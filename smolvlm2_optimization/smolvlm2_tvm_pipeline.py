#!/usr/bin/env python3
"""A100 SmolVLM2 benchmark/demo pipeline with TVM vision-tower compilation.

The full autoregressive VLM has dynamic token generation, so the stable compiler
target is the fixed-shape vision tower. End-to-end generation is benchmarked as
native PyTorch and as a compiler-enabled PyTorch path. TVM compilation and
latency are recorded separately for the vision tower because that is the part
with static image tensor shapes and repeated work for video frames.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_PROMPT = (
    "Describe the important actions, people, objects, and scene context in this video. "
    "Answer in two concise sentences."
)

TASKS = {
    "source_busy_city_street": ["street", "city", "people", "walking"],
    "source_students_university": ["students", "people", "campus", "walking"],
    "source_anonymous_woman_street": ["woman", "street", "person", "walking"],
}


@dataclass
class GenerationResult:
    text: str
    latency_ms: float
    tokens: int


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def latency_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    avg = statistics.fmean(values)
    return {
        "n": len(values),
        "avg_ms": avg,
        "min_ms": min(values),
        "max_ms": max(values),
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "fps": 1000.0 / avg if avg > 0 else 0.0,
    }


def load_model(model_id: str) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(model_id)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": None,
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            _attn_implementation="flash_attention_2",
            **kwargs,
        )
    except Exception as exc:
        print(f"flash_attention_2 load failed, falling back to sdpa/eager: {exc}")
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                _attn_implementation="sdpa",
                **kwargs,
            )
        except Exception:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval().to("cuda")
    return processor, model


def prepare_inputs(processor: Any, video_path: Path, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": str(video_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    moved: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def generate_once(model: Any, processor: Any, inputs: dict[str, torch.Tensor], max_new_tokens: int) -> GenerationResult:
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    prompt_len = int(inputs["input_ids"].shape[-1])
    new_ids = output_ids[0][prompt_len:]
    text = processor.decode(new_ids, skip_special_tokens=True).strip()
    return GenerationResult(text=text, latency_ms=elapsed_ms, tokens=int(new_ids.numel()))


def keyword_score(text: str, expected: list[str]) -> dict[str, Any]:
    lowered = text.lower()
    hits = [word for word in expected if word in lowered]
    return {
        "expected_keywords": expected,
        "hits": hits,
        "score": len(hits) / len(expected) if expected else 0.0,
    }


def find_vision_module(model: Any) -> tuple[str, torch.nn.Module]:
    preferred = []
    for name, module in model.named_modules():
        lname = name.lower()
        if lname.endswith("vision_model") or lname.endswith("vision_tower"):
            preferred.append((name, module))
    if preferred:
        return preferred[0]
    for name, module in model.named_modules():
        if "vision" in name.lower() and len(list(module.children())) > 0:
            return name, module
    raise RuntimeError("could not find a vision tower module in SmolVLM2")


class VisionWrapper(torch.nn.Module):
    def __init__(self, vision: torch.nn.Module):
        super().__init__()
        self.vision = vision
        self._signature = inspect.signature(vision.forward)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if "pixel_values" in self._signature.parameters:
            out = self.vision(pixel_values=pixel_values)
        else:
            out = self.vision(pixel_values)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if isinstance(out, (tuple, list)):
            for item in out:
                if torch.is_tensor(item):
                    return item
        if torch.is_tensor(out):
            return out
        raise TypeError(f"unsupported vision tower output: {type(out)!r}")


def first_video_frame(video_path: Path) -> Image.Image:
    reader = imageio.get_reader(str(video_path))
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    return Image.fromarray(np.asarray(frame[..., :3], dtype=np.uint8)).convert("RGB")


def extract_pixel_values(processor: Any, image_path: Path, device: torch.device) -> torch.Tensor:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path)},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"]
    if pixel_values.ndim == 5:
        pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:])
    return pixel_values[:1].to(device=device, dtype=torch.float32).contiguous()


def benchmark_torch_vision(vision: torch.nn.Module, example: torch.Tensor, bench_iters: int) -> dict[str, Any]:
    latencies = []
    with torch.inference_mode():
        for _ in range(2):
            _ = vision(example.to(dtype=torch.bfloat16))
        torch.cuda.synchronize()
        for _ in range(bench_iters):
            start = time.perf_counter()
            _ = vision(example.to(dtype=torch.bfloat16))
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)
    return latency_stats(latencies)


def compile_and_benchmark_tvm(
    vision: torch.nn.Module,
    example: torch.Tensor,
    work_dir: Path,
    bench_iters: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "not_run",
        "target": "cuda -arch=sm_80",
        "input_shape": list(example.shape),
    }
    try:
        import tvm
        from tvm import relay
        from tvm.contrib import graph_executor
    except Exception as exc:
        result.update({"status": "import_failed", "error": repr(exc)})
        return result

    wrapper = VisionWrapper(vision).eval().cpu().float()
    cpu_example = example.detach().cpu().float()
    try:
        scripted = torch.jit.trace(wrapper, cpu_example, strict=False)
        mod, params = relay.frontend.from_pytorch(scripted, [("pixel_values", tuple(cpu_example.shape))])
        target = tvm.target.Target("cuda -arch=sm_80")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        artifact = work_dir / "artifacts" / "smolvlm2_vision_tvm_cuda_sm80.so"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        lib.export_library(str(artifact))

        dev = tvm.cuda(0)
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input("pixel_values", tvm.nd.array(cpu_example.numpy(), dev))
        for _ in range(2):
            module.run()
        prof = module.benchmark(dev, number=10, repeat=bench_iters)
        mean_ms = float(prof.mean * 1000.0)
        result.update(
            {
                "status": "ok",
                "artifact": str(artifact),
                "mean_ms": mean_ms,
                "results": [float(x * 1000.0) for x in prof.results],
                "speed_note": "TVM graph executor benchmark for the fixed-shape vision tower only.",
            }
        )
    except Exception as exc:
        result.update({"status": "compile_or_benchmark_failed", "error": repr(exc)})
    finally:
        vision.cuda().to(dtype=torch.bfloat16).eval()
    return result


def compile_model_with_torch(model: Any) -> tuple[Any, str]:
    try:
        compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        return compiled, "torch.compile(mode='reduce-overhead')"
    except Exception as exc:
        print(f"torch.compile failed, optimized end-to-end path will reuse native model: {exc}")
        return model, "native fallback after torch.compile failure"


def fit_frame(frame: np.ndarray, size: tuple[int, int]) -> Image.Image:
    img = Image.fromarray(frame[..., :3]).convert("RGB")
    img.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textbbox((0, 0), trial, font=font)[2] <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_panel(base: Image.Image, title: str, subtitle: str, body: str) -> np.ndarray:
    img = base.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 19)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 13)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        title_font = sub_font = body_font = ImageFont.load_default()
    draw.rectangle((0, 0, img.width, 74), fill=(0, 0, 0, 185))
    draw.text((10, 7), title, fill=(255, 255, 255, 255), font=title_font)
    draw.text((10, 35), subtitle, fill=(232, 232, 232, 255), font=sub_font)
    if body:
        lines = wrap_text(draw, body, body_font, img.width - 24)[:5]
        box_h = 18 + 20 * len(lines)
        y0 = img.height - box_h
        draw.rectangle((0, y0, img.width, img.height), fill=(0, 0, 0, 178))
        for i, line in enumerate(lines):
            draw.text((12, y0 + 8 + i * 20), line, fill=(255, 255, 255, 255), font=body_font)
    return np.asarray(img)


def make_concat_demo(
    video_path: Path,
    native: GenerationResult,
    optimized: GenerationResult,
    output: Path,
    seconds: float,
    fps: float = 24.0,
    panel_size: tuple[int, int] = (426, 240),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(str(video_path))
    meta = reader.get_meta_data()
    source_fps = float(meta.get("fps", fps) or fps)
    total = max(1, int(math.ceil(seconds * fps)))
    try:
        with imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            last_frame = None
            for out_idx in range(total):
                source_idx = int((out_idx / fps) * source_fps)
                try:
                    frame = reader.get_data(source_idx)
                    last_frame = frame
                except Exception:
                    if last_frame is None:
                        break
                    frame = last_frame
                base = fit_frame(np.asarray(frame, dtype=np.uint8), panel_size)
                panels = [
                    draw_panel(base, "Source video", f"{source_fps:.1f} FPS source | frame {source_idx}", ""),
                    draw_panel(
                        base,
                        "Native PyTorch SmolVLM2",
                        f"{native.latency_ms:.1f} ms / {native.tokens} new tokens",
                        native.text,
                    ),
                    draw_panel(
                        base,
                        "Optimized compiler path",
                        f"{optimized.latency_ms:.1f} ms / {optimized.tokens} new tokens",
                        optimized.text,
                    ),
                ]
                writer.append_data(np.concatenate(panels, axis=1))
    finally:
        reader.close()


def write_summary(work_dir: Path, results: dict[str, Any]) -> None:
    lines = [
        "# SmolVLM2 A100 TVM Optimization Results",
        "",
        f"Model: `{results['model_id']}`",
        "",
        "## Compiler Strategy",
        "",
        "- TVM Relay compiles the fixed-shape SmolVLM2 vision tower for CUDA `sm_80`.",
        "- End-to-end text generation is benchmarked as native PyTorch and as a compiler-enabled PyTorch path.",
        "- Accuracy is a lightweight keyword-hit score for the three demo videos, used only as a reproducible smoke benchmark.",
        "",
        "## Latency",
        "",
        "| Track | Avg latency | Notes |",
        "| --- | ---: | --- |",
    ]
    for track, stats in results["latency_summary"].items():
        avg = stats.get("avg_ms")
        avg_s = f"{avg:.1f} ms" if isinstance(avg, float) else "n/a"
        lines.append(f"| {track} | {avg_s} | n={stats.get('n', 0)} |")
    tvm_result = results["tvm_vision"]
    tvm_avg = tvm_result.get("mean_ms")
    lines.extend(
        [
            "",
            "## TVM Vision Tower",
            "",
            f"- Status: `{tvm_result.get('status')}`",
            f"- Input shape: `{tvm_result.get('input_shape')}`",
            f"- Mean latency: `{tvm_avg:.3f} ms`" if isinstance(tvm_avg, float) else "- Mean latency: `n/a`",
        ]
    )
    if tvm_result.get("artifact"):
        lines.append(f"- Artifact: `{tvm_result['artifact']}`")
    if tvm_result.get("error"):
        lines.append(f"- Error: `{tvm_result['error']}`")
    lines.extend(["", "## Demo Videos", ""])
    for item in results["videos"]:
        lines.append(f"- `{item['demo_video']}`")
    (work_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--bench-iters", type=int, default=5)
    parser.add_argument("--demo-seconds", type=float, default=10.0)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load_model(args.model_id)
    device = torch.device("cuda")

    vision_name, vision_module = find_vision_module(model)
    sample_image = first_video_frame(args.videos[0])
    sample_path = args.work_dir / "artifacts" / "tvm_sample_frame.jpg"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_image.save(sample_path)
    vision_input = extract_pixel_values(processor, sample_path, device)
    torch_vision_stats = benchmark_torch_vision(VisionWrapper(vision_module).cuda().eval(), vision_input, args.bench_iters)
    tvm_vision = compile_and_benchmark_tvm(vision_module, vision_input, args.work_dir, args.bench_iters)

    prepared = {str(video): prepare_inputs(processor, video, args.prompt, device) for video in args.videos}

    native_results: dict[str, GenerationResult] = {}
    for video in args.videos:
        inputs = prepared[str(video)]
        _ = generate_once(model, processor, inputs, min(8, args.max_new_tokens))
        native_results[str(video)] = generate_once(model, processor, inputs, args.max_new_tokens)

    optimized_model, optimized_method = compile_model_with_torch(model)
    optimized_results: dict[str, GenerationResult] = {}
    for video in args.videos:
        inputs = prepared[str(video)]
        try:
            _ = generate_once(optimized_model, processor, inputs, min(8, args.max_new_tokens))
            optimized_results[str(video)] = generate_once(optimized_model, processor, inputs, args.max_new_tokens)
        except Exception as exc:
            print(f"optimized generation failed for {video}, falling back to native model: {exc}")
            optimized_method = f"{optimized_method}; per-video native fallback after generation failure"
            optimized_results[str(video)] = generate_once(model, processor, inputs, args.max_new_tokens)

    video_rows = []
    for video in args.videos:
        stem = video.stem
        native = native_results[str(video)]
        optimized = optimized_results[str(video)]
        expected = TASKS.get(stem, [])
        demo_video = args.work_dir / "demo_outputs" / f"{stem}_source_native_optimized.mp4"
        make_concat_demo(video, native, optimized, demo_video, seconds=args.demo_seconds)
        video_rows.append(
            {
                "video": str(video),
                "native": {
                    "text": native.text,
                    "latency_ms": native.latency_ms,
                    "tokens": native.tokens,
                    "keyword_accuracy": keyword_score(native.text, expected),
                },
                "optimized": {
                    "method": optimized_method,
                    "text": optimized.text,
                    "latency_ms": optimized.latency_ms,
                    "tokens": optimized.tokens,
                    "keyword_accuracy": keyword_score(optimized.text, expected),
                },
                "demo_video": str(demo_video),
            }
        )

    results = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "vision_module": vision_name,
        "optimized_method": optimized_method,
        "torch_vision": torch_vision_stats,
        "tvm_vision": tvm_vision,
        "latency_summary": {
            "native_end_to_end": latency_stats([row["native"]["latency_ms"] for row in video_rows]),
            "optimized_end_to_end": latency_stats([row["optimized"]["latency_ms"] for row in video_rows]),
            "torch_vision_tower": torch_vision_stats,
        },
        "videos": video_rows,
    }
    out_json = args.work_dir / "benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(args.work_dir, results)
    print(f"wrote {out_json}")
    print(f"wrote {args.work_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
