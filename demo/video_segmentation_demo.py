#!/usr/bin/env python3
"""Video/frame sequence segmentation demo for RepNeXt variants.

The module supports three workflows:

1. extract: video -> numbered PNG frames
2. run: frames/video -> segmented overlay frames/video + timing JSON
3. encode: numbered PNG frames -> video

Use frame-directory mode when running in a minimal LiteRT environment that lacks
video I/O libraries. Use video mode when imageio or torchvision+PyAV is present.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = 150


def add_conversion_path() -> None:
    path = str(ROOT / "conversion")
    if path not in sys.path:
        sys.path.insert(0, path)


def ade_palette() -> np.ndarray:
    import colorsys

    colors = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
    golden = 0.61803398875
    h = 0.0
    for idx in range(NUM_CLASSES):
        h = (h + golden) % 1.0
        s = 0.45 + 0.5 * ((idx * 7) % 5) / 4.0
        v = 0.55 + 0.45 * ((idx * 3) % 4) / 3.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors[idx] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


PALETTE = ade_palette()


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "avg_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "p50_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "fps": 1000.0 / statistics.fmean(values),
        "n": len(values),
    }


def colorize(label: np.ndarray) -> np.ndarray:
    safe = np.where((label < 0) | (label >= NUM_CLASSES), 0, label)
    return PALETTE[safe]


def overlay(rgb: np.ndarray, label: np.ndarray, alpha: float) -> np.ndarray:
    seg = Image.fromarray(colorize(label)).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)
    seg_arr = np.asarray(seg).astype(np.float32)
    out = (1.0 - alpha) * rgb.astype(np.float32) + alpha * seg_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_hud(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    pad = 8
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 2 * pad
    h = bbox[3] - bbox[1] + 2 * pad
    draw.rectangle((8, 8, 8 + w, 8 + h), fill=(0, 0, 0, 150))
    draw.text((8 + pad, 8 + pad), text, fill=(255, 255, 255, 255), font=font)
    return np.asarray(img)


def resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR))


def preprocess_rgb(rgb: np.ndarray, size: int, normalize: str) -> np.ndarray:
    arr = resize_rgb(rgb, size).astype(np.float32) / 255.0
    if normalize == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
    elif normalize == "minus-one-one":
        arr = arr * 2.0 - 1.0
    elif normalize != "zero-one":
        raise ValueError(f"unknown normalize mode: {normalize}")
    return arr


def frame_paths(path: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in exts)


def read_frames_from_dir(path: Path, stride: int, max_frames: int | None) -> Iterable[tuple[int, np.ndarray]]:
    kept = 0
    for idx, frame_path in enumerate(frame_paths(path)):
        if idx % stride:
            continue
        rgb = np.asarray(Image.open(frame_path).convert("RGB"))
        yield idx, rgb
        kept += 1
        if max_frames is not None and kept >= max_frames:
            break


def _imageio_reader(path: Path):
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))
    return reader, fps


def read_frames_from_video(path: Path, stride: int, max_frames: int | None) -> tuple[list[tuple[int, np.ndarray]], float]:
    try:
        reader, fps = _imageio_reader(path)
        frames = []
        kept = 0
        for idx, frame in enumerate(reader):
            if idx % stride:
                continue
            frames.append((idx, np.asarray(frame[..., :3], dtype=np.uint8)))
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break
        reader.close()
        return frames, fps
    except Exception as imageio_exc:
        try:
            from torchvision.io import read_video

            video, _, info = read_video(str(path), pts_unit="sec")
            fps = float(info.get("video_fps", 30.0))
            frames = []
            kept = 0
            for idx in range(video.shape[0]):
                if idx % stride:
                    continue
                frames.append((idx, video[idx].numpy().astype(np.uint8)))
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    break
            return frames, fps
        except Exception as tv_exc:
            raise RuntimeError(
                "Video decoding requires imageio[ffmpeg] or torchvision with PyAV. "
                "Install `pip install imageio imageio-ffmpeg`, or run extract on a machine "
                "with video I/O and then use --input-frames."
            ) from tv_exc


def write_video(path: Path, frames: Iterable[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
        return
    except Exception as imageio_exc:
        try:
            import torch
            from torchvision.io import write_video

            arr = np.stack(list(frames), axis=0)
            write_video(str(path), torch.from_numpy(arr), fps=fps)
            return
        except Exception as tv_exc:
            raise RuntimeError(
                "Video encoding requires imageio[ffmpeg] or torchvision with PyAV. "
                "Install `pip install imageio imageio-ffmpeg`, or use --output-frames."
            ) from tv_exc


class PytorchBackend:
    def __init__(self, weights: Path, activation: str, sparse: bool, threads: int):
        import torch

        add_conversion_path()
        import export_onnx

        torch.set_num_threads(threads)
        if activation == "relu":
            act_cls = torch.nn.ReLU
        elif activation == "tanh-gelu":
            act_cls = lambda: torch.nn.GELU(approximate="tanh")
        else:
            act_cls = torch.nn.GELU
        export_onnx.SPARSE_EQUIV_DOWNSAMPLE = sparse
        export_onnx.TPU_FRIENDLY_DOWNSAMPLE = False
        model = export_onnx.RepNeXtSeg(act=act_cls).eval()
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        if sparse:
            state, _ = export_onnx.rewrite_sparse_downsample_weights(state)
        model.load_state_dict(state, strict=False)
        self.torch = torch
        self.model = model

    def infer(self, image_hwc: np.ndarray) -> np.ndarray:
        x = self.torch.from_numpy(image_hwc.transpose(2, 0, 1)).unsqueeze(0).contiguous()
        with self.torch.inference_mode():
            y = self.model(x)
        return y[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)


class TfliteBackend:
    def __init__(self, model_path: Path, threads: int, delegate: str):
        Interpreter = None
        load_delegate = None
        try:
            from ai_edge_litert.interpreter import Interpreter as LiteInterpreter
            from ai_edge_litert.interpreter import load_delegate as lite_load_delegate

            Interpreter = LiteInterpreter
            load_delegate = lite_load_delegate
        except Exception:
            try:
                from tflite_runtime.interpreter import Interpreter as LiteInterpreter
                from tflite_runtime.interpreter import load_delegate as lite_load_delegate

                Interpreter = LiteInterpreter
                load_delegate = lite_load_delegate
            except Exception:
                from tensorflow.lite.python.interpreter import Interpreter as LiteInterpreter
                from tensorflow.lite.python.interpreter import load_delegate as lite_load_delegate

                Interpreter = LiteInterpreter
                load_delegate = lite_load_delegate

        delegates = []
        if delegate == "edgetpu":
            for lib in ("libedgetpu.so.1", "libedgetpu.so"):
                try:
                    delegates = [load_delegate(lib)]
                    break
                except Exception:
                    delegates = []
            if not delegates:
                raise RuntimeError("Could not load EdgeTPU delegate. Is libedgetpu installed?")

        kwargs = {"model_path": str(model_path)}
        if delegate == "none":
            kwargs["num_threads"] = threads
        if delegates:
            kwargs["experimental_delegates"] = delegates
        try:
            self.interp = Interpreter(**kwargs)
        except TypeError:
            kwargs.pop("num_threads", None)
            self.interp = Interpreter(**kwargs)
        try:
            self.interp.allocate_tensors()
        except RuntimeError as exc:
            if "XNNPACK" not in str(exc) or delegates:
                raise
            self.interp = Interpreter(model_path=str(model_path), experimental_preserve_all_tensors=True)
            self.interp.allocate_tensors()
        self.input_info = self.interp.get_input_details()[0]
        self.output_info = self.interp.get_output_details()[0]

    def infer(self, image_hwc: np.ndarray) -> np.ndarray:
        scale, zero_point = self.input_info["quantization"]
        dtype = self.input_info["dtype"]
        if scale and np.issubdtype(dtype, np.integer):
            q = np.rint(image_hwc[None, ...] / scale + zero_point)
            limits = np.iinfo(dtype)
            x = np.clip(q, limits.min, limits.max).astype(dtype)
        else:
            x = image_hwc[None, ...].astype(dtype)
        self.interp.set_tensor(self.input_info["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.output_info["index"])
        out_scale, out_zp = self.output_info["quantization"]
        if out_scale and np.issubdtype(y.dtype, np.integer):
            y = (y.astype(np.float32) - out_zp) * out_scale
        else:
            y = y.astype(np.float32)
        y = y[0]
        if y.shape[0] == NUM_CLASSES:
            y = np.transpose(y, (1, 2, 0))
        return y


def build_backend(args):
    if args.backend == "pytorch":
        weights = args.weights
        if not weights.exists() and Path("/workspace/tvm/handoff/repnext_m5_ade20k.pth").exists():
            weights = Path("/workspace/tvm/handoff/repnext_m5_ade20k.pth")
        return PytorchBackend(weights, args.activation, args.sparse_equiv_downsample, args.threads)
    if args.backend == "tflite":
        if args.model is None:
            raise SystemExit("--model is required for --backend tflite")
        return TfliteBackend(args.model, args.threads, args.delegate)
    raise ValueError(args.backend)


def load_input_frames(args) -> tuple[list[tuple[int, np.ndarray]], float]:
    if args.input_frames:
        return list(read_frames_from_dir(args.input_frames, args.stride, args.max_frames)), args.fps
    if args.input_video:
        return read_frames_from_video(args.input_video, args.stride, args.max_frames)
    raise SystemExit("Provide --input-video or --input-frames")


def save_output_frames(frames: list[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(out_dir / f"seg_{idx:06d}.png")


def run_demo(args) -> None:
    frames, input_fps = load_input_frames(args)
    if not frames:
        raise SystemExit("No frames decoded")
    backend = build_backend(args)

    out_frames = []
    rows = []
    start_all = time.perf_counter()
    for out_idx, (src_idx, rgb) in enumerate(frames):
        model_input = preprocess_rgb(rgb, args.size, args.normalize)
        t0 = time.perf_counter()
        logits = backend.infer(model_input)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        pred = logits.argmax(axis=-1).astype(np.int64)
        composed = overlay(rgb, pred, args.alpha)
        if args.hud:
            text = f"{args.name} | frame {out_idx + 1}/{len(frames)} | infer {infer_ms:.1f} ms | {1000.0 / infer_ms:.2f} FPS"
            composed = draw_hud(composed, text)
        out_frames.append(composed)
        rows.append({"output_index": out_idx, "source_index": src_idx, "infer_ms": infer_ms, "fps": 1000.0 / infer_ms})
        print(f"[{out_idx + 1:04d}/{len(frames):04d}] src={src_idx} infer={infer_ms:.1f} ms fps={1000.0 / infer_ms:.2f}", flush=True)

    total_s = time.perf_counter() - start_all
    output_fps = args.output_fps if args.output_fps else min(input_fps / max(args.stride, 1), max(1.0, stats([r["infer_ms"] for r in rows])["fps"]))
    if args.output_frames:
        save_output_frames(out_frames, args.output_frames)
    if args.output_video:
        write_video(args.output_video, out_frames, output_fps)

    result = {
        "name": args.name,
        "backend": args.backend,
        "model": str(args.model) if args.model else None,
        "weights": str(args.weights) if args.backend == "pytorch" else None,
        "input_video": str(args.input_video) if args.input_video else None,
        "input_frames": str(args.input_frames) if args.input_frames else None,
        "output_video": str(args.output_video) if args.output_video else None,
        "output_frames": str(args.output_frames) if args.output_frames else None,
        "size": args.size,
        "normalize": args.normalize,
        "frames": len(rows),
        "input_fps": input_fps,
        "output_fps": output_fps,
        "wall_time_s": total_s,
        "wall_fps": len(rows) / total_s if total_s > 0 else 0.0,
        "inference": stats([float(r["infer_ms"]) for r in rows]),
        "per_frame": rows,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_frame"}, indent=2))


def extract_frames(args) -> None:
    frames, fps = read_frames_from_video(args.input_video, args.stride, args.max_frames)
    args.output_frames.mkdir(parents=True, exist_ok=True)
    for out_idx, (src_idx, rgb) in enumerate(frames):
        Image.fromarray(rgb).save(args.output_frames / f"frame_{out_idx:06d}.png")
    meta = {"input_video": str(args.input_video), "fps": fps, "stride": args.stride, "frames": len(frames)}
    (args.output_frames / "frames_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def encode_frames(args) -> None:
    frames = [np.asarray(Image.open(path).convert("RGB")) for path in frame_paths(args.input_frames)]
    write_video(args.output_video, frames, args.fps)
    print(f"wrote {args.output_video} from {len(frames)} frames at {args.fps} fps")


def summarize_runs(args) -> None:
    rows = []
    for path in args.metrics:
        data = json.loads(path.read_text(encoding="utf-8"))
        infer = data.get("inference", {})
        rows.append(
            {
                "name": data.get("name", path.stem),
                "backend": data.get("backend", ""),
                "size": data.get("size", ""),
                "frames": data.get("frames", 0),
                "avg_ms": float(infer.get("avg_ms", 0.0)),
                "fps": float(infer.get("fps", 0.0)),
                "wall_fps": float(data.get("wall_fps", 0.0)),
                "output_video": data.get("output_video") or "",
            }
        )

    if not rows:
        raise SystemExit("No metrics files provided")
    baseline_ms = rows[0]["avg_ms"] or 1.0
    lines = [
        "| Run | Backend | Size | Frames | Avg infer | Infer FPS | Wall FPS | Speedup vs first |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        speedup = baseline_ms / row["avg_ms"] if row["avg_ms"] else 0.0
        lines.append(
            f"| {row['name']} | {row['backend']} | {row['size']} | {row['frames']} | "
            f"{row['avg_ms']:.1f} ms | {row['fps']:.2f} | {row['wall_fps']:.2f} | {speedup:.2f}x |"
        )
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    print(text)


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-video", type=Path)
    parser.add_argument("--input-frames", type=Path)
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--output-frames", type=Path)
    parser.add_argument("--metrics", type=Path, default=ROOT / "demo" / "video_runs" / "metrics.json")
    parser.add_argument("--name", default="RepNeXt demo")
    parser.add_argument("--backend", choices=["pytorch", "tflite"], required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "repnext_m5_ade20k.pth")
    parser.add_argument("--activation", choices=["gelu", "tanh-gelu", "relu"], default="gelu")
    parser.add_argument("--sparse-equiv-downsample", action="store_true")
    parser.add_argument("--delegate", choices=["none", "edgetpu"], default="none")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--normalize", choices=["zero-one", "imagenet", "minus-one-one"], default="zero-one")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=float, default=30.0, help="input FPS for frame-directory mode")
    parser.add_argument("--output-fps", type=float)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--hud", action=argparse.BooleanOptionalAction, default=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="segment a video or frame directory")
    add_common_run_args(p)

    p = sub.add_parser("extract", help="extract video frames to PNG files")
    p.add_argument("--input-video", type=Path, required=True)
    p.add_argument("--output-frames", type=Path, required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int)

    p = sub.add_parser("encode", help="encode PNG frames into a video")
    p.add_argument("--input-frames", type=Path, required=True)
    p.add_argument("--output-video", type=Path, required=True)
    p.add_argument("--fps", type=float, default=5.0)

    p = sub.add_parser("summarize", help="summarize run metrics JSON files as a markdown table")
    p.add_argument("metrics", nargs="+", type=Path)
    p.add_argument("--out", type=Path)

    args = parser.parse_args()
    if args.cmd == "run":
        run_demo(args)
    elif args.cmd == "extract":
        extract_frames(args)
    elif args.cmd == "encode":
        encode_frames(args)
    elif args.cmd == "summarize":
        summarize_runs(args)


if __name__ == "__main__":
    main()
