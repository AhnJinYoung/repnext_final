#!/usr/bin/env python3
"""Create a wall-clock video comparison from segmented frame directories."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Panel:
    label: str
    frames_dir: Path
    metrics: Path
    frames: list[np.ndarray]
    ready_times: list[float]
    avg_ms: float
    fps: float


def fit_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (18, 18, 18))
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return np.asarray(canvas)


def load_panel(label: str, frames_dir: Path, metrics_path: Path, panel_size: tuple[int, int]) -> Panel:
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"no PNG frames in {frames_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_frame = metrics.get("per_frame", [])
    infer_ms = [float(row.get("infer_ms", 0.0)) for row in per_frame[: len(frame_paths)]]
    if len(infer_ms) < len(frame_paths):
        avg = float(metrics.get("inference", {}).get("avg_ms", 1000.0))
        infer_ms.extend([avg] * (len(frame_paths) - len(infer_ms)))
    ready_times = []
    elapsed = 0.0
    for ms in infer_ms:
        elapsed += max(ms, 1.0) / 1000.0
        ready_times.append(elapsed)
    avg_ms = float(metrics.get("inference", {}).get("avg_ms", np.mean(infer_ms)))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    frames = [fit_image(path, panel_size) for path in frame_paths]
    return Panel(label, frames_dir, metrics_path, frames, ready_times, avg_ms, fps)


def draw_panel(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
    draw.rectangle((0, 0, img.width, 58), fill=(0, 0, 0, 175))
    draw.text((12, 8), title, fill=(255, 255, 255, 255), font=title_font)
    draw.text((12, 33), subtitle, fill=(230, 230, 230, 255), font=sub_font)
    return np.asarray(img)


def frame_for_time(panel: Panel, t: float) -> tuple[np.ndarray, int]:
    idx = 0
    for pos, ready in enumerate(panel.ready_times):
        if t >= ready:
            idx = pos
        else:
            break
    return panel.frames[idx], idx


def encode(panels: list[Panel], output: Path, fps: float, pad_seconds: float) -> None:
    max_duration = max(panel.ready_times[-1] for panel in panels) + pad_seconds
    total_frames = max(1, math.ceil(max_duration * fps))
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for out_idx in range(total_frames):
            t = out_idx / fps
            rendered = []
            for panel in panels:
                frame, frame_idx = frame_for_time(panel, t)
                subtitle = f"{panel.avg_ms:.1f} ms/frame | {panel.fps:.2f} FPS | shown {frame_idx + 1}/{len(panel.frames)}"
                rendered.append(draw_panel(frame, panel.label, subtitle))
            writer.append_data(np.concatenate(rendered, axis=1))
    print(f"wrote {output} at {fps:g} fps for {max_duration:.2f}s ({total_frames} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument("--panel-height", type=int, default=240)
    parser.add_argument("--pad-seconds", type=float, default=1.0)
    args = parser.parse_args()

    size = (args.panel_width, args.panel_height)
    specs = [
        ("Native PyTorch 512", "native512_frames", "native512_metrics.json"),
        ("Intel CPU LiteRT 192", "litert192_frames", "litert192_metrics.json"),
        ("RPi5 CPU LiteRT 256", "rpi5_cpu_frames", "rpi5_cpu_metrics.json"),
        ("RPi5 + Coral TPU INT8 192", "rpi5_tpu_frames", "rpi5_tpu_metrics.json"),
    ]
    panels = [load_panel(label, args.root / frames, args.root / metrics, size) for label, frames, metrics in specs]
    encode(panels, args.output, args.fps, args.pad_seconds)


if __name__ == "__main__":
    main()
