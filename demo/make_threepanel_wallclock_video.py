#!/usr/bin/env python3
"""Create a 3-panel wall-clock video comparison for the RPi5-only demo path."""

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
    frames: list[np.ndarray]
    ready_times: list[float]
    avg_ms: float
    fps: float


def fit_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return np.asarray(canvas)


def load_panel(label: str, frames_dir: Path, metrics_path: Path, panel_size: tuple[int, int]) -> Panel:
    paths = sorted(frames_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"no PNG frames in {frames_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    avg_ms = float(metrics.get("inference", {}).get("avg_ms", 1000.0))
    per_frame = metrics.get("per_frame", [])
    infer_ms = [float(row.get("infer_ms", avg_ms)) for row in per_frame[: len(paths)]]
    if len(infer_ms) < len(paths):
        infer_ms.extend([avg_ms] * (len(paths) - len(infer_ms)))
    elapsed = 0.0
    ready_times = []
    for ms in infer_ms:
        elapsed += max(ms, 1.0) / 1000.0
        ready_times.append(elapsed)
    return Panel(label, [fit_image(path, panel_size) for path in paths], ready_times, avg_ms, 1000.0 / avg_ms)


def draw_panel(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except Exception:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
    draw.rectangle((0, 0, img.width, 62), fill=(0, 0, 0, 175))
    draw.text((12, 8), title, fill=(255, 255, 255, 255), font=title_font)
    draw.text((12, 36), subtitle, fill=(235, 235, 235, 255), font=sub_font)
    return np.asarray(img)


def frame_for_time(panel: Panel, t: float) -> tuple[np.ndarray, int]:
    idx = 0
    for pos, ready in enumerate(panel.ready_times):
        if t >= ready:
            idx = pos
        else:
            break
    return panel.frames[idx], idx


def encode(panels: list[Panel], output: Path, fps: float, duration: float) -> None:
    total_frames = max(1, math.ceil(duration * fps))
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for out_idx in range(total_frames):
            t = out_idx / fps
            rendered = []
            for panel in panels:
                frame, idx = frame_for_time(panel, t)
                subtitle = f"{panel.avg_ms:.1f} ms/frame | {panel.fps:.2f} FPS | shown {idx + 1}/{len(panel.frames)}"
                rendered.append(draw_panel(frame, panel.label, subtitle))
            writer.append_data(np.concatenate(rendered, axis=1))
    print(f"wrote {output} at {fps:g} fps for {duration:.2f}s ({total_frames} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--panel-width", type=int, default=426)
    parser.add_argument("--panel-height", type=int, default=240)
    args = parser.parse_args()

    size = (args.panel_width, args.panel_height)
    panels = [
        load_panel("Native PyTorch 512", args.root / "native512_frames", args.root / "native512_metrics.json", size),
        load_panel("Intel CPU LiteRT 192", args.root / "litert192_frames", args.root / "litert192_metrics.json", size),
        load_panel("RPi5 CPU LiteRT 256", args.root / "rpi5_cpu_frames", args.root / "rpi5_cpu_metrics.json", size),
    ]
    encode(panels, args.output, args.fps, args.duration)


if __name__ == "__main__":
    main()
