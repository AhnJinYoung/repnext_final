#!/usr/bin/env python3
"""Create a Source + Native + RPi5 CPU realtime comparison video."""

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
class Event:
    source_index: int
    ready_time: float
    frame: np.ndarray


@dataclass
class Panel:
    label: str
    subtitle_prefix: str
    events: list[Event]


def fit_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return np.asarray(canvas)


def draw_panel(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 21)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
    draw.rectangle((0, 0, img.width, 62), fill=(0, 0, 0, 178))
    draw.text((12, 7), title, fill=(255, 255, 255, 255), font=title_font)
    draw.text((12, 35), subtitle, fill=(235, 235, 235, 255), font=sub_font)
    return np.asarray(img)


def read_metrics(metrics_path: Path, frame_count: int) -> tuple[float, list[float]]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    avg_ms = float(metrics.get("inference", {}).get("avg_ms", 1000.0))
    infer_ms = [float(row.get("infer_ms", avg_ms)) for row in metrics.get("per_frame", [])]
    if len(infer_ms) < frame_count:
        infer_ms.extend([avg_ms] * (frame_count - len(infer_ms)))
    return avg_ms, infer_ms[:frame_count]


def load_sparse_mapping(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [int(x) for x in data["source_indices"]]
    return [int(x) for x in data]


def load_source_panel(root: Path, size: tuple[int, int], source_fps: float, source_frames: int) -> Panel:
    paths = sorted((root / "input_frames").glob("*.jpg"))
    if not paths:
        paths = sorted((root / "input_frames").glob("*.png"))
    if not paths:
        raise SystemExit(f"no source frames in {root / 'input_frames'}")
    events = [
        Event(idx, idx / source_fps, fit_image(paths[min(idx, len(paths) - 1)], size))
        for idx in range(source_frames)
    ]
    return Panel("Source video", f"{source_fps:.1f} FPS source", events)


def load_native_panel(root: Path, size: tuple[int, int]) -> Panel:
    paths = sorted((root / "native_realtime_sparse_frames").glob("*.png"))
    if not paths:
        raise SystemExit(f"no native frames in {root / 'native_realtime_sparse_frames'}")
    avg_ms, infer_ms = read_metrics(root / "native_realtime_sparse_metrics.json", len(paths))
    mapping = load_sparse_mapping(root / "native_realtime_sparse_mapping.json")
    elapsed = 0.0
    events = []
    for idx, source_index in enumerate(mapping[: len(paths)]):
        elapsed += max(infer_ms[idx], 1.0) / 1000.0
        events.append(Event(int(source_index), elapsed, fit_image(paths[idx], size)))
    return Panel("Native PyTorch 512", f"{avg_ms:.1f} ms/frame, {1000.0 / avg_ms:.2f} FPS", events)


def load_rpi5_panel(root: Path, size: tuple[int, int], source_fps: float, source_frames: int) -> Panel:
    paths = sorted((root / "rpi5_cpu_frames").glob("*.png"))
    if not paths:
        raise SystemExit(f"no RPi5 frames in {root / 'rpi5_cpu_frames'}")
    avg_ms, infer_ms = read_metrics(root / "rpi5_cpu_metrics.json", len(paths))
    duration = source_frames / source_fps
    current_time = 0.0
    events = []
    while current_time < duration:
        source_index = int(math.floor(current_time * source_fps))
        if source_index >= source_frames or source_index >= len(paths):
            break
        ready_time = current_time + max(infer_ms[min(source_index, len(infer_ms) - 1)], 1.0) / 1000.0
        events.append(Event(source_index, ready_time, fit_image(paths[source_index], size)))
        current_time = ready_time
    if not events:
        events.append(Event(0, avg_ms / 1000.0, fit_image(paths[0], size)))
    return Panel("RPi5 CPU LiteRT 256", f"{avg_ms:.1f} ms/frame, {1000.0 / avg_ms:.2f} FPS", events)


def frame_for_time(panel: Panel, t: float) -> tuple[np.ndarray, Event]:
    event = panel.events[0]
    for candidate in panel.events:
        if t >= candidate.ready_time:
            event = candidate
        else:
            break
    return event.frame, event


def encode(panels: list[Panel], output: Path, fps: float, duration: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, math.ceil(duration * fps))
    with imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for out_idx in range(total_frames):
            t = out_idx / fps
            rendered = []
            for panel in panels:
                frame, event = frame_for_time(panel, t)
                rendered.append(draw_panel(frame, panel.label, f"{panel.subtitle_prefix} | frame {event.source_index}"))
            writer.append_data(np.concatenate(rendered, axis=1))
    print(f"wrote {output} at {fps:g} fps for {duration:.2f}s ({total_frames} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--panel-width", type=int, default=426)
    parser.add_argument("--panel-height", type=int, default=240)
    args = parser.parse_args()

    meta = json.loads((args.root / "input_frames" / "frames_meta.json").read_text(encoding="utf-8"))
    source_fps = float(meta["fps"])
    source_frames = int(meta["frames"])
    duration = source_frames / source_fps
    size = (args.panel_width, args.panel_height)

    panels = [
        load_source_panel(args.root, size, source_fps, source_frames),
        load_native_panel(args.root, size),
        load_rpi5_panel(args.root, size, source_fps, source_frames),
    ]
    encode(panels, args.output, args.fps, duration)


if __name__ == "__main__":
    main()
