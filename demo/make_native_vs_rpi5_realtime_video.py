#!/usr/bin/env python3
"""Create a two-panel source-time-synced Native vs RPi5 CPU video.

The segmentation frames are reused from an existing run directory. Each panel
keeps the same wall-clock video duration as the source. If inference is slower
than the source frame rate, the panel holds the latest completed segmentation
frame, making the slower path look choppy without stretching the video.
"""

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
    avg_ms: float
    fps: float
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
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except Exception:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
    draw.rectangle((0, 0, img.width, 68), fill=(0, 0, 0, 178))
    draw.text((14, 8), title, fill=(255, 255, 255, 255), font=title_font)
    draw.text((14, 40), subtitle, fill=(235, 235, 235, 255), font=sub_font)
    return np.asarray(img)


def load_sparse_mapping(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [int(x) for x in data["source_indices"]]
    return [int(x) for x in data]


def read_avg_ms(metrics_path: Path) -> float:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(metrics.get("inference", {}).get("avg_ms", 1000.0))


def read_infer_ms(metrics_path: Path, count: int, avg_ms: float) -> list[float]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    values = [float(row.get("infer_ms", avg_ms)) for row in metrics.get("per_frame", [])]
    if len(values) < count:
        values.extend([avg_ms] * (count - len(values)))
    return values[:count]


def load_native_panel(root: Path, panel_size: tuple[int, int], source_fps: float) -> Panel:
    frames_dir = root / "native_realtime_sparse_frames"
    paths = sorted(frames_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"no native PNG frames in {frames_dir}")

    metrics_path = root / "native_realtime_sparse_metrics.json"
    avg_ms = read_avg_ms(metrics_path)
    infer_ms = read_infer_ms(metrics_path, len(paths), avg_ms)
    mapping = load_sparse_mapping(root / "native_realtime_sparse_mapping.json")

    events: list[Event] = []
    elapsed = 0.0
    for idx, source_index in enumerate(mapping[: len(paths)]):
        elapsed += max(infer_ms[idx], 1.0) / 1000.0
        # Clamp impossible stale frame numbers if old mapping files were made
        # from rounded source FPS metadata.
        source_index = max(0, int(source_index))
        events.append(Event(source_index, elapsed, fit_image(paths[idx], panel_size)))

    return Panel("Native PyTorch 512", avg_ms, 1000.0 / avg_ms, events)


def load_full_panel(
    root: Path,
    label: str,
    frames_name: str,
    metrics_name: str,
    panel_size: tuple[int, int],
    source_fps: float,
    source_frames: int,
) -> Panel:
    frames_dir = root / frames_name
    paths = sorted(frames_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"no PNG frames in {frames_dir}")

    metrics_path = root / metrics_name
    avg_ms = read_avg_ms(metrics_path)
    infer_ms = read_infer_ms(metrics_path, len(paths), avg_ms)

    events: list[Event] = []
    current_time = 0.0
    duration = source_frames / source_fps
    while current_time < duration:
        source_index = int(math.floor(current_time * source_fps))
        if source_index >= source_frames or source_index >= len(paths):
            break
        ms = infer_ms[min(source_index, len(infer_ms) - 1)]
        ready_time = current_time + max(ms, 1.0) / 1000.0
        events.append(Event(source_index, ready_time, fit_image(paths[source_index], panel_size)))
        current_time = ready_time

    if not events:
        events.append(Event(0, avg_ms / 1000.0, fit_image(paths[0], panel_size)))
    return Panel(label, avg_ms, 1000.0 / avg_ms, events)


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
                subtitle = f"{panel.avg_ms:.1f} ms/frame | {panel.fps:.2f} FPS | source frame {event.source_index}"
                rendered.append(draw_panel(frame, panel.label, subtitle))
            writer.append_data(np.concatenate(rendered, axis=1))
    print(f"wrote {output} at {fps:g} fps for {duration:.2f}s ({total_frames} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=360)
    args = parser.parse_args()

    meta = json.loads((args.root / "input_frames" / "frames_meta.json").read_text(encoding="utf-8"))
    source_fps = float(meta["fps"])
    source_frames = int(meta["frames"])
    duration = source_frames / source_fps
    panel_size = (args.panel_width, args.panel_height)

    panels = [
        load_native_panel(args.root, panel_size, source_fps),
        load_full_panel(
            args.root,
            "RPi5 CPU LiteRT 256",
            "rpi5_cpu_frames",
            "rpi5_cpu_metrics.json",
            panel_size,
            source_fps,
            source_frames,
        ),
    ]
    encode(panels, args.output, args.fps, duration)


if __name__ == "__main__":
    main()
