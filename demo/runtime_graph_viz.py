#!/usr/bin/env python3
"""Render advisor-facing runtime graphs for the RepNeXt optimization project."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "runtime_graphs"


def box(ax, xy, wh, label, fc, ec="#222", fs=9):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=1.4, color="#333"),
    )


def render_pipeline_graph() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.8))
    fig.suptitle("Scene Understanding Optimization via DL Compiler: End-to-End Runtime Graphs", fontsize=15, y=0.98)

    # 1. Old hybrid pipeline.
    ax = axes[0]
    ax.set_title(
        "A. RPi5 ARM CPU w/ Google Coral TPU x2: old hybrid path accelerates only the middle segment",
        fontsize=11,
        loc="left",
    )
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2)
    ax.axis("off")
    y = 0.75
    box(ax, (0.2, y), (1.2, 0.5), "RGB\n512", "#e8f1ff")
    box(ax, (1.8, y), (2.7, 0.5), "RPi5 ARM CPU\nprefix stage0/1\n1706 ms", "#ffd9d9")
    box(ax, (4.9, y), (1.25, 0.5), "quant\n7 ms", "#fff0bf")
    box(ax, (6.5, y), (1.9, 0.5), "Coral TPU\nmiddle\n29 ms", "#d7f7d7")
    box(ax, (8.8, y), (1.25, 0.5), "dequant\n1 ms", "#fff0bf")
    box(ax, (10.4, y), (2.7, 0.5), "RPi5 ARM CPU\nsuffix FPN/head\n2321 ms", "#ffd9d9")
    box(ax, (13.3, y), (0.5, 0.5), "seg", "#e8f1ff")
    for x0, x1 in [(1.4, 1.8), (4.5, 4.9), (6.15, 6.5), (8.4, 8.8), (10.05, 10.4), (13.1, 13.3)]:
        arrow(ax, x0, 1.0, x1, 1.0)
    ax.text(
        0.2,
        0.25,
        "Total: 4064 ms/frame. Lesson: partial offload is not enough; the DL compiler must expose a full accelerator graph.",
        fontsize=10,
    )

    # 2. Accuracy-recovered CPU low-res path.
    ax = axes[1]
    ax.set_title(
        "B. RPi5 ARM CPU w/o Google Coral TPU x2: full-graph compiled LiteRT path gives the best accuracy-valid demo",
        fontsize=11,
        loc="left",
    )
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2)
    ax.axis("off")
    box(ax, (0.2, y), (1.3, 0.5), "RGB\n256", "#e8f1ff")
    box(ax, (2.1, y), (5.0, 0.5), "Full RepNeXt graph\nbackbone + FPN + head\nLiteRT/XNNPACK", "#dceeff")
    box(ax, (7.7, y), (2.0, 0.5), "64x64x150\nlogits", "#e8f1ff")
    box(ax, (10.3, y), (2.4, 0.5), "upsample /\nvisual overlay", "#f1f1f1")
    for x0, x1 in [(1.5, 2.1), (7.1, 7.7), (9.7, 10.3)]:
        arrow(ax, x0, 1.0, x1, 1.0)
    ax.text(
        0.2,
        0.25,
        "Result: 4223 -> 351 ms/frame on RPi5 ARM CPU (12.0x faster) with 0.2135 mIoU.",
        fontsize=10,
        fontweight="bold",
    )

    # 3. Compiler-centric TPU candidate.
    ax = axes[2]
    ax.set_title(
        "C. RPi5 ARM CPU w/ Google Coral TPU x2: DL-compiler-friendly full graph maps completely to Coral TPU",
        fontsize=11,
        loc="left",
    )
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2)
    ax.axis("off")
    box(ax, (0.2, y), (1.3, 0.5), "RGB\n192", "#e8f1ff")
    box(ax, (2.0, y), (1.3, 0.5), "int8\ninput", "#fff0bf")
    box(ax, (3.8, y), (4.6, 0.5), "w48 full CNN graph\n960/960 ops on Coral TPU\n1 subgraph", "#d7f7d7")
    box(ax, (8.9, y), (2.0, 0.5), "48x48x150\nint8 logits", "#e8f1ff")
    box(ax, (11.4, y), (2.0, 0.5), "dequant /\nstitch", "#f1f1f1")
    for x0, x1 in [(1.5, 2.0), (3.3, 3.8), (8.4, 8.9), (10.9, 11.4)]:
        arrow(ax, x0, 1.0, x1, 1.0)
    ax.text(
        0.2,
        0.25,
        "Compiler result: 84 ms/frame, 960/960 TPU ops, 57.75 KiB streaming. Accuracy needs distillation/QAT.",
        fontsize=10,
        fontweight="bold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "end_to_end_runtime_graphs.png", dpi=160)


def render_latency_bars() -> None:
    labels = [
        "RPi5 ARM CPU\nnative PyTorch 512",
        "RPi5 ARM CPU +\nCoral TPU x2 hybrid",
        "RPi5 ARM CPU\nLiteRT 256\naccuracy-valid",
        "Intel CPU\ncompiled pipeline",
        "RPi5 ARM CPU +\nCoral TPU x2\nw48 compiler-clean",
    ]
    values = [4222.626, 4064.330, 351.377, 2210.749, 84.130]
    colors = ["#cc5c5c", "#d8893a", "#4c78a8", "#8f7aa8", "#2f9e44"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Latency (ms/frame)")
    ax.set_title("Optimization Result: DL-Compiler Paths Turn Multi-Second Inference into Demo-Ready Latency")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25, which="both")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.08, f"{value:.1f} ms", ha="center", va="bottom", fontsize=9)
    fig.text(
        0.5,
        0.02,
        "Best accuracy-valid demo: RPi5 ARM CPU LiteRT 256 = 12.0x faster than native. "
        "Best TPU compiler result: full graph mapped to Coral TPU = 84 ms/frame.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "latency_comparison_logscale.png", dpi=160)


def annotate_bars(ax, bars, values, suffix=" ms"):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.04,
            f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def paired_track_plot(filename, title, labels, latency, accuracy, colors, note, acc_ylim=(0, 0.26)):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    x = range(len(labels))
    bars = ax.bar(x, latency, color=colors)
    ax.set_xticks(list(x), labels)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms/frame, log)")
    ax.set_title("Latency")
    ax.grid(axis="y", alpha=0.25, which="both")
    annotate_bars(ax, bars, latency)

    ax = axes[1]
    bars = ax.bar(x, accuracy, color=colors)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(*acc_ylim)
    ax.set_ylabel("ADE20K mIoU")
    ax.set_title("Accuracy")
    ax.grid(axis="y", alpha=0.25)
    ymin, ymax = acc_ylim
    label_offset = (ymax - ymin) * 0.03
    for bar, value in zip(bars, accuracy):
        y_text = min(value + label_offset, ymax - label_offset)
        ax.text(bar.get_x() + bar.get_width() / 2, y_text, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=14)
    fig.text(0.5, 0.02, note, ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename, dpi=160)


def render_paired_track_comparisons() -> None:
    paired_track_plot(
        "intel_cpu_latency_accuracy.png",
        "Intel CPU Track: Native vs Compiler Optimized",
        ["Native", "torch.compile", "OpenVINO+\nLiteRT"],
        [3972.223, 2608.011, 2210.749],
        [0.22245, 0.22245, 0.00338],
        ["#cc5c5c", "#6c8ebf", "#2f9e44"],
        "Accuracy is ADE20K val40. OpenVINO+LiteRT used the ReLU/sparse model family, which is fast but not accuracy-valid.",
        acc_ylim=(0, 0.26),
    )
    paired_track_plot(
        "rpi5_cpu_latency_accuracy.png",
        "Raspberry Pi 5 ARM CPU w/o Coral TPU x2: Native vs Optimized",
        ["Native", "LiteRT 256"],
        [4222.626, 351.377],
        [0.22245, 0.21347],
        ["#cc5c5c", "#2f9e44"],
        "Same accuracy protocol where available: ADE20K val40. LiteRT 256 is dynamic-range quantized and is the best accuracy-valid demo.",
        acc_ylim=(0, 0.26),
    )
    paired_track_plot(
        "coral_tpu_latency_accuracy.png",
        "Raspberry Pi 5 ARM CPU w/ Coral TPU x2: TPU Baseline vs Optimized",
        ["Partial\noffload", "Low-res\nTPU", "w48\nTPU"],
        [4064.330, 184.880, 84.130],
        [0.00338, 0.00861, 0.00309],
        ["#d8893a", "#72b7b2", "#2f9e44"],
        "All shown TPU-track variants are fast/compiler-useful but not accuracy-valid yet; QAT/distillation is required.",
        acc_ylim=(0, 0.01),
    )


def render_track_latency() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tracks = [
        (
            "intel_cpu_latency.png",
            "Intel CPU: Native vs DL-Compiler Optimized",
            ["Intel PyTorch\nbaseline", "Intel torch.compile", "Intel OpenVINO+LiteRT\npersistent"],
            [3972.223, 2608.011, 2210.749],
            ["#cc5c5c", "#6c8ebf", "#2f9e44"],
            "Best Intel result: 3972.2 -> 2210.7 ms/frame (1.80x faster) on the same i5 CPU.",
        ),
        (
            "rpi5_cpu_latency.png",
            "Raspberry Pi 5 ARM CPU w/o Coral TPU x2: Native vs Optimized",
            ["RPi5 PyTorch\n512 native", "RPi5 TFLite\nmiddle hybrid", "RPi5 LiteRT\n256 dyn-range"],
            [4222.626, 3652.424, 351.377],
            ["#cc5c5c", "#d8893a", "#2f9e44"],
            "Best accuracy-valid edge demo: 4222.6 -> 351.4 ms/frame (12.0x faster) on the same RPi5 ARM CPU.",
        ),
        (
            "coral_tpu_latency.png",
            "Raspberry Pi 5 ARM CPU w/ Coral TPU x2: Hybrid vs TPU-Compiler Optimized",
            ["Hybrid TPU\nmiddle only", "Full RepNeXt 96\nall ops mapped", "w48 192\ncompiler-clean"],
            [4064.330, 184.880, 84.130],
            ["#d8893a", "#72b7b2", "#2f9e44"],
            "Best TPU compiler result: 4064.3 -> 84.1 ms/frame (48.3x faster), but the w48 model still needs QAT/distillation.",
        ),
    ]

    for filename, title, labels, values, colors, note in tracks:
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        bars = ax.bar(labels, values, color=colors)
        ax.set_yscale("log")
        ax.set_ylabel("Latency (ms/frame, log scale)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, which="both")
        annotate_bars(ax, bars, values)
        fig.text(0.5, 0.02, note, ha="center", fontsize=8.5)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(OUT / filename, dpi=160)


def render_accuracy_graph() -> None:
    labels = [
        "LiteRT 96\nfloat32",
        "Low-res TPU\nINT8",
        "LiteRT 256\ndyn-range",
        "LiteRT 256\nfloat16",
        "LiteRT 256\nINT8",
    ]
    miou = [0.0648, 0.0086, 0.2135, 0.2155, 0.0027]
    colors = ["#8f7aa8", "#cc5c5c", "#2f9e44", "#72b7b2", "#d8893a"]

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    bars = ax.bar(labels, miou, color=colors)
    ax.set_ylabel("ADE20K mIoU (higher is better)")
    ax.set_ylim(0, 0.30)
    ax.set_title("Accuracy Ablation on the Same ADE20K val40 Protocol")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, miou):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.4f}", ha="center", fontsize=9)
    fig.text(
        0.5,
        0.02,
        "Same ADE20K val40 protocol. Main accuracy losses: low resolution and full INT8 activations.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "accuracy_ablation_miou.png", dpi=160)


def main() -> None:
    render_pipeline_graph()
    render_latency_bars()
    render_track_latency()
    render_paired_track_comparisons()
    render_accuracy_graph()
    print(f"wrote {OUT / 'end_to_end_runtime_graphs.png'}")
    print(f"wrote {OUT / 'latency_comparison_logscale.png'}")
    print(f"wrote {OUT / 'intel_cpu_latency.png'}")
    print(f"wrote {OUT / 'rpi5_cpu_latency.png'}")
    print(f"wrote {OUT / 'coral_tpu_latency.png'}")
    print(f"wrote {OUT / 'accuracy_ablation_miou.png'}")
    print(f"wrote {OUT / 'intel_cpu_latency_accuracy.png'}")
    print(f"wrote {OUT / 'rpi5_cpu_latency_accuracy.png'}")
    print(f"wrote {OUT / 'coral_tpu_latency_accuracy.png'}")


if __name__ == "__main__":
    main()
