#!/usr/bin/env python3
"""Visual + quantitative comparison of RepNeXt segmentation variants on ADE20K.

Renders a grid of overlays (input | ground truth | each model variant) so the
accuracy collapse of the ReLU / low-resolution / INT8 candidates is visible, and
prints per-variant mIoU / pixel-accuracy over the chosen images.

Runs in the system python (torch + PIL + matplotlib). PyTorch variants are run
here; INT8 TFLite predictions are read from an optional ``.npz`` produced by
``dump_tflite_preds.py`` (which runs in the conversion env).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "conversion"))
import export_onnx  # noqa: E402

NUM_CLASSES = 150
IGNORE_INDEX = 255
EVAL_SIZE = 512  # resolution at which mIoU is scored, matching the val baseline


def ade_palette() -> np.ndarray:
    """Deterministic, visually distinct 150-class palette (golden-ratio HSV)."""
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


def load_variant(activation: str, sparse: bool) -> torch.nn.Module:
    export_onnx.SPARSE_EQUIV_DOWNSAMPLE = sparse
    export_onnx.TPU_FRIENDLY_DOWNSAMPLE = False
    if activation == "relu":
        act_cls = torch.nn.ReLU
    elif activation == "tanh-gelu":
        act_cls = lambda: torch.nn.GELU(approximate="tanh")
    else:
        act_cls = torch.nn.GELU
    model = export_onnx.RepNeXtSeg(act=act_cls).eval()
    ckpt = torch.load(ROOT / "repnext_m5_ade20k.pth", map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    if sparse:
        sd, _ = export_onnx.rewrite_sparse_downsample_weights(sd)
    model.load_state_dict(sd, strict=False)
    return model


def preprocess(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()


def load_gt(path: Path, size: int) -> np.ndarray:
    mask = Image.open(path).resize((size, size), Image.Resampling.NEAREST)
    raw = np.asarray(mask).astype(np.int64)
    out = raw - 1
    out[raw == 0] = IGNORE_INDEX
    out[(out < 0) | (out >= NUM_CLASSES)] = IGNORE_INDEX
    return out


def pred_to_eval(pred_label: np.ndarray, size: int = EVAL_SIZE) -> np.ndarray:
    """Nearest-resize a label map to the scoring resolution."""
    img = Image.fromarray(pred_label.astype(np.int32), mode="I")
    img = img.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(img).astype(np.int64)


def score(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Return (mIoU over present classes, pixel accuracy) ignoring IGNORE_INDEX."""
    valid = gt != IGNORE_INDEX
    p = pred[valid]
    t = gt[valid]
    conf = np.bincount(t * NUM_CLASSES + p, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    tp = np.diag(conf).astype(np.float64)
    denom = conf.sum(1) + conf.sum(0) - tp
    present = denom > 0
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=present)
    miou = float(iou[present].mean()) if present.any() else 0.0
    pacc = float(tp.sum() / max(conf.sum(), 1))
    return miou, pacc


def colorize(label: np.ndarray) -> np.ndarray:
    safe = np.where((label < 0) | (label >= NUM_CLASSES), 0, label)
    rgb = PALETTE[safe]
    rgb[(label < 0) | (label >= NUM_CLASSES)] = 0  # ignore -> black
    return rgb


def overlay(base_rgb: np.ndarray, label: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    seg = colorize(label).astype(np.float32)
    out = (1 - alpha) * base_rgb.astype(np.float32) + alpha * seg
    return out.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ade-root", type=Path, default=Path("/workspace/tvm/datasets/ade20k"))
    parser.add_argument("--stems", nargs="+", default=[
        "ADE_val_00000001", "ADE_val_00000002", "ADE_val_00000003",
        "ADE_val_00000006", "ADE_val_00000011", "ADE_val_00000015",
    ])
    parser.add_argument("--tflite-npz", type=Path, default=None,
                        help="optional npz of INT8 predictions from dump_tflite_preds.py")
    parser.add_argument("--tflite-title", default="INT8 ReLU @96 (deployed)")
    parser.add_argument("--out", type=Path, default=ROOT / "demo" / "seg_compare" / "comparison.png")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    img_dir = args.ade_root / "ADEChallengeData2016" / "images" / "validation"
    ann_dir = args.ade_root / "ADEChallengeData2016" / "annotations" / "validation"

    # (title, activation, sparse, input_size)
    variants = [
        ("GELU @512 (orig)", "gelu", False, 512),
        ("tanh-GELU @256 (fix)", "tanh-gelu", False, 256),
        ("ReLU @512", "relu", True, 512),
        ("tanh-GELU @96", "tanh-gelu", False, 96),
    ]

    tflite = None
    if args.tflite_npz and args.tflite_npz.exists():
        tflite = np.load(args.tflite_npz)

    col_titles = ["input", "ground truth"] + [v[0] for v in variants]
    if tflite is not None:
        col_titles.append(args.tflite_title)

    n_rows = len(args.stems)
    n_cols = len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.6 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    # Cache models so each is built once.
    models = {(a, s): load_variant(a, s) for _, a, s, _ in variants}
    agg = {t: [] for t in [v[0] for v in variants] + ([args.tflite_title] if tflite is not None else [])}

    for r, stem in enumerate(args.stems):
        base = np.asarray(Image.open(img_dir / f"{stem}.jpg").convert("RGB").resize((EVAL_SIZE, EVAL_SIZE)))
        gt = load_gt(ann_dir / f"{stem}.png", EVAL_SIZE)

        axes[r, 0].imshow(base)
        axes[r, 1].imshow(overlay(base, gt))

        for c, (title, act, sparse, size) in enumerate(variants, start=2):
            x = preprocess(img_dir / f"{stem}.jpg", size)
            with torch.inference_mode():
                logits = models[(act, sparse)](x)
            pred = logits[0].argmax(0).cpu().numpy().astype(np.int64)
            pred_eval = pred_to_eval(pred)
            miou, pacc = score(pred_eval, gt)
            agg[title].append((miou, pacc))
            axes[r, c].imshow(overlay(base, pred_eval))
            axes[r, c].set_xlabel(f"mIoU {miou:.3f} / acc {pacc:.2f}", fontsize=7)

        if tflite is not None:
            pred = tflite[stem].astype(np.int64)
            pred_eval = pred_to_eval(pred)
            miou, pacc = score(pred_eval, gt)
            agg[args.tflite_title].append((miou, pacc))
            axes[r, n_cols - 1].imshow(overlay(base, pred_eval))
            axes[r, n_cols - 1].set_xlabel(f"mIoU {miou:.3f} / acc {pacc:.2f}", fontsize=7)

        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(stem.replace("ADE_val_0000", "#"), fontsize=8)

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=9)

    fig.suptitle("RepNeXt-M5 ADE20K segmentation: variant comparison (overlay)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")

    print("\n=== mean over images ===")
    print(f"{'variant':28s} {'mIoU':>8s} {'pixelAcc':>9s}")
    for title, vals in agg.items():
        if vals:
            arr = np.array(vals)
            print(f"{title:28s} {arr[:,0].mean():8.4f} {arr[:,1].mean():9.4f}")


if __name__ == "__main__":
    main()
