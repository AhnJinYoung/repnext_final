#!/usr/bin/env python3
"""ADE20K validation accuracy/latency benchmark for RepNeXt variants."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = 150
IGNORE_INDEX = 255


def add_paths() -> None:
    for path in (ROOT / "benchmark", ROOT / "conversion", ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


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
        "n": len(values),
    }


def find_pairs(ade_root: Path, limit: int | None) -> list[tuple[Path, Path]]:
    base = ade_root / "ADEChallengeData2016"
    image_dir = base / "images" / "validation"
    mask_dir = base / "annotations" / "validation"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Expected ADEChallengeData2016 validation layout under {ade_root}. "
            "Need images/validation and annotations/validation."
        )
    pairs = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        mask_path = mask_dir / f"{image_path.stem}.png"
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def preprocess_image(path: Path, size: int, normalize: str) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    if normalize == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
    elif normalize == "minus-one-one":
        arr = arr * 2.0 - 1.0
    elif normalize != "zero-one":
        raise ValueError(f"unknown normalize mode: {normalize}")
    return arr


def load_mask(path: Path, size: int) -> np.ndarray:
    mask = Image.open(path).resize((size, size), Image.Resampling.NEAREST)
    raw = np.asarray(mask).astype(np.int64)
    # ADEChallengeData2016 annotations: 0 is ignore, 1..150 are classes.
    out = raw - 1
    out[raw == 0] = IGNORE_INDEX
    out[(out < 0) | (out >= NUM_CLASSES)] = IGNORE_INDEX
    return out


def upsample_logits(logits: np.ndarray, size: int) -> np.ndarray:
    """Return HWC logits resized to size x size."""
    if logits.shape[0] == size and logits.shape[1] == size:
        return logits
    try:
        import cv2  # type: ignore

        return cv2.resize(logits, (size, size), interpolation=cv2.INTER_LINEAR)
    except Exception:
        channels = []
        for c in range(logits.shape[-1]):
            img = Image.fromarray(logits[..., c].astype(np.float32), mode="F")
            img = img.resize((size, size), Image.Resampling.BILINEAR)
            channels.append(np.asarray(img, dtype=np.float32))
        return np.stack(channels, axis=-1)


def confusion_from_pred(confusion: np.ndarray, pred: np.ndarray, target: np.ndarray) -> None:
    valid = target != IGNORE_INDEX
    pred = pred[valid].astype(np.int64)
    target = target[valid].astype(np.int64)
    keep = (pred >= 0) & (pred < NUM_CLASSES) & (target >= 0) & (target < NUM_CLASSES)
    idx = target[keep] * NUM_CLASSES + pred[keep]
    counts = np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES)
    confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES)


def metrics(confusion: np.ndarray) -> dict[str, float | int]:
    tp = np.diag(confusion).astype(np.float64)
    gt = confusion.sum(axis=1).astype(np.float64)
    pred = confusion.sum(axis=0).astype(np.float64)
    denom = gt + pred - tp
    valid = denom > 0
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=valid)
    acc = tp.sum() / max(confusion.sum(), 1)
    mean_acc = np.divide(tp, gt, out=np.zeros_like(tp), where=gt > 0)
    return {
        "mIoU": float(iou[valid].mean()) if valid.any() else 0.0,
        "mAcc": float(mean_acc[gt > 0].mean()) if (gt > 0).any() else 0.0,
        "pixel_acc": float(acc),
        "classes_present": int(valid.sum()),
        "valid_pixels": int(confusion.sum()),
    }


class TfliteBackend:
    def __init__(self, model_path: Path, threads: int):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except Exception:
            try:
                from tflite_runtime.interpreter import Interpreter
            except Exception:
                from tensorflow.lite.python.interpreter import Interpreter
        try:
            self.interp = Interpreter(model_path=str(model_path), num_threads=threads)
        except TypeError:
            self.interp = Interpreter(model_path=str(model_path))
        try:
            self.interp.allocate_tensors()
        except RuntimeError as exc:
            if "XNNPACK" not in str(exc):
                raise
            # Some converted RepNeXt FlatBuffers trip XNNPACK locally while the
            # reference interpreter can still run them. Recreate with preserved
            # tensors, which disables default delegates in TF/LiteRT builds.
            try:
                self.interp = Interpreter(
                    model_path=str(model_path),
                    num_threads=threads,
                    experimental_preserve_all_tensors=True,
                )
            except TypeError:
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


class PytorchBackend:
    def __init__(self, weights: Path, activation: str, sparse_equiv_downsample: bool, threads: int):
        import torch

        add_paths()
        import hybrid_e2e_benchmark as hybrid

        torch.set_num_threads(threads)
        self.torch = torch
        self.model, self.meta = hybrid.load_model(str(weights), activation, sparse_equiv_downsample)

    def infer(self, image_hwc: np.ndarray) -> np.ndarray:
        x = self.torch.from_numpy(image_hwc.transpose(2, 0, 1)).unsqueeze(0).contiguous()
        with self.torch.inference_mode():
            y = self.model(x)
        return y[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ade-root", type=Path, required=True)
    parser.add_argument("--backend", choices=["tflite", "pytorch"], required=True)
    parser.add_argument("--model", type=Path, help="TFLite model path")
    parser.add_argument("--weights", type=Path, default=ROOT / "repnext_m5_ade20k.pth")
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--normalize", choices=["zero-one", "imagenet", "minus-one-one"], default="zero-one")
    parser.add_argument("--activation", choices=["gelu", "relu", "tanh-gelu"], default="relu")
    parser.add_argument("--sparse-equiv-downsample", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pairs = find_pairs(args.ade_root, args.limit)
    if args.backend == "tflite":
        if args.model is None:
            raise SystemExit("--model required for tflite backend")
        backend = TfliteBackend(args.model, args.threads)
        backend_meta = {"model": str(args.model)}
    else:
        backend = PytorchBackend(args.weights, args.activation, args.sparse_equiv_downsample, args.threads)
        backend_meta = {"weights": str(args.weights), "activation": args.activation}

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    latencies = []
    for image_path, mask_path in pairs:
        image = preprocess_image(image_path, args.size, args.normalize)
        target = load_mask(mask_path, args.size)
        start = time.perf_counter()
        logits = backend.infer(image)
        latencies.append((time.perf_counter() - start) * 1000)
        logits = upsample_logits(logits, args.size)
        pred = logits.argmax(axis=-1).astype(np.int64)
        confusion_from_pred(confusion, pred, target)

    result = {
        "benchmark": "ADE20K validation semantic segmentation",
        "metric": "mIoU over 150 classes; ADEChallenge labels 1..150 mapped to 0..149, 0 ignored",
        "backend": args.backend,
        "backend_meta": backend_meta,
        "size": args.size,
        "limit": len(pairs),
        "normalize": args.normalize,
        "latency": stats(latencies),
        "accuracy": metrics(confusion),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
