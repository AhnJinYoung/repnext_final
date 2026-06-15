#!/usr/bin/env python3
"""Build an INT8 calibration tensor from real ADE20K images.

The existing low-res build sweep calibrated INT8 quantization on random Gaussian
noise (``rng.normal(0, 1, ...)``), whose distribution and range do not match the
real model input (preprocessed RGB in ``[0, 1]``). That mis-calibrates every
activation range and is a major contributor to the INT8 accuracy collapse. This
script produces a representative NHWC float32 calibration array from real ADE20K
images instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ade-root", type=Path, default=Path("/workspace/tvm/datasets/ade20k"))
    parser.add_argument("--split", choices=["training", "validation"], default="training")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--stride", type=int, default=37,
                        help="pick every Nth image for diversity across the split")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    img_dir = args.ade_root / "ADEChallengeData2016" / "images" / args.split
    paths = sorted(img_dir.glob("*.jpg"))[:: args.stride][: args.samples]
    if not paths:
        raise SystemExit(f"no images found under {img_dir}")

    arr = np.empty((len(paths), args.size, args.size, 3), dtype=np.float32)
    for i, path in enumerate(paths):
        image = Image.open(path).convert("RGB").resize((args.size, args.size), Image.Resampling.BILINEAR)
        arr[i] = np.asarray(image, dtype=np.float32) / 255.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, arr)
    print(f"wrote {args.out} shape={arr.shape} "
          f"range=[{arr.min():.3f},{arr.max():.3f}] mean={arr.mean():.3f}")


if __name__ == "__main__":
    main()
