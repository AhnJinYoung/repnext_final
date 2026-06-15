#!/usr/bin/env python3
"""Build a low-res full-RepNeXt logits TFLite/EdgeTPU model with the accuracy fixes.

Two changes versus ``build_lowres_logits_sweep.py``:
  * activation defaults to ``tanh-gelu`` (EdgeTPU-supported, accuracy-preserving)
    instead of ``relu`` (which collapses mIoU on the GELU-trained checkpoint);
  * INT8 calibration uses a real ADE20K image tensor instead of Gaussian noise.

Pipeline per size: export ONNX -> onnx2tf int8 -> depthwise patch -> edgetpu compile.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], log: Path, env: dict[str, str] | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("[cmd] " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def parse_compile_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    mapped = total = None
    for line in text.splitlines():
        if line.startswith("Total number of operations:"):
            total = int(line.rsplit(":", 1)[1].strip())
        if line.startswith("Number of operations that will run on Edge TPU:"):
            mapped = int(line.rsplit(":", 1)[1].strip())
    return {
        "compiled": "Compilation succeeded!" in text,
        "large_activation_failure": "large activation tensors" in text,
        "total_ops": total,
        "mapped_ops": mapped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[96])
    parser.add_argument("--activation", choices=["gelu", "relu", "tanh-gelu"], default="tanh-gelu")
    parser.add_argument("--calib-dir", type=Path, default=ROOT / "build/calib_real")
    parser.add_argument("--weights", default=str(ROOT / "repnext_m5_ade20k.pth"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "build/20260613_lowres_fixed")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/20260613_lowres_fixed")
    parser.add_argument("--no-compile", action="store_true", help="skip edgetpu_compiler step")
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--convert-python", default="/workspace/tvm/local-convert-env/bin/python3")
    parser.add_argument("--onnx2tf", default="/workspace/tvm/local-convert-env/bin/onnx2tf")
    parser.add_argument("--compiler", default="/usr/bin/edgetpu_compiler")
    args = parser.parse_args()

    tag = args.activation.replace("-", "")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = "/workspace/tvm/local-convert-env/bin:" + env.get("PATH", "")
    results: dict[str, object] = {}

    for size in args.sizes:
        name = f"repnext_m5_{tag}_real_full_{size}_logits"
        onnx = args.out_dir / f"{name}.onnx"
        saved = args.out_dir / f"onnx2tf_{tag}_{size}_logits"
        calib = args.calib_dir / f"calib_real_{size}_nhwc_float32.npy"
        raw_tflite = saved / f"{name}_full_integer_quant.tflite"
        patched = args.out_dir / f"{name}_full_integer_quant_dwpatched.tflite"
        res = {"size": size, "tflite": str(patched), "calib": str(calib)}

        if not calib.exists():
            res["error"] = f"missing calib {calib}; run make_real_calib.py --size {size}"
            results[str(size)] = res
            continue

        rc = run([args.python, "conversion/export_full_tile_logits_onnx.py",
                  "--ckpt", args.weights, "--out", str(onnx), "--size", str(size),
                  "--activation", args.activation, "--sparse-equiv-downsample"],
                 args.log_dir / f"export_{size}.log")
        res["export_rc"] = rc
        if rc:
            results[str(size)] = res
            continue

        rc = run([args.onnx2tf, "-i", str(onnx), "-o", str(saved), "-dsm", "-n",
                  "-oiqt", "-iqd", "int8", "-oqd", "int8",
                  "-cind", "input", str(calib), "[[[[0.0]]]]", "[[[[1.0]]]]"],
                 args.log_dir / f"convert_{size}.log", env=env)
        res["convert_rc"] = rc
        if rc:
            results[str(size)] = res
            continue

        rc = run([args.convert_python, "conversion/patch_depthwise_tflite.py",
                  str(raw_tflite), str(patched)],
                 args.log_dir / f"patch_{size}.log")
        res["patch_rc"] = rc
        if rc:
            results[str(size)] = res
            continue

        if not args.no_compile:
            rc = run([args.compiler, "-s", "-a", "-o", str(args.out_dir), str(patched)],
                     args.log_dir / f"compile_{size}.log")
            res["compile_rc"] = rc
            res["compile"] = parse_compile_log(args.log_dir / f"compile_{size}.log")

        results[str(size)] = res
        print(json.dumps({str(size): res}, indent=2))
        sys.stdout.flush()

    summary = args.out_dir / "build_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {summary}")


if __name__ == "__main__":
    main()
