#!/usr/bin/env python3
"""TVM CPU baseline / tuned benchmark for RepNeXt-M5 stage2 downsample.

Usage (RPi5):
    PYTHONPATH=$HOME/tvm-src/python python3 tvm_cpu_benchmark.py \
        --onnx ~/repnext-pipeline/repnext_m5_relu_tpu_stage2_downsample_512_simplified_kernelshape.onnx \
        --tag iter1_cpu_baseline \
        --out  ~/repnext-pipeline/runs/20260427_iter1_cpu_baseline/result.json \
        --warmup 10 --runs 50
"""
import argparse, json, time, os, sys
import numpy as np
import onnx
import tvm
from tvm import relay
from tvm.contrib import graph_executor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--input-name", default=None)
    ap.add_argument("--shape", default="1,3,512,512",
                    help="comma-separated shape, NCHW (ONNX expects NCHW)")
    ap.add_argument("--target",
                    default="llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a76 "
                            "-mattr=+v8.2a,+fullfp16,+dotprod")
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--disabled-pass", action="append", default=[],
                    help="TVM pass name to disable; can be repeated")
    args = ap.parse_args()

    shape = tuple(int(x) for x in args.shape.split(","))

    onnx_model = onnx.load(args.onnx)
    if args.input_name is None:
        args.input_name = onnx_model.graph.input[0].name
    print(f"[load] onnx={args.onnx} input={args.input_name} shape={shape}", flush=True)

    mod, params = relay.frontend.from_onnx(
        onnx_model, shape={args.input_name: shape}, freeze_params=True)

    target = tvm.target.Target(args.target)
    print(f"[build] target={target} opt={args.opt_level}", flush=True)
    t0 = time.perf_counter()
    with tvm.transform.PassContext(opt_level=args.opt_level, disabled_pass=args.disabled_pass):
        lib = relay.build(mod, target=target, params=params)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"[build] done in {build_ms:.0f} ms", flush=True)

    dev = tvm.cpu(0)
    gm = graph_executor.GraphModule(lib["default"](dev))
    x = np.random.rand(*shape).astype("float32")
    gm.set_input(args.input_name, tvm.nd.array(x))

    for _ in range(args.warmup):
        gm.run()
    times = []
    for _ in range(args.runs):
        t = time.perf_counter()
        gm.run()
        times.append((time.perf_counter() - t) * 1000)
    arr = np.array(times)
    res = {
        "model": "RepNeXt-M5 ADE20K stage2_downsample",
        "tag": args.tag,
        "framework": "TVM",
        "tvm_version": tvm.__version__,
        "target": args.target,
        "opt_level": args.opt_level,
        "disabled_pass": args.disabled_pass,
        "shape": list(shape),
        "warmup": args.warmup,
        "runs": args.runs,
        "build_ms": build_ms,
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
