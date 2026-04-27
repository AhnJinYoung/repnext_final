#!/usr/bin/env python3
"""PyTorch baseline benchmark using torch.compile (default backend).

Loads RepNeXtSeg from export_onnx.py and benchmarks it under PyTorch eager
or torch.compile (default inductor backend) on RPi5 CPU.

Usage (RPi5):
    cd ~/creative_design/RepNeXt-tpu
    python3 ~/repnext-pipeline/runs/<ts>_pytorch_compile/pytorch_compile_baseline.py \
        --weights ~/repnext-pipeline/repnext_m5_ade20k.pth \
        --shape 1,3,512,512 \
        --tag iter1_pytorch_compile \
        --out ~/repnext-pipeline/runs/<ts>_pytorch_compile/result.json \
        --warmup 10 --runs 50 --threads 4 --mode compile
"""
import argparse, json, os, time, sys
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--src-dir", default=os.path.expanduser("~/creative_design/RepNeXt-tpu"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shape", default="1,3,512,512")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mode", choices=["eager", "compile"], default="compile")
    args = ap.parse_args()

    sys.path.insert(0, args.src_dir)
    torch.set_num_threads(args.threads)
    shape = tuple(int(x) for x in args.shape.split(","))

    from export_onnx import RepNeXtSeg
    print(f"[load] RepNeXtSeg, weights={args.weights}", flush=True)
    model = RepNeXtSeg().eval()
    state = torch.load(args.weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    if args.mode == "compile":
        print("[compile] torch.compile(default)", flush=True)
        t0 = time.perf_counter()
        model = torch.compile(model)
        compile_setup_ms = (time.perf_counter() - t0) * 1000
    else:
        compile_setup_ms = 0.0

    x = torch.randn(*shape)

    with torch.inference_mode():
        # first call triggers compile under torch.compile
        t0 = time.perf_counter()
        model(x)
        first_ms = (time.perf_counter() - t0) * 1000
        for _ in range(args.warmup - 1 if args.warmup > 0 else 0):
            model(x)
        times = []
        for _ in range(args.runs):
            t = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t) * 1000)

    arr = np.array(times)
    res = {
        "model": "RepNeXt-M5 ADE20K (full)",
        "tag": args.tag,
        "framework": "PyTorch",
        "torch_version": torch.__version__,
        "mode": args.mode,
        "threads": args.threads,
        "shape": list(shape),
        "warmup": args.warmup,
        "runs": args.runs,
        "compile_setup_ms": compile_setup_ms,
        "first_run_ms": first_ms,
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
