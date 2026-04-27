#!/usr/bin/env python3
"""Run two (or more) EdgeTPU TFLite files on the same Coral device and compare latency.

Usage (RPi5, in coral-env):
    python3 tpu_compare_tflites.py \
        --tflite ~/repnext-pipeline/repnext_stage2_int8_dwpatched_edgetpu.tflite \
                 ~/repnext-pipeline/repnext_m5_relu_tpu_stage2_downsample_512_simplified_kernelshape_full_integer_quant_edgetpu.tflite \
        --tag iter1_tpu_compare \
        --out ~/repnext-pipeline/runs/20260427_iter1_tpu_search/result.json \
        --warmup 10 --runs 50 --device 0
"""
import argparse, json, os, time, hashlib
import numpy as np
import tflite_runtime.interpreter as tflite

EDGETPU_LIB = "libedgetpu.so.1"


def sha8(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def bench(path, device, warmup, runs):
    delegate = tflite.load_delegate(EDGETPU_LIB, options={"device": f"usb:{device}"})
    itp = tflite.Interpreter(model_path=path, experimental_delegates=[delegate])
    itp.allocate_tensors()
    in_det = itp.get_input_details()[0]
    out_det = itp.get_output_details()[0]
    shape = in_det["shape"]
    dtype = in_det["dtype"]
    if dtype == np.int8:
        x = np.random.randint(-128, 127, size=shape, dtype=np.int8)
    elif dtype == np.uint8:
        x = np.random.randint(0, 255, size=shape, dtype=np.uint8)
    else:
        x = np.random.rand(*shape).astype(dtype)
    itp.set_tensor(in_det["index"], x)
    for _ in range(warmup):
        itp.invoke()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        itp.invoke()
        times.append((time.perf_counter() - t) * 1000)
    arr = np.array(times)
    return {
        "tflite": path,
        "sha8": sha8(path),
        "size_bytes": os.path.getsize(path),
        "input_shape": [int(x) for x in shape],
        "input_dtype": str(dtype),
        "device": device,
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n": runs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    out = {
        "tag": args.tag,
        "warmup": args.warmup,
        "runs": args.runs,
        "results": [],
    }
    for p in args.tflite:
        print(f"[bench] {p}", flush=True)
        r = bench(p, args.device, args.warmup, args.runs)
        print(json.dumps(r, indent=2), flush=True)
        out["results"].append(r)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
