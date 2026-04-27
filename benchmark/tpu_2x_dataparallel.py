#!/usr/bin/env python3
"""Coral USB 2x data-parallel benchmark for RepNeXt-M5 partition.

두 USB Edge TPU 에 같은 모델을 로드하고 워커 스레드로 동시에 invoke 해 throughput 측정.

Usage (RPi5):
    source ~/coral-env/bin/activate
    python3 tpu_2x_dataparallel.py \
        --tflite ~/repnext-pipeline/repnext_stage2_int8_dwpatched_edgetpu.tflite \
        --tag iter2_tpu_2x --out result.json \
        --warmup 10 --runs 50
"""
import argparse, json, os, time, threading, hashlib
import numpy as np
import tflite_runtime.interpreter as tflite

EDGETPU_LIB = "libedgetpu.so.1"


def sha8(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def make_interp(path, device):
    delegate = tflite.load_delegate(EDGETPU_LIB, options={"device": f"usb:{device}"})
    itp = tflite.Interpreter(model_path=path, experimental_delegates=[delegate])
    itp.allocate_tensors()
    return itp


def bench_single(itp, warmup, runs):
    in_det = itp.get_input_details()[0]
    shape, dtype = in_det["shape"], in_det["dtype"]
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
    return times


def bench_parallel(itps, warmup, runs):
    # warmup
    for itp in itps:
        in_det = itp.get_input_details()[0]
        shape, dtype = in_det["shape"], in_det["dtype"]
        if dtype == np.int8:
            x = np.random.randint(-128, 127, size=shape, dtype=np.int8)
        elif dtype == np.uint8:
            x = np.random.randint(0, 255, size=shape, dtype=np.uint8)
        else:
            x = np.random.rand(*shape).astype(dtype)
        itp.set_tensor(in_det["index"], x)
        for _ in range(warmup):
            itp.invoke()

    # measure: each iteration kicks both interpreters concurrently and waits
    iter_times = []
    for _ in range(runs):
        results = [None] * len(itps)
        threads = []
        t0 = time.perf_counter()

        def worker(i):
            itps[i].invoke()
            results[i] = True

        for i in range(len(itps)):
            th = threading.Thread(target=worker, args=(i,))
            threads.append(th)
            th.start()
        for th in threads:
            th.join()
        iter_times.append((time.perf_counter() - t0) * 1000)
    return iter_times


def stats(times, n_per_iter=1):
    arr = np.array(times)
    return {
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n": len(times),
        "throughput_ips": float(n_per_iter * 1000.0 / arr.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    print(f"[load] {args.tflite}  sha8={sha8(args.tflite)}", flush=True)

    itp0 = make_interp(args.tflite, 0)
    itp1 = make_interp(args.tflite, 1)

    print("[bench] device 0 alone", flush=True)
    t_dev0 = bench_single(itp0, args.warmup, args.runs)
    print("[bench] device 1 alone", flush=True)
    t_dev1 = bench_single(itp1, args.warmup, args.runs)
    print("[bench] device 0+1 parallel", flush=True)
    t_par = bench_parallel([itp0, itp1], args.warmup, args.runs)

    out = {
        "tflite": args.tflite,
        "sha8": sha8(args.tflite),
        "tag": args.tag,
        "warmup": args.warmup,
        "runs": args.runs,
        "results": {
            "device0_only": stats(t_dev0),
            "device1_only": stats(t_dev1),
            "data_parallel_2x": {**stats(t_par, n_per_iter=2),
                                 "note": "n_per_iter=2 (one infer per device per iteration)"},
        },
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
