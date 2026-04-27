#!/usr/bin/env python3
"""Benchmark a sequential EdgeTPU pipeline made of multiple TFLite segments.

Example on RPi5:
    python tpu_pipeline_chain.py \
      --segments seg0_edgetpu.tflite seg1_edgetpu.tflite \
      --devices 0,1 --warmup 10 --runs 50 --out result.json
"""
import argparse
import json
import os
import statistics
import time

import numpy as np


def make_input(shape, dtype):
    rng = np.random.default_rng(0)
    if dtype == np.int8:
        return rng.integers(-128, 127, size=shape, dtype=np.int8)
    if dtype == np.uint8:
        return rng.integers(0, 255, size=shape, dtype=np.uint8)
    return rng.normal(0, 1, size=shape).astype(dtype)


def stats(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n": int(arr.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", nargs="+", required=True)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.utils.edgetpu import list_edge_tpus

    devices = [int(x) for x in args.devices.split(",") if x.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one device id")

    interpreters = []
    metas = []
    for idx, segment in enumerate(args.segments):
        dev = devices[idx % len(devices)]
        interp = make_interpreter(str(segment), device=f"usb:{dev}")
        interp.allocate_tensors()
        input_details = interp.get_input_details()
        output_details = interp.get_output_details()
        interpreters.append(interp)
        metas.append({
            "segment": segment,
            "device": dev,
            "inputs": [
                {
                    "index": int(item["index"]),
                    "name": item["name"],
                    "shape": [int(x) for x in item["shape"]],
                    "dtype": str(item["dtype"]),
                }
                for item in input_details
            ],
            "outputs": [
                {
                    "index": int(item["index"]),
                    "name": item["name"],
                    "shape": [int(x) for x in item["shape"]],
                    "dtype": str(item["dtype"]),
                }
                for item in output_details
            ],
        })

    first_input = make_input(
        interpreters[0].get_input_details()[0]["shape"],
        interpreters[0].get_input_details()[0]["dtype"],
    )

    def invoke_chain(measure=False):
        tensor_pool = {}
        invoke_times = []
        copy_times = []
        for idx, interp in enumerate(interpreters):
            stage_copy_ms = 0.0
            for input_pos, in_info in enumerate(interp.get_input_details()):
                if idx == 0 and input_pos == 0:
                    x = first_input
                else:
                    if in_info["name"] not in tensor_pool:
                        raise KeyError(f"missing boundary tensor for input {in_info['name']!r}")
                    x = tensor_pool[in_info["name"]]
                if x.dtype != in_info["dtype"]:
                    t_copy = time.perf_counter()
                    x = x.astype(in_info["dtype"])
                    stage_copy_ms += (time.perf_counter() - t_copy) * 1000

                t_copy = time.perf_counter()
                interp.set_tensor(in_info["index"], x)
                stage_copy_ms += (time.perf_counter() - t_copy) * 1000

            t_invoke = time.perf_counter()
            interp.invoke()
            invoke_ms = (time.perf_counter() - t_invoke) * 1000

            for out_info in interp.get_output_details():
                t_copy = time.perf_counter()
                tensor_pool[out_info["name"]] = interp.get_tensor(out_info["index"])
                stage_copy_ms += (time.perf_counter() - t_copy) * 1000

            if measure:
                invoke_times.append(invoke_ms)
                copy_times.append(stage_copy_ms)
        return invoke_times, copy_times

    for _ in range(args.warmup):
        invoke_chain(measure=False)

    total_times = []
    per_stage_invoke = [[] for _ in interpreters]
    per_stage_copy = [[] for _ in interpreters]
    for _ in range(args.runs):
        t0 = time.perf_counter()
        invoke_times, copy_times = invoke_chain(measure=True)
        total_times.append((time.perf_counter() - t0) * 1000)
        for idx, value in enumerate(invoke_times):
            per_stage_invoke[idx].append(value)
        for idx, value in enumerate(copy_times):
            per_stage_copy[idx].append(value)

    result = {
        "segments": args.segments,
        "devices": devices,
        "detected_edge_tpus": [str(x) for x in list_edge_tpus()],
        "warmup": args.warmup,
        "runs": args.runs,
        "total": stats(total_times),
        "stages": [
            {
                **meta,
                "invoke": stats(per_stage_invoke[idx]),
                "tensor_set_get_cast": stats(per_stage_copy[idx]),
            }
            for idx, meta in enumerate(metas)
        ],
        "throughput_ips": len(args.segments) * 1000.0 / statistics.mean(total_times),
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
