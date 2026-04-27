"""Benchmark RepNeXt on RPi5 CPU, one Coral, and two Coral USB devices."""
import argparse
import json
import os
import statistics
import threading
import time
from pathlib import Path

import numpy as np


def latency_stats(times):
    return {
        "avg_ms": round(statistics.mean(times), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "p50_ms": round(statistics.median(times), 3),
        "p95_ms": round(np.percentile(times, 95), 3),
        "n": len(times),
    }


def make_input(shape, dtype):
    rng = np.random.default_rng(0)
    if dtype == np.int8:
        return rng.integers(-128, 127, size=shape, dtype=np.int8)
    if dtype == np.uint8:
        return rng.integers(0, 255, size=shape, dtype=np.uint8)
    return rng.normal(0, 1, size=shape).astype(np.float32)


def import_tflite():
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        return Interpreter, load_delegate
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
        from tensorflow.lite.python.interpreter import load_delegate
        return Interpreter, load_delegate


def bench_interpreter(interpreter, warmup, runs):
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    x = make_input(input_info["shape"], input_info["dtype"])
    input_idx = input_info["index"]

    def invoke_once():
        interpreter.set_tensor(input_idx, x)
        interpreter.invoke()

    for _ in range(warmup):
        invoke_once()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        invoke_once()
        times.append((time.perf_counter() - t0) * 1000)
    return latency_stats(times), {"shape": input_info["shape"].tolist(), "dtype": str(input_info["dtype"])}


def bench_tflite_cpu(model_path, warmup, runs, threads):
    Interpreter, _ = import_tflite()
    interp = Interpreter(model_path=str(model_path), num_threads=threads)
    stats, input_meta = bench_interpreter(interp, warmup, runs)
    return {"mode": f"tflite_int8_cpu_{threads}t", **stats, "input": input_meta}


def bench_edgetpu(model_path, warmup, runs, device):
    from pycoral.utils.edgetpu import make_interpreter
    import gc

    dev_arg = f"usb:{device}" if device is not None else None
    interp = make_interpreter(str(model_path), device=dev_arg)
    stats, input_meta = bench_interpreter(interp, warmup, runs)
    del interp
    gc.collect()
    time.sleep(1.0)
    return {"mode": "tflite_int8_edgetpu_1x", "device": device, **stats, "input": input_meta}


def bench_edgetpu_data_parallel(model_path, warmup, runs, devices):
    from pycoral.utils.edgetpu import make_interpreter

    interpreters = []
    inputs = []
    for dev in devices:
        interp = make_interpreter(str(model_path), device=f"usb:{dev}")
        interp.allocate_tensors()
        info = interp.get_input_details()[0]
        x = make_input(info["shape"], info["dtype"])
        for _ in range(3):
            interp.set_tensor(info["index"], x)
            interp.invoke()
        interpreters.append(interp)
        inputs.append((info["index"], x))
        time.sleep(0.5)

    def invoke(interp, item):
        input_idx, x = item
        interp.set_tensor(input_idx, x)
        interp.invoke()

    for _ in range(warmup):
        threads = [threading.Thread(target=invoke, args=(interp, item)) for interp, item in zip(interpreters, inputs)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    batch_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        threads = [threading.Thread(target=invoke, args=(interp, item)) for interp, item in zip(interpreters, inputs)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        batch_times.append((time.perf_counter() - t0) * 1000)

    stats = latency_stats(batch_times)
    throughput = len(devices) * 1000.0 / statistics.mean(batch_times)
    return {
        "mode": "tflite_int8_edgetpu_2x_data_parallel",
        "devices": devices,
        **stats,
        "batch_size": len(devices),
        "throughput_ips": round(throughput, 3),
    }


def list_edgetpu_devices():
    try:
        from pycoral.utils.edgetpu import list_edge_tpus

        devices = list_edge_tpus()
        return [str(device) for device in devices]
    except Exception as exc:
        return [f"device_probe_failed: {exc}"]


def bench_edgetpu_pipeline(split_a, split_b, warmup, runs, devices):
    from pycoral.utils.edgetpu import make_interpreter

    first = make_interpreter(str(split_a), device=f"usb:{devices[0]}")
    second = make_interpreter(str(split_b), device=f"usb:{devices[1]}")
    first.allocate_tensors()
    second.allocate_tensors()
    first_in = first.get_input_details()[0]
    first_out = first.get_output_details()[0]
    second_in = second.get_input_details()[0]
    x = make_input(first_in["shape"], first_in["dtype"])

    def invoke_once():
        first.set_tensor(first_in["index"], x)
        first.invoke()
        y = first.get_tensor(first_out["index"])
        if y.dtype != second_in["dtype"]:
            y = y.astype(second_in["dtype"])
        second.set_tensor(second_in["index"], y)
        second.invoke()

    for _ in range(warmup):
        invoke_once()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        invoke_once()
        times.append((time.perf_counter() - t0) * 1000)
    return {"mode": "tflite_int8_edgetpu_2x_pipeline_split", "devices": devices, **latency_stats(times)}


def bench_pytorch_repnext(ckpt, size, warmup, runs, threads):
    import torch
    from export_onnx import RepNeXtSeg

    torch.set_num_threads(threads)
    model = RepNeXtSeg().eval()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("state_dict", sd), strict=False)
    x = torch.randn(1, 3, size, size)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return {"mode": f"pytorch_cpu_{threads}t", **latency_stats(times)}


def append_result(results, mode, fn):
    try:
        results.append(fn())
    except Exception as exc:
        results.append({"mode": mode, "error": str(exc)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default="repnext_m5_ade20k_int8.tflite")
    ap.add_argument("--edgetpu", default="repnext_m5_ade20k_int8_edgetpu.tflite")
    ap.add_argument("--ckpt", default="repnext_m5_ade20k.pth")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--split-a", default="")
    ap.add_argument("--split-b", default="")
    ap.add_argument("--out", default="results_repnext.json")
    ap.add_argument("--skip-pytorch", action="store_true")
    args = ap.parse_args()

    os.chdir(Path(__file__).resolve().parent)
    devices = [int(x) for x in args.devices.split(",") if x.strip()]
    results = []

    if not args.skip_pytorch and Path(args.ckpt).exists():
        append_result(
            results,
            f"pytorch_cpu_{args.threads}t",
            lambda: bench_pytorch_repnext(args.ckpt, args.size, args.warmup, args.runs, args.threads),
        )
    if Path(args.tflite).exists():
        append_result(
            results,
            f"tflite_int8_cpu_{args.threads}t",
            lambda: bench_tflite_cpu(args.tflite, args.warmup, args.runs, args.threads),
        )
    if Path(args.edgetpu).exists():
        append_result(
            results,
            "tflite_int8_edgetpu_1x",
            lambda: bench_edgetpu(args.edgetpu, args.warmup, args.runs, devices[0] if devices else None),
        )
        import gc; gc.collect(); time.sleep(2.0)
        if len(devices) >= 2:
            append_result(
                results,
                "tflite_int8_edgetpu_2x_data_parallel",
                lambda: bench_edgetpu_data_parallel(args.edgetpu, args.warmup, args.runs, devices[:2]),
            )
    if args.split_a and args.split_b:
        append_result(
            results,
            "tflite_int8_edgetpu_2x_pipeline_split",
            lambda: bench_edgetpu_pipeline(args.split_a, args.split_b, args.warmup, args.runs, devices[:2]),
        )

    payload = {
        "model": "RepNeXt-M5 ADE20K",
        "hardware": "RPi5 BCM2712, 4x Cortex-A76, 2x Coral USB",
        "runs": args.runs,
        "warmup": args.warmup,
        "requested_devices": devices,
        "detected_edge_tpus": list_edgetpu_devices(),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
