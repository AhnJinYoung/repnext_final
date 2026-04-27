"""Generate a partition report from EdgeTPU compiler logs and benchmark JSON."""
import argparse
import json
import re
from pathlib import Path


def parse_compiler_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    ops = []
    pattern = re.compile(r"^\s*([A-Z0-9_]+)\s+(\d+)\s+(.+?)\s*$")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        op, count, status = match.groups()
        if op in {"Operator", "Total"}:
            continue
        count = int(count)
        mapped = "Mapped to Edge TPU" in status
        ops.append({"op": op, "count": count, "status": status, "mapped": mapped})
    mapped_count = sum(item["count"] for item in ops if item["mapped"])
    total_count = sum(item["count"] for item in ops)
    cpu_count = total_count - mapped_count
    return {
        "path": str(path),
        "ops": ops,
        "mapped_count": mapped_count,
        "cpu_count": cpu_count,
        "total_count": total_count,
        "mapped_pct": (mapped_count / total_count * 100.0) if total_count else None,
    }


def load_results(path):
    if not path or not Path(path).exists():
        return []
    return json.loads(Path(path).read_text(encoding="utf-8")).get("results", [])


def fmt_ms(item):
    if not item:
        return "pending"
    return f"{item.get('avg_ms', 0):.3f}"


def find_mode(results, needle):
    for item in results:
        if needle in item.get("mode", ""):
            return item
    return None


def make_report(model_name, log_info, results, out_path):
    pt = find_mode(results, "pytorch_cpu")
    tflite_cpu = find_mode(results, "tflite_int8_cpu")
    tpu1 = find_mode(results, "edgetpu_1x")
    tpu2 = find_mode(results, "data_parallel")
    pipe = find_mode(results, "pipeline_split")

    mapped = "pending"
    if log_info["mapped_pct"] is not None:
        mapped = f"{log_info['mapped_pct']:.1f}% ({log_info['mapped_count']}/{log_info['total_count']})"

    op_rows = []
    for item in log_info["ops"]:
        target = "TPU" if item["mapped"] else "CPU"
        op_rows.append(f"| {item['op']} | {item['count']} | {target} | {item['status']} |")
    if not op_rows:
        op_rows.append("| pending | pending | pending | compiler log not available or unparsable |")

    throughput = "pending"
    if tpu2 and "throughput_ips" in tpu2:
        throughput = f"{tpu2['throughput_ips']:.3f} img/s"

    report = f"""# {model_name} Partition Report

**Target**: Raspberry Pi 5 BCM2712, 4x Cortex-A76 @ 2.4GHz, NEON+FP16+dotprod, 2x Coral USB
**Compiler target for CPU subgraphs**: `llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a76 -mattr=+neon,+fp16,+dotprod`

## Summary

| Configuration | Avg latency (ms) | TPU op mapping | Notes |
|---|---:|---:|---|
| PyTorch CPU (RPi5, 4T) | {fmt_ms(pt)} | n/a | baseline |
| TFLite INT8 CPU (4T) | {fmt_ms(tflite_cpu)} | n/a | CPU interpreter |
| TFLite INT8 + 1x EdgeTPU | {fmt_ms(tpu1)} | {mapped} | automatic delegate partition |
| TFLite INT8 + 2x EdgeTPU data parallel | {fmt_ms(tpu2)} | {mapped} | batch 2 throughput: {throughput} |
| TFLite INT8 + 2x EdgeTPU pipeline split | {fmt_ms(pipe)} | pending | requires split submodels |
| TVM(A76 tuned) CPU subgraph | pending | {mapped} | Phase D, cache-bounded tile search |

## EdgeTPU Compiler Op Placement

| Op | Count | Placement | Compiler status |
|---|---:|---|---|
{chr(10).join(op_rows)}

## 2x TPU Scheduling Notes

Data parallel mode binds two full interpreters with `make_interpreter("model.tflite,:0")` and `make_interpreter("model.tflite,:1")`. It improves throughput for independent frames but does not reduce single-frame latency.

Pipeline split mode needs two compiled submodels with a compatible boundary tensor. For RepNeXt, the useful split point is after a large stage boundary only if the intermediate tensor transfer is cheaper than the saved compute time. This remains pending until split TFLite graphs are produced.

## CPU Fallback / BYOC Notes

RepNeXt's expected fallback is GELU exported as `Erf`. The first mitigation to measure is tanh-GELU export or ReLU fine-tuning because removing CPU fallback transitions can beat TVM on tiny activation-only subgraphs. If fallback remains, TVM should be limited to large conv or matmul blocks and built for Cortex-A76 with cache-bounded schedules.
"""
    Path(out_path).write_text(report, encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="repnext_m5_ade20k_int8_edgetpu.log")
    ap.add_argument("--results", default="results_repnext.json")
    ap.add_argument("--out", default="partition_report.md")
    args = ap.parse_args()

    log_info = parse_compiler_log(args.log) if Path(args.log).exists() else {
        "ops": [],
        "mapped_count": 0,
        "cpu_count": 0,
        "total_count": 0,
        "mapped_pct": None,
    }
    report = make_report("RepNeXt-M5 ADE20K", log_info, load_results(args.results), args.out)
    print(report)


if __name__ == "__main__":
    main()
