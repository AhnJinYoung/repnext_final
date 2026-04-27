"""
Enumerate ONNX op types and classify against Coral EdgeTPU compatibility.

EdgeTPU op support reference: https://coral.ai/docs/edgetpu/models-intro/#supported-operations
After INT8 quantization (ONNX -> TFLite), these ONNX ops map to TFLite ops.
We flag ONNX ops whose TFLite equivalent is unsupported (CPU fallback) or restricted.

Usage:
    python analyze_ops.py model.onnx
"""
import argparse
import collections
import json
import sys
import onnx


# ONNX op -> EdgeTPU compatibility classification.
# OK       : directly maps to an EdgeTPU-supported TFLite op.
# RANK_OK  : supported only when operating on <=3D tensors (rank limit).
# RISKY    : conditionally supported, often falls back to CPU.
# BAD      : not supported by EdgeTPU, forces CPU fallback.
CLASSIFY = {
    # core OK
    "Conv": "OK", "Gemm": "OK", "MatMul": "OK",
    "Add": "OK", "Sub": "OK", "Mul": "OK", "Div": "OK",
    "Concat": "OK", "Split": "OK",
    "Relu": "OK", "Clip": "OK", "Sigmoid": "OK", "Tanh": "OK",
    "MaxPool": "OK", "AveragePool": "OK", "GlobalAveragePool": "OK",
    "Pad": "OK", "Slice": "OK",
    "Softmax": "RANK_OK",
    "Resize": "RISKY",  # bilinear/nearest usually OK; cubic / rank>4 not
    "ConvTranspose": "OK",
    # reshape-ish (ok as long as resulting rank fits)
    "Reshape": "RANK_OK", "Transpose": "RANK_OK",
    "Flatten": "RANK_OK", "Squeeze": "RANK_OK", "Unsqueeze": "RANK_OK",
    "Expand": "RANK_OK",
    "Gather": "RISKY",  # supported as embedding lookup, restrictions
    "ReduceMean": "RANK_OK",
    "ReduceSum": "RANK_OK",
    "ReduceMax": "RANK_OK",
    "Constant": "OK", "Shape": "OK", "Cast": "RISKY",
    "Identity": "OK",
    # NOT supported on EdgeTPU
    "LayerNormalization": "BAD",
    "InstanceNormalization": "BAD",
    "GroupNormalization": "BAD",
    "Erf": "BAD",
    "Gelu": "BAD",
    "HardSwish": "RISKY",
    "HardSigmoid": "RISKY",
    "Elu": "BAD",
    "LeakyRelu": "RISKY",
    "Pow": "RISKY",
    "Exp": "BAD",
    "Log": "BAD",
    "Sqrt": "BAD",
    "ReciprocalSqrt": "BAD",
    "Reciprocal": "BAD",
    "Neg": "OK",
    "Where": "RISKY",
    "ScatterND": "BAD",
    "ArgMax": "RISKY",
    "Einsum": "BAD",
    "Range": "BAD",
    "ConstantOfShape": "OK",
    "Floor": "RISKY",
    "Ceil": "RISKY",
    "Equal": "RISKY", "Less": "RISKY", "Greater": "RISKY",
    "Not": "RISKY",
    "BatchNormalization": "OK",  # folded into conv after INT8 quantization
    "Dropout": "OK",  # no-op at inference
}

LEVEL_ORDER = {"OK": 0, "RANK_OK": 1, "RISKY": 2, "BAD": 3}
LEVEL_LABEL = {
    "OK": "OK (TPU)",
    "RANK_OK": "OK if rank<=3 (TPU)",
    "RISKY": "RISKY (may fall back)",
    "BAD": "UNSUPPORTED (CPU fallback)",
    "UNKNOWN": "UNKNOWN (not in ref table)",
}


def classify(op_type):
    return CLASSIFY.get(op_type, "UNKNOWN")


def analyze(path):
    m = onnx.load(path, load_external_data=False)
    counter = collections.Counter(n.op_type for n in m.graph.node)
    total = sum(counter.values())
    by_class = collections.defaultdict(int)
    for op, n in counter.items():
        by_class[classify(op)] += n
    return m, counter, total, by_class


def format_report(path, counter, total, by_class):
    lines = []
    lines.append(f"# ONNX op analysis: {path}")
    lines.append(f"Total nodes: {total}")
    lines.append("")
    lines.append("## Compatibility summary")
    for lvl in ["OK", "RANK_OK", "RISKY", "BAD", "UNKNOWN"]:
        c = by_class.get(lvl, 0)
        pct = 100.0 * c / total if total else 0.0
        lines.append(f"  {LEVEL_LABEL[lvl]:<30s}: {c:>5d}  ({pct:5.1f}%)")
    lines.append("")
    lines.append("## Per-op breakdown")
    rows = sorted(counter.items(), key=lambda kv: (LEVEL_ORDER.get(classify(kv[0]), 4), -kv[1]))
    lines.append(f"  {'op_type':<28s} {'count':>6s}  class")
    for op, n in rows:
        cls = classify(op)
        lines.append(f"  {op:<28s} {n:>6d}  {LEVEL_LABEL[cls]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to .onnx")
    ap.add_argument("--json", help="write machine-readable summary json to this path")
    args = ap.parse_args()

    m, counter, total, by_class = analyze(args.model)
    report = format_report(args.model, counter, total, by_class)
    print(report)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "model": args.model,
                "total_nodes": total,
                "by_class": dict(by_class),
                "per_op": {op: {"count": n, "class": classify(op)} for op, n in counter.items()},
            }, f, indent=2)
        print(f"\n[json] wrote {args.json}")


if __name__ == "__main__":
    main()
