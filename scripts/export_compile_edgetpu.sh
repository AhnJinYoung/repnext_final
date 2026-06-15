#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CKPT=""
ADE_ROOT=""
SIZE="192"
OUT_DIR="build/qat_distill_edgetpu"
LOG_DIR="logs/qat_distill_edgetpu"
PYTHON_BIN="${PYTHON_BIN:-python}"
ONNX2TF_BIN="${ONNX2TF_BIN:-onnx2tf}"
EDGETPU_COMPILER="${EDGETPU_COMPILER:-edgetpu_compiler}"
CALIB_SAMPLES="${CALIB_SAMPLES:-200}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --ade-root) ADE_ROOT="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --onnx2tf) ONNX2TF_BIN="$2"; shift 2 ;;
    --compiler) EDGETPU_COMPILER="$2"; shift 2 ;;
    --calib-samples) CALIB_SAMPLES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CKPT" || -z "$ADE_ROOT" ]]; then
  echo "Usage: $0 --ckpt CHECKPOINT.pth --ade-root /path/to/ade20k [--size 192]" >&2
  exit 2
fi

CALIB_DIR="$OUT_DIR/calib"
mkdir -p "$OUT_DIR" "$LOG_DIR" "$CALIB_DIR"

echo "[1/3] building real-image calibration set"
"$PYTHON_BIN" conversion/make_real_calib.py \
  --ade-root "$ADE_ROOT" \
  --split training \
  --size "$SIZE" \
  --samples "$CALIB_SAMPLES" \
  --out "$CALIB_DIR/calib_real_${SIZE}_nhwc_float32.npy"

echo "[2/3] exporting ONNX -> TFLite INT8 and patching depthwise ops"
"$PYTHON_BIN" conversion/build_lowres_fixed.py \
  --sizes "$SIZE" \
  --activation tanh-gelu \
  --weights "$CKPT" \
  --calib-dir "$CALIB_DIR" \
  --out-dir "$OUT_DIR" \
  --log-dir "$LOG_DIR" \
  --python "$PYTHON_BIN" \
  --convert-python "$PYTHON_BIN" \
  --onnx2tf "$ONNX2TF_BIN" \
  --compiler "$EDGETPU_COMPILER"

echo "[3/3] compiled artifacts"
find "$OUT_DIR" -maxdepth 1 -type f \( -name '*_edgetpu.tflite' -o -name '*_dwpatched.tflite' -o -name '*.log' \) -print | sort

echo
echo "Expected Edge TPU binary:"
find "$OUT_DIR" -maxdepth 1 -type f -name '*_edgetpu.tflite' -print | sort | tail -1
