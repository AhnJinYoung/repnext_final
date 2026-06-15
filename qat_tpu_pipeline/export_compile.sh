#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ADE_ROOT="${ADE_ROOT:-}"
CKPT="${CKPT:-training/checkpoints/w48_192_distill_qat_latest_best.pth}"
SIZE="${SIZE:-192}"
OUT_DIR="${OUT_DIR:-build/w48_192_qat_edgetpu}"
LOG_DIR="${LOG_DIR:-logs/w48_192_qat_edgetpu}"
CALIB_SAMPLES="${CALIB_SAMPLES:-200}"
ONNX2TF_BIN="${ONNX2TF_BIN:-onnx2tf}"
EDGETPU_COMPILER="${EDGETPU_COMPILER:-edgetpu_compiler}"

if [[ -z "$ADE_ROOT" ]]; then
  echo "Set ADE_ROOT=/path/to/ade20k before running." >&2
  exit 2
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 2
fi

bash scripts/export_compile_edgetpu.sh \
  --ckpt "$CKPT" \
  --ade-root "$ADE_ROOT" \
  --size "$SIZE" \
  --out-dir "$OUT_DIR" \
  --log-dir "$LOG_DIR" \
  --python "$(which python)" \
  --onnx2tf "$ONNX2TF_BIN" \
  --compiler "$EDGETPU_COMPILER" \
  --calib-samples "$CALIB_SAMPLES"
