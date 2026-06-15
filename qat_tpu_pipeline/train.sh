#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ADE_ROOT="${ADE_ROOT:-}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-repnext_m5_ade20k.pth}"
OUT="${OUT:-training/checkpoints/w48_192_distill_qat_latest.pth}"
SIZE="${SIZE:-192}"
TEACHER_SIZE="${TEACHER_SIZE:-384}"
TRAIN_LIMIT="${TRAIN_LIMIT:-2000}"
VAL_LIMIT="${VAL_LIMIT:-200}"
BATCH="${BATCH:-16}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-20}"
QAT_EPOCHS="${QAT_EPOCHS:-10}"
LR="${LR:-2e-4}"
WORKERS="${WORKERS:-8}"

if [[ -z "$ADE_ROOT" ]]; then
  echo "Set ADE_ROOT=/path/to/ade20k before running." >&2
  exit 2
fi

python qat_tpu_pipeline/train_qat_distill.py \
  --ade-root "$ADE_ROOT" \
  --teacher-weights "$TEACHER_WEIGHTS" \
  --out "$OUT" \
  --size "$SIZE" \
  --teacher-size "$TEACHER_SIZE" \
  --base-width 48 \
  --depth 4 4 8 2 \
  --fpn-out 96 \
  --head-ch 64 \
  --train-limit "$TRAIN_LIMIT" \
  --val-limit "$VAL_LIMIT" \
  --batch "$BATCH" \
  --distill-epochs "$DISTILL_EPOCHS" \
  --qat-epochs "$QAT_EPOCHS" \
  --lr "$LR" \
  --lambda-kd 1.0 \
  --temperature 2.0 \
  --workers "$WORKERS" \
  --amp
