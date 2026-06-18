#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOL_DIR="${ROOT_DIR}/smolvlm2_optimization"
WORK_DIR="${WORK_DIR:-${SMOL_DIR}/runs/a100_smolvlm2_2b}"
VENV_DIR="${VENV_DIR:-${WORK_DIR}/.venv}"
MODEL_ID="${MODEL_ID:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
BENCH_ITERS="${BENCH_ITERS:-5}"
DEMO_SECONDS="${DEMO_SECONDS:-10}"
VIDEO_DIR="${VIDEO_DIR:-${ROOT_DIR}/demo/video_sources_eye_new}"

VIDEOS=(
  "${VIDEO_DIR}/source_busy_city_street.mp4"
  "${VIDEO_DIR}/source_students_university.mp4"
  "${VIDEO_DIR}/source_anonymous_woman_street.mp4"
)

mkdir -p "${WORK_DIR}"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${SMOL_DIR}/requirements-smolvlm2.txt"

if ! python - <<'PY'
import tvm  # noqa: F401
PY
then
  python -m pip install apache-tvm || \
  python -m pip install --pre -f https://tlcpack.ai/wheels tlcpack-nightly-cu121
fi

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("flash_attn") else 1)
PY
then
  echo "flash-attn already installed"
else
  FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE python -m pip install flash-attn==2.7.4.post1 --no-build-isolation || \
    echo "flash-attn install failed; pipeline will use SDPA/eager attention fallback"
fi

python "${SMOL_DIR}/smolvlm2_tvm_pipeline.py" \
  --model-id "${MODEL_ID}" \
  --work-dir "${WORK_DIR}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --bench-iters "${BENCH_ITERS}" \
  --demo-seconds "${DEMO_SECONDS}" \
  --videos "${VIDEOS[@]}"
