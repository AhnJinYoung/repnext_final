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
VIDEO_SAMPLE_FRAMES="${VIDEO_SAMPLE_FRAMES:-8}"
AUTOTVM_TRIALS="${AUTOTVM_TRIALS:-64}"
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
import importlib
import tvm
autotvm = importlib.import_module("tvm.autotvm")
relay = importlib.import_module("tvm.relay")
from tvm.contrib import graph_executor
assert tvm.runtime.enabled("cuda"), "TVM was installed without CUDA runtime support"
PY
then
  python -m pip uninstall -y tvm apache-tvm tlcpack-nightly tlcpack-nightly-cu121 || true
  python -m pip install apache-tvm
fi

python - <<'PY'
import importlib
import tvm
autotvm = importlib.import_module("tvm.autotvm")
relay = importlib.import_module("tvm.relay")
from tvm.contrib import graph_executor
if not tvm.runtime.enabled("cuda"):
    raise SystemExit("TVM CUDA runtime is not enabled. Install a CUDA-enabled TVM wheel/build before running.")
print("TVM:", getattr(tvm, "__version__", "unknown"), "CUDA enabled:", tvm.runtime.enabled("cuda"))
print("TVM path:", tvm.__file__)
print("Relay/AutoTVM import check: ok")
PY

if ! command -v nvcc >/dev/null 2>&1; then
  echo "warning: nvcc is not in PATH. TVM CUDA codegen may fail if the wheel needs nvcc for module export."
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
  --video-sample-frames "${VIDEO_SAMPLE_FRAMES}" \
  --autotvm-trials "${AUTOTVM_TRIALS}" \
  --videos "${VIDEOS[@]}"
