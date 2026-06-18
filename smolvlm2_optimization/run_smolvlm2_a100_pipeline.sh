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
TVM_TUNING_TRIALS="${TVM_TUNING_TRIALS:-${AUTOTVM_TRIALS:-64}}"
TVM_PIP_PACKAGE="${TVM_PIP_PACKAGE:-apache-tvm==0.14.dev273}"
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
import importlib.util
import tvm
relay = importlib.import_module("tvm.relay")
if importlib.util.find_spec("tvm.autotvm") is None and importlib.util.find_spec("tvm.meta_schedule") is None:
    raise ImportError("neither tvm.autotvm nor tvm.meta_schedule is available")
from tvm.contrib import graph_executor
assert tvm.runtime.enabled("cuda"), "TVM was installed without CUDA runtime support"
PY
then
  python -m pip uninstall -y tvm apache-tvm tlcpack-nightly tlcpack-nightly-cu121 || true
  python -m pip install --pre "${TVM_PIP_PACKAGE}"
fi

python - <<'PY'
import importlib
import importlib.util
import tvm
relay = importlib.import_module("tvm.relay")
has_autotvm = importlib.util.find_spec("tvm.autotvm") is not None
has_meta_schedule = importlib.util.find_spec("tvm.meta_schedule") is not None
if not has_autotvm and not has_meta_schedule:
    raise SystemExit("TVM has neither AutoTVM nor MetaSchedule. Install a TVM build with an auto-tuning module.")
from tvm.contrib import graph_executor
if not tvm.runtime.enabled("cuda"):
    raise SystemExit(
        "TVM CUDA runtime is not enabled. The selected TVM package exposes Relay/AutoTVM, "
        "but it was not built with CUDA. Install a CUDA-enabled TVM build and rerun with "
        "TVM_PIP_PACKAGE pointing to that wheel, or build TVM from source with USE_CUDA=ON."
    )
print("TVM:", getattr(tvm, "__version__", "unknown"), "CUDA enabled:", tvm.runtime.enabled("cuda"))
print("TVM path:", tvm.__file__)
print("Relay import check: ok")
print("AutoTVM available:", has_autotvm)
print("MetaSchedule available:", has_meta_schedule)
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
  --tvm-tuning-trials "${TVM_TUNING_TRIALS}" \
  --videos "${VIDEOS[@]}"
