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
TVM_GIT_REF="${TVM_GIT_REF:-v0.14.0}"
TVM_SOURCE_DIR="${TVM_SOURCE_DIR:-${WORK_DIR}/apache-tvm-src}"
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

check_tvm() {
  python - <<'PY'
import importlib
import importlib.util
import tvm
relay = importlib.import_module("tvm.relay")
if importlib.util.find_spec("tvm.autotvm") is None and importlib.util.find_spec("tvm.meta_schedule") is None:
    raise ImportError("neither tvm.autotvm nor tvm.meta_schedule is available")
from tvm.contrib import graph_executor
assert tvm.runtime.enabled("cuda"), "TVM was installed without CUDA runtime support"
PY
}

build_tvm_from_source() {
  if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required to build TVM from source. Install cmake first." >&2
    return 1
  fi
  if ! command -v git >/dev/null 2>&1; then
    echo "git is required to build TVM from source. Install git first." >&2
    return 1
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is required for a CUDA-enabled TVM build. Make sure CUDA toolkit is in PATH." >&2
    return 1
  fi

  python -m pip install "attrs" "cloudpickle" "decorator" "psutil" "scipy" "tornado"

  if [ ! -d "${TVM_SOURCE_DIR}/.git" ]; then
    rm -rf "${TVM_SOURCE_DIR}"
    git clone --recursive --branch "${TVM_GIT_REF}" https://github.com/apache/tvm.git "${TVM_SOURCE_DIR}"
  else
    git -C "${TVM_SOURCE_DIR}" fetch --tags
    git -C "${TVM_SOURCE_DIR}" checkout "${TVM_GIT_REF}"
    git -C "${TVM_SOURCE_DIR}" submodule update --init --recursive
  fi

  mkdir -p "${TVM_SOURCE_DIR}/build"
  cp "${TVM_SOURCE_DIR}/cmake/config.cmake" "${TVM_SOURCE_DIR}/build/config.cmake"
  python - "${TVM_SOURCE_DIR}/build/config.cmake" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
settings = {
    "USE_CUDA": "ON",
    "USE_LLVM": "ON",
    "USE_CUBLAS": "ON",
    "USE_CUDNN": "OFF",
}
for key, value in settings.items():
    needle = f"set({key} "
    lines = []
    replaced = False
    for line in text.splitlines():
        if line.startswith(needle):
            lines.append(f"set({key} {value})")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.append(f"set({key} {value})")
    text = "\n".join(lines) + "\n"
path.write_text(text)
PY

  if command -v ninja >/dev/null 2>&1; then
    cmake -S "${TVM_SOURCE_DIR}" -B "${TVM_SOURCE_DIR}/build" -G Ninja
  else
    cmake -S "${TVM_SOURCE_DIR}" -B "${TVM_SOURCE_DIR}/build"
  fi
  cmake --build "${TVM_SOURCE_DIR}/build" --parallel "$(nproc)"
  export TVM_LIBRARY_PATH="${TVM_SOURCE_DIR}/build"
  export LD_LIBRARY_PATH="${TVM_SOURCE_DIR}/build:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${TVM_SOURCE_DIR}/python:${PYTHONPATH:-}"
  python -m pip install -e "${TVM_SOURCE_DIR}/python"
}

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
  if ! python -m pip install --pre "${TVM_PIP_PACKAGE}"; then
    build_tvm_from_source
  fi
fi

if ! check_tvm; then
  echo "Installed TVM still does not expose Relay/AutoTVM with CUDA; rebuilding from source." >&2
  python -m pip uninstall -y tvm apache-tvm tlcpack-nightly tlcpack-nightly-cu121 || true
  build_tvm_from_source
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
