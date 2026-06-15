#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  bash qat_tpu_pipeline/install_env.sh
fi

source "${VENV:-.venv_qat}/bin/activate"
bash qat_tpu_pipeline/train.sh
bash qat_tpu_pipeline/export_compile.sh

echo
echo "Final Edge TPU binaries:"
find "${OUT_DIR:-build/w48_192_qat_edgetpu}" -maxdepth 1 -type f -name '*_edgetpu.tflite' -print | sort
