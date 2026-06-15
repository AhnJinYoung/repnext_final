#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-.venv_qat}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
pip install -r qat_tpu_pipeline/requirements.txt

cat <<EOF

Environment ready.

Activate it with:
  source $VENV/bin/activate

Optional, for local Edge TPU compilation:
  sudo apt install edgetpu-compiler

EOF
