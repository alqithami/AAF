#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3,10)<=sys.version_info[:2]<(3,14), "Use Python 3.10–3.13, preferably 3.11 or 3.12"'
# Do not change an existing project environment or its GPU installation.
if [ ! -d .venv ]; then "$PYTHON_BIN" -m venv .venv; fi
. .venv/bin/activate
python -m pip install --upgrade pip
if ! python -c 'import torch' >/dev/null 2>&1; then
  if [ "$(uname -s)" = Linux ]; then
    python -m pip install 'torch>=2.2,<2.11' --index-url https://download.pytorch.org/whl/cpu
  else
    python -m pip install 'torch>=2.2,<2.11'
  fi
fi
python -m pip install -r requirements.txt
python -m pip freeze > INSTALLED_ENVIRONMENT.txt
python run_review3.py preflight --device cpu --physics
python -m pytest -q
printf '\nSetup complete. Activate using: source .venv/bin/activate\n'
