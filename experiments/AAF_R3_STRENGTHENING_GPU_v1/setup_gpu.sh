#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs
exec > >(tee -a logs/setup.log) 2>&1
trap 'rc=$?; trap - ERR; echo "Setup stopped. No existing project environment was modified."; if command -v python3 >/dev/null; then python3 collect_failure.py >/dev/null 2>&1 || true; fi; exit "$rc"' ERR
PY="${AAF_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for candidate in python3.11 python3.12 python3; do
    if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; assert (3,11)<=sys.version_info[:2]<(3,13)' 2>/dev/null; then PY="$candidate"; break; fi
  done
fi
if [[ -z "$PY" ]]; then echo 'Python 3.11 or 3.12 is required. Set AAF_PYTHON to its executable; do not change system Python.'; exit 2; fi
"$PY" -c 'import sys; assert (3,11)<=sys.version_info[:2]<(3,13), sys.version; print(sys.version)'
"$PY" verify_package.py
if [[ -L .venv ]]; then echo 'Refusing a symlinked .venv'; exit 2; fi
if [[ -e .venv && ! -f .venv/AAF_STRENGTHENING_ENV ]]; then
  echo 'An unrelated .venv exists. Extract this package to a fresh directory.'; exit 2
fi
if [[ ! -e .venv ]]; then "$PY" -m venv .venv; touch .venv/AAF_STRENGTHENING_ENV; fi
PY="$PWD/.venv/bin/python"
if [[ ! -f .venv/AAF_INSTALL_COMPLETE ]]; then
  "$PY" -m pip install --upgrade pip setuptools wheel
  "$PY" -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
  "$PY" -m pip install -r requirements.txt
  "$PY" -m pip check
  touch .venv/AAF_INSTALL_COMPLETE
fi
"$PY" -m pip check
"$PY" - <<'PY'
import torch, importlib.metadata as im
assert torch.__version__.split('+')[0]=='2.10.0',torch.__version__
assert im.version('vmas')=='1.5.2'
assert torch.cuda.is_available(),'CUDA unavailable. Return logs; do not replace the system driver.'
x=torch.randn(16,16,device='cuda'); y=x@x; torch.cuda.synchronize()
print('CUDA device:',torch.cuda.get_device_name(0),'Torch:',torch.__version__)
print('Installation ready. Next: bash run_ibm_diagnostics.sh')
PY
