#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
export AAF_TEST_DEVICE=cuda
PY="${AAF_RUN_PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then echo 'Run bash setup_gpu.sh first, or set AAF_RUN_PYTHON to an isolated compatible Python executable.'; exit 2; fi
exec > >(tee -a logs/run.log) 2>&1
trap 'rc=$?; trap - ERR; echo "STOPPED: do not delete old results or launch a larger run."; "$PY" collect_failure.py || true; exit "$rc"' ERR
"$PY" verify_package.py
echo 'AAF strengthening development diagnostics. No old project is modified.'
# Hardware inspection only: never kill processes, reset GPUs, or change drivers.
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv; fi
"$PY" -m pip freeze > logs/packages.txt
"$PY" run_strengthening.py plan --profile diagnostic > logs/PLAN.json
"$PY" run_strengthening.py preflight --device cuda --out results/preflight
PYTHONPATH="$PWD/vendor:$PWD${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m pytest -q tests vendor/tests --junitxml=logs/tests.xml | tee logs/tests.txt
"$PY" run_strengthening.py run --profile smoke --suite all --device cuda --out results/smoke
"$PY" run_strengthening.py analyze --out results/smoke
"$PY" run_strengthening.py run --profile diagnostic --suite all --device cuda --out results/development
"$PY" run_strengthening.py analyze --out results/development
mkdir -p results/development/execution_logs
cp logs/run.log logs/packages.txt logs/PLAN.json logs/tests.txt logs/tests.xml results/development/execution_logs/
cp results/preflight/PREFLIGHT.json results/development/execution_logs/INITIAL_PREFLIGHT.json
"$PY" run_strengthening.py pack --out results/development --zip AAF_R3_STRENGTHENING_RESULTS.zip
echo 'FINISHED. Upload AAF_R3_STRENGTHENING_RESULTS.zip. No main confirmation run is started.'
