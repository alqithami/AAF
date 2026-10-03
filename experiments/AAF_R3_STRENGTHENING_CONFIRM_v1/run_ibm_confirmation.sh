#!/usr/bin/env bash
# Uses the already working IBM environment; never installs or changes dependencies.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE="${1:-start}"
PY="${AAF_RUN_PYTHON:-/mnt/caenl/active/aaf-r3-strengthening-v1/AAF_R3_STRENGTHENING_GPU_v1/.venv/bin/python}"
stop() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
[[ "$(uname -s)" == Linux ]] || stop 'Run on the IBM Linux server, not on the Mac.'
if [[ "$MODE" == status ]]; then
    printf '=== STATUS ===\n'; [[ ! -f RUN.status ]] || cat RUN.status
    if [[ -s RUN.pid ]]; then ps -p "$(cat RUN.pid)" -o pid=,etime=,args= || true; fi
    [[ ! -s RUN.exitcode ]] || { printf 'Exit code: '; cat RUN.exitcode; }
    if [[ -s RUN.logpath ]]; then logfile="$(cat RUN.logpath)"; printf '\n=== %s ===\n' "$logfile"; [[ ! -f "$logfile" ]] || tail -n 60 "$logfile"; fi
    exit 0
fi
[[ -x "$PY" ]] || stop "Existing experiment Python not found: $PY. No installation or fallback was attempted."
CACHE=/mnt/caenl/active/aaf-r3-strengthening-v1/runtime-storage
export TMPDIR="$CACHE/tmp" TMP="$CACHE/tmp" TEMP="$CACHE/tmp"
export XDG_CACHE_HOME="$CACHE/cache" MPLCONFIGDIR="$CACHE/cache/matplotlib"
export TORCH_HOME="$CACHE/cache/torch" CUDA_CACHE_PATH="$CACHE/cache/cuda"
export PIP_NO_CACHE_DIR=1 PIP_CACHE_DIR="$CACHE/pip"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$PIP_CACHE_DIR" logs
if [[ "$MODE" == --worker ]]; then
    [[ "${AAF_CONFIRM_WORKER:-}" == 1 && -e /proc/$$/fd/9 ]] || stop 'Start with no argument, not --worker.'
    fail() { rc=$?; trap - ERR; printf '%s\n' "$rc" > RUN.exitcode; printf 'COLLECTING_FAILURE\n' > RUN.status; "$PY" confirm.py failure || true; printf 'FAILED\n' > RUN.status; exit "$rc"; }
    trap fail ERR
    printf 'RUNNING\n' > RUN.status
    "$PY" confirm.py run
    printf 'ANALYZING\n' > RUN.status
    "$PY" confirm.py analyze
    printf 'PACKAGING\n' > RUN.status
    "$PY" confirm.py pack
    unzip -tq AAF_R3_CONFIRMATION_REVIEW.zip
    sha256sum -c AAF_R3_CONFIRMATION_REVIEW.zip.sha256
    printf '0\n' > RUN.exitcode
    printf 'COMPLETE\n' > RUN.status
    echo 'COMPLETE. Return AAF_R3_CONFIRMATION_REVIEW.zip. Keep the full server traces and checkpoints.'
    exit 0
fi
[[ "$MODE" == start ]] || stop 'Use no argument to start, or: status'
command -v flock >/dev/null || stop 'flock is required.'
command -v unzip >/dev/null || stop 'unzip is required.'
exec 9> "$ROOT/.confirmation.lock"
flock -n 9 || stop 'A confirmation worker is already active. Use status; do not start a duplicate.'
if [[ -f RUN.status ]] && grep -qx COMPLETE RUN.status; then
    echo 'Already COMPLETE; no repeat was launched. Download the existing review archive.'; exit 0
fi
"$PY" confirm.py verify
"$PY" confirm.py plan | tee logs/PLAN.json
"$PY" - <<'PY'
import os,shutil,tempfile
free=shutil.disk_usage(os.getcwd()).free/1024**3
print('Free output-storage space: %.1f GiB' % free)
if free<25:raise SystemExit('STOP: at least 25 GiB free output storage is required. Do not delete previous experiments.')
for folder in (os.getcwd(),os.environ['TMPDIR']):
    with tempfile.TemporaryFile(dir=folder) as f:
        f.write(b'0'*1024**2);f.flush();os.fsync(f.fileno())
PY
"$PY" -m pip freeze > logs/packages.txt
"$PY" -m pytest -q tests --junitxml=logs/confirmation_tests.xml | tee logs/confirmation_tests.txt
export AAF_CONFIRM_WORKER=1 AAF_RUN_PYTHON="$PY"
logfile="$ROOT/logs/confirmation_$(date -u +%Y%m%dT%H%M%S)_$$.log"
printf 'STARTING\n' > RUN.status
printf '%s\n' "$logfile" > RUN.logpath
: > RUN.exitcode
nohup bash "$ROOT/run_ibm_confirmation.sh" --worker > "$logfile" 2>&1 < /dev/null 9>&9 &
pid=$!;printf '%s\n' "$pid" > RUN.pid
printf 'LAUNCHED PID %s. Launch is not completion.\n' "$pid"
printf 'Check: bash %s/run_ibm_confirmation.sh status\n' "$ROOT"
