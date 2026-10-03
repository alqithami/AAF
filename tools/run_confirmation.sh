#!/usr/bin/env bash
# Portable front end; never changes the frozen experiment algorithm or protocol.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE="$ROOT/experiments/AAF_R3_STRENGTHENING_CONFIRM_v1"
MODE="${1:-verify}"
PY="${AAF_RUN_PYTHON:-python}"
command -v "$PY" >/dev/null || { echo 'Activate a compatible environment or set AAF_RUN_PYTHON.' >&2; exit 2; }
case "$MODE" in verify|plan|test|run|analyze|pack|all) ;; *) echo 'Usage: bash tools/run_confirmation.sh verify|plan|test|run|analyze|pack|all [output-directory]' >&2; exit 2;; esac
OUT="$("$PY" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${2:-$ROOT/results/confirmation}")"
export TMPDIR="$ROOT/.runtime/tmp" TMP="$ROOT/.runtime/tmp" TEMP="$ROOT/.runtime/tmp"
export XDG_CACHE_HOME="$ROOT/.runtime/cache" MPLCONFIGDIR="$ROOT/.runtime/cache/matplotlib"
export TORCH_HOME="$ROOT/.runtime/cache/torch" CUDA_CACHE_PATH="$ROOT/.runtime/cache/cuda"
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH"
"$PY" "$ROOT/tools/verify_release.py"
cd "$PACKAGE"
"$PY" confirm.py verify
case "$MODE" in
  verify) exit 0;;
  plan) exec "$PY" confirm.py plan;;
  test) "$PY" -m pytest -q tests; cd base; exec "$PY" -m pytest -q tests vendor/tests;;
esac
# Hold the same outer lock across training, analysis and packaging.
command -v flock >/dev/null || { echo 'Linux flock is required.' >&2; exit 2; }
mkdir -p "$OUT"
exec 9> "$OUT/.publication.lock"
flock -n 9 || { echo 'Another portable workflow owns this output directory.' >&2; exit 2; }
if [[ "$MODE" == all ]]; then
  "$PY" -m pytest -q tests
  (cd base && "$PY" -m pytest -q tests vendor/tests)
fi
if [[ "$MODE" == all || "$MODE" == run ]]; then "$PY" confirm.py run --out "$OUT"; fi
if [[ "$MODE" == all || "$MODE" == analyze ]]; then "$PY" confirm.py analyze --out "$OUT"; fi
if [[ "$MODE" == all || "$MODE" == pack ]]; then
  "$PY" confirm.py pack --out "$OUT" --zip "${OUT}.review.zip"
  echo "Archive: ${OUT}.review.zip"
  echo "Checksum: ${OUT}.review.zip.sha256"
fi
