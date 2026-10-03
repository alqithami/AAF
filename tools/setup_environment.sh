#!/usr/bin/env bash
# Portable installer. Environment/cache live outside the immutable experiment packages.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-cpu}"
[[ "$MODE" == cpu || "$MODE" == cuda ]] || { echo 'Usage: bash tools/setup_environment.sh cpu|cuda' >&2; exit 2; }
[[ "$(uname -s)" == Linux ]] || { echo 'This pinned installer targets Linux. No system changes were made.' >&2; exit 2; }
PY="${AAF_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for CANDIDATE in python3.11 python3.12; do
    if command -v "$CANDIDATE" >/dev/null; then PY="$CANDIDATE"; break; fi
  done
fi
[[ -n "$PY" ]] || { echo 'Python 3.11 or 3.12 required; set AAF_PYTHON to its executable.' >&2; exit 2; }
"$PY" -c 'import sys; assert (3,11)<=sys.version_info[:2]<(3,13), sys.version'
ENV="$ROOT/.venv-aaf-$MODE"
[[ ! -L "$ENV" ]] || { echo 'Refusing a symlinked environment.' >&2; exit 2; }
if [[ -e "$ENV" && ! -f "$ENV/AAF_CODE_ENVIRONMENT" ]]; then
  echo 'An unrelated environment exists at the destination; it was not modified.' >&2; exit 2
fi
if [[ ! -e "$ENV" ]]; then "$PY" -m venv "$ENV"; printf '%s\n' "$MODE" > "$ENV/AAF_CODE_ENVIRONMENT"; fi
export PIP_NO_CACHE_DIR=1
export TMPDIR="$ROOT/.runtime/tmp" XDG_CACHE_HOME="$ROOT/.runtime/cache"
export PIP_CACHE_DIR="$ROOT/.runtime/pip"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"
PY="$ENV/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel
INDEX=cpu; [[ "$MODE" != cuda ]] || INDEX=cu128
"$PY" -m pip install 'torch==2.10.0' --index-url "https://download.pytorch.org/whl/$INDEX"
"$PY" -m pip install -r "$ROOT/experiments/AAF_R3_STRENGTHENING_CONFIRM_v1/base/requirements.txt"
"$PY" -m pip check
"$PY" "$ROOT/tools/verify_release.py"
printf '\nActivate: source "%s/bin/activate"\n' "$ENV"
echo 'No experiment was launched and no GPU driver was changed.'
