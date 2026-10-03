#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi
DEVICE="${DEVICE:-cpu}"
JOBS="${JOBS:-2}"
if [ "$DEVICE" = cuda ]; then JOBS=1; fi
mkdir -p logs
python run_review3.py preflight --device "$DEVICE" --physics 2>&1 | tee logs/pilot_preflight.log
python run_review3.py run --profile pilot --suite all --device "$DEVICE" --jobs "$JOBS" --out results/R3_pilot 2>&1 | tee logs/pilot_run.log
python run_review3.py analyze --out results/R3_pilot 2>&1 | tee logs/pilot_analysis.log
python run_review3.py pack --out results/R3_pilot --zip AAF_R3_PILOT_RESULTS.zip
printf '\nReturn this file for review: %s/AAF_R3_PILOT_RESULTS.zip\n' "$PWD"
printf 'Do not start the main study until the pilot has been reviewed.\n'
