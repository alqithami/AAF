#!/bin/bash
#SBATCH --job-name=aaf_grid
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

# Usage (recommended): run one shard per array task
#   sbatch --array=0-19 slurm/run_array.sh configs/grid_q1_deep.jsonl out/grid_q1_deep 20 cuda
#
# Args:
#   1) GRID_PATH   : JSONL grid file
#   2) OUT_ROOT    : root output dir
#   3) NUM_SHARDS  : total shards (must match --array range)
#   4) DEVICE      : cuda|cpu|auto (default cuda)
#   5) LOG_MODE    : summary|agents|steps|full (default summary)

set -euo pipefail

GRID_PATH="${1:-}"
OUT_ROOT="${2:-}"
NUM_SHARDS="${3:-20}"
DEVICE="${4:-cuda}"
LOG_MODE="${5:-summary}"

if [[ -z "$GRID_PATH" || -z "$OUT_ROOT" ]]; then
  echo "Usage: sbatch --array=0-<S-1> slurm/run_array.sh <grid.jsonl> <out_root> <num_shards> [device] [log_mode]"
  exit 1
fi

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Put each shard under its own subdir to avoid write contention.
OUT_DIR="${OUT_ROOT}/shard_${SHARD_ID}"

echo "[INFO] shard_id=${SHARD_ID}/${NUM_SHARDS}"
echo "[INFO] grid=${GRID_PATH}"
echo "[INFO] out=${OUT_DIR}"
echo "[INFO] device=${DEVICE} log_mode=${LOG_MODE}"

python -m scripts.run_grid \
  --grid "$GRID_PATH" \
  --out "$OUT_DIR" \
  --jobs 1 \
  --device "$DEVICE" \
  --log_mode "$LOG_MODE" \
  --shard_id "$SHARD_ID" \
  --num_shards "$NUM_SHARDS" \
  --write_config
