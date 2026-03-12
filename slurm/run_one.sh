#!/bin/bash
#SBATCH --job-name=aaf_one
#SBATCH --output=slurm-one-%j.out
#SBATCH --error=slurm-one-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

CONFIG="${1:-configs/example_single.yaml}"
OUT="${2:-out/single_slurm}"

python -m scripts.run_single --config "$CONFIG" --out "$OUT" --device cuda
