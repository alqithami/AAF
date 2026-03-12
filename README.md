# AAF Q1 Experiment Pipeline

This repository provides an **end-to-end, reproducible** experiment pipeline for the paper:

> *Adaptive Accountability in Networked Multi‑Agent Systems (AAF): Tracing and Mitigating Emergent Norms at Scale*

It includes:

- **Resource-sharing game** environment (paper §7.1 style).
- Optional **Public-goods** environment (second domain) for stronger Q1 evaluation.
- **Multi-agent parameter-sharing PPO** baseline (PPO-only).
- **Static guard** baseline (action restriction).
- **Constrained PPO** baseline (Lagrangian constraint on norm violations).
- **Fairness-regularized PPO** baseline (Gini regularization).
- **AAF supervisor**: online detection (adaptive CUSUM) + responsibility scoring + interventions (reward shaping + policy patch).
- **Grid runner**, **aggregator**, and **LaTeX/figure exporters** for paper-ready assets.
- Optional **Slurm array** scripts (IBM / cluster).

The code is designed to run on:
- **Apple Silicon (M4 Max)** via PyTorch **MPS** (Metal)
- **NVIDIA GPUs** via PyTorch **CUDA**
- CPU-only as fallback.

---

## 0) TL;DR commands

### One run
```bash
python -m scripts.run_single --config configs/example_single.yaml --out out/single
```

### Paper-style grid (fast)
```bash
python -m scripts.make_grid --preset paper_fast --out configs/grid_paper_fast.jsonl
python -m scripts.run_grid  --grid configs/grid_paper_fast.jsonl --out out/grid_paper_fast --jobs 4 --device cpu --log_mode summary --torch_threads 1
python -m scripts.aggregate --root out/grid_paper_fast
python -m scripts.make_latex --summary out/grid_paper_fast/analysis/final_summary.csv --outdir out/grid_paper_fast/analysis/paper_assets
python -m scripts.make_figures --summary out/grid_paper_fast/analysis/final_summary.csv --outdir out/grid_paper_fast/analysis/paper_assets
```

### Q1 “deep evaluation” grid (bigger; 2 domains; more baselines/ablations)
```bash
python -m scripts.make_grid --preset q1_deep --out configs/grid_q1_deep.jsonl
python -m scripts.run_grid  --grid configs/grid_q1_deep.jsonl --out out/grid_q1_deep --jobs 8 --device cpu --log_mode summary --torch_threads 1 --max_tasks_per_child 50
python -m scripts.aggregate --root out/grid_q1_deep
python -m scripts.stats     --summary out/grid_q1_deep/analysis/final_summary.csv --outdir out/grid_q1_deep/analysis/paper_assets
python -m scripts.make_latex --summary out/grid_q1_deep/analysis/final_summary.csv --outdir out/grid_q1_deep/analysis/paper_assets
python -m scripts.make_figures --summary out/grid_q1_deep/analysis/final_summary.csv --outdir out/grid_q1_deep/analysis/paper_assets
```

---

## 1) Installation

### 1.1 Python environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 1.2 PyTorch notes (GPU)
**Apple Silicon (M4 Max / macOS):**
- Install PyTorch as recommended by the official PyTorch install page for macOS.
- MPS is used automatically when you pass `--device mps` (or `--device auto`).

**NVIDIA GPU (IBM cloud):**
- Install the CUDA wheel matching your CUDA version (PyTorch provides cu11x/cu12x wheels).
- Use `--device cuda`.

You can confirm device visibility with:
```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), 'mps', torch.backends.mps.is_available())"
```

---

## 2) Output structure

Each run creates:

```
out/<exp_name>/<run_id>/
  config.json
  summary.json
  step_logs.csv        # only when log_mode=steps/full
  agent_logs.csv       # only when log_mode=agents/full
  artifacts/
```

Aggregation writes:

```
out/<exp_name>/analysis/
  all_runs_flat.csv
  final_summary.csv
  paper_assets/
    table_main_results.tex
    table_ablation.tex
    fig_compromise_bar.png
    fig_detection_delay_cdf.png
    fig_pareto_compromise_gini.png
    stats_paired_tests.csv
```

---

## 3) What “deep Q1 evaluation” means here

The `q1_deep` preset is designed to address typical Q1 reviewer demands:

1) **Multiple domains** (resource-sharing + public-goods)  
2) **Baselines beyond PPO-only**  
   - action restriction (“static guard”)
   - constrained PPO
   - fairness-regularized PPO
3) **AAF ablations**  
   - detector-only
   - shaping-only
   - patch-only
   - no-attribution (uniform blame)
4) **Byzantine stress tests**  
   - fractions: 0%, 5%, 10% (configurable)
   - change-point at t = 200 (configurable)
5) **Attribution quality** (top‑k accuracy vs known Byzantine IDs)
6) **Detector calibration** (alarm rate, delay distribution)
7) **Statistical tests** (paired deltas, confidence intervals)

All of these are produced by `scripts/stats.py` and `scripts/make_figures.py`.

---

## 4) Running on Slurm (IBM / cluster)

1) Create a grid file (JSONL):
```bash
python -m scripts.make_grid --preset q1_deep --out configs/grid_q1_deep.jsonl
```

2) Submit job array:
```bash
sbatch --array=0-19 slurm/run_array.sh configs/grid_q1_deep.jsonl out/grid_q1_deep 20 cuda summary
```

Then aggregate after jobs finish:
```bash
python -m scripts.aggregate --root out/grid_q1_deep
```

---

## 5) Troubleshooting

- **macOS / Apple Silicon:** for large grids, prefer `--device cpu`.
  - The environment loop is NumPy/CPU, and some distribution samplers can fall back from MPS.
  - If you insist on MPS, this repo includes an MPS-friendly Beta sampler to avoid Dirichlet fallback warnings.
- **Avoid cloud-synced output folders** for large grids (Google Drive / Dropbox / iCloud). Use a local scratch path.
- If results look inconsistent, ensure you are not mixing output folders from different code versions.
- If results look inconsistent, ensure you are not mixing output folders from different code versions.
- For reproducibility, every run stores `config.json` and the seed.

---

## 6) License

This code is provided for research use. You may re-license it as you like for your submission.
