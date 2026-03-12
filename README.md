# Adaptive Accountability Framework (AAF)

<img width="3172" height="1350" alt="banner_aaf" src="https://github.com/user-attachments/assets/415d72f9-e731-4786-b899-d02e8c717b65" />

This repository provides a **reproducible experiment pipeline** for evaluating the **Adaptive Accountability Framework (AAF)** in networked multi-agent systems.

It is designed for *paper-grade* empirical evaluation:

- Two benchmark environments: `resource_sharing`, `public_goods`
- AAF variants (full + ablations) and strong baselines (PPO + fairness/constraint variants)
- Large, resumable sweeps with sharding (local multiprocess or multi-node)
- Aggregation + paired significance tests
- Paper-ready LaTeX tables and figures

---

## What you get

### Methods / baselines

- `ppo_only` — parameter-sharing PPO (no accountability)
- `static_guard` — fixed rule-based guardrail
- `constrained_ppo` — PPO with constraint-style penalty handling
- `fair_ppo` — PPO with fairness-aware shaping

AAF variants:

- `aaf_full` — full AAF pipeline (detector + attribution + intervention)
- `aaf_detector_only`
- `aaf_shaping_only`
- `aaf_patch_only`
- `aaf_no_attrib`

### Key metrics (written to `summary.json` per run)

- compromise ratios (attempted + executed)
- social welfare
- allocation/reward inequality (Gini)
- alarms count and detection delay
- attribution quality (top-1 / recall@k)
- bandwidth + runtime

---

## Installation

### 1) Create a clean environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Device selection (CPU / Apple MPS / CUDA)

The runner supports `--device {auto,cpu,mps,cuda}`.

- **Apple Silicon (M1–M4):**
  - For *single runs*, `--device mps` is fine.
  - For *large sweeps*, `--device cpu` is often **more stable** (the env loop is mostly NumPy/CPU; some PyTorch ops may fall back).
  - This repo includes an MPS-friendly Beta sampler to avoid the common `Dirichlet` fallback warnings during PPO action sampling.

- **NVIDIA GPUs (Linux/cluster):**
  - Install a CUDA-enabled PyTorch build.
  - Use `--device cuda` (or `--device auto`).
  - For GPU runs, set `--jobs 1` (one process should own the GPU).

---

## Quickstart

### A) Run a single experiment (sanity check)

```bash
python -m scripts.run_single \
  --env resource_sharing \
  --baseline aaf_full \
  --n_agents 50 \
  --penalty_factor 0.20 \
  --dist_alpha 1.0 \
  --partial_obs off \
  --t_steps 1000 \
  --seed 0 \
  --device auto \
  --out out/run_single_aaf
```

You will get:

- `out/run_single_aaf/<timestamp_baseline_hash>/summary.json`
- optional `step_logs.csv` / `agent_logs.csv` depending on `--log_mode`

### B) Run a sweep end-to-end (grid → runs → aggregate → paper assets)

1) Generate a grid (JSONL)

```bash
python -m scripts.make_grid --preset q1_main --out configs/grid_q1_main.jsonl
```

2) Execute the grid

```bash
python -m scripts.run_grid \
  --grid configs/grid_q1_main.jsonl \
  --out out/q1_main \
  --jobs 4 \
  --device cpu \
  --log_mode summary \
  --max_tasks_per_child 50 \
  --torch_threads 1
```

3) Aggregate all runs

```bash
python -m scripts.aggregate --root out/q1_main
```

4) Export LaTeX tables and figures

```bash
python -m scripts.make_latex \
  --summary out/q1_main/analysis/final_summary.csv \
  --outdir out/q1_main/analysis/paper_assets

python -m scripts.make_figures \
  --summary out/q1_main/analysis/final_summary.csv \
  --outdir out/q1_main/analysis/paper_assets
```

5) Run paired significance tests (method vs baseline)

```bash
python -m scripts.stats \
  --summary out/q1_main/analysis/final_summary.csv \
  --outdir out/q1_main/analysis \
  --method aaf_full \
  --baseline ppo_only
```

---

## Presets

`make_grid` supports the following presets:

- `paper_fast` — small sweep for CI / quick validation
- `paper_full` — fuller sweep (matches the original “paper grid” style)
- `q1_main` — strong, review-friendly main comparison across both tasks
- `q1_ablation` — AAF ablations on a canonical slice
- `q1_deep` — the **largest** sweep (both tasks × more baselines × 10 seeds)

Example:

```bash
python -m scripts.make_grid --preset q1_deep --out configs/grid_q1_deep.jsonl
```

---

## Running large grids safely

### 1) Don’t write huge outputs into cloud-synced folders

Large sweeps create **tens of thousands of small files**.
Avoid output paths under iCloud / Google Drive / Dropbox / OneDrive.

Good:

```bash
python -m scripts.run_grid --grid configs/grid_q1_deep.jsonl --out ./out/q1_deep
```

Risky:

```bash
# Avoid: can trigger OS kills or extreme slowdown
python -m scripts.run_grid --grid configs/grid_q1_deep.jsonl --out ~/Library/CloudStorage/...
```

### 2) Use `--log_mode summary` for sweeps

`steps`/`full` logging can explode disk usage. For sweeps, keep:

```bash
--log_mode summary
```

### 3) Resume is automatic

`run_grid` is **resumable**. If a run folder already contains `summary.json`, it is skipped.
So you can safely re-run the same command after interruption.

### 4) Shard across machines (or multiple terminal sessions)

You can split a grid by `--shard_id / --num_shards`.

Example: 4 shards (run these in parallel on 4 machines):

```bash
python -m scripts.run_grid --grid configs/grid_q1_deep.jsonl --out out/q1_deep/shard_0 --jobs 1 --device cpu --shard_id 0 --num_shards 4
python -m scripts.run_grid --grid configs/grid_q1_deep.jsonl --out out/q1_deep/shard_1 --jobs 1 --device cpu --shard_id 1 --num_shards 4
python -m scripts.run_grid --grid configs/grid_q1_deep.jsonl --out out/q1_deep/shard_2 --jobs 1 --device cpu --shard_id 2 --num_shards 4
python -m scripts.run_grid --grid configs/grid_q1_deep.jsonl --out out/q1_deep/shard_3 --jobs 1 --device cpu --shard_id 3 --num_shards 4
```

Then aggregate across *all shards*:

```bash
python -m scripts.aggregate --root out/q1_deep
```

Aggregation searches recursively for `summary.json`, so shard subdirectories are supported.

### 5) Smoke-test a big grid first

Before launching a large sweep, run a tiny subset:

```bash
python -m scripts.run_grid \
  --grid configs/grid_q1_deep.jsonl \
  --out out/q1_deep_smoke \
  --jobs 1 \
  --device cpu \
  --max_runs 10
```

---

## Outputs (what gets written where)

After `run_grid`:

```
out/<exp>/
  <run_id_1>/
    summary.json
    config.json            # if --write_config
    step_logs.csv          # if log_mode includes steps
    agent_logs.csv         # if log_mode includes agents
  <run_id_2>/
    ...
```

After `aggregate`:

```
out/<exp>/analysis/
  all_runs_flat.csv        # per-run flat table (one row per seed/run)
  final_summary.csv        # grouped means/std/CI per config
```

After `make_latex` + `make_figures`:

```
out/<exp>/analysis/paper_assets/
  table_main_results.tex
  table_ablation.tex
  fig_compromise_bar.png
  fig_pareto_compromise_gini.png
  fig_detection_delay_cdf.png   # only if byzantine runs exist
```

---

## If you are collaborating with the paper author

When sharing results for incorporation into LaTeX, the most useful artifacts are:

- `out/<exp>/analysis/final_summary.csv`
- `out/<exp>/analysis/all_runs_flat.csv`
- `out/<exp>/analysis/paper_assets/` (tables + figures)
- `out/<exp>/analysis/stats_paired_tests.csv` (after running `scripts.stats`)

Zipping just `out/<exp>/analysis/` is usually sufficient.

---

## Troubleshooting

### “error: the following arguments are required: --out”

This repo uses `--out` (not `--out_dir`). Examples:

```bash
python -m scripts.make_grid --preset q1_deep --out configs/grid_q1_deep.jsonl
python -m scripts.run_grid  --grid configs/grid_q1_deep.jsonl --out out/q1_deep
```

### `--device cuda` fails on macOS

CUDA is not supported on Apple Silicon. Use:

```bash
--device mps
# or
--device cpu
```

### `zsh: killed` during a long sweep

Common causes:

- output directory is cloud-synced (massive file churn)
- memory growth over many runs

Mitigations:

- write outputs to a local scratch path
- add `--max_tasks_per_child 50`
- keep `--log_mode summary`
- shard the grid across multiple processes/machines

### Multiprocessing warnings about leaked semaphores

If you see `resource_tracker: leaked semaphore objects`, try:

- `--max_tasks_per_child 20` (or 50)
- reduce `--jobs`
- or run `--jobs 1`

---

## Citation

If you use this codebase, please cite the associated AAF paper (add/update BibTeX once the journal version is final):

```bibtex
@article{alqithami2025aaf,
  title   = {Adaptive Accountability in Networked Multi-Agent Systems: Tracing and Mitigating Emergent Norms at Scale},
  author  = {Alqithami, Saad},
  journal = {arXiv preprint},
  year    = {2025}
}
```

---

## License

Research code, provided as-is. Add an explicit OSS license (MIT/Apache-2.0/etc.) if you plan to distribute or accept external contributions.
