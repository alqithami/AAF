# Reproduction guide

## Source identity

Use the exact Git commit cited by the camera-ready article. `jii-confirmation-code-1.0.0` is the software version label. The new source release does not claim a journal acceptance or provide manuscript PDFs. `tools/verify_release.py` verifies the imported source snapshots before long runs.

The confirmation's `confirm.py`, protocol, statistical routines, tests, and 50-file `base/` are unchanged from the executed packages. Only the public transfer guide and its checksum line were sanitized to remove the original machine address. New portable launchers are outside those snapshots. This changes the wrapper's source-identity record: **use a fresh output directory**, not the old IBM result directory.

## Setup

The portable installer targets Linux and Python 3.11/3.12, uses an isolated `.venv-aaf-cpu` or `.venv-aaf-cuda`, and keeps pip/build caches under `.runtime`. It never changes the system driver. The experiment package pins numerical dependencies, including VMAS 1.5.2; the frozen confirmation specifically requires PyTorch 2.10.0+cu128. See the original protocol for the complete pins. A site without CUDA can run source checks, most tests, and the corrected games but not claim reproduction of the pinned GPU confirmation.

```bash
bash tools/setup_environment.sh cuda
source .venv-aaf-cuda/bin/activate
bash tools/run_confirmation.sh test
bash tools/run_confirmation.sh plan
```

The inherited CUDA preflight checks at least 8 GiB of free GPU memory and applies a 12 GiB allocator ceiling. A ceiling is not a reservation. Provide adequate persistent disk storage for full traces and do not share one device between duplicate workers. The recorded research platform was NVIDIA L40S; numerical equality on other hardware is not promised.

## Held-out confirmation

```bash
bash tools/run_confirmation.sh all "$PWD/results/confirmation"
```

Individual stages are also supported:

```bash
bash tools/run_confirmation.sh run "$PWD/results/confirmation"
bash tools/run_confirmation.sh analyze "$PWD/results/confirmation"
bash tools/run_confirmation.sh pack "$PWD/results/confirmation"
```

The portable workflow is foreground. Use a persistent session or site batch scheduler. An exclusive output lock prevents concurrent portable workflows. Complete compatible treatments/final checkpoints are reused. An incomplete treatment or training restarts from its recorded seed, not a mid-update checkpoint. Changed configuration, numerical source, or environment is rejected by the experiment identity check.

The plan includes 20 independent navigation policies, 82 configurations per policy, 32 episodes per configuration, and 20 independent game policies per domain. The replay compares four selectors, four stream types, and three budgets in both domains. The six game and 24 navigation primary comparisons form one Holm-adjusted family. Analysis averages episodes within training seed and reports paired effects separately from exact sign tests. Preserve null and adverse results. No tuning or seed selection is authorized by a completed run.

## Initial corrected-learning studies

These are different experiments and must not be pooled with confirmation. Activate the environment from the repository root, then:

```bash
cd experiments/AAF_R3_CORRECTED_v1
python run_review3.py preflight --device cpu --physics
python -m pytest -q tests
python run_review3.py run --profile main --suite games --device cpu --jobs 2 --out results/games_main
python run_review3.py analyze --out results/games_main
python run_review3.py run --profile main --suite physics --device cpu --jobs 2 --out results/physics_main
python run_review3.py analyze --out results/physics_main
```

This reproduces 840 corrected game treatments plus 240 separate legacy-update diagnostics and 560 initial navigation evaluations. The historical-update diagnostic arms intentionally retain the earlier convention; do not merge them into corrected estimates. Read the included `EXPERIMENT_PROTOCOL.md` for seeds, calibration, endpoints, and bootstrap/multiplicity definitions.

## Development diagnostics

The two-policy development code is preserved separately. It is not a substitute for held-out confirmation:

```bash
cd experiments/AAF_R3_STRENGTHENING_GPU_v1
python run_strengthening.py --help
```

Its original setup/IBM scripts remain unchanged for provenance. For future work, change the study version and use new seeds/output directories instead of silently retuning the frozen confirmation.

## Mechanism and historical audits

Use the explicit seed counts in `experiments/mechanism_tests/README.md`. For the separate statistical-unit audit of the historical CSVs, install `statsmodels` into a separate audit environment if needed, then:

```bash
python tools/audit_historical_results.py --main /path/to/all_runs_flat.csv --scaling /path/to/scaling_all_runs_flat.csv --out results/historical_audit
```

This audit needs the original CSVs. It cannot rebuild them from current confirmation results or validate the old update rule as corrected PPO. The legacy `aaf_q1/` tree and original commands remain unchanged. Their result labels must retain their historical status.

## Code versus data availability

This repository supplies executable code, plans, source hashes, tests, and statistical/export routines. It does **not** currently publish the original raw results, trained-policy binaries, full trajectories, manuscript, or editorial documents. Rerunning creates result records and checkpoints. Reanalysis without rerunning requires the corresponding external research-output archive; no public dataset URL or DOI is invented here.

The confirmation review ZIP contains all episode summaries and policy weights but just world 0 of every configuration's physical trace. All 32 worlds remain in the full run directory. A review ZIP is therefore not the complete physical dataset. Keep both.

## Scientific scope

The fixed-proposal study measures direct suppression, not changes in online learning. Available intervention limits differ from realized command changes. The predictive candidate search is not a certified invariant-set filter; gateway risk-evidence experiments retain trusted simulated local state for that primitive. The code models omission and forgery, not a compromised operating system, real trusted hardware, or deployed signatures. No passing checksum, unit test, or simulator run establishes universal safety.
