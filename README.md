# Adaptive Accountability Framework (AAF)

[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![arXiv](https://img.shields.io/badge/arXiv-2512.18561-b31b1b.svg)](https://arxiv.org/abs/2512.18561)

<img width="3172" height="1350" alt="banner_aaf" src="https://github.com/user-attachments/assets/415d72f9-e731-4786-b899-d02e8c717b65" />

**Code, fixed protocols, tests, and analysis for reproducible runtime-accountability experiments.** AAF separates evidence integrity, telemetry authenticity, and enforceable intervention authority. The executable studies combine sequential detection, recent-evidence ranking, bounded control, and simulated evidence failures.

The banner above is retained from the original repository as an architectural overview, not a claim that every conceptual service is implemented. The evaluated ranker uses recent violation evidence, not SHAP or a causal graph. Signatures, trusted hardware, and a production replicated ledger are not implemented as an integrated security stack.

## Start here

| Code path | Study and correct interpretation |
|---|---|
| [`experiments/AAF_R3_CORRECTED_v1`](experiments/AAF_R3_CORRECTED_v1) | Proposal-consistent online-learning comparisons: 840 game treatments, 240 separate legacy-update diagnostics, and the initial ten-policy navigation study. |
| [`experiments/AAF_R3_STRENGTHENING_CONFIRM_v1`](experiments/AAF_R3_STRENGTHENING_CONFIRM_v1) | Held-out confirmation: 20 navigation policies, 1,640 evaluations, and 1,920 fixed-proposal game comparisons using 20 policies per domain. |
| [`experiments/AAF_R3_STRENGTHENING_GPU_v1`](experiments/AAF_R3_STRENGTHENING_GPU_v1) | Earlier two-policy development diagnostics. Do not pool these with confirmation. |
| [`experiments/mechanism_tests`](experiments/mechanism_tests) | Separate telemetry, omission, alert-flood, policy-depth, and predictive-screening stress tests. |
| `aaf_q1/`, `scripts/`, `configs/`, `slurm/` | Preserved historical implementation. Its action/log-probability mismatch and archive multiplicities preclude treating its results as corrected-PPO confirmation. |

The root `requirements.txt` belongs to the historical runner; use the versioned setup instructions below for the current studies. Original package notes describe their pre-execution stage and are retained for provenance.

The current source release is **`jii-confirmation-code-1.0.0`**. This is a code-version label, not a journal acceptance statement. Cite the full Git commit SHA for an exact, immutable snapshot. The repository does not include the new manuscript, title page, cover letter, or reviewer letters.

## Quick verification — no experiment launched

```bash
git clone https://github.com/alqithami/AAF.git
cd AAF
python3 tools/verify_release.py
```

For publication reproduction, check out the exact commit cited by the article before verifying. The original README is preserved in [`docs/LEGACY_README.md`](docs/LEGACY_README.md); its commands and claims concern only the historical implementation.

## Environment and tests

Use Linux and Python 3.11 or 3.12. Setup creates a dedicated environment; it never replaces a GPU driver or an unrelated environment.

```bash
bash tools/setup_environment.sh cpu
source .venv-aaf-cpu/bin/activate
bash tools/run_confirmation.sh test
```

For the frozen GPU confirmation, install its CUDA build instead:

```bash
bash tools/setup_environment.sh cuda
source .venv-aaf-cuda/bin/activate
bash tools/run_confirmation.sh plan
```

The confirmation requires PyTorch `2.10.0+cu128`, VMAS `1.5.2`, and the versions in its [`CONFIRM_PROTOCOL.json`](experiments/AAF_R3_STRENGTHENING_CONFIRM_v1/CONFIRM_PROTOCOL.json). CPU checks are not a substitute for actual CUDA/VMAS preflight. Original machine-specific launchers remain for provenance; use the portable `tools/` entry points on other machines.

## Reproduce the held-out confirmation

**This is a substantial GPU study, not a quick demo.** Use persistent storage, one GPU process, and an output directory separate from earlier runs. The default is `results/confirmation` under this checkout.

```bash
source .venv-aaf-cuda/bin/activate
bash tools/run_confirmation.sh all "$PWD/results/confirmation"
```

The portable launcher verifies sources, runs tests, trains/evaluates, validates trajectories, performs the fixed analysis, and creates `results/confirmation.review.zip` plus its SHA-256 sidecar. `all` runs in the foreground; use your cluster's batch scheduler or a persistent terminal for long jobs. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for individual stages, the earlier studies, output interpretation, and resume boundaries.

## Evidence and reproducibility boundaries

The experimental unit is the independent policy-training seed. Episodes, configurations, agents, and time steps are not independent policy replications. The confirmation retains all 30 prespecified primary comparisons and applies one joint Holm correction. Fixed-proposal targeting gains are not online-learning gains, and reduced contact duration is not contact prevention.

**This Git release is code-only.** It includes protocols and analysis, not the original raw results, trained weights, or full physical traces. A fresh run generates those files. Reanalysis without retraining additionally requires the corresponding experiment-output archive supplied with the research materials; no public data DOI or download is asserted here. The compact review exporter retains every episode summary and policy, but only world index 0 of each physical trajectory. Keep the full output directory for all 32 worlds.

Neither a source hash nor passing tests proves field safety, signed execution, or bitwise agreement across different hardware. Do not resume an original IBM output directory with this public packaging snapshot: document-only sanitization changes its recorded source identity. Preserve the original run package for that purpose.

## Reuse and citation

The MIT license is in [`LICENSE.txt`](LICENSE.txt). [`CITATION.cff`](CITATION.cff) describes the software. In a camera-ready reproducibility statement, name this repository and the full release commit; identify data availability separately. No final journal DOI is invented.

Tests and publication packaging do not initiate experiments on an external server. See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the exact checks performed for this release.
