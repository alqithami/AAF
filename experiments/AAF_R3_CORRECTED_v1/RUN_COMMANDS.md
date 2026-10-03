# Exact run commands

Work inside the `experiments` directory after extracting `AAF_R3_revision.zip`.
Python 3.10–3.13 is supported; 3.11 or 3.12 is recommended for dependency availability.
No external dataset, paid service, credentials, or rendering/display server is needed.

## First: install, validate, and run the pilot

```bash
cd AAF_R3_revision/experiments
bash setup.sh
source .venv/bin/activate
python run_review3.py run --profile pilot --suite all --device cpu --jobs 2 --out results/R3_pilot
python run_review3.py analyze --out results/R3_pilot
python run_review3.py pack --out results/R3_pilot --zip AAF_R3_PILOT_RESULTS.zip
```

Send `AAF_R3_PILOT_RESULTS.zip` back for inspection **before** starting the main profile.
The same sequence can be run after setup with `bash run_pilot.sh`.

The pilot checks installation, runtime, learning traces, intervention activation,
nominal navigation learning, and data completeness. Its seeds are disjoint from main
seeds. It is NOT a publication result and is never inserted into manuscript tables.
`setup.sh` creates a new CPU environment without changing an existing environment.
VMAS preflight must pass on the real installed engine before an all-suite run starts.

An existing CUDA-capable PyTorch environment may be used instead of `setup.sh`:
install `requirements.txt` there, run preflight with `--device cuda --physics`, and
use `--device cuda --jobs 1`. Do not install a CPU Torch wheel into that environment.
The code does not choose MPS automatically.

## Main run after the pilot review

```bash
python run_review3.py run --profile main --suite all --device cpu --jobs 2 --out results/R3_main
python run_review3.py analyze --out results/R3_main
python run_review3.py pack --out results/R3_main --zip AAF_R3_MAIN_RESULTS.zip
```

The main plan contains 840 corrected game treatments, 240 explicitly labeled
legacy-update sensitivity treatments, and 560 VMAS policy-seed/condition/method
evaluations. These are NOT 1,640 statistically independent replications: comparisons
are paired within 20 game seeds or 10 independently trained navigation-policy seeds.
Each VMAS evaluation averages 32 held-out episodes within the policy seed.

## Resume and inspect

Repeat the **identical run command with the identical output directory** after an
interruption. Verified completed treatments and nominal-policy checkpoints are reused.
Only the incomplete treatment restarts from its recorded seed. This is treatment-level
resume, not mid-minibatch checkpoint recovery. Never delete old results to rerun.

```bash
python run_review3.py plan --profile main --suite all
python run_review3.py analyze --out results/R3_main --allow-partial
```

Partial analysis is diagnostic only. Standard analysis refuses missing or mismatched
runs. Code/configuration/dependency changes require a new output directory; this
prevents accidental mixing of different implementations.

## Optional isolated smoke check

```bash
python run_review3.py run --profile smoke --suite games --device cpu --jobs 2 --out results/R3_smoke
python run_review3.py analyze --out results/R3_smoke
```

To exercise the actual VMAS integration, use `--suite physics` or `--suite all` after
installation. No fake or substitute engine is used when VMAS is unavailable.

## Outputs

- `RUN_MANIFEST.json`: complete plan, code fingerprint, package versions, device.
- `runs/<id>/summary.json`: scalar metrics, configuration, all alarm timestamps,
  target admissions, and training/evaluation diagnostics.
- `runs/<id>/steps.json.gz`: full game step traces.
- `initial_policies/`: reusable nominal model and optimizer checkpoints.
- `analysis/COMPLETENESS.json`: missing/invalid-run checks.
- `analysis/seed_results.csv`: one record per treatment and independent seed.
- `analysis/paired_comparisons.csv`: prespecified paired effects and Holm-adjusted tests.
- `analysis/legacy_update_sensitivity.csv`: diagnostic, not primary evidence.
- `analysis/PHYSICS_POLICY_QUALITY.json`: nominal PPO versus random-policy check.
- `analysis/figures/`: new data-derived figures; original manuscript figures untouched.
- `analysis/REPORT.md`: analysis status and limitations.

Main numerical tables are emitted only for a complete main dataset. Even then,
scientific interpretation and the response letter require review; an automatically
created table is not an automatic acceptance or submission-readiness certificate.
