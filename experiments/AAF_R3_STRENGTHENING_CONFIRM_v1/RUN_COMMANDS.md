# Public reproduction entry points

The original run-transfer instructions contained a private machine address. They are replaced here by portable commands; no experiment implementation or protocol is changed.

From the repository root on Linux:

```bash
bash tools/setup_environment.sh cuda
source .venv-aaf-cuda/bin/activate
bash tools/run_confirmation.sh verify
bash tools/run_confirmation.sh plan
bash tools/run_confirmation.sh all "$PWD/results/confirmation"
```

The full workflow runs in the foreground. Use a persistent terminal or your site's batch scheduler. It trains all held-out policies, executes every configuration, analyzes all recorded conditions, and writes `results/confirmation.review.zip` and its checksum. Complete compatible records and final checkpoints are reused; interrupted work restarts from its seed. Do not mix this public source identity into the original IBM run directory.

The frozen `confirm.py` also provides `verify`, `plan`, `run`, `analyze`, `pack`, and `failure`. Run and analysis accept `--out`; pack additionally accepts `--zip`. The original IBM launcher is retained as source provenance, not the general-use entry point.

The compact exporter includes every summary, all final policies, all game proposal streams, and the fixed first-world physical trace from every configuration. Keep the complete run directory for the other 31 physical worlds. The source repository does not contain the original result archives or a public data deposit.

See the root README and `docs/REPRODUCIBILITY.md` for the earlier studies, CPU tests, environment details, and interpretation boundaries.
