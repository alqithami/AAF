# IBM server: exact commands and output

Run commands in the Linux terminal on the GPU machine, not on the Mac and not in
the watsonx Prompt Lab. Use the same established SSH/SCP method used for prior jobs.
No server address or credentials are embedded in this package.

Upload the ZIP to your chosen persistent working directory, then:

```bash
unzip -n AAF_R3_STRENGTHENING_GPU_v1.zip
cd AAF_R3_STRENGTHENING_GPU_v1
bash setup_gpu.sh
bash run_ibm_diagnostics.sh
```

A previously created directory from a DIFFERENT release must not be mixed with this
release. `verify_package.py` checks all release files before installation/run.
The scripts never search for or modify old AAF, CAENL, or PBRC data.

## Python selection

The setup script first tries python3.11, then python3.12. The IBM server's old default
`python3` may be 3.9; that does not require changing system Python. An explicit path:

```bash
AAF_PYTHON=/usr/bin/python3.11 bash setup_gpu.sh
```

This creates a new `.venv` under this package. It refuses unrelated existing `.venv`
directories or symlinks. Installing the new CUDA wheel requires internet and disk
space. It does not install a CUDA driver or modify another project's PyTorch.

An existing isolated, compatible environment can be used without setup:

```bash
AAF_RUN_PYTHON=/absolute/path/to/your/environment/bin/python bash run_ibm_diagnostics.sh
```

Only do so after checking VMAS 1.5.2 and CUDA availability. Its actual dependencies
are recorded and must remain unchanged on resume. Do not pip-install over another
running project's environment. The supplied setup route is safer.

## Long SSH session

Use your existing terminal multiplexer or cluster job allocation so disconnecting
SSH does not kill training. On a machine with tmux installed:

```bash
tmux new -s aaf-strengthening
# Within that session, cd to this package, then run:
bash run_ibm_diagnostics.sh
```

Detach with Ctrl-b then d. Reattach with `tmux attach -t aaf-strengthening`.
The code does not create remote scheduled jobs or start background work on its own.
Do not start a second run in the same output directory; a lock prevents concurrent writers.

## Manual stage commands

The wrapper already runs these. Do not run them concurrently with the wrapper.

```bash
source .venv/bin/activate
python run_strengthening.py preflight --device cuda --out results/preflight
python run_strengthening.py run --profile smoke --suite all --device cuda --out results/smoke
python run_strengthening.py analyze --out results/smoke
python run_strengthening.py run --profile diagnostic --suite all --device cuda --out results/development
python run_strengthening.py analyze --out results/development
python run_strengthening.py pack --out results/development --zip AAF_R3_STRENGTHENING_RESULTS.zip
```

The manual route omits wrapper test and execution-log collection unless run separately;
the wrapper is recommended. There is deliberately no `--profile main` option.

## Resume

Repeat the same wrapper. Completed treatment checksums and nominal-policy identities
are checked before reuse. A partially trained policy restarts from its seed. An
interrupted evaluation restarts only that treatment. A partial replay block restarts
the replay block. There is no unsafe mid-minibatch continuation claim.

Code/config/environment/device changes require a new output directory. Do not delete
completed records to bypass a mismatch. The plan includes every method and condition.

## Failure

Return `AAF_R3_STRENGTHENING_FAILURE.zip` after any failed setup/preflight/test/run.
It contains logs and safe metadata, not API keys. When absent, manually create it:

```bash
python3 collect_failure.py
```

A failed preflight is not a completed experiment. In particular, do not loosen the
model-validation tolerance or swap in a fake simulator to make it pass.

## CPU debug only

The normal launcher requires CUDA. An explicit CPU smoke test is possible in an
already compatible environment, with a separate directory:

```bash
python run_strengthening.py run --profile smoke --suite games --device cpu --out results/cpu_game_smoke
python run_strengthening.py analyze --out results/cpu_game_smoke
```

Running physics on CPU still requires the actual VMAS installation. CPU and CUDA
outputs are never silently pooled or resumed across devices.

## Return

Upload `AAF_R3_STRENGTHENING_RESULTS.zip` after completion. Preserve all directories.
The return includes final policy weights and full physics traces (unlike the earlier
R3 export). Runtime and output size depend on the actual hardware; no unmeasured
wall-clock estimate is used to declare a stalled process. Training prints every
16 updates and evaluation progress prints once per treatment.
