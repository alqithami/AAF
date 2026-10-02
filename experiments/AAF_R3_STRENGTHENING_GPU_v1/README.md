# AAF: GPU-enabled strengthening diagnostics — development v1

**Run this package on the IBM GPU server. It is separate from the completed R3 paper and experiments.**

This is a diagnostic method-development package, not a new final submission, a
confirmatory test, or a promise of favorable results. No old manuscript, figure,
experiment folder, Python environment, or GitHub branch is replaced.

## Run now

Upload `AAF_R3_STRENGTHENING_GPU_v1.zip` to a new working directory on the GPU server.
Use persistent storage with room for a new CUDA environment and trajectory files
(allow approximately 15 GB for installation and working space; final use varies).
Extract the ZIP. In the server's SSH terminal:

```bash
unzip -n AAF_R3_STRENGTHENING_GPU_v1.zip
cd AAF_R3_STRENGTHENING_GPU_v1
bash setup_gpu.sh
bash run_ibm_diagnostics.sh
```

The setup script finds Python 3.11 or 3.12 and creates **only this folder's `.venv`**.
It installs Torch 2.10.0 from the official CUDA 12.8 wheel index and VMAS 1.5.2.
It does not install a driver, alter system Python, or touch CAENL/PBRC environments.
The run uses one CUDA process, 16 parallel training worlds and 8 evaluation worlds.
At least 8 GiB free GPU memory is checked, with a 12 GiB PyTorch allocator ceiling.
That ceiling is not a GPU reservation and does not include all driver allocations.
Use only a GPU allocated to this work; no other processes are stopped.

The launcher runs actual-engine preflight, unit/integration tests, an isolated
smoke run, development diagnostics, complete-record analysis, and return packaging.
It stops on failed checks. It does not automatically launch any main confirmation.

## What to return

**`AAF_R3_STRENGTHENING_RESULTS.zip`**, produced beside the launcher.
It includes source/configuration fingerprints, policy weights, complete physics
state/action/evidence traces, episode timing, nominal training traces, all methods
and conditions, game replay streams, results, and checksums. Nothing is selected
for favorable performance. Large original AAF archives are not copied into it.

On failure, return **`AAF_R3_STRENGTHENING_FAILURE.zip`**. No API key is needed.
The failure collector deliberately does not dump environment variables.

## Interruption

Repeat `bash run_ibm_diagnostics.sh` in the same directory. Completed compatible
physics treatments and complete nominal policy checkpoints are reused. An unfinished
policy training or treatment restarts from its seed; there is no mid-update checkpoint.
The game replay block is atomic as a block and is repeated if interrupted before completion.
Do not edit code/configuration inside an existing output run: fingerprints reject mixing.
Never delete an earlier result directory to get past an identity mismatch.

## Scope

1. **Capacity and selection opportunity:** fixed-proposal replay from a frozen,
   corrected PPO learner, plus clearly labeled scripted controls; these are NOT
   online-learning ablations. A future-window reference gives an exact hindsight
   maximum for the fixed recorded proposals and disjoint fixed intervention windows.
2. **Timing and authority:** actual VMAS navigation, original four control rules,
   three target-count caps, reactive brake versus a shared candidate predictive filter.
3. **State-model positive diagnostics:** adaptive model-benefit targeting and a
   zero-delay model-based variant. These have privileged gateway-state assumptions
   and are neither optimal oracles nor new demonstrated causal attribution methods.
4. **End-to-end evidence/timing faults:** forged reports, selectively suppressed
   records, explicit next-checkpoint gaps, permanent suppression, independent/burst
   loss, and separated evidence/command delays.

The candidate filter holds commands constant over an eight-step prediction and
omits contact forces. It is a diagnostic prototype, NOT a certified MPC/barrier
filter. Preflight checks its free-motion integration against the actual VMAS
engine. Actual state prediction errors are recorded throughout the evaluation.

The original R3 primary contact endpoint is retained. New timing and duration
measures are prospectively collected for this development extension, not substituted
into the earlier confirmatory analysis. Two fresh policy seeds are not enough for
publication-level inference; no p-values or non-inferiority claims are generated.

## GPU and watsonx

CUDA is used for PPO training, VMAS physics, and vectorized candidate prediction.
The current orchestration still includes CPU-side evidence/scheduling and array
transfers; no hardware speedup is promised without measurements. A CPU override is
available in `docs/RUN_COMMANDS.md` for debugging, never silently selected.

watsonx foundation-model calls are intentionally absent. They would add an unrelated
inference model to a deterministic control comparison and cannot verify simulated
outcomes. This code sends no manuscripts, credentials, traces, or results to an API.
An evidence-grounded explanation study could be separate future work.

## Read next

- `docs/EXPERIMENT_PROTOCOL.md`: precise arms, timing and evidence contracts.
- `docs/RUN_COMMANDS.md`: server preparation, resume, targeted troubleshooting.
- `docs/METHOD_EXTENSION.tex`: mathematical diagnostics and proposed methods text;
  not a replacement manuscript or a report of completed main findings.
- `docs/BASELINE_PROVENANCE.json`: identity of the retained executed R3 source.
- `validation/LOCAL_VALIDATION.md`: what was tested locally and what was not.
