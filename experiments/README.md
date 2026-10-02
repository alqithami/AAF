# AAF experiment packages

## Current development package: GPU strengthening v1

`AAF_R3_STRENGTHENING_GPU_v1/` is the complete, directly runnable source package
from the separately delivered ZIP, imported without modifying its 50 files.
This is development-stage code, not a completed confirmatory study or a replacement
for the paper's previous results. No new experimental outcome is claimed here.

The original repository code and all existing paper directories are preserved.
The corrected R3 implementation needed by this package is included under its own
`vendor/aaf_r3/` directory; no external repository checkout is needed at runtime.

### IBM GPU server: start from a fresh clone

Use an allocated GPU and a persistent working directory. These commands create a
new clone and an isolated Python environment; do not execute them inside a running
experiment's directory.

```bash
git clone --single-branch --branch revision/jii-major-revision-redline \
  https://github.com/alqithami/AAF.git AAF-gpu-strengthening
cd AAF-gpu-strengthening/experiments/AAF_R3_STRENGTHENING_GPU_v1
python3 verify_package.py
bash setup_gpu.sh
bash run_ibm_diagnostics.sh
```

The same code is already in the delivered `AAF_R3_STRENGTHENING_GPU_v1.zip`.
An existing run from that ZIP does not need restarting or duplicating because of
this repository import. Keep its code, environment, output path, and fingerprints
unchanged. Resume it using its original launcher in its original directory.

The launcher performs GPU/actual-engine preflight, tests, smoke execution,
development diagnostics, analysis, and packaging. It never automatically starts
a confirmatory main study. GPU experiments are not run by a Git commit.

### Outputs and evidence status

Return `AAF_R3_STRENGTHENING_RESULTS.zip` after successful completion, or
`AAF_R3_STRENGTHENING_FAILURE.zip` after failure. Preserve the on-server results,
policy checkpoints, and full state/action traces. Local result, cache, log, and
virtual-environment folders are excluded from Git by the sibling `.gitignore`.

A repository-import validation reran the supplied tests on CPU: **61 passed,
4 actual-VMAS-dependent tests skipped**. CUDA and VMAS execution remain required
on the IBM machine. The source-package validation records are preserved as
historical preparation records, not rewritten as new server validation.

Read the package's `docs/EXPERIMENT_PROTOCOL.md`, `docs/RUN_COMMANDS.md`, and
`validation/LOCAL_VALIDATION.md` before interpreting outputs. The candidate
predictive controller is not a certified physical-safety filter. This package
requires no watsonx model API, credentials, or paid inference service.

### Provenance

`AAF_R3_STRENGTHENING_GPU_v1_IMPORT.json` records the source archive hash and exact
package tree identity. The package's `SHA256SUMS.txt` verifies its 49 other files.
Its preparation-time sentence saying the package was not yet committed describes
the earlier standalone handoff; this repository import publishes that exact source
unchanged. No source results or experimental claims were edited for publication.
