# Confirmation-wrapper validation

The 50 files under `base/` match the successfully completed strengthening-v1 package byte-for-byte. That returned IBM run reported 65 tests passed on NVIDIA L40S with PyTorch 2.10.0+cu128 and actual VMAS 1.5.2 preflight.

New wrapper checks: **13 tests passed locally**. Tests cover source preservation, exact configuration expansion, development/confirmation seed separation, all 30 primary cells, exact sign-test examples/ties, Holm multiplicity, deterministic paired resampling, refusal of incomplete/nonfinite seed pairs, fixed trace selection, and analysis/export wiring.

The analysis/export wiring tests use explicitly synthetic fixtures. They test serialization and statistics plumbing only; they are not simulator evidence. The local environment does not run the real CUDA confirmation study. Actual-engine preflight and pinned-dependency checks execute on IBM before confirmation.

Bash syntax and Python compilation were checked. No dependency upgrades, accelerator model calls, experiment-result fabrication, baseline edits, or GitHub writes are included in this delivery. No confirmatory results have been generated.
