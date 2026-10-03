# Retained mechanism-level stress tests

`run_targeted_robustness.py` is copied without numerical changes from the final research archive. It tests controlled event streams, bounded escalation, policy-depth/action-predicate enforcement, and predictive-screening failure. These are not field-security or integrated online-recovery experiments.

From the repository root in the installed environment, reproduce the reported seed counts explicitly:

```bash
python experiments/mechanism_tests/run_targeted_robustness.py --out results/mechanisms --stream-seeds 100 --flood-seeds 500 --shield-seeds 12 --screening-seeds 250
```

The script's default stream count is 500; the publication used the explicit 100-seed setting above. Shield seeds are per network depth. No saved numerical outcomes are included in this code-only release.
