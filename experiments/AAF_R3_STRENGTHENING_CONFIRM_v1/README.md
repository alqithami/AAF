# AAF R3 strengthening: held-out confirmation v1

This package **does not modify any previous manuscript, figure, repository, environment, or experiment output**. `base/` is an exact 50-file copy of the reviewed GPU strengthening-v1 source. The new `confirm.py` fixes a separate confirmation design; it does not modify the detector, ranking, filter, calibration, attack, dynamics, or authority settings.

## Why this is the next stage

The development run supports testing (i) selective authority allocation on heterogeneous public-goods proposal records, (ii) the shared predictive primitive, and (iii) the consequences of telemetry and omission handling. It also establishes the zero-selection-opportunity resource-sharing negative control and serious remaining limitations of bounded contact prevention. Confirmation must retain those negative conditions and simpler comparators.

The methods are now frozen. Fresh independent seeds are used; there is no favorable-checkpoint selection or policy exclusion. This is a prospective design **after viewing separate development data**, not an externally registered experiment and not a claim that the results are already favorable.

## Study size

- Navigation: 20 new independent policy-training seeds `291000`–`291019`, 82 configurations each, 32 evaluation episodes/configuration: **1,640 configurations / 52,480 episode records**.
- Games: 20 new separately trained seeds per domain `291100`–`291119`; both domains, all four stream types, three budgets and four selectors: **1,920 fixed-proposal replay comparisons**.
- All no-attack, random, threshold, periodic, unrestricted-guard, state-informed, and communication conditions remain included.
- The scripted streams remain engineering controls; the game replay study remains a fixed-proposal diagnostic, not an online-learning experiment.
- The independent unit is the policy-training seed within each domain, not the agent, time step, episode, or configuration count.

Only the seed list, navigation evaluation batch size and study labels differ from the reviewed design. The unchanged baseline files are retained for exact source provenance; inherited comments referring to development apply to the original package. The top-level confirmation protocol and wrapper define this new study. The wrapper corrects returned scope metadata without changing numerical outputs.

## Predeclared analysis

`CONFIRM_PROTOCOL.json` is the executable specification. There are **30 primary paired contrasts**, jointly Holm-adjusted:

1. Six game tests: recent score versus random selection on direct suppression in both frozen-policy domains at k=3,10,25.
2. Twenty-four navigation tests: six controller/evidence comparisons at k=1 under pursuit, each assessed on contact incidence, contact duration, goal progress and command modification. Comparisons include the shared predictor versus brake, ranking versus random targeting, adaptive versus threshold/periodic triggering, gap-aware versus naive selective-loss handling, and gateway evidence versus forged reports.

The primary p-value is an exact two-sided paired **sign test** of equal positive/negative sign probabilities among non-ties, using independent training seeds. It is not specifically a test of equal population means. All-zero comparisons have p=1 and remain in the Holm family. Mean paired differences and 10,000-resample paired-seed percentile 95% intervals are reported separately; these are marginal intervals, not simultaneous intervals. Ties use a fixed absolute 1e-12 numerical tolerance. No post hoc non-inferiority, equivalence, or efficiency-frontier assertion is authorized.

Contact duration is prospectively a primary endpoint for THIS new confirmation, alongside contact incidence. This does not retroactively change the earlier R3 primary endpoints. All other authority levels, no-attack conditions, diagnostics, and timing outcomes are retained descriptively. No early stop based on favorable significance or expansion of seeds after inspecting the primary results.

Useful documentation for the analysis primitives:
- SciPy binomial test: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
- SciPy paired bootstrap/resampling: https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.bootstrap.html

The wrapper implements resampling of the paired difference vector directly. Sign tests require the stated sign-probability null, and bootstrap intervals are approximations. Neither makes the study assumption-free.

## Run without reinstalling anything

Use `RUN_COMMANDS.md`. The launch script uses the already working environment at:

`/mnt/caenl/active/aaf-r3-strengthening-v1/AAF_R3_STRENGTHENING_GPU_v1/.venv/bin/python`

It checks the pinned dependencies, source hashes, free disk space, CUDA, and actual VMAS preflight. Caches remain on `/mnt/caenl`, not the home partition. No pip installation, driver changes, model-service calls, or API keys are involved. No other GPU process is stopped. The worker is detached and holds an exclusive lock to refuse duplicate launches.

## Resume and completion

Repeat the same start command after an interruption. Complete compatible records and final policy checkpoints are reused. Interrupted training or an incomplete evaluation restarts from its recorded seed; no mid-optimizer-step resume is claimed. Incompatible source/protocol/environment causes a stop, not a forced overwrite.

`RUN.status` progresses through `STARTING`, `RUNNING`, `ANALYZING`, `PACKAGING`, and `COMPLETE`; failures become `FAILED`. Execution completion is not scientific acceptance. Keep all policies even when nominal performance is low; report the policy-quality table rather than excluding them.

## Return archive and full-data preservation

Return **`AAF_R3_CONFIRMATION_REVIEW.zip`** and retain the entire server output directory.

The review ZIP includes every condition summary and all 52,480 episode records, all final policy weights, all game proposal streams, code, protocol, analysis and checksums. It additionally includes the **first evaluation world (fixed index 0) from every navigation configuration** as a full physical trace. This selection is fixed before execution and is not based on outcomes.

The full 32-world arrays remain on IBM under `results/confirmation/runs/`. Every full trajectory is validated by the server-side analysis before the review ZIP is made. `FULL_SHA256SUMS.txt` identifies the full retained dataset. The review ZIP is intentionally not a full raw-trajectory export, to keep the transfer manageable. A later record-level review can reconstruct all means from episode records but cannot independently reconstruct all physical transitions using only the index-0 slices.

On failure, return `AAF_R3_CONFIRMATION_FAILURE.zip` or the status output. Do not remove old data, alter settings in place, change the primary family, or weaken integrity tests.
