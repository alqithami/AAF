# AAF strengthening development protocol v1

## Status and preservation

This package is a new **development experiment**, motivated by the completed R3
findings. It is not a concealed rerun of the earlier main study. All earlier null,
unfavorable, and historical results remain preserved. It contains no final new
performance numbers and no automatic confirmation stage. The immutable inherited
`vendor/aaf_r3` source is extracted from the returned `AAF_R3_NEXT_STAGE_RESULTS.zip`;
its per-file hashes are checked on every run.

Development choices below are specified before this package's outputs are read.
They are not externally registered. Subsequent method/configuration changes require
a new version and new held-out confirmation data. No outcome-driven seed exclusion.

## A. Game capacity and targeting opportunity

Both original environments and the corrected Beta-PPO learner are retained without
edits. Each domain uses fresh development seeds 281100 and 281101. Nominal training
runs 6,024 steps (not a claim of equality to the old main treatment), followed by
512 separate calibration steps. The final nominal checkpoint is fixed, not selected
by mitigation results. The frozen policy generates a 2,000-step ungoverned proposal
stream with five scripted maximum-request/zero-contribution agents from step 400.

We replay the SAME proposals under k = 3, 10, 25, H = 50, cooldown = 25,
B = 4 starts per W = 300. Fixed starts occur at 75,150,... and are identical for
all selectors. Recent score uses the 50 preceding proposal-violation indicators;
current-violation targeting uses the previous recorded proposal. Uniform random
targeting chooses k of all N agents, even on zero-score ties. This deliberately
isolates selection at exactly matched intervention opportunity, not detector value.

The future-window diagnostic knows every proposal in the next admitted window. It
chooses the k agents with the greatest total future violations. Its maximum is exact
ONLY for the fixed proposal stream, per-agent clamp, and disjoint fixed schedule.
It is neither an online policy nor an optimum over future learning trajectories.
The analytically expected uniform-random reduction and maximum targeting opportunity
are reported as well as the realized random draw.

Additional `scripted_sparse`, `scripted_rotating`, and `scripted_diffuse` streams
are explicit mechanism controls: baseline violation probability .03, five persistent
or rotating (.95 violation) sources after step 400, or .985 for all agents throughout.
They are not learned multi-agent behaviors. All four streams and all selectors are
reported; they cannot be selected as replacements because their results look better.

The native resource-sharing mean reward is mechanically 2.6 - .2 * violation rate
when capacity is fully allocated at N=50. It is not counted as independent evidence
of throughput, legitimate demand satisfaction, or service quality.

## B. Physics-based development

### Policies and units

Standard VMAS 1.5.2 navigation; four agents; unchanged first-order actuator lag .5.
New policy seeds: 281000 and 281001. Each uses 16 training worlds, 128 updates,
128 rollout steps and a 256-step episode/block horizon, as in the earlier main
budget. Calibration: 512 steps with separate seeds. Runtime policies are frozen.
Eight evaluation worlds per configuration, held apart from training/calibration.
Final checkpoint only. Main R3 seeds 191000--191009 are not used or pooled.

### Authority/filter block

For each policy seed and no-attack/pursuit condition:
- k in {1,2,4}, H=10, cooldown=5, B=4 starts, W=60, history=20, minimum coverage=.5.
- adaptive evidence ranking, adaptive random-k, instantaneous-threshold ranking,
  periodic ranking, with the inherited calibration rules and NEXT-step admissions.
- each rule uses both the original reactive brake and the same predictive candidate
  primitive. No method receives a private stronger filter within a filter block.
- unconstrained PPO, random policy, and unrestricted guards for both filter primitives
  are kept visible as references with explicitly different authority.

This varies maximum concurrent target count, not every authority parameter. The
long-run scheduled agent-time caps are 1/6, 1/3 and 2/3 for k=1,2,4; finite episodes
have truncation effects. Realized active time and actual modification are measured.
This is NOT an optimally tuned Pareto frontier and does not equalize realized use by
post hoc selection. Further equal-budget development calibration must precede claims
of calibrated efficiency in confirmation.

### Candidate predictive primitive (new prototype)

Read mass, drag, timestep, substeps and radii from the real engine. The model uses
semi-implicit Euler with drag once per control step and the same actuator lag. It
omits collision forces and holds all candidate commands constant for eight steps.
Physical pair clearance is radius-corrected. The dimensionless objective sums over
predicted steps: squared positive clearance deficit normalized by .05, and squared
speed excess normalized by .3. Thus model cost zero requires both predicted
clearance >= .05 and speed <= .3 over the finite horizon.

For each authorized target, candidate commands are incoming/current coordinate,
velocity damping, zero, and eight equally spaced unit directions. Two deterministic
coordinate passes minimize model cost; numerical ties minimize squared change from
the full incoming command. Non-authorized agents remain unchanged. For k>1 this is
coordinate search, not exhaustive joint optimization. There is no invariant-set,
terminal-feasibility, or certified physical-safety guarantee. Cost decrease in this
approximate model is NOT counted as an empirical safety result.

Preflight verifies one-step free-motion integration against actual VMAS at separated
states. Full-run one-step prediction errors are exported, including during contact.
Candidate model and actual-engine errors must be inspected before selecting this
primitive for a held-out confirmatory study.

### State-model diagnostics (not deployable evidence-only AAF claims)

`adaptive_benefit_state`: same received-risk CUSUM and next-step admission as
adaptive_rank, but target score is predicted reduction from a single-agent candidate
intervention, using trusted current gateway positions/velocities/previous force and
held current commands. Scores are computed after the transition for the next action;
next proposals and future true states are NOT observed. Trusted-state eligibility is
separate from evidence-history coverage. The method is expressly labeled privileged.

`immediate_state`: k=1, predictive primitive, pre-action predicted-risk trigger,
state-model target scores, and zero-step admission delay; same admission/duration
limits. This intentionally changes timing, trigger and selection together as a
positive diagnostic; it is not an isolated timing ablation or a claim of optimality.
The partial/full-information differences must not be hidden when interpreting it.

### Evidence and command block

At k=1, adaptive evidence ranking and the predictive primitive:
- `gateway`: trusted simulated gateway risk, clean delivery (already in main block).
- `forged_self_report`: the attack process overwrites its source risk payload with
  zero. The collector can preserve it without establishing its truth.
- `selective_naive`: incriminating source reports dropped; naive controller treats
  absence as zero. Actual missing values remain nan in the recorded received trace.
- `selective_gapaware`: same selective drop; a next-step simulated trusted checkpoint
  exposes the previous interior omission. It supplies no missing payload. A separate
  omission alarm may authorize bounded reversible control of that missing-evidence
  source, with `omission_target` explicitly logged; NOT responsibility/culpability.
  The two policies differ in omission handling; do not call it a pure detector ablation.
- `permanent_suppression`: all attack-source records and future checkpoints suppressed.
  No gap proof is invented. There is no separate trusted heartbeat-timeout assumption.
- `gateway_iid`: 20% independent supervisory evidence loss.
- `gateway_burst`: common-mode temporal loss, p(good->bad)=.025, p(bad->good)=.10.
- `stale_evidence_only`, `stale_commands_only`, `stale_both`: separate channel effects;
  delays {0,1,2,4}, expiry after two steps; command expiry invokes the same brake
  fallback for every relevant method. A shared-fallback gain is not solely AAF's gain.

The attack injector knows identity solely to modify proposals/payloads; original
controllers receive no identity labels. Gateway positions/velocities remain available
to the local filter, as in the original wrapper. These are simulated trust profiles,
not real TEEs, signed packet transport, radio interference or OS-level compromise.

## C. Timing, endpoints, and exports

Index t denotes the action producing transition s_t to s_(t+1). Histories/alarms
observed at the end of t start normal interventions at t+1. The immediate diagnostic
is separately labeled delay=0. Record first risk, model risk, alarm, admission,
actual command change, and contact; missing events remain null rather than zero.
Initial contact is recorded separately. Record all episodes including those that
finish before attack and report attack exposure; no conditional favorable exclusions.

Retain whole-episode contact/proximity (r_i+r_j+.005), goal progress, modification,
active time, contact-involved time, native completion, overspeed, evidence coverage,
omission controls, fallback and model error. New post-attack/timing endpoints are
prospectively collected here, not retroactively substituted into R3 primary results.

Every physics transition stores pre/post positions/velocities, proposals, submitted,
received and executed commands, forces, risk, received/controller evidence, sequence
gaps, governor state, selection masks, live/done masks, goals and contact geometry.
Policy weights and training metadata are exported. Game proposal streams and all
fixed-window starts are retained. Checksums prove file consistency, not authenticity
of a remote experiment or correctness of the scientific assumptions.

## D. Analysis and decision gate

Report each independent development policy seed and the across-seed descriptive
mean/range. Episodes are nested within policy seed. No significance testing,
non-inferiority claims, or automatic promotion to publication-ready performance.
The diagnostic is complete whether AAF wins, loses or ties. Unit/preflight failures
stop execution; poor task performance is retained and discussed, not hidden.

After inspection, freeze a defensible method, training/authority calibration and
confirmatory question using fresh seeds. Do not keep adding seeds or changing
endpoints until a favorable comparison appears. Existing R3 results remain intact.
