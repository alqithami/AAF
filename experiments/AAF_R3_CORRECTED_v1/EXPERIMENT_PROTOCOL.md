# AAF-R3-v1: prospective component and transfer evaluation

Status: code and experiment protocol supplied; main results NOT completed.
The pilot is for implementation/runtime/learning checks. Main seeds are held out.
Archive: alqithami/AAF, commit a7fb5c0a2a2a55ef823658d7786389fd5337b264.

## R3.1 Architectural contribution

The tested proposition is not that CUSUM, evidence scoring, shaping, or shielding is
new. It is that explicitly constrained triggering and evidence-based targeting can
change the operational safety/performance/intervention trade-off. Components need
not contribute equally. Negative and null results are reportable.

## R3.2 Component study in both original games

The resource-sharing and public-goods equations and default local observation features
are adapted from the recorded repository version. Population 50; resource exponent 1;
penalty 0.2; observation noise 0.01; shared social reward weight 0.3. The learner is
shared Beta-PPO with two 128-unit ReLU layers, learning rate 3e-4, discount .99,
GAE .95, clip .2, entropy .01, value-loss coefficient .5, gradient norm .5,
128-step rollouts, four optimizer epochs and minibatches up to 1024.

Each main seed first produces a shared 1,024-step ungoverned nominal checkpoint.
An independently seeded 512-step frozen-policy nominal calibration rollout supplies
the mean baseline and 95th percentile for the instantaneous rule. No attack or test
outcome enters calibration. Every treatment starts from the same nominal checkpoint
and optimizer state within seed and continues online learning for 5,000 steps.

Conditions: no scripted attack, persistent maximum-request/zero-contribution attack,
and 50-step pulses separated by 150 inactive steps. Five of 50 identities are selected
at the start by an independent RNG. Attack onset is step 1,000. No-attack runs still
contain learning-induced norm violations; their alarms are not automatically false.

Seven primary methods: PPO only; full AAF including shaping; adaptive-CUSUM with
recent-evidence ranking and clamp; the same detector with random-k targeting;
instantaneous threshold with ranking; periodic triggering with ranking; unrestricted
Static Guard. CUSUM initial threshold 5, floor .05, slack .01, target update rate .05,
gain (t+1)^(-.6). Instant threshold is the nominal 95th percentile. The period is 75
steps, derived from the authority cap rather than fitted to test outcomes. Trigger probabilities are
NOT claimed equal; the comparison matches enforceable authority limits and reports
realized utilization separately.

Bounded arms share k=3 targets, 50-step duration, no overlap or extension of an active
window, 25-step cooldown, and four admitted starts in any 300-step rolling window.
A newly admitted intervention begins on the next action. Minimum history coverage
is .5 over a 50-step window; unavailable evidence is NaN, not zero. All-zero or
insufficient histories abstain. Random-k selects the same number from the same
coverage-eligible population, without the historical all-agent switch. Static Guard
has unrestricted authority and is identified as such, not presented as budget matched.

Primary outcomes: executed violation fraction, unshaped environment reward, and
constrained agent-time. Secondary diagnostics: attempts, actual action modification,
L1 displacement, shaping debit, Gini, admission/denial/trigger counts, all alarm times,
first post-injection alarm coverage/delay and operational ranking. A targeted identity
not in the scripted set is not automatically falsely accused: honest learners may
also violate the declared norm. Intervention costs are reported as quantities, not
combined using an outcome-selected cost weight.

Four primary contrasts: adaptive_rank minus adaptive_random, threshold_rank,
periodic_rank; aaf_full minus adaptive_rank. Twenty independent seed blocks per domain,
with initialization, environment, attacker and target-selection RNGs separated.

## Additional training-semantics check

The historical runner replaces sampled actions through the attacker or shield and
then stores the changed action with the original proposal's log probability. The
new learner stores proposals and matching likelihoods. Scripted actions after takeover
are excluded from actor and value regression. This is not a cosmetic refactor.

Two additional diagnostic labels, legacy__ppo_only and legacy__aaf_full, intentionally
use executed actions/original likelihoods and include scripted samples. Their results
quantify sensitivity to that update convention under the SAME new authority protocol.
They do not reproduce the entire historical grid or prove which original numerical
outcomes would remain unchanged. They must never be pooled with corrected methods.
The original 19,440-record audit remains a provenance result, not validation of a
corrected PPO training implementation.

## R3.3 Physics-based runtime transfer

Dependency: VMAS 1.5.2, standard `navigation` scenario, four holonomic agents,
collisions enabled, individual progress/collision reward, default scenario observations
and LIDAR. Original VMAS positions, velocities, contacts and task rewards are used.
Our wrapper adds bounded force commands and first-order actuator lag:
    applied_force[t] = .5 applied_force[t-1] + .5 command[t].
This is an experimental extension, not a stock VMAS radio or actuator model.

Ten independent nominal PPO policies are trained with 16 parallel worlds,
128 updates of 128 vectorized steps (262,144 team-world steps, 1,048,576 agent
transitions per policy). Fixed-horizon episodes have 256 steps; completed worlds
are excluded from further learning within the batch. Calibration uses disjoint
world seeds. Final policies are frozen for runtime evaluation; there is NO claim of
closed-loop adaptive-policy training under network faults in this transfer study.

Seven runtime methods: PPO only, adaptive ranking, adaptive random-k, instantaneous
ranking, periodic ranking, unrestricted guard, and random policy as a policy-quality
reference. Full AAF plus shaping is omitted because reward shaping cannot alter a
frozen policy. The shaping contribution is tested in the online-learning game study.

Each method/condition is evaluated on 32 held-out episodes per nominal-policy seed.
Conditions cross no attack/scripted nearest-peer pursuit beginning at step 32 with four communication profiles:
- clean;
- independent 20% supervisory-evidence loss;
- common-mode Gilbert-style burst loss, good->bad .025 and bad->good .10,
  initialized at stationary bad probability .20; all agent reports in a world are
  lost in a bad step (correlation across agents and time);
- evidence AND upstream-command delays sampled uniformly from 0,1,2,4 control steps.
  Sequence numbers prevent rollback. Maximum acceptable age is two steps; stale
  evidence is discarded, and a stale command invokes the same local brake fallback
  in ALL methods. This is a simulated age deadline, not measured hardware real time.

The governor sees received post-step speed/contact risk, not attacker identities.
Risk is speed above .3 simulator units/time or an observed contact. A local gateway
has current trusted simulator velocity and range. For targeted agents it applies
bounded damping force clip(-2 velocity, -1, 1) on overspeed or nearby-contact risk.
This is a common heuristic braking filter for all controlled methods, not a new
shield primitive and NOT a collision-free or invariant-set certificate. Inertia,
contact forces, stale evidence and unseen intentions can leave residual collisions.

Authority: one target; 10-step intervals; four starts per 60 steps; five-step cooldown;
20-step evidence history and .5 minimum coverage. Missing evidence cannot become
negative blame evidence. A clock tick alone is not a new observation.

Primary outcomes: episode contact incidence, mean progress toward goals, and actual
command-modification fraction. Also report goal completion, final distance, speed
violations, constrained agent-time, evidence coverage, expired-command fallback,
and measured filter-only CPU/GPU-synchronized runtime. Measured compute excludes
policy, physics, transport and rendering. No simulated time quantity is labeled WCET.

The nominal-policy quality check compares clean, no-attack PPO with random policy
using mean episode goal progress aggregated within trained-policy seed. The paired
95% bootstrap interval for PPO-minus-random progress should exclude zero before
interpreting the transfer study as evidence about a competent learned controller.
Failure requires inspecting/training the nominal policy, not dropping poor seeds.
Pilot and main have different seed ranges. Any protocol change after pilot requires
a new fingerprint/output directory and is documented before main runs.

## Statistics, preservation and acceptance

Inferential unit: independent training seed, NOT an agent, time step, vectorized
world, repeated same-seed export, or individual held-out episode. VMAS episodes are
averaged within each trained-policy seed. Use paired seed effects, 10,000 paired-seed
bootstrap resamples for marginal intervals, paired t-tests as secondary tests, and
Holm adjustment over all primary contrasts/outcomes within each suite. Do not interpret
overlapping intervals as equivalence. No claim that every component wins is required.

Main plan: 840 corrected game treatments + 240 diagnostic legacy treatments + 560
VMAS seed/condition/method evaluations = 1,640 records. There are 20 seed blocks per
game and 10 nominal-policy seed blocks in VMAS. A more complex norm-discovery task,
learned adaptive adversary, real attestation stack, physical robot, and hard-real-time
certification remain outside this protocol.

No existing figure or result file is overwritten. New plots are generated only from
real output records, in separate directories with a consistent semantic palette.
Smoke/pilot/incomplete outputs cannot generate the main numerical LaTeX table.
The response remains a WORKING DRAFT until data, plots, interpretation and final
cross-references have been reviewed together.

## Final timing details
The periodic comparator uses max(H + cooldown, ceil(W/B)): 75 steps in games and 15 in navigation. This uses the same available authority rather than imposing an arbitrarily sparse periodic schedule. Navigation pursuit starts at step 32; attack-exposed episode fraction is reported separately to reveal early completions.

Non-scripted constrained agent-time is the mean of the indicator that an agent is constrained and is not in the scripted identity set; it is not the conditional proportion of targets that are non-scripted and is not a false-accusation rate. Calibration metadata counts scalar calibration observations (pooled world observations for VMAS), not independent trained-policy seeds.
