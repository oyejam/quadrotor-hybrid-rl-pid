# Results

Fill in as experiments complete. Keep this in sync with `README.md`'s "Description of
the Obtained Results" section (link here, don't duplicate).

## 1. Training curves
- PPO: episode reward vs. timesteps (from `results/ppo/evaluations.npz`)
- Hybrid: episode reward vs. timesteps (from `results/hybrid/evaluations.npz`)
- Compare sample efficiency (H2): steps to reach a fixed reward/RMSE threshold.

**PPO run log (2026-11-20):** 500,000 timesteps (503,808 actual, PPO completes full
rollouts), 2 parallel envs, ~1199s wall-clock (~420 steps/sec) on this CPU-only
machine. Final eval: mean reward -75.2 ± 67.5, mean episode length 176/480 steps.
The large variance and short mean episode length (well under the 480-step full
trajectory) suggest the policy frequently triggers the truncation condition (flies
too far off-course or tilts too much) before completing a lap. Raw-RPM control is a
hard RL problem to learn from scratch in this budget. This is itself a relevant data
point for the hybrid comparison: if Hybrid converges to a lower-variance,
longer-surviving policy in the same budget, that is direct evidence for H2 (the PID
term reduces the exploration burden). Needs the Hybrid run to compare against before
drawing conclusions.

**Hybrid run log (2026-11-20):** 500,000 timesteps (503,808 actual), 2 parallel envs,
~1582s wall-clock (~318 steps/sec, slower than PPO's ~420 due to the per-step PID
computation in the wrapper). Final eval: mean reward -149.3 ± 108.9, mean episode
length 263/480 steps.

These raw numbers aren't directly comparable: `episode_reward` is a sum over the
episode, and Hybrid's episodes ran much longer (263 vs 176 steps) before truncating,
so a larger (more negative) total is expected even if per-step tracking is similar or
better. Normalized, PPO is about -0.43 reward/step and Hybrid about -0.57 reward/step,
so on a per-step basis PPO looks slightly better here. Hybrid surviving roughly 50%
longer before truncating is still meaningful on its own: it suggests the PID floor
keeps the drone within bounds for longer, even if position tracking during training
wasn't tighter. The formal comparison in section 3 uses RMSE over a fixed one-lap
trajectory instead of this raw training-eval reward.

## 2. In-distribution tracking accuracy
Position RMSE for PID / PPO / Hybrid at the nominal mass and within the training DR
range (±10%).

## 3. Zero-shot robustness sweep (the core result, H1)

Run 2026-11-20: 7 mass scales (0.7-1.3x nominal, 5% steps near center), 10 episodes
each, 4 controllers (PID, PPO, Hybrid clip=0.2, Hybrid clip=0.08), 280 episodes total.
Plots: `results/robustness_plot.png` (RMSE), `results/success_rate_plot.png` (success
rate). Raw data: `results/robustness_sweep.csv`.

> **Correction.** The first pass of this sweep recorded RMSE only, not whether each
> episode actually completed the trajectory. Adding success tracking
> (`episode_length`, `success`, see docs/research_plan.md §4.4-4.5) showed that
> several of the low RMSE numbers reported earlier came from episodes that crashed
> almost immediately. PPO's previously reported best RMSE (0.028m at nominal mass)
> came from episodes with a mean length of 12 of 480 steps: it wasn't tracking the
> trajectory, it was crashing before covering any real distance. The tables below
> replace that reading.

**Success rate (fraction of 10 episodes that completed in-bounds):**

| mass_scale | PID | PPO | Hybrid (clip=0.2) | Hybrid (clip=0.08) |
|---|---|---|---|---|
| 0.70 | 1.0 | 0.0 | 0.0 | 0.0 |
| 0.80 | 1.0 | 0.0 | 0.0 | 0.0 |
| 0.90 | 1.0 | 0.0 | 0.0 | 0.0 |
| 1.00 (nominal) | 1.0 | 0.0 | 0.0 | **1.0** |
| 1.10 | 1.0 | 1.0 | 0.0 | 0.0 |
| 1.20 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1.30 | 1.0 | 1.0 | 1.0 | 1.0 |

**Mean episode length (of 480 steps):**

| mass_scale | PID | PPO | Hybrid (clip=0.2) | Hybrid (clip=0.08) |
|---|---|---|---|---|
| 0.70 | 480 | 16 | 57 | 57 |
| 0.80 | 480 | 18 | 88 | 85 |
| 0.90 | 480 | 19 | 78 | 140 |
| 1.00 (nominal) | 480 | 12 | 44 | 482 |
| 1.10 | 480 | 482 | 290 | 189 |
| 1.20 | 480 | 482 | 482 | 482 |
| 1.30 | 480 | 482 | 482 | 482 |

PID is the only controller that reliably completes the trajectory across the full
±30% range tested. Both PPO and the original hybrid (clip=0.2) crash on nearly every
episode at or below nominal mass (0% success from -30% to 0%), and only become
reliable once the true mass is at least 10% heavier than what they were trained
around. That's a sharp, asymmetric failure mode the RMSE-only view hid completely: a
drone lighter than or equal to its training-nominal mass causes near-total failure for
both learned controllers, while a heavier drone doesn't.

A plausible physical explanation, worth stating as a hypothesis rather than a proven
mechanism: the RPM action mapping used here
(`rpm = HOVER_RPM * (1 + 0.05*action)`, see `src/envs/traj_track_aviary.py` via
`BaseRLAviary._preprocessAction`) computes `HOVER_RPM` from the *nominal* mass,
fixed regardless of the actual randomized mass. For the same policy output, a lighter
drone receives more thrust per unit mass than a heavier one (same force, less
inertia), so the same action is more likely to cause overshoot and
instability when the drone is lighter, and more forgiving when it's heavier. PID does
not have this problem because it reacts to the currently measured error rather than
applying a fixed-mass-calibrated baseline action.

**RMSE, unconditional (all episodes, including crashed ones, what was originally
reported):**

| mass_scale | PID | PPO | Hybrid (clip=0.2) | Hybrid (clip=0.08) |
|---|---|---|---|---|
| 0.70 | 0.107 | 0.120 | 0.907 | 0.924 |
| 0.80 | 0.098 | 0.094 | 1.234 | 0.898 |
| 0.90 | 0.093 | 0.063 | 0.697 | 0.688 |
| 1.00 (nominal) | 0.090 | 0.028 | 0.063 | 0.088 |
| 1.10 | 0.094 | 0.996 | 1.744 | 0.924 |
| 1.20 | 0.100 | 0.447 | 0.435 | 0.434 |
| 1.30 | 0.109 | 0.434 | 0.432 | 0.432 |

**RMSE, conditional on success only (N/A = zero successful episodes at that point;
the only version of this table that is a fair tracking-quality comparison):**

| mass_scale | PID | PPO | Hybrid (clip=0.2) | Hybrid (clip=0.08) |
|---|---|---|---|---|
| 0.70 | 0.107 | N/A | N/A | N/A |
| 0.80 | 0.098 | N/A | N/A | N/A |
| 0.90 | 0.093 | N/A | N/A | N/A |
| 1.00 (nominal) | 0.090 | N/A | N/A | 0.088 |
| 1.10 | 0.094 | 0.996 | N/A | N/A |
| 1.20 | 0.100 | 0.447 | 0.435 | 0.434 |
| 1.30 | 0.109 | 0.434 | 0.430 | 0.432 |

The unconditional table alone makes PPO and Hybrid (clip=0.2) look best at nominal
mass, exactly where their success rate is 0%. The conditional table is the fair
comparison: wherever a learned controller does complete the trajectory, its tracking
error is still 4-10x worse than PID's.

H1 is rejected, and more decisively than the uncorrected RMSE numbers suggested. The
hypothesis predicted the hybrid would degrade more gracefully than PID or PPO outside
the training range. Instead, the hybrid (either clip setting) fails to complete the
trajectory at all across most of the tested range, including inside its own training
distribution, while PID never fails. The hybrid inherited PPO's brittleness to light
drone mass without reliably inheriting PID's floor.

### 3.1 Ablation: does a tighter residual bound close the gap?

Section 4 flagged `residual_clip_fraction=0.2` as a likely cause and proposed testing
a tighter bound. Ran the identical setup with `residual_clip_fraction=0.08`
(`config/hybrid_tight_clip.yaml`, everything else unchanged). See the 4-column
tables in section 3 above for the full comparison.

Tightening the clip produced one specific improvement: the tight-clip hybrid is the
only learned controller that reliably completes the trajectory at nominal mass (100%
success, RMSE 0.088m, matching PID). It did not broaden the operating range. Both
clip settings still fail on every episode below nominal mass, and the tight-clip
version also fails again at +10% before recovering at +20%/+30%. With only 10
episodes per point, that dip could be a real narrow failure mode or sampling
variance; more episodes would be needed to tell which.

A tighter residual bound can make the hybrid reliable exactly at the mass it was
tuned around, but it doesn't fix the asymmetric brittleness to lighter-than-nominal
mass. That's consistent with the action-mapping explanation above: a clip bound
constrains the residual's magnitude, not the miscalibration in the baseline RPM it's
added to.

## 4. Discussion

**H1 (robustness): rejected.** The failure mode is sharper than the RMSE-only numbers
first suggested. The predicted mechanism was that the PID term gives the hybrid a
stability floor the residual builds on. Instead, both the hybrid and plain PPO fail
to complete the trajectory at all across most of the tested mass range, not just
track it worse. Causes investigated:
- ~~`residual_clip_fraction=0.2` may simply be too large~~: tested in section 3.1. A
  tighter 0.08 clip fixed reliability at nominal mass but left the broader asymmetric
  failure unresolved, so the clip bound is a partial cause at most.
- **Leading hypothesis (section 3):** the fixed-nominal-mass `HOVER_RPM` baseline in
  the RPM action mapping is miscalibrated for lighter actual masses, making
  instability more likely below nominal and progressively less likely above it. This
  matches the sharp success-rate transition at the nominal-mass boundary for both
  PPO and Hybrid.
- 500k timesteps on CPU may not be enough for the hybrid policy to learn a refinement
  of PID rather than noisy interference. Raw-RPM control is already a hard RL
  problem to learn cleanly in this budget (see section 1).
- The non-monotonic success dip at +10% for the tight-clip hybrid (section 3.1) may
  be a genuine edge-of-distribution effect or partly sampling noise from only 10
  episodes per point; more episodes or seeds would help confirm which.

This result still shows classical PID's robustness advantage in concrete, measured
terms: 100% success everywhere tested against a hard failure boundary for both
learned controllers. It also narrows down why the hybrid idea didn't transfer here:
not simply undertraining, but a specific, physically-motivated miscalibration in how
the residual's baseline action is computed relative to the true mass.

**H2 (sample efficiency):** not yet evaluated with a proper steps-to-threshold
comparison; the raw final-reward numbers in section 1 aren't a fair substitute (see
that section's caveat about episode-length normalization). Left as follow-up: extract
the reward-vs-timesteps curves from `results/ppo/evaluations.npz` and
`results/hybrid/evaluations.npz` and compare steps-to-threshold directly.

## 5. Limitations
- Training budget was CPU-only (document actual total timesteps used, wall-clock time,
  number of parallel envs). Note how this may affect how well PPO/Hybrid converged
  compared to a GPU-trained reference.
- Single reference trajectory (document which): results may not generalize to other
  trajectories.
- Success rate is a binary outcome estimated from only 10 episodes per (controller,
  mass) cell, so a reported rate of 0.0 or 1.0 could flip with a different random
  seed at the boundary points (e.g. the ±10% transitions in section 3, and the
  non-monotonic dip noted in section 3.1). The qualitative pattern (hard failure
  below nominal mass, recovery above it) is large and consistent enough across
  controllers to trust, but exact transition points should not be over-interpreted
  with this sample size.
- Single policy seed per controller (seed=0 in config/*.yaml): a different training
  seed could land in a different region of this failure landscape; the finding here
  is about this specific trained policy, not a guarantee about PPO/Hybrid in general
  on this task.
