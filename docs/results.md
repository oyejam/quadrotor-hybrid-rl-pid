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
too far off-course or tilts too much) before completing a lap — raw-RPM control is a
genuinely hard RL problem to learn from scratch in this budget. This is itself a
relevant data point for the hybrid comparison: if Hybrid converges to a lower-variance,
longer-surviving policy in the same budget, that is direct evidence for H2 (the PID
term reduces the exploration burden). Needs the Hybrid run to compare against before
drawing conclusions.

**Hybrid run log (2026-11-20):** 500,000 timesteps (503,808 actual), 2 parallel envs,
~1582s wall-clock (~318 steps/sec, slower than PPO's ~420 due to the per-step PID
computation in the wrapper). Final eval: mean reward -149.3 ± 108.9, mean episode
length 263/480 steps.

Raw numbers are **not directly comparable to PPO's** here: `episode_reward` is a sum
over the episode, and Hybrid's episodes ran much longer (263 vs 176 steps) before
truncating, so a larger (more negative) total is expected even if per-step tracking is
similar or better. Normalizing: PPO ≈ -0.43 reward/step, Hybrid ≈ -0.57 reward/step —
on this raw per-step basis PPO looks slightly better, not worse. But Hybrid surviving
~50% longer before truncating (going too far off-course or tilting too much) is itself
meaningful: it suggests the PID floor keeps the drone within bounds for longer, even
if the exact position tracking during training wasn't tighter. This is exactly why the
formal comparison uses RMSE over a fixed one-lap trajectory (`evaluate_robustness.py`),
not this raw cumulative training-eval reward — see section 3 below for the actual
hypothesis test.

## 2. In-distribution tracking accuracy
Position RMSE for PID / PPO / Hybrid at the nominal mass and within the training DR
range (±10%).

## 3. Zero-shot robustness sweep (the core result, H1)

Run 2026-11-20: 7 mass scales (0.7-1.3x nominal, 5% steps near center), 10 episodes
each, all 3 controllers, 210 episodes total. Plot: `results/robustness_plot.png`.
Raw data: `results/robustness_sweep.csv`.

**Mean RMSE (m) by controller and test-time mass scale:**

| mass_scale | PID | PPO | Hybrid |
|---|---|---|---|
| 0.70 | 0.107 | 0.120 | 0.907 |
| 0.80 | 0.098 | 0.094 | 1.234 |
| 0.90 | 0.093 | 0.063 | 0.697 |
| 1.00 (nominal) | 0.090 | 0.028 | 0.063 |
| 1.10 | 0.094 | 0.996 | 1.744 |
| 1.20 | 0.100 | 0.447 | 0.435 |
| 1.30 | 0.109 | 0.434 | 0.430 |

**H1 is not supported by this data — the opposite pattern appears.** PID is by far
the most robust controller: its RMSE barely moves (0.090-0.109 m) across the entire
±30% range. PPO is the best of the three exactly at the nominal mass it was centered
on (0.028 m, beating PID) but degrades sharply and asymmetrically away from it,
including inside its own training DR range (0.996 m at +10% mass). Hybrid is
competitive only right at nominal (0.063 m) and is the *worst* controller everywhere
else, by a wide margin (0.4-1.7 m) — including inside the training range, where it is
roughly an order of magnitude worse than PID.

So the hybrid controller did not inherit PID's robustness; if anything, adding the
learned residual made the combined controller *less* robust than PID alone across
most of the tested range.

## 4. Discussion

**H1 (robustness) — rejected by this run.** The predicted mechanism was that the PID
term gives the hybrid a "stability floor" that the residual builds on. Instead, the
residual appears to actively destabilize that floor outside a narrow band around the
training conditions: Hybrid underperforms plain PID everywhere except exactly at
nominal mass, including inside its own ±10% training DR range. Plausible causes worth
investigating before final write-up:
- `residual_clip_fraction=0.2` (config/hybrid.yaml) may simply be too large — the
  policy can override up to 20% of PID's commanded RPM per motor, which may be enough
  to meaningfully destabilize a controller that is otherwise quite well-behaved. Worth
  re-running with a smaller clip (e.g. 0.05-0.1) to see if robustness improves.
- 500k timesteps on CPU may not be enough for the hybrid policy to learn a *refinement*
  of PID rather than noisy interference — PPO's own learning curve suggests the
  RPM-level control task is already hard to learn cleanly in this budget (see section
  1), and the hybrid adds another layer of complexity (the residual must learn when
  to stay near zero vs. correct).
- Both PPO and Hybrid show a non-monotonic spike in error exactly at +10% mass (right
  at the training range's edge), which is the single worst point for both — this could
  be a genuine edge-of-distribution effect, or partly sampling noise from only 10
  episodes per point. More episodes (or more seeds) would help confirm this isn't
  noise before treating it as a real finding.

This is still a usable, honest result: it shows classical PID's well-known robustness
advantage over learned control in a concrete, measured way, and shows that naively
wrapping RL around a PID controller does not automatically inherit that robustness —
the wrapping has to be more careful (tighter residual bounds, more training, possibly
a different reward that penalizes residual magnitude) for the hybrid idea to work as
intended. That's a legitimate, more nuanced conclusion than the original hypothesis,
and worth presenting as such rather than forcing the data to fit H1.

**H2 (sample efficiency)** — not yet evaluated with a proper steps-to-threshold
comparison; the raw final-reward numbers in section 1 aren't a fair substitute (see
that section's caveat about episode-length normalization). Left as follow-up: extract
the reward-vs-timesteps curves from `results/ppo/evaluations.npz` and
`results/hybrid/evaluations.npz` and compare steps-to-threshold directly.

## 5. Limitations
- Training budget was CPU-only (document actual total timesteps used, wall-clock time,
  number of parallel envs) — note how this may affect how well PPO/Hybrid converged
  compared to a GPU-trained reference.
- Single reference trajectory (document which) — results may not generalize to other
  trajectories.
