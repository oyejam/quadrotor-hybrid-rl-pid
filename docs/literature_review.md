# Analytical Review of the Field

This project sits at the intersection of four lines of work: classical feedback
control for quadrotor trajectory tracking, pure reinforcement-learning (RL) control,
residual/hybrid RL-on-classical-control, and domain randomization (DR) for
sim-to-real robustness. This review covers each briefly, then identifies the specific
gap this project's experiment addresses.

## 1. Classical control (PID) for quadrotor trajectory tracking

PID and cascaded-PID controllers remain the default for quadrotor low-level control:
well understood, computationally cheap, and, critically for this project, robust
to parameter uncertainty without needing to be retrained. Their robustness comes from
being a *reactive* feedback law (correcting the currently observed error) rather than
a *learned mapping* from state to action that implicitly encodes assumptions about the
system's dynamics. The trade-off is that a fixed-gain PID controller can't improve its
behavior by exploiting experience, and its performance is capped by how well a human
tuned the gains for the nominal system.

## 2. Pure RL for quadrotor control

End-to-end RL policies (PPO, SAC, TD3, etc.) that output raw motor commands have been
shown to match or exceed hand-tuned PID in specific regimes (e.g. aggressive
maneuvers or wind rejection) by exploiting learned nonlinear corrections a linear PID
law cannot express [Hsiao 2019](https://web.stanford.edu/class/aa228/reports/2019/final62.pdf);
[Reinforcement Learning-Based Control of CrazyFlie 2.X](https://arxiv.org/pdf/2306.03951).
However, comparative studies also report the opposite: PID giving "much smoother
control... and smoother action outputs" than PPO/A2C on the same task
([Comparison of Traditional Optimal Control Methodologies to RL Agents in Quadcopter
Attitude Control](https://www.researchgate.net/publication/363736123_Comparison_of_Traditional_Optimal_Control_Methodologies_to_Reinforcement_Learning_Agents_in_Quadcopter_Attitude_Control_Applications)).
The field's conclusion is mixed and condition-dependent, not a clean win for either
side, which is itself one of the reasons a controlled, apples-to-apples comparison
(as attempted here) is worth running rather than relying on a prior result from a
different setup.

A separate, well-documented limitation of raw low-level RL is sample inefficiency:
learning a mapping directly to motor-level commands is harder than learning at a
higher level of abstraction, and hierarchical approaches that separate high-level
decisions from low-level control are a standard mitigation
([systematic review of DRL for legged locomotion](https://doi.org/10.3390/instruments10010008)).
This is directly relevant to this project's own finding (docs/results.md §1) that a
from-scratch PPO policy over raw RPM commands struggled to reliably complete the
reference trajectory within a 500k-timestep CPU-only budget.

## 3. Residual / hybrid RL on top of classical control

The approach this project implements, an RL policy that learns a bounded *residual*
correction added to a classical controller's output rather than replacing it, was
established by two foundational papers: residual policy learning
([Silver et al. 2018](https://arxiv.org/pdf/2106.08050) builds on this; original is
arXiv:1812.06298) and
[Residual Reinforcement Learning for Robot Control (Johannink et al., ICRA 2019)](https://www.researchgate.net/publication/335140884_Residual_Reinforcement_Learning_for_Robot_Control),
both motivated by the same argument used in this project's research plan: a classical
controller provides a stable baseline and a sample-efficiency head start, so the RL
component only has to learn what the baseline gets wrong.

Follow-on work has generally reported positive results for this architecture:
- **Optimising PID Control with Residual Policy RL** (2020,
  [RPRL](https://www.researchgate.net/publication/346488030_Optimising_PID_Control_with_Residual_Policy_Reinforcement_Learning))
  improved PID performance with a learned residual.
- **Improving Wind Resistance Performance of Cascaded PID Controlled Quadcopters
  using Residual RL** ([Shu et al. 2023](https://arxiv.org/abs/2308.01648)) is the
  closest prior work to this project's architecture: a residual RL policy on top of
  cascaded PID for a real quadcopter. They report ~50% position-deviation reduction
  over PID alone under wind disturbance, and state that
  **the trained controller "preserves its performance even when the quadcopter's
  mass and propeller's lift coefficient is changed between 50% to 150% from original
  training time."**

That last point is the key tension this project's results surface. Shu et al. found
residual-RL-on-PID *robust* to mass variation far outside their training distribution;
this project's own mass-perturbation sweep (docs/results.md §3) found the opposite:
the hybrid controller was the *least* robust of the three tested, including inside
its own training DR range. Three plausible reasons for the discrepancy:

1. **Where the residual acts.** Shu et al.'s residual corrects a *cascaded* PID
   structure (position loop → attitude loop → motor mixing), likely at an
   upper/mid-level command. This project's residual acts directly on raw per-motor
   RPM commands (ActionType.RPM in `gym_pybullet_drones`), the lowest and least
   structured level in the stack. A residual at a higher level of abstraction has a
   smaller, more semantically meaningful space to search and is plausibly easier to
   learn robustly than one at the raw-actuator level, consistent with the
   sample-efficiency literature in section 2.
2. **Disturbance type.** Wind is an *additive* external force/torque the policy can
   learn to counter with a bounded correction. Mass scaling is a *multiplicative*
   change to the plant's own dynamics (inertia, thrust-to-weight ratio). The
   "correct" compensation is not a fixed-magnitude patch but a state-dependent
   rescaling, which may be fundamentally harder for a residual bounded to a fixed
   fraction of PID's output to represent, especially when undertrained.
3. **Training budget.** This project's training ran on a CPU-only laptop (500k
   timesteps, ~20-30 min wall-clock); Shu et al.'s results likely reflect a
   substantially larger training effort. Section 3.1 of docs/results.md shows that
   even a large change to the residual's allowed magnitude (clip 0.2 → 0.08) did not
   close the gap, which is more consistent with an undertrained policy than with a
   structurally wrong residual bound.

This project's domain-randomization limitation is also consistent with a known issue
in the DR literature (below): a policy trained under DR is "limited to
within-the-distribution scenarios," and this project's sharpest failure point for
both learned controllers was right at +10% mass, the edge of the training
distribution, not deep outside it.

## 4. Domain randomization for sim-to-real robustness

DR (training across a distribution of simulated dynamics parameters rather than one
fixed value) is the standard tool for making a sim-trained policy robust to
real-world parameter uncertainty, used directly in this project's methodology
(docs/research_plan.md §4.3). It has demonstrated real transfer success for
quadrotors, including zero-shot sim-to-multi-real transfer across different physical
airframes
([Molchanov et al., Sim-to-(Multi)-Real, 2019](https://arxiv.org/pdf/1903.04628)) and
cross-platform racing policies
([One Net to Rule Them All, 2025](https://arxiv.org/pdf/2504.21586)).

The literature also documents DR's known failure modes, both observed in this
project's own results:
- Policies trained with DR remain reliable mainly *within* the training distribution,
  with no guarantee of graceful degradation just outside it, which matches this
  project's sharp error spike exactly at the training range's edge (+10% mass) rather
  than a smooth, monotonic degradation further out.
- Relying on DR alone (rather than, say, online adaptation) can push a policy toward
  overly conservative or "hyper-reactive" behavior as it tries to hedge across the
  whole randomized range, which may partly explain the high-variance training curves
  observed for both PPO and the original hybrid controller (docs/results.md §1).

## 5. Where this project sits

The specific question this project investigates, whether residual RL preserves or
improves a PID controller's robustness to unmodeled *mass* perturbation under a
constrained (CPU-only, raw-actuator-level) training setup, is not directly answered
by the closest prior work (Shu et al. 2023), which tests a different disturbance type
(wind) on a different residual level (cascaded, not raw-motor) with more training
resources. The result for H1 turned out to be a sharper negative than the first pass
of the analysis showed: once episode success was tracked, not just RMSE, both PPO and
the hybrid controller failed to complete the reference trajectory at all on
essentially every episode with mass at or below the value they were centered on
during training, recovering only once the true mass exceeded it (docs/results.md
§3-4). That is a concrete, measured data point narrowing where the
residual-RL-on-classical-control idea does and doesn't straightforwardly transfer,
rather than a refutation of the broader idea. The gap is plausibly in *how* the
residual's baseline action is computed relative to the true (randomized) mass and
*how much* it was trained, not in the underlying hybrid concept.

## Sources

- [Residual Reinforcement Learning for Robot Control (Johannink et al., ICRA 2019)](https://www.researchgate.net/publication/335140884_Residual_Reinforcement_Learning_for_Robot_Control)
- [Optimising PID Control with Residual Policy Reinforcement Learning (2020)](https://www.researchgate.net/publication/346488030_Optimising_PID_Control_with_Residual_Policy_Reinforcement_Learning)
- [Residual Reinforcement Learning from Demonstrations](https://arxiv.org/pdf/2106.08050)
- [Improving Wind Resistance Performance of Cascaded PID Controlled Quadcopters using Residual Reinforcement Learning (Shu et al. 2023)](https://arxiv.org/abs/2308.01648)
- [Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors (Molchanov et al. 2019)](https://arxiv.org/pdf/1903.04628)
- [One Net to Rule Them All: Domain Randomization in Quadcopter Racing](https://arxiv.org/pdf/2504.21586)
- [Reinforcement Learning Based Quadcopter Controller (Hsiao, Stanford AA228 2019)](https://web.stanford.edu/class/aa228/reports/2019/final62.pdf)
- [Reinforcement Learning-Based Control of CrazyFlie 2.X Quadrotor](https://arxiv.org/pdf/2306.03951)
- [Comparison of Traditional Optimal Control Methodologies to RL Agents in Quadcopter Attitude Control Applications](https://www.researchgate.net/publication/363736123_Comparison_of_Traditional_Optimal_Control_Methodologies_to_Reinforcement_Learning_Agents_in_Quadcopter_Attitude_Control_Applications)
- [A Systematic Review of Deep Reinforcement Learning for Legged Robot Locomotion](https://doi.org/10.3390/instruments10010008)
