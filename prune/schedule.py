"""
Sparsity schedule for gradual pruning.

Implements the cubic sparsity ramp from Zhu & Gupta (2017),
"To Prune, or Not to Prune: Exploring the Efficacy of Pruning for
Model Compression."

The cubic schedule:
    s(t) = s_f + (s_i - s_f) * (1 - (t - t_0) / (t_n - t_0))^3

Where:
    s_f = final target sparsity (e.g. 0.90)
    s_i = initial sparsity (typically 0.0, i.e. start fully dense)
    t_0 = step at which pruning begins
    t_n = step at which pruning ends (then fine-tune at fixed sparsity)
    t   = current step

Why cubic and not linear?
    - Early on, the cubic ramp prunes slowly, giving the network time to
      learn initial representations before connections are removed.
    - Mid-training, pruning accelerates, removing connections while the
      network can still adapt (learning rate hasn't decayed too much).
    - Near the end of the pruning phase, the rate tapers off, making
      fewer changes as the network fine-tunes around its sparse structure.
    - The 20% post-pruning fine-tuning phase (after t_n) gives the network
      time to recover from the accumulated pruning without any further
      disruption.

This is strictly better than one-shot pruning (removing all connections at
once at the end), which gives the network zero chance to adapt its remaining
connections around the pruned topology. See DESIGN.md for the detailed
comparison.
"""


def target_sparsity(step, total_steps, final_sparsity,
                    initial_sparsity=0.0, begin_step=None, end_step=None):
    """Compute the target sparsity at the given training step.

    Args:
        step:             current training step (0-indexed)
        total_steps:      total number of training steps
        final_sparsity:   target sparsity at end of pruning phase (e.g. 0.90)
        initial_sparsity: sparsity at the start (default 0.0 = fully dense)
        begin_step:       step at which pruning begins (default: 0)
        end_step:         step at which pruning ends (default: 80% of total)

    Returns:
        float in [initial_sparsity, final_sparsity]: the target sparsity
        at this step.
    """
    if begin_step is None:
        begin_step = 0
    if end_step is None:
        end_step = int(0.8 * total_steps)

    # Before pruning begins: stay at initial sparsity
    if step < begin_step:
        return initial_sparsity

    # After pruning ends: stay at final sparsity
    if step >= end_step:
        return final_sparsity

    # Cubic interpolation between begin_step and end_step
    # s(t) = s_f + (s_i - s_f) * (1 - (t - t_0) / (t_n - t_0))^3
    fraction = (step - begin_step) / (end_step - begin_step)
    sparsity = final_sparsity + (initial_sparsity - final_sparsity) * (1 - fraction) ** 3

    return sparsity
