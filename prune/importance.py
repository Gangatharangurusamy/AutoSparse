"""
Importance scoring for pruning decisions.

Two criteria are implemented, selectable via a config flag:

1. SALIENCY (primary, recommended):
     importance[i,j] = |weight[i,j] * grad[i,j]|

   This is a first-order Taylor estimate of the loss increase from removing
   connection (i,j). See DESIGN.md for the full derivation.

2. MAGNITUDE (baseline for comparison):
     importance[i,j] = |weight[i,j]|

   This is the simplest pruning heuristic: remove the smallest weights.
   It's explicitly called out as the weak baseline in the challenge brief.

Both operate on the same data structures and are swappable via the
`criterion` argument, not two divergent code paths.

Note on masked entries: the gradient at a masked (pruned) entry is exactly
zero (forced by the chain rule via w * mask in the forward graph), so:
  - saliency importance = |0 * 0| = 0 at masked entries
  - magnitude importance = |0| = 0 at masked entries
Both correctly assign zero importance to already-pruned connections.
"""

import numpy as np


def compute_importance(weight_tensor, criterion='saliency'):
    """Compute per-element importance scores for a weight tensor.

    Args:
        weight_tensor: Tensor with .data, .grad, and .mask attributes.
                       Must have been through a recent backward() pass
                       if criterion='saliency' (so .grad is populated).
        criterion:     'saliency' for |w * g| (first-order Taylor),
                       'magnitude' for |w| (baseline).

    Returns:
        numpy array of shape weight_tensor.data.shape, with non-negative
        importance scores. Higher = more important = prune last.
    """
    w = weight_tensor.data

    if criterion == 'saliency':
        g = weight_tensor.grad
        if g is None:
            raise ValueError(
                "Saliency criterion requires gradients. "
                "Call backward() before compute_importance()."
            )
        # Add a tiny fraction of weight magnitude to break ties deterministically when gradients shrink to zero
        importance = np.abs(w * g) + 1e-15 * np.abs(w)

    elif criterion == 'magnitude':
        importance = np.abs(w)

    else:
        raise ValueError(f"Unknown criterion: {criterion!r}. "
                         f"Use 'saliency' or 'magnitude'.")

    return importance
