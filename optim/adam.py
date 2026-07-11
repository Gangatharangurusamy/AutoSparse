"""
Adam optimizer, written from scratch, with explicit handling for masked
(pruned) parameters.

THE TRAP (read this before touching anything else):

Masking a weight via `w * mask` in the forward pass (see engine/tensor.py)
already guarantees dL/dw = 0 at masked entries via the ordinary chain rule.
That looks like it's "enough" -- and a naive Adam step looks correct:

    m = beta1*m + (1-beta1)*grad         # grad is 0 at masked entries
    v = beta2*v + (1-beta2)*grad**2      # also 0 at masked entries
    w -= lr * m_hat / (sqrt(v_hat)+eps)

It is NOT enough. `grad == 0` does not make `m == 0`. If the weight had
nonzero momentum *before* it was pruned, then after pruning:

    m_new = beta1 * m_old + (1-beta1)*0 = beta1 * m_old

m decays geometrically toward zero but never actually reaches it. Every
subsequent step still computes a nonzero update `lr * m_hat / (sqrt(v_hat)+eps)`
from that stale momentum and applies it to `w` -- so a "pruned" weight
silently drifts away from zero even though its mask is 0 and its gradient
is 0. If you only re-apply the mask to `w` after the update (`w *= mask`),
you hide the symptom (w stays displayed as 0) but corrupt anything that
reads `w` before the next masking, and you still corrupt `m`/`v` for good.

Two things must both be true:
  1. The update itself must be masked: `update *= mask` before applying to
     w, not just w *= mask afterward. This guarantees a masked weight
     literally never moves, regardless of stale m/v.
  2. Optimizer state must be RESET (m=0, v=0, step count local to that
     param entry conceptually) at the moment an entry transitions from
     active -> pruned. Otherwise the stale momentum is still sitting there
     the moment the weight is revived (mask 0 -> 1), and the very first
     post-revival update will be a large, spurious jump driven by
     ancient gradient history that has nothing to do with the current
     loss landscape. This is exactly the failure mode the challenge
     brief calls out ("the weight suddenly jumps because of old
     optimizer history").

On revival we do NOT try to be clever and restore old state -- we start
that entry from m=0, v=0, i.e. as if it were a freshly initialized
parameter. This is the conservative, defensible choice: any other choice
(e.g. keeping stale m/v around "just in case") reintroduces the exact bug
described above. We reason about this trade-off explicitly in DESIGN.md.
"""

import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        params: list of Tensor objects (must have .data, .grad).
                Optionally each param may have a `.mask` attribute (numpy
                array of 0/1, same shape as .data). If absent, treated as
                all-ones (no pruning).
        """
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        # remember previous mask per-param so we can detect 0->1 / 1->0
        # transitions and reset state accordingly.
        self._prev_mask = [self._get_mask(p).copy() for p in params]

    @staticmethod
    def _get_mask(p):
        m = getattr(p, "mask", None)
        return np.ones_like(p.data) if m is None else m

    def sync_masks(self):
        """Call this immediately after a pruning step changes any param's
        .mask. Detects newly-pruned entries (1->0) and resets their Adam
        state to zero, so stale momentum can never leak into a future
        revival. Detects newly-revived entries (0->1) too -- for symmetry
        and to document the decision explicitly, though since we already
        zero state on prune, revived entries are already clean.
        """
        for i, p in enumerate(self.params):
            cur_mask = self._get_mask(p)
            prev_mask = self._prev_mask[i]
            newly_pruned = (prev_mask == 1) & (cur_mask == 0)
            if newly_pruned.any():
                self.m[i][newly_pruned] = 0.0
                self.v[i][newly_pruned] = 0.0
            # also hard-zero the underlying data at newly pruned entries
            p.data[newly_pruned] = 0.0
            self._prev_mask[i] = cur_mask.copy()

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            mask = self._get_mask(p)
            grad = p.grad * mask  # belt-and-suspenders: force-zero grad at masked entries

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            update = update * mask  # THE FIX: mask the update itself, not just the weight

            p.data -= update

            # defensive: even if something upstream went wrong, a masked
            # entry's data must be exactly zero after every step.
            p.data[mask == 0] = 0.0
