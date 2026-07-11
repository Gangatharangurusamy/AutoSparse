"""
THE PART 1 CORRECTNESS REQUIREMENT.

This is the single most heavily-weighted correctness test in the whole
challenge, and the thing the live defense will focus on. Three things must
be demonstrated:

  (A) Masking via `w * mask` in the forward graph gives dL/dw == 0 exactly
      at masked entries, via the ordinary chain rule (not a special case).

  (B) A NAIVE Adam implementation (masking only the forward pass, applying
      the raw Adam update without also masking the update or resetting
      state) lets a pruned weight DRIFT away from zero over subsequent
      steps, driven by stale momentum -- even though its instantaneous
      gradient is exactly zero every single step. This is the bug the
      brief says "most quick implementations get subtly wrong."

  (C) Our Adam (optim/adam.py) does NOT drift: a pruned weight stays
      exactly 0 across many subsequent steps. And when later revived
      (mask 0 -> 1), it starts from a clean m=0, v=0 state rather than
      inheriting stale momentum that would otherwise cause a spurious
      jump on the first post-revival update.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from engine.tensor import Tensor
from optim.adam import Adam

np.random.seed(0)


class MaskedParam(Tensor):
    """A Tensor with an attached prunable mask, for test purposes."""
    def __init__(self, data, mask=None):
        super().__init__(data, requires_grad=True)
        self.mask = np.ones_like(self.data) if mask is None else mask


def loss_fn(w, x, y):
    """Simple linear model loss: mean((w_masked @ x - y)^2), masked in the
    forward graph (so dL/dw at masked entries is forced to 0 by the chain
    rule -- see engine/tensor.py docstring)."""
    mask_t = Tensor(w.mask)
    w_eff = w * mask_t          # <-- masking happens IN the graph
    pred = w_eff @ x
    diff = pred - y
    return (diff * diff).mean()


def test_masked_weight_gradient_is_exactly_zero():
    w = MaskedParam(np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 4))
    w.mask[0, 1] = 0.0  # prune the 2nd weight
    x = np.random.randn(4, 5)
    y = np.random.randn(1, 5)

    loss = loss_fn(w, x, y)
    loss.backward()

    assert w.grad[0, 1] == 0.0, "masked weight must receive exactly zero gradient"
    print(f"[PASS] masked weight gradient is exactly 0.0 (grad={w.grad[0,1]})")


def test_naive_adam_lets_pruned_weight_drift():
    """Demonstrates the BUG: masking only the forward pass, then applying
    a textbook Adam update without masking the update itself and without
    resetting m/v on prune, lets a pruned weight silently drift."""

    class NaiveAdam:
        def __init__(self, params, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
            self.params = params
            self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
            self.t = 0
            self.m = [np.zeros_like(p.data) for p in params]
            self.v = [np.zeros_like(p.data) for p in params]

        def step(self):
            self.t += 1
            for i, p in enumerate(self.params):
                grad = p.grad  # NOTE: not masked -- but it's already 0 here anyway
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                p.data -= update  # BUG: update not masked -> stale momentum moves w

    w = MaskedParam(np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 4))
    x = np.random.randn(4, 8)
    y = np.random.randn(1, 8)
    opt = NaiveAdam([w])

    # Run a few steps UNPRUNED first so entry [0,1] builds up real momentum.
    for _ in range(5):
        w.zero_grad()
        loss = loss_fn(w, x, y)
        loss.backward()
        opt.step()

    # Now prune entry [0,1]. Its gradient will be exactly 0 from now on,
    # but its stale momentum (m, v) is still sitting there.
    w.mask[0, 1] = 0.0
    w.data[0, 1] = 0.0
    value_right_after_prune = w.data[0, 1]

    for _ in range(20):
        w.zero_grad()
        loss = loss_fn(w, x, y)
        loss.backward()
        assert w.grad[0, 1] == 0.0  # gradient really is 0 every step
        opt.step()

    drifted = w.data[0, 1]
    print(f"[DEMONSTRATED BUG] naive Adam: pruned weight was {value_right_after_prune:.6f} "
          f"right after pruning, drifted to {drifted:.6f} after 20 more steps "
          f"despite zero gradient every step (stale momentum).")
    assert drifted != 0.0, (
        "expected the naive optimizer to demonstrate drift; if this fails, "
        "the demo itself needs a larger lr or more pre-prune steps"
    )


def test_our_adam_prevents_drift_and_clean_revival():
    w = MaskedParam(np.array([1.0, 2.0, 3.0, 4.0]).reshape(1, 4))
    x = np.random.randn(4, 8)
    y = np.random.randn(1, 8)
    opt = Adam([w], lr=0.1)

    # Build up real momentum on entry [0,1] before pruning.
    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn(w, x, y)
        loss.backward()
        opt.step()
    assert opt.m[0][0, 1] != 0.0, "sanity check: momentum should be nonzero before pruning"

    # Prune entry [0,1] and sync the optimizer's view of the mask.
    w.mask[0, 1] = 0.0
    opt.sync_masks()

    assert w.data[0, 1] == 0.0, "pruned weight must be hard-zeroed immediately"
    assert opt.m[0][0, 1] == 0.0, "momentum must be reset to zero on prune"
    assert opt.v[0][0, 1] == 0.0, "second moment must be reset to zero on prune"

    # Run many more steps: the pruned weight must NOT drift, unlike the naive case.
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(w, x, y)
        loss.backward()
        assert w.grad[0, 1] == 0.0
        opt.step()
        assert w.data[0, 1] == 0.0, "pruned weight drifted away from zero -- BUG"

    print("[PASS] our Adam: pruned weight stays exactly 0.0 across 50 subsequent steps")

    # Now revive it (mask 0 -> 1) and confirm it starts clean (no stale jump).
    w.mask[0, 1] = 1.0
    w.data[0, 1] = 0.05  # simulate re-initializing the revived weight to a small value
    opt.sync_masks()

    pre_revival_value = w.data[0, 1]
    opt.zero_grad()
    loss = loss_fn(w, x, y)
    loss.backward()
    opt.step()
    post_step_value = w.data[0, 1]
    step_size = abs(post_step_value - pre_revival_value)

    # With m=0, v=0 at revival, the very first Adam update for this entry is
    # bounded (Adam's first step is always ~lr in magnitude, no bias-corrected
    # blowup from ancient history). If state had NOT been reset, m_hat/v_hat
    # would be dominated by pre-prune history unrelated to the current
    # gradient, and could produce a much larger, spurious jump.
    assert step_size <= opt.lr * 1.5, (
        f"revived weight moved by {step_size:.4f} in one step, "
        f"suspiciously large for a freshly-reset Adam state (lr={opt.lr})"
    )
    print(f"[PASS] revived weight: first post-revival step size={step_size:.4f} "
          f"(bounded, no stale-momentum jump)")


if __name__ == "__main__":
    test_masked_weight_gradient_is_exactly_zero()
    test_naive_adam_lets_pruned_weight_drift()
    test_our_adam_prevents_drift_and_clean_revival()
    print("\nAll masked-weight correctness tests passed.")
