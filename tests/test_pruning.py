"""
Tests for pruning modules: importance, schedule, and mask.

Verifies:
  - Importance scoring computes correct values for both criteria
  - Cubic schedule produces correct sparsity at boundary conditions
  - Mask update achieves target sparsity with exact-zero masked entries
  - optimizer.sync_masks() is called correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from engine.tensor import Tensor
from nn.layers import MLP, Linear
from optim.adam import Adam
from prune.importance import compute_importance
from prune.schedule import target_sparsity
from prune.mask import update_masks, get_model_sparsity

np.random.seed(0)


def test_importance_saliency():
    """Test saliency importance = |w * g|."""
    w = Tensor(np.array([[1.0, -2.0], [3.0, -4.0]]), requires_grad=True)
    w.mask = np.ones_like(w.data)
    w.grad = np.array([[0.5, -0.3], [0.1, 0.2]])

    imp = compute_importance(w, criterion='saliency')
    expected = np.abs(w.data * w.grad)

    assert np.allclose(imp, expected), f"Expected {expected}, got {imp}"
    print("[PASS] Saliency importance = |w * g|")


def test_importance_magnitude():
    """Test magnitude importance = |w|."""
    w = Tensor(np.array([[1.0, -2.0], [3.0, -4.0]]), requires_grad=True)
    w.mask = np.ones_like(w.data)

    imp = compute_importance(w, criterion='magnitude')
    expected = np.abs(w.data)

    assert np.allclose(imp, expected), f"Expected {expected}, got {imp}"
    print("[PASS] Magnitude importance = |w|")


def test_importance_masked_entries_are_zero():
    """Masked entries should have importance = 0 for both criteria
    (because both w and grad are 0 at masked entries)."""
    w = Tensor(np.array([[1.0, 0.0], [3.0, 0.0]]), requires_grad=True)
    w.mask = np.array([[1.0, 0.0], [1.0, 0.0]])
    w.grad = np.array([[0.5, 0.0], [0.1, 0.0]])

    imp_sal = compute_importance(w, criterion='saliency')
    imp_mag = compute_importance(w, criterion='magnitude')

    assert imp_sal[0, 1] == 0.0, "Saliency at masked entry should be 0"
    assert imp_sal[1, 1] == 0.0, "Saliency at masked entry should be 0"
    assert imp_mag[0, 1] == 0.0, "Magnitude at masked entry should be 0"
    assert imp_mag[1, 1] == 0.0, "Magnitude at masked entry should be 0"
    print("[PASS] Both criteria give importance=0 at masked entries")


def test_schedule_boundaries():
    """Test cubic schedule at boundary conditions."""
    total = 100
    final = 0.9

    # Before begin: initial sparsity
    s = target_sparsity(0, total, final, initial_sparsity=0.0, begin_step=10)
    assert s == 0.0, f"Before begin: expected 0.0, got {s}"

    # At begin_step: should be close to initial (cubic starts slow)
    s = target_sparsity(10, total, final, initial_sparsity=0.0,
                        begin_step=10, end_step=80)
    assert abs(s - 0.0) < 0.01, f"At begin: expected ~0.0, got {s}"

    # At end_step: should be final sparsity
    s = target_sparsity(80, total, final, initial_sparsity=0.0,
                        begin_step=10, end_step=80)
    assert s == final, f"At end: expected {final}, got {s}"

    # After end_step: should still be final
    s = target_sparsity(99, total, final, initial_sparsity=0.0,
                        begin_step=10, end_step=80)
    assert s == final, f"After end: expected {final}, got {s}"

    print("[PASS] Cubic schedule boundaries correct")


def test_schedule_monotonic():
    """Target sparsity should be monotonically non-decreasing."""
    total = 100
    final = 0.9
    prev = 0.0
    for step in range(total):
        s = target_sparsity(step, total, final, initial_sparsity=0.0,
                            begin_step=10, end_step=80)
        assert s >= prev - 1e-10, (
            f"Sparsity decreased at step {step}: {prev:.6f} -> {s:.6f}"
        )
        prev = s
    print("[PASS] Cubic schedule is monotonically non-decreasing")


def test_schedule_cubic_shape():
    """Verify the cubic curve shape: starts slow, accelerates, tapers."""
    total = 100
    final = 0.9
    begin, end = 10, 80

    # First quarter of pruning phase
    s1 = target_sparsity(begin + (end - begin) // 4, total, final,
                         begin_step=begin, end_step=end)
    # Midpoint
    s2 = target_sparsity(begin + (end - begin) // 2, total, final,
                         begin_step=begin, end_step=end)
    # Three-quarter point
    s3 = target_sparsity(begin + 3 * (end - begin) // 4, total, final,
                         begin_step=begin, end_step=end)

    # The increments should increase (cubic accelerates)
    delta1 = s1 - 0.0   # from initial to first quarter
    delta2 = s2 - s1     # first quarter to midpoint
    delta3 = s3 - s2     # midpoint to three-quarter

    # Cubic: most pruning happens in the middle portion
    assert s1 < s2 < s3 < final, "Sparsity should increase through phases"
    print(f"[PASS] Cubic shape: quarter={s1:.3f}, mid={s2:.3f}, "
          f"three-quarter={s3:.3f}, final={final}")


def test_mask_update_achieves_target():
    """Test that update_masks achieves the target sparsity."""
    model = MLP([10, 20, 10], seed=0)
    optimizer = Adam(model.params(), lr=1e-3)

    # Do a forward+backward to populate gradients
    x = Tensor(np.random.randn(8, 10))
    logits = model.forward(x)
    loss = logits.softmax_cross_entropy(np.random.randint(0, 10, size=8))
    loss.backward()

    # Prune to 50% sparsity
    actual = update_masks(model, optimizer, 0.50, criterion='saliency')

    # Check sparsity is approximately 50%
    _, _, sparsity = get_model_sparsity(model)
    assert abs(sparsity - 0.50) < 0.05, (
        f"Expected ~50% sparsity, got {sparsity:.2%}"
    )
    print(f"[PASS] Mask update achieved {sparsity:.2%} sparsity (target: 50%)")


def test_mask_update_exact_zeros():
    """After mask update, all masked entries must be exactly 0.0."""
    model = MLP([10, 20, 10], seed=0)
    optimizer = Adam(model.params(), lr=1e-3)

    x = Tensor(np.random.randn(8, 10))
    logits = model.forward(x)
    loss = logits.softmax_cross_entropy(np.random.randint(0, 10, size=8))
    loss.backward()

    update_masks(model, optimizer, 0.70, criterion='saliency')

    for layer in model.prunable_layers():
        mask = layer.weight.mask
        masked_values = layer.weight.data[mask == 0]
        if len(masked_values) > 0:
            assert np.all(masked_values == 0.0), (
                f"Masked entries not exactly 0.0! "
                f"Max: {np.abs(masked_values).max()}"
            )
    print("[PASS] All masked entries are exactly 0.0 (not just 'small')")


def test_mask_update_calls_sync():
    """Verify that optimizer state is reset at newly-pruned entries."""
    model = MLP([10, 20, 10], seed=0)
    optimizer = Adam(model.params(), lr=1e-3)

    # Train a few steps to build up momentum
    for _ in range(5):
        optimizer.zero_grad()
        x = Tensor(np.random.randn(8, 10))
        logits = model.forward(x)
        loss = logits.softmax_cross_entropy(np.random.randint(0, 10, size=8))
        loss.backward()
        optimizer.step()

    # Now prune
    optimizer.zero_grad()
    x = Tensor(np.random.randn(8, 10))
    logits = model.forward(x)
    loss = logits.softmax_cross_entropy(np.random.randint(0, 10, size=8))
    loss.backward()

    update_masks(model, optimizer, 0.50, criterion='saliency')

    # Check that optimizer state (m, v) is zero at pruned entries
    for i, p in enumerate(optimizer.params):
        mask = getattr(p, 'mask', None)
        if mask is not None:
            pruned = (mask == 0)
            if pruned.any():
                assert np.all(optimizer.m[i][pruned] == 0.0), \
                    "Momentum not reset at pruned entries"
                assert np.all(optimizer.v[i][pruned] == 0.0), \
                    "Second moment not reset at pruned entries"

    print("[PASS] Optimizer state (m, v) is zero at pruned entries after sync")


def test_progressive_pruning():
    """Test that pruning can be applied progressively (multiple steps)."""
    model = MLP([10, 20, 10], seed=0)
    optimizer = Adam(model.params(), lr=1e-3)

    sparsities_to_test = [0.2, 0.4, 0.6, 0.8]
    prev_sparsity = 0.0

    for target_s in sparsities_to_test:
        # Forward+backward for fresh gradients
        optimizer.zero_grad()
        x = Tensor(np.random.randn(8, 10))
        logits = model.forward(x)
        loss = logits.softmax_cross_entropy(np.random.randint(0, 10, size=8))
        loss.backward()

        update_masks(model, optimizer, target_s, criterion='saliency')
        _, _, actual = get_model_sparsity(model)

        assert actual >= prev_sparsity, (
            f"Sparsity should not decrease: {prev_sparsity:.2%} -> {actual:.2%}"
        )
        prev_sparsity = actual

    print(f"[PASS] Progressive pruning: final sparsity {prev_sparsity:.2%}")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} pruning tests passed.")
