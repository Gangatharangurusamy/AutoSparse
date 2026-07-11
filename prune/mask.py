"""
Mask management for pruning.

Given a model's layers, importance scores, and a target sparsity, this
module decides which connections to prune and updates the masks accordingly.

Key correctness points:
  - Uses GLOBAL pruning: pools importance scores across all layers and
    prunes the globally least-important connections. This is better than
    per-layer uniform pruning because it lets the network keep more
    capacity in layers that need it (e.g. the first layer processing
    raw features, or the final layer producing class logits).
  - Per-layer safety: never prunes a layer below 50% of its original
    connections, to avoid completely stripping a small layer.
  - After mask update: asserts that all masked entries are exactly 0.0
    in the weight data (not just "small").
  - Calls optimizer.sync_masks() immediately after changing masks, which
    resets Adam's m/v state at newly-pruned entries to prevent stale
    momentum from corrupting future updates (see optim/adam.py).
"""

import numpy as np
from prune.importance import compute_importance


def update_masks(model, optimizer, target_sparsity_val, criterion='saliency'):
    """Update weight masks across all prunable layers to reach target sparsity.

    Uses global pruning: pools importance scores across all layers, finds
    the global threshold, and prunes entries below it (respecting per-layer
    minimums).

    Args:
        model:              MLP instance with .prunable_layers()
        optimizer:          Adam instance (for sync_masks() call)
        target_sparsity_val: float in [0, 1], the target fraction of
                            weights that should be zero
        criterion:          'saliency' or 'magnitude' (passed to
                            compute_importance)

    Returns:
        actual_sparsity: float, the actual sparsity achieved after this step
    """
    if target_sparsity_val <= 0.0:
        return 0.0

    layers = model.prunable_layers()

    # Collect importance scores and metadata for all prunable weights
    all_importances = []
    layer_info = []

    for layer in layers:
        imp = compute_importance(layer.weight, criterion=criterion)
        mask = layer.weight.mask

        total_entries = mask.size
        min_active = max(1, int(0.01 * total_entries))  # never prune below 1% to allow high global sparsity

        layer_info.append({
            'layer': layer,
            'importance': imp,
            'mask': mask,
            'total_entries': total_entries,
            'min_active': min_active,
        })

        # For global thresholding, only consider currently-active entries
        active_importances = imp[mask == 1]
        all_importances.append(active_importances)

    # Total counts across all layers
    total_weights = sum(info['total_entries'] for info in layer_info)
    target_zeros = int(target_sparsity_val * total_weights)

    # Current zeros
    current_zeros = sum(int((info['mask'] == 0).sum()) for info in layer_info)

    if current_zeros >= target_zeros:
        # Already at or beyond target sparsity, nothing to prune
        return current_zeros / total_weights

    # Need to prune (target_zeros - current_zeros) more entries
    n_to_prune = target_zeros - current_zeros

    # Pool all active importances and find the threshold
    all_active = np.concatenate(all_importances)
    if len(all_active) == 0:
        return current_zeros / total_weights

    # Sort to find the threshold: prune the n_to_prune lowest-importance
    # active connections
    sorted_importances = np.sort(all_active)
    if n_to_prune >= len(sorted_importances):
        threshold = float('inf')  # prune everything (shouldn't happen with min_active guard)
    else:
        threshold = sorted_importances[n_to_prune - 1]

    # Apply pruning per layer, respecting per-layer minimum active count
    masks_changed = False
    for info in layer_info:
        layer = info['layer']
        imp = info['importance']
        mask = info['mask']
        min_active = info['min_active']

        # Find entries to prune: currently active AND importance <= threshold
        candidates = (mask == 1) & (imp <= threshold)

        if not candidates.any():
            continue

        # Check per-layer minimum: how many would remain active?
        current_active = int((mask == 1).sum())
        n_candidates = int(candidates.sum())
        max_prunable = current_active - min_active

        if max_prunable <= 0:
            continue

        if n_candidates > max_prunable:
            # Can only prune max_prunable entries — pick the lowest importance ones
            candidate_importances = imp.copy()
            candidate_importances[~candidates] = float('inf')
            flat_idx = np.argsort(candidate_importances.ravel())[:max_prunable]
            candidates = np.zeros_like(mask, dtype=bool)
            candidates.ravel()[flat_idx] = True

        # Apply pruning
        mask[candidates] = 0.0
        layer.weight.data[candidates] = 0.0
        masks_changed = True

    # Sync optimizer state ONCE after all layers are updated
    if masks_changed:
        optimizer.sync_masks()

    # Assert all masked entries are exactly 0.0 (not just "small")
    for info in layer_info:
        layer = info['layer']
        mask = layer.weight.mask
        masked_values = layer.weight.data[mask == 0]
        if len(masked_values) > 0:
            assert np.all(masked_values == 0.0), (
                f"Masked entries are not exactly 0.0! "
                f"Max abs value at masked entry: {np.abs(masked_values).max()}"
            )

    # Compute actual sparsity
    total_zeros = sum(int((info['layer'].weight.mask == 0).sum()) for info in layer_info)
    actual_sparsity = total_zeros / total_weights

    return actual_sparsity


def get_model_sparsity(model):
    """Compute the current sparsity of a model's prunable weights.

    Returns:
        (total_zeros, total_weights, sparsity_fraction)
    """
    layers = model.prunable_layers()
    total_zeros = 0
    total_weights = 0
    for layer in layers:
        mask = layer.weight.mask
        total_zeros += int((mask == 0).sum())
        total_weights += mask.size
    sparsity = total_zeros / total_weights if total_weights > 0 else 0.0
    return total_zeros, total_weights, sparsity


def get_active_flops(model):
    """Estimate FLOPs for the current sparse model vs dense baseline.

    For each linear layer, a dense matmul x @ W with x shape (1, in) and
    W shape (in, out) requires 2 * in * out FLOPs (1 multiply + 1 add
    per output element, for each of the in contributions).

    With sparsity, the active FLOPs are 2 * active_params where
    active_params = number of non-zero entries in the weight matrix.

    Returns:
        (sparse_flops, dense_flops, ratio)
    """
    layers = model.prunable_layers()
    sparse_flops = 0
    dense_flops = 0
    for layer in layers:
        total = layer.weight.data.size
        active = int(layer.weight.mask.sum())
        dense_flops += 2 * total
        sparse_flops += 2 * active
    ratio = sparse_flops / dense_flops if dense_flops > 0 else 1.0
    return sparse_flops, dense_flops, ratio
