"""
Part 3: Self-pruning during training.

Usage (from repo root):
    python scripts/run_part3.py
    python scripts/run_part3.py --regrowth     # enable regrowth (bonus)

Trains an MLP with gradual pruning using the cubic sparsity schedule.
Every N steps, computes importance scores, determines the target sparsity
from the schedule, and prunes the lowest-importance remaining connections.

Produces:
  - outputs/part3_pruning_curve.png  (accuracy + sparsity vs epoch)
  - outputs/part3_metrics.json       (all per-epoch numbers)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np

from engine.tensor import Tensor
from nn.layers import MLP
from optim.adam import Adam
from train.data import load_data, iterate_batches
from train.train import train_epoch, evaluate
from prune.importance import compute_importance
from prune.schedule import target_sparsity
from prune.mask import update_masks, get_model_sparsity, get_active_flops


def do_importance_accumulation(model, X, y, batch_size=64, seed=None):
    """Run a forward+backward pass to populate gradients for importance scoring.

    This is separate from the training step: we want gradients that reflect
    the current loss landscape, but we don't take an optimizer step here.
    We accumulate gradients over a few batches for a more stable importance
    estimate.

    Args:
        model:      MLP instance
        X, y:       training data
        batch_size: batch size for gradient accumulation
        seed:       random seed for batch selection

    Returns:
        None (gradients are accumulated in-place on model params)
    """
    # Zero all gradients first
    for p in model.params():
        p.zero_grad()

    # Accumulate gradients over up to 3 batches for a more stable estimate
    n_accum = 0
    for X_batch, y_batch in iterate_batches(X, y, batch_size, shuffle=True, seed=seed):
        x_tensor = Tensor(X_batch)
        logits = model.forward(x_tensor)
        loss = logits.softmax_cross_entropy(y_batch)
        loss.backward()
        n_accum += 1
        if n_accum >= 3:
            break

    # Average the accumulated gradients
    if n_accum > 1:
        for p in model.params():
            if p.grad is not None:
                p.grad /= n_accum


def regrowth_step(model, optimizer, regrowth_fraction=0.05, seed=None):
    """Allow a small fraction of pruned entries to come back (bonus feature).

    For each layer, find pruned entries (mask=0) whose recomputed importance
    (using current gradient) would be highest, and revive up to
    regrowth_fraction of currently-pruned entries.

    Respects trap #9: sync_masks() already ensures m=0, v=0 at pruned
    entries, so revival starts with clean optimizer state. The revived
    weight is re-initialized to a small random value (not the old value,
    which was zeroed on prune).

    Args:
        model:              MLP instance
        optimizer:          Adam instance
        regrowth_fraction:  fraction of pruned entries to potentially revive
        seed:               random seed for re-initialization
    """
    rng = np.random.RandomState(seed)
    masks_changed = False

    for layer in model.prunable_layers():
        mask = layer.weight.mask
        pruned_count = int((mask == 0).sum())
        if pruned_count == 0:
            continue

        n_revive = max(1, int(regrowth_fraction * pruned_count))

        # Compute importance at pruned entries using current gradient
        # For pruned entries, grad is 0 (chain rule via mask), so we can't
        # use saliency directly. Instead, use the gradient magnitude at
        # pruned entries as a proxy for "how much would this connection
        # contribute if it existed." But since grad at pruned entries is
        # exactly 0 (by the chain rule), we need a different signal.
        #
        # The standard approach for regrowth (from Rigging the Lottery,
        # Evci et al. 2020) is to use the gradient of the loss w.r.t. the
        # pruned weight as if the mask were temporarily 1 — i.e., the
        # "dense gradient." We approximate this by using the gradient
        # magnitude of nearby (active) weights in the same layer as a
        # noisy proxy, combined with randomness.
        #
        # For simplicity and correctness (no custom backward path), we use
        # random regrowth: randomly select which pruned entries to revive.
        # This is a valid baseline used in the literature.
        pruned_indices = np.argwhere(mask == 0)
        if len(pruned_indices) > n_revive:
            selected = rng.choice(len(pruned_indices), size=n_revive, replace=False)
            revive_indices = pruned_indices[selected]
        else:
            revive_indices = pruned_indices

        for idx in revive_indices:
            idx = tuple(idx)
            mask[idx] = 1.0
            # Re-initialize to small random value (He-scale for this layer)
            fan_in = layer.weight.data.shape[0]
            layer.weight.data[idx] = rng.randn() * np.sqrt(2.0 / fan_in) * 0.1

        masks_changed = True

    if masks_changed:
        optimizer.sync_masks()


def main():
    # ---- Configuration ----
    SEED = 42
    EPOCHS = 150
    BATCH_SIZE = 64
    LR = 1e-3
    FINAL_SPARSITY = 0.90
    PRUNE_EVERY = 5       # prune every N epochs
    PRUNE_BEGIN = 10       # start pruning at this epoch
    PRUNE_END = 120        # stop pruning at this epoch (fine-tune remaining 30)
    CRITERION = 'saliency'

    # Check for --regrowth flag
    enable_regrowth = '--regrowth' in sys.argv

    np.random.seed(SEED)

    # ---- Setup ----
    X_train, y_train, X_val, y_val = load_data(seed=SEED)
    model = MLP([64, 128, 64, 10], seed=SEED)
    optimizer = Adam(model.params(), lr=LR)

    total_params = sum(p.data.size for p in model.params())
    _, total_weights, _ = get_model_sparsity(model)

    print("=" * 70)
    print("PART 3: Self-Pruning During Training")
    print("=" * 70)
    print(f"Architecture: [64, 128, 64, 10]")
    print(f"Total params: {total_params} (prunable weights: {total_weights})")
    print(f"Target sparsity: {FINAL_SPARSITY * 100:.0f}%")
    print(f"Criterion: {CRITERION}")
    print(f"Schedule: cubic ramp, prune epochs {PRUNE_BEGIN}-{PRUNE_END}, "
          f"every {PRUNE_EVERY} epochs")
    print(f"Regrowth: {'enabled' if enable_regrowth else 'disabled'}")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LR}")
    print()

    # ---- Training loop ----
    metrics = {
        "config": {
            "architecture": [64, 128, 64, 10],
            "total_params": total_params,
            "total_prunable_weights": total_weights,
            "final_sparsity": FINAL_SPARSITY,
            "criterion": CRITERION,
            "prune_every": PRUNE_EVERY,
            "prune_begin": PRUNE_BEGIN,
            "prune_end": PRUNE_END,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "seed": SEED,
            "regrowth": enable_regrowth,
        },
        "epochs": []
    }

    for epoch in range(EPOCHS):
        # 1. Train one epoch
        train_loss = train_epoch(model, X_train, y_train, optimizer,
                                 batch_size=BATCH_SIZE,
                                 seed=SEED * 1000 + epoch)

        # 2. Pruning step (if scheduled)
        pruned_this_epoch = False
        current_sparsity = get_model_sparsity(model)[2]

        if (epoch >= PRUNE_BEGIN and epoch < PRUNE_END and
                (epoch - PRUNE_BEGIN) % PRUNE_EVERY == 0):

            # Compute target sparsity at this epoch
            target_s = target_sparsity(
                step=epoch,
                total_steps=EPOCHS,
                final_sparsity=FINAL_SPARSITY,
                initial_sparsity=0.0,
                begin_step=PRUNE_BEGIN,
                end_step=PRUNE_END
            )

            # Accumulate gradients for importance scoring
            do_importance_accumulation(model, X_train, y_train,
                                       batch_size=BATCH_SIZE,
                                       seed=SEED + epoch)

            # Update masks
            current_sparsity = update_masks(model, optimizer, target_s,
                                            criterion=CRITERION)
            pruned_this_epoch = True

            # Optional regrowth
            if enable_regrowth and current_sparsity > 0.1:
                regrowth_step(model, optimizer, regrowth_fraction=0.05,
                              seed=SEED + epoch + 10000)
                current_sparsity = get_model_sparsity(model)[2]

        # 3. Evaluate
        _, train_acc = evaluate(model, X_train, y_train)
        val_loss, val_acc = evaluate(model, X_val, y_val)
        sparse_flops, dense_flops, flops_ratio = get_active_flops(model)

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "sparsity": round(current_sparsity, 4),
            "flops_ratio": round(flops_ratio, 4),
            "pruned": pruned_this_epoch,
        }
        metrics["epochs"].append(epoch_data)

        if (epoch + 1) % 10 == 0 or epoch == 0 or pruned_this_epoch:
            marker = " [PRUNED]" if pruned_this_epoch else ""
            print(f"Epoch {epoch + 1:3d}/{EPOCHS}  "
                  f"loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  "
                  f"val_acc={val_acc:.4f}  "
                  f"sparsity={current_sparsity:.2%}  "
                  f"FLOPs={flops_ratio:.2%}{marker}")

    # ---- Final report ----
    final_sparsity = get_model_sparsity(model)[2]
    _, final_train_acc = evaluate(model, X_train, y_train)
    _, final_val_acc = evaluate(model, X_val, y_val)
    sparse_flops, dense_flops, flops_ratio = get_active_flops(model)

    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Final sparsity:   {final_sparsity:.2%}")
    print(f"Final train acc:  {final_train_acc:.4f}")
    print(f"Final val acc:    {final_val_acc:.4f}")
    print(f"FLOPs ratio:      {flops_ratio:.2%} of dense")
    print(f"Active weights:   {int(sparse_flops // 2)} / {int(dense_flops // 2)}")

    metrics["final"] = {
        "sparsity": round(final_sparsity, 4),
        "train_acc": round(final_train_acc, 4),
        "val_acc": round(final_val_acc, 4),
        "flops_ratio": round(flops_ratio, 4),
        "sparse_flops": sparse_flops,
        "dense_flops": dense_flops,
    }

    # ---- Save outputs ----
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/part3_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved: outputs/part3_metrics.json")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_list = [e["epoch"] for e in metrics["epochs"]]
        train_accs = [e["train_acc"] for e in metrics["epochs"]]
        val_accs = [e["val_acc"] for e in metrics["epochs"]]
        sparsities = [e["sparsity"] for e in metrics["epochs"]]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Accuracy on left axis
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.plot(epochs_list, train_accs, color="#2ecc71", linewidth=2,
                 label="Train Acc", alpha=0.8)
        ax1.plot(epochs_list, val_accs, color="#3498db", linewidth=2,
                 linestyle="--", label="Val Acc", alpha=0.8)
        ax1.set_ylim(0, 1.05)

        # Sparsity on right axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Sparsity", color="#e74c3c", fontsize=12)
        ax2.plot(epochs_list, sparsities, color="#e74c3c", linewidth=2,
                 linestyle=":", label="Sparsity", alpha=0.8)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelcolor="#e74c3c")

        # Pruning region shading
        ax1.axvspan(PRUNE_BEGIN, PRUNE_END, alpha=0.08, color="red",
                    label="Pruning phase")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right",
                   fontsize=10)

        title = "Part 3: Self-Pruning Training"
        if enable_regrowth:
            title += " (with regrowth)"
        plt.title(title, fontsize=14)
        fig.tight_layout()
        plt.savefig("outputs/part3_pruning_curve.png", dpi=150)
        plt.close()
        print("Saved: outputs/part3_pruning_curve.png")
    except ImportError:
        print("WARNING: matplotlib not installed, skipping plot generation.")

    print("\nPart 3 complete.")


if __name__ == "__main__":
    main()
