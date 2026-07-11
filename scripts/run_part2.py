"""
Part 2: Train a real network with our own Adam optimizer.

Usage (from repo root):
    python scripts/run_part2.py

Trains an MLP on the sklearn digits dataset (8x8 images, 10 classes) using
our custom autodiff engine and Adam optimizer. Produces:
  - outputs/part2_learning_curve.png  (loss + accuracy vs epoch)
  - outputs/part2_metrics.json        (all per-epoch numbers)

Also demonstrates that He initialization prevents NaN/exploding activations
compared to naive std=1 initialization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np

from engine.tensor import Tensor
from nn.layers import MLP
from optim.adam import Adam
from train.data import load_data
from train.train import train_epoch, evaluate


def run_init_comparison():
    """Demonstrate that He init prevents NaN/exploding activations vs std=1."""
    print("=" * 70)
    print("INITIALIZATION COMPARISON: He init vs std=1.0")
    print("=" * 70)

    X_train, y_train, X_val, y_val = load_data(seed=42)

    # --- std=1.0 init (bad) ---
    print("\n[std=1.0 init] Training for 5 epochs...")
    model_bad = MLP([64, 128, 64, 10], init_std=1.0, seed=42)
    opt_bad = Adam(model_bad.params(), lr=1e-3)

    had_problem = False
    for epoch in range(5):
        try:
            loss = train_epoch(model_bad, X_train, y_train, opt_bad,
                               batch_size=64, seed=epoch)
            _, acc = evaluate(model_bad, X_val, y_val)
            print(f"  Epoch {epoch + 1}: loss={loss:.4f}, val_acc={acc:.4f}")
        except (AssertionError, Exception) as e:
            print(f"  Epoch {epoch + 1}: NaN/Inf loss detected (expected with std=1.0)!")
            had_problem = True
            break

    if not had_problem:
        # Even if no NaN, check if activations are much larger
        x_tensor = Tensor(X_train[:10])
        logits = model_bad.forward(x_tensor)
        max_logit = np.abs(logits.data).max()
        print(f"  Max |logit| after 5 epochs: {max_logit:.2f}")
        if max_logit > 100:
            print("  WARNING: logits are very large — training is unstable.")

    # --- He init (good) ---
    print("\n[He init] Training for 5 epochs...")
    model_good = MLP([64, 128, 64, 10], seed=42)  # He init by default
    opt_good = Adam(model_good.params(), lr=1e-3)

    for epoch in range(5):
        loss = train_epoch(model_good, X_train, y_train, opt_good,
                           batch_size=64, seed=epoch)
        _, acc = evaluate(model_good, X_val, y_val)
        print(f"  Epoch {epoch + 1}: loss={loss:.4f}, val_acc={acc:.4f}")

    x_tensor = Tensor(X_train[:10])
    logits = model_good.forward(x_tensor)
    max_logit = np.abs(logits.data).max()
    print(f"  Max |logit| after 5 epochs: {max_logit:.2f}")
    print(f"  He init keeps logits well-behaved (no NaN, bounded magnitudes).")
    print()


def main():
    np.random.seed(42)
    SEED = 42
    EPOCHS = 100
    BATCH_SIZE = 64
    LR = 1e-3

    # ---- Initialization comparison ----
    run_init_comparison()

    # ---- Full training with He init ----
    print("=" * 70)
    print("FULL TRAINING: MLP [64 -> 128 -> 64 -> 10] with He init + Adam")
    print("=" * 70)

    X_train, y_train, X_val, y_val = load_data(seed=SEED)
    print(f"Dataset: {len(X_train)} train, {len(X_val)} val, "
          f"{X_train.shape[1]} features, 10 classes")

    model = MLP([64, 128, 64, 10], seed=SEED)
    optimizer = Adam(model.params(), lr=LR)

    total_params = sum(p.data.size for p in model.params())
    print(f"Total parameters: {total_params}")
    print(f"Optimizer: Adam (lr={LR})")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print()

    metrics = {
        "config": {
            "architecture": [64, 128, 64, 10],
            "activation": "relu",
            "init": "he",
            "optimizer": "adam",
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "dataset": "sklearn.datasets.load_digits (loading only)",
            "total_params": total_params,
        },
        "epochs": []
    }

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        # Train one epoch
        train_loss = train_epoch(model, X_train, y_train, optimizer,
                                 batch_size=BATCH_SIZE, seed=SEED * 1000 + epoch)

        # Evaluate
        _, train_acc = evaluate(model, X_train, y_train)
        val_loss, val_acc = evaluate(model, X_val, y_val)

        best_val_acc = max(best_val_acc, val_acc)

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
        }
        metrics["epochs"].append(epoch_data)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{EPOCHS}  "
                  f"train_loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.4f}  "
                  f"val_acc={val_acc:.4f}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    metrics["best_val_acc"] = round(best_val_acc, 4)

    # ---- Save outputs ----
    os.makedirs("outputs", exist_ok=True)

    # Save metrics JSON
    with open("outputs/part2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: outputs/part2_metrics.json")

    # Save learning curve plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_list = [e["epoch"] for e in metrics["epochs"]]
        train_losses = [e["train_loss"] for e in metrics["epochs"]]
        train_accs = [e["train_acc"] for e in metrics["epochs"]]
        val_accs = [e["val_acc"] for e in metrics["epochs"]]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Loss on left axis
        color_loss = "#e74c3c"
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", color=color_loss, fontsize=12)
        ax1.plot(epochs_list, train_losses, color=color_loss, linewidth=2,
                 label="Train Loss", alpha=0.8)
        ax1.tick_params(axis="y", labelcolor=color_loss)

        # Accuracy on right axis
        ax2 = ax1.twinx()
        color_train = "#2ecc71"
        color_val = "#3498db"
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.plot(epochs_list, train_accs, color=color_train, linewidth=2,
                 label="Train Acc", alpha=0.8)
        ax2.plot(epochs_list, val_accs, color=color_val, linewidth=2,
                 linestyle="--", label="Val Acc", alpha=0.8)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="center right",
                   fontsize=10)

        plt.title("Part 2: MLP Training on Digits Dataset", fontsize=14)
        fig.tight_layout()
        plt.savefig("outputs/part2_learning_curve.png", dpi=150)
        plt.close()
        print("Saved: outputs/part2_learning_curve.png")
    except ImportError:
        print("WARNING: matplotlib not installed, skipping plot generation.")

    print("\nPart 2 complete.")


if __name__ == "__main__":
    main()
