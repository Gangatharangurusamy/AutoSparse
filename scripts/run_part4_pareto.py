"""
Part 4: Prove it — Pareto evaluation across sparsities, seeds, and criteria.

Usage (from repo root):
    python scripts/run_part4_pareto.py

For target sparsities [0, 50, 75, 90, 95] percent, runs the Part 3
training+pruning pipeline with both saliency (|w*g|) and magnitude (|w|)
criteria, across 3 seeds. Records final accuracy and FLOPs estimates.

Produces:
  - outputs/part4_pareto_raw.csv    (one row per seed × sparsity × criterion)
  - outputs/part4_pareto_curve.png  (accuracy vs sparsity, error bars)

Also implements a genuine sparse forward pass using scipy.sparse for
wall-clock timing comparison (clearly labeled, not conflated with the
dense path).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
import time
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
    """Run forward+backward passes to populate gradients for importance scoring."""
    for p in model.params():
        p.zero_grad()
    n_accum = 0
    for X_batch, y_batch in iterate_batches(X, y, batch_size, shuffle=True, seed=seed):
        x_tensor = Tensor(X_batch)
        logits = model.forward(x_tensor)
        loss = logits.softmax_cross_entropy(y_batch)
        loss.backward()
        n_accum += 1
        if n_accum >= 3:
            break
    if n_accum > 1:
        for p in model.params():
            if p.grad is not None:
                p.grad /= n_accum


def train_with_pruning(X_train, y_train, X_val, y_val, target_sparsity_pct,
                       criterion='saliency', seed=42, epochs=150,
                       batch_size=64, lr=1e-3):
    """Train an MLP with gradual pruning to a target sparsity.

    Args:
        X_train, y_train, X_val, y_val: dataset splits
        target_sparsity_pct: target sparsity as a percentage (0-100)
        criterion: 'saliency' or 'magnitude'
        seed: random seed
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate

    Returns:
        dict with final metrics
    """
    np.random.seed(seed)
    final_sparsity_frac = target_sparsity_pct / 100.0

    # Config for pruning schedule
    prune_every = 5
    prune_begin = 10 if final_sparsity_frac > 0 else epochs + 1
    prune_end = int(0.8 * epochs) if final_sparsity_frac > 0 else epochs + 1

    model = MLP([64, 128, 64, 10], seed=seed)
    optimizer = Adam(model.params(), lr=lr)

    for epoch in range(epochs):
        # Train one epoch
        train_loss = train_epoch(model, X_train, y_train, optimizer,
                                 batch_size=batch_size,
                                 seed=seed * 1000 + epoch)

        # Pruning step (if scheduled and target > 0)
        if (final_sparsity_frac > 0 and
                epoch >= prune_begin and epoch < prune_end and
                (epoch - prune_begin) % prune_every == 0):

            target_s = target_sparsity(
                step=epoch,
                total_steps=epochs,
                final_sparsity=final_sparsity_frac,
                initial_sparsity=0.0,
                begin_step=prune_begin,
                end_step=prune_end
            )

            do_importance_accumulation(model, X_train, y_train,
                                       batch_size=batch_size,
                                       seed=seed + epoch)

            update_masks(model, optimizer, target_s, criterion=criterion)

    # Final evaluation
    _, train_acc = evaluate(model, X_train, y_train)
    _, val_acc = evaluate(model, X_val, y_val)
    _, total_weights, actual_sparsity = get_model_sparsity(model)
    sparse_flops, dense_flops, flops_ratio = get_active_flops(model)

    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'actual_sparsity': actual_sparsity,
        'flops_ratio': flops_ratio,
        'sparse_flops': sparse_flops,
        'dense_flops': dense_flops,
        'model': model,  # keep for timing measurements
    }


def measure_forward_time(model, X, n_repeats=50):
    """Measure wall-clock time for forward passes.

    Returns:
        (dense_time_ms, sparse_time_ms) — average per-forward-pass time.

    dense_time_ms: standard dense forward using our engine (w * mask matmul)
    sparse_time_ms: genuine sparse forward using scipy.sparse CSR matrices
    """
    # Dense forward time (our standard path)
    x_tensor = Tensor(X)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = model.forward(x_tensor)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    dense_time_ms = np.median(times)

    # Sparse forward time using scipy.sparse
    try:
        from scipy.sparse import csr_matrix
        sparse_time_ms = _measure_sparse_forward(model, X, n_repeats)
    except ImportError:
        sparse_time_ms = -1.0  # scipy not available

    return dense_time_ms, sparse_time_ms


def _measure_sparse_forward(model, X, n_repeats=50):
    """Genuine sparse matmul forward pass using scipy.sparse CSR.

    This is NOT used for training (no autodiff support). It's purely for
    measuring whether sparsity actually translates to wall-clock savings
    in a genuinely sparse representation, as opposed to the fake speedup
    of dense-times-zero.
    """
    from scipy.sparse import csr_matrix

    layers = model.prunable_layers()
    activations = ['relu'] * (len(layers) - 1) + [None]

    # Convert weights to CSR format
    sparse_weights = []
    biases = []
    for layer in layers:
        w_masked = layer.weight.data * layer.weight.mask
        sparse_weights.append(csr_matrix(w_masked))
        biases.append(layer.bias.data)

    # Time the sparse forward pass
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        x = X.copy()
        for i, (sw, b, act) in enumerate(zip(sparse_weights, biases, activations)):
            # scipy sparse @ dense — genuinely skips zero entries
            x = x @ sw + b
            # x is a numpy matrix after sparse matmul, convert back
            x = np.asarray(x)
            if act == 'relu':
                x = np.maximum(x, 0)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return np.median(times)


def main():
    # ---- Configuration ----
    SPARSITIES = [0, 50, 75, 90, 95]  # percent
    SEEDS = [42, 123, 777]
    CRITERIA = ['saliency', 'magnitude']
    EPOCHS = 150
    BATCH_SIZE = 64
    LR = 1e-3

    print("=" * 70)
    print("PART 4: Pareto Evaluation")
    print("=" * 70)
    print(f"Sparsities: {SPARSITIES}%")
    print(f"Seeds: {SEEDS}")
    print(f"Criteria: {CRITERIA}")
    print(f"Total runs: {len(SPARSITIES) * len(SEEDS) * len(CRITERIA)}")
    print()

    # Load data once (same split for all runs)
    X_train, y_train, X_val, y_val = load_data(seed=42)

    # ---- Run all combinations ----
    results = []
    total_runs = len(SPARSITIES) * len(SEEDS) * len(CRITERIA)
    run_idx = 0

    for sparsity_pct in SPARSITIES:
        for criterion in CRITERIA:
            for seed in SEEDS:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] sparsity={sparsity_pct}%, "
                      f"criterion={criterion}, seed={seed} ... ", end="", flush=True)

                result = train_with_pruning(
                    X_train, y_train, X_val, y_val,
                    target_sparsity_pct=sparsity_pct,
                    criterion=criterion,
                    seed=seed,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    lr=LR
                )

                # Measure forward time
                dense_time_ms, sparse_time_ms = measure_forward_time(
                    result['model'], X_val[:50], n_repeats=30
                )

                row = {
                    'seed': seed,
                    'target_sparsity_pct': sparsity_pct,
                    'criterion': criterion,
                    'actual_sparsity': round(result['actual_sparsity'], 4),
                    'train_accuracy': round(result['train_acc'], 4),
                    'val_accuracy': round(result['val_acc'], 4),
                    'flops_ratio': round(result['flops_ratio'], 4),
                    'dense_time_ms': round(dense_time_ms, 3),
                    'sparse_time_ms': round(sparse_time_ms, 3),
                }
                results.append(row)
                print(f"val_acc={result['val_acc']:.4f}, "
                      f"actual_sparsity={result['actual_sparsity']:.2%}")

    # ---- Save raw results ----
    os.makedirs("outputs", exist_ok=True)

    # CSV
    csv_path = "outputs/part4_pareto_raw.csv"
    fieldnames = ['seed', 'target_sparsity_pct', 'criterion', 'actual_sparsity',
                  'train_accuracy', 'val_accuracy', 'flops_ratio',
                  'dense_time_ms', 'sparse_time_ms']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")

    # ---- Analysis and comparison ----
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n{'Sparsity':>10s} | {'Criterion':>10s} | {'Val Acc (mean±std)':>20s} | "
          f"{'FLOPs ratio':>12s}")
    print("-" * 65)

    summary = {}
    for sparsity_pct in SPARSITIES:
        for criterion in CRITERIA:
            accs = [r['val_accuracy'] for r in results
                    if r['target_sparsity_pct'] == sparsity_pct
                    and r['criterion'] == criterion]
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            flops = np.mean([r['flops_ratio'] for r in results
                            if r['target_sparsity_pct'] == sparsity_pct
                            and r['criterion'] == criterion])

            key = (sparsity_pct, criterion)
            summary[key] = {'mean': mean_acc, 'std': std_acc, 'n': len(accs)}

            print(f"{sparsity_pct:>9d}% | {criterion:>10s} | "
                  f"{mean_acc:.4f} ± {std_acc:.4f}       | "
                  f"{flops:.4f}")

    # ---- Comparison at each sparsity level ----
    print(f"\n{'=' * 70}")
    print("CRITERION COMPARISON (saliency vs magnitude)")
    print(f"{'=' * 70}")

    for sparsity_pct in SPARSITIES:
        if sparsity_pct == 0:
            continue  # no pruning, criteria are irrelevant
        sal = summary[(sparsity_pct, 'saliency')]
        mag = summary[(sparsity_pct, 'magnitude')]
        diff = sal['mean'] - mag['mean']
        # Simple comparison: is the difference larger than the pooled std?
        pooled_std = np.sqrt((sal['std']**2 + mag['std']**2) / 2)

        # Two-sample t-test (Welch's)
        if sal['std'] > 0 or mag['std'] > 0:
            se = np.sqrt(sal['std']**2/sal['n'] + mag['std']**2/mag['n'])
            if se > 0:
                t_stat = diff / se
                # With n1+n2-2 dof (small sample), |t| > 2.78 is significant at p<0.05
                significant = abs(t_stat) > 2.78
            else:
                t_stat = float('inf') if diff != 0 else 0
                significant = diff != 0
        else:
            t_stat = float('inf') if diff != 0 else 0
            significant = diff != 0

        sig_str = "YES (p<0.05)" if significant else "NO (within noise)"
        print(f"\nAt {sparsity_pct}% sparsity:")
        print(f"  Saliency:  {sal['mean']:.4f} ± {sal['std']:.4f}")
        print(f"  Magnitude: {mag['mean']:.4f} ± {mag['std']:.4f}")
        print(f"  Difference: {diff:+.4f} (t={t_stat:.2f})")
        print(f"  Significant: {sig_str}")

    # ---- Falsifiable claim ----
    sal_90 = summary.get((90, 'saliency'), {'mean': 0, 'std': 0, 'n': 0})
    mag_90 = summary.get((90, 'magnitude'), {'mean': 0, 'std': 0, 'n': 0})
    print(f"\n{'=' * 70}")
    print("FALSIFIABLE CLAIM")
    print(f"{'=' * 70}")
    claim = (f"At 90% sparsity, saliency pruning (|w*g|) retains "
             f"{sal_90['mean']*100:.1f}% accuracy "
             f"(± {sal_90['std']*100:.1f}% across N={sal_90['n']} seeds) "
             f"versus {mag_90['mean']*100:.1f}% for magnitude pruning (|w|).")
    print(claim)

    # Save claim to a file for DESIGN.md reference
    with open("outputs/part4_claim.txt", "w") as f:
        f.write(claim + "\n")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        for criterion, color, marker in [('saliency', '#3498db', 'o'),
                                          ('magnitude', '#e74c3c', 's')]:
            means = []
            stds = []
            sparsities = []
            for sp in SPARSITIES:
                accs = [r['val_accuracy'] for r in results
                        if r['target_sparsity_pct'] == sp
                        and r['criterion'] == criterion]
                means.append(np.mean(accs))
                stds.append(np.std(accs))
                sparsities.append(sp)

            label_name = 'Saliency (|w·g|)' if criterion == 'saliency' else 'Magnitude (|w|)'
            ax.errorbar(sparsities, means, yerr=stds, color=color,
                       marker=marker, markersize=8, linewidth=2, capsize=5,
                       label=label_name, alpha=0.8)

            # Shaded region for ±1 std
            ax.fill_between(sparsities,
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           color=color, alpha=0.15)

        ax.set_xlabel("Target Sparsity (%)", fontsize=12)
        ax.set_ylabel("Validation Accuracy", fontsize=12)
        ax.set_title("Part 4: Accuracy vs Sparsity (Pareto Curve)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(SPARSITIES)

        fig.tight_layout()
        plt.savefig("outputs/part4_pareto_curve.png", dpi=150)
        plt.close()
        print(f"\nSaved: outputs/part4_pareto_curve.png")
    except ImportError:
        print("WARNING: matplotlib not installed, skipping plot generation.")

    print("\nPart 4 complete.")


if __name__ == "__main__":
    main()
