# AutoSparse — Self-Pruning Neural Network

A from-scratch implementation of a self-pruning neural network with reverse-mode autodiff, custom Adam optimizer with masked-weight support, and gradual magnitude/saliency-based pruning.

**Pure Python + NumPy only.** No PyTorch, TensorFlow, JAX, or any external autodiff/NN library. `scikit-learn` is used **exclusively** for loading the digits dataset (`sklearn.datasets.load_digits()`) — never for training, models, gradients, or optimizers.

## Quick Start

### Install

```bash
pip install numpy scipy matplotlib scikit-learn
```

Or with uv:
```bash
uv sync
```

### Run Tests

```bash
# Gradient checks (13 tests) — verifies autodiff correctness
python tests/test_gradients.py

# Masked weight tests (3 tests) — verifies Adam handles pruned weights correctly
python tests/test_masked_weights.py

# Layer tests — verifies Linear/MLP layers, He init, gradient correctness
python tests/test_layers.py

# Pruning tests — verifies importance scoring, cubic schedule, mask updates
python tests/test_pruning.py

# Or run all tests at once:
python -m pytest tests/ -v
```

### Run Training & Evaluation

```bash
# Part 2: Train an MLP on the digits dataset (produces learning curve)
python scripts/run_part2.py

# Part 3: Train with self-pruning to 90% sparsity
python scripts/run_part3.py

# Part 3 with regrowth (bonus):
python scripts/run_part3.py --regrowth

# Part 4: Pareto evaluation across sparsities, seeds, and criteria
python scripts/run_part4_pareto.py
```

## Project Structure

```
engine/tensor.py           — Reverse-mode autodiff engine (Tensor class)
optim/adam.py              — Adam optimizer with masked-weight support

nn/layers.py               — Linear layer + MLP (He init, mask-aware forward)
train/data.py              — Dataset loading (sklearn digits, loading only)
train/train.py             — Training loop (mini-batch SGD with Adam)

prune/importance.py        — Importance scoring (|w*g| saliency, |w| magnitude)
prune/schedule.py          — Cubic sparsity schedule (Zhu & Gupta 2017)
prune/mask.py              — Mask management (global pruning, sync_masks)

scripts/run_part2.py       — End-to-end training script
scripts/run_part3.py       — Self-pruning training script
scripts/run_part4_pareto.py — Pareto evaluation (accuracy vs sparsity)

tests/test_gradients.py    — Gradient checks (centered finite differences)
tests/test_masked_weights.py — Masked weight + Adam correctness tests
tests/test_layers.py       — Layer gradient checks and shape tests
tests/test_pruning.py      — Pruning module tests

outputs/                   — Generated plots and metrics
DESIGN.md                  — Detailed design document with derivations
```

## Outputs

After running all scripts, the `outputs/` directory will contain:

| File | Description |
|------|-------------|
| `part2_learning_curve.png` | Loss + accuracy vs epoch (unpruned baseline) |
| `part2_metrics.json` | Per-epoch training metrics |
| `part3_pruning_curve.png` | Accuracy + sparsity vs epoch during pruning |
| `part3_metrics.json` | Per-epoch metrics with pruning events |
| `part4_pareto_curve.png` | Accuracy vs sparsity (saliency vs magnitude, with error bars) |
| `part4_pareto_raw.csv` | Raw results: seed × sparsity × criterion |

## Key Design Decisions

See [DESIGN.md](DESIGN.md) for full derivations and justifications:

1. **`|w·g|` importance criterion**: First-order Taylor estimate of loss increase from removing a connection
2. **Masked weight gradients**: The chain rule forces `dL/dw = 0` at masked entries, but Adam's stale momentum requires `sync_masks()` to reset state
3. **Gradual cubic pruning**: Better than one-shot because the network can adapt after each pruning event
4. **Global pruning with per-layer minimum**: Allocates capacity non-uniformly across layers

## Falsifiable Claim

> At 90% sparsity, saliency pruning (|w*g|) retains 97.9% accuracy (± 0.3% across N=3 seeds) versus 97.7% for magnitude pruning (|w|).

## Cost Measurement

We report three honest cost metrics:
- **Active parameter count**: `mask.sum()` across layers (fraction of dense baseline)
- **FLOPs estimate**: `2 × active_params` per layer (multiply-accumulate ops)
- **Wall-clock with genuine sparse matmul**: `scipy.sparse` CSR-based forward pass (clearly labeled, not dense-times-zero)

See DESIGN.md §4 for why dense `W * mask` matmul is NOT a real speedup measurement.
