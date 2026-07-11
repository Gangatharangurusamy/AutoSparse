"""
Dataset loading for the Self-Pruning Network challenge.

Uses sklearn.datasets.load_digits() for loading ONLY — sklearn is never
used for training, models, gradients, or optimizers.

The digits dataset: 1797 samples of 8×8 grayscale images (64 features),
10 classes (digits 0–9). No download required — it's bundled with sklearn.
This is a good choice because it's small enough to train quickly with our
Python-level autodiff engine but complex enough to demonstrate real pruning
behavior (10-class classification, meaningful feature interactions).
"""

import numpy as np
from sklearn.datasets import load_digits


def load_data(val_fraction=0.2, seed=42):
    """Load and split the digits dataset.

    Args:
        val_fraction: fraction of data to hold out for validation
        seed:         random seed for reproducible train/val split

    Returns:
        (X_train, y_train, X_val, y_val) — numpy arrays.
        X arrays are float64 with shape (N, 64), normalized to zero mean
        and unit variance (statistics computed on training split only, then
        applied to validation split, to avoid data leakage).
        y arrays are int64 with shape (N,), values in {0, ..., 9}.
    """
    digits = load_digits()
    X = digits.data.astype(np.float64)  # shape (1797, 64)
    y = digits.target.astype(np.int64)  # shape (1797,)

    # Deterministic shuffle + split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))
    n_val = int(len(X) * val_fraction)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Normalize: compute statistics on training split only (no data leakage)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    # Avoid division by zero for constant features (e.g. corner pixels that
    # are always 0 in 8x8 digit images)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, y_train, X_val, y_val


def iterate_batches(X, y, batch_size, shuffle=True, seed=None):
    """Yield (X_batch, y_batch) mini-batches.

    Args:
        X:          input features, shape (N, D)
        y:          labels, shape (N,)
        batch_size: number of samples per batch
        shuffle:    if True, shuffle indices before batching
        seed:       random seed for shuffle (for reproducibility across epochs,
                    pass a different seed each epoch, e.g. epoch_num)

    Yields:
        (X_batch, y_batch) numpy arrays. Last batch may be smaller than
        batch_size if N is not divisible by batch_size.
    """
    N = len(X)
    indices = np.arange(N)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    for start in range(0, N, batch_size):
        idx = indices[start:start + batch_size]
        yield X[idx], y[idx]
