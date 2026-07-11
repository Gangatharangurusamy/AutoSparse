"""
Training loop for the Self-Pruning Network.

All training uses engine/tensor.py for forward/backward and optim/adam.py
for parameter updates. No external autodiff or optimizer libraries.

Key correctness points:
  - Uses Tensor.softmax_cross_entropy() for the loss (numerically stable,
    subtracts row-max before exponentiating).
  - Uses the stashed _probs from softmax_cross_entropy for accuracy
    computation, reusing the numerically stable softmax path rather than
    reimplementing exp(x)/sum(exp(x)).
  - Asserts finite loss every batch to catch NaN/Inf early.
  - Gradient accumulation is handled correctly by the engine (+=, not =).
  - Broadcasting gradients for bias (batch, out) + (out,) are handled
    correctly by the engine's __add__ and _unbroadcast.
"""

import numpy as np
from engine.tensor import Tensor
from train.data import iterate_batches


def train_epoch(model, X, y, optimizer, batch_size=64, seed=None):
    """Train for one epoch over the dataset.

    Args:
        model:      MLP instance
        X:          training features, shape (N, D)
        y:          training labels, shape (N,)
        optimizer:  Adam instance (from optim/adam.py)
        batch_size: mini-batch size
        seed:       random seed for batch shuffling

    Returns:
        epoch_loss: float, mean loss over all batches in this epoch
    """
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in iterate_batches(X, y, batch_size, shuffle=True, seed=seed):
        # 1. Zero gradients
        optimizer.zero_grad()

        # 2. Forward pass
        x_tensor = Tensor(X_batch)
        logits = model.forward(x_tensor)

        # 3. Loss (numerically stable softmax + cross-entropy)
        loss = logits.softmax_cross_entropy(y_batch)

        # 4. Assert finite loss (catch NaN/Inf early)
        assert np.isfinite(loss.data), (
            f"Non-finite loss detected: {loss.data}. "
            "Check initialization (He init?), learning rate, or data normalization."
        )

        # 5. Backward pass
        loss.backward()

        # 6. Optimizer step
        optimizer.step()

        total_loss += float(loss.data)
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, X, y, batch_size=256):
    """Compute loss and accuracy on a dataset.

    Uses softmax_cross_entropy's stashed _probs for accuracy computation,
    reusing the numerically stable softmax path.

    Args:
        model:      MLP instance
        X:          features, shape (N, D)
        y:          labels, shape (N,)
        batch_size: batch size for evaluation (larger is fine, no gradients)

    Returns:
        (loss, accuracy): floats
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in iterate_batches(X, y, batch_size, shuffle=False):
        x_tensor = Tensor(X_batch)
        logits = model.forward(x_tensor)
        loss = logits.softmax_cross_entropy(y_batch)

        # Use the stashed _probs from softmax_cross_entropy (numerically
        # stable path) for accuracy, not a separate exp(x)/sum(exp(x)).
        probs = loss._probs  # shape (batch, num_classes)
        preds = np.argmax(probs, axis=1)
        total_correct += (preds == y_batch).sum()
        total_samples += len(y_batch)
        total_loss += float(loss.data) * len(y_batch)

    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    return avg_loss, accuracy
