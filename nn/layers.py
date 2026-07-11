"""
Neural network layers built on top of engine/tensor.py.

Design decisions:
  - Linear uses He initialization (std = sqrt(2/fan_in)) for ReLU-activated
    layers. This compensates for ReLU killing ~half the activations: if we
    used std = sqrt(1/fan_in) (Xavier), the variance of activations would
    halve at each layer, leading to vanishing signals in deep networks.
    With std=1 (naive), variance GROWS as fan_in at each layer, causing
    exploding activations and NaN loss within a few epochs on any
    non-trivial network.
  - Biases are initialized to zero (standard practice; the He init on
    weights already handles the variance).
  - Weight and bias Tensor objects are given a .mask attribute (all-ones by
    default) so the pruning code (Part 3) can attach to them uniformly
    without special-casing "does this param have a mask?".
  - Forward: x @ W + b relies on the engine's broadcasting-correct __add__
    for the (batch, out) + (out,) broadcast. No raw numpy ops in forward.
"""

import numpy as np
from engine.tensor import Tensor


class Linear:
    """Fully connected layer: y = x @ W + b.

    Weight shape: [in_features, out_features]
    Bias shape:   [out_features]

    Args:
        in_features:  number of input features
        out_features: number of output features
        init_std:     if provided, override He init with this std (for
                      ablation / demonstration of exploding activations)
        seed:         random seed for weight initialization
    """

    def __init__(self, in_features, out_features, init_std=None, seed=None):
        rng = np.random.RandomState(seed)

        # He initialization: std = sqrt(2 / fan_in)
        # Derivation: For a ReLU layer, Var[y_j] = (fan_in / 2) * Var[w] * Var[x]
        # (the 1/2 comes from E[ReLU(z)^2] = Var[z]/2 for symmetric z).
        # Setting Var[w] = 2/fan_in gives Var[y] = Var[x], preserving signal
        # variance across layers.
        if init_std is not None:
            std = init_std
        else:
            std = np.sqrt(2.0 / in_features)

        w_data = rng.randn(in_features, out_features) * std
        b_data = np.zeros(out_features)

        self.weight = Tensor(w_data, requires_grad=True)
        self.bias = Tensor(b_data, requires_grad=True)

        # Default masks (all-ones): no pruning active.
        # Pruning code sets these to 0/1 arrays and calls optimizer.sync_masks().
        self.weight.mask = np.ones_like(self.weight.data)
        self.bias.mask = np.ones_like(self.bias.data)  # biases are never pruned, but uniform interface

    def forward(self, x):
        """Forward pass: x @ W + b.

        Masking (if active) is applied here so the computational graph
        includes w * mask, which forces dL/dw = 0 at masked entries via
        the chain rule (see engine/tensor.py docstring).

        Args:
            x: Tensor of shape (batch, in_features)
        Returns:
            Tensor of shape (batch, out_features)
        """
        # Apply weight mask in the graph so gradient is exactly 0 at pruned entries
        mask_t = Tensor(self.weight.mask)
        w_eff = self.weight * mask_t
        return x @ w_eff + self.bias

    def params(self):
        """Return list of trainable parameters (for the optimizer)."""
        return [self.weight, self.bias]


class MLP:
    """Multi-layer perceptron: stack of Linear layers with ReLU activations.

    The final layer outputs raw logits (no activation). Softmax + CE loss
    is applied externally via Tensor.softmax_cross_entropy().

    Args:
        layer_sizes: list of ints, e.g. [64, 128, 64, 10].
                     First entry is input dim, last is output (num classes).
        activation:  'relu' or 'tanh' (activation between hidden layers)
        init_std:    if provided, override He init for all layers
        seed:        base random seed (each layer gets seed+i for reproducibility)
    """

    def __init__(self, layer_sizes, activation='relu', init_std=None, seed=42):
        assert len(layer_sizes) >= 2, "need at least input and output sizes"
        self.layers = []
        self.activation = activation
        for i in range(len(layer_sizes) - 1):
            layer_seed = seed + i if seed is not None else None
            self.layers.append(
                Linear(layer_sizes[i], layer_sizes[i + 1],
                       init_std=init_std, seed=layer_seed)
            )

    def forward(self, x):
        """Forward pass through all layers.

        Applies activation after every hidden layer, but NOT after the
        final layer (which outputs raw logits for softmax_cross_entropy).

        Args:
            x: Tensor of shape (batch, input_dim)
        Returns:
            Tensor of shape (batch, num_classes) — raw logits
        """
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            # Apply activation to all layers EXCEPT the last one
            if i < len(self.layers) - 1:
                if self.activation == 'relu':
                    x = x.relu()
                elif self.activation == 'tanh':
                    x = x.tanh()
                else:
                    raise ValueError(f"Unknown activation: {self.activation}")
        return x

    def params(self):
        """Return flat list of all trainable parameters across all layers."""
        all_params = []
        for layer in self.layers:
            all_params.extend(layer.params())
        return all_params

    def prunable_layers(self):
        """Return list of Linear layers whose weights can be pruned.
        (All layers — biases are not pruned, but layers are returned
        so the pruning code can access layer.weight.)"""
        return self.layers
