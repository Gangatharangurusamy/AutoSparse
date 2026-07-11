"""
Tests for nn/layers.py — Linear layer and MLP.

Gradient checks use the same centered-difference framework as
tests/test_gradients.py. Also tests shape correctness and He vs bad init.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from engine.tensor import Tensor
from nn.layers import Linear, MLP

np.random.seed(0)


def numerical_grad(f, x, h=1e-5):
    """Centered finite-difference gradient."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + h
        plus = float(np.asarray(f(x.copy()).data).reshape(-1)[0])
        x[idx] = orig - h
        minus = float(np.asarray(f(x.copy()).data).reshape(-1)[0])
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * h)
        it.iternext()
    return grad


def test_linear_forward_shape():
    """Test that Linear forward produces correct output shape."""
    layer = Linear(10, 5, seed=0)
    x = Tensor(np.random.randn(8, 10))
    out = layer.forward(x)
    assert out.shape == (8, 5), f"Expected (8, 5), got {out.shape}"
    print("[PASS] Linear forward shape: (8, 10) @ (10, 5) -> (8, 5)")


def test_linear_gradient_wrt_input():
    """Gradient check: dL/dx through a Linear layer."""
    layer = Linear(4, 3, seed=0)

    def f(x_np):
        x = Tensor(x_np, requires_grad=True)
        return layer.forward(x).sum()

    x_np = np.random.randn(5, 4)
    num = numerical_grad(f, x_np.copy())

    x_t = Tensor(x_np.copy(), requires_grad=True)
    out = layer.forward(x_t)
    out.sum().backward()
    ana = x_t.grad

    max_err = np.abs(num - ana).max()
    assert max_err < 1e-4, f"Linear input grad check failed: {max_err:.3e}"
    print(f"[PASS] Linear gradient wrt input     max_err={max_err:.3e}")


def test_linear_gradient_wrt_weight():
    """Gradient check: dL/dW through a Linear layer."""
    layer = Linear(4, 3, seed=0)
    x_data = np.random.randn(5, 4)

    def f(w_np):
        layer.weight = Tensor(w_np, requires_grad=True)
        layer.weight.mask = np.ones_like(w_np)
        x = Tensor(x_data)
        return layer.forward(x).sum()

    w_np = layer.weight.data.copy()
    num = numerical_grad(f, w_np.copy())

    # Analytical
    layer.weight = Tensor(w_np.copy(), requires_grad=True)
    layer.weight.mask = np.ones_like(w_np)
    x = Tensor(x_data)
    out = layer.forward(x)
    out.sum().backward()
    ana = layer.weight.grad

    max_err = np.abs(num - ana).max()
    assert max_err < 1e-4, f"Linear weight grad check failed: {max_err:.3e}"
    print(f"[PASS] Linear gradient wrt weight    max_err={max_err:.3e}")


def test_linear_gradient_wrt_bias():
    """Gradient check: dL/db through a Linear layer (tests broadcasting)."""
    layer = Linear(4, 3, seed=0)
    x_data = np.random.randn(5, 4)

    def f(b_np):
        layer.bias = Tensor(b_np, requires_grad=True)
        x = Tensor(x_data)
        return layer.forward(x).sum()

    b_np = layer.bias.data.copy()
    num = numerical_grad(f, b_np.copy())

    # Analytical
    layer.bias = Tensor(b_np.copy(), requires_grad=True)
    x = Tensor(x_data)
    out = layer.forward(x)
    out.sum().backward()
    ana = layer.bias.grad

    max_err = np.abs(num - ana).max()
    assert max_err < 1e-4, f"Linear bias grad check failed: {max_err:.3e}"
    print(f"[PASS] Linear gradient wrt bias      max_err={max_err:.3e}")


def test_mlp_forward_shape():
    """Test MLP forward produces correct output shape."""
    model = MLP([64, 128, 64, 10], seed=0)
    x = Tensor(np.random.randn(16, 64))
    out = model.forward(x)
    assert out.shape == (16, 10), f"Expected (16, 10), got {out.shape}"
    print("[PASS] MLP forward shape: (16, 64) -> (16, 10)")


def test_mlp_params_count():
    """Test MLP params count is correct."""
    model = MLP([64, 128, 64, 10], seed=0)
    params = model.params()
    total = sum(p.data.size for p in params)
    # Layer 0: 64*128 + 128 = 8320
    # Layer 1: 128*64 + 64 = 8256
    # Layer 2: 64*10 + 10 = 650
    expected = 8320 + 8256 + 650
    assert total == expected, f"Expected {expected} params, got {total}"
    print(f"[PASS] MLP params count: {total} (expected {expected})")


def test_he_init_vs_bad_init():
    """Verify He init keeps activations bounded vs std=1.0 which explodes."""
    x = np.random.randn(32, 64)

    # He init: activations should be ~O(1)
    model_he = MLP([64, 256, 256, 10], seed=0)
    out_he = model_he.forward(Tensor(x))
    max_he = np.abs(out_he.data).max()

    # std=1.0 init: activations grow with sqrt(fan_in) at each layer
    model_bad = MLP([64, 256, 256, 10], init_std=1.0, seed=0)
    out_bad = model_bad.forward(Tensor(x))
    max_bad = np.abs(out_bad.data).max()

    print(f"[INFO] He init max |activation|: {max_he:.2f}")
    print(f"[INFO] std=1.0 init max |activation|: {max_bad:.2f}")

    # He init should produce much smaller activations
    assert max_he < max_bad, (
        f"He init ({max_he:.2f}) should produce smaller activations "
        f"than std=1.0 ({max_bad:.2f})"
    )
    # He init activations should be reasonable (not exploding)
    assert max_he < 100, f"He init activations too large: {max_he:.2f}"
    print("[PASS] He init produces bounded activations, std=1.0 is much larger")


def test_masked_linear_forward():
    """Test that masking a weight in Linear makes the corresponding
    connection contribute exactly 0 to the output."""
    layer = Linear(4, 3, seed=0)
    x = Tensor(np.ones((1, 4)))

    # Get output with all weights active
    out_full = layer.forward(x).data.copy()

    # Mask out weight[2, 1] (connection from input 2 to output 1)
    layer.weight.mask[2, 1] = 0.0
    layer.weight.data[2, 1] = 0.0

    out_masked = layer.forward(x).data.copy()

    # The outputs should differ only in column 1 (output neuron 1)
    # and the difference should be exactly the original weight value
    # times the input value (which is 1.0 here)
    assert out_masked[0, 0] == out_full[0, 0], "Masking weight[2,1] shouldn't affect output 0"
    assert out_masked[0, 2] == out_full[0, 2], "Masking weight[2,1] shouldn't affect output 2"
    print("[PASS] Masked weight produces exactly zero contribution to output")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} layer tests passed.")
