"""
Gradient checking: compare analytical gradients (from Tensor.backward())
against numerical gradients from finite differences.

We use the CENTERED difference:
    (f(x+h) - f(x-h)) / (2h)
not the forward difference (f(x+h)-f(x))/h. The forward difference has
O(h) error; the centered difference has O(h^2) error, so it's the
standard choice for gradient checks and lets us use a tighter tolerance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from engine.tensor import Tensor

np.random.seed(0)


def numerical_grad(f, x, h=1e-5):
    """f: function taking a Tensor (built fresh from a numpy array each call)
    and returning a scalar Tensor. x: numpy array.
    Returns numerical gradient, same shape as x."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]

        x[idx] = orig + h
        plus = float(np.asarray(f(Tensor(x.copy())).data).reshape(-1)[0])

        x[idx] = orig - h
        minus = float(np.asarray(f(Tensor(x.copy())).data).reshape(-1)[0])

        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * h)
        it.iternext()
    return grad


def analytical_grad(f, x):
    t = Tensor(x.copy(), requires_grad=True)
    out = f(t)
    out.backward()
    return t.grad.copy()


def check(name, f, x, atol=1e-4):
    num = numerical_grad(f, x.copy())
    ana = analytical_grad(f, x.copy())
    max_err = np.abs(num - ana).max()
    ok = max_err < atol
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:30s} max_abs_err={max_err:.3e}")
    assert ok, f"{name} gradient check failed: max_abs_err={max_err:.3e}"
    return ok


def test_add():
    x = np.random.randn(4, 3)
    check("add (scalar const)", lambda t: (t + 2.0).sum(), x)


def test_add_broadcast():
    # (N, D) + (D,) -- the classic broadcasting bug case
    bias = np.random.randn(3)

    def f(t):
        b = Tensor(bias)
        return (t + b).sum()

    x = np.random.randn(5, 3)
    check("add broadcast (N,D)+(D,)", f, x)


def test_sub():
    x = np.random.randn(4, 3)
    check("sub", lambda t: (t - 1.5).sum(), x)


def test_mul():
    x = np.random.randn(4, 3)
    y = np.random.randn(4, 3)

    def f(t):
        other = Tensor(y)
        return (t * other).sum()

    check("mul", f, x)


def test_div():
    x = np.random.randn(4, 3) + 5.0  # avoid div by zero region
    y = np.random.randn(4, 3) + 5.0

    def f(t):
        other = Tensor(y)
        return (t / other).sum()

    check("div", f, x)


def test_matmul():
    W = np.random.randn(3, 2)

    def f(t):
        w = Tensor(W)
        return (t @ w).sum()

    x = np.random.randn(4, 3)
    check("matmul", f, x)


def test_sum_mean():
    x = np.random.randn(4, 3)
    check("sum", lambda t: t.sum(), x)
    check("mean", lambda t: t.mean(), x)


def test_relu():
    x = np.random.randn(10) * 3  # spread across +/- so both branches hit
    check("relu", lambda t: t.relu().sum(), x)


def test_tanh():
    x = np.random.randn(10)
    check("tanh", lambda t: t.tanh().sum(), x)


def test_sigmoid():
    x = np.random.randn(10)
    check("sigmoid", lambda t: t.sigmoid().sum(), x)


def test_gradient_accumulation():
    # a*a + a  -> da/db = 2a + 1, and `a` is used twice, so its grad must
    # ACCUMULATE across both branches, not get overwritten by the second use.
    def f(t):
        return t * t + t

    x = np.array([2.0])
    num = numerical_grad(f, x.copy())
    ana = analytical_grad(f, x.copy())
    expected = 2 * x + 1
    assert np.allclose(ana, expected, atol=1e-6), f"expected {expected}, got {ana}"
    assert np.allclose(num, expected, atol=1e-4)
    print(f"[PASS] gradient accumulation (a*a+a)  analytical={ana}, expected={expected}")


def test_softmax_cross_entropy():
    N, C = 5, 4
    logits = np.random.randn(N, C)
    targets = np.random.randint(0, C, size=N)

    def f(t):
        return t.softmax_cross_entropy(targets)

    check("softmax_cross_entropy", f, logits, atol=1e-4)


def test_softmax_numerical_stability():
    # logits with huge magnitude would overflow a naive exp(x)/sum(exp(x))
    huge = np.array([[1000.0, 1001.0, 999.0]])
    t = Tensor(huge, requires_grad=True)
    loss = t.softmax_cross_entropy(np.array([1]))
    assert np.isfinite(loss.data), "softmax_cross_entropy overflowed on large logits"
    loss.backward()
    assert np.all(np.isfinite(t.grad)), "gradient is non-finite on large logits"
    print(f"[PASS] softmax numerical stability  loss={float(loss.data):.4f} (finite)")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} gradient-check tests passed.")
