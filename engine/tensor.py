"""
Minimal reverse-mode autodiff engine.

Design notes (see DESIGN.md for the full derivations):

- Every Tensor produced by an op stores its parents and a `_backward` closure
  that knows how to push gradient from this node to its parents.
- `backward()` builds a topological order via DFS-postorder (not naive
  recursion into parents, which breaks / double-counts on diamond graphs
  like a = x; b = a*a + a) and then walks it in reverse, calling each
  node's `_backward` exactly once.
- Gradients are ACCUMULATED (+=) into `.grad`, never overwritten, because a
  node can have multiple children feeding gradient back to it (the a*a + a
  case: `a` gets contributions from both the `a*a` branch and the `+a`
  branch).
- Broadcasting: when a forward op broadcasts (e.g. (N,D) + (D,)), the
  backward pass must SUM the incoming gradient over the broadcast axes to
  get back to the original (pre-broadcast) shape. This is what
  `_unbroadcast` does. Missing this is the single most common bug in
  from-scratch autodiff engines.
- Masking (for pruning) is NOT a special case bolted onto the engine. A
  masked weight is implemented as `effective_w = w * mask` where `mask` is
  a plain (non-trainable) Tensor of 0s/1s. Because multiplication's
  backward rule is `dL/da = dL/d(a*b) * b`, this means:
      dL/dw = dL/d(effective_w) * mask
  i.e. the gradient of a weight whose mask entry is 0 is EXACTLY ZERO,
  forced by the ordinary chain rule -- not a convention we impose
  separately. This is the "well-defined gradient treatment" the challenge
  asks for. See optim/adam.py for why this alone is NOT sufficient to
  protect a pruned weight from drifting (stale momentum).
"""

import numpy as np


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_children", "_backward", "_op", "_probs", "mask")

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._children = _children
        self._backward = lambda: None
        self._op = _op

    # ---------------------------------------------------------------- utils
    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self.grad is not None:
            self.grad[...] = 0.0

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op='{self._op}', requires_grad={self.requires_grad})"

    @staticmethod
    def _coerce(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    @staticmethod
    def _unbroadcast(grad, target_shape):
        """Sum `grad` down to `target_shape`, undoing whatever numpy broadcast
        happened in the forward pass. Handles both:
          (a) extra leading dimensions (e.g. (N,D) + (D,) -> grad shape (N,D),
              target shape (D,))
          (b) size-1 dimensions that were stretched (e.g. (N,1) + (N,D))
        """
        # (a) remove extra leading dims
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        # (b) sum over dims that were size 1 in the original but broadcast up
        for i, dim in enumerate(target_shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def _accumulate(self, incoming_grad):
        """Accumulate `incoming_grad` into self.grad, unbroadcasting first."""
        g = self._unbroadcast(incoming_grad, self.data.shape)
        self.grad += g

    # ------------------------------------------------------------ backward
    def backward(self):
        assert self.data.size == 1, "backward() only from a scalar (e.g. a loss)"
        topo = []
        visited = set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------- add/sub
    def __add__(self, other):
        other = self._coerce(other)
        out_data = self.data + other.data
        req = self.requires_grad or other.requires_grad
        out = Tensor(out_data, requires_grad=req, _children=(self, other), _op="+")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad)
            if other.requires_grad:
                other._accumulate(out.grad)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op="neg")

        def _backward():
            if self.requires_grad:
                self._accumulate(-out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) + (-self)

    # ------------------------------------------------------------- mul/div
    def __mul__(self, other):
        other = self._coerce(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="*")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * other.data)
            if other.requires_grad:
                other._accumulate(out.grad * self.data)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._coerce(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="/")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad / other.data)
            if other.requires_grad:
                other._accumulate(-out.grad * self.data / (other.data ** 2))

        out._backward = _backward
        return out

    # ------------------------------------------------------------- matmul
    def __matmul__(self, other):
        other = self._coerce(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="matmul")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad @ other.data.T)
            if other.requires_grad:
                other._accumulate(self.data.T @ out.grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------ reductions
    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if self.requires_grad:
                g = out.grad
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis)
                self._accumulate(np.ones_like(self.data) * g)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ------------------------------------------------------------ activations
    def relu(self):
        mask = (self.data > 0).astype(self.data.dtype)
        out = Tensor(self.data * mask, requires_grad=self.requires_grad, _children=(self,), _op="relu")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * mask)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * (1 - t ** 2))

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * s * (1 - s))

        out._backward = _backward
        return out

    # ------------------------------------------------------ softmax + CE
    def softmax_cross_entropy(self, targets):
        """
        self: logits, shape (N, C)
        targets: integer class labels, shape (N,)
        Returns scalar mean cross-entropy loss.

        Numerically stable: subtract row-max before exponentiating
        (log-sum-exp trick). Naive exp(x)/sum(exp(x)) overflows for
        logits as small as ~1000.
        """
        x = self.data
        N = x.shape[0]
        shifted = x - x.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        log_probs = shifted - np.log(exp.sum(axis=1, keepdims=True))
        loss_val = -log_probs[np.arange(N), targets].mean()

        out = Tensor(loss_val, requires_grad=self.requires_grad, _children=(self,), _op="softmax_ce")

        def _backward():
            if self.requires_grad:
                grad = probs.copy()
                grad[np.arange(N), targets] -= 1.0
                grad /= N
                self._accumulate(out.grad * grad)

        out._backward = _backward
        out._probs = probs  # stashed for convenience (accuracy computation etc.)
        return out
