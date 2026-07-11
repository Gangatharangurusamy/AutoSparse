# Design Document — Self-Pruning Network

## 1. Deriving the `|w·g|` Importance Criterion from a First-Order Taylor Expansion

Consider a trained network with loss function $L(\mathbf{w})$ where $\mathbf{w}$ is the vector of all weights. We want to estimate: **if we remove a single connection $w_{ij}$ (set it to zero), how much does the loss increase?**

### The derivation

Let $\delta\mathbf{w}$ be a perturbation to the weight vector. A first-order Taylor expansion of the loss gives:

$$L(\mathbf{w} + \delta\mathbf{w}) \approx L(\mathbf{w}) + \nabla L(\mathbf{w})^\top \delta\mathbf{w}$$

"Removing" connection $w_{ij}$ means setting it to zero, so the perturbation for that single weight is $\delta w_{ij} = -w_{ij}$ (we move it from its current value to zero). All other perturbations are zero.

Substituting:

$$\Delta L_{ij} = L(\mathbf{w} + \delta\mathbf{w}) - L(\mathbf{w}) \approx \frac{\partial L}{\partial w_{ij}} \cdot (-w_{ij}) = -g_{ij} \cdot w_{ij}$$

where $g_{ij} = \frac{\partial L}{\partial w_{ij}}$ is the gradient of the loss with respect to weight $w_{ij}$.

The *magnitude* of this loss change is:

$$|\Delta L_{ij}| \approx |w_{ij} \cdot g_{ij}|$$

### Why small `|w·g|` implies small expected loss increase

A weight with small $|w_{ij} \cdot g_{ij}|$ satisfies one or both of:
- **Small weight**: the connection is weak, so removing it doesn't perturb the network's output much.
- **Small gradient**: the loss is locally flat with respect to this weight, so even a large perturbation has little effect on the loss.

If *both* are small, the connection is both weak and unimportant to the current loss landscape — a safe pruning candidate.

### Why this beats magnitude-only pruning

Magnitude pruning uses $|w_{ij}|$ alone. This misses cases where a small weight has a very large gradient (the connection is small but the loss is very sensitive to it — pruning it would cause a large loss increase). Conversely, it penalizes large weights that have near-zero gradients (the connection is large but in a "flat valley" of the loss — pruning it barely changes the loss).

The saliency criterion $|w \cdot g|$ captures both effects in a single score.

---

## 2. Masked Weight Gradients and Adam's Stale Momentum Problem

### What the engine computes as "the gradient of a masked weight"

In `engine/tensor.py`, a masked weight is represented as:

$$w_{\text{eff}} = w \odot m$$

where $m$ is a binary mask (0/1) and $\odot$ is elementwise multiplication. This product happens *inside the computational graph*, not as a post-hoc override.

By the chain rule, the gradient with respect to the original weight $w$ is:

$$\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial w_{\text{eff},ij}} \cdot m_{ij}$$

Where $m_{ij} = 0$ (masked/pruned), this gives $\frac{\partial L}{\partial w_{ij}} = 0$ **exactly**, not approximately. This is mathematically correct: the weight has no path to influence the loss (the mask blocks it), so the loss is flat with respect to it.

### Why this alone is NOT sufficient for Adam

Zero gradient at every step does **not** mean zero Adam update. Adam maintains exponential moving averages of the gradient ($m_t$) and squared gradient ($v_t$):

```
m_t = β₁·m_{t-1} + (1 - β₁)·g_t
v_t = β₂·v_{t-1} + (1 - β₂)·g_t²
```

If a weight had nonzero momentum $m_{t-1}$ **before** it was pruned, then after pruning (when $g_t = 0$ every step):

```
m_t = β₁·m_{t-1} + (1 - β₁)·0 = β₁·m_{t-1}
```

The momentum decays geometrically toward zero (by factor $β_1 = 0.9$) but never reaches it. The bias-corrected update $\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ remains nonzero, and Adam applies it to the weight — so a "pruned" weight silently drifts away from zero.

This is proven empirically in `tests/test_masked_weights.py`: a naive Adam lets a pruned weight drift to -0.71 over 20 steps from stale momentum alone.

### What `sync_masks()` does about it

`Adam.sync_masks()` (in `optim/adam.py`) does two things when a mask entry transitions from 1→0:

1. **Resets optimizer state**: Sets $m_i = 0$ and $v_i = 0$ at the newly-pruned entry. This eliminates the stale momentum entirely — there is nothing left to drive a spurious update.

2. **Hard-zeros the weight data**: Sets $w_i = 0$ immediately, so the weight is exactly zero from the moment of pruning.

Additionally, `Adam.step()` masks the update itself (`update *= mask`) before applying it to the weight. This belt-and-suspenders approach guarantees a pruned weight literally never moves, regardless of any numerical edge case.

On **revival** (mask 0→1), the weight starts from clean $m = 0$, $v = 0$ state — as if it were a freshly initialized parameter. We deliberately do **not** restore old optimizer state, because that state reflects a loss landscape that no longer exists (the network has been extensively retrained since the weight was pruned). Restoring it would reintroduce the exact stale-momentum bug that pruning was designed to avoid.

---

## 3. Autodiff Engine Bottlenecks

### Where this engine bottlenecks

1. **Python-level per-operation overhead**: Every `+`, `*`, `@`, `.relu()` creates a new `Tensor` object, allocates a numpy array for the output, and builds a Python closure for the backward pass. For a 3-layer MLP with batch size 64, this is ~15-20 Tensor operations per forward+backward pass. The Python interpreter overhead (object allocation, closure creation, dict lookups) dominates over the actual numpy array computation time for the small arrays in this problem.

2. **No operator fusion**: A ReLU applied after a matrix multiply does `out1 = x @ W` (allocates array), then `out2 = max(out1, 0)` (allocates another array, reads `out1`, writes `out2`). A fused kernel would compute `max(x @ W, 0)` in a single pass over the output, saving one memory allocation and one full read/write of the intermediate result. For large models, memory bandwidth (not compute) is the bottleneck, so fusion matters enormously.

3. **No GPU parallelism**: All operations run sequentially on CPU via numpy. For the matrix multiplies that dominate MLP compute, a GPU can provide 100-1000× speedup via massively parallel SIMD execution.

4. **No batched sparse operations**: Even with sparsity masks, the engine still does dense `W * mask` multiplies (full FLOPs), then dense `x @ (W * mask)`. A true sparse BLAS library (cuSPARSE, Intel MKL sparse) would skip zero entries entirely.

5. **Graph rebuilding every forward pass**: The computational graph (parent pointers, backward closures) is rebuilt from scratch on every forward pass. A compiled/traced approach (like XLA or TorchScript) would build the graph once, optimize it, and reuse it.

### How to optimize for production

- **Compile the graph**: Trace the forward computation once, convert to an intermediate representation (IR), apply optimization passes (constant folding, dead code elimination, operator fusion), and lower to CUDA/XLA/LLVM for execution.
- **Fuse elementwise chains**: Merge sequences of elementwise operations (add, multiply, ReLU, mask) into single kernels to reduce memory traffic.
- **Use sparse BLAS**: For structured sparsity (block-sparse, 2:4), use hardware-accelerated sparse matrix multiply (NVIDIA Ampere's sparse tensor cores provide 2× throughput for 2:4 structured sparsity).
- **Vectorize the backward pass**: Batch gradient computations across samples instead of accumulating per-sample.
- **Use mixed precision**: FP16/BF16 for forward/backward, FP32 for optimizer state. Halves memory bandwidth and doubles throughput on modern GPUs.

---

## 4. Serving a Self-Pruned Model at Scale

### The core tension: unstructured sparsity vs hardware efficiency

Our pruning produces **unstructured sparsity** — individual weight entries are zeroed in arbitrary locations. This is optimal for accuracy (the network decides which connections matter) but problematic for hardware:

- **GPU**: Modern GPUs execute instructions in warps (32 threads). If a weight matrix has scattered zeros, every warp still loads the full row and executes the full multiply — it just multiplies some entries by zero. There is no speedup. The memory access pattern is unchanged, and memory bandwidth (not compute) is the bottleneck for inference.
- **CPU**: Similar issues with SIMD lanes (AVX-512 operates on 16 floats at once). Scattered zeros don't help unless you convert to a sparse format (CSR/CSC/COO), which adds indexing overhead that only pays off at very high sparsity (>95%).

### Structured sparsity for real speedups

To actually speed up inference, convert unstructured sparsity to **structured sparsity**:

- **Block sparsity**: Prune entire blocks (e.g., 4×4 or 8×8 blocks of weights). The GPU can skip entire blocks, and the memory access pattern becomes regular. Costs some accuracy vs unstructured (less flexibility in which connections to keep).
- **N:M sparsity** (e.g., NVIDIA's 2:4): In every group of 4 consecutive weights, exactly 2 are zero. NVIDIA Ampere/Hopper sparse tensor cores natively accelerate this pattern, providing exactly 2× throughput with ~1-2% accuracy loss. This is the most practical path to production speedup today.
- **Channel/filter pruning**: Remove entire rows or columns of weight matrices (corresponding to neurons/channels). This directly reduces matrix dimensions, giving linear speedup with no special sparse hardware needed. But it's the most accuracy-destructive form of pruning.

### Multi-tenant batching implications

In a multi-tenant inference service (thousands of requests/sec, multiple models):

- **If each model has a different sparsity pattern**: Batching across models is impossible because the weight matrices have different shapes (structured pruning) or different zero locations (unstructured). Each model needs its own forward pass, killing GPU utilization.
- **Shared sparsity pattern across models**: If all models in a tenant class share the same pruning mask (e.g., pruned once, then fine-tuned per-tenant), batching works normally — the shared sparse weight matrix is loaded once and applied to all requests in the batch.
- **Distillation**: The most practical approach for production may be to use the pruned model as a *teacher* and distill its knowledge into a smaller, fully-dense *student* model. The student has regular, batch-friendly matrix shapes and runs at full hardware efficiency.

### Cost measurement choices in Part 4

We report three cost metrics:

1. **Active parameter count**: `active_params = mask.sum()` across all layers. Honest proxy for "how much computation is *mathematically necessary*" regardless of hardware. Reported as a fraction of the dense baseline.

2. **FLOPs estimate**: `flops = 2 * active_params` per layer (one multiply + one add per active connection). This is the standard FLOPs metric used in the pruning literature.

3. **Wall-clock time with genuine sparse matmul**: We use `scipy.sparse.csr_matrix` for the forward pass (not `dense * mask`, which still does full FLOPs). This measures whether sparsity *actually* translates to CPU wall-clock savings via a CSR representation. We clearly label this as "sparse CSR timing on CPU" and do not conflate it with the dense-times-zero path.

---

## Design Decisions

### Weight initialization: He init

We use He initialization (`std = sqrt(2/fan_in)`) for ReLU-activated layers. This compensates for ReLU killing approximately half the activations at each layer. With Xavier init (`std = sqrt(1/fan_in)`), activation variance halves per layer, leading to vanishing signals. With naive `std = 1.0`, variance grows as `fan_in` per layer, causing exploding activations. He init maintains unit variance across layers. This is demonstrated empirically in `run_part2.py`.

### Gradual (cubic) vs one-shot pruning

We use the cubic sparsity schedule from Zhu & Gupta (2017) rather than one-shot pruning because:
- **Recovery opportunity**: After each pruning event, the remaining weights can be retrained to compensate for the lost connections before the next pruning step.
- **Better final accuracy**: At the same target sparsity, gradual pruning consistently achieves higher accuracy than one-shot, because the network adapts its internal representations as connections are removed.
- **Cubic shape**: The cubic ramp prunes slowly at first (letting the network learn initial features), accelerates mid-training, and tapers off near the end (smaller adjustments as sparsity approaches target).
- **Fine-tuning phase**: The last 20% of training runs at fixed sparsity with no further pruning, giving the network time to fully optimize within its sparse topology.

### Global vs per-layer pruning

We use **global pruning** with a per-layer minimum (1% of original connections). Global pruning pools importance scores across all layers and prunes the globally least-important connections. This allows the network to allocate capacity non-uniformly across layers — e.g., keeping more connections in the first layer (which processes raw features) while pruning more aggressively in redundant hidden layers. The per-layer minimum prevents any single layer from being completely stripped, which would sever the information flow entirely.

### Regrowth strategy

The optional regrowth feature uses **random regrowth** — randomly selecting which pruned entries to revive, rather than using gradient-based selection. This is because:
- The gradient at pruned entries is exactly zero (by the chain rule via `w * mask`), so we cannot use the actual gradient to assess which pruned connections would be most valuable if restored.
- Computing "dense gradients" (the gradient as if the mask were temporarily all-ones) would require a separate forward+backward pass without masks, doubling compute.
- Random regrowth is a valid baseline used in the literature (e.g., SET by Mocanu et al. 2018).
- Revived weights are re-initialized to small random values (He-scaled, times 0.1) and start with clean optimizer state (m=0, v=0) per `sync_masks()`. This prevents stale-momentum corruption (trap #9).

### Optimizer state on revival

When a weight is revived (mask 0→1), its Adam state starts from $m = 0$, $v = 0$ — as if it were a freshly initialized parameter. We do NOT restore old state because:
1. The old state reflects gradients from a different loss landscape (before pruning changed the network topology).
2. Restoring stale momentum would cause the exact same spurious-jump bug that Part 1 was designed to catch and prevent.
3. Starting from zero is the conservative, defensible choice: Adam will build up appropriate momentum for the revived weight based on its actual gradients in the current landscape.

### scikit-learn usage

`sklearn.datasets.load_digits()` is used **exclusively for loading** the digits dataset. No sklearn models, optimizers, gradient computation, or training utilities are used anywhere in this project.
