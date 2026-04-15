# triton-tagi: Engineering Plan

This document describes the direction, conventions, and implementation roadmap for triton-tagi. It is the single source of truth for engineering decisions. Update it when decisions change.

---

## 1. Vision

triton-tagi is a Python/Triton library for Bayesian neural networks using Tractable Approximate Gaussian Inference (TAGI). Its goals, in priority order:

1. **Correct** — numerically validated against cuTAGI, the reference C++/CUDA implementation.
2. **Accessible** — installable via `pip install triton-tagi`, no C++/CUDA compilation required.
3. **Extensible** — typed, documented abstractions that make adding a new layer a two-hour task.
4. **Fast** — fused Triton kernels for GPU acceleration, benchmarked against cuTAGI.

The target audience is TAGI researchers and practitioners who want to experiment with Bayesian inference in Python without compiling cuTAGI. cuTAGI remains the production-grade, multi-GPU reference; triton-tagi is its Python-native counterpart, and the two must remain numerically consistent.

### Long-term architectural direction: TAGI autograd

The current architecture requires every layer to manually implement both `forward()` (moment propagation) and `backward()` (delta propagation). This is viable for a fixed catalog of layers but breaks down the moment a user wants to compose operations freely — a residual sum, an elementwise product, an attention score — because each composition requires a new hand-derived backward pass.

The long-term target is a **TAGI computation graph**: the user defines only the forward moment propagation, and the library derives the backward delta propagation automatically, analogously to how PyTorch autograd derives gradients from the forward computation.

This requires three components:

1. A `GaussianTensor` type wrapping `(ma, Sa)` that records the operations applied to it, building a computation graph as the forward pass runs.
2. A set of **primitive TAGI ops** — `add`, `mul`, `linear`, `conv`, `relu`, etc. — each registering its own Bayesian backward rule (the TAGI analogue of a `grad_fn`). These rules are the closed-form delta propagation equations derived from the TAGI framework.
3. A **graph traversal** at backward time that walks the recorded graph in reverse and applies each op's backward rule to propagate `delta_ma` and `delta_Sa` through the full computation, regardless of branching or skip connections.

Example of what this enables:

```python
class MyResBlock(Module):
    # User defines only the forward pass. No backward() needed.
    def forward(self, x: GaussianTensor) -> GaussianTensor:
        z = self.bn(self.relu(self.conv(x)))
        skip = self.proj(x)          # projection shortcut — graph branches here
        return z + skip              # AddOp recorded; backward distributes deltas to both branches
```

This eliminates the `Add` layer workaround currently needed in `resblock.py`, makes attention mechanisms and arbitrary graph topologies straightforward to implement, and means a new layer requires only a forward implementation and a single backward rule for its primitive op — not a full backward pass that reasons about the surrounding graph.

**This is a Phase 5+ endeavor.** Phases 0 through 4 must be complete first because the primitive op backward rules need to be individually correct and validated before they can be composed automatically. The explicit `forward/backward/update` contract of the current `LearnableLayer` ABC is the foundation — it gives individually verified building blocks that phase 5 promotes into registered `grad_fn`-style ops.

---

## 2. Current State Assessment

### What is solid

- Core layers: `Linear`, `Conv2D`, `BatchNorm2D`, `ResBlock`
- Triton kernels: `kernels/common.py` (variance-forward, backward-delta, im2col)
- Activation moment propagation: `ReLU`, `LeakyReLU`, `EvenSoftplus`
- Parameter update logic: `update/observation.py`, `update/parameters.py`
- Initialization: `param_init.py`, `init.py`
- Novel contributions not in cuTAGI: `Remax`, `Bernoulli`, `AdamTAGI`, `NadamTAGI`, `StateSpaceMomentum`, `TAGIMonitor`, `auto_tune`

### What needs cleanup before any new features

- No formal layer interface (ABC) — layer duck-typing via `_LEARNABLE_LAYERS` tuple is fragile
- No type annotations
- Inconsistent naming (`Sw` vs `sw` across layers)
- `src/` is not an installable package
- Tests scattered at root level with no structure
- Many training scripts and `run_logs*/` directories pollute the root
- Commented-out code in `remax.py`, `tlu.py`, others

### What is missing (compared to cuTAGI)

| Layer | Priority |
|---|---|
| `LayerNorm` | P1 |
| `MaxPool2D` | P1 |
| `ConvTranspose2D` | P2 |
| `LSTM` | P3 |
| `SelfAttention` | P3 |
| `Embedding` | P3 |

---

## 3. Repository Structure

The library should live in a standalone repository. The current layout mixes library code, experiments, logs, and the cuTAGI submodule at the same level. The target layout:

```
triton-tagi/
├── triton_tagi/                  # installable package (replaces src/)
│   ├── __init__.py               # public API: Sequential, all layers, optimizers
│   ├── base.py                   # Layer and LearnableLayer ABCs
│   ├── network.py                # Sequential container
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── linear.py
│   │   ├── conv2d.py
│   │   ├── batchnorm2d.py
│   │   ├── resblock.py
│   │   ├── avgpool2d.py
│   │   ├── flatten.py
│   │   ├── relu.py
│   │   ├── leaky_relu.py
│   │   ├── even_softplus.py
│   │   ├── silu.py
│   │   ├── remax.py
│   │   ├── bernoulli.py
│   │   ├── frn2d.py
│   │   ├── tlu.py
│   │   └── shared/               # shared-variance variants
│   │       ├── __init__.py
│   │       ├── linear.py
│   │       ├── conv2d.py
│   │       ├── batchnorm2d.py
│   │       └── resblock.py
│   ├── kernels/
│   │   ├── __init__.py
│   │   └── common.py
│   ├── optimizers/               # consolidates optimizer.py, nadam_optimizer.py, momentum.py
│   │   ├── __init__.py
│   │   ├── adam.py               # AdamTAGI
│   │   ├── nadam.py              # NadamTAGI
│   │   └── momentum.py           # StateSpaceMomentum
│   ├── init/                     # consolidates param_init.py, init.py, inference_init.py
│   │   ├── __init__.py
│   │   ├── params.py             # He / Xavier weight init
│   │   ├── network.py            # reinit_net, init_residual_aware
│   │   └── inference.py          # inference-mode parameter scaling
│   ├── diagnostics/              # consolidates monitor.py, auto_tune.py
│   │   ├── __init__.py
│   │   ├── monitor.py
│   │   └── auto_tune.py
│   └── update/
│       ├── __init__.py
│       ├── observation.py
│       ├── parameters.py
│       └── shared_var.py
├── tests/
│   ├── conftest.py               # shared fixtures, device marks
│   ├── unit/                     # per-component, no cuTAGI dependency
│   │   ├── layers/
│   │   │   ├── test_linear.py
│   │   │   ├── test_conv2d.py
│   │   │   ├── test_batchnorm2d.py
│   │   │   ├── test_relu.py
│   │   │   └── ...
│   │   ├── test_network.py
│   │   └── test_update.py
│   └── validation/               # numerical comparison against cuTAGI
│       ├── conftest.py           # cuTAGI import guard + shared fixtures
│       ├── test_linear.py
│       ├── test_conv2d.py
│       ├── test_batchnorm2d.py
│       ├── test_resblock.py
│       ├── test_activations.py
│       └── test_end_to_end.py
├── examples/                     # replaces root-level run_*.py / train_*.py
│   ├── mnist_mlp.py
│   ├── mnist_cnn.py
│   ├── cifar10_cnn.py
│   ├── cifar10_resnet18.py
│   └── custom_layer.py           # minimal guide to implementing a new layer
├── benchmarks/
│   ├── bench_linear.py
│   ├── bench_conv2d.py
│   └── bench_vs_cutagi.py
├── pyproject.toml
├── .gitignore                    # includes run_logs*/, __pycache__/, *.egg-info/
├── PLAN.md                       # this document
├── CONTRIBUTING.md
├── STYLE_GUIDE.md                # fill in §3 Code from this plan
└── README.md
```

Concretely: rename `src/` to `triton_tagi/`, move all root-level scripts to `examples/`, add `pyproject.toml`. The cuTAGI directory becomes an optional dependency for validation tests only.

---

## 4. Engineering Conventions

### 4.1 Tensor Naming

These names are fixed. They match the cuTAGI C++ convention and the TAGI paper notation. Do not rename to `mu/var`, `mean/variance`, or any other scheme.

| Symbol | Meaning | Typical shape |
|---|---|---|
| `ma` | activation mean | `(N, ...)` |
| `Sa` | activation variance | `(N, ...)` |
| `mw` | weight mean | `(out, in)` |
| `Sw` | weight variance | `(out, in)` |
| `mb` | bias mean | `(out,)` |
| `Sb` | bias variance | `(out,)` |
| `delta_ma` | backward delta on activation mean | `(N, ...)` |
| `delta_Sa` | backward delta on activation variance | `(N, ...)` |
| `delta_mw` | stored weight mean delta | same as `mw` |
| `delta_Sw` | stored weight variance delta | same as `Sw` |

All capital-`S` symbols are variances (σ²), not standard deviations. This distinction is enforced throughout.

### 4.2 Python Naming

- Classes: `PascalCase`
- Functions and variables: `snake_case`
- Module-level constants: `UPPER_SNAKE_CASE`
- Private helpers: leading underscore (`_compute_col2im`)
- One primary class per file; file name matches the class in lowercase (e.g. `linear.py` → `Linear`)

### 4.3 Type Annotations

All public functions and class methods have full type annotations. Use `from __future__ import annotations` at the top of every file to enable forward references without runtime cost.

```python
from __future__ import annotations
from torch import Tensor

def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]: ...
def update(self, cap_factor: float) -> None: ...
```

Do not annotate private helper functions unless the types are non-obvious.

### 4.4 Docstrings

Google-style docstrings for all public classes and methods. Document the mathematical operation, not the implementation.

```python
def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
    """Propagate Gaussian moments through a linear layer.

    Computes the first two moments of z = a W^T + b, where W and b
    are independent Gaussians and a has mean ma and variance Sa.

    Args:
        ma: Activation means, shape (N, in_features).
        Sa: Activation variances, shape (N, in_features).

    Returns:
        mz: Output means, shape (N, out_features).
        Sz: Output variances, shape (N, out_features).
    """
```

Private methods do not need docstrings. The mathematical formulas belong in the docstring, not in inline comments.

### 4.5 Device Handling

- Learnable layers accept `device` at construction and store all parameters on that device.
- Activation layers (ReLU, Remax, etc.) are device-agnostic; they operate on whatever tensors are passed in.
- `Sequential` moves all parameters to `device` during `__init__`.
- Never perform device placement inside `forward()` or `backward()`.

### 4.6 Code Style

- Formatter: `ruff format` (line length: 100)
- Linter: `ruff check` with rules `E, F, I, UP, B`
- No commented-out code in committed files; remove or convert to a docstring
- No `print()` calls in library code; use `logging.getLogger(__name__)` if output is needed
- No bare `except:` clauses

Run before every commit:
```bash
ruff format triton_tagi/ tests/
ruff check triton_tagi/ tests/
```

### 4.7 Invariants Every Layer Must Maintain

1. `Sa` is always non-negative after `forward`. Clamp with `Sa.clamp(min=0)` at the end of forward if numerical drift is possible.
2. `forward` is stateless for activation layers (no `self._cache`).
3. Learnable layers may store intermediate values from `forward` on `self._cache` (a dict). The cache is cleared at the start of each `forward` call.
4. All tensors in `_cache` and all delta tensors reside on the same device as the layer's parameters.

---

## 5. Layer Contract (ABCs)

Define in `triton_tagi/base.py`:

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from torch import Tensor


class Layer(ABC):
    """Base class for all TAGI layers."""

    @abstractmethod
    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """Propagate Gaussian moments forward through the layer."""
        ...

    @abstractmethod
    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        """Propagate innovation deltas backward through the layer."""
        ...


class LearnableLayer(Layer):
    """Base class for layers with trainable parameters (weights, biases, etc.)."""

    @abstractmethod
    def update(self, cap_factor: float) -> None:
        """Apply the capped parameter update using deltas stored during backward."""
        ...

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """Total number of learnable scalars (means + variances combined)."""
        ...
```

`Sequential` dispatches `update()` calls via `isinstance(layer, LearnableLayer)`, replacing the current `_LEARNABLE_LAYERS` tuple. Adding a new layer type requires only inheriting from the correct ABC — no edits to `network.py`.

---

## 6. Systematic Validation Against cuTAGI

This is the highest-priority quality gate before any new feature work. Every layer shared between triton-tagi and cuTAGI must have a validation test demonstrating numerical agreement.

### 6.1 Strategy

Validation tests live in `tests/validation/`. They are skipped automatically if cuTAGI is not installed, so they never block contributors who only have triton-tagi:

```python
# tests/validation/conftest.py
import pytest
pytagi = pytest.importorskip("pytagi", reason="cuTAGI (pytagi) not installed")
```

Comparison uses three levels, always in this order:

**Level 1 — Forward pass.** Feed identical inputs and weights to both libraries; check that output means and variances match.

**Level 2 — Backward pass.** Feed identical innovation deltas from the output; check that the backward-propagated deltas match.

**Level 3 — Parameter update.** After a complete forward + backward step on the same data, check that updated weight and bias means and variances match.

Tolerance: `atol=1e-4, rtol=0` for fp32 (GPU arithmetic order differs between implementations). Always set `torch.manual_seed(0)` and cuTAGI's equivalent at the start of every test.

### 6.2 Validation Matrix

| Layer | Forward | Backward | Update | Priority |
|---|---|---|---|---|
| `Linear` | ☐ | ☐ | ☐ | P0 |
| `Conv2D` | ☐ | ☐ | ☐ | P0 |
| `BatchNorm2D` | ☐ | ☐ | ☐ | P0 |
| `ResBlock` (identity shortcut) | ☐ | ☐ | ☐ | P0 |
| `ResBlock` (projection shortcut) | ☐ | ☐ | ☐ | P0 |
| `AvgPool2D` | ☐ | ☐ | — | P0 |
| `ReLU` | ☐ | ☐ | — | P0 |
| `LeakyReLU` | ☐ | ☐ | — | P0 |
| `EvenSoftplus` | ☐ | ☐ | — | P1 |
| Full MLP (MNIST, 3 epochs) | ☐ | — | — | P0 |
| Full CNN (CIFAR-10, 3 epochs) | ☐ | — | — | P1 |

P0 must pass before any new feature work begins. P1 must pass before Phase 2.

### 6.3 Test Template

```python
# tests/validation/test_linear.py
import pytest
import torch
pytagi = pytest.importorskip("pytagi")

DEVICE = "cuda"
ATOL = 1e-4

def make_triton_layer(in_f, out_f, mw, Sw, mb, Sb):
    from triton_tagi.layers import Linear
    layer = Linear(in_f, out_f, device=DEVICE)
    layer.mw = mw.clone()
    layer.Sw = Sw.clone()
    layer.mb = mb.clone()
    layer.Sb = Sb.clone()
    return layer

def make_cutagi_layer(in_f, out_f, mw, Sw, mb, Sb):
    # Use cuTAGI Python API to build equivalent layer
    # Exact API TBD after inspecting pytagi bindings
    ...

def test_linear_forward():
    torch.manual_seed(0)
    N, in_f, out_f = 8, 32, 16
    mw = torch.randn(out_f, in_f, device=DEVICE)
    Sw = torch.rand(out_f, in_f, device=DEVICE).abs() * 0.1
    mb = torch.randn(out_f, device=DEVICE)
    Sb = torch.rand(out_f, device=DEVICE).abs() * 0.1
    ma = torch.randn(N, in_f, device=DEVICE)
    Sa = torch.rand(N, in_f, device=DEVICE).abs() * 0.1

    tri = make_triton_layer(in_f, out_f, mw, Sw, mb, Sb)
    cut = make_cutagi_layer(in_f, out_f, mw, Sw, mb, Sb)

    mz_tri, Sz_tri = tri.forward(ma, Sa)
    mz_cut, Sz_cut = cut.forward(ma, Sa)

    torch.testing.assert_close(mz_tri, mz_cut, atol=ATOL, rtol=0)
    torch.testing.assert_close(Sz_tri, Sz_cut, atol=ATOL, rtol=0)
```

### 6.4 End-to-End Validation

Beyond layer tests, run a 3-layer MLP on MNIST for 3 epochs with both libraries, using identical batch order, sigma_v, and initialization. Pass criteria:

- Final test accuracy within 0.5%
- Training loss at each epoch within 2% of each other

This catches cross-layer interaction bugs that unit tests miss. It must run in under 5 minutes on a single GPU.

### 6.5 Finding the cuTAGI Python API

Before writing validation tests, inspect `triton/cuTAGI/` to identify:

1. The installed Python package name (`pytagi`? `cutagi`? Check `cuTAGI/pyproject.toml`)
2. How to set layer weights programmatically (search for `mw`, `theta`, or `param` in the binding files)
3. How to run a forward pass on a single layer vs. a full network

This reconnaissance step is the first task of Phase 1.

---

## 7. Implementation Roadmap

### Phase 0: Foundation (prerequisite for everything)

These tasks are independent and can be done in any order:

1. **Create `triton_tagi/base.py`** with `Layer` and `LearnableLayer` ABCs. Update all existing layers to inherit from the correct base. Verify `isinstance` checks in `network.py` work.

2. **Add type annotations** to all public methods in all existing layers and `network.py`. Run `ruff check` to catch obvious issues.

3. **Add `pyproject.toml`** to make the package installable with `pip install -e .`. Use `hatchling`. Add `ruff` to dev dependencies.

4. **Run `ruff format` and `ruff check`** on all files. Fix all violations. Remove commented-out code (commit it to a scratch branch if it feels useful to preserve).

5. **Normalize variance naming**: all layers must use `Sw`/`Sb` (capital S, matching the TAGI paper). Audit `shared_var_*.py` files which currently use `sw`/`sb` in some places.

6. **Move test files** from root to `tests/unit/`. Run `pytest tests/unit/` and confirm they pass.

7. **Move training scripts** to `examples/`. Add `run_logs*/`, `*.json`, `*.png` (generated artifacts) to `.gitignore`.

8. **Fill in §3 Code of `STYLE_GUIDE.md`** based on §4 of this document.

### Phase 1: Systematic Validation

1. Inspect the cuTAGI Python API (see §6.5). Document findings in a `tests/validation/README.md`.
2. Write and pass validation tests for all P0 layers (§6.2). Fix any discrepancies — they are bugs, not acceptable differences.
3. Write and pass the MNIST end-to-end validation test.
4. Check off the P0 rows in the validation matrix (§6.2).

### Phase 2: Missing Layers

In order:

1. **`LayerNorm`** — same moment-propagation pattern as `BatchNorm2D` but normalizes over the feature dimension, not the batch. No running statistics. Straightforward extension.

2. **`MaxPool2D`** — requires propagating which input position achieved the maximum through the Gaussian distribution, typically via a soft-argmax or the expected indicator formulation. Validate against cuTAGI.

3. **`ConvTranspose2D`** — transpose of `Conv2D`; shares the im2col infrastructure already in `kernels/common.py`. The forward is the current backward and vice versa.

For each new layer: write unit tests first, then the implementation, then the validation test.

### Phase 3: Performance Benchmarks

1. Implement `benchmarks/bench_vs_cutagi.py` measuring wall-clock time per forward+backward+update step for `Linear`, `Conv2D`, `BatchNorm2D` at batch sizes {64, 256, 1024}.
2. Identify the top 3 bottlenecks by profiling with `torch.profiler`.
3. Address the top bottleneck with a targeted Triton kernel improvement.
4. Report results in `benchmarks/results.md`.

### Phase 4: Documentation & Contribution Guide

1. Write `CONTRIBUTING.md` — development setup, running tests, how to add a new layer (referencing `examples/custom_layer.py`).
2. Write `examples/custom_layer.py` — a minimal, documented example of implementing a new activation layer.
3. Add equations (from the paper) to all layer forward/backward docstrings that do not already have them.

---

## 8. Testing Strategy

### Unit Tests (`tests/unit/`)

One test file per module. Minimum test set per layer:

| Test | What it checks |
|---|---|
| `test_<layer>_forward_shape` | Output shape is correct for standard input |
| `test_<layer>_forward_zero_sa` | Sa=0 (deterministic input) reduces to deterministic forward; Sa_out is non-negative |
| `test_<layer>_backward_shape` | Delta output shape matches delta input shape |
| `test_<layer>_backward_passthrough` | Backward with zero deltas returns zero deltas |
| `test_<layer>_update_decreases_sw` | After update with a non-zero learning signal, Sw decreases |

Additional tests for `Sequential`:

| Test | What it checks |
|---|---|
| `test_sequential_step_shapes` | `step()` returns `(mu, var)` with the correct output shape |
| `test_sequential_train_eval` | Switching modes changes BatchNorm behavior |
| `test_sequential_num_parameters` | Count matches manual calculation |

### Testing Conventions

- Framework: `pytest`
- Tensor comparisons: `torch.testing.assert_close(atol=1e-5, rtol=0)` — never bare `assert a == b`
- GPU tests: mark with `@pytest.mark.cuda`; skip automatically in CI if no GPU
- Each test is independent: no shared mutable state between tests
- Seed all random operations: `torch.manual_seed(42)` at the start of each test

### What Not to Test

- Convergence speed or final accuracy after training (non-deterministic, environment-dependent)
- Internal caches or intermediate tensor values
- The mathematical derivation of TAGI (assumed correct from the paper)
- `ruff` formatting (CI handles this separately)

---

## 9. Packaging & CI

### `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "triton-tagi"
version = "0.1.0"
description = "Bayesian neural networks via Tractable Approximate Gaussian Inference, implemented in Python/Triton"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "triton>=2.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "ruff>=0.4"]
vis = ["matplotlib>=3.7"]
validation = ["pytagi"]  # cuTAGI Python bindings

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.ruff.format]
quote-style = "double"

[tool.pytest.ini_options]
markers = ["cuda: requires a CUDA GPU"]
testpaths = ["tests"]
```

### GitHub Actions

Three jobs, all triggered on every PR:

```
lint      → ruff check + ruff format --check     (CPU, ~30s, blocks merge)
unit      → pytest tests/unit/ -m "not cuda"     (CPU, ~2min, blocks merge)
validate  → pytest tests/validation/ --gpu        (GPU runner, ~10min, advisory)
```

PRs from forks skip the `validate` job (no GPU access). Internal PRs run all three. A PR cannot be merged if `lint` or `unit` fails.

---

## 10. Run Directories, File Naming, and Reproducibility

Every training run, sweep, and benchmark produces artifacts (checkpoints, metrics, figures). These must follow a single convention so any run can be identified, resumed, and reproduced without reading the script that produced it.

### 10.1 The Run Directory

All outputs from a run go under a single directory rooted at `runs/`. The directory name encodes the four dimensions that distinguish runs:

```
runs/{dataset}_{arch}_{optimizer}_{YYYYMMDD-HHMMSS}/
```

| Component | Examples |
|---|---|
| `dataset` | `mnist`, `cifar10`, `cifar100`, `imagenet` |
| `arch` | `mlp`, `lenet`, `cnn3`, `resnet18`, `resnet18-frn`, `resnet18-sv` |
| `optimizer` | `tagi`, `adam`, `nadam`, `momentum` |
| `YYYYMMDD-HHMMSS` | `20260414-143205` |

Example: `runs/cifar10_resnet18_adam_20260414-143205/`

The timestamp makes every run unique. The prefix makes runs from the same setup sort together. No run ever silently overwrites another.

Within the run directory, the layout is fixed:

```
runs/cifar10_resnet18_adam_20260414-143205/
├── config.json           # all hyperparameters; written at run start before training
├── metrics.csv           # one row per epoch: epoch, train_loss, test_acc, sigma_v, wall_s
├── checkpoints/
│   ├── epoch_0001.pt
│   ├── epoch_0010.pt
│   └── epoch_0100.pt     # only saved at configured intervals, not every epoch
├── figures/
│   ├── training_curves.pdf
│   ├── training_curves.png
│   ├── monitor.png
│   └── ece_epoch_0100.png
└── monitor.csv           # TAGIMonitor per-epoch layer statistics
```

**Rules:**
- `config.json` is written before the first training step. A run without `config.json` is incomplete.
- `metrics.csv` is appended after each epoch (not rewritten), so a killed run has partial metrics up to the last completed epoch.
- Checkpoints are saved at a configurable interval (default: every 10 epochs) plus the final epoch. Not every epoch.
- Figures always save both `.pdf` (vector) and `.png` (raster), per the figure conventions in STYLE_GUIDE.md.
- `monitor.csv` is written by `TAGIMonitor.save_csv()` at the end of training.

### 10.2 Checkpoint Format

Every checkpoint file is a dict with a fixed schema:

```python
{
    "epoch": int,              # epoch at which this checkpoint was saved (1-indexed)
    "config": dict,            # copy of config.json contents
    "net_state": {             # keyed by layer index (int)
        0: {"mw": Tensor, "Sw": Tensor, "mb": Tensor, "Sb": Tensor},
        2: {"mw": Tensor, "Sw": Tensor, "mb": Tensor, "Sb": Tensor},
        # ... learnable layers only, by their index in Sequential.layers
    },
    "opt_state": dict | None,  # optimizer state if applicable, else None
}
```

Filename: `epoch_{epoch:04d}.pt`. No optimizer prefix. The optimizer type is already encoded in the run directory name.

### 10.3 The `config.json` Format

```json
{
    "dataset": "cifar10",
    "arch": "resnet18",
    "optimizer": "adam",
    "n_epochs": 100,
    "batch_size": 256,
    "sigma_v": 0.01,
    "gain_w": 0.1,
    "gain_b": 0.1,
    "seed": 42,
    "checkpoint_interval": 10,
    "triton_tagi_version": "0.1.0",
    "started_at": "2026-04-14T14:32:05"
}
```

All hyperparameters that affect the result go here. A run is fully reproducible from its `config.json` and the code at the commit listed in `triton_tagi_version`.

### 10.4 Sweep Results

For hyperparameter sweeps (e.g., inference-init sweep, auto-tune), the output goes in `runs/sweeps/`:

```
runs/sweeps/
└── {sweep_name}_{YYYYMMDD-HHMMSS}/
    ├── config.json        # sweep-level configuration (parameter grid, dataset, etc.)
    ├── results.json       # all trial results
    └── figures/
        ├── heatmap.pdf
        ├── heatmap.png
        └── curves.png
```

No JSON files at the project root. No figures at the project root.

### 10.5 Shared Utilities: `triton_tagi/checkpoint.py`

`save_checkpoint` and `load_checkpoint` are currently duplicated verbatim in 7+ scripts. They move to a single module:

```python
# triton_tagi/checkpoint.py

class RunDir:
    """Manages the directory structure for a single training run."""

    def __init__(self, dataset: str, arch: str, optimizer: str, base: str = "runs"):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{dataset}_{arch}_{optimizer}_{timestamp}"
        self.path = Path(base) / name
        self.checkpoints = self.path / "checkpoints"
        self.figures = self.path / "figures"
        self.config_json = self.path / "config.json"
        self.metrics_csv = self.path / "metrics.csv"
        self.monitor_csv = self.path / "monitor.csv"
        self.path.mkdir(parents=True, exist_ok=True)
        self.checkpoints.mkdir()
        self.figures.mkdir()

    def save_config(self, config: dict) -> None:
        """Write config.json. Call before the first training step."""
        ...

    def save_checkpoint(self, net: Sequential, epoch: int, config: dict,
                        opt=None) -> Path:
        """Save a checkpoint to checkpoints/epoch_{epoch:04d}.pt."""
        ...

    def load_checkpoint(self, net: Sequential, path: str | Path | None = None,
                        opt=None) -> int:
        """Load a checkpoint. If path is None, loads the latest in checkpoints/."""
        ...

    def append_metrics(self, epoch: int, **kwargs: float) -> None:
        """Append one row to metrics.csv. Creates the file with a header on first call."""
        ...
```

Every example script instantiates a `RunDir` and uses it for all I/O. No script defines its own `save_checkpoint` or `load_checkpoint`.

### 10.6 `.gitignore` Additions

```gitignore
# Generated run artifacts — never committed
runs/
data/

# Root-level generated files from old scripts — removed after migration
run_logs*/
results_*.json
*.png
*.pdf
*.csv
```

Exception: figures committed to the repository for documentation purposes live in `docs/figures/` and are explicitly tracked.

---

## 12. Decisions Log

Decisions made here are final unless updated with a reason. New decisions are appended.

| Decision | Rationale |
|---|---|
| Keep `ma/Sa/mw/Sw/mb/Sb` naming | Matches cuTAGI and the TAGI paper; changing it would make validation harder |
| One class per file for layers | Makes navigation predictable and diffs clean |
| `ruff` over `black + isort + flake8` | Single tool, faster, modern |
| `hatchling` as build backend | Simple, well-maintained, no legacy configuration |
| Validation tests skip if cuTAGI absent | Contributors without cuTAGI can still run the full test suite |
| Capital `S` for all variance tensors | Matches paper notation; `Sw` not `sw`, `Sb` not `sb` |
| `UPPER_SNAKE_CASE` for kernel constants | Triton kernels use C-style constants; keep them visually distinct |
| `runs/` as the single artifact root | Eliminates 13 ad-hoc `run_logs_*` directories; run name encodes all dimensions |
| Timestamp in run directory name | Makes every run unique without a collision risk; prefix keeps same-setup runs adjacent |
| Checkpoint interval not every epoch | Full checkpoints are large; saving every 10 epochs is sufficient for most recovery needs |
| `RunDir` class centralizes all I/O | Removes duplicated `save_checkpoint`/`load_checkpoint` from 7+ scripts |
