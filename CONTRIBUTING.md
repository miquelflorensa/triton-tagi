# Contributing to triton-tagi

## Development setup

**Prerequisites:** Python ≥ 3.10, CUDA-capable GPU, PyTorch ≥ 2.0 with CUDA.

```bash
git clone https://github.com/miquelflorensa/triton-tagi.git
cd triton-tagi

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Optional: install matplotlib for examples that produce figures
pip install -e ".[dev,vis]"
```

Verify the installation:

```bash
python -c "import triton_tagi; print('ok')"
```

---

## Running tests

```bash
# Unit tests — no GPU or cuTAGI required
pytest tests/unit/

# Validation tests — requires a CUDA GPU and pytagi (cuTAGI Python bindings)
pytest tests/validation/

# Both together
pytest tests/
```

Validation tests skip automatically if `pytagi` is not installed — they will never
block a contributor who only has triton-tagi.

To run only tests that do not touch the GPU (useful in CI without a GPU runner):

```bash
pytest tests/unit/ -m "not cuda"
```

### Linting and formatting

```bash
ruff format triton_tagi/ tests/
ruff check triton_tagi/ tests/
```

Both must pass before opening a pull request.  Run them together with:

```bash
ruff format triton_tagi/ tests/ && ruff check triton_tagi/ tests/
```

---

## Adding a new layer

Every layer is a subclass of either `Layer` (activation layers, pooling) or
`LearnableLayer` (layers with trainable parameters).  Both ABCs live in
`triton_tagi/base.py`.

```
Layer               — forward(ma, Sa) → (ma, Sa)
                    — backward(delta_ma, delta_Sa) → (delta_ma, delta_Sa)

LearnableLayer(Layer)
                    — forward / backward (same contract as Layer)
                    — update(cap_factor) → None
                    — num_parameters: int  (property)
```

A worked example implementing an activation layer from scratch is in
`examples/custom_layer.py`.  Follow this sequence:

### 1. Derive the moment equations

For a pointwise function f applied to z ~ N(μ_z, S_z):

```
μ_a ≈ f(μ_z)  +  ½ · f''(μ_z) · S_z          (second-order Taylor)
J   = f'(μ_z)                                   (Jacobian at mean)
S_a = J² · S_z
```

The Jacobian is also needed during backward:

```
δ_μ_z = J · δ_μ_a
δ_S_z = J² · δ_S_a
```

### 2. Create the file

One class per file; file name is the class name in lowercase snake_case.

```
triton_tagi/layers/my_activation.py
```

Inherit from `Layer`:

```python
from ..base import Layer

class MyActivation(Layer):
    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]: ...
    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]: ...
```

Export the class from `triton_tagi/layers/__init__.py`.

### 3. Write unit tests first

Minimum test set in `tests/unit/test_my_activation.py`:

| Test | What it checks |
|---|---|
| `test_forward_shape` | Output shape matches input shape |
| `test_forward_zero_Sa` | Sa=0 input gives Sa=0 output; μ_a = f(μ_z) |
| `test_backward_shape` | Delta output shape matches delta input shape |
| `test_backward_zero_delta` | Zero deltas propagate as zero deltas |
| `test_Sa_nonneg` | S_a ≥ 0 for all inputs |

All tensor comparisons use `torch.testing.assert_close(atol=1e-5, rtol=0)`.
GPU tests are marked `@pytest.mark.cuda`.

### 4. Write the Triton kernel

Fuse forward and backward into a single kernel when both branches share the
same intermediate values (e.g., the Jacobian).  See `relu.py` for the pattern.

Key conventions:
- Kernel name: `_<layer>_kernel`
- Wrapper: `_triton_<layer>(ma, Sa) -> (ma_out, Sa_out, J)` returns the Jacobian
  for use in `backward`.
- Block size: constexpr `BLOCK` defaulting to 1024 (tuned separately if needed).

### 5. Add a validation test (optional but recommended)

If the activation exists in cuTAGI, add a validation test in
`tests/validation/test_my_activation.py` that compares forward and backward
outputs at tolerance `atol=1e-4, rtol=0`.

---

## Adding a learnable layer

A learnable layer additionally stores parameter tensors (`mw`, `Sw`, `mb`, `Sb`),
computes parameter deltas during `backward`, and applies them via `update`.

The weight update convention (cuTAGI-style):

```
Δμ_w = S_w · (∂L/∂μ_w)      Δμ_b = S_b · (∂L/∂μ_b)
ΔS_w = S_w² · (∂L/∂S_w)     ΔS_b = S_b² · (∂L/∂S_b)
```

where the raw gradients are:

```
∂L/∂μ_w = μ_a^T @ δ_μ_z          ∂L/∂μ_b = Σ_batch δ_μ_z
∂L/∂S_w = (μ_a²)^T @ δ_S_z       ∂L/∂S_b = Σ_batch δ_S_z
```

Call `update_parameters` from `triton_tagi/update/parameters.py` to apply the
capped update — do not re-implement the capping logic.

---

## Tensor naming

These names are fixed throughout the library:

| Symbol | Meaning |
|---|---|
| `ma`, `Sa` | activation mean / variance |
| `mw`, `Sw` | weight mean / variance |
| `mb`, `Sb` | bias mean / variance |
| `delta_ma`, `delta_Sa` | backward delta on activation mean / variance |
| `delta_mw`, `delta_Sw` | stored weight mean / variance delta |

Capital `S` denotes variance (σ²), never standard deviation.

---

## Code style

- Formatter: `ruff format` (line length 100)
- Linter: `ruff check` with rules `E, F, I, UP, B`
- No commented-out code in committed files
- No `print()` in library code; use `logging.getLogger(__name__)` if needed
- One public class per file

See `STYLE_GUIDE.md` for figure and run-directory conventions.
