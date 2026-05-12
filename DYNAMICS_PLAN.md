# DYNAMICS_PLAN — TAGI Update Dynamics and Observation Models

## Goal

Build a principled TAGI training framework that separates three issues currently
entangled in practice:

1. **Mini-batch parameter assimilation**
   - Replace cap-factor clipping with a Bayesian precision-space update.

2. **Observation noise in regression**
   - Use TAGI-V / AGVI to infer heteroscedastic aleatory uncertainty instead of
     manually tuning `sigma_v`.

3. **Observation model in classification**
   - Replace Gaussian pseudo-observations with a categorical Remax likelihood,
     avoiding manual `sigma_v` for classification.

The unifying object is the posterior information ratio

```text
chi_p = S_p Lambda_p = -Delta_S_p / S_p
```

where `chi_p` measures how much prior variance a batch tries to consume for
parameter `theta_p`.

---

## Part A — Precision-Space Parameter Update

### A.1 Current capped update

```text
delta_bar = sqrt(S) / cap_factor

m_new = m + sign(dm) * min(|dm|, delta_bar)
S_new = S + sign(dS) * min(|dS|, delta_bar)
```

This is stable but heuristic. The cap factor is not derived from the posterior.

### A.2 New precision-space update

Given TAGI backward deltas:

```text
dm = S eta
dS = -S^2 Lambda
```

the exact local Gaussian batch posterior is:

```text
info = max(-dS / S, 0)
d    = 1 + rho * info

m_new = m + rho * dm / d
S_new = S / d
```

Important invariant:

```text
S_new <= S
S_new > 0
```

Positive `dS` is treated as zero information, not as variance inflation. If
variance inflation is desired, it should be added explicitly as process noise.

### A.3 Tempering policy

`rho` must not become a new arbitrary cap factor. Provide deterministic modes:

```text
rho_mode = "full"       -> rho = 1
rho_mode = "sqrt_batch" -> rho = 1 / sqrt(B)
rho_mode = "batch_avg"  -> rho = 1 / B
rho_mode = "custom"     -> rho = user value
```

Recommended experimental order:

```text
1. full
2. sqrt_batch
3. batch_avg only if full/sqrt_batch are unstable
```

---

## Phase 1 — Precision Kernel

**File:** `triton_tagi/update/parameters.py`

Add a precision-space update kernel after the capped kernel.

```python
@triton.jit
def _precision_param_update_kernel(
    m_ptr,
    S_ptr,
    dm_ptr,
    dS_ptr,
    rho,
    variance_floor,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    m = tl.load(m_ptr + offs, mask=valid)
    S = tl.load(S_ptr + offs, mask=valid)
    dm = tl.load(dm_ptr + offs, mask=valid)
    dS = tl.load(dS_ptr + offs, mask=valid)

    S_safe = tl.maximum(S, variance_floor)

    # dS should be <= 0 for a Bayesian variance update.
    # Positive dS is treated as zero information.
    info = tl.maximum(-dS / S_safe, 0.0)
    d = 1.0 + rho * info

    m_new = m + rho * dm / d
    S_new = S_safe / d
    S_new = tl.maximum(S_new, variance_floor)

    tl.store(m_ptr + offs, m_new, mask=valid)
    tl.store(S_ptr + offs, S_new, mask=valid)
```

Python wrapper:

```python
def update_parameters_precision(
    m,
    S,
    delta_m,
    delta_S,
    rho: float = 1.0,
    variance_floor: float = 1e-10,
):
    """In-place precision-space TAGI parameter update."""
    n = m.numel()
    _precision_param_update_kernel[(triton.cdiv(n, BLOCK),)](
        m.view(-1),
        S.view(-1),
        delta_m.view(-1),
        delta_S.view(-1),
        rho,
        variance_floor,
        n,
        BLOCK=BLOCK,
    )
```

Do not modify:

```text
update_parameters
get_cap_factor
```

They are required for cuTAGI-parity and regression guards.

---

## Phase 2 — Update Routing in `Sequential`

Prefer centralized routing first. Do not add an abstract method to every
`LearnableLayer` until the API stabilizes.

**File:** `triton_tagi/network.py`

Add constructor arguments:

```python
def __init__(
    self,
    layers: list,
    device: str = "cuda",
    update_mode: str = "cap",      # "cap" | "precision"
    rho_mode: str = "full",        # "full" | "sqrt_batch" | "batch_avg" | "custom"
    rho: float = 1.0,
) -> None:
    self.update_mode = update_mode
    self.rho_mode = rho_mode
    self.rho = rho
```

Add helper:

```python
def _get_rho(self, batch_size: int) -> float:
    if self.rho_mode == "full":
        return 1.0
    if self.rho_mode == "sqrt_batch":
        return batch_size ** -0.5
    if self.rho_mode == "batch_avg":
        return batch_size ** -1
    if self.rho_mode == "custom":
        return self.rho
    raise ValueError(f"Unknown rho_mode: {self.rho_mode}")
```

Add centralized update helper:

```python
def _apply_param_update(self, batch_size: int) -> None:
    if self.update_mode == "cap":
        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                layer.update(cap_factor)
        return

    if self.update_mode == "precision":
        rho = self._get_rho(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                self._update_precision_layer(layer, rho)
        return

    raise ValueError(f"Unknown update_mode: {self.update_mode}")
```

Handle composite layers centrally:

```python
def _update_precision_layer(self, layer: LearnableLayer, rho: float) -> None:
    if isinstance(layer, ResBlock):
        for sub in layer._learnable:
            self._update_precision_layer(sub, rho)
        return

    if isinstance(layer, MultiheadAttentionV2):
        self._update_precision_layer(layer.q_proj, rho)
        self._update_precision_layer(layer.k_proj, rho)
        self._update_precision_layer(layer.v_proj, rho)
        return

    update_parameters_precision(layer.mw, layer.Sw, layer.delta_mw, layer.delta_Sw, rho)

    if getattr(layer, "has_bias", False):
        update_parameters_precision(layer.mb, layer.Sb, layer.delta_mb, layer.delta_Sb, rho)
```

Replace update blocks in `step` and `step_hrc`:

```python
self._apply_param_update(batch_size)
```

Default behavior must remain:

```python
Sequential(layers)
# equivalent to update_mode="cap"
```

---

## Phase 3 — Example Wiring

Update ResNet and MLP examples with:

```text
--update_mode {cap,precision}
--rho_mode {full,sqrt_batch,batch_avg,custom}
--rho FLOAT
```

Examples:

```bash
python examples/cifar10_resnet18.py \
  --update_mode precision \
  --rho_mode full \
  --sigma_v 0.07
```

```bash
python examples/cifar10_resnet18.py \
  --update_mode precision \
  --rho_mode sqrt_batch \
  --sigma_v 0.01
```

The old behavior remains:

```bash
python examples/cifar10_resnet18.py --update_mode cap --sigma_v 0.01
```

---

## Part B — Regression Observation Noise: TAGI-V / AGVI

For regression, the Gaussian observation model is meaningful:

```text
Y = Z^(O) + V
V ~ N(0, Sigma_V(x))
```

Manual `sigma_v` should be avoided through TAGI-V / AGVI, where the network
predicts a variance head:

```text
overline{V^2}(x)
```

and infers the aleatory noise moments analytically.

Theory statement:

```text
Regression:
    Use TAGI-V / AGVI to infer Sigma_V(x).

Classification:
    Do not use AGVI on one-hot residuals. Use a discrete observation model.
```

Implementation note:

```text
Do not change TAGI-V code unless update_mode routing breaks it.
The parameter update mode should be orthogonal to the heteroscedastic output update.
```

Add to theory document:

```text
Regression-side solution to sigma_v:
    TAGI-V / AGVI

Classification-side solution to sigma_v:
    categorical Remax or HRC/probit likelihood
```

---

## Part C — Classification Observation Model: Categorical Remax

### C.1 Problem with current Gaussian pseudo-observation

Current Remax training still behaves like:

```text
A = Remax(Z)
Y = A + V
V ~ N(0, sigma_v^2 I)
```

So `sigma_v` is still required.

Setting `sigma_v = 0` is not correct. It means the one-hot vector is observed as
the exact value of the probability vector.

### C.2 Categorical Remax model

Use:

```text
C | A ~ Categorical(A)
Y = e_C
```

Then:

```text
E[Y | A] = A
Var(Y | A) = diag(A) - A A^T
```

Moment-matched approximation:

```text
mu_Y = mu_A
Sigma_Y ≈ diag(mu_A) - mu_A mu_A^T
```

If including epistemic uncertainty in `A`:

```text
Sigma_Y ≈ Sigma_A + diag(mu_A) - mu_A mu_A^T
```

Diagonal first implementation:

```text
sigma2_Yi ≈ S_Ai + mu_Ai * (1 - mu_Ai)
delta_mu_i = (y_i - mu_Ai) / sigma2_Yi
delta_S_i  = -1 / sigma2_Yi
```

This is an experimental diagonal approximation. It discards the negative
off-diagonal categorical covariance terms, so it should not be claimed as the
full categorical update.

Targets are one-hot:

```text
y_i in {0, 1}
```

not `±1`.

---

## Phase 4 — Experimental Categorical-Remax Innovation

**File:** `triton_tagi/update/observation.py`

Add:

```python
def compute_innovation_remax_cat(y, y_pred_mu, y_pred_var, eps: float = 1e-10):
    """
    Diagonal categorical-Remax innovation.

    Uses:
        sigma2_Yi = S_Ai + mu_Ai * (1 - mu_Ai)

    No sigma_v parameter.
    """
```

Kernel formula:

```python
mu = clamp(y_pred_mu, eps, 1.0 - eps)
sigma2_y = y_pred_var + mu * (1.0 - mu)
sigma2_y = max(sigma2_y, eps)

delta_mu = (y - mu) / sigma2_y
delta_S = -1 / sigma2_y
```

Add a new step method:

```python
def step_remax_cat(self, x_batch: Tensor, y_batch: Tensor):
    batch_size = x_batch.shape[0]

    y_pred_mu, y_pred_var = self.forward(x_batch)
    delta_mu, delta_var = compute_innovation_remax_cat(
        y_batch,
        y_pred_mu,
        y_pred_var,
    )

    for layer in reversed(self.layers):
        delta_mu, delta_var = layer.backward(delta_mu, delta_var)

    self._apply_param_update(batch_size)
    return y_pred_mu, y_pred_var
```

Recommended first experiment:

```bash
python examples/cifar10_resnet18.py \
  --classification_observation remax_cat \
  --update_mode precision \
  --rho_mode sqrt_batch
```

---

## Phase 5 — Exports

**File:** `triton_tagi/update/__init__.py`

Export:

```python
update_parameters_precision
compute_innovation_remax_cat
```

Top-level exports are optional. Prefer keeping experimental functions under
`triton_tagi.update` until stable.

---

## Phase 6 — Tests

### CPU formula tests

These should not call Triton.

| Test | Purpose |
|---|---|
| `test_precision_formula_exact` | Checks scalar formula for `m_new`, `S_new` |
| `test_precision_positive_dS_ignored` | Positive `dS` is treated as zero information |
| `test_rho_modes` | `full`, `sqrt_batch`, `batch_avg`, `custom` return expected values |
| `test_remax_cat_formula_shapes` | Diagonal formula returns correct shape |
| `test_remax_cat_onehot_targets` | Reject or document non-one-hot targets |

### CUDA kernel tests

Mark with `@pytest.mark.cuda`.

| Test | Purpose |
|---|---|
| `test_precision_kernel_matches_cpu` | Triton precision kernel equals CPU formula |
| `test_remax_cat_kernel_matches_cpu` | Triton categorical innovation equals CPU formula |
| `test_sequential_precision_no_nan` | One small train step leaves parameters finite |

### Regression guards

Must pass before merge:

```bash
pytest tests/unit
pytest tests/validation/test_linear.py
pytest tests/validation/test_conv2d.py
pytest tests/validation/test_batchnorm.py
```

The default `update_mode="cap"` must preserve cuTAGI parity.

---

## Phase 7 — Experiments

### 7.1 Precision update sweep

Run:

```text
update_mode = cap, precision
rho_mode = full, sqrt_batch, batch_avg
sigma_v = 0.01, 0.03, 0.05, 0.07
```

For:

```text
MNIST MLP depth sweep
CIFAR-10 ResNet-18 Remax
CIFAR-10 ResNet-18 HRC
```

Primary metrics:

```text
accuracy
NaN/collapse rate
median S_w per layer
information ratio chi per layer
posterior KL per layer
```

### 7.2 Categorical Remax sweep

Compare:

```text
Gaussian Remax + sigma_v
Categorical Remax diagonal
```

with:

```text
update_mode = cap
update_mode = precision + sqrt_batch
```

Watch for:

```text
early collapse
overconfidence
plateau accuracy
variance collapse in final layer
```

---

## Key Invariants

- Do not remove or modify `update_parameters`.
- Do not remove or modify `get_cap_factor`.
- Default `Sequential(layers)` must behave exactly as current cuTAGI-parity mode.
- Backward delta computation is unchanged.
- Precision update changes only the final parameter assimilation rule.
- TAGI-V / AGVI remains the regression answer to observation noise.
- Categorical Remax is experimental and should be labeled as a diagonal approximation.