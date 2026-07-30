# Unified parameter-free calibration for TAGI

This note explains the intuition and the math behind `triton_tagi.calibrate`: one
projection operator **T** on the Gaussian parameter state `(μ_W, σ²_W, μ_b, σ²_b)`
of every learnable layer, applied **at init** and — online — **after every Kalman
step**. It removes the usual init/training knobs (`init_method`, `gain_w/gain_b`,
`sigma_v` and its decay schedule) and replaces them with three conditions that are
*derived*, not tuned.

The reference derivation lives in
`Tagi_julia/proof/initialization/proof_unified.jl`; this is the triton-tagi port.

---

## 0. Notation

A layer computes the pre-activation `z = aᵀ W + b`, where `a` is the input
activation. Augment the input with a constant bias feature, `ã = [a; 1]`, and
define the **input metric**

```
G = E[ã ãᵀ]          (second-moment / Gram of the augmented input)
Gᶜ = Cov(a)          (its centred block — covariance of a)
g_j = E[ã_j²]        (diagonal of G; g = 1 for the bias feature)
```

TAGI carries a **diagonal** Gaussian over every parameter and activation — it
assumes the coordinates are mutually independent. Three quantities matter:

- **Signal** `Var[z_o]` — the prior spread of output neuron `o`.
- **Per-parameter Kalman gain** `K` — how far one update moves a parameter.
- **Epistemic budget** `S_z = Σ_j g_j σ²_{W,j} + σ²_b` — the model's predictive
  variance from parameter uncertainty.

The operator T fixes all three in the *same* metric `G`.

---

## 1. The three init conditions

### (I) Signal = 1, centred  → `diag(μ_Wᵀ Gᶜ μ_W) = 1`, `E[z] = 0`

Whiten the weight means in the input metric and seed them with an orthonormal
rotation `R` (orthonormal columns):

```
μ_W = (Gᶜ)^(−1/2) · R ,     μ_b = −μ_Wᵀ E[a]
```

Then the signal of each output is exactly one:

```
signal_o = μ_W[:,o]ᵀ Gᶜ μ_W[:,o]
         = R[:,o]ᵀ (Gᶜ)^(−1/2) Gᶜ (Gᶜ)^(−1/2) R[:,o]
         = R[:,o]ᵀ R[:,o] = 1
```

**Why it matters.** This is a He/Xavier-style variance-preservation, but *exact*
and *per output*: `Var[z] = 1` at every layer, so the signal neither explodes nor
vanishes with depth. A 10-, 50-, 100-layer net is born well-scaled, and the prior
predictive variance is on the same scale as the observation noise `σ_v²` (see III).
Centring via `μ_b` keeps `E[z] = 0`.

### (II) Balanced gain  →  `K_w ≈ K_b` (uniform across all parameters)

The magnitude of a TAGI parameter update is proportional to its prior variance
times the typical activation magnitude. For weight `w_{j,o}` that scale is
`√g_j · σ²_{W,j}`; for the bias it is `1 · σ²_b`. Choose

```
σ²_{W,j} = c / √g_j ,     σ²_b = c
```

so that **every** per-parameter gain equals the *same* scalar `c`:

```
K_w = √g_j · σ²_{W,j} = c ,     K_b = σ²_b = c   ⇒   K_b / K_w = 1
```

**Why it matters.** Without this, a few high-variance parameters absorb most of
each innovation while the rest barely move — the update is lop-sided and unstable.
Balanced gains let weights and the bias *share trust* equally; no coordinate
dominates. This is the `K_b/K_w → 1` invariant in the verification table.

### (III) Calibrated budget = σ_v²  →  output Kalman gain `J = ½`

Condition (III) is a **global output** condition, not a per-layer one. The
quantity that must equal `σ_v²` is the *network output's* total epistemic variance
`S_z^final` — the variance that sits in the output Kalman gain — **not** any single
layer's local budget.

**Per-layer local budget.** With (II), one layer's *own* budget contribution is

```
S_z^(ℓ)_own = Σ_j g_j (c/√g_j) + c = c (Σ_j √g_j + 1) = c · budget_denom_ℓ
```

**Why local ≠ global.** TAGI's forward pass *accumulates* epistemic variance
through depth: a Conv/Linear layer propagates the upstream variance amplified by
`Σ_i μ_w[i,j]² = 1/v_analytic` (condition I), ReLU multiplies it by
`_VAR_A_RELU = 1/2 − 1/2π`, BatchNorm passes it through and adds its own budget,
AvgPool divides by `k²`, and a ResBlock **adds** the skip and branch budgets
(`var_a += var_s`, TAGI's diagonal merge — which double-counts the shared input,
overestimating variance). Across a deep ResNet these factors compound, so setting
each layer's *local* budget to `σ_v²` makes the output budget blow up to
`millions × σ_v²`, driving `J → 0` and killing learning from the first batch.

**The global fix.** Because the input is deterministic (`S_z = 0`) and every
propagation formula is linear in `S_z`, the output budget with one shared gain `c`
is exactly linear:

```
S_z^final(c) = c · A(net)
```

where `A(net)` is a purely architectural amplification constant — the output
variance produced by propagating a **unit budget** (`c = 1`) forward through the
network with those same analytic formulas (`_forward_sz_analytic`). The unique
global gain that calibrates the output is then closed-form:

```
c_global = σ_v² / A(net)     ⇒     S_z^final = σ_v²
```

The output **Kalman gain** that scales the innovation `(y − μ_z)` is then

```
J = S_z^final / (S_z^final + σ_v²) = σ_v² / (σ_v² + σ_v²) = ½
```

**Why ½.** The network starts *exactly half-confident*. With `J → 0` the prior is
frozen and ignores data; with `J → 1` it over-commits to the first batch. `J = ½`
is the balanced, maximal-information starting point — the Bayesian analogue of a
well-chosen learning rate, but derived rather than tuned.

> One **global** scalar `c_global` (shared by every layer) fixes (II) and (III)
> jointly; the per-layer `budget_denom_ℓ` now only sets the `1/√g_j` *shape* of each
> Sw row, not its magnitude. `A(net)` is architectural (independent of the weights
> and of `σ_v²`), so it is computed once at init and stored (`net._calib_A`,
> `OnlineCalibration.calib_A`). Online, `c_global = σ_v² / A(net)` is recomputed
> from the live `σ_v²` estimate (below), so `J = ½` keeps holding as the noise is
> learned. The per-layer local budgets `c_global · budget_denom_ℓ` therefore differ
> from layer to layer — this is correct and expected.

---

## 2. Online operator T — re-inflation as process noise `Q`

A Kalman *measurement* update only ever **shrinks** posterior variance:
`S⁺ ≤ S⁻`. Iterated over thousands of steps, `S → 0 ⇒ gain → 0`, and the layers
**freeze** — TAGI's well-known variance-collapse failure in deep or long training.

The textbook Kalman remedy is a **predict step that adds process noise** `Q`:
`S⁻ = S⁺ + Q` (a fading-memory / adaptive filter). Our re-inflation is precisely
that injection, written as a relaxation toward the calibrated prior `S_target`:

```
(S)   σ² ← (1 − λ)·σ²  +  λ·S_target ,     S_target = c_global/√g  (weights),  c_global (bias)
```

`λ = 0` → vanilla TAGI (monotone collapse). `λ → 1` → reset to the calibrated
prior (`J → ½`). `λ` is the forgetting rate, and `S_target` uses the single global
gain `c_global = σ_v²/A(net)` from (II)+(III), so re-inflation restores the OUTPUT
budget `S_z^final = σ_v²` and `J = ½`.

### Surprise-driven λ — adaptive `Q`, no schedule

Drive `λ` from the batch **normalized innovation** (NIS in Kalman terms):

```
χ² = E[ (y − μ_z)² / (Var[z] + σ_v²) ]
λ  = λ_max · (1 − exp(−(χ² − 1)₊ / τ))
```

- `χ² ≈ 1` — residuals match predicted uncertainty → data is unsurprising →
  `λ → 0`: keep converging, no forgetting.
- `χ² ≫ 1` — residuals exceed predicted uncertainty → distribution shift / new
  regime → `λ ↑`: re-inflate, **reopen the gain** ("new data ⇒ become undecided
  again").

This is the standard adaptive-process-noise idea: **increase `Q` when innovations
are too large for the current uncertainty.** No hand-tuned forgetting schedule.

### Learning the noise: TAGI-V / AGVI (the Bayesian way, not a hyperparameter)

TAGI learns the observation noise as a **latent variable**, via **TAGI-V** and
**Approximate Gaussian Variance Inference (AGVI)** — already implemented in this
repo (`examples/regression_heteros.py`, the `_output_innovation_kernel_heteros`
path). The output layer carries a second head whose value is the error variance
treated as a latent Gaussian `V`; `EvenSoftplus` (cuTAGI: `EvenExp`) keeps it
positive. AGVI then infers that latent analytically every step:

- the squared-noise prior predictive uses Gaussian moments —
  `E[V²] = μ_{V̄}`, `Var[V²] = 3·Var[V̄] + 2·μ_{V̄}²`;
- the total output variance is `Var_sum = Var[z] (epistemic) + E[V²] (aleatoric)`,
  which drives the mean update `δμ = (y − μ_z)/Var_sum`;
- a Rao–Blackwellised Gaussian update yields the posterior of `V`, then of
  `V² = V·V`, and a smoother step updates the underlying `V̄` parameter.

So `σ_v²` is **inferred**, not set, and it can be input-dependent (heteroscedastic).
With `sigma_v_mode="heteroscedastic"` the online operator defers the budget's noise
to this AGVI head.

#### AGVI extension of condition (III): initialising J = ½

For AGVI the output Kalman gain for the **mean head** is:

```
J_mean = S_z / (S_z + E[V²])
```

where `S_z` is the epistemic budget and `E[V²]` the AGVI-predicted aleatoric noise.
For `J_mean = ½` at initialisation we need:

```
S_z_init = E[V²]_init                                         (III-AGVI)
```

The calibration already sets `S_z_init = σ_v²` (condition III).  So the AGVI noise
head must also start at `E[V²]_init = σ_v²`.

With `EvenSoftplus`, the Muon centering step enforces `E[mz_odd] = 0` after every
projection.  Left alone, this gives:

```
E[V²]_init = softplus(0) ≈ log 2 ≈ 0.693   ≠   σ_v²
```

The mismatch breaks `J = ½` at init and — more critically — causes the re-inflation
feedback to diverge: re-inflating all L layers to match a budget that no longer
equals `E[V²]` amplifies the second-order correction `0.5·S_z·σ(mz)·(1−σ(mz))`
across every layer, making the recurrence factor `r = 1 − ρ + ρ·L·0.125 > 1`.

**Fix: a fixed pre-activation offset `noise_mu` inside `EvenSoftplus`.**

```
noise_mu = softplus⁻¹(σ_v²) = log(exp(σ_v²) − 1)
```

The kernel computes `effective_mz_odd = mz_odd + noise_mu` before applying
softplus.  After Muon centering restores `E[mz_odd] = 0`:

```
E[V²] = softplus(0 + noise_mu) = softplus(softplus⁻¹(σ_v²)) = σ_v²   ✓
```

The offset is a **scalar constant**, not a trainable parameter.  It lives inside
`EvenSoftplus`, is invisible to the weight update and to Muon, and requires no
per-step correction.  Usage:

```python
EvenSoftplus(half_width=K, noise_mu=math.log(math.expm1(sigma2_obs)))
```

> **Gradient tradeoff.**  The EvenSoftplus Jacobian for the noise head is
> `J_bp = sigmoid(mz_odd + noise_mu)`.  With `E[mz_odd] = 0`:
> `J_bp ≈ sigmoid(softplus⁻¹(σ_v²)) ≈ 1 − e^{−σ_v²} ≈ σ_v²` for small `σ_v²`.
> Small `σ_v²` (e.g. 0.001) gives `J_bp ≈ 0.001` — the noise head learns 500× slower
> than the mean head.  This is expected: a tight prior (`σ_v²` small) means the
> network already "knows" the noise is small and only fine-tunes it.  For fast noise
> adaptation choose `σ_v² ≈ log 2` (natural EvenSoftplus scale, `J_bp = 0.5`).

> **Online re-inflation for AGVI.**  `cfg.sigma_v2` must track the **same**
> σ_v² that sits in the Kalman gain denominator:
> `var_sum = var_a_col + mu_v2_bar_tilde` (observation.py).
> `mu_v2_bar_tilde` is the batch-mean of the odd output columns after EvenSoftplus
> (`y_pred_mu[..., 1::2]`), i.e. the actual `E[V²]` the AGVI kernel consumed.
> `update_sigma_v2` maintains a slow EMA of that value so the global re-inflation
> target `c_global = cfg.sigma_v2 / A(net)` stays aligned with the live noise
> prediction and `J = ½` is preserved as the AGVI head learns.  The `noise_mu` offset is still
> required at init (so that `E[V²]_init = σ_v²_seed` before the EMA has warmed up).

> **Homoscedastic fallback** (`sigma_v_mode="homoscedastic"`, used by the CIFAR
> example) — for nets *without* a V2 head. A method-of-moments running estimate
> `σ̂_v² ← (1 − ρ)·σ̂_v² + ρ·max(E[(y − μ)²] − E[Var_z], floor)` (total residual minus
> epistemic). This is a convenience heuristic, **not AGVI**; the global budget target
> `c_global = σ̂_v²/A(net)` tracks it so the OUTPUT `J = ½` is preserved.

---

## 3. Muon mean projection — making TAGI's independence assumption *true*

TAGI propagates only **diagonal** covariances: it assumes output neurons are
mutually independent, `Cov(z_o, z_{o'}) ≈ 0` for `o ≠ o'`. At init this is *exactly*
true, because orthonormal `R` decorrelates the outputs in the metric:

```
Cov(z_o, z_{o'}) = μ_W[:,o]ᵀ Gᶜ μ_W[:,o'] = R[:,o]ᵀ R[:,o'] = δ_{o,o'}
```

But Kalman updates push `μ_W` **off** the `Gᶜ`-orthonormal manifold: the columns
acquire correlations, output neurons become statistically dependent, and the
diagonal approximation that TAGI's moment propagation rests on starts to bias both
the mean and variance forward pass.

The Muon step periodically re-orthogonalizes the means. Take the signal block
`B = (Gᶜ)^(1/2) μ_W` and find its nearest column-orthonormal matrix `Π(B)` via the
**exact polar factor `U Vᵀ` from an SVD** (approximations such as Newton–Schulz are
intentionally not used — exactness keeps `signal = 1` and decorrelation precise),
map back to weight space, and relax toward it:

```
(μ)   μ_W ← (1 − β)·μ_W  +  β·(Gᶜ)^(−1/2) Π((Gᶜ)^(1/2) μ_W) ,    μ_b ← −μ_Wᵀ E[a]
```

This simultaneously (a) preserves `signal = 1` and (b) **re-decorrelates the
outputs**. So the projection is not a generic optimizer trick: it actively
restores the independence assumption that makes TAGI's diagonal inference valid.

---

## 4. The metric: analytic vs data

The same operator T runs in two regimes that differ only in how `Gᶜ` is obtained:

| metric        | `Gᶜ`                              | needs data | scope            |
|---------------|-----------------------------------|------------|------------------|
| `"analytic"`  | `v·I` (ReLU Gaussian fixed point) | no         | all archs (incl. conv) |
| `"data"`      | measured `Cov(a)` of real samples | yes        | dense (Linear) nets |

- **analytic** uses the closed-form ReLU moments `v = Var[ReLU(N(0,V))]`, so
  `(Gᶜ)^(−1/2) = v^(−1/2)·I` is a scalar and `g_j = E[ReLU²]` is constant — exactly
  the He-style fixed point. Works for every layer, including convolutions.
- **data** propagates a real calibration batch through the (calibrated) means
  layer-by-layer, measures `Ea = E[a]`, `g = E[a²]`, and the centred Gram
  `Gᶜ = Cov(a)`, and whitens with `(Gᶜ)^(−1/2)` (a full matrix). Signal = 1 then
  holds on the **true, correlated** input distribution rather than the iid-Gaussian
  assumption. Currently dense-only (the conv case would need a patch/im2col Gram).

```python
# analytic (any architecture)
calibrate(net, sigma2_obs=0.01, metric="analytic")

# data-driven whitening (dense / MLP), pass a calibration batch
calibrate(net, sigma2_obs=0.01, metric="data", data_batch=x_calib)
```

---

## 5. How to use it end-to-end

```python
from triton_tagi import calibrate, OnlineCalibration

calibrate(net, sigma2_obs=0.01, metric="analytic")          # init operator T (λ = 1)

online = OnlineCalibration(
    mode="surprise",          # λ_t from the batch χ² (or "const" / "off")
    project=True,             # Muon mean projection (exact SVD polar factor)
    sigma_v_mode="heteroscedastic",  # TAGI-V / AGVI head learns σ_v² (needs a 2·K
                                     # output head + EvenSoftplus); use
                                     # "homoscedastic" only for nets without one
)
for xb, yb in batches:
    net.step(xb, yb, sigma_v=0.0, online=online)            # online operator T after each update
```

A head-to-head on a deep net (same ResNet-18 backbone, TAGI-V noise in both arms,
only the calibration differs) is in `examples/compare_calibration_resnet18.py`.

Invariants you can check (see `tests/validation/test_calibrate.py`):

- `signal_o = v · Σ_i μ_W[i,o]² ≈ 1` per output (condition I)
- `AW + B ≈ σ_v²` per layer ⇒ `J = ½` (condition III)
- `K_b / K_w ≈ 1`, gains uniform across parameters (condition II)
- online re-inflation lifts a collapsed `σ²` back toward its calibrated target
- polar projection returns column-orthonormal means (signal preserved)
