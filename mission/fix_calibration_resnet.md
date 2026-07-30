# Mission: Fix condition (III) for ResNet architectures

You are working on `triton-tagi`, a Triton/PyTorch implementation of TAGI (Tractable Approximate Gaussian Inference by James-A. Goulet). This is serious Bayesian deep learning research — the calibration system you are about to fix is the mathematical core that makes parameter-free training possible. There is no room for approximations, heuristics, or shortcuts. Everything asked of you is provably tractable in closed form. If you find yourself reaching for "approximately" or "we can assume", stop and rederive. The math is clean; trust it.

---

## The Bug: condition (III) is local but TAGI's forward pass is global

### What condition (III) currently does

`calibrate()` in `triton_tagi/calibrate.py:368` processes each learnable layer independently. For every dense/conv block (`calibrate_dense_block`, line 267) it computes a **per-layer** scalar `c`:

```python
# calibrate.py:300-301
budget_denom = sqrt_g.sum().item() + 1.0   # Σ_i √g_i + 1
c = sigma2 / budget_denom                   # condition (III): local budget = σ_v²
```

This sets the **local** epistemic variance of that layer's output:

```
S_z^(ℓ)_own = c · (Σ_i √g_i + 1) = σ_v²
```

The goal is Kalman gain `J = S_z / (S_z + σ_v²) = ½` at the final output. This requires the **final output's total epistemic variance** to equal `σ_v²`. But TAGI's forward pass ACCUMULATES epistemic variance from every layer — the final `Sz` is not the local contribution of the last layer alone.

### How TAGI propagates epistemic variance forward

**Conv/Linear layer** (`kernels/common.py` — `triton_fused_var_forward`):

```
Sz_out_j = Σ_i μ_w[i,j]² · Sa_in[i]  +  Σ_i g_i · Sw[i,j]  +  Sb_j
           ↑_________________________↑    ↑__________________________↑
             propagated upstream Sz         this layer's own budget
```

With calibrated `μ_w` where `Σ_i μ_w[i,j]² = 1/v_analytic` (from condition I), and uniform `Sa_in = V`:

```
Sz_out = V · (1/v_analytic)  +  c · budget_denom
```

The propagated term is **amplified by `1/v_analytic ≈ 2.93`** (where `v_analytic = Vz · _VAR_A_RELU ≈ 0.3408 · Vz`).

**ReLU** (`layers/relu.py`):

```
Sa_out = v_analytic · Sz_in      (Var[ReLU(N(0,Sz))] ≈ v · Sz)
```

This **divides by `1/v`**, cancelling the conv amplification exactly for a Conv-ReLU pair → net factor ≈ 1 per Conv-ReLU.

**BatchNorm2D** (`layers/batchnorm2d.py` forward kernel, lines ~111-120):

```
Ŝ = Sz_in / (run_s + ε)          # run_s ≈ 1 at the calibrated mean fixed-point
Sa_out = μ_γ² · Ŝ + Sγ · (μ̂² + Ŝ) + Sβ
       ≈ Sz_in + Sγ · Sz_in + sigma2_obs   (with μ_γ=1, Sγ=Sβ=σ_v²/2, μ̂≈0, run_s≈1)
       ≈ (1 + σ_v²/2) · Sz_in + σ_v²
```

BN **does not normalize the epistemic variance** — it passes it through (run_s ≈ 1 when the mean path is unit-variance, which calibration ensures) and adds its own `σ_v²` contribution.

**ResBlock addition** (`layers/resblock.py:382`, kernel line 88):

```python
# resblock.py — triton_add_shortcut kernel, line 88
var_a += var_s      # Sz_out = Sz_branch + Sz_skip
```

The skip and branch are added as if **independent**. This is TAGI's diagonal approximation. It is a deliberate modeling choice for the inference engine and must NOT be changed. However, since the skip path and branch path share the same input `x`, this addition **double-counts** the upstream epistemic variance:

```
True Var[f(x) + x] = Var[f(x)] + Var[x] + 2 Cov(f(x), x)
TAGI computes:       Var[f(x)] + Var[x]                     ← drops the covariance term
```

The dropped `2 Cov(f(x), x)` is **positive** (branch and skip are correlated), so TAGI overestimates variance at every ResBlock. For an identity ResBlock where the branch preserves variance (net factor ≈ 1 from Conv-ReLU-BN-Conv-ReLU-BN), TAGI gives:

```
Sz_out ≈ Sz_skip + Sz_branch ≈ Sz_in + Sz_in = 2 · Sz_in
```

For a projection ResBlock (skip goes through 1×1 Conv-BN without ReLU → net factor `1/v_analytic ≈ 2.93`):

```
Sz_out ≈ (1/v) · Sz_in + Sz_in + 2 · σ_v² ≈ (1 + 1/v) · Sz_in ≈ 3.93 · Sz_in
```

### The exponential blow-up in ResNet-18

The CIFAR-10 ResNet-18 in `examples/cifar10_resnet18_calibrated.py:104` has:

```
Conv-ReLU-BN → [ResBlock(64,64)×2] → ResBlock(64→128) → [ResBlock(128,128)×2]
            → ResBlock(128→256) → [ResBlock(256,256)×2] → ResBlock(256→512)
            → [ResBlock(512,512)×2] → AvgPool → Flatten → Linear(512,20) → EvenSoftplus
```

Starting from `Sz_0 ≈ σ_v²` after the first Conv, the amplification through 8 ResBlocks compounds:

| Stage | Factor | Cumulative Sz / σ_v² |
|-------|--------|----------------------|
| Initial Conv-ReLU-BN | 1 | ≈ 1 |
| ×2 identity blocks (64→64) | 2² = 4 | ≈ 4 |
| Projection block (64→128) | ≈ 3.93 | ≈ 16 |
| ×2 identity (128→128) | 4 | ≈ 64 |
| Projection (128→256) | ≈ 3.93 | ≈ 252 |
| ×2 identity (256→256) | 4 | ≈ 1000 |
| Projection (256→512) | ≈ 3.93 | ≈ 3930 |
| ×2 identity (512→512) | 4 | ≈ 15700 |
| AvgPool(4): divide by 16 | 1/16 | ≈ 980 |
| Linear(512,20): amplify by 1/v | ≈ 2.93 | ≈ 2870 |

So `Sz_odd` at the `EvenSoftplus` input is approximately **2870 × σ_v²**. The EvenSoftplus second-order correction (`layers/even_softplus.py:74`):

```python
mu_sp = softplus_base + 0.5 * Sz * sig * (1.0 - sig)
      ≈ σ_v²   +   0.5 · 2870·σ_v² · 0.25        (at mz≈0, sig≈0.5)
      ≈ σ_v² · (1 + 358.75)
      ≈ 359 · σ_v²
```

With `σ_v² = log(2) = 0.693`:

```
E[V²] ≈ 359 · 0.693 ≈ 249     →    sqrt(E[V²]) ≈ 15.8
```

The observed `σ_v(AGVI) = 2826` at epoch 1 confirms the actual blow-up exceeds even this estimate (inter-block BN `Sγ·Sz` terms and the projection Conv amplifications compound further). Whatever the exact figure, **the final `Sz` is millions of times larger than `σ_v²`**, making `J ≈ 0` and killing learning from the first batch.

---

## The Fix: a two-pass calibration with a global `c`

### Why it is tractable in closed form

The key insight: the relationship between the per-layer scalar `c` and the final output epistemic variance `Sz_final` is **linear in `c`**:

```
Sz_final(c) = c · A(net)
```

where `A(net)` is a purely architectural constant — the total amplification factor computed by propagating a unit-budget variance forward through the network using the same analytic formulas TAGI uses at runtime. This linearity holds because every variance-propagation formula in TAGI is linear in `Sz` and hence linear in `c`.

Therefore:

```
c_correct = σ_v² / A(net)
```

This is the **unique global scalar** that makes `Sz_final = σ_v²` exactly, giving `J = ½` at the output.

### The two-pass algorithm

**Pass 1 — set means (already correct, no change needed):**

For each layer in `_iter_learnable(net.layers)`, call the existing mean-setting code (condition I). Also collect the per-layer `budget_denom` and layer metadata.

**Pass 2 — forward Sz sensitivity pass with unit budget (c = 1):**

Walk `net.layers` in order (not the flattened `_iter_learnable` list, but the structured list that includes `ResBlock` objects). For each layer, propagate a running `Sz_running` forward using the **same formulas TAGI uses at inference time**:

| Layer type | Forward Sz rule |
|------------|-----------------|
| `Conv2D` / `Linear` | `Sz_out = Sz_in · (1/v_analytic) + budget_denom_ℓ` (unit c = 1) |
| `ReLU` | `Sz_out = Sz_in · v_analytic` |
| `BatchNorm2D` | `Sz_out = Sz_in + budget_denom_ℓ` (run_s = 1 at fixed point, own contrib = 2.0 · 1 = 2) |
| `AvgPool2D` | `Sz_out = Sz_in / (kernel² · H_out · W_out)` — use actual spatial dims from the layer |
| `Flatten` | identity |
| `ResBlock` | Recurse: propagate through main path and skip/proj path separately; **add** their Sz at the end (mirrors `var_a += var_s` in the kernel) |
| `EvenSoftplus` | identity (not a weight layer, no budget) |
| `Remax` | identity |

For `AvgPool2D`, the spatial reduction at the analytic level uses the fixed-point spatial size (for CIFAR-10, the feature map is 4×4 before AvgPool(4), giving factor 16). Since the spatial dimensions are known from the network structure, this is fully computable without a data forward pass.

At the end of this pass: `A(net) = Sz_running` at the final output.

**Compute global c:**

```
c_global = σ_v² / A(net)
```

**Pass 3 — set all Sw with corrected c:**

For each calibrated layer, update its `Sw` and `Sb`:

```python
# For dense/conv layers:
reinflate_weight_(layer.Sw, meta["sqrt_g"], c_global, lam=1.0)
reinflate_const_(layer.Sb, c_global, lam=1.0)   # if has_bias

# For BatchNorm layers:
reinflate_const_(layer.Sw, c_global, lam=1.0)   # Sγ = c_global
reinflate_const_(layer.Sb, c_global, lam=1.0)   # Sβ = c_global
```

Also update `meta["c"]` and `meta["budget_denom"]` so that `online_recalibrate` uses the correct re-inflation targets.

### What the online operator must track

After this fix, every layer shares the same global `c_global`. The online re-inflation in `_reinflate_layer` (`calibrate.py:556`) computes `c = sigma_v2 / meta["budget_denom"]` per layer. This must be replaced with the **same global formula**: `c_global = sigma_v2 / A(net_at_current_sigma_v2)`.

Since `A(net)` depends on the architecture (not the current weights or σ_v²), it is a constant after init. Store `A` on the `OnlineCalibration` object. Then online:

```
c_online = cfg.sigma_v2 / A_global
```

The `budget_denom` stored per-layer in `meta` is no longer used as the denominator — it only determines the **shape** of the Sw tensor (the `1/sqrt(g_j)` profile). The magnitude is set entirely by `c_online`.

---

## Files to modify

1. **`triton_tagi/calibrate.py`** — the main change:
   - Add a `_forward_sz_analytic(net, sigma2_obs, input_var)` function that walks `net.layers` (structured, not flattened) and returns `A(net)` = total epistemic amplification with unit budget
   - Modify `calibrate()` (line 368) to: (a) run the existing mean-setting pass, (b) call `_forward_sz_analytic`, (c) recompute and write all Sw/Sb with `c_global = sigma2_obs / A`
   - Store `A` in the returned metadata and attach it to the network (e.g. `net._calib_A = A`)
   - Modify `_reinflate_layer()` (line 556) to accept and use `c_global` directly instead of computing `c = sigma_v2 / meta["budget_denom"]`
   - Modify `online_recalibrate()` (line 694) to compute `c_global = cfg.sigma_v2 / cfg.calib_A` and pass it to `_reinflate_layer`
   - Add `calib_A: float = 1.0` field to `OnlineCalibration` dataclass

2. **`triton_tagi/layers/resblock.py`** — no change to the forward/backward. The variance addition `var_a += var_s` stays. Only the init Sw values change.

3. **`examples/cifar10_resnet18_calibrated.py`** — update `OnlineCalibration(...)` to pass `calib_A` after calibration.

4. **`docs/calibration.md`** — update condition (III) to distinguish local budget vs global output budget.

---

## Verification

After the fix, the calibration invariant check in `print_calibration_properties` (example, line 165) should show:

- `(I) signal = v·Σmw²` → still 1.0 per layer (unchanged)
- `(II) K_b/K_w` → still 1.0 (unchanged, only c changed, not the relative Sw shape)
- `(III) budget AW+B` → will show **different values per layer** (now `c_global · budget_denom_ℓ`, not σ_v²) — this is CORRECT and expected
- **NEW check**: a forward Sz pass through the network should give `Sz_final = σ_v²` exactly → `J = 0.5` at the output
- `σ_v(AGVI)` at epoch 1 should be close to `sqrt(σ_v²) = sqrt(log 2) ≈ 0.83`

Do not paper over the invariant check by modifying `print_calibration_properties` to hide the per-layer change — instead update the documentation comment to say condition (III) is now a GLOBAL output condition rather than a per-layer condition.

---

Now go. This is solvable, everything is closed form, and there is no reason to approximate. The math is all in front of you.
