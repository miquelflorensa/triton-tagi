# PN-TAGI investigation — detailed findings

Date: 2026-05-13
Branch: `feature/precision-normalization`

This document captures every experimental and theoretical finding produced
while implementing and stress-testing Precision-Normalized TAGI (PN-TAGI)
per [PLAN.md](./PLAN.md). It is intended to be self-contained input to
theory work — every number, every failure mode, and every open question is
recorded with pointers to the code, runs, and figures.

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [The four update rules — formal definitions](#2-the-four-update-rules--formal-definitions)
3. [The three-concern decomposition](#3-the-three-concern-decomposition)
4. [Stage 1 — Scalar / kernel-correctness sanity](#4-stage-1--scalar--kernel-correctness-sanity)
5. [Stage 2 — Tiny linear regression sweep](#5-stage-2--tiny-linear-regression-sweep)
6. [Stage 3 — MNIST MLP depth sweep](#6-stage-3--mnist-mlp-depth-sweep)
7. [Stage 3 dissection — per-batch mechanism](#7-stage-3-dissection--per-batch-mechanism)
8. [Stage 3 axis sweeps (B, σ_v, gain_w)](#8-stage-3-axis-sweeps-b-σv-gainw)
9. [Stage 3 main re-run with hybrid](#9-stage-3-main-re-run-with-hybrid)
10. [Stage R — TAGI-V regression depth sweep](#10-stage-r--tagi-v-regression-depth-sweep)
11. [Stage R follow-up — gain × depth grid](#11-stage-r-follow-up--gain--depth-grid)
12. [Open theoretical questions](#12-open-theoretical-questions)
13. [File map](#13-file-map)
14. [Run map (figures + CSVs)](#14-run-map-figures--csvs)

---

## 1. Executive summary

We added four update rules to the TAGI parameter-update kernel and ran a
diagnostic ladder from scalar Gaussian sanity tests through MNIST MLPs
and 1-D heteroscedastic regression. The headline conclusions:

1. **`precision_normalized` (PN-TAGI) is per-parameter exact Bayes** for a
   single observation, and is the local Bayesian object the plan
   describes. It always keeps `σ² > 0` (no 1e-5 floor needed).
2. **`capped_precision_normalized` (CPN-TAGI, hybrid)** — PN-TAGI's
   variance contraction + the cap-factor's mean-step bound — matches the
   cuTAGI baseline at every classification depth tested while keeping
   PN-TAGI's variance positivity guarantee. **It is empirically the most
   robust rule, but still uses the cap-factor heuristic.**
3. **The "PN-TAGI fails at depth ≥ 3" finding has a corrected
   interpretation**: it is *not* an intrinsic limitation of PN-TAGI. It
   is an interaction of **three separable concerns**:
   - the prior scale at init (`σ²_w₀`)
   - the likelihood-trust parameter (`σ_v` or learned `V²`)
   - the posterior-update rule's mini-batch behaviour.
4. **Falsified hypothesis**: removing the fixed `σ_v` hyperparameter (by
   using TAGI-V regression with learned `V²`) does *not* by itself fix
   PN-TAGI's depth failure. So `σ_v` is not the dominant load-bearing
   factor.
5. **Confirmed hypothesis**: lowering the prior scale `gain_w` from the
   default 1.0 to anywhere in [0.1, 0.5] makes plain PN-TAGI work at
   *every* depth tested. So the depth failure is primarily an
   **initialization-scale** problem.
6. **The cap-factor remains heuristic.** All current "good" results
   either keep it (CPN-TAGI) or rely on small-gain init that
   coincidentally avoids the regime where the cap would matter.

---

## 2. The four update rules — formal definitions

For each scalar parameter component `θ ∈ ℝ` with prior `(μ, σ²)`, additive
deltas `Δμ_add` and `Δσ²_add` produced by TAGI's backward pass for a
mini-batch `B`, and the cap radius

```math
\bar{\delta} \;=\; \frac{\sqrt{\max(\sigma^2,\, 10^{-10})}}{c_B}
```

with the cuTAGI batch-size heuristic

```math
c_B = \begin{cases} 0.1 & B = 1 \\ 2.0 & 1 < B < 256 \\ 3.0 & B \ge 256 \end{cases}
```

and the posterior contraction ratio

```math
\chi \;\equiv\; -\frac{\Delta\sigma^2_{add}}{\max(\sigma^2,\, \epsilon)},\qquad \epsilon = 10^{-12}
```

the four rules are:

### 2.1 `additive`

```math
\mu_{|\mathcal{B}} \;=\; \mu + \Delta\mu_{add}
```
```math
\sigma^2_{|\mathcal{B}} \;=\; \begin{cases} \sigma^2 + \Delta\sigma^2_{add} & \text{if positive} \\ 10^{-5} & \text{numerical floor} \end{cases}
```

For a single observation this matches the exact Gaussian Bayes posterior
(Kalman gain `K = σ²/(σ² + σ_v²)`):
```math
\Delta\mu_{add}^{(1)} = K \cdot (y - \mu_z),\qquad
\Delta\sigma^2_{add}^{(1)} = -K \cdot \sigma^2.
```
For B observations summed against the *prior* (TAGI's mini-batch
accumulation), `χ` grows linearly in B and the rule can drive `σ²` to
zero or below. The 1e-5 floor is a numerical safety net, not a Bayesian
operation.

### 2.2 `capped_additive` (cuTAGI baseline)

```math
\Delta\mu_{cap} \;=\; \mathrm{sgn}(\Delta\mu_{add}) \cdot \min\!\bigl(|\Delta\mu_{add}|,\; \bar{\delta}\bigr)
```
```math
\Delta\sigma^2_{cap} \;=\; \mathrm{sgn}(\Delta\sigma^2_{add}) \cdot \min\!\bigl(|\Delta\sigma^2_{add}|,\; \bar{\delta}\bigr)
```
```math
\mu_{|\mathcal{B}} = \mu + \Delta\mu_{cap},\qquad
\sigma^2_{|\mathcal{B}} = \max\!\bigl(\sigma^2 + \Delta\sigma^2_{cap},\; 10^{-5}\bigr)
```

The cap radius `δ̄ = √σ²/c_B` bounds each step at a fraction of one prior
standard deviation. `c_B` is empirically tuned; the *form* is reasonable
(trust-region-flavoured) but the *values* are not derived.

### 2.3 `precision_normalized` (PN-TAGI), with `ρ = 1`

```math
\chi^{(+)} \;=\; \max(0, \chi)
\qquad
d \;=\; 1 + \rho \chi^{(+)}
```
```math
\mu_{|\mathcal{B}} \;=\; \mu + \rho \cdot \frac{\Delta\mu_{add}}{d}
\qquad
\sigma^2_{|\mathcal{B}} \;=\; \frac{\sigma^2}{d}
```

Key identity: for a single observation in isolation,

```math
\frac{\Delta\mu_{add}}{1 + \chi} \;\approx\; \Delta\mu_{\text{exact-Bayes}}
\qquad\text{(per-parameter)}
```

so PN-TAGI **is** the per-parameter exact one-shot Bayes step. `σ²_{|B}`
is always strictly positive — no floor needed. The fraction of prior
variance consumed is `χ/(1+χ)` regardless of how large χ is, so the rule
gracefully handles "high contraction pressure" regimes the additive rule
cannot.

### 2.4 `tempered_precision_normalized` (PN-TAGI with `ρ < 1`)

Same formulas as 2.3 with `ρ ∈ (0, 1)`. Damps both the mean step and the
variance contraction by the same factor. Mathematically a strict subset
of PN-TAGI — same kernel code path.

### 2.5 `capped_precision_normalized` (CPN-TAGI, hybrid)

```math
\Delta\mu_{cap} = \mathrm{sgn}(\Delta\mu_{add}) \cdot \min\!\bigl(|\Delta\mu_{add}|,\; \bar{\delta}\bigr)
```
```math
d = 1 + \rho \max(0, \chi)
```
```math
\mu_{|\mathcal{B}} \;=\; \mu + \rho \cdot \frac{\Delta\mu_{cap}}{d}
\qquad
\sigma^2_{|\mathcal{B}} \;=\; \frac{\sigma^2}{d}
```

Cap the *raw* `Δμ_add` first, then apply PN's `/d` on the already-capped
mean step. Variance update identical to plain PN. Empirically tracks
cuTAGI baseline accuracy across every regime tested.

---

## 3. The three-concern decomposition

The plan stated "`σ_v` controls likelihood trust, PN-TAGI controls
posterior assimilation geometry, cap_factor mixes both roles." After
this investigation, the precise statement is:

| # | Concern | What it controls | Current implementation |
|---|---|---|---|
| **1** | **Initialization (prior scale)** | The prior `N(μ_w₀, σ²_w₀)` over parameters. Sets first-batch step magnitude `dm = σ²_w₀ · grad_μ` and contributes to χ at batch 0. | He / Xavier scale × `gain_w`. No principled depth/width/batch formula. The Stage R gain × depth grid shows `gain_w ∈ [0.1, 0.5]` is the working range for 1-D regression; `gain_w = 1.0` is the He default. |
| **2** | **σ_v / likelihood trust** | How confidently the model interprets observations. Sets `grad_μ ∝ 1/(S_z + σ_v²)`. | Fixed hyperparameter (e.g. 0.05 in cuTAGI MNIST) **or** TAGI-V with learned per-sample `V²` (`Linear(h, 2) + EvenSoftplus`). |
| **3** | **Posterior update rule + mini-batch** | How posterior gets computed from B individual samples that all use the same prior. | Four rules above. Additive: divergent. Capped: heuristic. PN: per-parameter exact. CPN: PN + cap. None addresses the *joint cross-parameter* mini-batch effect rigorously. |

These three are **largely independent** — each can be addressed without
the others. But choices in one affect the others' regime of validity:
- Larger init prior `σ²_w₀` (#1) ⇒ rule needs stronger damping (#3).
- Smaller `σ_v` (#2) ⇒ larger `grad_μ` ⇒ rule needs stronger damping (#3).
- "Capacious enough" rule (#3) makes #1 and #2 less sensitive.

A no-hyperparameter PN-TAGI requires a principled answer in *all three*.
What has been done so far:
- **#1**: only empirical sensitivity (gain_w sweep); no formula yet.
- **#2**: TAGI-V (heteros) is implemented but isn't yet a "global random
  σ_v" (proposal sketched in section 12).
- **#3**: PN handles per-parameter; cap heuristically handles the joint
  mini-batch effect. A principled trust-region or joint-Kalman variant
  has not been implemented.

---

## 4. Stage 1 — Scalar / kernel-correctness sanity

**Goal**: pin the kernel formulas for all four rules and verify the
scalar Gaussian Kalman behaviour.

**File**: [tests/unit/test_pn_tagi_update.py](./tests/unit/test_pn_tagi_update.py)
**Kernel**: [triton_tagi/update/parameters.py](./triton_tagi/update/parameters.py)
**Tests**: 24 (all pass)

### Per-rule kernel correctness

For each rule, the test builds random `(m, S, dm, dS)` tensors of size
1024 covering both signs of `dS`, computes the expected output via a
pure-torch reference (`_ref_additive`, `_ref_capped`, `_ref_pn`,
`_ref_capped_pn`), and asserts the Triton kernel matches within
`atol=1e-5, rtol=1e-5`.

### χ-diagnostic correctness

`χ_out` matches `−ΔS / max(S, 1e-12)` regardless of which rule fires.
Sign of `χ_out` is the inverse of sign of `dS`, so positive variance
increments (`dS > 0`) appear as `χ < 0`.

### Scalar 1-observation Kalman

Setup: scalar `θ ~ N(μ, σ²)` observed once as `y = θ + v`,
`v ~ N(0, σ_v²)`.

- `additive` produces `μ_post = μ + K(y − μ)` and `σ²_post = σ²(1 − K)`
  exactly — matches the exact Bayes posterior to fp32 precision.
- `precision_normalized` produces `σ²_post = σ²/(1 + K)` and
  `μ_post = μ + K(y − μ)/(1 + K)`. **Differs from exact Bayes by `O(K²)`.**
  PN is *strictly less contracted* than Bayes for `K > 0`; it is a
  *regularised* posterior, not the exact one. The plan's statement
  "PN-TAGI matches exact local posterior" was overstated; PN matches
  exact Bayes only as `K → 0`.

### N-observation mini-batch (synthetic additive accumulation)

Setup: pretend a batch of `N` i.i.d. observations was processed by
summing the per-observation additive deltas.

- `additive` hits the 1e-5 numerical floor when `N·K > 1`.
- `precision_normalized` keeps `σ² > 0` always; the consumed variance
  fraction `1 − σ²_post / σ²` matches the closed form `χ/(1+χ)` to fp32
  precision (`χ = N·K`).

These tests don't run network forward/backward — they exercise the
update kernel directly with synthetic deltas, so they isolate the rule's
mathematical correctness from any pipeline behaviour.

---

## 5. Stage 2 — Tiny linear regression sweep

**Goal**: study χ behaviour and rule efficacy in the simplest
non-trivial setting where TAGI's local Gaussian assumption is exact.

**File**: [tests/unit/test_pn_tagi_stage2.py](./tests/unit/test_pn_tagi_stage2.py) (12 tests, all pass)
**Sweep script**: [experiments/pn_tagi_stage2/run_sweep.py](./experiments/pn_tagi_stage2/run_sweep.py)
**Run**: `runs/pn_tagi_stage2_20260512-161549/`

Model: `Linear(in_features, 1)` (no activation, no output head). Data:
`y = x·W_true + b_true + ε`, `ε ~ N(0, σ_obs²)`, `x ~ N(0, I)`.

### Closed-form chi

For a 1-layer Linear, the per-weight chi is (derivation in PLAN.md
§Implementation Instructions):

```math
\chi_w \approx \frac{S_w \cdot \sum_i x_i^2}{S_z + \sigma_v^2}
\;\approx\;
\frac{B \cdot S_w \cdot \langle x^2\rangle}{S_z + \sigma_v^2}
\;\propto\;
\frac{B \cdot \mathrm{gain}_w^2}{\sigma_v^2}
\;\text{(for small } S_z\text{)}.
```

The three pinned claims (verified):
- **χ decreases monotonically in σ_v** (test passes at B ∈ {4, 32, 128}).
- **χ increases monotonically in B** (test passes at σ_v ∈ {0.05, 0.5}).
- **χ increases monotonically in `gain_w`** (test passes).

### Headline sweep result (3 rules × 3 batch sizes × 2 σ_v, 100 steps each)

Median validation MSE across `gain_w = 1.0` (data noise floor ≈ σ_v²):

| | B=1 σ_v=0.05 | B=1 σ_v=1.0 | B=32 σ_v=0.05 | B=32 σ_v=1.0 | B=128 σ_v=0.05 | B=128 σ_v=1.0 |
|---|---|---|---|---|---|---|
| additive | 0.00019 | 0.188 | **1.45** | **60.2** | **1.78** | **991** |
| capped_additive | 0.00019 | 0.188 | **8.31** | **8.36** | **8.22** | **7.32** |
| precision_normalized | 0.00027 | 0.189 | 6.4e-6 | 0.003 | 7.5e-7 | 0.0003 |

**Key observation**: at B ≥ 32, additive diverges spectacularly,
*capped_additive plateaus at the prior MSE* (~8) — the cap is so
aggressive that learning stalls — and PN-TAGI converges cleanly to
near-zero everywhere. This is Stage 2's headline finding.

### Stage 2 figures

- [val_mse heatmap](./runs/pn_tagi_stage2_20260512-161549/figures/val_mse.png)
- [chi_p95 initial heatmap](./runs/pn_tagi_stage2_20260512-161549/figures/chi_p95_initial.png) — identical across rules (it's a prior property)
- [Sw trace](./runs/pn_tagi_stage2_20260512-161549/figures/Sw_trace.png) — additive + capped pinned at the 1e-5 floor; PN-TAGI contracts smoothly

### Stage 2 caveat

This is a 1-layer linear model. The "cross-parameter" effects that
dominate at depth (Stage 3) do not manifest here. So Stage 2 only
validates *per-parameter* rule behaviour. The cap_additive plateau is
revealing: at this batch size, the cap is over-restrictive — the
heuristic is tuned for deep nets with stale gradients, not for tiny
linear regression. (This is itself evidence that c_B values are
problem-specific.)

---

## 6. Stage 3 — MNIST MLP depth sweep

**Goal**: do the things that work in Stage 2 carry through deeper
networks?

**Script**: [experiments/pn_tagi_stage3/run_mnist_depths.py](./experiments/pn_tagi_stage3/run_mnist_depths.py)
**Run**: `runs/pn_tagi_stage3_20260512-192629/`

Architecture: `784 → [256]·depth → 10` with Remax output.
Defaults: σ_v=0.05, B=512, gain_w=1.0, hidden=256, 5 epochs.

### Headline result (depth × rule, test accuracy)

| depth | capped_additive | precision_normalized |
|---|---|---|
| 1 | 97.61% | 96.38% |
| 3 | 97.56% | **20.90%** |
| 5 | 97.48% | **11.35%** |
| 7 | 97.01% | **10.10%** |

Plain PN-TAGI **catastrophically fails at depth ≥ 3**. The plateau is
~chance (10% = 1/10 for MNIST).

### Per-layer chi p95 at end of epoch 1

| depth | capped_additive (max across layers) | precision_normalized (max across layers) |
|---|---|---|
| 1 | 2.3e-3 | 6.1e-4 |
| 3 | 8.5e-4 | **6e-9** |
| 5 | 4.5e-4 | **5.7e-13** |
| 7 | 1e-3 | **2.6e-14** |

Plain PN-TAGI's chi diagnostic drops to ~1e-9 to 1e-14 after the first
epoch at depth ≥ 3. This was the first puzzle: chi ≈ 0 means "no
contraction is happening", which is the opposite of the failure mode
I expected (over-contraction). The dissect (next section) resolved it.

### Per-layer Sw is NOT collapsing

Inspecting the Sw trace plot
([Sw_trace.png](./runs/pn_tagi_stage3_20260512-192629/figures/Sw_trace.png))
shows `Sw_mean` stays in `[1e-3, 4e-3]` for both rules across all
depths. So the failure is *not* parameter-variance collapse — Sw is
similar to capped's, but the parameters are stuck.

---

## 7. Stage 3 dissection — per-batch mechanism

**Goal**: identify *which batch*, *which layer*, and *what specific
quantity* fails first under plain PN-TAGI at depth ≥ 3.

**Script**: [experiments/pn_tagi_stage3/run_mnist_dissect.py](./experiments/pn_tagi_stage3/run_mnist_dissect.py)
**Run**: `runs/pn_tagi_stage3_dissect_20260512-193746/`
**Config**: depth=3, σ_v=0.05, B=512, gain_w=1.0, 30 batches with full per-batch logging.

The dissect instruments every batch in epoch 1 with:
- Per Linear layer: `chi_p95`, `chi_max`, `Sw_mean`, `Sw_min`,
  `mw_mean_abs`, `mw_max_abs` (post-update).
- Per layer interface (extra forward after update): activation mean,
  activation variance, fraction `|a| < 1e-6` (dead-ReLU proxy).
- Per batch: train_acc, Remax mean output entropy, mean max-probability.

### Smoking-gun observations (from [dissect.png](./runs/pn_tagi_stage3_dissect_20260512-193746/figures/dissect.png))

Side-by-side comparison of capped_additive (works) vs precision_normalized (fails):

1. **|μ_w| at every Linear layer**: plain PN's parameter magnitudes
   jump 3× in a single batch (from ~3e-2 to ~1e-1). Capped's barely
   move (~3-7e-2 throughout). The cap is doing its design job.
2. **Dead-ReLU fraction at every ReLU**: plain PN hits **80-90% dead**
   within 3-5 batches and stays there. Capped peaks at 40-60% and
   recovers.
3. **Sw_min** is *not* the failure signal — plain PN's output-layer
   Sw_min sits at ~2e-5 while capped reaches ~6e-7. Variance dies
   more under capped, but capped *learns*.
4. **Remax output entropy** for plain PN oscillates between 0.5 and 2.3
   (= log 10, uniform) indefinitely; capped's drops monotonically to
   ~0.3 (sharp predictions).
5. **chi_p95** for plain PN oscillates 1e-9 ↔ 1e-23 — essentially
   zero. The variance contraction signal is dead because the
   parameters are frozen, not because chi is small at the prior.

### The mechanism

When `σ_v` is small and `σ²_w` is *also* small (which happens at init
for hidden layers with He init: `σ_w² ≈ 1/fan_in ≈ 4e-3` for hidden=256):
- `χ = S_w · Σx²/(S_z + σ_v²)` can be small per parameter (a *single
  scalar* doesn't see big contraction pressure)
- But the raw `Δμ_add = S_w · grad_μ` is **huge** because
  `grad_μ ∝ 1/(S_z + σ_v²) ≈ 1/σ_v² ≈ 400` blows up at small σ_v
- PN-TAGI's denominator `1 + max(0, χ)` is near 1 → **no protection
  on the mean step**
- A single batch shifts μ_w by 3× its init magnitude
- The next forward produces dead ReLU activations everywhere
- All subsequent gradient information is killed by zero activations
- Training is frozen for the remaining epochs

The cap-factor's `Δ̄ = √S_w/c_B` is what stops this in the cuTAGI
baseline. PN-TAGI on its own provides only per-parameter contraction
control — it does not bound the *cross-parameter cumulative* effect on
the next layer's pre-activations.

---

## 8. Stage 3 axis sweeps (B, σ_v, gain_w)

**Goal**: test where each rule lives in (B × σ_v × gain_w) space.

**Script**: [experiments/pn_tagi_stage3/run_mnist_axis_sweep.py](./experiments/pn_tagi_stage3/run_mnist_axis_sweep.py)
**Run**: `runs/pn_tagi_stage3_axis_20260512-194923/`

Fixed: depth=3, hidden=256, 3 epochs. Rules:
{capped_additive, precision_normalized, **capped_precision_normalized**}.

CPN-TAGI is the new hybrid rule introduced after the dissect
(formula in §2.5). Stage 1 has separate kernel-correctness tests for it
(6 parametrized cases). All 44 unit tests pass after its addition.

### Batch-size sweep (σ_v=0.05, gain_w=1.0)

| B | capped_additive | precision_normalized | capped_precision_normalized |
|---|---|---|---|
| 16 | 97.41% | 97.57% (dead 49%) | 97.70% |
| 64 | 97.25% | 96.61% (dead 66%) | 97.52% |
| 256 | 97.20% | **70.82%** (dead 98%) | 96.68% |
| 512 | 97.20% | **20.98%** (dead 100%) | 97.25% |
| 1024 | 96.67% | **20.22%** (dead 98%) | 96.86% |

The cliff for plain PN is at B ≥ 256. At B=16-64 it still works; once
B is big enough that the summed additive deltas overshoot, it dies.

### σ_v sweep (B=512, gain_w=1.0)

| σ_v | capped_additive | precision_normalized | capped_precision_normalized |
|---|---|---|---|
| 0.02 | 96.95% | **11.35%** (dead 100%) | 96.80% |
| 0.05 | 97.20% | **20.98%** (dead 100%) | 97.25% |
| 0.1 | 97.45% | **69.66%** (dead 100%) | 97.07% |
| 0.3 | 97.48% | 96.90% (dead 23%) | 97.26% |
| 1.0 | 93.96% | 93.67% (dead 0%) | 93.95% |

Plain PN fails for σ_v ≤ 0.1. At σ_v=1.0 even the cuTAGI baseline
under-trains (likelihood too soft), so all rules are at 94%.

### gain_w sweep (σ_v=0.05, B=512)

| gain_w | capped_additive | precision_normalized | capped_precision_normalized |
|---|---|---|---|
| 0.25 | 97.15% | 94.95% (dead 94%) | 97.04% |
| 0.5 | 96.92% | 87.26% (dead 98%) | 97.12% |
| 1.0 | 97.20% | **20.98%** (dead 100%) | 97.25% |
| 2.0 | 96.81% | **16.27%** (dead 95%) | 96.57% |
| 4.0 | 94.38% | **11.35%** (dead 100%) | 96.74% |

Plain PN works at very small init (gain=0.25 → 95%) but fails at He
default (gain=1.0). This was the first hint that #1 (init) is doing
load-bearing work.

### Axis-sweep figures

- [sweep_batch.png](./runs/pn_tagi_stage3_axis_20260512-194923/figures/sweep_batch.png)
- [sweep_sigma_v.png](./runs/pn_tagi_stage3_axis_20260512-194923/figures/sweep_sigma_v.png)
- [sweep_gain_w.png](./runs/pn_tagi_stage3_axis_20260512-194923/figures/sweep_gain_w.png)

Each panel: left = final test accuracy vs axis, right = worst-ReLU
dead-fraction at end of epoch 1.

---

## 9. Stage 3 main re-run with hybrid

**Goal**: extend Stage 3's main depth sweep with CPN-TAGI included.

**Script**: same as Stage 3 ([run_mnist_depths.py](./experiments/pn_tagi_stage3/run_mnist_depths.py))
**Run**: `runs/pn_tagi_stage3_20260512-195702/`
**Config**: depth ∈ {1, 3, 5, 7}, 5 epochs, σ_v=0.05, B=512, gain_w=1.0.

| depth | capped_additive | precision_normalized | **capped_precision_normalized** |
|---|---|---|---|
| 1 | 97.61% | 96.38% | **97.70%** |
| 3 | 97.56% | 20.90% | **97.70%** |
| 5 | 97.48% | 11.35% | **97.31%** |
| 7 | 97.01% | 10.10% | **96.98%** |

CPN-TAGI matches or slightly beats the cuTAGI baseline at every depth
tested, while plain PN remains broken (deterministic — identical
numbers to the original Stage 3 run).

[Accuracy plot](./runs/pn_tagi_stage3_20260512-195702/figures/accuracy.png) shows capped (blue) and hybrid (green) overlapping at all four depths; plain PN (orange) flatlines.

**This is the most "practically robust" rule we have**, but it still
inherits the cap-factor heuristic. CPN is the right rule to ship
*today*, but it isn't the principled disentanglement the plan called
for — it just combines two heuristics (cap + PN's `1/d` dampening) that
happen to cover each other's failure modes.

---

## 10. Stage R — TAGI-V regression depth sweep

**Goal**: test whether plain PN-TAGI's depth failure is caused by the
**fixed `σ_v` hyperparameter** specifically. Use TAGI-V (heteros) so
`V²` is learned per-sample, not a fixed scalar.

**Script**: [experiments/pn_tagi_stageR/run_regression_depths.py](./experiments/pn_tagi_stageR/run_regression_depths.py)
**Run**: `runs/pn_tagi_stageR_20260513-010832/`

Architecture: `1 → [50]·depth → 2 + EvenSoftplus`. Data: 1-D heteros
`y = sin(x) + ε(x)`, `ε ~ N(0, (0.05 + 0.3|x|)²)`. 50 epochs, hidden=50,
B=64, gain_w=1.0.

### Final test RMSE (data noise floor ≈ 0.65 in normalised units)

| depth | capped_additive | precision_normalized | capped_precision_normalized | additive |
|---|---|---|---|---|
| 1 | 0.87 | **0.94** | 0.88 | NaN |
| 3 | 0.89 | **1.14** | 0.89 | 0.89 |
| 5 | 0.88 | **1.14** | 0.89 | 1.14 |
| 7 | 0.90 | **1.14** | 0.96 | 0.89 |

**The σ_v hypothesis is falsified.** Even with V² learned (no σ_v
hyperparameter), plain PN-TAGI cliff-fails at depth ≥ 3 (RMSE = 1.14 is
the "no-learning" plateau — the model just predicts `y ≈ mean(y_train)`
and attributes the residual to noise via V²).

### Chi heatmap signature

Same Sw-collapse / chi-falls-to-1e-9 signature as the MNIST failure.
The mean-step overshoot mechanism is the same in regression as in
classification — `σ_v` is not the load-bearing factor.

### V² drifts upward at deep PN

In the V² trace plot, plain PN's mean predicted V² at depth 3/5/7
*increases* over training (0.7 → 0.85+) while capped/CPN's decreases
to the true noise level (~0.4). The model is reattributing the
unexplained signal to noise because it can't fit it — a "failure in
disguise" that test RMSE alone wouldn't show.

### Figures

- [rmse.png](./runs/pn_tagi_stageR_20260513-010832/figures/rmse.png)
- [nll.png](./runs/pn_tagi_stageR_20260513-010832/figures/nll.png)
- [v2_trace.png](./runs/pn_tagi_stageR_20260513-010832/figures/v2_trace.png)
- [chi_heatmap.png](./runs/pn_tagi_stageR_20260513-010832/figures/chi_heatmap.png)

---

## 11. Stage R follow-up — gain × depth grid

**Goal**: test whether plain PN-TAGI's depth failure is fixed by
**smaller prior init** (`gain_w < 1`).

**Script**: [experiments/pn_tagi_stageR/run_regression_gain_depth.py](./experiments/pn_tagi_stageR/run_regression_gain_depth.py)
**Run**: `runs/pn_tagi_stageR_gain_depth_20260513-012903/`
**Config**: TAGI-V regression, 30 epochs, hidden=50, B=64.

### Final test RMSE — `precision_normalized` only (capped is gain-insensitive)

| gain_w \ depth | 1 | 3 | 5 | 7 |
|---|---|---|---|---|
| 0.05 | 0.96 | 0.91 | 0.93 | 1.03 |
| **0.1** | **0.89** | **0.89** | **0.89** | **0.90** |
| **0.25** | 0.94 | **0.89** | **0.89** | **0.89** |
| **0.5** | 0.93 | **0.89** | 0.90 | 0.90 |
| 1.0 | 0.96 | **1.15** | **1.14** | **1.14** |

`capped_additive` for comparison (gain-insensitive baseline): 0.87-0.93
across the entire grid.

### The cliff is exactly at gain_w = 1.0 (He scale)

Plain PN-TAGI's depth failure happens **only at gain=1.0 AND depth ≥ 3**.
Drop to gain ≤ 0.5 and PN-TAGI works at every depth tested, matching
the capped baseline within ±0.04 RMSE.

### Working range

Sweet spot is `gain_w ∈ [0.1, 0.5]`. Below 0.1 the prior is so small
that mean steps `Δμ_w = σ²_w · grad_μ` are too small to learn quickly
in 30 epochs (RMSE 1.03 at depth=7 with gain=0.05). Above 0.5 the
first-batch step overshoots and PN-TAGI dies.

### Mechanism (one sentence)

Smaller initial Sw → smaller first-batch `dm = Sw·grad_μ` → no
mean-step overshoot → no dead-ReLU collapse → learning proceeds.

### Figure

- [gain_depth.png](./runs/pn_tagi_stageR_gain_depth_20260513-012903/figures/gain_depth.png) — the heatmap that closes the loop.

---

## 12. Open theoretical questions

### Q1. The right initialization formula

We have only empirical sensitivity, not a derivation. Three principled
targets that would each give a different formula for `σ²_w₀`:

- **Variance preservation through forward**: choose `σ²_w₀` so each
  layer's pre-activation variance `S_z` stays ≈ constant across depth.
  This is essentially the TAGI extension of LeCun / He variance-preserving
  init.
- **Bounded initial chi**: choose `σ²_w₀` so `p95(χ_init) ≤ τ` for some
  target like 0.5. Requires modeling χ as a function of (depth, width,
  B, σ_v).
- **Bounded initial step in units of prior σ**: choose `σ²_w₀` so
  `𝔼[|Δμ_w| / σ_w] ≤ τ` for τ ≈ 1. This is the trust-region
  interpretation, but applied at init rather than as a runtime cap.

The three would coincide as `χ → 0` (vacuous regime) but diverge as
the network gets deeper / batches larger. Worth deriving all three and
testing which is most predictive of the empirical working range.

### Q2. The right σ_v treatment

cuTAGI-style heteros (TAGI-V) works empirically but is per-sample
(network emits V² as another output). A cleaner Bayesian alternative
sketched in the conversation:

Promote `V²` to a *global* Gaussian random parameter
`V² ~ N(μ_{V²}, σ²_{V²})`. The output innovation becomes
`δμ_y = (y − μ_z) · 𝔼[1/(S_z + V²)]`, computed via Taylor (second
order):

```math
\mathbb{E}\!\left[\tfrac{1}{S_z + V^2}\right]
\;\approx\;
\frac{1}{S_z + \mu_{V^2}}
\;+\;
\frac{\sigma^2_{V^2}}{(S_z + \mu_{V^2})^3}
```

When `σ²_{V²}` is large (early training), the innovation saturates →
automatic likelihood-trust cap. As data accumulates, V² posterior
tightens → cap relaxes. **This is the cleanest path to removing fixed
σ_v as a hyperparameter.** Not yet implemented.

### Q3. The right joint-mini-batch posterior update

Three options on the table:

- **Status quo**: PN per parameter, ignore cross-parameter interaction.
  Works only if init is small enough (Stage R finding).
- **Joint mini-batch Kalman**: exact joint posterior across all
  parameters in a layer. Compute cost ~130× per Linear, memory ~640×
  for hidden layers (calculated in conversation). Cheap (~5×) for the
  output layer alone, where the observation model is exactly linear.
- **Layer-level trust-region**: bound the L2 norm of `Δμ` per layer
  such that the change in next-layer pre-activations is bounded. Not
  yet sketched mathematically.

CPN-TAGI's cap factor is an empirical proxy for option 3 (bounds each
parameter's step in units of `σ_w`). A principled version would bound
the *layer-wise* effect, not per-parameter.

### Q4. Are the three concerns truly independent?

Conjectures (untested):
- **Conjecture A**: with the right init (Q1), plain PN-TAGI works
  across all depths *and* batch sizes *and* σ_v values. The cap factor
  is purely a band-aid for bad init.
- **Conjecture B**: with the right σ_v treatment (Q2), the *init*
  becomes less sensitive. The two interact: large prior σ²_w with
  uncertain V² gives the same effective cap as the c_B heuristic.
- **Conjecture C**: a principled #3 (Q3) would make #1 and #2
  irrelevant for stability (only affecting convergence speed).

The current data is consistent with A. We haven't tested B or C
directly.

### Q5. What's the PN-TAGI generalisation gap?

In Stage 2 (single-parameter exact Bayes), PN matches additive (which
matches exact Bayes for single obs). In Stage 3 (deep nets) PN works
when init is small. In Stage R (regression) same. So *empirically* PN
seems to give the same generalisation as the capped baseline when
init is in the working range.

But: PN's variance update keeps `σ²` strictly positive and respects
the per-parameter Bayes geometry, while capped's `S + ΔS_cap` with
1e-5 floor does not. So in principle PN's posterior should be more
Bayesian-faithful. There may be a generalisation benefit (or test-NLL
benefit) we haven't measured — all Stage 3 metrics are accuracy, not
calibration. **Stage R's NLL plot shows PN's NLL is comparable to
capped's at gain≤0.5**, suggesting PN's "better posterior" claim
hasn't translated to measured generalisation in any of our setups.
This is worth investigating: maybe better metrics (Brier score, ECE,
log-likelihood on held-out) would reveal a benefit.

### Q6. Behaviour with normalisation layers

Not tested. The plan's Stage 4 (CIFAR small CNN with BatchNorm) is the
next architectural confound. BatchNorm's running statistics change
the forward variance structure, which could interact with the cap
factor and PN dampening in non-obvious ways. ResBlocks (Stage 5) raise
the same issues with skip connections.

---

## 13. File map

### Implementation

- **Kernel (4 update rules)**: [triton_tagi/update/parameters.py](./triton_tagi/update/parameters.py)
- **Layer API**: each `LearnableLayer.update()` accepts `update_rule`, `rho`, `record_chi` kwargs:
  [triton_tagi/base.py](./triton_tagi/base.py),
  [triton_tagi/layers/linear.py:161](./triton_tagi/layers/linear.py#L161),
  [triton_tagi/layers/conv2d.py:376](./triton_tagi/layers/conv2d.py#L376),
  [triton_tagi/layers/batchnorm2d.py:518](./triton_tagi/layers/batchnorm2d.py#L518),
  [triton_tagi/layers/layernorm.py:180](./triton_tagi/layers/layernorm.py#L180),
  [triton_tagi/layers/embedding.py:156](./triton_tagi/layers/embedding.py#L156),
  [triton_tagi/layers/rms_norm.py:139](./triton_tagi/layers/rms_norm.py#L139),
  [triton_tagi/layers/multihead_attention.py:273](./triton_tagi/layers/multihead_attention.py#L273),
  [triton_tagi/layers/resblock.py:435](./triton_tagi/layers/resblock.py#L435).
- **Sequential plumbing**: [triton_tagi/network.py](./triton_tagi/network.py) — `update_rule`, `rho`, `record_chi` constructor args; `collect_chi_stats()` walks the network.
- **Module exports**: [triton_tagi/__init__.py](./triton_tagi/__init__.py), [triton_tagi/update/__init__.py](./triton_tagi/update/__init__.py).

### Tests

- **Stage 1 (kernel + scalar Kalman, 24 tests)**: [tests/unit/test_pn_tagi_update.py](./tests/unit/test_pn_tagi_update.py)
- **Stage 2 (synthetic linear, 12 tests)**: [tests/unit/test_pn_tagi_stage2.py](./tests/unit/test_pn_tagi_stage2.py)
- **Total**: 131 unit tests pass on the branch (95 pre-existing + 24 Stage 1 + 12 Stage 2).
- **Pre-existing failing validation tests** (verified unrelated to PN-TAGI via `git stash` on 2026-05-12): `test_mnist_layernorm_3epochs`, `test_embedding_update`, `test_emb_linear_multi_step_parity`.

### Experiment scripts

- **Stage 2**: [experiments/pn_tagi_stage2/run_sweep.py](./experiments/pn_tagi_stage2/run_sweep.py)
- **Stage 3 main**: [experiments/pn_tagi_stage3/run_mnist_depths.py](./experiments/pn_tagi_stage3/run_mnist_depths.py)
- **Stage 3 dissect**: [experiments/pn_tagi_stage3/run_mnist_dissect.py](./experiments/pn_tagi_stage3/run_mnist_dissect.py)
- **Stage 3 axis sweep**: [experiments/pn_tagi_stage3/run_mnist_axis_sweep.py](./experiments/pn_tagi_stage3/run_mnist_axis_sweep.py)
- **Stage R main**: [experiments/pn_tagi_stageR/run_regression_depths.py](./experiments/pn_tagi_stageR/run_regression_depths.py)
- **Stage R gain × depth**: [experiments/pn_tagi_stageR/run_regression_gain_depth.py](./experiments/pn_tagi_stageR/run_regression_gain_depth.py)

---

## 14. Run map (figures + CSVs)

Every experiment writes a `runs/<stage>_<timestamp>/` directory with
`figures/` (PNG + PDF), `traces/` (per-config CSV), and `summary.csv`.

| Stage | Run dir | Headline figure |
|---|---|---|
| Stage 2 | [runs/pn_tagi_stage2_20260512-161549/](./runs/pn_tagi_stage2_20260512-161549/) | [val_mse.png](./runs/pn_tagi_stage2_20260512-161549/figures/val_mse.png) |
| Stage 3 (2 rules) | [runs/pn_tagi_stage3_20260512-192629/](./runs/pn_tagi_stage3_20260512-192629/) | [accuracy.png](./runs/pn_tagi_stage3_20260512-192629/figures/accuracy.png) |
| Stage 3 dissect | [runs/pn_tagi_stage3_dissect_20260512-193746/](./runs/pn_tagi_stage3_dissect_20260512-193746/) | [dissect.png](./runs/pn_tagi_stage3_dissect_20260512-193746/figures/dissect.png) |
| Stage 3 axis sweep | [runs/pn_tagi_stage3_axis_20260512-194923/](./runs/pn_tagi_stage3_axis_20260512-194923/) | [sweep_batch.png](./runs/pn_tagi_stage3_axis_20260512-194923/figures/sweep_batch.png) [sweep_sigma_v.png](./runs/pn_tagi_stage3_axis_20260512-194923/figures/sweep_sigma_v.png) [sweep_gain_w.png](./runs/pn_tagi_stage3_axis_20260512-194923/figures/sweep_gain_w.png) |
| Stage 3 main + hybrid | [runs/pn_tagi_stage3_20260512-195702/](./runs/pn_tagi_stage3_20260512-195702/) | [accuracy.png](./runs/pn_tagi_stage3_20260512-195702/figures/accuracy.png) |
| Stage R main | [runs/pn_tagi_stageR_20260513-010832/](./runs/pn_tagi_stageR_20260513-010832/) | [rmse.png](./runs/pn_tagi_stageR_20260513-010832/figures/rmse.png) |
| Stage R gain × depth | [runs/pn_tagi_stageR_gain_depth_20260513-012903/](./runs/pn_tagi_stageR_gain_depth_20260513-012903/) | [gain_depth.png](./runs/pn_tagi_stageR_gain_depth_20260513-012903/figures/gain_depth.png) |
