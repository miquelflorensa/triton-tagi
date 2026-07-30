# Mission: Scale TAGI calibration to the 98%-CIFAR-10 regime

You are working on `triton-tagi`, a Triton/PyTorch implementation of TAGI (Tractable
Approximate Gaussian Inference, James-A. Goulet). The parameter-free calibration core
(`triton_tagi/calibrate.py`, conditions I/II/III + the online operator) now trains a
ResNet-18 stably with output Kalman gain `J = ½`. This mission is about **scalability of
the inference itself** — closing the gap between the current accuracy plateau and the
~98% regime — **without abandoning what makes TAGI TAGI.**

> **Non-negotiable constraint.** TAGI does **not** linearize. There is no Jacobian
> taping, no Extended-Kalman step, no automatic differentiation of the forward map.
> Every operation must be an **exact closed-form Gaussian** computation: the forward
> pass is exact moment-matching (the closed-form moments of ReLU/softplus/etc. through a
> Gaussian), and the parameter/state update is exact Gaussian conditioning. Where an
> exact closed form exists, use it. **The single approximation we are allowed to keep is
> the diagonal (mean-field) covariance — and even that we will relax with a low-rank
> correction.** If a step tempts you to linearize or sample, stop: there is a closed
> form, and it is the point of TAGI.

Everything below is validated by `mission/scalable_bayes_demo.py` (exact float64
references, CPU). Run it: `python mission/scalable_bayes_demo.py`.

---

## 1. TAGI is already a closed-form Gaussian filter **and** smoother

Frame training as inference on a Gaussian state-space model — but keep it **exact**, not
extended/linearized.

### 1.1 Training-time SSM (weights as a slowly-drifting state)

Let `θ ∈ ℝ^p` be all weights with Gaussian belief `θ ~ 𝒩(μ_t, P_t)`. One minibatch is one
inference step:

```
 drift   :  θ_t = θ_{t-1} + w_t ,        w_t ~ 𝒩(0, Q_t)         (process / re-inflation)
 observe :  the network output, with noise R_t                    (likelihood)
```

For a **single linear layer**, the pre-activation `z = aᵀθ` is *exactly linear* in `θ`.
So the conditioning of `θ` on a (smoothed) target for `z` is **closed-form and exact** —
there is nothing to linearize:

```
 S_i = a_iᵀ P a_i + r           (predictive / innovation variance — scalar, exact)
 k_i = P a_i / S_i              (gain)
 μ  ← μ + k_i (target_i − a_iᵀ μ)
 P  ← P − (P a_i)(P a_i)ᵀ / S_i
```

This sequential recursion reproduces the batch Bayesian-linear-regression posterior to
machine precision — **T1** below shows `max|Δμ| = 1.1e-15`, `max|ΔP| = 3.8e-17`. The
nonlinearities (ReLU, softplus, AvgPool…) are handled by TAGI's **exact Gaussian
moment** formulas in the forward pass, not by a first-order Taylor expansion. So the
*only* place an approximation enters is the representation of `P`.

### 1.2 Depth-time SSM (the network is a chain; the backward pass is a smoother)

Treat depth as time: state `s_ℓ` = layer-`ℓ` activations, transition = the layer map,
"process noise" = the per-layer epistemic budget (condition III's local budget). TAGI's
forward pass is the **filter** (it conditions `s_ℓ` on inputs `x` only). Its backward
pass — which propagates the corrected moments from the output back through the layers —
is, for the linear-Gaussian segments, **exactly the Rauch–Tung–Striebel (RTS) smoother**:

```
 G_ℓ   = P_ℓ^f F_ℓᵀ (P_{ℓ+1}^-)⁻¹                              (smoother gain)
 μ_ℓ^s = μ_ℓ^f + G_ℓ (μ_{ℓ+1}^s − μ_{ℓ+1}^-)
 P_ℓ^s = P_ℓ^f + G_ℓ (P_{ℓ+1}^s − P_{ℓ+1}^-) G_ℓᵀ
```

`F_ℓ` is the layer's exact linear map (e.g. `μ_w` for a dense layer), **not** a numerical
Jacobian. The smoother is what lets a label at the output revise the belief about an
early layer — i.e. credit assignment. **T4** shows that without the backward smoothing a
filter literally cannot move an early layer (`Var[x₀|y]` stays at the prior `1.0000`),
whereas the smoother recovers the exact posterior (`0.6023` vs exact `0.6023`).

> **Takeaway.** TAGI = exact closed-form Gaussian **filter + smoother**, with a
> **diagonal `P`**. Calibration sets the initial covariance so the filter is
> well-conditioned. The accuracy ceiling is the diagonal `P`, and the cure is to keep a
> little structure in `P` — staying 100% inside closed-form Bayes.

---

## 2. Why the diagonal assumption is the accuracy ceiling

A diagonal `P` is the **mean-field** posterior. With a diagonal `P`, the update
`μ ← μ + P aᵀ(·)/S` is a *diagonal-preconditioned* Bayesian step — the Gaussian analogue
of an Adam-like, per-coordinate learning rule. On correlated features (which is every
real conv/linear layer) it **mis-estimates the predictive (innovation) variance** — the
very `S` that condition (III) is about.

**T2** makes this exact. On strongly correlated inputs we compute the true posterior
`P_exact`, then approximate it as `diag(P) + rank-k`:

```
 rank k | mean S_approx/S_exact | mean|ratio-1| | ‖P̂−P‖_F
      0  |        1.4800         |    0.4800     |  6.15e-02   diagonal = current TAGI
      1  |        1.4977         |    0.4977     |  3.15e-02
      2  |        1.3010         |    0.3012     |  2.62e-02
      4  |        1.2381         |    0.2381     |  1.81e-02
      8  |        1.0236         |    0.0266     |  6.10e-03
     12  |        1.0000         |    0.0000     |  8.59e-17   exact (full rank)
```

The pure diagonal mis-estimates the predictive variance by **48%** — it is *not*
Bayes-calibrated on correlated data. The covariance error falls **monotonically** with
rank (Eckart–Young), and a modest rank recovers calibration (`k=8` → 2.7% error). This
is the whole thesis: **a small low-rank correction to the diagonal closes most of the
gap to exact Bayes, at `O(p·k)` cost instead of `O(p²)`.**

---

## 3. What to build (all closed-form, all Bayes-compatible)

### Upgrade 1 — Calibrate the innovation **covariance**, not its trace

Condition (III) currently makes a *scalar* output budget equal `σ_v²`
(`Sz_final = σ_v²`). The correct object is the output **innovation covariance**

```
 S = Σ_ŷ + R ,     Σ_ŷ = predictive (epistemic) output covariance,  R = obs noise
```

and the gain is the **matrix** `J = Σ_ŷ (Σ_ŷ + R)⁻¹`. A scalar budget only matches
`trace(Σ_ŷ) = trace(R)`; it leaves `Σ_ŷ` correlated, so the gain is unbalanced across
output directions. **T3** (output covariance with condition number ≈ 25):

```
 scalar (trace-matched) :  gain eigenvalues  min = 0.094 ,  max = 0.722
 matrix (Σ_ŷ = R)       :  gain eigenvalues  min = 0.500 ,  max = 0.500
```

Some class directions learn at gain 0.72, others at 0.09 — a 7× imbalance hidden behind
a correct *average*. **Do:** generalize `_forward_sz_analytic` to propagate a covariance
(not a scalar) and solve the closed-form (Lyapunov-type) condition `𝒜(net)∘P₀ = R` so
that `Σ_ŷ = R` ⇒ `J = ½·I` in **every** direction. The current scalar `c_global = σ_v²/A`
is the isotropic special case.

### Upgrade 2 — Diagonal **+ low-rank** posterior (the scalability core)

Represent every layer's weight posterior as

```
 P = D + U Uᵀ ,    D diagonal (the current TAGI state),  U ∈ ℝ^{p×k},  k ≪ p.
```

The exact Gaussian conditioning of a linear layer over a minibatch of `B` observations
adds a **rank-`B` correction** to `P` (each observation subtracts one rank-1 term). TAGI
currently throws that structure away by re-diagonalizing; **keep it** as a rank-`k`
factor (truncate by eigendecomposition of the correction — optimal in Frobenius norm).
The gain needs only `S⁻¹`, available in `O(mk + k³)` by **Woodbury**:

```
 S⁻¹ = M⁻¹ − M⁻¹ B (I_k + Bᵀ M⁻¹ B)⁻¹ Bᵀ M⁻¹ ,   M = (diagonal part),  B = (low-rank part)ᵀHᵀ
```

This is exact Gaussian conditioning on a structured covariance — **no sampling, no
linearization**. It captures the cross-parameter correlations (and the ResBlock
`Cov(f(x),x)` that `var_a += var_s` drops) that the diagonal cannot. Memory `O(pk)`.

### Upgrade 3 — Use the smoother; stop dropping cross-layer covariance

Make the depth-time RTS recursion (§1.2) explicit and **carry the cross-layer
covariance** instead of discarding it. Two concrete consequences:

- **ResBlock merge.** `Sz_out = Sz_branch + Sz_skip` is only correct if branch ⟂ skip.
  They share the input, so the exact merge is `Sz_branch + Sz_skip + 2·Cov`. With a
  low-rank `P` the covariance term is available in closed form — restore it.
- **A *learned* per-layer budget.** The lag-one smoothed covariance `P_{ℓ,ℓ+1}^s` gives
  the closed-form EM update for the per-layer process noise `Q_ℓ` (the local budget).
  Condition (III)'s `c·budget_denom` becomes the warm-start; the smoother provides the
  maximum-evidence value online. No new hyperparameter.

### Upgrade 4 — Adaptive, anisotropic process noise (closed form)

The re-inflation `Q_t` is currently a scalar relaxation. Replace the scalar surprise-`λ`
by **innovation-covariance matching**: require the realized NIS `eᵀ S⁻¹ e` to match its
expectation `m` (the dof), inflating only the directions where the model is surprised
(`S⁻¹` and the direction come from the low-rank `P`). This is the matrix generalization
of the existing `χ²`-driven `λ` — same idea, now per-direction, and still closed-form.

---

## 4. Unit tests — what to expect and why it matters

`mission/scalable_bayes_demo.py` is the executable spec. Each test compares against a
**closed-form** Bayesian ground truth (never Monte-Carlo), so "passing" means
"provably equals exact Bayes / provably approaches it."

| Test | Demonstrates | Why it matters | Observed |
|------|--------------|----------------|----------|
| **T1** | the sequential gain/cov recursion = exact batch posterior | TAGI's per-layer update needs **no linearization** — it is exact | `Δμ=1.1e-15`, `ΔP=3.8e-17` |
| **T2** | diagonal mis-calibrates `S`; `diag+rank-k` → exact, monotone in `k` | the diagonal is the **only** error and a small rank fixes Bayes calibration | diag err **48%** → exact at full rank |
| **T3** | scalar budget ⇒ unbalanced gain; matrix `Σ_ŷ=R` ⇒ `J=½·I` | calibration is a **covariance** condition; trace-matching hides 7× gain imbalance | scalar `0.094–0.722`; matrix `0.5–0.5` |
| **T4** | filter can't credit early layers; smoother = exact posterior | the **backward smoother** (cross-layer covariance) is mandatory for depth | filter `1.0000` (prior); smoother `0.6023` = exact |

New code must keep all four green and add: (a) a covariance-propagation calibration test
asserting `J ≈ ½·I` on a real layer, (b) a low-rank online step test asserting predictive
NIS → 1 as `k` grows, (c) a ResBlock test asserting the merge includes the covariance
term and the deep-stack output `J` stays ≈ ½.

---

## 5. Files & order of work

1. **`triton_tagi/calibrate.py`** — Upgrade 1: a covariance-valued `_forward_sz_cov` that
   propagates `Σ` (reusing the exact per-layer moment maps already there) and the
   closed-form solve `Σ_ŷ = R`. Keep the scalar path as the isotropic special case.
2. **New `triton_tagi/lowrank.py`** — the `P = D + UUᵀ` state: storage, Woodbury gain,
   rank-`k` truncation (eigh of the correction), all as fused Triton kernels mirroring
   the existing `reinflate_*`/`lerp_` style. This is the scalability core (Upgrade 2).
3. **`triton_tagi/layers/resblock.py`** — restore the `+2·Cov` term in the merge using
   the low-rank factor (Upgrade 3). The diagonal-only path stays as the `k=0` fallback.
4. **`triton_tagi/network.py`** — expose the backward pass's smoothed cross-layer
   covariance so Upgrade 3's EM budget and Upgrade 4's anisotropic `Q` can read it.
5. **`mission/scalable_bayes_demo.py`** — extend with the three new tests in §4.

---

## 6. Acceptance criteria

- All four closed-form demonstrations (T1–T4) pass, plus the three new tests.
- The low-rank path reduces to the current diagonal calibration exactly at `k = 0`
  (no regression; `OnlineCalibration` default behaviour unchanged).
- On the ResNet-18 AGVI head, the **matrix** gain `J` has eigenvalues within `±0.02` of
  `0.5` after init (vs the current scalar-only guarantee on the trace).
- Predictive NIS on a held-out batch trends to `1.0` as `k` increases — i.e. the
  network becomes **Bayes-calibrated**, not merely stable.
- Nowhere in the new code is the forward map linearized or sampled: every update is an
  exact closed-form Gaussian conditioning on a (diagonal + low-rank) covariance.

The math is closed-form throughout. The diagonal was a tractability choice, not a
modelling truth; a rank-`k` correction buys back the Bayes calibration that depth and
correlation destroy — which is exactly what the jump to 98% needs.

---

## 7. The classification likelihood, exactly — Pólya–Gamma as closed-form Gaussian conditioning

Everything above assumes a Gaussian observation `R` at the head. But CIFAR-10 is
**categorical**: the label is Bernoulli/multinomial, not Gaussian. The current AGVI head
sidesteps this by regressing one-hot targets with a learned Gaussian noise `E[V²]`. We
can do better — and stay exact — with **Pólya–Gamma (PG) augmentation**, which makes the
classification likelihood *exactly Gaussian in the logit* without linearizing the
sigmoid/softmax.

### 7.1 The identity (no information lost)

For a logit `ψ = xᵀθ` and label `y ∈ {0,1}` (`κ = y − ½`), the Bernoulli likelihood is an
**exact** Gaussian mixture over an auxiliary `ω`:

```
 σ(ψ)^y (1−σ(ψ))^{1−y}  =  ½ · e^{κψ} · E_{ω~PG(1,0)}[ e^{−ωψ²/2} ]
```

Conditional on `ω`, the term `e^{κψ − ωψ²/2}` is Gaussian in `ψ` (hence in `θ`): it is the
quadratic the earlier note wrote as

```
 p(y | ψ) ∝ exp(κ ψ − ω ψ²/2)   ⟹   pseudo-obs  z = κ/ω,   noise  R = 1/ω.
```

The Gaussianity is **conditional**, so nothing is approximated by it — this is the
opposite of a Taylor/Jacobian linearization of the sigmoid.

### 7.2 Why this respects the no-sampling constraint

A Gibbs sampler would *draw* `ω ~ PG`. We never need a draw — only its **closed-form
mean**:

```
 E[ω | ξ] = (1/2ξ)·tanh(ξ/2),     ξ² = E[ψ²] = (xᵀm)² + xᵀP x.
```

Plugging `E[ω]` back is exactly the **Jaakkola–Jordan variational bound**, i.e. a
deterministic coordinate ascent on a provable **evidence lower bound (ELBO)** — principled
variational Bayes, fully closed-form, no Monte-Carlo. The two half-steps are:

- **E-step (closed form):** `ξ_i² = (x_iᵀm)² + x_iᵀP x_i`, then `ω_i = (1/2ξ_i)tanh(ξ_i/2)`.
- **M-step (TAGI's own exact Gaussian update):** condition on `z_i = κ_i/ω_i` with noise
  `R_i = 1/ω_i`, giving `P⁻¹ = P₀⁻¹ + Σ_i ω_i x_i x_iᵀ`, `m = P(P₀⁻¹m₀ + Σ_i κ_i x_i)`.

So `z = κ/ω, R = 1/ω` is precisely the M-step pseudo-likelihood — it slots into the same
conditioning kernel as every other layer. The `ω` are a per-sample, per-class
**heteroscedastic precision computed in closed form from the logit's own moments** — the
principled counterpart of AGVI's learned `E[V²]`, specialized to the categorical
likelihood.

### 7.3 Multi-class

CIFAR-10 is 10-way. Two closed-form routes, both PG-augmentable:

- **Stick-breaking** (Linderman–Johnson–Adams): factor the `K`-way categorical into
  `K−1` conditionally-independent binary logits, each PG-augmented as above. Exact
  mean-field, `K−1` heads.
- **One-vs-each** bound: a sum of binary terms upper-bounding the softmax partition;
  again PG-augmentable. Simpler to wire to the existing per-class head.

Either replaces the regression-style AGVI head with a likelihood that *matches the data
type*, which is where the last points of calibration/accuracy on CIFAR-10 hide.

### 7.4 Proof of concept — `mission/polya_gamma_classification_demo.py`

Validated against a **deterministic quadrature** posterior (the exact reference, never
random):

| Proof | Claim | Observed |
|-------|-------|----------|
| **P1/P2** | PG-induced Gaussian is a valid lower bound on `σ`, tight at `ψ=ξ`; `E[ω]=2λ(ξ)` | min gap `≥ −1e-16`; gap at `ψ=ξ` `≤ 1e-7` |
| **P3** | deterministic PG-VB → exact Bayesian-logistic posterior; ELBO monotone | `Δmean = 0.053`, **predictive MAE = 0.0029**, ELBO `−37.3 → −33.1` ↑ |
| **P4** | properly Bayes-calibrated: posterior contracts with data | `tr(P)`: `0.78 → 0.18 → 0.04 → 0.01` as `n: 20→1280` |

`python mission/polya_gamma_classification_demo.py` reproduces all of it. No sampling, no
linearization — only `tanh`, the closed-form PG mean, and TAGI's exact Gaussian
conditioning.

### 7.5 Where it plugs in

This is **Upgrade 4b**, sitting at the output head:

5. **`triton_tagi/layers/` (new head) + `triton_tagi/calibrate.py`** — replace the
   AGVI/EvenSoftplus regression head with a PG categorical head: emit logits, compute
   `ω = (1/2ξ)tanh(ξ/2)` from the logit moments (closed form), and feed `z=κ/ω, R=1/ω`
   into the existing innovation/update kernels. Condition (III) generalizes cleanly — the
   output innovation covariance is now `S = Σ_ŷ + diag(1/ω)`, and the matrix calibration
   of §3 (Upgrade 1) still targets `J = ½·I`, with `R` now the closed-form PG precision
   instead of a hand-set `σ_v²`.

**Acceptance (addendum to §6).** The PG head must (a) keep P1–P4 green, (b) reduce to a
Bernoulli/categorical innovation that matches the quadrature predictive within MAE `<0.01`
on a held-out batch, and (c) introduce no sampled or differentiated quantity — `ω` is
closed form, the update is exact Gaussian conditioning.
