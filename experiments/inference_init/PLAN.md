# Inference-Based Initialization (IBI) — Plan

Status: **draft 2026-04-23** — algorithm derived from Contribution 3 of the
user's thesis. Re-implementing from scratch (existing archived code in
`_archive/triton_tagi/inference_init.py` deliberately not consulted).

This is a research track, not part of the library's parity goal. It lives
under `experiments/` so it can grow figures, sweeps, and notebooks without
bloating the main library plan.

---

## 1. Goal

Replace He initialization in TAGI training with a **pre-training calibration
phase** that, for each layer $l$, drives the empirical hidden-unit moments
$(\mu_{Z_i}, \sigma_{Z_i}^2)$ toward a prescribed target $(\sigma_M, \sigma_Z)$
using one epoch of forward-and-correct passes over the data.

Acceptance for V1: reproduce Figure 4 of the thesis on MNIST+MLP.
Specifically:

- $\mathtt{L}=7$, $\sigma_V=0.01$, $(\sigma_M=1.0, \sigma_Z=0.5)$ → ~96.8%
  (vs He → 14.5%)
- $\mathtt{L}=5$, $\sigma_V=0.05$, $(\sigma_M=0.5, \sigma_Z=0.5)$ → ~97.7%
  (vs He → collapse)
- He baseline: matches the thesis numbers within run-to-run variance

---

## 2. Algorithm (one batch, one layer)

Notation: $A^{(l)}$ = layer $l$ width. $Z_i^{(l)}$ = pre-activation unit $i$.
Targets are derived from $(\sigma_M, \sigma_Z)$ (global hyperparameters).

### 2.1 Per-layer targets (depend only on $A$, $\sigma_M$, $\sigma_Z$)

$$
\mu_{\tilde S} = 0,\quad \sigma_{\tilde S}^2 = A\,\sigma_Z^2
$$
$$
\mu_{\tilde{S2}} = A(\sigma_M^2 + \sigma_Z^2),\quad
\sigma_{\tilde{S2}}^2 = A(2\sigma_Z^4 + 4\sigma_M^2\sigma_Z^2)
$$

### 2.2 Forward to layer $l$

Get current $(\mu_{Z_i}, \sigma_{Z_i}^2)$ from a TAGI forward pass on the
current batch through (already-calibrated) layers $1..l-1$ then layer $l$.

### 2.3 S projection (moment-matching, diagonal innovation)

$$
\mu_S = \sum_i \mu_{Z_i},\quad \sigma_S^2 = \sum_i \sigma_{Z_i}^2
$$
$$
\delta\mu_S = (\mu_{\tilde S} - \mu_S)/\sigma_S^2,\quad
\delta\sigma_S^2 = (\sigma_{\tilde S}^2 - \sigma_S^2)/\sigma_S^4
$$
$$
\mu_{Z_i|S} = \mu_{Z_i} + \sigma_{Z_i}^2 \delta\mu_S,\quad
\sigma_{Z_i|S}^2 = \sigma_{Z_i}^2 (1 + \sigma_S^2 \delta\sigma_S^2)
$$

### 2.4 S2 RTS update (quadratic obs, applied AFTER S)

Using the post-S moments as the new $\mu_{Z_i}, \sigma_{Z_i}^2$:

$$
\mu_{Z_i^2} = \mu_{Z_i}^2 + \sigma_{Z_i}^2,\quad
\sigma_{Z_i^2}^2 = 2\sigma_{Z_i}^4 + 4\sigma_{Z_i}^2\mu_{Z_i}^2
$$
$$
\mu_{S2} = \sum_i \mu_{Z_i^2},\quad \sigma_{S2}^2 = \sum_i \sigma_{Z_i^2}^2,\quad
J_i = 2\mu_{Z_i}\sigma_{Z_i}^2 / \sigma_{S2}^2
$$
$$
\delta\mu_{S2} = (\mu_{\tilde{S2}} - \mu_{S2})/\sigma_{S2}^2,\quad
\delta\sigma_{S2}^2 = (\sigma_{\tilde{S2}}^2 - \sigma_{S2}^2)/\sigma_{S2}^4
$$
$$
\mu_{Z_i|S2} = \mu_{Z_i} + 2\mu_{Z_i}\sigma_{Z_i}^2 \delta\mu_{S2},\quad
\sigma_{Z_i|S2}^2 = \sigma_{Z_i}^2 + (2\mu_{Z_i}\sigma_{Z_i}^2)^2 \delta\sigma_{S2}^2
$$

After 2.3 + 2.4 we have the calibrated targets $(\mu_{Z_i|\cdot}, \sigma_{Z_i|\cdot}^2)$.

### 2.5 Decoupled inverse on layer params

$$
\gamma_i = \sqrt{\sigma_{Z_i|\cdot}^2 / \sigma_{Z_i}^2}
$$
$$
\mu_{W_{ji}} \leftarrow \gamma_i \mu_{W_{ji}},\quad
\sigma_{W_{ji}}^2 \leftarrow \gamma_i^2 \sigma_{W_{ji}}^2,\quad
\sigma_{B_i}^2 \leftarrow \gamma_i^2 \sigma_{B_i}^2
$$
$$
\tilde\mu_{Z_i} = \gamma_i (\mu_{Z_i} - \mu_{B_i}) + \mu_{B_i},\quad
\Delta\mu_{Z_i} = \mu_{Z_i|\cdot} - \tilde\mu_{Z_i}
$$
$$
\mu_{B_i} \leftarrow \mu_{B_i} + \Delta\mu_{Z_i}
$$

### 2.6 Outer loop

```
for batch in dataloader:                 # one epoch
    ma, Sa = preprocess(x)
    for layer in net:
        if isinstance(layer, LearnableLayer):
            mz, Sz = layer.forward(ma, Sa)            # uncalibrated forward
            calibrate_layer(layer, mz, Sz, sigma_m, sigma_z)
            mz, Sz = layer.forward(ma, Sa)            # re-forward with new params
        else:
            mz, Sz = layer.forward(ma, Sa)            # passthrough (ReLU, etc.)
        ma, Sa = mz, Sz
```

S target is hit exactly per batch (analytical projection). S2 target is
only approached asymptotically across the full dataset because the S2 RTS
update is a linearization of a quadratic observation.

---

## 3. Design decisions (open for review)

These are choices I'll make for V1 unless you say otherwise:

**D1. Aggregation across the batch.** Forward pass gives per-sample
$(\mu_{Z_i}, \sigma_{Z_i}^2)$ of shape $(B, A)$. Two interpretations:

- **(a)** Per-sample S/S2 projection → $(B, A)$ corrected moments → batch-mean
  for the inverse step (single $\gamma_i$, $\Delta\mu_{Z_i}$ per output unit).
- **(b)** Batch-mean first → single $(\mu_{Z_i}, \sigma_{Z_i}^2)$ per output
  unit → S/S2 projection on those scalars → inverse on those scalars.

**Default: (b)** (user decision, 2026-04-23). Per-sample projection is too
expensive; batch-mean-first is the scalar-vector form the §2 formulas are
already written in. Mean over batch is taken on the raw $(\mu_{Z_i}, \sigma_{Z_i}^2)$
entering §2.3; everything after that is per-unit scalars of shape $(A,)$.

**D2. Output layer target — regression.** Same $(\sigma_M, \sigma_Z)$ as
hidden layers. Symmetric, simple.

**D3. Output layer target — classification (Remax head).** User wants
the post-Remax distribution to be uniform $1/C$ at the first batch.
Options:

- **(a)** Calibrate the Linear *before* Remax to a target derived from
  inverting Remax at the uniform output (closed-form for MixtureReLU? probably
  not; would need a numerical / Monte Carlo back-derivation).
- **(b)** Skip the Remax-head calibration in V1; apply standard target on the
  pre-Remax Linear. Verify post-Remax distribution empirically.

**Default: (b)** for V1 — get the MNIST-MLP result first, then revisit.

**D4. Bias prior $\alpha$.** Use whatever the layer's existing init produces.
Don't sweep $\alpha$ in V1.

**D5. Numerical guards.**
- $\sigma_{Z_i}^2 < \epsilon$ → skip the inverse update for unit $i$ (γ undefined).
- $\sigma_S^2, \sigma_{S2}^2 < \epsilon$ → skip the S/S2 update for that batch.

**D6. ReLU passthrough.** ReLU has no learnable params; the calibration
walks past it without modification. The next Linear sees post-ReLU
$(\mu_A, \sigma_A^2)$ as input.

---

## 4. File layout

```
experiments/inference_init/
    PLAN.md                  ← this file
    mnist_mlp_sweep.py       ← reproduces thesis Figure 4 heatmap
    figures/                 ← sweep output (PDF/PNG)
    results/                 ← run logs / .json metrics
triton_tagi/
    inference_init.py        ← the algorithm (lives in the library; clean
                                public API: `inference_init(net, loader,
                                sigma_m, sigma_z)`)
tests/
    unit/test_inference_init.py    ← per-step correctness:
                                     - S projection lands on target
                                     - S2 RTS Jacobian sanity
                                     - decoupled inverse round-trip
    validation/test_ibi_mnist.py   ← end-to-end: 7-layer MLP MNIST,
                                     (σ_M=0.5, σ_Z=0.5), σ_V=0.05,
                                     L=5 → ≥97% (slow, marked @pytest.mark.slow)
```

---

## 5. Phasing

### Phase 1 — Linear + ReLU MLP (this PLAN's V1)

- Implement `inference_init.py` with `Linear` calibration only.
- Pass-through for `ReLU`, `Flatten`, `Remax` (the latter just for completeness;
  per D3, no special treatment in V1).
- Sweep MNIST MLP at L ∈ {1, 3, 5, 7} × $(\sigma_M, \sigma_Z) \in \{0.5, 1.0\}^2$
  × $\sigma_V \in \{0.01, 0.05\}$. Reproduce Figure 4.
- Done when the heatmap qualitatively matches the thesis.

### Phase 2 — Conv2D + BatchNorm2D (implemented 2026-05-08)

**D7. Conv2D width.** $A = C_{\text{out}}$, batch-aggregate over
$N \cdot H_{\text{out}} \cdot W_{\text{out}}$. The $C_{\text{out}}$ channels at
one spatial position are exactly the "layer of width $A$" the algorithm
assumes (shared weight column across positions, per-channel bias). The inverse
update is per-output-channel — same math as Linear, with $\gamma$ broadcast as
$(1, C_{\text{out}})$ over the patch-row axis of $\mu_W \in \mathbb{R}^{K \times C_{\text{out}}}$.

The other interpretations were rejected:
- $A = C_{\text{out}} \cdot H \cdot W$: breaks weight sharing — would require
  per-position $\gamma$, but Conv2D has only $C_{\text{out}}$ scaling
  parameters.
- Per-spatial-position with $A = C_{\text{out}}$: same as the chosen
  formulation if you batch-aggregate, just stated differently.

**D8. BatchNorm2D.** Treat $\text{out} = \gamma \cdot \hat z + \beta$ as
structurally identical to Linear/Conv2D from the inverse-update perspective:
$A = $ num_features (= $C$), batch-aggregate over $N \cdot H \cdot W$, scale
$\gamma$'s mean and variance by $\gamma_c$, $\gamma_c^2$ and shift $\beta$'s
mean by $\Delta\mu_c$. Identical algebra to Linear (verified: scaling the
scalar parameter $\mu_\gamma$ by $\gamma_{c}$ scales the output variance by
$\gamma_c^2$).

**D9. BN training mode during IBI.** Run IBI in train mode so BN computes batch
stats (eval mode would normalize against zero-initialized running stats and
defeat the purpose of BN). Each BN forward is called twice per batch (probe +
re-forward), which roughly doubles the effective EMA momentum on running
stats during IBI. Benign — running stats refresh quickly during real training.

- Validate on MNIST CNN, then CIFAR-10 CNN.

### Phase 3 — ResNet-18 (implemented 2026-05-08)

**D10. ResBlock — option (a): per-sub-layer calibration.** Recurse into the
block: walk `main_layers` (Conv→ReLU→BN→Conv→ReLU→BN), then `proj_layers` if
present, calibrating each Conv2D / BN as if standalone. ReLU is pass-through.
Then sum into the merged moments and continue.

Option (b) (per-block) deferred — would require deriving an inverse for
$(\text{main} + \text{skip})$ jointly, which is no longer per-layer.

**Caveat — residual doubling.** After the add, variance is roughly $2\times$
either path alone. The next block's first conv still calibrates its OUTPUT to
$\sigma_M, \sigma_Z$ regardless of input scale, so this isn't catastrophic —
but it means the prior is implicitly enforced "post-add", not "post-conv".
First knob to revisit if ResNet18+IBI underperforms He.

- Validate against the current 89% CIFAR-10 baseline.

---

## 6. Out of scope for V1

- MultiheadAttentionV2 / LayerNorm / RMSNorm calibration (transformer track).
- Inverse-Remax target derivation (D3 alternative).
- Universal $(\sigma_M, \sigma_Z)$ search across architectures.
- Bias-prior $\alpha$ sweep.
- Multi-epoch calibration (paper says one epoch; we follow that).

---

## 7. Open empirical questions (post-V1, defer)

- Does a universal $(\sigma_M, \sigma_Z)$ exist across architectures?
- Does multi-epoch calibration help, or hurt (over-fitting the prior to the
  calibration set)?
- For deep networks under high $\sigma_V$, is the ReLU-clipping bottleneck
  fundamental or fixable by clipping $\sigma_M$ adaptively per layer?
- Does IBI compose with later regularization (Contribution 3 §regularization,
  not yet read)?
