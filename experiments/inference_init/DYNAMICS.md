# Training Dynamics in TAGI: A Problem Statement

## Overview

This document identifies a fundamental open problem in the practical deployment of
Tractable Approximate Gaussian Inference (TAGI) for deep neural networks: the three
main design choices that govern training stability — weight initialization, observation
noise, and the parameter update regularization — are deeply coupled, yet are currently
treated as independent hyperparameters. This coupling is not merely a nuisance; it
reflects the underlying inference dynamics of the algorithm. A principled theory of
these dynamics does not yet exist, and its absence forces practitioners to compensate
empirically in ways that are brittle, depth-dependent, and poorly understood.

---

## Background: The Irreversibility of Posterior Contraction in TAGI

TAGI frames neural network training as sequential Bayesian inference. At each step,
the posterior over the parameters is updated using a Kalman-style rule. A defining
characteristic of this process is that the posterior variance is **monotonically
non-increasing**: once epistemic uncertainty is consumed by an update, it cannot be
recovered. There is no mechanism analogous to gradient noise or learning rate warmup
that injects uncertainty back into the system.

This property has a direct consequence: the **prior distribution at initialization is a
hard ceiling on the model's exploratory capacity throughout all of training**. If the
prior is miscalibrated — too narrow, too wide, or statistically misaligned with the
data — the model's ability to learn is compromised from the very first update, and
there is no way to undo this from within the training procedure itself.

This makes initialization not a secondary engineering concern, as it is often treated
in deterministic deep learning, but a **first-class inference decision**.

---

## The Three Coupled Quantities

In practice, training a TAGI network requires specifying three quantities:

**1. The initial weight distribution (initialization)**

The initial prior variance on the weights, `S_w`, determines the scale of the
prior and the magnitude of the initial Kalman gain. Current practice borrows
initialization schemes from deterministic deep learning (He, Xavier), which were
designed to stabilize gradient flow, not Bayesian updates. These schemes have
no principled relationship to the TAGI update rule and offer no guarantees about
the behavior of the inference dynamics.

**2. The observation noise standard deviation, `sigma_v`**

`sigma_v` parameterizes how much the model trusts each observed data point. It
appears directly in the innovation covariance `S_z + sigma_v²`, which is the
denominator of the Kalman gain. A small `sigma_v` produces a large gain and
aggressive updates; a large `sigma_v` dampens updates and slows learning. In
current practice, `sigma_v` is treated as a scalar hyperparameter tuned by
trial and error. There is no principled rule for choosing it, and it is
routinely set to values that are large enough to prevent divergence — not
because those values are theoretically justified, but because they happen to
work.

**3. The parameter update cap factor**

The cap factor limits the magnitude of each parameter update relative to the
current posterior standard deviation. In the cuTAGI reference implementation,
it is a lookup table that returns a scalar based on batch size alone (0.1 for
batch=1, 2.0 for small batches, 3.0 for large batches). This is a purely
empirical heuristic with no derivation from the inference dynamics. It exists
to prevent large Kalman gains from destabilizing training, but it operates
independently of both the initialization scale and the value of `sigma_v`.

---

## The Coupling Problem

These three quantities are not independent. They interact through the Kalman gain
at every layer on every update step:

```
K  ~  S_w · E[μ_a²] / (S_z + sigma_v²)
```

The gain determines how aggressively the posterior contracts. Its magnitude depends
simultaneously on the current weight variance `S_w` (set by initialization and
modified by all previous updates), the expected activation energy `E[μ_a²]` (set by
the data and the current network state), and the observation noise `sigma_v²`.

The cap factor then limits how much of this gain is actually applied. But the
"correct" cap — the one that prevents instability without unduly slowing learning —
depends on the same quantities: the scale of `S_w`, the magnitude of the gain, and
the size of `sigma_v`.

Currently, none of these interdependencies are accounted for. The result is a system
where:

- `sigma_v` is tuned large enough to suppress the instability caused by a
  miscalibrated initialization
- The cap factor is set large enough to suppress the instability caused by a
  miscalibrated `sigma_v`
- Initialization is copied from deterministic methods that have no awareness of
  either `sigma_v` or the cap

Each quantity is compensating for the others. The system works, in the sense that
training does not diverge, but it works for the wrong reasons. The three quantities
have absorbed each other's errors rather than reflecting a coherent inference regime.

---

## Empirical Evidence of the Problem

The coupling manifests clearly in practice. Across several experiments on MNIST
classification with a multi-layer perceptron:

- At moderate depth (9 layers) and `sigma_v = 0.05`, neither He initialization
  nor Inference-Based Initialization (IBI) collapses, and the differences between
  them are negligible. The large `sigma_v` masks any initialization advantage.

- At the same depth with `sigma_v = 0.01`, IBI produces a marginal improvement
  over He (≈ 0.3%), suggesting that initialization begins to matter when
  `sigma_v` is tighter, but the effect is small.

- At depth 20 with `sigma_v = 0.01`, both He and IBI collapse completely to
  chance performance (≈ 11%). IBI, which provably calibrates the forward-pass
  statistics of every layer, provides no benefit. The collapse happens regardless
  of how well the prior is initialized.

This last result is the most revealing. IBI is working correctly — the layer
statistics are well-calibrated before training begins. Yet training fails anyway.
The only explanation is that the **update dynamics**, not the initialization, are
responsible for the collapse at depth. The Kalman gain propagated through 20 layers
under `sigma_v = 0.01` and the current cap factor overwhelms the carefully
constructed prior within the first few training steps.

---

## Why a Theory Is Needed

Without a theory of the update dynamics, the following questions cannot be answered
from first principles:

1. **What is the correct `sigma_v` for a network of given depth and width?**
   Currently there is no answer. The value is tuned per-experiment and per-depth.
   A theory should derive `sigma_v` from the statistical properties of the
   network — ideally from the same parameters that govern the initialization.

2. **What is the correct cap factor?**
   The batch-size heuristic is a proxy for something we do not understand. The
   cap exists because the Kalman gain can be too large, but we have no formula
   for how large is too large, and no derivation of the threshold from the
   network's parameters.

3. **How do the update dynamics interact with depth?**
   The variance decay rate at each layer depends on the activation statistics at
   that layer, which in turn depend on the weights in all previous layers. The
   dynamics are coupled across layers in a way that is not currently characterized.
   A theory should describe how the decay rate accumulates with depth and under
   what conditions the overall system remains in a stable learning regime.

4. **Is there a regime where all three quantities are jointly consistent?**
   The empirical evidence suggests that there is a well-defined operating regime —
   a region of (initialization scale, `sigma_v`, cap factor) space where TAGI
   networks train stably and efficiently regardless of depth. Whether such a
   regime exists, what characterizes it, and how to reach it from first principles
   is entirely open.

5. **How does the choice of output layer and loss interact with the dynamics?**
   The observation noise `sigma_v` is defined at the output layer, but its effect
   propagates through the entire network via the backward pass. The structure of
   the output (e.g., a plain linear layer versus a Bayesian softmax like Remax)
   changes the effective innovation signal. This interaction is not characterized.

---

## What Is Not Being Claimed

This document does not claim to have answers to the above questions. The decay rate
equation and the SNR intuition sketched informally here are heuristics, not a
rigorous theory. They are suggestive of the right structure — that `sigma_v` should
be set relative to the initialization scale, and that the cap factor should follow
from that — but they have not been derived rigorously, validated formally, or
extended to the multi-layer setting.

The goal of the research program outlined here is to build that theory from the
ground up: starting from the single-layer update dynamics, characterizing the
conditions for stability, extending to the multi-layer case under the TAGI diagonal
approximation, and ultimately deriving practical initialization rules that make
explicit the joint constraints on all three quantities.

---

## Open Questions to Drive the Research

- Can the per-step variance decay rate be characterized analytically as a function
  of `S_w`, `sigma_v`, and the layer-wise activation statistics?
- Under what conditions does the product of decay rates across layers lead to
  network-level collapse, and how does this scale with depth?
- If IBI is used to fix the activation statistics before training, does this
  decouple the layer dynamics in a way that makes the multi-layer analysis
  tractable?
- Is there a natural parameterization of (`sigma_v`, cap factor) in terms of the
  IBI hyperparameters (`sigma_m`, `sigma_z`) that yields stable dynamics at any
  depth?
- How does the diagonal covariance approximation in TAGI affect the theoretical
  stability conditions relative to the exact Kalman filter?
