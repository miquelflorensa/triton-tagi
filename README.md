# triton-tagi

**Tractable Approximate Gaussian Inference (TAGI) for Bayesian Neural Networks, in Python + Triton.**

A minimal, GPU-accelerated reimplementation of [cuTAGI](https://github.com/lhnguyen102/cuTAGI)
(C++/CUDA) with numerical parity on its headline examples and fused
[Triton](https://triton-lang.org/) kernels for the hot paths.

> **Idea.** TAGI treats every weight and activation as a Gaussian. The forward
> pass propagates `(mean, variance)` analytically through each layer; the
> backward pass applies closed-form Bayesian updates to the parameters. No
> sampling, no variational bounds, no autograd.

---

## Status

- **Library version:** 0.2.0 (scoped down 2026-04-19 to a minimal cuTAGI-parity core).
- **Parity:** every kept example reproduces its cuTAGI counterpart at Phase-1
  tolerance; see [`PLAN.md`](PLAN.md) §3 for the table.
- **Tests:** 95 unit + 89 validation pass on CUDA.
- **Archive:** layers, optimizers, and diagnostics outside the minimal scope
  live under [`_archive/`](_archive/); nothing deleted.

### Recent milestones

- **2026-04-23.** Per-call dispatch overhead in `kernels/attention.py` cut
  17–24% by replacing `@triton.autotune` with a shape-adaptive `_pick_blocks`
  heuristic; attention validation suite (18 tests) still passes. CIFAR-10
  ResNet-18 + Remax reaches **89%** test accuracy after exact cuTAGI parity.
- **2026-04-22.** Self-attention added (`Embedding`, `PositionalEncoding`,
  `MultiheadAttentionV2`, `RMSNorm`) plus the `reverse_predictor` example —
  sequence reversal with sinusoidal PE + MHA-V2 + RMSNorm + HRC head,
  matching cuTAGI's `feat/attn-debug` branch.
- **2026-04-20.** Remax reimplemented to match cuTAGI's MixtureReLU plus
  log-normal covariance path. CIFAR-10 ResNet-18 + Remax now trains to ≥80%
  test accuracy in ~15 epochs with `gain_w=gain_b=0.1` and σ_v ∈ {0.01, 0.05}.

---

## Install

```bash
git clone https://github.com/miquelflorensa/triton-tagi.git
cd triton-tagi
pip install -e .           # core: torch, triton, numpy
pip install -e ".[vis]"    # + matplotlib for figures
pip install -e ".[dev]"    # + pytest, ruff
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0 with CUDA, and Triton ≥ 2.0.

---

## Quick start

```python
import torch
from triton_tagi import Linear, ReLU, Remax, Sequential

device = torch.device("cuda")
net = Sequential(
    [
        Linear(784, 256, device=device),
        ReLU(),
        Linear(256, 128, device=device),
        ReLU(),
        Linear(128, 10, device=device),
        Remax(),
    ],
    device=device,
)

# One closed-form Bayesian update step
y_pred_mu, y_pred_var = net.step(x_batch, y_batch_onehot, sigma_v=0.05)
```

Every example under [`examples/`](examples/) follows the same `RunDir` and
`argparse` convention (see [PLAN.md](PLAN.md) §3.1):

```bash
python examples/mnist_mlp.py --n_epochs 5
python examples/mnist_cnn.py
python examples/cifar10_cnn.py --n_epochs 100
python examples/cifar10_resnet18.py --n_epochs 100 --gain_w 0.1 --gain_b 0.1
python examples/cifar10_resnet18_hrc.py
python examples/cifar10_resnet18_probitree.py --smoke --no-augment
python examples/regression.py
python examples/regression_heteros.py
python examples/reverse_predictor.py    # sinusoidal PE + MHA-V2 + RMSNorm + HRC head
python examples/custom_layer.py         # tutorial: write your own Triton layer
```

---

## Library surface

Everything lives under `triton_tagi/`. The package is deliberately small;
reading it top-to-bottom in an evening is a goal, not an accident.

### Layers (`triton_tagi/layers/`)

| Layer | Used by |
|---|---|
| `Linear` | MLP, CNN/ResNet heads |
| `Conv2D` | CNN, ResNet |
| `BatchNorm2D` | CNN, ResNet |
| `LayerNorm` | MLP variant |
| `AvgPool2D`, `MaxPool2D` | CNN, ResNet stem/head |
| `ReLU` | all non-linear examples |
| `Flatten` | conv→FC boundary |
| `Remax` | classification head (cuTAGI-native) |
| `ResBlock` + `Add` | ResNet-18 |
| `EvenSoftplus` | heteroscedastic regression noise head |
| `Embedding` | reverse_predictor (token → vector) |
| `PositionalEncoding` | reverse_predictor (sinusoidal, fixed) |
| `MultiheadAttentionV2` | reverse_predictor (separate Q/K/V projections, Remax over scores) |
| `RMSNorm` | reverse_predictor |

### Top-level

- `base.py`: `Layer`, `LearnableLayer` ABCs.
- `network.py`: `Sequential` (forward / step / train / eval).
- `param_init.py`: He / Xavier / Gaussian init.
- `hrc_softmax.py`: hierarchical softmax output for many-class classification.
- `hrc_probit.py`: full K-leaf trees and the unit-probit class likelihood,
  plus the scale ablations it is compared against.
- `probitree.py`: exact-K balanced `ProbiTree` construction, stable direct
  probit moment matching, log-space prediction, and TAGI output messages.
- `cdf_remax.py`: Laplace-Remax mixture kernels, conditional and scale moments,
  the epistemic/aleatoric decomposition, and forward diagnostics.
- `cdf_variance.py`: exact Gaussian moments of the bounded variance activation
  `h(u) = eps + kappa*Phi(u)`.
- `remax_kernels.py`: the `phi(alpha) I_n(beta)` kernel forms the Remax moments
  are built on, with the negative-tail series the textbook forms lose to
  cancellation.
- `remax_scale.py`: `LogScalePosterior` plus the ADF and grid fits for the
  shared Remax log-scale.
- `checkpoint.py`: `RunDir` (run-directory manager) and `load_model`.
- `kernels/common.py`: fused Triton kernels for Linear / Conv2D / BN.
- `kernels/attention.py`: fused Triton kernels for `MultiheadAttentionV2`
  (`bmm_tagi_var` for the QKᵀ / Score@V variance, `bmm_shared_left/right`
  for the four backward reductions).
- `update/observation.py`, `update/parameters.py`: innovation and parameter update rules.

---

## How TAGI works (one page)

Every weight `W` and activation `a` is a Gaussian random variable described
by its mean `μ` and variance `σ²`.

**Forward (moment propagation).** For `z = a W + b`,

$$\mu_z = \mu_a\,\mu_W + \mu_b$$

$$\sigma^2_z = \mu_a^2\,\sigma^2_W + \sigma^2_a\,\mu_W^2 + \sigma^2_a\,\sigma^2_W + \sigma^2_b$$

Nonlinearities propagate moments analytically. ReLU uses the Alric (2024)
closed-form; Remax uses MixtureReLU plus log-normal identities, matching cuTAGI.

**Backward (observation innovation).** At the output,

$$\delta_\mu = \frac{y - \mu_z}{\sigma^2_z + \sigma^2_v}, \qquad \delta_\sigma = \frac{-1}{\sigma^2_z + \sigma^2_v}.$$

Deltas propagate backward through each layer; no autograd.

**Parameter update (capped, cuTAGI-style).**

$$\mu_W^{\text{new}} = \mu_W + \sigma^2_W \cdot \Delta_\mu$$

$$\sigma^{2,\text{new}}_W = \max\!\left(\sigma^2_W + (\sigma^2_W)^2 \cdot \Delta_\sigma,\;\epsilon\right)$$

---

## Writing a custom layer

Subclass `Layer` (pure moment propagation) or `LearnableLayer` (adds `update`
and `num_parameters`), implement `forward` and `backward`, and you are done.
No registry, no decorators. See [`examples/custom_layer.py`](examples/custom_layer.py)
for an end-to-end ELU tutorial: Triton kernel, `Layer` subclass, MNIST run.

---

## Tests

```bash
pytest tests/unit                 # ~95 tests, fast
pytest tests/validation           # ~89 tests; compares to pytagi reference
pytest -m "not slow and not cuda" # CPU subset
```

Validation tests assert `torch.testing.assert_close(atol=1e-5, rtol=0)`
against a pytagi reference run on the same batch.

---

## Benchmarks

See [`benchmarks/results.md`](benchmarks/results.md). Summary on an RTX 4070
Ti SUPER, median of 50 runs:

| Layer / batch 1024 | triton-tagi | cuTAGI | Speedup |
|---|---:|---:|---:|
| Linear(512, 512) | 0.64 ms | 45.0 ms | **70×** |
| Conv2D net | 11.0 ms | 106 ms | **9.7×** |
| BatchNorm2D net | 12.5 ms | 109 ms | **8.7×** |

triton-tagi wins on throughput; cuTAGI wins on small-batch latency, where
dispatch overhead dominates.

---

## Relation to cuTAGI

This is a Triton-based reimplementation of
[cuTAGI](https://github.com/lhnguyen102/cuTAGI), the reference C++/CUDA TAGI
library. Parity is load-bearing: every kept example must reproduce the cuTAGI
result at Phase-1 tolerance on a fixed seed. In particular:

- **Capped parameter updates** with batch-size-dependent cap factors.
- **Backward order:** compute deltas first, apply capped updates after.
- **ResBlock** identical to cuTAGI's `ResNetBlock` (projection shortcut).
- **Remax** uses cuTAGI's MixtureReLU plus log-normal covariance path, not a
  Softplus+Taylor approximation. See `triton_tagi/layers/remax.py`.

TF32 matmul is disabled at import time: cuTAGI uses scalar FMA with near-fp64
accuracy, so leaving TF32 on would introduce systematic ~1e-3 variance errors
and break parity.

---

## References

- Goulet, J.-A., Nguyen, L. H., & Amiri, S. (2021). *Tractable Approximate
  Gaussian Inference for Bayesian Neural Networks*. JMLR 22(228), 1–23.
  [[paper]](https://www.jmlr.org/papers/v22/20-1009.html)
- Alric, L. (2024). Closed-form MixtureReLU moments.
- cuTAGI: [github.com/lhnguyen102/cuTAGI](https://github.com/lhnguyen102/cuTAGI).
- Triton: [triton-lang.org](https://triton-lang.org/).

## Frozen PyTorch backbone + TAGI heads

Install the example dependency, train a conventional deterministic model, then
compare its softmax probabilities with TAGI Remax and HRC heads trained on
frozen penultimate features or logits:

```bash
pip install -e ".[examples,vis]"
python examples/train_cifar10_resnet18_torch.py \
  --output runs/deterministic_cifar10_resnet18.pt
python examples/compare_cifar10_tagi_heads.py \
  --checkpoint runs/deterministic_cifar10_resnet18.pt
```

A fast end-to-end check uses `--smoke` on both commands. Laplace-Remax is
available without changing the default cuTAGI-parity lognormal comparison:

```bash
python examples/compare_cifar10_tagi_heads.py \
  --checkpoint runs/deterministic_cifar10_resnet18.pt \
  --remax-approximation laplace --remax-jacobian diag
```

The comparison writes accuracy, 15-bin ECE, multiclass NLL, Brier score, and
SVHN OOD AUROC/AUPR/FPR95 using predictive entropy and negative maximum
probability. It does not fit temperature scaling or any other post-hoc
calibration. For another PyTorch architecture, construct it explicitly, load
it with `load_torch_checkpoint`, and give `FrozenTorchBackbone` a callback that
returns `(features, logits)`.
The reusable API accepts cached feature tensors directly:

```python
from triton_tagi import TAGILastLayerClassifier

head = TAGILastLayerClassifier(
    features.shape[1],
    100,
    head="remax_laplace_diag",
    sigma_v=0.05,
    gain_w=0.1,
    gain_b=0.1,
)
history = head.fit(
    features,
    labels,
    epochs=20,
    validation=(validation_features, validation_labels),
)
prediction = head.predict(test_features)
```

Choose `probit_ovr`, `remax_lognormal`, `remax_laplace_diag`, `hrc`,
`hrc_probit`, `probitree`, `cdf_remax`, `multinomial_probit`, `agci`, `agci_remax`,
`gumbel_agci`, `logit_site`, `ct_agci`, `categorical_tagiv`, `hrc_tagiv`, or
`logit_tagiv`.
The
`hrc_probit` head fixes its latent link variance at one and does not accept
`sigma_v`. TAGI-V heads omit
`sigma_v` because they learn their observation-variance channel.

### Hierarchical probit

`head="hrc_probit"` is a hierarchical classifier with no free scale. Each
decision node of a full K-leaf tree carries a latent variable with unit probit
noise,

    Z_j | x, D ~ N(mu_j, S_j),   R_j = Z_j + o_j + eps_j,   eps_j ~ N(0, 1),

so the class probability is

    p(c | x, D) = prod_{j in path(c)} Phi( s_cj (mu_j + o_j) / sqrt(S_j + 1) ),

and these sum to one without normalization. Training uses the matching probit
moment update: with `d = sqrt(S_j + 1)`, `gamma = s_cj (mu_j + o_j) / d` and
`lambda = phi(gamma) / Phi(gamma)`, the innovations are `delta_mu = s lambda /
d` and `delta_S = -lambda (lambda + gamma) / d^2`. That is the whole
observation model.

Nothing in it is tuned. The unit noise variance fixes the latent unit that
`(mu, S, tau) -> (a mu, a^2 S, a tau)` would otherwise leave undetermined,
exactly as standard probit regression does. The branch offsets are
deterministic, `o_j = Phi^-1(pi_j)` for the left-subtree share `pi_j` of the
node's prior mass, which is zero at every node when K is a power of two. So the
head takes no `sigma_v`, no temperature, and no calibration step:

```python
head = TAGILastLayerClassifier(features.shape[1], 10, head="hrc_probit")
head.fit(train_features, train_labels, epochs=20)
prediction = head.predict(test_features)
```

`class_to_obs_full(K)` builds the tree: exactly K leaves and K - 1 decision
nodes, so CIFAR-10 needs nine outputs. The older `class_to_obs(K)` pads K
classes into 2^ceil(log2 K) leaves and discards the rest, so for K = 10 its ten
leaf products do not sum to one and a categorical normalizer is required before
any proper score is computed.

Evaluation is ordinary categorical cross-entropy, which makes the comparison
with a softmax classifier one-to-one in nats. `triton_tagi.hrc_probit` exposes
`hrc_log_probs`, `hrc_negative_log_likelihood`, and
`hrc_log_partition_deviation` for the `max_n |sum_c p_nc - 1|` invariant.

The padded tree (`hrc_tree="padded"`), the Gaussian +/-1 update (`head="hrc"`),
a fitted latent scale (`fit_hrc_log_tau`, `calibrate_hrc_log_tau`), and its
Laplace posterior (`fit_hrc_log_tau_laplace`) are research ablations against
that model, not part of it.
`experiments/last_layer/run_hrc_calibration.py` runs the main comparison and
those ablations on frozen CIFAR-10 features.

### ProbiTree direct branch updates

`head="probitree"` is the exact-K tree without deterministic gate offsets. It
uses an explicit fixed branch-noise variance `probitree_r` consistently in
training and prediction. A class label observes only the branch signs on its
path; the network receives the analytical probit messages `g` and `h`, not a
Gaussian regression update toward encoded +/-1 targets.

The first TAGI adapter deliberately processes one observation at a time:

```python
from triton_tagi import ProbiTree, TAGILastLayerClassifier

tree = ProbiTree(10)  # nine internal gates, preorder IDs, no dummy leaves
head = TAGILastLayerClassifier(
    features.shape[1],
    10,
    head="probitree",
    probitree_tree=tree,
    probitree_r=1.0,
)
history = head.fit(features, labels, epochs=20, batch_size=1)
prediction = head.predict(test_features)
```

For non-power-of-two class counts, the constructor initializes gate biases so
the zero-input prediction is uniform at recorded reference variances. Pass
`probitree_reference_variances=` to choose a representative variance vector,
or call `initialize_probitree_uniform(...)` later. Checkpoints store `r`, those
reference variances, and the complete class-to-path mapping. The lower-level
`probitree_label_update` API returns posterior moments, moment changes, stable
`g/h` messages, and pre-update log evidence for custom TAGI networks.

For end-to-end TAGI ResNet-18 training, use:

```bash
python examples/cifar10_resnet18_probitree.py --epochs 100 --probitree-r 1.0
```

The runner records `r`, the exact tree, reference variances, NLL/Brier/ECE,
and checkpoints. Its `--smoke` mode executes the same full network on four
training examples and eight test examples.


### CDF-TAGI-V / Remax

`head="cdf_remax"` is one interleaved `2K` layer followed by `EvenProbit`. The
even stream is the prediction head `Z_i`; the odd stream is a pre-activation
variance head `U_i` whose bounded activation `h(u) = eps + kappa*Phi(u)` is the
learned aleatoric logit noise, so that noise is confined to `(eps, eps+kappa)`
by construction. No `Remax` layer sits in the forward path: class probabilities
are not an activation for this head but the analytic Laplace-Remax average over
both heads and a shared positive deviation scale `s = e^L`.

```python
from triton_tagi import TAGILastLayerClassifier

head = TAGILastLayerClassifier(
    features.shape[1], 10, head="cdf_remax",
    cdf_epsilon=0.01, cdf_kappa=0.05, cdf_aleatoric_init=0.02,
)
head.fit(features, labels, epochs=10)
# The log-scale is a separate inference channel: fit it with the network
# frozen, on a split disjoint from training. It is worth ~0.17 nats.
head.calibrate_remax_log_scale(calibration_features, calibration_labels, method="grid")
prediction = head.predict(test_features)
```

`eps` and `kappa` must be sized to the logit scale the head actually sees.
The values above were selected against a converged 95%-accurate frozen
backbone; on an untrained one the variance head saturates against its own cap.
`examples/cifar10_resnet18_cdf_remax.py` trains this head end to end on a TAGI
ResNet-18 and prints a `band = (h - eps)/kappa` occupancy column for exactly
that reason. **That end-to-end run does not converge** — see
[experiments/last_layer/FINDINGS.md](experiments/last_layer/FINDINGS.md) §5.

### A hidden layer in front of the head

Every head takes `hidden_dims`, which puts `Linear + ReLU` blocks before the
output layer (default `()` is the single-layer head):

```python
head = TAGILastLayerClassifier(512, 100, head="hrc", hidden_dims=(512,))
```

This is worth +2.4 to +2.8 points and 0.21 nats to `hrc` on CIFAR-100 and +19
points on ImageNet, and nothing to the Remax family. It is not a capacity
argument: the deterministic MAP reference on the same frozen features is flat
in depth.

The `logit_tagiv` head is the only one that trains on continuous targets. It
regresses a teacher's logits over frozen features and learns the observation
variance of that regression, so it takes `targets` rather than `labels`:

```python
from triton_tagi import TAGILastLayerClassifier, prepare_logit_targets

targets, scale = prepare_logit_targets(teacher_logits)
head = TAGILastLayerClassifier(
    features.shape[1], 10, head="logit_tagiv", logit_scale=scale
)
head.fit(features, targets=targets, epochs=5)
head.calibrate(validation_features, validation_labels, mode="joint")
prediction = head.predict(test_features)
```

When repeated observations of the same input are available, prefer the
two-phase fit: `fit_mean` under a fixed observation variance, then
`reset_variance_head`, then `fit_variance` driven by the replicated sample
variance. A cold start learns its mean and its variance from the same first
step, so an early residual on a large-magnitude target is stored as noise and
then throttles that channel's own mean update through the `1 / (v_Z + mu_S)`
gain. The replicate-aware update observes the noise without passing through the
mean head at all, and `fit_variance(weight_shrinkage=...)` bounds how much
across-input spread the log-variance develops, which is the remaining error once
the contamination is gone.

Its output layer is the interleaved `2K` TAGI-V head followed by `EvenExp`, so
the observation variance is `s_min2 + exp(G)` with `G` Gaussian and
unconstrained. The exponential is what makes both the forward moments and the
cross-covariance `Cov(G, S) = v_G mu_X` analytic. Training conditions the
latent logit and the residual on the observed teacher logit, projects the
posterior residual square onto the AGVI moments of that variance, and maps the
result back through the exponential. `calibrate` then fits the post-hoc
temperature and the aleatoric multiplier `alpha` by validation NLL on shared
Sobol draws; `logit_tagiv_uncertainty` splits the predictive entropy into its
aleatoric and epistemic parts. The variance prior must not collapse:
`logit_variance_cv` sets the prior coefficient of variation of `exp(G)`, and a
zero would make the AGVI gain zero and freeze the head.

The `multinomial_probit` head instead uses `probit_tau2=1.0` for canonical
probit utility noise or `probit_tau2=0.0` for an epistemic-only model. Its
predictions and training update both use the TAGI output variances.

The `agci` head conditions jointly on the observed noisy-utility argmax event.
It uses fixed `agci_tau=1.0` by default and shifted one-dimensional Gaussian
quadrature to obtain the class-event moments for diagonal TAGI outputs before
the ordinary TAGI backward pass.
`agci_remax` uses the same AGCI training update but maps the resulting Gaussian
utilities to predictive probabilities and output variances with ReMax. It
centers each utility vector first so the map remains invariant to the shared
offset that an argmax likelihood cannot identify.

`gumbel_agci` swaps the Gaussian decision noise for Gumbel noise, which makes
the argmax event an exact multinomial logit and gives a class probability that
decays linearly rather than quadratically in the utility margin. `logit_site`
is its `S -> 0` limit, taking the probabilities from the prior means and
applying an ordinary Gaussian site.

`ct_agci` replaces softmax in that site with the Core-Tail link, the convex
choice model whose regularizer is Shannon negative entropy plus an interior
correction `-a* p^2 (1 - p)^2` that vanishes at `p = 0` and `p = 1`. It
therefore keeps the exact softmax tail and has the variance-matched probit
slope at a tie. The coefficient
`a* = (pi sqrt(2 pi) / sqrt(3) - 4) / 2 = 0.2732603854486113` is derived from
that slope match rather than fitted, and `core_tail_a_star=0.0` reduces the
head exactly to `logit_site`. Score and diagonal Fisher curvature are both
`O(C)` and exact.

Open-set detection requires a separate feature-support hypothesis; a
closed-set argmax likelihood cannot represent “none of the above.” The
`BayesianFeatureSupportGate` fits objective-Bayes Student-t posterior
predictives to labeled ID features and combines their domain evidence with any
conditional class probabilities:

```python
from triton_tagi import BayesianFeatureSupportGate

gate = BayesianFeatureSupportGate.fit(train_features, train_labels)
open_set = gate.predict(test_features, prediction.probabilities)
# open_set.probabilities has K known-class columns plus one OOD column.
```

The gate uses no OOD fitting data or statistical tuning parameters. Its
`log_bayes_factor` should be preferred as a ranking score in high dimensions,
where the corresponding posterior probability can saturate numerically and
statistically.


### Reproducible CIFAR-10/CIFAR-100 last-layer study

The staged study in [experiments/last_layer](experiments/last_layer/README.md)
trains one pinned deterministic ResNet-18 per dataset, caches frozen features for
clean CIFAR, all 15 canonical CIFAR-C corruptions at severities 1-5, and SVHN,
then screens and confirms Probit OVR, lognormal Remax, diagonal Laplace-Remax,
HRC, dense categorical TAGI-V, and hierarchical TAGI-V heads. It records
calibration, proper scoring, selective prediction, OOD detection, and
long-horizon epistemic convergence without post-hoc scaling.

The initialization study built on top of it — every head on CIFAR-10,
CIFAR-100 and ImageNet-1k across accuracy, calibration and OOD detection, as a
function of how the last layer is initialized — is written up in
[experiments/last_layer/FINDINGS.md](experiments/last_layer/FINDINGS.md), with
the deliverable tables in
[MEETING_RESULTS.md](experiments/last_layer/MEETING_RESULTS.md). Read FINDINGS
first: it records the negative results and the two known instrumentation
defects alongside what worked.
