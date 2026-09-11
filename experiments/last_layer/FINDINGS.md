# Last-layer TAGI — everything tried, and what came out

Branch: `feature/pytorch-last-layer-tagi`, 15 commits ahead of `main`.
**Not intended for merge.** This is a research record, published so the work and
its negative results are recoverable; `main` is unaffected.

Written 2026-09-11, closing the line of work. Start here, then:

| document | what it is |
|---|---|
| `MEETING_PLAN.md` | the working contract — scope, axes, execution plan, session handoffs |
| `MEETING_RESULTS.md` | the deliverable tables: every head × dataset × {accuracy, UQ, OOD} |
| `MEETING_TABLES.md` | the generated tables plus appendices A–C (every cell, not just selected) |
| `README.md` (§ last-layer) | library-level API for the heads |
| `AGCI_LAST_LAYER_THEORY.md`, `HSM_CALIBRATION_THEORY.md`, `CDF_TAGIV_Remax_joint_calibration.tex` | the derivations |

**All numbers below come from `runs/`, which is gitignored and not published.**
The JSON result dumps that *are* committed (`cdf_remax_*.json`,
`hidden_layer_cifar100.json`, `probitree_uq*.json`) carry the subset that fits
in the repo. Re-running from scratch needs the cached feature extraction; see
§Reproduction.

---

## 1. The question

Given a **frozen** deterministic backbone, how does each TAGI classification
head behave on accuracy, calibration and OOD detection — and how much of that
is decided by *how the last layer is initialized*?

Three datasets (CIFAR-10, CIFAR-100, ImageNet-1k), four heads, three
initialization arms, four gains, three `sigma_v` values, plus a post-hoc
calibration axis.

---

## 2. What was tried

### Heads carried through the study

| head | link | observation noise | output dim (10 / 100 / 1000) |
|---|---|---|---|
| `hrc` | hierarchical binary probit tree, padded | fixed `sigma_v` | 11 / 102 / 1001 |
| `hrc:full` | same, full K-leaf tree (calibratable) | fixed `sigma_v` | 9 / 99 / 999 |
| `remax_lognormal` | Remax + lognormal moment match | fixed `sigma_v` | K |
| `remax_laplace_diag` | Remax + diagonal Laplace Jacobian | fixed `sigma_v` | K |
| `logit_tagiv` | logit-target TAGI-V (distillation) | learned | 2K interleaved |

### Heads built and tested but outside the four-head study

- `probitree` — exact-K balanced tree, direct probit branch messages, fixed
  branch noise `r`. Used as the clean instrument for the epistemic question
  (§4.2). Confirmed cell: 0.9462 / 0.1969 / 0.0117 against `hrc`'s 0.9479 /
  0.1816 / 0.0049 — competitive on accuracy, behind on calibration.
- `cdf_remax` — CDF-TAGI-V / Remax with a bounded variance activation
  `h(u) = eps + kappa*Phi(u)` and a shared calibrated log-scale. Frozen-feature
  results in §3.6; the end-to-end attempt is §5, and it failed.
- Six heads deleted in `ae4787c` (agci, agci_remax, ct_agci, gumbel_agci,
  logit_site, multinomial_probit). Recoverable from `93aaa3f`.

### Axes

1. **`mean_init`** ∈ {`random` (He), `zero`, `backbone` (warm start from the
   trained `fc`)} — the primary axis, new in this branch.
2. **gain** ∈ {0.03, 0.1, 0.3, 1.0}, with `Sw = Sb = (gain·scale)²`.
3. **`sigma_v`** ∈ {0.05, 0.1, 0.3}, fixed-noise heads only.
4. **HSM gain calibration** — a Gaussian belief over each tree group's
   positive-branch log gain, at `global` / `level` / `node` sharing.
5. **Depth** — an extra `Linear + ReLU` block in front of the head.

Compute actually run: 152-cell screen × 2 CIFAR datasets, 200-epoch × 5-seed
confirms, a 60-cell ImageNet screen, 4-epoch × 3-seed ImageNet confirms, the
calibration axis at three sharing levels, a 192-cell hidden-layer grid on
CIFAR-100 and a 48-cell one on ImageNet.

---

## 3. What worked

### 3.1 `zero` means are the recommendation, and the reason is robustness

Not because they reach a better optimum — tuning gain per arm mostly closes
that — but because **`zero` is the only arm that is flat in gain**.
`remax_laplace_diag` with `zero` sits at 0.1973–0.1983 validation NLL across
the whole gain axis while `random` swings 1.4256 → 0.2002, a 7-fold range.
`zero` removes a tuning axis. At 1000 classes it is worth far more than that:
`random` means collapse the remax heads to 0.196 / 0.047 top-1 against 0.54 /
0.69 for `zero` / `backbone`.

### 3.2 The arms disagree about which prior they want

`random` improves as gain rises, `backbone` as gain **falls**, `zero` is flat.
A warm start does not want a prior wide enough to contain it; it wants one
tight enough to keep it. The two axes cannot be tuned independently — an
earlier draft of the plan predicted the opposite and is corrected in place.

### 3.3 Tree gain calibration buys likelihood, not bin-wise calibration

Per-node wins on NLL (CIFAR-10 0.1875 → 0.1783; CIFAR-100 1.2810 → 1.1576),
**a global gain does nothing** (±0.0003 — the α = 3 convention already sits at
the global optimum), and ECE is flat or slightly worse at every level. The
learned CIFAR-100 per-level gains rise monotonically with depth (0.188 at the
root to 0.629 at the leaves): the head is systematically over-sharp near the
root, which is structure a single global gain cannot express.

**Per-level is the recommendation at a fixed budget** — it beats uncalibrated
from n = 300 up and beats per-node everywhere below n ≈ 3000, where per-node is
*worse* than doing nothing (99 groups on 100 rows is one visit per node).

### 3.4 Depth fixes `hrc`, and only `hrc`

CIFAR-100: +2.4 to +2.8 points and −0.21 nats, halving ECE. ImageNet: **+19
points** (0.4260 → 0.6168 top-1 at width 2048, still improving). The Remax
family gains nothing or regresses.

**This is not a capacity argument** — the deterministic MAP reference on the
same frozen features is 0.7649 flat vs 0.7646 with a hidden layer, so depth
buys a deterministic model nothing. `hrc` asks a linear map to score tree nodes
over arbitrary *class subsets*, which the backbone never made linearly
separable; Remax's per-class logits are exactly what the softmax already
computes.

### 3.5 `logit_tagiv` is insensitive to initialization, as a distillation head should be

0.1770 / 0.1770 / 0.1771 across all three arms. Its `random` and `zero` arms
are bit-identical by construction. It is also the best head on CIFAR-100
(0.7673 / 0.9566) and matches the teacher exactly on ImageNet.

### 3.6 CDF-Remax works on frozen features, and the log-scale is load-bearing

CIFAR-10, 10 epochs, gain 0.1, eps 0.01, kappa 0.05, `zero`: test accuracy
0.9498, NLL **0.354 uncalibrated → 0.183 calibrated** at a fitted scale
s ≈ 2.68. The separate log-scale channel is worth 0.17 nats and is not
optional. Implementation verified against the design note: the predictive
vector, all-zero probability, the 15-point scale sweep, four auxiliary one-step
moments and the pre-fit population NLL all reproduce exactly
(`verification/`). Three published values do not reproduce and the reasons are
settled in the note's margin — do not re-litigate them.

---

## 4. What did not work, and why

### 4.1 The native-epistemic OOD column measures feature norm, not uncertainty

A frozen *deterministic* backbone hands the head zero input variance, so the
output variance collapses to `Sz = ma² @ Sw + Sb` — a monotone function of
`||f||²`. Measured spearman against feature energy: **+0.9988** (ProbiTree),
+0.7136 (`hrc`). SVHN's features are *smaller* than CIFAR-10's (24.04 vs
33.04), so a score rising with feature norm points the wrong way: AUROC 0.1191
where `||f||²` alone gives 0.1149 — the head contributes nothing.

Over 75 CIFAR-10-C shards the epistemic AUROC is below 0.5 in **75/75**, and
the more a corruption destroys accuracy the more confidently the score reports
*reduced* uncertainty (spearman +0.94). Entropy behaves correctly on the same
runs. Where the energy gap vanishes the AUROC is chance to three decimals.

Recovering a real epistemic column needs input variance to reach the head — an
unfrozen or stochastic backbone. That is a different study. **Do not present
0.2374 or 0.1191 as an uncertainty result.**

### 4.2 `hrc` degrades badly with class count

0.4331 top-1 on ImageNet against a 0.6976 softmax reference. This is a genuine
scaling result about the flat hierarchical tree at 1000 classes, and §3.4 shows
a hidden layer recovers most of it.

### 4.3 The remax heads collapse under `random` means at 1000 classes

0.196 (lognormal) and 0.047 (laplace_diag) top-1. They also rank well while
reporting near-uniform probability (mean confidence 0.017), so accuracy and NLL
rank the ImageNet cells almost independently — read both columns or neither.

### 4.4 Softmax temperature scaling still beats every calibration arm here

0.1722 / 0.9578 against HRC's best 0.1783 / 1.1576. Say this before a
supervisor finds it. On CIFAR-100 the fitted temperature is T ≈ 0.98, i.e. the
softmax baseline was already at its optimum.

### 4.5 Selection on validation NLL rewards hedging

On CIFAR-100 it prefers badly underconfident cells (`remax_lognormal`
backbone, gain 0.03: NLL 1.582, ECE 0.440, mean confidence 0.3235 at 76%
accuracy). For a calibration study, NLL with Brier and ECE only as tiebreakers
is the wrong objective. Unresolved.

### 4.6 The backbone moves OOD more than the head does

Swapping the study ResNet-18 (512-d) for a pretrained RepVGG-A2 (1408-d) over
identical data, transform and OOD source moves entropy AUROC 0.9118 → 0.9557
and FPR95 0.2401 → 0.1460 — larger than the entire spread between the best and
worst TAGI head that trains at all. The confound runs *backwards*: RepVGG is
worse in-distribution on accuracy, NLL and ECE, so "the better OOD model is
just the better model" is ruled out rather than merely acknowledged.

Every OOD number in this study is a joint statement about head **and**
backbone. Caveat that must travel with it: the pretrained RepVGG saw all 50k
CIFAR-10 train images, so no in-distribution column from the study's validation
split is comparable for it. SVHN is unseen by both.

---

## 5. End-to-end TAGI with the CDF-Remax head — attempted, failed

`examples/cifar10_resnet18_cdf_remax.py`, 2026-09-11. TAGI ResNet-18 (23.4M
params, the same topology the `hrc` and `probitree` examples train) terminated
by `Linear(512, 2K) → EvenProbit(K, eps, kappa)`, trained with
`Sequential.step_cdf_tagiv`, with the log-scale fitted on a held-out 5000-image
split carved out of train.

**Cost is not the problem.** Warm (post-JIT) timings on one RTX 4070 Ti SUPER:

| phase | cost |
|---|---|
| `step_cdf_tagiv`, batch 256 | 361 ms → 63.5 s/epoch, same as ProbiTree's 68 s |
| — at hermite order 64 / 32 / 16 / 8 | 361 ms throughout: the head's quadrature is free, the conv stack dominates |
| forward summaries, 10k rows | 4.0 s |
| `remax_scale_moments` L80 H16 S20, 10k rows | 8.3 s |
| `fit_remax_log_scale` grid=65, 5k rows | 40.5 s (grid=17 → 10.6 s; cost is linear in grid) |

**Three configurations, three failures, one mechanism.** Accuracy tracks the
variance head's operating point inside its own bounded range
(`band = (h − eps)/kappa`, 0 at the floor, 1 at the cap):

| arm | epochs | accuracy trajectory | band trajectory |
|---|---|---|---|
| eps 0.01, kappa 0.05 (the frozen-study cell) | 5 | 13.9 → 13.8 → 28.1 → 26.7 → 20.1 | 0.913 → 0.949 → 0.987 → 0.991 → **0.994** (pinned at the cap) |
| eps 0.02, kappa 1.5 (library default) | 4 | 18.2 → 11.6 → 10.0 → **9.5** | 0.307 → 0.014 → **0.000** → 0.938 (floor to ceiling) |
| kappa 1.5 + damped variance head (`v2bar_bias_var` 1e-6) | 10 | 10.1 → 26.5 → 26.6 → 26.6 → **14.3** → 26.5 → 23.5 → **15.5** → 27.3 → 20.7 | 0.011 → 0.19 → 0.17 → 0.15 → **0.310** → 0.14 → 0.21 → **0.005** → 0.14 → 0.10 |

Reference on the same backbone: ProbiTree reaches 36.2% at epoch 1 and 40.1%
at epoch 2 and keeps climbing.

**Diagnosis.** The variance head and the prediction head are coupled and
oscillate: the variance head moves → the observation variance rescales the
innovation for the mean head → the residuals change → the variance head
overshoots back. At kappa 0.05 the ceiling clamps the oscillation into a pin;
at kappa 1.5 nothing clamps it and it rings freely, taking the prediction head
into chance-level accuracy. Damping the head's *learned* component reduces the
amplitude but does not remove it — every epoch the band leaves its ~0.15
operating point, accuracy craters, and every epoch it returns, accuracy
recovers to ~26.5%. Ceiling across all arms: **~27%**, raw NLL 1.94–2.30
against a chance NLL of ln 10 = 2.303, mean confidence stuck near 0.17 where
uniform is 0.10. `epistemic_median` collapses to ~3e-4 in all three arms.

**The frozen-feature study never exercised this.** There the backbone is fixed,
so only the head moves and the feedback loop cannot close. End-to-end is the
first setting where both sides move at once, so this may be a real property of
the head at this scale rather than a driver bug.

**The next step, not taken:** put the same batch through `step_cdf_tagiv`,
`step_hrc` and `step_probitree` on the same initialized network and compare the
magnitudes of the parameter deltas. The other two heads train fine on this
exact backbone, so an order-of-magnitude discrepancy would localise the fault
to the channel rather than to any constant — one script, no training. Specific
hypothesis to test there: `logit_tagiv` routes through
`initialize_mean_from_teacher` with an explicit `logit_scale` division and
centering because its targets must live at a particular logit scale;
`step_cdf_tagiv` builds targets as a raw ±1 signed encoding with no such
scaling. On frozen 95%-accurate features the logits are already well scaled and
this never mattered.

Run directories (local, gitignored): `runs/cifar10_resnet18_cdf_remax_tagi_
20260911-{114559, 115724, 120718}`, each with `config.json`, `metrics.csv` and
`train.log`.

---

## 6. Defects this study found in its own instrumentation

Both are **documented and unfixed**. Numbers affected by them are reported with
the caveat rather than withdrawn.

1. **`hrc:full` trained with an uncompensated readout bias.**
   `class_to_obs_full` stores `offset[j] = Phi^-1(pi_j)` and `hrc_log_probs`
   adds `tau * offset` at inference, but `network.step_hrc` computes its
   innovation against the raw network output and never reads `hrc.offset`, so
   training fits the branch frequencies from data and the readout applies the
   prior a second time. Worth ~75% of the padded-vs-full NLL gap on CIFAR-100.
   `study.json` now pins `hrc_full_prior_offsets: false`; the cells already on
   disk report as `hrc:full+offsets`, a measured ablation. The deeper fix —
   training against the shifted latent, so the offset is a prior rather than a
   bias — is a model change and is **not** done.

2. **The native-epistemic OOD column is two different scores.**
   `predict_batches` reduces the network's raw output variance (node space for
   `hrc`); `calibrated_predict_batches` reduces `hsm_class_moments(...).variance`
   (class-probability space). Its docstring claims they compare like with like;
   the reduction matches, the quantity does not. So the 0.1052 → 0.8962 jump
   between uncalibrated and calibrated rows is **a change of score, not an
   effect of calibration**. The like-for-like comparison is table 6c. Fixing it
   is one line in `predict_batches` plus re-running `evaluate` over 155
   checkpoints, about 26 minutes.

Also worth carrying forward: `select_stage` could pick **epoch 0**, the
untrained prior. For `backbone` init that is the trained `fc` read through the
head's link, so it scores like the baseline it copies and *was winning* — best
validation NLL fell at epoch 0 in 7/12 `remax_lognormal` backbone cells and
11/12 `remax_laplace_diag` cells on CIFAR-10. Epoch 0 is now excluded from
selection and reported in its own warm-start table. One leak remains:
`evaluate`'s own best-checkpoint search does not exclude it, so a single row in
appendix A is the untrained prior, flagged in the caption.

---

## 7. What is on this branch

### Library (`triton_tagi/`)

| file | what |
|---|---|
| `classification.py` | `TAGILastLayerClassifier`: all heads, `mean_init`, `hidden_dims`, calibration entry points |
| `probitree.py` | exact-K balanced tree, direct probit moments, log-space prediction, TAGI messages |
| `cdf_remax.py` | Laplace-Remax mixture kernels, conditional and scale moments, uncertainty decomposition, forward diagnostics |
| `cdf_variance.py` | exact moments of `h(u) = eps + kappa*Phi(u)` |
| `remax_kernels.py` | the `phi(alpha) I_n(beta)` kernel forms |
| `remax_scale.py` | `LogScalePosterior`, ADF and grid fits for the shared log-scale |
| `layers/even_probit.py` | the interleaved 2K CDF variance-head activation |
| `network.py` | `step_cdf_tagiv`, `step_probitree` and the other per-head update entry points |
| `hsm_calibration.py`, `hierarchical_softmax_calibration.py` | the tree gain belief |

Two numerical facts in the Remax kernels that cost real time to find and are
load-bearing: the textbook forms of `D` and `F` die of cancellation as `t`
grows and get *worse* with more quadrature nodes, so everything must go through
`phi(alpha) I_n(beta)` with a four-term negative-tail series below
`beta = -30`; and that product must be carried *as a product*, because
`phi(alpha)` underflows while `erfcx(-beta/sqrt2)` overflows in the same regime.

### Drivers (`experiments/last_layer/`)

`run_study.py` is the main driver (`screen` / `select` / `confirm` / `evaluate`
/ `calibrate` stages, content-hashed run directories). Everything else is a
focused one-question script: `run_cdf_remax_cifar.py`,
`run_hidden_layer_cifar100.py`, `run_imagenet_hidden_layer.py`,
`run_probitree_uq.py`, `run_backbone_ood_comparison.py`, `run_hrc_longrun.py`,
`run_hrc_grid_extension.py`. `build_meeting_tables.py` regenerates
`MEETING_TABLES.md` from whatever is on disk and says so when inputs are
missing rather than dropping rows.

### Examples

`examples/cifar10_resnet18_probitree.py` and
`examples/cifar10_resnet18_cdf_remax.py` — end-to-end TAGI ResNet-18 with the
two new heads. The second one does not converge; see §5.

### Tests

`tests/unit/`, all passing at the time of writing. New in this branch:
`test_probitree.py`, `test_cdf_remax.py`, `test_cdf_variance.py`,
`test_cdf_tagiv_channel.py`, `test_remax_kernels.py`, `test_remax_scale.py`,
`test_last_layer_cdf_remax.py`, `test_last_layer_hidden.py`, plus additions to
`test_hrc_softmax.py` and `test_last_layer_runner.py`.

---

## 8. Reproduction

`runs/` is gitignored and ~37 GB, of which 14.9 GB is the frozen ImageNet
feature cache and 17 GB the `logit_tagiv` train shard. Nothing here is
reproducible without first re-extracting features.

```bash
# CIFAR: screen -> select -> confirm -> evaluate -> calibrate
python experiments/last_layer/run_study.py screen   --dataset cifar10 --stage init_screen
python experiments/last_layer/run_study.py select   --dataset cifar10 --stage init_screen
python experiments/last_layer/run_study.py confirm  --dataset cifar10 --stage init_confirm
python experiments/last_layer/run_study.py evaluate --dataset cifar10 --stage init_confirm
python experiments/last_layer/run_study.py calibrate --dataset cifar10 --stage init_confirm

# Regenerate the deliverable tables from whatever is on disk
python experiments/last_layer/build_meeting_tables.py

# End-to-end examples (the CDF one does not converge; see section 5)
python examples/cifar10_resnet18_probitree.py --epochs 100
python examples/cifar10_resnet18_cdf_remax.py --smoke --no-augment
```

Hardware everything ran on: 2 × RTX 4070 Ti SUPER, 16 GB each; GPU 0 only.
`remax_laplace_diag` trains at batch 134 on ImageNet against 256 for every
other head — its Laplace Jacobian is O(K²) and OOMs otherwise.

---

## 9. Open threads, in the order worth picking up

1. **The CDF-Remax innovation scale** (§5) — one script, no training, and it
   decides whether the head is salvageable end to end.
2. **The `hrc` offset fix** (§6.1) — train against the shifted latent so the
   prior offset is a prior, not a bias.
3. **The native-epistemic score unification** (§6.2) — one line plus 26 minutes
   of re-evaluation, and it makes that column readable across heads.
4. **Confirm the hidden-layer arm** (§3.4) — it is 20 epochs / 1 seed on
   CIFAR-100 and 1 epoch / 1 seed on ImageNet, with no OOD. The +19-point
   ImageNet result deserves a real confirm run.
5. **`hrc:full` init coverage** — no `zero` confirm on CIFAR-10, no `backbone`
   confirm anywhere, so "does calibration absorb the init delta" is answerable
   on CIFAR-100 only. (There, it does not: node calibration moves both init
   arms by the same 0.12 nats.)
6. **A selection objective that does not reward hedging** (§4.5).
