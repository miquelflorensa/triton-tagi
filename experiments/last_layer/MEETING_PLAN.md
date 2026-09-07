# Frozen last-layer TAGI — initialization study

Target: a supervisor-facing account of how every remaining head behaves on
CIFAR-10, CIFAR-100 and ImageNet-1k, across accuracy, calibration and OOD
detection, as a function of **how the last layer is initialized**.

Deadline: results in two days. Written 2026-09-07.

This file is the contract. A session that has lost its context should be able
to pick the work up from here alone.

---

## 1. Scope

### Heads (four, after the cull in `ae4787c`)

| head | link | observation noise | output dim |
|---|---|---|---|
| `hrc` | hierarchical binary probit tree | fixed `sigma_v` | tree nodes: 11 / 102 / 1001 |
| `remax_lognormal` | Remax + lognormal moment match | fixed `sigma_v` | K classes |
| `remax_laplace_diag` | Remax + diagonal Laplace Jacobian | fixed `sigma_v` | K classes |
| `logit_tagiv` | logit-target TAGI-V (distillation) | learned | 2K interleaved |

**Tree decision (settled 2026-09-07).** `hrc` runs on the *padded* tree,
`hrc_tree="auto"`, which is `run_study.py`'s default and what every existing
screen and refine run used, so the reference gain selections transfer. The
node counts above are the padded ones (K+1). An earlier draft of this table
said 9 / 99 / 999, which is the **full** tree (`hrc_tree="full"`, K-1 nodes);
that is the tree `hsm_calibration` used, so the finished calibration slide
still speaks of 99 groups on CIFAR-100 while this study's `hrc` head has 102
nodes. The two are not comparable node-for-node, and the deck must not imply
they are.

**Amended 2026-09-07 (session 2): `hrc` runs on BOTH trees.** The decision
above stands for the headline row, but it turned out to exclude the whole
calibration question: `gain_groups` — and therefore
`calibrate_hsm_log_gain` at every sharing level — **rejects the padded tree
outright**, with

    ValueError: gain_groups requires a proper K-leaf tree from
    class_to_obs_full; the padded tree of class_to_obs discards leaves that
    hold probability mass

Verified on GPU at K = 10: padded is refused at `global`, `level` and `node`;
full fits all three and the calibrated rows sum to one. Only
`calibrate_hrc_log_tau`, the *point-value* ablation, accepts the padded tree,
and that is explicitly not the gain-belief method.

So the grid now crosses `hrc` with `hrc_trees = ["padded", "full"]`:

| arm | selection key | nodes (10 / 100 / 1000) | role |
|---|---|---|---|
| padded | `hrc` | 11 / 102 / 1001 | headline row; the v2 gain selections transfer |
| full | `hrc:full` | 9 / 99 / 999 | the calibratable arm, and node-for-node comparable with the finished calibration slide |

This retires the non-comparability caveat above **for the `hrc:full` arm
only**: at 99 nodes on CIFAR-100 it is the same tree the calibration slide
used. The padded row still is not comparable node-for-node, and the deck must
keep saying so for that row.

The padded arm deliberately emits **no** `hrc_tree` key, so its run hash still
matches the 116 cells per dataset already on disk; verified that the new
152-cell grid matches all 116 existing CIFAR-10 runs, will run exactly the 36
new full-tree cells, and orphans nothing. A blank `hrc_tree` in the report
means "the head's default tree", which for `hrc` is padded.

`hrc_probit`, `hrc_tagiv`, `categorical_tagiv`, `probit_ovr` and
`remax_laplace` are still importable but are **not** part of this study.
`hrc_probit.py` must stay: `hsm_calibration` imports `hrc_log_probs` from it.

### Datasets

| dataset | features | classes | train / val | OOD source |
|---|---|---|---|---|
| CIFAR-10 | `runs/last_layer/cifar_frozen_last_layer_v2_40k10k/features/cifar10` | 10 | 40 000 / 10 000 | SVHN + CIFAR-10-C (15 × 5) |
| CIFAR-100 | same root, `/cifar100` | 100 | 40 000 / 10 000 | SVHN + CIFAR-100-C (15 × 5) |
| ImageNet-1k | `runs/imagenet/resnet18_agci/features_shuffled` | 1000 | 1 281 167 / 25 000 | **none cached — see §6** |

### Explicitly out of scope

- **Full-covariance last layer.** Discussed and dropped: only tractable on
  CIFAR-10, so it cannot appear in a three-dataset table. Do not add it.
- The six heads deleted in `ae4787c` (agci, agci_remax, ct_agci, gumbel_agci,
  logit_site, multinomial_probit). Recoverable from `93aaa3f` if ever needed.

---

## 2. The axes

### Axis 1 — mean initialization (PRIMARY, and entirely new)

**Implemented 2026-09-07.** `TAGILastLayerClassifier` now takes
`mean_init` and `backbone_fc`; see `_apply_mean_init`. 30 tests in
`tests/unit/test_last_layer_mean_init.py` cover it, and the 455 pre-existing
tests are unchanged, because `random` is a deliberate no-op.

| arm | weight means | rationale |
|---|---|---|
| `random` | `mu_w ~ N(0, 1/fan_in)` (current default) | the status quo baseline |
| `zero` | `mu_w = 0`, `mu_b = 0` | class-symmetric prior; no prior committed to one random classifier |
| `backbone` | copied from the trained backbone's `fc` layer | warm start from the network the features came from |

Prior variances are untouched by this axis: `Sw = Sb = (gain · scale)²` in all
three arms. Only the means move.

**Backbone weight sources (verified present):**

| dataset | file | tensor |
|---|---|---|
| CIFAR-10 | `.../backbones/cifar10_resnet18.pt` | `model_state_dict["network.fc.weight"]` `[10, 512]`, `.bias` `[10]` |
| CIFAR-100 | `.../backbones/cifar100_resnet18.pt` | `[100, 512]` + bias |
| ImageNet | `runs/imagenet/resnet18_agci/features/pretrained_fc.pt` | `weight [1000, 512]`, `bias [1000]` |

**The HRC projection — RESOLVED, and it works.** The proposed branch-contrast
rule was implemented as
`triton_tagi.hrc_softmax.project_classes_to_nodes` and verified against an
explicit per-node reference loop at K = 10, 100 and 1000 on both trees:

    w_node = mean(w_class for classes on the node's +1 branch)
           - mean(w_class for classes on the node's -1 branch)

One amendment was needed. The contrast is invariant to adding a constant
vector to every class — the right gauge, since the tree's latents are only
defined up to that shift — but **only on the full tree**. The padded tree we
are running has single-branch nodes, where one of the two means is missing and
the contrast degenerates into a raw branch mean whose scale depends on the
class gauge. The projection therefore takes the *class-centered* weights,
`w - mean_k w`, which fixes the gauge on those nodes and is an exact no-op on
the full tree. Verified: gauge invariance fails on the padded tree without
centering and holds with it.

So `backbone` is reported for `hrc`, not marked N/A. `remax_*` is a direct
copy (transposed to `[in, out]`). `logit_tagiv` routes through the head's own
`initialize_mean_from_teacher`, which writes the even (mean) channel with the
centering and `logit_scale` division its distillation targets were built with,
and leaves the odd (variance) channel at the prior `_initialize_logit_tagiv_prior`
solved for.

**`logit_tagiv` has only two distinct init arms, not three.**
`_initialize_logit_tagiv_prior` zeroes its own latent means by design — its
docstring argues a class-symmetric prior is the correct one for a head
regressing centered logits — so `random` and `zero` are bit-identical for it.
Confirmed numerically: val NLL 0.1770 vs 0.1770 on CIFAR-10, 0.9620 vs 0.9620
on CIFAR-100. Its screen is therefore 8 cells per dataset, not 12, and the
deck should report one row for it with the arms merged rather than two
identical rows.

### Axis 2 — gain (prior parameter variance)

`Sw = Sb = (gain · scale)²`, `scale = sqrt(1/512) = 0.04419`.

Sweep `gain_w = gain_b ∈ {0.03, 0.1, 0.3, 1.0}`.

Reference: the v2 study selected 0.3 for CIFAR-10 (`hrc`) and gain_w 0.3 /
gain_b 0.1 for CIFAR-100. Prior sweeps at 20 epochs live in
`heads/screen/*/selection.json` — checkpoints are gone, histories are not.

### Axis 3 — sigma_v (SECONDARY; watch it, don't chase it)

Fixed-noise heads only (`hrc`, `remax_lognormal`, `remax_laplace_diag`);
`logit_tagiv` learns its variance and rejects the argument.

Sweep `{0.05, 0.1, 0.3}`. Established values: **0.3 on CIFAR-10, 0.1 on
CIFAR-100** — selected per dataset, not transferable. On the frozen CIFAR-10
backbone the whole grid is flat (val NLL 0.1665–0.1730); on CIFAR-100 it
matters (1.330 → 1.900). Expect ImageNet to behave like CIFAR-100 or worse.

---

### Axis 4 — HSM gain calibration (added 2026-09-07, session 2)

Requested during the meeting prep, and it is the axis that asks whether the
initialization effect survives calibration — a fair question, since `zero` and
`backbone` differ mainly in **calibration** (NLL / ECE) rather than accuracy.

This is the hierarchical probit calibration of Goulet, Nguyen and
Florensa-Montilla: a Gaussian belief over each group's positive-branch log
gain, fitted on a split disjoint from training with the network frozen, and
*integrated over* at prediction rather than plugged in. `prior_mean = 0` is
the uncalibrated head, so the `evaluate` rows are the baseline these read
against.

| setting | value | why |
|---|---|---|
| sharing | `global`, `level`, `node` — all three | mirrors the finished calibration study, so the two are directly comparable on the `hrc:full` arm |
| fit split | the full 10 000-row validation split | decided; maximum data for the fit. **Caveat for the caption:** it is the same split that selected the cell |
| eligible runs | `hrc:full` only | the padded tree is refused; the remax and logit heads have no tree gain to fit |
| groups | 1 / 4 / 9 at K=10, 1 / 7 / 99 at K=100, 1 / 10 / 999 at K=1000 | per-node is data-starved on ImageNet: 999 groups against 25 000 validation rows |

Driver: `run_study.py calibrate --dataset <ds> --stage init_confirm`, which
fits the gain and re-evaluates clean + SVHN + corruptions through
`hsm_class_moments`, so the calibrated rows carry the **native epistemic** OOD
column too, reduced exactly as `predict_batches` reduces the uncalibrated
head's. The sharing level rides in the run config as `calibration`, so it
reaches `report.csv` as its own column and never averages into the
uncalibrated row.

## 3. Metrics — every cell reports all three families

Already implemented in `triton_tagi/metrics.py` and emitted by
`run_study.py evaluate`; no new metric code needed.

| family | metrics |
|---|---|
| accuracy | top-1, top-5 |
| calibration / UQ | NLL, ECE, adaptive ECE, classwise ECE, Brier, mean confidence, epistemic share, AURC, risk@{80,90,95} coverage |
| OOD | AUROC / AUPR-in / AUPR-out / FPR95 under three scores: predictive entropy, negative max probability, **native epistemic variance** |

The native-epistemic column is the one that justifies TAGI over softmax, so it
carries the story. Note the known result to check against: on CIFAR-10 `hrc`
native epistemic AUROC was **0.2374** — worse than chance, while entropy gave
0.9244. If that reproduces across heads it is a finding, not a bug, and the
supervisors should see it.

---

## 4. Execution plan

### Hour 0–1 — unblock — **DONE 2026-09-07**

1. ~~**Install torchvision.**~~ Installed `torchvision 0.29.0+cu130` from the
   PyTorch cu130 index, which left `torch 2.14.0+cu130` untouched (checked
   with `--dry-run` first). `python -m pytest tests/unit -q` → **485 passed**
   (455 pre-existing + 30 new `mean_init` tests). Hardware: 2 × RTX 4070 Ti
   SUPER, 16 GB each.
2. ~~**Implement `mean_init`.**~~ Done; see §2 above for the design and the
   one amendment the HRC projection needed. `mean_init` is recorded in
   `config()`, so it lands in every run's `config.json` and in the run hash,
   which means new cells cannot collide with the v2 runs already on disk.
   `load()` rebuilds on the no-op arm and restores the recorded value, since
   the backbone tensors are not in the checkpoint and the saved state
   supersedes the prior means anyway.
3. ~~**Measure throughput.**~~ Measured, not extrapolated. All numbers below
   are wall clock on one GPU, batch 256.

**CIFAR, 20 epochs, one cell (seconds).** The plan's 20 s/cell estimate holds.

| head | out dim | CIFAR-10 | CIFAR-100 |
|---|---|---|---|
| `hrc` (padded) | 11 / 102 | 10–16 | 17–22 |
| `remax_lognormal` | 10 / 100 | 6–10 | 13–15 |
| `remax_laplace_diag` | 10 / 100 | 14–16 | 27–29 |
| `logit_tagiv` | 20 / 200 | 10–14 | 19–23 |

Screen budget: 116 cells/dataset × 2 datasets ≈ **70 min**, one GPU.

**ImageNet-1k, 1 epoch over 1.29 M samples** (129 shards × 10 000, streamed
from `features_shuffled`; measured on 3 shards and scaled, load time is
~0.04 s/shard warm and negligible against compute).

| head | out dim | 1 epoch |
|---|---|---|
| `remax_lognormal` | 1000 | **2.85 min** |
| `logit_tagiv` | 2000 | **2.71 min** |
| `hrc` (padded) | 1001 | **4.25 min** |
| `remax_laplace_diag` | 1000 | **14.14 min** |

This changes the overnight plan: see the revised §4 ImageNet budget below.

### Day 1 — CIFAR screen, then CIFAR confirm

**Screen** (20 epochs, seed 0, val split):

| heads | init | gain | sigma_v | cells |
|---|---|---|---|---|
| hrc, remax_lognormal, remax_laplace_diag | 3 | 4 | 3 | 108 |
| logit_tagiv | 3 | 4 | — | 12 |
| **per dataset** | | | | **120** |
| **CIFAR-10 + CIFAR-100** | | | | **240** |

Estimate ≈ 20 s/cell → **~80 min**. Then `select` on validation NLL.

**Confirm** (200 epochs, seeds 0–4) at the selected `(init, gain, sigma_v)`
per head per dataset: 4 heads × 2 datasets × 5 seeds = 40 runs ≈ **2.2 h**.

Keep one extra confirm arm per head: the *best* init and the *`random`*
init, so the deck can show the initialization delta at full protocol rather
than only at 20 epochs. That doubles confirm to ~4.5 h — start it before
leaving for the ImageNet queue.

**Then `evaluate` + `report`** for both datasets: clean, SVHN, 15 corruptions
× 5 severities.

### Day 1 night — ImageNet screen (overnight, unattended)

Reduced grid, 1 epoch, seed 0, on `features_shuffled` (129 train shards ×
10 000 × 512, 5 val shards, teacher `logits` present for `logit_tagiv`):

| heads | init | gain | sigma_v | cells | measured cost |
|---|---|---|---|---|---|
| hrc | 3 | 3 (0.1, 0.3, 1.0) | 2 (0.1, 0.3) | 18 | 77 min |
| remax_lognormal | 3 | 3 | 2 | 18 | 51 min |
| remax_laplace_diag | 3 | 3 | 2 | 18 | **255 min** |
| logit_tagiv | 2 (see §2) | 3 | — | 6 | 16 min |
| | | | **60** | **6.6 h, one GPU** |

**Budget decision (measured, 2026-09-07).** The plan's cut rule — "if a cell
exceeds ~4 min, cut the gain axis to {0.1, 0.3}" — is triggered, by
`remax_laplace_diag` at 14.1 min/epoch and marginally by `hrc` at 4.25. Do
**not** cut it: there are two RTX 4070 Ti SUPERs in this box, and splitting
the queue across them brings 6.6 h down to **≈3.4 h**, which fits the
overnight window with the gain axis intact. The gain axis is deliverable table
3, so parallelism is the cheaper thing to spend. Cut gains only if one GPU is
lost.

`logit_tagiv` contributes 6 cells rather than 9 because `random` and `zero`
are bit-identical for it (§2).

### Day 2 — ImageNet confirm, tables, deck

1. ImageNet confirm: 4 heads × best init × 3 seeds × 4 epochs, plus the
   `random`-init arm for the delta.
2. Evaluate everything; regenerate `report.csv` / `report_summary.csv`.
3. Build the four tables in §5.
4. Reserve the last 3 hours for writing, not computing.

---

## 5. Deliverable tables

1. **Head × dataset headline.** 4 heads × 3 datasets × {top-1, NLL, ECE,
   OOD AUROC}, each against the softmax reference for that dataset
   (CIFAR-10 / CIFAR-100 `pytorch_softmax`; ImageNet pretrained fc:
   acc 0.6976, NLL 1.2469, ECE 0.0263).
2. **Initialization delta.** For each head × dataset, the three init arms at
   fixed best gain — the centrepiece, since it is the question asked.
3. **Gain sensitivity.** Metric vs gain curve per head, best init fixed.
   One figure, three panels (one per dataset).
4. **sigma_v sensitivity.** Small table, fixed-noise heads only, with the
   flatness on CIFAR-10 vs the real sensitivity on CIFAR-100 called out.

5. **Calibration × initialization** (new, axis 4). For the `hrc:full` arm:
   uncalibrated vs `global` vs `level` vs `node`, each at all three init arms,
   on CIFAR-10 and CIFAR-100. The question it answers: does the
   initialization delta survive calibration, or does a fitted gain absorb it?
   If a global gain absorbs the whole init effect, the recommendation becomes
   "initialize however you like and calibrate", which is a *different* slide
   from "never use He means on a many-class last layer".

6. **The validation-sweep calibration for HRC** — already computed, from
   `runs/last_layer/hsm_calibration/{cifar10,cifar100}_base_hrc/report.csv`.
   Test NLL against the size of the split the gain is fitted on:

   | dataset | arm (groups) | n=100 | n=300 | n=1000 | n=3000 | n=10000 |
   |---|---|---|---|---|---|---|
   | CIFAR-10 | uncalibrated | 0.1893 | 0.1893 | 0.1893 | 0.1893 | 0.1893 |
   | | `hsm_global` (1) | 0.1956 | 0.1899 | 0.1901 | 0.1896 | 0.1891 |
   | | `hsm_level` (4) | 0.2043 | 0.1869 | 0.1845 | 0.1825 | **0.1818** |
   | | `hsm_node` (9) | 0.2123 | 0.1910 | 0.1852 | 0.1814 | **0.1802** |
   | CIFAR-100 | uncalibrated | 1.3150 | 1.3150 | 1.3150 | 1.3150 | 1.3150 |
   | | `hsm_global` (1) | 1.3170 | 1.3152 | 1.3149 | 1.3153 | 1.3149 |
   | | `hsm_level` (7) | 1.2301 | 1.2092 | 1.2035 | 1.2012 | **1.2003** |
   | | `hsm_node` (99) | 1.4105 | 1.2882 | 1.2268 | 1.2044 | **1.1943** |

   Four things to say with it:

   - **The crossover is real.** Per-node is *worse* than uncalibrated below
     n≈1000 (CIFAR-100 1.4105 vs 1.3150 at n=100 — one visit per node at 99
     groups) and best above n≈3000.
   - **`hsm_level` is the recommendation at a fixed budget.** It beats
     uncalibrated from n=300 up on both datasets and beats per-node
     everywhere below n≈3000, with 4 and 7 groups.
   - **A global gain does nothing on CIFAR-100** (1.3149 vs 1.3150), i.e. the
     α = 3 convention already sits at the global optimum. The gain belief only
     pays once it can vary *across* the tree.
   - **It buys likelihood, not bin-wise calibration.** Test ECE: CIFAR-10
     uncalibrated 0.0049 and every calibrated arm is worse at every n;
     CIFAR-100 flat at 0.087-0.098 except per-node at n=100, which is 0.1853.
     And softmax temperature scaling still beats all of it on NLL (0.1734 /
     0.9541 vs HRC's best 0.1802 / 1.1943) — say so rather than letting a
     supervisor find it.

   This is the sweep that tells axis 4 whether its single n = 10 000 fit is in
   the data-rich regime. It is, for `global` and `level`; per-node at 99
   groups sits right at the edge.

Plus the calibration slide that is **already finished** and needs no compute —
base HRC uncalibrated vs global / per-level / per-node gain, from
`runs/last_layer/hsm_calibration/{cifar10,cifar100}_base_hrc/report.csv`.
Headline: read at its trained `sigma_v = 0.1` the CIFAR-100 head sits at
3.2584 NLL / 0.1766 ECE; through the α = 3 convention channel, 1.3150 /
0.0871; a single global gain fitted on 1000 points collapses the difference to
1.3149. Per-node needs data — worse than uncalibrated below n≈1000 at 99
groups, best above n≈3000.

---

## 6. Known gaps — surface these, do not paper over them

1. ~~**No ImageNet OOD set is cached.**~~ **DECIDED 2026-09-07: report
   ImageNet with accuracy + calibration only**, and say so in the table
   caption. No OOD source will be cached, so the Day-2 budget stands as
   written. Consequence to state plainly in the deck: the native-epistemic OOD
   column — the one that carries the TAGI-over-softmax story — is a
   **CIFAR-10 / CIFAR-100 result only**, and nothing here shows whether it
   holds at 1000 classes. The ImageNet row's OOD cells are `n/a`, not blank.
2. ~~**`hrc` backbone init needs the class→node projection.**~~ Resolved and
   tested; see §2. It needed one amendment (class-centering, to fix the gauge
   on the padded tree's single-branch nodes).
3. ~~**ImageNet timing is extrapolated**, never measured.~~ Measured; see the
   Hour 0–1 tables. The extrapolation was badly wrong in one place:
   `remax_laplace_diag` costs 14.1 min/epoch at 1000 classes, roughly 5× the
   other heads, because its diagonal Laplace Jacobian scales with K.
4. **`hrc` on ImageNet was catastrophic** in the earlier 1-epoch screens:
   acc 0.4232, NLL 2.7443 against a 0.6976 softmax reference. If that
   reproduces, it is a genuine scaling result about the hierarchical tree at
   1000 classes and belongs in the deck.

---

## 7. Repository state this plan assumes

- Branch `feature/pytorch-last-layer-tagi`, at `ae4787c`.
  `93aaa3f` is the full pre-cull snapshot.
- `runs/` is 37 GB: `last_layer` 4.3 GB, `imagenet` 33 GB (of which 14.9 GB is
  the frozen ImageNet feature cache and 17 GB the logit_tagiv train shard —
  **all of it must be preserved**, it is the input to every ImageNet run here).
- CIFAR screen/refine checkpoints were deleted; their `history.json`,
  `config.json` and `selection.json` were kept, so old sweeps still read back.
- `experiments/scaling_theory_stage2` (3.9 GB) is untracked, unbacked and
  awaiting a delete/keep decision. Unrelated to this plan.

---

## 8. Preliminary signal from the Hour 0–1 timing probes

These are **single cells, not the screen**: seed 0, 20 epochs, gain 0.3,
`sigma_v` 0.1, one cell per (head, init). They are recorded because they
already answer the question the study asks, and because they tell the screen
what to look for. Validation top-1 / NLL:

| head | dataset | `random` | `zero` | `backbone` |
|---|---|---|---|---|
| `remax_lognormal` | CIFAR-10 | 0.9520 / 0.4029 | 0.9524 / 0.3439 | 0.9528 / 0.3571 |
| `remax_laplace_diag` | CIFAR-10 | 0.9525 / 0.2912 | 0.9526 / 0.2264 | 0.9529 / 0.6233 |
| `hrc` | CIFAR-10 | 0.9520 / 0.1668 | 0.9525 / 0.1665 | 0.9499 / 0.1757 |
| `logit_tagiv` | CIFAR-10 | 0.9539 / 0.1770 | 0.9539 / 0.1770 | 0.9539 / 0.1771 |
| `remax_lognormal` | CIFAR-100 | **0.5271** / 2.4907 | **0.7471** / 2.0277 | 0.7473 / 2.1615 |
| `remax_laplace_diag` | CIFAR-100 | **0.5220** / 2.7391 | **0.7533** / 2.2433 | 0.7501 / 3.2509 |
| `hrc` | CIFAR-100 | 0.7248 / 1.3296 | 0.7284 / 1.3048 | 0.7213 / 1.4180 |
| `logit_tagiv` | CIFAR-100 | 0.7628 / 0.9620 | 0.7628 / 0.9620 | 0.7636 / 0.9640 |

Three things to carry into the screen:

1. **The headline is the remax heads at 100 classes.** Going from `random` to
   `zero` means is worth **+22 accuracy points** on CIFAR-100
   (`remax_lognormal` 0.527 → 0.747, `remax_laplace_diag` 0.522 → 0.753) at
   20 epochs, and it costs nothing to do. On CIFAR-10 the same change is worth
   ~0.0005. The effect is a function of class count, which is exactly why the
   study needed three datasets, and it makes the ImageNet arm the interesting
   one rather than a formality. Check whether it is a *rate* effect that 200
   epochs closes, or a floor the head never leaves — the confirm stage at 200
   epochs answers this, and the answer changes the recommendation.

2. **`backbone` is not the best arm, and it hurts calibration.** Accuracy
   ties `zero` or beats it slightly, but NLL is consistently worse, badly so
   for `remax_laplace_diag` (0.2264 → 0.6233 on CIFAR-10; 2.2433 → 3.2509 on
   CIFAR-100). Mechanism, worth a slide: the trained `fc` weights have RMS
   0.082 (CIFAR-10), 0.059 (CIFAR-100), 0.069 (ImageNet), i.e. 1.3–1.9× the He
   *mean* scale but **4–6× the prior standard deviation** `gain · scale` at
   gain 0.3 = 0.0133. The warm start therefore places the mean 4–6 prior
   sigmas out, and the posterior is over-committed before it sees data. This
   predicts `backbone` should look much better at gain 1.0 — the gain axis and
   the init axis interact, so read table 2 at more than one gain.

   **MEASURED, AND THE PREDICTION WAS BACKWARDS (session 2).** The full
   CIFAR-10 grid is in, and `backbone` gets monotonically *better as gain
   falls*, not as it rises. Val NLL at 20 epochs, each arm at its best
   `sigma_v`:

   | head | arm | gain 0.03 | gain 0.1 | gain 0.3 | gain 1.0 |
   |---|---|---|---|---|---|
   | `remax_lognormal` | random | 0.5008 | 0.4730 | 0.3687 | **0.2932** |
   | | zero | 0.3174 | 0.3153 | 0.3034 | **0.2907** |
   | | backbone | **0.2027** | 0.2098 | 0.2645 | 0.3276 |
   | `remax_laplace_diag` | random | 1.4256 | 1.1686 | 0.2912 | **0.2002** |
   | | zero | **0.1973** | 0.1983 | 0.1983 | 0.1983 |
   | | backbone | **0.2607** | 0.2954 | 0.7189 | — |
   | `hrc` (padded) | random | 0.1673 | 0.1666 | **0.1665** | 0.1666 |
   | | zero | 0.1668 | **0.1662** | **0.1662** | 0.1666 |
   | | backbone | 0.1805 | 0.1763 | **0.1733** | 0.1746 |

   The mechanism above is right about the magnitudes and wrong about what
   follows from them. A warm start does not want a prior wide enough to
   contain it; it wants one **tight enough to keep it**, so the data cannot
   pull the mean off a solution that is already good. Read the other way, the
   arms disagree about which prior they want: `random` needs a loose one,
   `backbone` a tight one, and that is why the two axes cannot be tuned
   independently.

   **The real headline is robustness, not the best cell.** `zero` is the only
   arm that is flat in gain — `remax_laplace_diag` sits at 0.1973–0.1983
   across the whole axis, while `random` swings from 1.4256 to 0.2002, a
   7-fold NLL range. So `zero` does not merely win by a little; it removes a
   tuning axis. That is a stronger recommendation than the +22 points, and it
   is the one to lead with.

   **And CIFAR-10's flatness in §8 was gain-confounded.** Tuning gain per arm
   collapses the init effect to nothing there (`hrc` 0.1665 vs 0.1662). The
   single-gain probes in the table above were reading an interaction, not a
   main effect. Whether the CIFAR-100 effect survives per-arm gain tuning is
   the thing the finished screen answers.

3. **`logit_tagiv` is insensitive to all of it** (0.1770 / 0.1770 / 0.1771).
   Expected for a distillation head: it regresses the teacher's logits, so the
   teacher determines the fixed point regardless of where the mean starts.
   Say so rather than presenting three near-identical numbers as a null result.

---

## 9. Handoff — state as of 2026-09-07, end of session 2

Session 1's handoff is preserved in git (`6ef20ff`); this replaces it.

### Where the compute is

| stage | state |
|---|---|
| CIFAR-10 padded screen | **done**, 116/116 |
| CIFAR-100 padded screen | running, ~103/116 |
| CIFAR-10 + CIFAR-100 full-tree `hrc` arm | queued, 36 cells each, waits on the padded screen |
| `select` both datasets | queued behind that |
| `init_confirm` | **not started** — inspect the selections first, it is ~4.5 h |
| `evaluate --stage init_confirm` | not started |
| `calibrate --stage init_confirm` | not started |
| ImageNet screen | **never run**, not one cell |

Measured cell cost, all-in through the driver (CIFAR-10, 20 epochs, ~29 s/cell
average): `remax_lognormal` 4.8 s training, `hrc` 8.7 s, `logit_tagiv` 8.1 s,
`remax_laplace_diag` 12.9 s; CIFAR-100 roughly 1.3-1.8x that. The full CIFAR-10
screen took ~52 min, so §4's 70-min two-dataset estimate was optimistic mainly
on CIFAR-100.

### What session 2 changed

1. **Report carries the cell identity** (`d0d8e55`). `report.csv` named rows by
   `(dataset, head, seed, epoch, evaluation_kind)` and the summary grouped by
   `(dataset, head, evaluation_kind)`. `init_confirm` evaluates **two arms per
   head**, so both would have collapsed into one summary row reporting their
   mean — the delta this study exists to measure, averaged away — and tables
   2-4 had no gain or `sigma_v` column at all. Rows now carry
   `stage/mean_init/hrc_tree/gain_w/gain_b/sigma_v/calibration`. Regenerating
   the v2 report reproduces all 132 rows and 1664 summary rows unchanged.
2. **`evaluate` can read the init tree** (`3bb25da`). It hardcoded
   `stage_root(..., "confirm", ...)`, so the plan's own Day-2 command would
   have re-evaluated the v2 runs and produced **no init rows at all**, leaving
   every table empty with nothing visibly wrong. Now `--stage`.
3. **ImageNet arm covered, and one duplicate cell dropped** (`1129a9b`). It
   had no tests despite running unattended. Also `confirm_configs` paired
   `logit_tagiv` with a `random` counterpart that is bit-identical to `zero`:
   half an hour of the overnight window, and a delta of exactly zero reported
   as though measured.
4. **Axis 4 exists** — see §2. `hrc` now runs both trees (§1 amendment),
   because the padded tree cannot be gain-calibrated at all.
5. **`select` keeps the trees apart and carries `hrc_tree` forward.** Grouping
   by head alone made padded and full compete for one slot, and the config
   projection dropped `hrc_tree` — so a winning full-tree cell would have been
   confirmed on the **padded** tree, silently, since padded is what `auto`
   resolves to. This is the same failure mode as the `mean_init` fallback
   session 1 fixed; the whitelist is the thing to check whenever an axis is
   added.
6. **§8 note 2 was wrong and is corrected in place.** See §8.

### The epoch-0 trap, and what it says about three of the four heads

Found while previewing what `select` would choose, and it is the most
consequential thing session 2 turned up.

`select_stage` could pick **epoch 0**, the prior before any data. For the
`random` and `zero` arms that is a chance-level model that never wins, which
is why the hole survived. For `backbone` epoch 0 is not untrained at all — it
is the backbone's own trained `fc` read through the head's link — so it scores
like the baseline it copies: CIFAR-10 `remax_lognormal` 0.9539 / 0.1913 at
epoch 0 against `pytorch_softmax` 0.9500 / 0.1941.

**And it was winning.** Best validation NLL fell at epoch 0 in 7/12
`remax_lognormal` backbone cells, 11/12 `remax_laplace_diag`, 4/4
`logit_tagiv` on CIFAR-10. The study was on course to select untrained cells,
confirm them at 200 epochs, and report its own baseline as a TAGI result.

**Decided (2026-09-07):** epoch 0 is excluded from selection, matching
`run_imagenet_init_study.py`, which already skipped it. The warm start is
reported in its own table instead — it needs no seeds, since nothing has
trained and the prior variances are a deterministic function of gain.

The wider finding, which is *not* an initialization result and must not be
presented as one:

| dataset | head | epoch 0 NLL | best trained NLL | at epoch |
|---|---|---|---|---|
| CIFAR-10 | `hrc` | 0.6846 | **0.1733** | 20 |
| CIFAR-10 | `hrc:full` | 0.4559 | **0.1878** | 20 |
| CIFAR-10 | `remax_lognormal` | **0.1913** | 0.2000 | 1 |
| CIFAR-10 | `remax_laplace_diag` | **0.1969** | 0.2514 | 1 |
| CIFAR-10 | `logit_tagiv` | **0.1739** | 0.1767 | 1 |
| CIFAR-100 | `hrc` | 1.6503 | **1.3683** | 20 |
| CIFAR-100 | `remax_lognormal` | 1.8647 | **1.3068** | 2 |
| CIFAR-100 | `logit_tagiv` | **0.9575** | 0.9639 | 1 |

So on CIFAR-10 the untrained warm start beats every trained checkpoint for all
three non-hierarchical heads, while `hrc` trains properly and enormously.
Those three heads peak at epoch 1-2 and decay after, with mean confidence
climbing to 0.99 at flat accuracy — the 20-epoch screen and the 200-epoch
confirm both read them well past their best. **Decided:** run confirm at 200
epochs as planned anyway; its checkpoint list already includes epochs 1, 2 and
3, so `evaluate` will report each head's true optimum and the 200-epoch
horizon side by side, with five seeds on both.

### The open questions, updated

1. ~~Is the +22 point gap a rate effect or a floor?~~ **Neither — it was a
   gain confound, and it is largely gone.** The §8 probe sat at gain 0.3 /
   `sigma_v` 0.1, where `remax_lognormal` random is 0.5271 and zero 0.7471.
   Tuning gain per arm on the finished CIFAR-100 grid, random reaches 0.7587
   (gain 0.03) and `remax_laplace_diag` random reaches 0.7482 (gain 1.0), so
   the accuracy gap essentially closes. He random means are not on a floor and
   are not merely slow: they need a *particular* gain, and the probe used the
   wrong one for them.

   What survives, and is the better result: **`zero` is the arm that does not
   care.** `remax_laplace_diag` zero is 0.7550 / 1.569 at every one of the
   four gains, identical to four decimals, while random spans 0.4007 to 0.7482
   — a 35-point accuracy range. Recommend `zero` for insensitivity, not for a
   better optimum.

   Still open at full protocol: whether that holds at 200 epochs, and whether
   selection on NLL is even the right rule here — on CIFAR-100 it prefers
   badly underconfident cells (`remax_lognormal` backbone gain 0.03: NLL
   1.582, ECE 0.440, mean confidence 0.3235 at 76% accuracy, against gain 0.3
   at NLL 1.649 and ECE 0.045). Worth raising: for a calibration study, NLL
   with Brier and ECE only as tiebreakers rewards hedging.
2. ~~Init and gain interact.~~ **Answered, and the plan's predicted direction
   was backwards.** See §8. Supersedes the old note: read table 3 as "which
   prior width does each arm want", and lead with `zero`'s flatness.
3. **Does calibration absorb the init effect?** New, and it is now the
   question that decides the recommendation — deliverable table 5. If a global
   gain absorbs the whole delta, the advice is "initialize however you like and
   calibrate"; if it does not, it is "never use He means on a many-class last
   layer".
4. **`hrc` on ImageNet was catastrophic before** (§6 gap 4, acc 0.4232).
   Unchanged, still untested.

### Housekeeping

- Nothing in `runs/` was deleted this session. The v2 `report.csv` /
  `report_summary.csv` were **regenerated** in place after the schema change,
  verified row-for-row identical on the shared columns; a pre-change copy is
  in the session scratchpad, which is not durable.
- `runs/last_layer/.../heads/init_screen/` is ~150 MB for the padded screen and
  will roughly grow by a third with the full-tree arm.
- The ImageNet inputs were pre-flighted: 129 train shards, 5 val shards,
  `pretrained_fc.pt` and `feature_statistics.pt` all present under
  `features_shuffled`, which is where the driver reads them. Note §2's table
  says `features/pretrained_fc.pt`; both exist, and the driver uses the
  `features_shuffled` copy.
- GPU 1 is running unrelated user jobs (a streamlit app and
  `next_token_predictor.py`). Everything here ran on GPU 0. The ImageNet
  overnight queue can halve its wall clock with `--gpu-shards 2` **only** if
  GPU 1 is free by then; one GPU at ~6.6 h still fits.
- `experiments/scaling_theory_stage2` (3.9 GB, untracked) still awaits its
  delete/keep decision. Still unrelated.
