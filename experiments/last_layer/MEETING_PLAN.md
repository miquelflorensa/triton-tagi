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

3. **`logit_tagiv` is insensitive to all of it** (0.1770 / 0.1770 / 0.1771).
   Expected for a distillation head: it regresses the teacher's logits, so the
   teacher determines the fixed point regardless of where the mean starts.
   Say so rather than presenting three near-identical numbers as a null result.

---

## 9. Handoff — state as of 2026-09-07, end of session 1

Hour 0–1 is complete and both blockers are gone. No screen has been run yet:
the CIFAR screen was started, found to be ~10x slower per cell than the
measurement predicted, stopped, and the cause fixed (see "What changed" #4).
It is ready to relaunch.

### What is done

1. **torchvision installed** — `0.29.0+cu130`, `torch 2.14.0+cu130` untouched.
2. **`mean_init` implemented and tested** — `triton_tagi/classification.py`,
   plus `project_classes_to_nodes` in `hrc_softmax.py`. 30 tests in
   `tests/unit/test_last_layer_mean_init.py`. See §2.
3. **Throughput measured** on both datasets and on ImageNet. See §4.
4. **`classwise_calibration_error` rewritten** — it looped over classes x bins
   with a device sync per bin, costing 4.6 s per evaluation on CIFAR-100. With
   21 evaluations per cell that was ~100 s of a ~150 s cell, and at 1000
   classes it would have made the ImageNet arm impractical. It now uses one
   `scatter_add` and the identity
   `(count/N) * |mean_t - mean_c| = |sum_t - sum_c| / N`. **19x faster at 100
   classes, 716x at 1000**, equal to the old definition to float32 precision.
   The old double loop is kept in `tests/unit/test_metrics.py` as the
   definition the fast path must agree with. Suite: **491 passed**.
5. **CIFAR driver plumbed** — `run_study.py` gained stages `init_screen` and
   `init_confirm`, the `init_study` section of `study.json` (116 cells per
   dataset), backbone `fc` loading for the warm-start arm, and teacher-logit
   training for `logit_tagiv`. `select_stage` now carries `mean_init` into its
   selection; **without that fix every confirm run would have silently fallen
   back to `random` means.** Verified end to end on a 2-epoch grid
   (screen -> select -> confirm configs), whose artifacts were then deleted.
6. **ImageNet driver written** — `run_imagenet_init_study.py` +
   `imagenet_init_study.json`, 60 cells. Streams `features_shuffled` shard by
   shard, because ImageNet will not fit the way `run_study.py` holds CIFAR.
   Splits across both GPUs with `--gpu-shard i --gpu-shards 2`.
   **Not yet run at all** — not even one cell.
7. **Two decisions taken** (§1 tree decision, §6 gap 1) and **three gaps
   closed** (§6 gaps 1–3).
8. **Preliminary result in hand** — §8. Zero means beat He random means by
   **+22 accuracy points** for the remax heads on CIFAR-100 at 20 epochs.

Commits: `a8b6bcd` (mean_init, tests, measured budgets), `d1a4968` (drivers,
classwise ECE). Working tree clean; nothing in `runs/` was deleted except two
of my own smoke/timing trees, described below.

### What remains

**Day 1 — CIFAR.** Relaunch the screen. 6 of 232 cells are already complete
and the driver skips completed runs, so this is safe to just re-run:

    python experiments/last_layer/run_study.py run --dataset cifar10  --stage init_screen
    python experiments/last_layer/run_study.py run --dataset cifar100 --stage init_screen

**Run these sequentially, not one per GPU.** The workload is CPU-launch-bound,
not GPU-bound: TAGI issues ~3000 small kernel launches per epoch from Python,
so two workers on two GPUs starve each other on CPU rather than running twice
as fast. That, together with the old `classwise_ece`, is what made the first
attempt look 10x slow — 150 s per cell instead of the 20 s predicted.

**Measured per-cell cost, uncontended, with the `classwise_ece` fix**
(CIFAR-100, 20 epochs, through the driver, so including the 21 validation
evaluations, the per-epoch native diagnostics and 6 checkpoint saves that the
bare §4 numbers exclude):

| head | seconds/cell |
|---|---|
| `remax_lognormal` | 43–50 |
| `hrc` | 47–58 |
| `remax_laplace_diag` | ~66 |

So budget the CIFAR screen at **≈105 min for CIFAR-100 and ≈55 min for
CIFAR-10, ~2.7 h sequential** — not the 70 min §4 implies, because §4 timed
training only. The driver overhead is ~2.5x the bare training cost and is
dominated by 21 full `classification_metrics` calls on CPU tensors; if that
ever needs to come down, evaluate at fewer epochs rather than optimizing
further.

These timing cells also reproduced §8 exactly (`remax_lognormal` CIFAR-100
`random` 0.5271 / 2.4907 vs `zero` 0.7471 / 2.0277), which confirms the driver
path and the standalone probe agree.

Then, per dataset:

    python experiments/last_layer/run_study.py select   --dataset <ds> --stage init_screen
    python experiments/last_layer/run_study.py run      --dataset <ds> --stage init_confirm
    python experiments/last_layer/run_study.py evaluate  --dataset <ds>

`init_confirm` runs the selected arm **and** its `random` counterpart per head
(8 configs per dataset x 5 seeds x 200 epochs), which is what makes table 2 a
full-protocol result rather than a 20-epoch one.

**Day 1 night — ImageNet.** Never run; start with a single cell to confirm the
measured 2.7–14.1 min/epoch still holds now that `classwise_ece` is fixed
(the ImageNet numbers in §4 were measured *without* per-epoch validation, so
they are training-only and slightly optimistic).

    python experiments/last_layer/run_imagenet_init_study.py run --stage screen \
        --gpu-shard 0 --gpu-shards 2   # and --gpu-shard 1 on the other GPU
    python experiments/last_layer/run_imagenet_init_study.py select --stage screen
    python experiments/last_layer/run_imagenet_init_study.py run --stage confirm
    python experiments/last_layer/run_imagenet_init_study.py report

**Day 2 — tables and deck.** The four tables of §5 plus the finished
calibration slide. Nothing there is written yet.

### Open questions for whoever picks this up

1. **Is the +22 point gap a rate effect or a floor?** The single most
   important thing the confirm stage answers. At 20 epochs `random` sits at
   0.527 on CIFAR-100 and `zero` at 0.747; if 200 epochs closes that, the
   recommendation is "converges anyway, but slowly", and if it does not, the
   recommendation is "never use He means on a many-class last layer". These
   are different slides.
2. **Init and gain interact, and the grid can see it.** `backbone` places the
   mean 4–6 prior sigmas out at gain 0.3 (§8 note 2), which predicts it should
   look much better at gain 1.0. Read table 2 at more than one gain before
   concluding `backbone` is simply worse.
3. **`hrc` on ImageNet was catastrophic before** (§6 gap 4, acc 0.4232). Now
   that `mean_init` exists, check whether `zero` or `backbone` rescues it. If
   the failure was really a bad random prior over 1001 tree nodes, this is the
   experiment that shows it, and it becomes a much better story than "the tree
   does not scale".

### Housekeeping

- `runs/` is untouched apart from `heads/init_screen/`, which holds the 6
  completed cells of the interrupted first screen attempt. Those are real
  20-epoch runs on the final grid and the driver will skip them on relaunch.
  Two throwaway trees (a 2-epoch smoke test and a timing scratch tree) were
  created and deleted within this session; nothing else in `runs/` was
  removed, and none of the ImageNet feature caches were touched.
- `experiments/scaling_theory_stage2` (3.9 GB, untracked) still awaits its
  delete/keep decision. Still unrelated to this plan.

