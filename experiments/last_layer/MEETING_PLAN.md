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
| `hrc` | hierarchical binary probit tree | fixed `sigma_v` | tree nodes: 9 / 99 / 999 |
| `remax_lognormal` | Remax + lognormal moment match | fixed `sigma_v` | K classes |
| `remax_laplace_diag` | Remax + diagonal Laplace Jacobian | fixed `sigma_v` | K classes |
| `logit_tagiv` | logit-target TAGI-V (distillation) | learned | 2K interleaved |

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

Nothing in the repo currently exposes this. `TAGILastLayerClassifier` always
uses He random means; the zero-mean path existed only inside the AGCI branch
that `ae4787c` deleted. **This must be implemented before anything runs.**

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

**Open design decision — the HRC projection.** `remax_*` heads output K
classes, so `backbone` is a direct copy (transposed to `[in, out]`). `hrc`
outputs *tree nodes*, not classes, so a class→node map is required. Proposed
rule, to be confirmed by whoever implements it:

    w_node = mean(w_class for classes on the node's +1 branch)
           - mean(w_class for classes on the node's -1 branch)

which makes the node's prior mean the contrast the node actually scores.
`logit_tagiv` writes the backbone weights into the even (mean) channel and
leaves the odd (variance) channel at its existing prior.

If the projection turns out to be wrong or unstable, **report `backbone` for
the remax heads and mark it N/A for `hrc`** rather than silently substituting
something else. An honest hole beats a fabricated row.

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

### Hour 0–1 — unblock (nothing can run until this is done)

1. **Install torchvision.** It is missing; `triton_tagi/cifar_study.py` imports
   it, so `run_study.py` and three test modules cannot even import.
   Verify with `python -m pytest tests/unit -q` — expect 442 + the 3 restored
   modules, all passing.
2. **Implement `mean_init`.** Add `mean_init: str = "random"` and
   `backbone_fc: tuple[Tensor, Tensor] | None = None` to
   `TAGILastLayerClassifier.__init__`; apply after `self.linear` is built and
   before any head-specific prior init, so the TAGI-V / logit priors still win
   on the channels they own. Record `mean_init` in `state_dict()` and in every
   run `config.json`. Add unit tests: `zero` gives exactly zero means,
   `backbone` reproduces the fc tensor for a remax head, and all three arms
   leave `Sw` identical.
3. **Measure throughput.** Time one 20-epoch CIFAR-100 cell and one 1-epoch
   ImageNet cell. Every budget below is extrapolated from
   16.2 s / 20 epochs (CIFAR-100, 99-node tree) and is *unverified for
   ImageNet* — replace the estimates with measurements before committing to
   the overnight queue.

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

| heads | init | gain | sigma_v | cells |
|---|---|---|---|---|
| hrc, remax_lognormal, remax_laplace_diag | 3 | 3 (0.1, 0.3, 1.0) | 2 (0.1, 0.3) | 54 |
| logit_tagiv | 3 | 3 | — | 9 |
| | | | | **63** |

Budget from the measurement in Hour 0–1. If a cell exceeds ~4 min, cut the
gain axis to {0.1, 0.3} and say so in the report.

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

1. **No ImageNet OOD set is cached.** CIFAR gets SVHN + CIFAR-C; ImageNet gets
   nothing. Either cache one (ImageNet-O / Places / Textures) during Day 1, or
   report ImageNet with accuracy + calibration only and say so in the table
   caption. Decide early — it changes the Day 2 budget.
2. **`hrc` backbone init needs the class→node projection** of §2. Unresolved.
3. **ImageNet timing is extrapolated**, never measured. Fix in Hour 0–1.
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
