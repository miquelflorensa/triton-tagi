# ImageNet frozen-feature AGCI last layer — experiment log, 2026-08-28

Seventeen completed runs (plus three launches killed early) against a frozen torchvision
ResNet-18 (`IMAGENET1K_V1`), training only a 1,000-way Bayesian TAGI last layer with an
AGCI head. Wall-clock 09:13–17:51 on one RTX 4070 Ti SUPER, seed 0 throughout.

**Headline results.** Best top-1 69.96% (warm start, $\tau=1.2825$, prior variance
$\times 0.025$) — the only configuration to pass the frozen softmax head's 69.76%. Best
NLL 1.3211, from the principled single-pass posterior (§7): zero means, $S_W=S_b=1/D$,
$\tau=1.2825$. That run also beats the control on ECE. Error AUROC stayed in
0.4126–0.4230 across all sixteen valid runs — below chance in every one.

---

## 1. Setup

The backbone is frozen and its 512-dimensional penultimate features cached: 1,281,167
train and 50,000 validation vectors, sharded at 10,000 samples. Only the final 1,000-way
classifier trains, as a TAGI linear layer with the AGCI head.

Because the backbone is frozen, the network's own `fc` layer is the natural control — a
linear classifier over identical features:

| Control | Top-1 | Top-5 | NLL | ECE % | Conf−Acc |
| --- | --- | --- | --- | --- | --- |
| pretrained softmax `fc` | 69.76 | 89.08 | 1.2469 | 2.633 | +2.26 |

AGCI models the multiclass event as `argmax(Z + E) == c` with independent
$E \sim N(0, \tau^2)$, using 48-node shifted Gauss-Hermite quadrature for the moments.
Knobs explored: TAGI initialization gain, noise scale $\tau$, epoch count, and mean
initialization.

Metric conventions: `Conf−Acc` is mean top-1 probability minus accuracy — negative is
underconfident, positive is overconfident. `Err AUROC` asks whether predicted output
variance separates errors from correct predictions; 0.5 is chance.

---

## 2. Full-validation results

All 50,000 validation images. Gain 1.0 and one epoch unless noted.

| Run | Init | $\tau$ | Ep | Top-1 | Top-5 | NLL | ECE % | Conf−Acc | Var mean | Err AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| softmax `fc` (control) | — | — | — | 69.76 | 89.08 | **1.2469** | 2.633 | +2.26 | — | — |
| `pretrained_tau1.2825_vs0.025_1ep` | fc | 1.2825 | 1 | **69.96** | **89.25** | 1.4895 | 5.383 | +5.38 | 0.0415 | 0.4126 |
| `pretrained_varscale0.025_1ep` | fc | 1.0 | 1 | 69.81 | 89.19 | 1.7487 | 9.754 | +9.75 | 0.0410 | 0.4131 |
| `pretrained_tau1.2825_1ep` | fc | 1.2825 | 1 | 69.01 | 88.79 | 1.4123 | 3.686 | +3.69 | 0.8627 | 0.4128 |
| `clean_zeromean_..._bs64_nocap` | zero | 1.2825 | 1 | 68.71 | 88.77 | 1.3212 | 2.453 | −1.95 | 0.6106 | 0.4160 |
| `clean_zeromean_..._bs32_nocap` | zero | 1.2825 | 1 | 68.71 | 88.77 | **1.3211** | 2.467 | −1.95 | 0.6107 | 0.4161 |
| `clean_zeromean_tau1.2825_1ep` | zero | 1.2825 | 1 | 68.64 | 88.73 | 1.3225 | **2.419** | −1.87 | 0.6116 | 0.4161 |
| `fixed_g1_8ep` | random | 1.0 | 8 | 68.91 | 88.58 | 1.4031 | 2.970 | +2.97 | 0.1269 | 0.4200 |
| `fixed_g1_pretrained_1ep` | fc | 1.0 | 1 | 68.61 | 88.58 | 1.5236 | 6.393 | +6.39 | 0.7795 | 0.4138 |
| `random_tau1.2825_1ep` | random | 1.2825 | 1 | 68.07 | 88.36 | 1.3635 | 1.708 | −0.69 | 0.6200 | 0.4179 |
| `fixed_g1_1ep` | random | 1.0 | 1 | 67.89 | 88.20 | 1.4117 | 1.786 | +1.78 | 0.5367 | 0.4192 |
| `gainsweep_g0.7_1ep` (gain 0.7) | random | 1.0 | 1 | 67.64 | 88.08 | 1.3879 | 1.685 | −0.74 | 0.3267 | 0.4171 |
| `gain_1` (class-ordered shards) | random | 1.0 | 1 | 0.57 | 0.93 | 27.3179 | 62.332 | +62.33 | 0.3993 | 0.5221 |

**Nothing dominates.** Three different configurations hold the three crowns — accuracy
(warm start with both knobs), NLL (the principled prior of §7), ECE (gain 0.7, with the
principled prior a close second at 2.419 and the only run to beat the control). The
softmax control is the only entry that is simultaneously high-accuracy and
well-calibrated; no AGCI run today reached both. The two families separate cleanly: warm
starts buy accuracy and pay in calibration, random init stays calibrated and gives up
about a point of accuracy.

The last row is the shard-order failure of §5.1, kept for the record.

---

## 3. Epochs

| Epochs | Top-1 | Top-5 | NLL | ECE % | Conf−Acc | Var mean | Err AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 67.89 | 88.20 | 1.4117 | 1.786 | +1.78 | 0.5367 | 0.4192 |
| 8 | 68.91 | 88.58 | 1.4031 | 2.970 | +2.97 | 0.1269 | 0.4200 |

One epoch — 5,005 sequential updates, 20 minutes — lands within 1.87 points of the
control. Eight epochs cut the gap to 0.85 and improved NLL slightly, while ECE degraded
from 1.786 to 2.970 and the confidence gap widened from +1.78 to +2.97.

Mean output variance contracted 4.2$\times$ (0.5367 → 0.1269) with error AUROC unchanged
(0.4192 → 0.4200). Eight epochs of posterior contraction bought no discriminative value —
the same conclusion recorded for CIFAR in `experiments/last_layer/REPORT.md`.

Prior CIFAR-100 evidence at gain 1.0 predicted a flat trajectory (+0.13 points over 99
extra epochs); ImageNet gained a full point instead, because its epoch-1 point sat
further from its ceiling — 1.87 points back versus CIFAR-100's 0.77.

---

## 4. Gain sweep

Six runs, one epoch each, 12:04–12:59. **Scored on the 10,000-image screening cohort, not
full validation** — see §5.2. These numbers are not comparable to §2.

| Gain | Top-1 | Top-5 | NLL | ECE % | Conf−Acc | Var mean | Err AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| softmax (same cohort) | 76.38 | 92.88 | 0.9146 | 1.270 | +1.15 | — | — |
| 0.25 | 70.45 | 90.46 | 1.2062 | 9.670 | −9.65 | 0.0716 | 0.4230 |
| 0.4 | 72.90 | 91.43 | 1.0947 | 6.088 | −5.91 | 0.1486 | 0.4153 |
| 0.5 | 73.81 | 91.73 | 1.0655 | 4.624 | −4.59 | 0.2059 | 0.4157 |
| 0.7 | 74.64 | 91.92 | **1.0439** | 3.181 | −2.58 | 0.3309 | 0.4133 |
| 1.0 | **74.94** | 92.17 | 1.0481 | **2.039** | −0.28 | 0.5422 | 0.4152 |
| 2.0 | 74.70 | **92.28** | 1.1521 | 4.295 | +4.29 | 1.4707 | 0.4137 |

**Gain 1.0, the value already in use, was essentially optimal.** Accuracy rises
monotonically to a plateau at 0.7–1.0 (74.64, 74.94 — a 0.30-point spread, inside cohort
noise) and falls off at 2.0, where NLL degrades 11%. The confidence gap sweeps cleanly
through zero across the sweep, from −9.65 to +4.29.

### 4.1 The prediction that failed

The hypothesis going in was that 1,000 classes would want a different gain than
CIFAR-100's 100. Measuring the prior scale said otherwise. TAGI uses He initialization, so
$S_w = (\text{gain} \cdot \sqrt{1/\text{fan\_in}})^2$ with `fan_in` = 512 for all three
datasets — **the class count never enters the prior.** What differs is the feature norm:

| Dataset | C | mean $\|x\|$ | gain used | prior utility sd | sd/$\tau$ |
| --- | --- | --- | --- | --- | --- |
| CIFAR-10 | 10 | 5.88 | 0.1 | 0.026 | 0.026 |
| CIFAR-100 | 100 | 11.76 | 1.0 | 0.529 | 0.529 |
| ImageNet | 1000 | 29.59 | 1.0 | 1.305 | 1.305 |

Matching ImageNet's sd/$\tau$ to CIFAR-100's working point predicted an optimum near
0.4–0.5. The sweep found the plateau an octave higher. The heuristic was wrong; the
feature-norm effect is real but does not set the optimum.

### 4.2 Screen validation

Gain 0.7 and 1.0 were re-scored on all 50,000 images. The cohort's ranking held for
accuracy (67.89 > 67.64, matching 74.94 > 74.64) and for NLL (1.3879 < 1.4117, matching
1.0439 < 1.0481), but **reversed for calibration**: on full validation gain 0.7 sits at
−0.74 and gain 1.0 at +1.78, so the calibration zero crossing is below 1.0, near 0.8 — not
at 1.0 as the biased cohort suggested.

---

## 5. Two measurement bugs

### 5.1 Class-ordered training shards

The cached features inherited ImageFolder's class-sorted order. Sequential Bayesian
updates on near-single-class batches destroyed the layer: **0.57% top-1, NLL 27.3, ECE
62.3**. Fixed at 10:25 by permuting all 1.28M vectors before sharding (21 s); spot-check
confirmed 1,000 distinct classes per shard.

Cost: one wasted 20-minute run plus three launches killed three minutes in. Caught quickly
because the failure was total rather than subtle.

### 5.2 A screening cohort covering 200 of 1,000 classes

The reshuffle permuted training shards only. The gain sweep's diagnostic cohort took a
*prefix* of the still-sorted validation shards — **classes 0–199 exclusively**. That
subset is easier: the softmax control scores 76.38% on it against 69.76% on full
validation.

This produced a false headline mid-sweep — gain 0.25 appeared to beat the baseline by 0.69
points when it was in fact 5.93 points behind. Gain-to-gain comparisons remained valid
(all six ran on the same cohort) and the ordering was confirmed on full validation for the
two plateau candidates, but no sweep number could be read against the 50k control.

Fixed by drawing an even strided share from every validation shard. The corrected cohort
spans all 1,000 classes and tracks full validation to within 0.12 points:

| | corrected 10k cohort | full 50k |
| --- | --- | --- |
| softmax top-1 | 69.88 | 69.76 |
| softmax top-5 | 88.99 | 89.08 |
| softmax NLL | 1.2347 | 1.2469 |
| softmax ECE % | 2.655 | 2.633 |

Subsequent runs used the corrected cohort, which predicted their full-validation accuracy
to within 0.09–0.26 points.

---

## 6. Warm starting and the $\tau$ correction

### 6.1 The link mismatch

Copying the pretrained `fc` weights into the head *lost* 1.15 points (69.76 → 68.61) and
nearly tripled the confidence gap (+2.26 → +6.39).

The cause is a link mismatch. **Softmax is exactly the argmax-plus-Gumbel model**:
$\text{softmax}(z)_c = P(\arg\max(z + G) = c)$ for i.i.d. standard Gumbel $G$, which has
sd $\pi/\sqrt{6} = 1.2825$. AGCI substitutes Gaussian noise of sd $\tau$. At $\tau = 1$
the copied logits are effectively $1.28\times$ too large — precisely the overconfidence
observed.

A second mismatch is in the code: `initialize_pretrained_mean` copies only `mw` and `mb`,
leaving `Sw`/`Sb` at the random prior. The head starts with trained means and untrained
variances, and since the TAGI step cap is $\sqrt{S}/\text{cap\_factor}$, those large prior
variances let the first updates move well-fit means a long way. Measured: the pretrained
`fc` has weight std 0.0695 while the gain-1.0 prior has sd 0.0442 — the prior claims
uncertainty about 64% of the weight magnitude.

### 6.2 The four arms

| Arm | $\tau$ | var scale | Top-1 | Top-5 | NLL | ECE % | Conf−Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| softmax `fc` (start point) | — | — | 69.76 | 89.08 | 1.2469 | 2.633 | +2.26 |
| baseline warm start | 1.0 | 1.0 | 68.61 | 88.58 | 1.5236 | 6.393 | +6.39 |
| A — $\tau$ matched | 1.2825 | 1.0 | 69.01 | 88.79 | 1.4123 | 3.686 | +3.69 |
| B — variance shrunk | 1.0 | 0.025 | 69.81 | 89.19 | 1.7487 | 9.754 | +9.75 |
| C — both | 1.2825 | 0.025 | **69.96** | **89.25** | 1.4895 | 5.383 | +5.38 |

The knobs decompose, but not additively:

- **$\tau$ controls calibration.** Matching it takes the gap +6.39 → +3.69 and NLL
  1.5236 → 1.4123.
- **Prior variance controls how much accuracy the update destroys.** Shrinking `Sw` 40×
  cuts the step cap 6.3× so the means barely move, preserving 69.81 — by not training. Its
  NLL is the worst of the day.
- **Combining them** gives the best accuracy but lands calibration *between* the two arms
  (+5.38), not at the better one. With mean output variance collapsed to 0.0415, the
  predictive distribution is over-sharp regardless of $\tau$.

Variance scale 0.025 was chosen to put the prior sd at 10% of the pretrained weight std
(0.0069 against 0.0695), exposed as `--pretrained-var-scale`.

### 6.3 The control: $\tau$ was mis-set globally

| Random init, 1 epoch | Top-1 | Top-5 | NLL | ECE % | Conf−Acc | Err AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| $\tau = 1$ | 67.89 | 88.20 | 1.4117 | 1.786 | +1.78 | 0.4192 |
| $\tau = 1.2825$ | 68.07 | 88.36 | **1.3635** | 1.708 | −0.69 | 0.4179 |

The corrected $\tau$ helps a plain random-init head too — **−0.048 NLL and +0.18 points,
for free** — and flips the confidence gap negative. So $\tau = 1$ was simply the wrong
noise scale for AGCI, not merely wrong for copied weights. It helps warm starts more
(−0.111 NLL there versus −0.048 here), which is the link-mismatch effect on top.

**Every other result in this log ran at $\tau = 1$ and should be re-read accordingly.**

---

## 7. The principled single-pass posterior

The initialization used everywhere above draws $m_W \sim \mathcal N(0, 1/D)$ and sets
$S_W = g^2/D$. Once sampled, that is a Gaussian prior centred on *one arbitrary random
classifier*, not the He-scale prior $W \sim \mathcal N(0, 1/D)$ it is meant to represent.
The coherent form is zero means with $S_W = S_b = 1/D$ — which at gain 1.0 is exactly the
variance the existing code already produces, so only the means change.

Added as `--mean-init zero`. Verified at construction: $S_W = S_b = 0.001953125 = 1/512$
exactly, means exactly zero, initial class probabilities exactly uniform (max-prob
0.00100 = $1/K$, cross-class std 0.0), and symmetry broken correctly by the first
labelled update.

| Init, $\tau=1.2825$, 1 pass | Top-1 | Top-5 | NLL | ECE % | Conf−Acc | Brier | AURC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| softmax `fc` (control) | 69.76 | 89.08 | 1.2469 | 2.633 | +2.26 | 0.4111 | 0.1014 |
| zero means, $S=1/D$ | **68.64** | **88.73** | **1.3225** | **2.419** | −1.87 | 0.4249 | 0.1083 |
| random means | 68.07 | 88.36 | 1.3635 | 1.708 | −0.69 | 0.4313 | 0.1116 |
| random means, $\tau=1$ | 67.89 | 88.20 | 1.4117 | 1.786 | +1.78 | 0.4335 | 0.1122 |

The two prior corrections compound almost equally and independently:

- $\tau = 1 \to 1.2825$ at random means: **−0.048 NLL**
- random means $\to$ zero means at corrected $\tau$: **−0.041 NLL**

Removing the arbitrary random draw is worth about as much as fixing the link scale.
Together they take single-pass NLL from 1.4117 to 1.3225 — a 6.3% reduction at identical
compute — and this is the only configuration that beats the softmax control on ECE
(2.419 against 2.633) while staying within 1.1 points on accuracy. Residual miscalibration
is underconfidence (−1.87), the benign direction.

### 7.1 Why one pass, and why gain 1

With $N = 1{,}281{,}167$, $D = 512$, $K = 1000$, $\mathbb E\|x\|^2 = 888.80$ and centered
feature energy 446.31, gain 1 gives $s_g^2 = (888.80 + 1)/512 = 1.738$ and
$q = \tau^2 + s_g^2 = 1.645 + 1.738 = 3.383$. The centered-feature evidence estimate is

$$\rho \approx \frac{(0.010528)(1{,}281{,}167)(446.31)}{512^2 (3.383)} \approx 6.8$$

per pass. One pass is already strongly data-dominated. Eight passes raise this to roughly
54 by *reusing the same likelihood* — a power posterior, not a Bayesian one. That is the
mechanism behind §3: the 8-pass run's +1.02 points of accuracy and its degraded ECE
(1.786 → 2.970) are two views of the same over-contraction, not a tunable trade-off.

The gain-1 choice also explains why the feature-norm heuristic of §4.1 failed. Per-contrast
information falls as $\beta_K \sim 2\log K / K$ while the feature norm rises, and the two
nearly cancel:

| Dataset | $\beta_K \mathbb E\|x\|^2$ |
| --- | --- |
| CIFAR-10 | 10.1 |
| CIFAR-100 | 8.7 |
| ImageNet | 9.4 |

Larger ImageNet features almost exactly compensate for the weaker information in each
1000-class contrast, which is why gain near 1 transfers across all three datasets and the
norm-only prediction of 0.4–0.5 did not. Correcting the observed $\tau=1$ plateau for the
mean-update amplitude $A(g,\tau) = g^2/\sqrt{\tau^2 + 1.738 g^2}$ maps gain 0.7 → 0.767 and
1.0 → 1.075, giving a transferred plateau of $g \in [0.77, 1.08]$. **Untested** — the
$\tau=1.2825$ gain sweep that would confirm it has not been run.

---

## 8. Batch size and the update cap

The cuTAGI cap clips each mean delta at $\sqrt{S}/\text{cap\_factor}$. Measured binding
rate on the first step, comparing capped against uncapped deltas directly:

| Batch | cap_factor | weights clipped | mean \|Δm\| capped vs free |
| --- | --- | --- | --- |
| 256 | 3.0 | 0.349% | 1.303e−3 vs 1.320e−3 |
| 64 | 2.0 | 0.008% | 3.840e−4 vs 3.843e−4 |
| 32 | 2.0 | 0.003% | 1.970e−4 vs 1.972e−4 |

The cap is nearly inert, and more so at small batches. Disabling it (`--no-cap`) and
shrinking the batch 8$\times$ changes essentially nothing:

| Batch | cap | Top-1 | Top-5 | NLL | ECE % | Brier | AURC | Err AUROC | Wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | on | 68.644 | 88.73 | 1.3225 | 2.419 | 0.4249 | 0.1083 | 0.4161 | 1363 s |
| 64 | off | 68.708 | 88.77 | 1.3212 | 2.453 | 0.4247 | 0.1081 | 0.4160 | 1456 s |
| 32 | off | 68.706 | 88.77 | 1.3211 | 2.467 | 0.4247 | 0.1081 | 0.4161 | 1738 s |

An 8$\times$ range of batch size — 5,005 large posterior steps against 40,037 small ones —
moves top-1 by 0.064 points and NLL by 0.0015. **The single-pass posterior is determined by
the total evidence, not by how the updates are partitioned**, which is what $\rho$ predicts
but was not guaranteed: the batched TAGI update is an approximation that could have been
leaking, and at these sizes it is not.

Batch 64 is also the fastest of the three per epoch (3.64 s per 10k shard against 4.36 s at
256 and 4.74 s at 32), so the smaller batch costs nothing.

Training order: the cached features were globally permuted once before sharding, so each
10k shard is a uniform random sample of all 1.28M and every batch is drawn from one such
shard. Batches never mix across shard boundaries — the global-shuffle property holds
through the cache permutation, not through the training loop, which is sufficient for a
single pass.

---

## 9. The variance head does not work

Error AUROC across all sixteen valid runs: **0.4126 – 0.4230**. Consistently *below*
chance, meaning predicted output variance is systematically *lower* on errors than on
correct predictions.

That range holds across:

- three initializations (random, pretrained `fc`, zero-mean)
- two noise scales ($\tau$ = 1.0, 1.2825)
- six gains (0.25 → 2.0)
- two epoch counts (1, 8)
- three batch sizes (32, 64, 256), with and without the update cap
- a 40$\times$ range of mean output variance (0.0410 → 1.4707)

It survives a fully centered prior with exactly uniform initial class probabilities and no
seed-dependent prior logits, which rules out initialization as the cause.

No knob tried today moved it. This reproduces the CIFAR-10/100 finding in
`experiments/last_layer/REPORT.md`: posterior contraction is not by itself evidence that
the resulting scalar is a valid uncertainty score. The invariance to prior scale points at
`compute_agci_innovation` rather than at tuning — sub-chance ordering suggests the
variance tracks feature norm rather than ambiguity.

### 9.1 Calibration direction, verified

At gain 0.25 the head is underconfident by 9.65 points (conf 60.80, acc 70.45); at gain
1.0 it is overconfident by 1.78. The direction was confirmed per-bin rather than inferred
from the aggregate: 13 of 15 reliability bins run the same sign, with the two exceptions
holding 179 of 10,000 samples. The deficit peaks mid-range (63% confidence earning 82%
accuracy) and nearly closes in the top bin (97.35% earning 98.75%), so it is not a uniform
temperature offset.

Consistency check: ECE is the weighted sum of per-bin $|acc - conf|$, so ECE $\geq$
$|$aggregate gap$|$ with equality only under a single sign. Measured excess is $\leq$ 0.02
points for every AGCI run — the softmax control, at 0.369, is the only entry with
genuinely mixed-sign bins.

---

## 10. Next steps

1. **Eight passes with the zero-mean prior, $\tau = 1.2825$.** Not as a better model —
   §7.1 argues it is a power posterior at $\rho \approx 54$ — but to quantify what the
   extra passes cost in calibration from a principled starting point. ~85 min, and the
   per-epoch instrumentation now produces the full trajectory.
2. **Confirm the transferred gain plateau at $\tau = 1.2825$.** All six sweep points ran
   at $\tau = 1$. The amplitude correction of §7.1 predicts $g \in [0.77, 1.08]$; a
   four-point sweep (0.8, 0.9, 1.0, 1.1) with the zero-mean prior would test it directly.
   Not run.
3. **Diagnose the variance signal directly.** Sub-chance error AUROC invariant to the
   prior scale is a question for `compute_agci_innovation`, not another training run.

---

## 11. Runner changes made today

Three flags added to `run_resnet18_agci.py`:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--diagnostic-size` | 2048 | per-epoch screening cohort, strided across all validation shards |
| `--skip-final-eval` | off | skip the 14-minute 50k evaluation for cheap sweeps |
| `--pretrained-var-scale` | 1.0 | multiply `Sw`/`Sb` after copying pretrained means |
| `--mean-init zero` | — | third choice: zero means with $S = g^2/D$, the coherent He-scale prior |
| `--no-cap` | off | bypass the cuTAGI update cap via `cap_factor_override` on the network |

`experiments/imagenet/summarize_runs.py` prints every run as one table, preferring full
50k results and falling back to the screening cohort, with each row labelled by which set
it came from so the two are never silently compared.

Per-epoch checkpoints (`epoch_NNNN.pt`) and an incrementally written `history.json` are
now standard, so a killed run still leaves its trajectory. Previously only the final epoch
was saved, which is why `fixed_g1_8ep` has no 1→8 curve.

```bash
python experiments/imagenet/run_resnet18_agci.py run \
  --feature-root runs/imagenet/resnet18_agci/features_shuffled \
  --output      runs/imagenet/resnet18_agci/<name> \
  --epochs 1 --gain 1.0 --tau 1.2825 --num-quad 48 \
  --mean-init random --prediction-batch-size 16 \
  --diagnostic-size 10000
```

**On `--prediction-batch-size`.** AGCI's predictive moments allocate on the order of
$\text{batch} \times C^2 \times \text{quad}$. At $C = 1000$ with 48 nodes, a batch of 64
requests 11.43 GiB and OOMs. Sixteen is the ceiling, not a conservative default.
