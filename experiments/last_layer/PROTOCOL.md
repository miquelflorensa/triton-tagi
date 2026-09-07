# Frozen protocol: logit-space TAGI-V last layer

Pre-registered before the ImageNet experiment. Nothing below is to be adjusted
after seeing an ImageNet result; anything that turns out to need adjusting is
reported as a finding, not folded into the method.

## Claim under test

> TAGI-V learns a well-ordered input-dependent surrogate for
> augmentation-induced logit variability and improves in-distribution
> calibration at unchanged accuracy on CIFAR-10 and CIFAR-100. The usefulness
> of that variance is dataset- and augmentation-dependent; absolute variance
> dispersion is dataset-dependent too.

Established on CIFAR-10 and CIFAR-100, and not established on ImageNet-100,
where the augmentation mean shift dominates. The axis is the dataset and its
augmentation distribution, not the class count: ImageNet-100 is also a
hundred-way problem, and a weak-teacher explanation is ruled out separately
below. ImageNet-1k is a pre-registered boundary test.

## Method

1. **Targets.** Center the teacher's logits per row and divide by one global
   scale computed on training data only. For each training input draw `M`
   augmented teacher logit vectors and reduce them to their mean `ybar` and
   unbiased sample variance `R`. Features stay on the evaluation transform.
2. **Mean warm-up.** Fit the latent stream alone against `ybar` under a fixed
   observation variance. The variance head keeps its prior throughout.
3. **Reset.** Restore the variance head's prior, keeping the fitted mean.
4. **Replicate-aware AGVI.** Fit the variance stream alone from `R`, with the
   mean frozen. For Gaussian replicates `var(R | S) = c_M^2 S^2` with
   `c_M^2 = 2 / (M - 1)`, giving `v_R = (1 + c_M^2) v_S + c_M^2 mu_S^2`,
   `Cov(S, R) = v_S`, and a posterior that always contracts.
5. **Initialization.** `mu_G = log(a_0 - s_min2) - v_G / 2` with
   `v_G = log(1 + c_0^2)`, split between weights and bias at the measured
   feature energy so that the head is input-dependent at all.
6. **Shrinkage.** Reapply a zero-mean Gaussian prior to the variance head's
   weights after every batch. The bias is never regularized: it carries the
   level, the weights carry the spread.
7. **Calibration.** Fit `(T, alpha)` jointly on the label-calibration split,
   with `alpha = 0` always a candidate.

Implementation: `triton_tagi.logit_tagiv` and
`TAGILastLayerClassifier.{fit_mean, reset_variance_head, fit_variance, calibrate}`.

## Fixed hyperparameters

| quantity | value |
| --- | --- |
| replicates `M` | 8 |
| gains `gain_w`, `gain_b` | 0.1 |
| initial aleatoric `a_0` | 0.1 |
| prior coefficient of variation `c_0` | 0.5 |
| variance weight share | 0.5 |
| mean warm-up observation variance | 0.01 |
| batch size | 256 |
| Monte Carlo draws for prediction | 256 Sobol |
| seeds | 0, 1, 2 |

Epochs are set by the mean stream reaching convergence on validation, not by a
fixed budget, and the budget actually used is reported.

## Selection

The shrinkage rate is chosen from the dimensionless grid

    kappa_0 = lambda_g * v_Wg,0  in  {0, 0.05, 0.10, 0.164, 0.25, 0.40}

converted per dataset by `lambda_g = kappa_0 / v_Wg,0`. A raw rate is never
transferred: the same `kappa_0 = 0.164` is `lambda_g = 200` on CIFAR-100 and
`lambda_g = 51` on CIFAR-10 because the feature energies differ.

The criterion is the mean absolute log ratio across p50, p90 and p99 of the
observed-class variance against the held-out replicated logits. Labels are never
used to select it. A squared error in log variance over all channels is not
usable: it is dominated by the `K - 1` channels a cold start already fits.

`kappa_0 = 0` is a legitimate outcome and was in fact selected on CIFAR-10.

## Controls, retained in every run

- **constant**: the learned variance replaced by its calibration-split class
  mean. Keeps the level, discards the input dependence.
- **shuffled**: the learned variance permuted across examples. Keeps the
  marginal, discards the association with the input.

Each control refits its own `(T, alpha)`.

## Reported metrics

Accuracy, NLL, ECE, Brier, AURC and risk at 90% coverage on the test split,
evaluated once; teacher and teacher-plus-temperature baselines; the two
controls; selected `(kappa_0, T, alpha)`; per-seed variability.

Variance diagnostics: measured against learned p50, p90, p99 and `sd(log a)`;
per `(example, class)` rank correlation; the observed-class channel separately
from the rest. Read the median beside the mean: the learned variance is
log-normal by construction, so its mean is set by its upper tail.

## Success criteria

Required:

1. Rank correlation between the learned variance and the measured replicate
   variance, per `(example, class)`, at least 0.5.
2. Learned beats both the constant and the shuffled control on NLL.
3. NLL improves on teacher-plus-temperature.
4. Accuracy within 0.2 points of the teacher.
5. The selector returns a rate whose behaviour is explicable, `kappa_0 = 0`
   included.

Diagnostic only, not required:

6. Absolute level agreement. p50, p90 and p99 are reported honestly in whichever
   direction they miss. CIFAR-100 over-dispersed before shrinkage and CIFAR-10
   under-disperses with no available correction, so a miss here is expected and
   is a limitation to describe, not a failure to hide.

## Explicit negative boundaries

Carried into any write-up as results, not as open questions.

- The learned variance does not extrapolate under unsupported covariate shift.
  On CIFAR-100-C the predicted-class variance still falls with severity after
  its level is corrected, and no aggregation beats the teacher's own energy
  score. The head only ever saw clean-feature support.
- Absolute dispersion is dataset-dependent and the shrinkage lever is one-sided:
  it removes spread and cannot add it.

## ImageNet questions

1. Does the variance ordering remain meaningful at K = 1000?
2. Does learned heteroscedasticity beat the constant and shuffled controls?
3. Does NLL improve beyond teacher temperature scaling?
4. Is accuracy unchanged?
5. Does the replicate selector choose a sensible `kappa_0`?
6. Is the computation competitive for a Bayesian last-layer method?

## ImageNet-100 outcome

The augmented-mean target is not representable by a clean-feature affine head:
it asks for `W E_aug[h(aug(x))] + b`, which is linear in the augmented feature
mean and not in `h(x)`. CIFAR hides this because its augmentation is mild enough
that the two nearly coincide; ImageNet's `RandomResizedCrop(0.08, 1.0)` does not.
The corrected model decouples the two heads, distilling the clean teacher logits
into the mean and the replicate variance into `G`, and initializes the mean from
the centered teacher classifier so the warm-up assimilates rather than
re-derives. That restores accuracy exactly (0.9116 against the teacher's
0.9116) where the original target lost 0.76 points.

The ceiling analysis then bounds what the variance channel is worth, on 2500
test rows with 10,000 paired bootstrap draws:

| comparison | mean gain (nats) | 95% CI |
| --- | --- | --- |
| learned vs teacher + T | +0.00033 | [-0.00018, +0.00088] |
| learned vs constant | +0.00041 | [-0.00002, +0.00090] |
| learned vs shuffled | +0.00074 | [+0.00027, +0.00129] |
| measured-variance oracle vs no variance | +0.00201 | [-0.00137, +0.00540] |
| calibrated TTA vs teacher + T | +0.04345 | [+0.02566, +0.06190] |

Only two intervals exclude zero. The learned variance beats a destroyed
ordering but not a constant one, and replacing it with the directly measured
replicate variance raises the ceiling only to about 0.002 nats, itself not
distinguishable from zero. Temperature-calibrated test-time augmentation gains
0.043 nats at eight forward passes, two orders of magnitude more than anything
the variance channel offers.

On ImageNet-100 the useful content of the augmentation distribution is therefore
its mean shift, not its variance. The supported statement is narrow:

> The usefulness of augmentation-induced logit variance is dataset- and
> augmentation-dependent. It improves calibration on CIFAR-10/100 but provides
> no established benefit on ImageNet-100, where augmentation mean shift
> dominates.

A "weak teacher" explanation is not supported: CIFAR-10 has a stronger teacher
than ImageNet-100, 95.0% against 91.2%, and a larger benefit.

## Pre-registered ImageNet-1k prediction

Written before the ImageNet-1k result exists.

Stratifying ImageNet-100 by teacher difficulty shows the aggregate null is a
cancellation, not an absence. On the examples the teacher gets wrong the
measured-variance oracle gains 0.079 nats and the learned head 0.013, both with
intervals excluding zero; on the examples it gets right both lose a small but
significant amount, 0.005 and 0.001. The hardest confidence quintile, teacher
accuracy 0.62, shows the same pattern: oracle +0.0216 with the interval
excluding zero, learned +0.0010 without.

Because ImageNet-1k's teacher is 69.8% top-1 against ImageNet-100's 91.2%, the
wrong-answer stratum grows from 9% of the test set to 30%. Reweighting the
per-stratum gains by that mix, and assuming they carry over unchanged:

| quantity | observed at K=100 | predicted at K=1000 |
| --- | --- | --- |
| learned vs teacher + T | +0.00033 | +0.0034 |
| measured-variance oracle | +0.00201 | +0.0201 |
| calibrated TTA | +0.04345 | +0.1528 |

The prediction is therefore a roughly tenfold increase in both the learned and
the oracle gain, with the learned gain still an order of magnitude below TTA.
The run is a boundary test: confirmation would show the ImageNet-100 null is a
class-mix artifact rather than a property of the method, and a flat result at
K=1000 would show the boundary is real. Either way the primary outcome is the
paired bootstrap interval on

    Delta_NLL = NLL(teacher + T) - NLL(learned),

on one seed, with additional seeds only if that interval excludes zero and the
effect is practically visible.

The defect is not a uniform scale error. A global `alpha` can correct one of
those, and the calibrator is free to choose it. What the strata show is a
compressed *relative* dispersion across difficulty: the hard tail needs
substantially more variance and the easy majority needs approximately none, so
any global `alpha` large enough to help the tail damages the bulk. That is why
the learned and oracle gains both decompose into a positive contribution from
the wrong-answer stratum and a significant negative one from the right-answer
stratum, and why raising `alpha` cannot recover the difference.

The implied next model is a difficulty-conditioned scale
`alpha(x) = softplus(b0 + b1 u(x))` for a fixed difficulty score `u`, which
permits `alpha(x)` near zero on easy inputs. It is not part of this protocol and
must not be adopted on the strength of these strata alone: confidence gating is
itself a more expressive calibrator, so it would have to beat a constant
variance under the same gate, a shuffled variance under the same gate, and a
confidence-conditioned temperature with no variance channel at all. A two-bin
version at a preselected confidence threshold is the safer first diagnostic.

Note what this does not license. The learned head captures about one sixth of
the oracle's gain on the hard stratum, so even a confirmed prediction leaves the
variance head well short of the information that is measurably there, and far
short of what the augmentation mean delivers.

## ImageNet engineering notes

Reduce the replicates online. Accumulate a running mean and sum of squared
deviations over the `M` passes and store only `ybar` and `R`; keeping all `M`
logit vectors for ImageNet-1k would be 41 GB in float32 against 5 GB for the two
reductions in float16.

The replicate-aware update trains on one row per image rather than one row per
replicate, so its variance phase is `M` times cheaper than the tiled
single-observation AGVI. That is the form to use at scale, and question 6 should
be answered against the tiled alternative as well as against the teacher.
