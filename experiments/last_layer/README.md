# Frozen-feature CIFAR last-layer study

This directory implements the staged CIFAR-10/CIFAR-100 comparison in
[study.json](study.json). The ResNet-18 backbone is deterministic and frozen;
every TAGI method trains only the final feature-to-output layer. The original
PyTorch classifier is the softmax baseline. No command fits temperature scaling
or any other post-hoc calibration.

The official 50,000-example CIFAR training set is deterministically stratified
before any supervised training. The backbone, its PyTorch softmax classifier,
and all last-layer heads train on 40,000 examples. Hyperparameters and epochs
are selected only on the remaining 10,000 validation examples. The official
10,000-example test set is never used for fitting or selection; its metrics are
computed only by the final `evaluate` command.

Install the optional dependencies with:

    pip install -e ".[examples,vis]"

Place the official CIFAR-10-C and CIFAR-100-C NumPy archives at the paths in the
manifest. Each directory must contain labels.npy and the 15 canonical corruption
files.

For each dataset, run:

    python experiments/last_layer/run_study.py backbone --dataset cifar10
    python experiments/last_layer/run_study.py cache --dataset cifar10
    python experiments/last_layer/run_study.py run --dataset cifar10 --stage screen
    python experiments/last_layer/run_study.py select --dataset cifar10 --stage screen
    python experiments/last_layer/run_study.py run --dataset cifar10 --stage refine
    python experiments/last_layer/run_study.py select --dataset cifar10 --stage refine
    python experiments/last_layer/run_study.py run --dataset cifar10 --stage tagiv
    python experiments/last_layer/run_study.py select --dataset cifar10 --stage tagiv
    python experiments/last_layer/run_study.py run --dataset cifar10 --stage confirm
    python experiments/last_layer/run_study.py evaluate --dataset cifar10

Repeat with cifar100, then generate the combined machine-readable report:

    python experiments/last_layer/run_study.py report

Runs are keyed by a hash of their complete configuration and skipped after a
complete.json marker is written. Use --force only to deliberately rerun an
identical configuration. Feature shards include the backbone SHA-256 and split
metadata. Screening and selection use only the fixed 10,000-example validation
split; clean test, CIFAR-C, and SVHN results are produced only by evaluate.

## Multinomial-probit ADF follow-up

Run the fixed-`tau` AGCI head on the same frozen ResNet-18 CIFAR-10 features
with:

    python experiments/last_layer/run_agci.py --dataset cifar10
    python experiments/last_layer/run_agci.py --dataset cifar100 \
        --prediction-batch-size 256

The runner estimates and freezes the training-feature mean, computes the
centered feature energy, and defaults to the dimensionless prior scale
`kappa=1`. It then derives `gain_w` from `kappa`, `tau`, feature energy, and
the fan-in-scaled bias variance. Use `--kappa` to choose another declared
prior ratio. `--gain` is an explicit raw-gain override, and
`--no-center-features` disables centering for an ablation. The frozen feature
transform is serialized in every last-layer checkpoint and reused for
validation, test, and OOD inputs.

To retain the AGCI class-event update but turn its Gaussian utilities into
probabilities and native output variances with ReMax, run:

    python experiments/last_layer/run_agci.py --dataset cifar10 \
        --predictive-link remax

This hybrid deliberately applies ReMax only at prediction time. AGCI training
continues to condition the pre-ReMax utilities on the observed argmax event.
The utility means are centered across classes before ReMax because AGCI
identifies only utility contrasts, while ReMax otherwise depends on their
arbitrary shared offset.

This uses `tau=1`, selects a checkpoint only from validation metrics, and then
evaluates the selected last layer on clean CIFAR and reports closed-set SVHN
OOD ranking from entropy, negative maximum probability, and native output variance.

After the CIFAR-10 feature cache exists, reproduce the dense output-layer ADF
screen for canonical and epistemic-only probit noise with:

    python experiments/last_layer/run_multinomial_probit.py

The runner screens both `probit_tau2=1` and `probit_tau2=0`, selects only from
validation checkpoints, evaluates the selected checkpoint on the clean test set,
and writes resumable JSON results under the study artifact root.

Track clean/SVHN OOD metrics over every ADF training epoch with:

    python experiments/last_layer/run_multinomial_probit_trajectory.py

Evaluate the directional full-covariance parameter posterior with block-EP site
replacement and intrinsic entropy, posterior variance, contrast-log-determinant,
epistemic-only entropy, and BALD scores with:

    python experiments/last_layer/run_full_covariance_adf.py

This variant keeps a dense 513 by 513 posterior precision for each class and
uses only CIFAR-10 training labels. Repeated epochs replace each observation's
old Gaussian event site before recomputing it, avoiding repeated-evidence
double counting. SVHN is used only for the reported diagnostic trajectory and
never for training or selection.

To fit the parameter-free Bayesian feature-support model on CIFAR-10 training
features only and evaluate the selected ADF checkpoint on SVHN and all 75
CIFAR-10-C conditions, run:

    python experiments/last_layer/run_feature_support_gate.py

This experiment reports conditional ADF scores, raw class-mixture evidence,
the support/background log Bayes factor, and the explicit K+1 open-set
distribution separately. It never uses SVHN or CIFAR-10-C for fitting,
selection, thresholding, or calibration.

For an explicitly supervised isolation test, fit a post-hoc linear domain
discriminant on CIFAR-10 validation features and one half of SVHN, then evaluate
on clean CIFAR-10 test and the disjoint SVHN half:

    python experiments/last_layer/run_posthoc_ood_probe.py

This is labeled as an oracle diagnostic rather than a candidate methodology. It
tests whether the fixed ResNet-18 representation contains enough linearly
accessible information for strong SVHN detection, while leaving the selected
ADF class head unchanged.

## Gumbel decision noise: the argmax-event logit link

The `gumbel_agci` head keeps the AGCI construction and changes only the law of
the decision noise, from `N(0, tau^2)` to `Gumbel(0, beta)`. That makes the
argmax event an exact multinomial logit conditional on the utilities, so the
class probability of a trailing class decays linearly rather than quadratically
in its utility margin. Section 1.2 of
[AGCI_LAST_LAYER_THEORY.md](AGCI_LAST_LAYER_THEORY.md) derives why that tail,
not the prior scale, dominates negative log likelihood.

`beta` is a pure gauge for this model exactly as `tau` is, so it stays at one
and the prior is described entirely by the dimensionless ratio `kappa`. The
update needs no competitor quadrature: the TAGI innovations are the first two
derivatives of the log evidence in the output mean, estimated from analytic
softmax derivatives on antithetic reparameterized draws.

Cost is `O(MK)` for both training and prediction. The Gaussian link matches
that for training at `O(QK)` but pays `O(QK^2)` for prediction, which needs
every candidate class. So the logit link is cheaper only at large `K`: measured
Gumbel-to-Gaussian wall-time ratios for one pass are 1.65 at `K=10`, 0.61 at
`K=100`, and 0.23 at `K=1000`. At `K=10` it is the slower of the two.

Which link scores better is dataset-dependent and must be measured, not
assumed. Sweeping `kappa` for both links and replicating four seeds each, the
logit link wins on ImageNet (+0.033 NLL) and CIFAR-10 (+0.0046) and *loses* on
CIFAR-100 (-0.0282); all three effects replicate at |t| >= 27. A comparison at a
single shared `kappa` is meaningless here, because the two links prefer
different priors. See section 13.5 of
[AGCI_LAST_LAYER_THEORY.md](AGCI_LAST_LAYER_THEORY.md) and
[the CIFAR comparison](../../runs/last_layer/decision_noise_cifar/RESULTS.md).

Reproduce the CIFAR comparison with:

    python experiments/last_layer/run_agci.py --dataset cifar100 \
        --decision-noise gumbel --kappa 1.5 --epochs 5 --validation-only

Two logit variants exist. `--decision-noise gumbel` integrates the link over
the prior predictive; the `logit_site` head takes probabilities from the prior
means alone and is the `S -> 0` limit of it. The prior-mean variant scores
better on NLL and is about four times cheaper; the integrated variant is
markedly better calibrated. Pick by objective.

On the ImageNet frozen ResNet-18 features, run it with:

    python experiments/imagenet/run_resnet18_agci.py run \
        --head gumbel_agci --kappa 1.0 --epochs 8 \
        --feature-root runs/imagenet/resnet18_agci/features_shuffled \
        --validation-root runs/imagenet/resnet18_agci/features \
        --output runs/imagenet/resnet18_agci/gumbel_agci_centered_kappa1.0_s32_8ep

`--gumbel-num-samples` sets the draw count and defaults to 32; results are
converged there and change by under 0.005 NLL at 8 draws. Draws are antithetic
and seeded from `--seed`, so a rerun reproduces a run exactly.

## The Core-Tail link: probit core, softmax tail, no fitted coefficient

The Gaussian and logit links disagree in two separate regimes, and section 13.5
of [AGCI_LAST_LAYER_THEORY.md](AGCI_LAST_LAYER_THEORY.md) measures both on the
same features: the logit link is better on the easy bulk and on the far tail,
the Gaussian link is better on near-misses. Neither dominates, so the sign of
the net depends on how a representation distributes mass across those strata,
which is exactly why the recommendation there has to be made per dataset.

The `ct_agci` head removes the compromise by construction rather than by
sweeping. It replaces softmax with the convex choice model

    p*(z) = argmax_{p in simplex} { z' p - sum_k [ p_k log p_k - a* p_k^2 (1 - p_k)^2 ] } ,

whose interior correction vanishes identically at `p = 0` and `p = 1`. Softmax
is the `a* = 0` member of the family, so the link keeps the exact softmax tail
including its leading constant, `-log p_c -> Delta`, and modifies only the
competitive interior. The coefficient is fixed by requiring that the binary
link have the same slope at a tie as a Gaussian decision model of matched noise
variance,

    1 / (4 + 2 a*) = sqrt(3) / (pi sqrt(2 pi))   =>   a* = 0.2732603854486113 ,

since the Gumbel utility difference is standard logistic with variance
`pi^2 / 3`. Nothing is fitted, and `beta` stays at one exactly as `tau` does,
so the prior is still described entirely by `kappa`.

Stationarity is the fixed point `p = softmax(z + 2 a* p (1 - p) (1 - 2 p))`,
which a Picard iteration contracts at rate `a*` and solves to below float32
resolution in six `O(C)` softmaxes. Differentiating it gives the exact
probability Jacobian `diag(w) - w w' / W` with `w_k = 1 / phi''(p_k)`, hence the
exact categorical score and diagonal Fisher curvature, both `O(C)` and both
reducing to `y - p` and `p (1 - p)` at `a* = 0`.

![Core-Tail against softmax and the variance-matched probit](report_assets/core_tail_link.png)

Reproduce the figure with:

    python experiments/last_layer/plot_core_tail_link.py

Run the controlled link comparison on the cached CIFAR features with:

    python experiments/last_layer/run_core_tail.py run --dataset cifar10
    python experiments/last_layer/run_core_tail.py run --dataset cifar100 \
        --kappas 0.4 0.6 0.8 1.0 1.5 2.0 3.0 --prediction-batch-size 256
    python experiments/last_layer/run_core_tail.py report

This holds the argmax-event construction and the prior parameterization fixed
and changes only the link, across `agci` (Gaussian), `gumbel_agci` (logit,
integrated over the prior predictive), `logit_site` (logit at the prior means)
and `ct_agci` (Core-Tail at the prior means). `logit_site` is the controlled
reference: it and `ct_agci` share every line of the update except the link, so
their difference is attributable to the link and to nothing else. Each link is
swept over `kappa` separately and compared at its own optimum with four seeds,
because the links prefer different priors. Only the validation split is read.

The report adds the decomposition of validation NLL by the rank the frozen
deterministic backbone assigns the observed class, which is the falsifiable
part of the proposal. Results are in
[the Core-Tail comparison](../../runs/last_layer/core_tail_cifar/RESULTS.md).

The Triton kernels launch on the current CUDA device, so `--device cuda:1` does
not work; select a second GPU with `CUDA_VISIBLE_DEVICES` instead.

## Logit-space TAGI-V: training through the teacher's logits

`run_logit_tagiv.py` replaces the one-hot label with the frozen backbone's own
logits and learns the observation variance of that regression with TAGI-V. The
observation model is `Y_k = Z_k + V_k` with `V_k | S_k ~ N(0, S_k)` and
`S_k = s_min2 + exp(G_k)`, so the variance head's pre-activation is Gaussian and
unconstrained and its exponential is the positive noise variance. The
exponential is what keeps the forward moments and the cross-covariance
`Cov(G, S) = v_G mu_X` analytic.

    python experiments/last_layer/run_logit_tagiv.py cache --repeats 8
    python experiments/last_layer/run_logit_tagiv.py run --seeds 0 1 2
    python experiments/last_layer/run_logit_tagiv.py report

Two target regimes are run because they differ in whether the observation noise
exists at all. In `clean` the targets are the backbone's logits on the same
evaluation-transform image whose features the head reads, so the teacher's own
last layer lies inside the student's hypothesis class and the residual is near
zero by construction. In `augmented` the features stay clean while the targets
are the backbone's logits under the training augmentation, so the residual is
the genuine spread a random crop and flip induce in the teacher's logits. That
spread is measurable by resampling, which makes the learned variance falsifiable
rather than merely plausible: `cache` stores eight repeats and the report scores
the learned `a(x)` against their empirical per-image variance on the held-out
validation split.

The learned variance is scored against three controls, each refitting its own
`(T, alpha)` so that none is handicapped by a calibration fitted for a different
channel. `constant` replaces it with its calibration-split class mean, keeping
the noise level and discarding the input dependence. `shuffled` permutes it
across examples, keeping its marginal and discarding the association with the
input; the SVHN permutation runs over the pooled test and SVHN rows, because
permuting inside each split would leave both marginals, and therefore the AUROC,
untouched. The teacher's own MSP, entropy, and energy scores are reported
alongside, since a learned variance that cannot beat a proxy already present in
the frozen logits is not worth its parameters.

Read the per (example, class) rank correlation, not the class-averaged one. With
many classes the class average is dominated by the observed class channel, whose
learned variance is both much larger than the others and the least reliable, so
that summary can invert a correlation that is positive channel by channel. On
CIFAR-100 the two disagree in sign.

    python experiments/last_layer/run_logit_tagiv.py corruptions --dataset cifar10 \
        --augmentation-repeats 8
    python experiments/last_layer/run_logit_tagiv.py corruption-report --dataset cifar10

The corruption sweep reuses one calibrated head across all 15 canonical
corruptions at all five severities and asks whether the learned variance rises
with severity, how well it separates clean from corrupted inputs, and what
happens to NLL, ECE, and the risk-coverage curve. With `--augmentation-repeats`
it also re-measures the teacher's augmentation spread on the corrupted images
themselves, so the learned quantity is checked against its intended target under
shift and not only in distribution.

The variance head is only heteroscedastic if its weights carry prior mass. The
log-variance pre-activation has prior variance `Sw * sum_d h_d**2 + Sb`, so a
weight prior that is negligible against the bias prior leaves a head that can
learn a global noise level and nothing else. `logit_variance_weight_share`
splits the prior between the two at the measured feature energy;
`--variance-weight-share 0` reproduces the homoscedastic head, which recovers
the mean noise level correctly and its input dependence barely at all.

Results are in
[the logit TAGI-V report](../../runs/last_layer/logit_tagiv/cifar10/REPORT.md).

The finalized protocol is pre-registered in [PROTOCOL.md](PROTOCOL.md).
The claim it supports is that TAGI-V learns a well-ordered input-dependent
surrogate for augmentation-induced logit variability and improves
in-distribution calibration at unchanged accuracy across K = 10 and K = 100,
with absolute variance dispersion dataset-dependent.

### Separating mean fitting from variance fitting

    python experiments/last_layer/run_logit_tagiv.py channels --dataset cifar100
    python experiments/last_layer/run_logit_tagiv.py variance-study --dataset cifar100
    python experiments/last_layer/run_logit_tagiv.py variance-study-report --dataset cifar100

A cold-started TAGI-V head learns its mean and its variance from the same first
step, so the early residual on a high-magnitude target is available to be stored
as noise, and the inflated variance then throttles that channel's own mean
update through the `1 / (v_Z + mu_S)` gain. `variance-study` compares the cold
start against two mean-first fits: `warmup_agvi` fits the mean under a fixed
observation variance, resets the variance head, and runs the same
single-observation AGVI with the mean frozen; `warmup_replicate` instead drives
the variance from the unbiased sample variance across augmentation repeats,
whose observation of the noise never passes through the mean head. For `M`
replicates that update uses `var(R | S) = 2 S^2 / (M - 1)`, giving
`v_R = (1 + c_M^2) v_S + c_M^2 mu_S^2` and a posterior that always contracts.

### Recommended formulation

    python experiments/last_layer/run_logit_tagiv.py power-diagnostic --dataset cifar100
    python experiments/last_layer/run_logit_tagiv.py shrinkage-study --dataset cifar100
    python experiments/last_layer/run_logit_tagiv.py shrinkage-study-report --dataset cifar100

Use `warmup_replicate` with persistent weight shrinkage. The mean-first fit
removes the contamination; the shrinkage removes what remains, which is not
contamination at all but excessive across-input spread in `G`. The exponential
is not the problem, it only exposes an over-dispersed log-variance: on CIFAR-100
the head's p10 is exact, its median 1.8x high and its p99 an order of magnitude
high, so its arithmetic mean is set almost entirely by a tail the real noise
does not have.

`power-diagnostic` establishes that causally without retraining. Compressing
`a` toward a fixed anchor is strictly increasing, so it cannot change any
ranking, and any improvement it produces is attributable to the level and the
predictive integration alone. It also shows why a global compression is not the
fix: anchored on one quantile it conflates level with spread, and it damages the
off-diagonal channels a cold start already fits.

Shrinkage separates the two, because the bias carries the level and the weights
carry the spread. Its rate belongs in the hundreds: the per-batch factor is
`1 / (1 + lambda_g * Sw)` with `Sw` of order `1e-3`, so a rate of order one is
three orders of magnitude too small to compete with the update pushing the
weights back out. Select it on held-out replicated logits by quantile agreement,
not by a squared error in log variance, which is dominated by the 99
off-diagonal channels and selects no shrinkage at all.

The formulation transfers across `K`. Rerunning the finalized protocol on
CIFAR-10 with the pre-registered dimensionless grid
`kappa_0 = lambda_g * v_Wg,0` selects `kappa_0 = 0`: CIFAR-10's variance head is
not over-dispersed, so the criterion correctly applies no shrinkage. That is the
protocol working, not a null result, and it is why the rate must never be
transferred raw. The same `kappa_0 = 0.164` that gives `lambda_g = 200` on
CIFAR-100 would be `lambda_g = 51` here, because the feature energy differs
(34.5 against 136) and with it the weight prior.

The two datasets miss in opposite directions, which bounds what shrinkage can
do. CIFAR-100's head is over-dispersed, `sd(log a) = 1.46` against an implied
`sd(log S) = 0.71`, and shrinkage fixes it. CIFAR-10's is under-dispersed,
`0.42` against `1.20`: the ordering is good, rank correlation `0.66`, but the
dynamic range is compressed, so the observed-class p99 is ten times too small.
Shrinkage only removes spread, so nothing in the current toolkit addresses that
direction.

The frozen formulation does not transfer to distribution shift. Rerunning the
selected head on CIFAR-100-C and SVHN with its existing `(T, alpha)` and no
further tuning fixes the level of the predicted class channel, from 13.8 to
0.48, and leaves its direction unchanged: it still falls with severity, 0.48
clean to 0.16 at severity 5, so its detection AUROC stays near 0.21. No
aggregation of the learned variance reaches a useful detector, and the
predictive entropy that does work, 0.74 at severity 5, sits below every
post-hoc score already available in the teacher's logits. The variance head
only ever saw clean-feature support, so this is a separate problem: modelling
shift sensitivity needs training perturbations that span it, not a better
in-distribution variance fit.

`channels` scores the aggregations of `a(x)` separately, because the predicted
class channel carries the largest target and is the one most exposed to storing
mean error as noise. It is the least informative channel on both datasets, so
`rest`, the median, or a trimmed mean is the aggregation to report; on CIFAR-10
dropping it lifts SVHN AUROC from 0.931 to 0.957.

Read the learned variance's median next to its mean. It is log-normal by
construction, so a linear score with too much spread inflates the mean through
its upper tail while the bulk stays close to the target.

---

## Hierarchical probit

`run_hrc_calibration.py` compares two models and reports everything else as an
ablation against them.

The main comparison is TAGI's hierarchical-probit head on the full K-leaf tree
— unit latent probit noise, deterministic prior branch offsets, no `sigma_v`,
no fitted scale, no normalizer, no post-hoc calibration — against a
conventional linear head trained with backprop cross-entropy and no
temperature. Both are scored with ordinary categorical cross-entropy in nats.

    python experiments/last_layer/run_hrc_calibration.py select --num-classes 10
    python experiments/last_layer/run_hrc_calibration.py run --num-classes 10 \
        --permutation-seeds 0 1 2 3
    python experiments/last_layer/run_hrc_calibration.py run --num-classes 8 \
        --permutation-seeds 0 1 2 3
    python experiments/last_layer/run_hrc_calibration.py report

It reads the same cached ResNet-18 features as `run_study.py`, so no backbone
training is needed. `select` picks the prior gain on validation NLL, plus
`sigma_v` for the Gaussian ablation, which is the only arm that has one. `run`
retrains under several class-to-leaf assignments and reads the test split once
per record. `report` prints the main comparison first and the ablations below
it; the `role` column carries the same split in the CSV.

The ablations vary one thing each. `probit_padded` keeps the probit update but
moves it to the padded 2^ceil(log2 K)-leaf tree, so its leaf products need a
categorical normalizer. `gaussian_padded` is the classical head: a Gaussian
+/-1 pseudo-target with a tuned `sigma_v`, reported at cuTAGI's `alpha = 3`.
Each trained arm is then read out five ways: at the model's own scale
(`unit_scale`), with the normalizer forced off (`unnormalized`) and on
(`normalized`), at a scale fitted on validation NLL (`fitted_scale`), and at
the mean of a Laplace posterior over `log tau` (`laplace_scale`). The softmax
baseline is read with and without a fitted temperature.

`test_simplex_deviation` is the column to read first. It is
`max_n |sum_c p_nc - 1|`, so any row where it is not zero is not scoring a
distribution and its NLL, Brier, and ECE mean nothing — those rows report the
diagnostic columns instead, with ECE blank. The full tree is zero there with
the normalizer switched off, which is the invariant that lets the proposed
model do without one.

The four class-to-leaf assignments matter because a hierarchical head is not
permutation invariant; `test_nll_std` in the report is the spread across them,
not a seed effect. The `--num-classes 8` run is the control: at K = 8 the
padded tree is already full and every branch offset is zero, so it separates
the normalization defect from every other difference between the arms.

## Hierarchical probit calibration with a positive uncertain gain

Adds a Gaussian log-gain to a frozen hierarchical head, following
[HSM_calibration_Goulet_Nguyen_Florensa_annotated.tex](HSM_calibration_Goulet_Nguyen_Florensa_annotated.tex).
Implementation notes, the two channels, and the notation mapping into
`hrc_probit` are in [HSM_CALIBRATION_THEORY.md](HSM_CALIBRATION_THEORY.md).

    python experiments/last_layer/run_hsm_calibration.py run
    python experiments/last_layer/run_hsm_calibration.py report
    python experiments/last_layer/plot_hsm_calibration.py

The gain is `G_r = exp(L_r)` with `L_r ~ N(lambda_r, q_r)`, so it is strictly
positive and it keeps its uncertainty into prediction. It multiplies the state
against the **fixed** noise of the branch channel the head was trained with,

    R_n = G_{r(n)} Z_n + sigma_v o_n + eps_n,    eps_n ~ N(0, sigma_v^2),

so `lambda = 0` with `q = 0` is the frozen head itself, which is the
`uncalibrated` reference, and `q = 0` alone is ordinary probit temperature
scaling, which is the `hsm_global_point` ablation. `sigma_v` is never inferred;
it is carried on the belief so that a gain fitted against one channel cannot be
read out against another.

Both hierarchical heads are supported and each brings its own channel.

    --head hrc_probit                  sigma_v = 1, the structural latent unit
    --head hrc --sigma-v S             sigma_v = S, the noise the head trained
                                       with; this is the paper's formulation
                                       instantiated on the base HRC head
    --head hrc --latent-scale convention   sigma_v = 1/3, cuTAGI's hard-coded
                                       alpha = 3 readout, whatever S was

For the base HRC head the uncalibrated arm is then
`obs_to_class_probs(ma, Sa, hrc, alpha = 1 / sigma_v)` to double precision. The
two base HRC channels are different models, not two parameterizations of one,
because cuTAGI reads a head trained at `sigma_v = 0.3` through `alpha = 3`;
calibration largely repairs the difference, since the gain absorbs a constant
scale, and `runs/last_layer/hsm_calibration/cifar10_base_hrc{,_sigmav}` are the
two sweeps to compare. Within one channel, changing `sigma_v` is exactly the
shift `lambda -> lambda - log sigma_v`: re-running the convention sweep after
the channel was introduced reproduced the earlier records to 1e-14 with every
log-gain shifted by exactly `log 3`.

One head is trained per class-to-leaf assignment on the 40,000-example train
split. Everything else is post-hoc on the frozen network, so every arm reads
the same forward summaries and the comparison isolates the calibration. Two
axes are crossed. The **sharing** axis is one global gain, one per tree level,
or one per internal node. The **calibration size** axis subsamples the 10,000
held-out examples down to 100, which is where the gain uncertainty stops being
a rounding error and where deep per-node gains become weakly identified. Three
subsamples are drawn at every size below the full split, and the visit count of
every gain group is recorded with its posterior.

The ablations vary one thing each. `*_point` reads the fitted `lambda` as a
point value, which drops `q` and leaves ordinary probit temperature scaling;
the gap to the matching main arm is what the gain uncertainty buys.
`hsm_global_laplace` replaces the normalized grid by the mode and its inverse
negative curvature. `hsm_global_adf` and `hsm_global_adf_reversed` run the
sequential assumed-density update over the same visits in two stream orders,
which is the ordering check the approximation requires. `hrc_fitted_scale` is
the existing NLL-fitted latent scale, and it agrees with `hsm_global_point`
because on a global gain the branch likelihood of the batch fit is the
categorical likelihood. `softmax_temperature` is temperature scaling on a
backprop cross-entropy head, fitted on the same subsample.

`test_epistemic_share` is `v_Q / [mu_Q (1 - mu_Q)]` at the predicted leaf: the
fraction of the class indicator's variance that is dispersion of the latent
probability rather than residual label randomness. It is zero for every
deterministic arm by construction. `test_simplex_deviation` stays at machine
zero for all of them, which is the invariant that lets the proper tree do
without a normalizer at any gain.

`plot_hsm_calibration.py` produces the two-panel summary in
[report_assets/hsm_calibration_cifar10.pdf](report_assets/hsm_calibration_cifar10.pdf).
The right panel plots the posterior deviation of every gain group against its
visits. The plateau at `sqrt(q) ~ 1.3` below a thousand visits is not a fitting
artifact: a node whose visits all agree has a monotone likelihood in `l`, so
its posterior is the prior tilted rather than a peak, and `q` leaves the prior
scale only once enough disagreeing visits create an interior mode.
