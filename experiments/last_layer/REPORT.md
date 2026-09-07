# Frozen-feature TAGI last-layer classification study

## Executive summary

This study asks whether analytic Bayesian last-layer inference can improve uncertainty quantification while preserving the accuracy of a strong deterministic classifier. Separate CIFAR ResNet-18 backbones were trained on deterministic 40,000/10,000 train/validation splits for CIFAR-10 and CIFAR-100 and then frozen. Six shared TAGI last-layer heads were trained from each backbone's 512-dimensional features; CIFAR-100 additionally evaluates unit-probit HRC. The original PyTorch softmax classifier is the baseline. No method uses temperature scaling or any other post-hoc calibration.

On CIFAR-10, the answer is positive for calibration and promising but mixed for OOD detection. Clean accuracy stays within 0.20 percentage points of the 95.00% baseline. Clean ECE improves from 2.78% to 0.53% with HRC, while clean NLL improves from 0.194 to 0.173 with categorical TAGI-V. On CIFAR-10-C, hierarchical TAGI-V reduces macro ECE from 19.68% to 12.39% and NLL from 1.358 to 1.090 at essentially unchanged accuracy. Entropy-based SVHN detection improves from 91.18% AUROC for softmax to 92.48% for HRC; the corresponding CIFAR-10-C improvement is smaller, 70.76% to 71.33%.

CIFAR-100 is more qualified. Categorical TAGI-V gives the strongest overall Bayesian tradeoff: 76.61% clean accuracy versus 76.67% for softmax, and on CIFAR-100-C it improves NLL from 2.452 to 2.411, Brier from 0.691 to 0.678, and ECE from 12.86% to 10.29% with a 0.28-point accuracy decrease. It also slightly improves entropy AUROC on SVHN (86.05% versus 85.49%) and CIFAR-100-C (69.72% versus 69.39%). The hierarchical heads lose 3.3–4.7 clean-accuracy points, so their lower shifted ECE cannot be read as a calibration-only improvement.

Native epistemic uncertainty is the main unresolved issue. ReMax variance is useful for OOD detection, reaching about 90% AUROC on SVHN and 70% on CIFAR-10-C, while the native variance of the other heads is anti-correlated with OOD examples. Moreover, HRC, probit OVR, and hierarchical TAGI-V continue contracting through epoch 200 even after accuracy and calibration have stabilized. Posterior contraction therefore cannot by itself be treated as evidence that the resulting scalar is a valid OOD score.

The learned-noise result is also asymmetric: hierarchical TAGI-V increases its aleatoric variance and becomes aleatoric-dominated, whereas dense categorical TAGI-V leaves its nominally trainable variance channel at initialization. The latter should be treated as an inactive-channel result until its update parameterization is corrected or shown to learn under another controlled configuration.

## Scope and experimental protocol

- Dataset split: deterministic stratified 40,000 train / 10,000 validation from each official CIFAR training set; each official 10,000-example test set is held out until final evaluation.
- Backbones: separate CIFAR ResNet-18 models trained for 200 epochs with SGD, momentum 0.9, weight decay $5\times10^{-4}$, cosine learning-rate schedule, and initial learning rate 0.1. CIFAR-10 reached 95.44% best validation accuracy at epoch 195 and 95.39% final accuracy. CIFAR-100 reached its best and final validation accuracy of 76.36% at epoch 200.
- Last-layer scope: the backbone and all 512-dimensional feature vectors are frozen. Only the feature-to-output layer is changed.
- Search: fixed-noise heads screen $\sigma_v$, weight gain, and bias gain. TAGI-V heads instead search the prior for learned observation variance. Hyperparameters are selected by minimum validation NLL among configurations within one percentage point of the best validation accuracy, breaking ties with Brier score and ECE.
- Confirmation: five last-layer seeds, 200 epochs, with validation-selected and epoch-200 checkpoints both retained.
- OOD: SVHN is semantic OOD. CIFAR-10-C and CIFAR-100-C use the 15 canonical corruptions at severities 1–5 and are better described as covariate shift; OOD metrics compare each clean CIFAR dataset against its corresponding corrupted set.
- Uncertainty scores: predictive entropy, $1-\max_k p_k$, and the mean native predictive epistemic variance across outputs. AUROC/AUPR-OOD are higher-is-better; FPR95 is lower-is-better.
- Replication boundary: within each dataset, the five seeds share one trained backbone, so intervals measure last-layer variability, not backbone-training variability. Reported $\pm$ values are sample standard deviations; each baseline is one fixed run.

### Selected configurations

| Method | Observation variance | Gain W/b | Selected epochs (5 seeds) |
| --- | --- | --- | --- |
| Probit OVR | $\sigma_v=0.3$ | 0.01/0.1 | 10 |
| ReMax moment matching | $\sigma_v=1.0$ | 3.0/3.0 | 1 |
| ReMax Laplace (diagonal) | $\sigma_v=3.0$ | 1.0/3.0 | 10, 20, 10, 20, 20 |
| Hierarchical classifier (HRC) | $\sigma_v=0.3$ | 0.3/0.3 | 200 |
| Categorical TAGI-V | learned; $\bar v^2_0=0.001$, $S_{b,v}=0.1$ | 0.1/0.1 | 75 |
| Hierarchical TAGI-V | learned; $\bar v^2_0=0.001$, $S_{b,v}=1.0$ | 0.1/0.1 | 20, 200, 200, 20, 200 |

The per-seed validation selector chose epoch 10 for probit OVR, epoch 1 for moment-matched ReMax, epochs 10–20 for diagonal Laplace ReMax, epoch 200 for fixed HRC, epoch 75 for categorical TAGI-V, and epochs 20 or 200 for hierarchical TAGI-V.

## Methods and formulations

### Common TAGI last-layer model

For frozen feature vector $h\in\mathbb{R}^D$, each weight and bias is represented by an independent Gaussian posterior,

$$W_{dk}\sim\mathcal N(m_{W,dk},S_{W,dk}),\qquad b_k\sim\mathcal N(m_{b,k},S_{b,k}).$$

Because $h$ is deterministic, a linear output has diagonal moments

$$m_{z,k}=\sum_d h_d m_{W,dk}+m_{b,k},\qquad S_{z,k}=\sum_d h_d^2 S_{W,dk}+S_{b,k}.$$

The regression-style Gaussian observation model is

$$y_k=z_k+v_k,\qquad v_k\sim\mathcal N(0,\sigma_v^2),$$

which yields the output innovations

$$\Delta m_{z,k}=\frac{y_k-m_{z,k}}{S_{z,k}+\sigma_v^2},\qquad
\Delta S_{z,k}=-\frac{1}{S_{z,k}+\sigma_v^2}.$$

TAGI propagates these two terms analytically through the last layer and performs capped Gaussian parameter updates. There is no gradient optimizer for these heads. Initial gains determine the prior parameter variances; $\sigma_v$ determines how strongly observations update that prior.

### Probit one-versus-rest

The true class receives target $+1$ and every other class receives $-1$. Each logit is updated with the Gaussian regression observation above. Its binary predictive probability is

$$\tilde p_k=\Phi\left(\frac{m_{z,k}}{\sqrt{S_{z,k}+\sigma_v^2}}\right),\qquad
p_k=\frac{\tilde p_k}{\sum_j\tilde p_j}.$$

This is an independent one-versus-rest construction followed by normalization; it is not the external NormCDF/Laplace method discussed elsewhere.

### ReMax probability-space observations

ReMax maps Gaussian logits to a simplex without exponentiation. Let $M_k=\max(0,Z_k)$ and

$$A_k=\frac{M_k}{\sum_j M_j}.$$

The class label is a one-hot vector in probability space and the same Gaussian observation update is applied after propagating moments through $A$. Two approximations were evaluated:

1. **Moment-matched ReMax:** truncated-Gaussian moments of $M_k$ are converted to log-normal moments; moments of $\log M_k-\log\sum_jM_j$ give $E[A_k]$, $\operatorname{Var}(A_k)$, and a diagonal statistical Jacobian $\operatorname{Cov}(A_k,Z_k)/S_{z,k}$.
2. **Laplace ReMax, diagonal:** fixed Gaussian quadrature evaluates the Laplace identities for $1/\sum_jM_j$ and its square. The implementation can form a full cross-class Jacobian, but this experiment deliberately uses only its diagonal.

### Fixed-noise hierarchical classification

For $K$ classes, each class is assigned a binary code of length $L=\lceil\log_2K\rceil$. CIFAR-10 uses four decisions and 11 unique tree nodes. A class $c$ has path nodes $j_{c,\ell}$ and signs $s_{c,\ell}\in\{-1,+1\}$. Only those four nodes receive an update for each training example. Node probabilities are

$$q_j=\Phi\left(\frac{m_{z,j}}{\sqrt{(1/3)^2+S_{z,j}}}\right),$$

and the class probability is the normalized path product

$$\tilde p(c)=\prod_{\ell=1}^L
q_{j_{c,\ell}}^{\mathbb 1[s_{c,\ell}=+1]}
(1-q_{j_{c,\ell}})^{\mathbb 1[s_{c,\ell}=-1]}.$$

This makes the observation update sparse and changes the multiclass geometry: examples update only their path rather than all $K$ outputs. CIFAR-100 paths contain seven decisions rather than four.

### Unit-probit HRC

The CIFAR-100 replication also evaluates an exact half-space probit update on the same sparse hierarchy. It fixes the structural link variance to one, accepts no $\sigma_v$, and predicts each positive branch with

$$q_j=\Phi\left(\frac{m_{z,j}}{\sqrt{1+S_{z,j}}}\right).$$

This separates the effect of a unit-probit observation model from the tuned fixed-noise HRC model. Its gains are searched, but its link scale is neither inferred nor used as a prediction-time temperature.

### Categorical TAGI-V

TAGI-V removes manual $\sigma_v$. The output width is $2K$, interleaving a logit latent and a positive learned variance latent for every class. Denote the epistemic logit variance by $S_{z,k}$ and the learned aleatoric mean by $\bar v_k^2$. The predictive probability uses a logistic–probit moment bridge,

$$a_k=\left(1+\frac{\pi}{8}(S_{z,k}+\bar v_k^2)\right)^{-1/2},\qquad
p_k=\operatorname{softmax}_k\left(a_k[m_{z,k}-\bar m_z]\right).$$

Training uses the integer categorical label directly. A centered $O(K)$ Gaussian approximation to the categorical innovation couples the classes through the simplex constraint. The variance channel is updated by matching the first two moments of the squared residual, and an even softplus activation keeps $\bar v_k^2$ positive. Thus $S_z$ is reported as epistemic uncertainty and $\bar v^2$ as learned aleatoric uncertainty; neither is fitted post hoc.

### Hierarchical TAGI-V

Hierarchical TAGI-V combines the sparse binary tree with a learned variance channel at every tree node, giving $2J$ interleaved outputs for $J$ nodes. For each selected node,

$$S_{\text{total},j}=S_{z,j}+\bar v_j^2,$$

$$\Delta m_{z,j}=\frac{s_j-m_{z,j}}{S_{\text{total},j}},\qquad
\Delta S_{z,j}=-\frac{1}{S_{\text{total},j}}.$$

The learned variance is again updated by squared-residual moment matching. At prediction time, $S_z+\bar v^2$ enters each node CDF before multiplying probabilities along the class path. This is the no-manual-noise hierarchical classifier proposed in this study.

## Clean CIFAR-10 classification and calibration

| Method | Accuracy ↑ | NLL ↓ | Brier ↓ | ECE ↓ | Adaptive ECE ↓ |
| --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 95.00 | 0.194 | 0.080 | 2.78 | 2.77 |
| Probit OVR | 94.95 ± 0.08 | 0.181 ± 0.001 | 0.077 ± 0.000 | 0.98 ± 0.03 | 1.52 ± 0.12 |
| ReMax moment matching | 94.97 ± 0.02 | 0.187 ± 0.000 | 0.077 ± 0.000 | 1.36 ± 0.02 | 1.54 ± 0.05 |
| ReMax Laplace (diagonal) | 94.93 ± 0.02 | 0.187 ± 0.002 | 0.078 ± 0.001 | 1.10 ± 0.20 | 1.57 ± 0.08 |
| Hierarchical classifier (HRC) | 94.83 ± 0.03 | 0.181 ± 0.000 | 0.078 ± 0.000 | 0.53 ± 0.04 | 0.69 ± 0.02 |
| Categorical TAGI-V | 94.99 ± 0.03 | 0.173 ± 0.000 | 0.077 ± 0.000 | 1.12 ± 0.02 | 1.48 ± 0.05 |
| Hierarchical TAGI-V | 94.80 ± 0.01 | 0.181 ± 0.001 | 0.078 ± 0.000 | 0.73 ± 0.10 | 0.66 ± 0.04 |

All heads preserve accuracy. The largest decrease from the 95.00% baseline is 0.20 percentage points. Categorical TAGI-V has the best NLL and Brier score, while fixed HRC has the best standard ECE. Hierarchical TAGI-V is close to fixed HRC without requiring a manually selected observation variance.

## SVHN semantic OOD detection

| Method | Entropy AUROC ↑ | Entropy AUPR-OOD ↑ | Entropy FPR95 ↓ | $1-\max p$ AUROC ↑ | Native epi. AUROC ↑ | Native epi. FPR95 ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 91.18 | 95.20 | 24.01 | 91.00 | — | — |
| Probit OVR | 90.39 ± 0.73 | 94.78 ± 0.31 | 26.08 ± 3.21 | 89.91 ± 0.76 | 12.32 ± 0.00 | 99.93 |
| ReMax moment matching | 88.15 ± 0.34 | 93.42 ± 0.14 | 32.02 ± 1.94 | 88.11 ± 0.24 | 90.21 ± 0.48 | 28.24 ± 0.72 |
| ReMax Laplace (diagonal) | 88.04 ± 0.98 | 93.85 ± 0.33 | 42.17 ± 12.02 | 87.98 ± 0.64 | 89.91 ± 1.15 | 28.91 ± 2.30 |
| Hierarchical classifier (HRC) | 92.48 ± 0.28 | 95.75 ± 0.20 | 20.61 ± 0.76 | 92.21 ± 0.27 | 23.74 ± 0.00 | 99.93 |
| Categorical TAGI-V | 91.53 ± 0.13 | 95.56 ± 0.07 | 24.17 ± 0.54 | 91.24 ± 0.12 | 11.86 ± 0.01 | 99.93 |
| Hierarchical TAGI-V | 92.24 ± 0.29 | 95.62 ± 0.21 | 21.52 ± 0.86 | 92.01 ± 0.26 | 16.48 ± 3.77 | 99.95 ± 0.01 |

Entropy and $1-\max p$ give the same ranking. HRC is strongest, improving entropy AUROC by about 1.30 percentage points and reducing FPR95 by about 3.40 points versus softmax. ReMax's native epistemic variance is qualitatively different: it reaches approximately 90% AUROC, while native variances from probit, HRC, and categorical TAGI-V are strongly anti-correlated with semantic OOD inputs. Consequently, predictive entropy is currently the reliable score for HRC/TAGI-V; native variance cannot yet be used indiscriminately.

![Clean CIFAR-10 and SVHN entropy histograms](report_assets/svhn_entropy_histograms.png)

Every panel uses the same entropy bins and axes. For learned heads, each example's entropy is averaged across the five validation-selected last-layer seeds before histogramming. HRC and hierarchical TAGI-V move more SVHN mass toward higher entropy while retaining a concentrated low-entropy clean distribution, explaining their stronger entropy AUROC. The overlap remains substantial, consistent with FPR95 values around 20–22% rather than near-perfect separation.

## CIFAR-10-C covariate shift

Each number below is a macro-average over 15 corruptions and five severities (75 equally sized conditions).

### Shifted classification and calibration

| Method | Accuracy ↑ | NLL ↓ | Brier ↓ | ECE ↓ | Adaptive ECE ↓ |
| --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 72.13 | 1.358 | 0.463 | 19.68 | 19.67 |
| Probit OVR | 72.18 ± 0.13 | 1.275 ± 0.017 | 0.444 ± 0.003 | 16.80 ± 0.18 | 16.91 ± 0.18 |
| ReMax moment matching | 72.30 ± 0.06 | 1.281 ± 0.006 | 0.437 ± 0.001 | 15.64 ± 0.15 | 15.83 ± 0.15 |
| ReMax Laplace (diagonal) | 72.23 ± 0.17 | 1.246 ± 0.031 | 0.438 ± 0.006 | 15.82 ± 1.04 | 15.93 ± 1.01 |
| Hierarchical classifier (HRC) | 71.89 ± 0.02 | 1.128 ± 0.003 | 0.423 ± 0.001 | 13.16 ± 0.06 | 13.16 ± 0.07 |
| Categorical TAGI-V | 72.15 ± 0.02 | 1.098 ± 0.002 | 0.434 ± 0.001 | 15.51 ± 0.04 | 15.53 ± 0.04 |
| Hierarchical TAGI-V | 71.92 ± 0.07 | 1.090 ± 0.008 | 0.419 ± 0.002 | 12.39 ± 0.19 | 12.40 ± 0.20 |

Accuracy is effectively tied, showing that the output head cannot repair corruption errors already embedded in frozen features. Calibration does improve materially. Hierarchical TAGI-V reduces ECE by 7.29 percentage points, NLL by 0.268, and Brier by 0.044 relative to softmax.

### Clean-versus-corrupted detection

| Method | Entropy AUROC ↑ | Entropy AUPR-OOD ↑ | Entropy FPR95 ↓ | $1-\max p$ AUROC ↑ | Native epi. AUROC ↑ | Native epi. FPR95 ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 70.76 | 69.46 | 80.55 | 70.67 | — | — |
| Probit OVR | 69.16 ± 0.35 | 68.26 ± 0.24 | 83.74 ± 1.44 | 69.17 ± 0.33 | 32.46 ± 0.00 | 98.48 ± 0.00 |
| ReMax moment matching | 69.59 ± 0.29 | 68.54 ± 0.19 | 82.64 ± 1.24 | 69.63 ± 0.25 | 70.49 ± 0.18 | 81.18 ± 0.78 |
| ReMax Laplace (diagonal) | 68.55 ± 1.27 | 68.11 ± 0.56 | 84.47 ± 3.64 | 68.64 ± 1.23 | 69.95 ± 0.65 | 82.46 ± 3.01 |
| Hierarchical classifier (HRC) | 71.33 ± 0.10 | 70.11 ± 0.08 | 78.04 ± 0.30 | 71.25 ± 0.10 | 36.88 ± 0.00 | 98.32 ± 0.00 |
| Categorical TAGI-V | 71.05 ± 0.05 | 69.83 ± 0.05 | 80.22 ± 0.22 | 70.93 ± 0.05 | 32.49 ± 0.01 | 98.47 ± 0.00 |
| Hierarchical TAGI-V | 70.99 ± 0.15 | 69.91 ± 0.12 | 78.77 ± 0.43 | 70.92 ± 0.14 | 34.74 ± 0.74 | 98.49 ± 0.04 |

The entropy-AUROC gains are modest: fixed HRC improves from 70.76% to 71.33%. ReMax is the only family whose native epistemic variance has the correct direction, at about 70% AUROC; the native scores of the other heads remain below 50%.

### Severity dependence

Corrupted accuracy (%):

| Method | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 |
| --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 86.89 | 80.36 | 73.52 | 65.88 | 53.99 |
| Probit OVR | 86.85 | 80.36 | 73.59 | 65.94 | 54.12 |
| ReMax moment matching | 86.93 | 80.48 | 73.71 | 66.07 | 54.28 |
| ReMax Laplace (diagonal) | 86.88 | 80.44 | 73.68 | 66.00 | 54.18 |
| Hierarchical classifier (HRC) | 86.67 | 80.10 | 73.32 | 65.52 | 53.86 |
| Categorical TAGI-V | 86.87 | 80.37 | 73.56 | 65.90 | 54.04 |
| Hierarchical TAGI-V | 86.67 | 80.11 | 73.33 | 65.58 | 53.89 |

ECE (%):

| Method | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 |
| --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 8.12 | 12.90 | 18.44 | 24.61 | 34.31 |
| Probit OVR | 5.71 | 10.21 | 15.55 | 21.57 | 30.98 |
| ReMax moment matching | 4.93 | 9.10 | 14.31 | 20.23 | 29.61 |
| ReMax Laplace (diagonal) | 5.16 | 9.34 | 14.50 | 20.38 | 29.71 |
| Hierarchical classifier (HRC) | 3.84 | 7.49 | 12.06 | 17.30 | 25.11 |
| Categorical TAGI-V | 5.28 | 9.33 | 14.35 | 19.92 | 28.65 |
| Hierarchical TAGI-V | 3.40 | 6.83 | 11.24 | 16.36 | 24.11 |

Clean-versus-corrupted entropy AUROC (%):

| Method | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 |
| --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 61.15 | 66.71 | 70.79 | 74.94 | 80.18 |
| Probit OVR | 60.04 | 65.28 | 69.07 | 73.11 | 78.29 |
| ReMax moment matching | 60.63 | 65.84 | 69.45 | 73.38 | 78.66 |
| ReMax Laplace (diagonal) | 59.91 | 64.89 | 68.32 | 72.19 | 77.44 |
| Hierarchical classifier (HRC) | 61.47 | 67.19 | 71.44 | 75.65 | 80.93 |
| Categorical TAGI-V | 61.36 | 67.01 | 71.12 | 75.27 | 80.47 |
| Hierarchical TAGI-V | 61.17 | 66.82 | 71.09 | 75.30 | 80.59 |

All heads degrade similarly in accuracy, from about 87% at severity 1 to 54% at severity 5. Hierarchical TAGI-V is consistently better calibrated, although its ECE still rises from 3.40% to 24.11%. Detection becomes easier as severity grows because corrupted representations move farther from the clean distribution.

The easiest corruption is brightness (about 93.5% accuracy averaged across severities); the hardest are Gaussian noise (about 37%), impulse noise (about 49.5%), and shot noise (about 51%).

## CIFAR-100 replication

The locked experiment was repeated on CIFAR-100 with the same deterministic stratified 40,000/10,000 train/validation split, official 10,000-example test set, frozen 512-dimensional ResNet-18 features, search rule, five last-layer confirmation seeds, and 200-epoch confirmation horizon. A separate CIFAR-100 ResNet-18 was trained from scratch for 200 epochs; its best and final validation accuracy was 76.36% at epoch 200. CIFAR-100-C contains the same 15 corruption families and five severities as CIFAR-10-C. The CIFAR-100 study additionally includes unit-probit HRC, whose latent probit scale is fixed to one and therefore has no searched observation-noise parameter.

The five-seed intervals below again measure last-layer variability under one shared backbone, not variability from retraining the backbone. Primary comparisons use each seed's validation-selected checkpoint; explicit epoch-200 tables show the effect of continued training.

### CIFAR-100 selected configurations

| Method | Observation variance | Gain W/b | Selected epochs (5 seeds) |
| --- | --- | --- | --- |
| Probit OVR | $\sigma_v=0.3$ | 0.1/0.3 | 200 |
| ReMax moment matching | $\sigma_v=0.3$ | 1.0/3.0 | 2, 3, 3, 3, 3 |
| ReMax Laplace (diagonal) | $\sigma_v=0.3$ | 1.0/0.3 | 2, 2, 3, 2, 2 |
| Hierarchical classifier (HRC) | $\sigma_v=0.1$ | 0.3/0.1 | 200 |
| Categorical TAGI-V | learned; $\bar v^2_0=0.001$, $S_{b,v}=0.1$ | 0.1/0.1 | 200 |
| Hierarchical TAGI-V | learned; $\bar v^2_0=0.01$, $S_{b,v}=0.1$ | 0.1/0.1 | 30, 30, 30, 30, 75 |
| Unit-probit HRC | unit probit scale | 3.0/1.0 | 1 |

### Clean CIFAR-100 classification and calibration

| Method | Accuracy ↑ | NLL ↓ | Brier ↓ | ECE ↓ | Adaptive ECE ↓ |
| --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 76.67 | 0.958 | 0.333 | 4.82 | 4.82 |
| Probit OVR | 76.08 ± 0.12 | 1.091 ± 0.004 | 0.348 ± 0.001 | 8.41 ± 0.11 | 8.41 ± 0.11 |
| ReMax moment matching | 76.56 ± 0.05 | 1.309 ± 0.033 | 0.403 ± 0.052 | 18.76 ± 10.97 | 18.73 ± 10.96 |
| ReMax Laplace (diagonal) | 76.04 ± 0.44 | 1.470 ± 0.077 | 0.400 ± 0.024 | 17.22 ± 8.74 | 17.18 ± 8.69 |
| Hierarchical classifier (HRC) | 72.65 ± 0.17 | 1.304 ± 0.002 | 0.394 ± 0.000 | 8.59 ± 0.20 | 8.71 ± 0.21 |
| Categorical TAGI-V | 76.61 ± 0.12 | 1.007 ± 0.002 | 0.343 ± 0.000 | 6.64 ± 0.13 | 6.68 ± 0.13 |
| Hierarchical TAGI-V | 73.36 ± 0.10 | 1.282 ± 0.003 | 0.413 ± 0.001 | 17.06 ± 0.28 | 17.06 ± 0.28 |
| Unit-probit HRC | 71.96 ± 0.13 | 1.288 ± 0.007 | 0.397 ± 0.001 | 5.33 ± 0.20 | 5.42 ± 0.21 |

The 100-class replication is materially harder than CIFAR-10. Categorical TAGI-V nearly matches softmax accuracy (76.61% versus 76.67%) but does not improve clean NLL or ECE. Probit OVR also retains most accuracy but is less calibrated. All hierarchical variants lose 3.3–4.7 clean-accuracy points; unit-probit HRC's 5.33% ECE is therefore paired with substantially worse accuracy and NLL.

### CIFAR-100 versus SVHN semantic OOD

| Method | Entropy AUROC ↑ | Entropy AUPR-OOD ↑ | Entropy FPR95 ↓ | $1-\max p$ AUROC ↑ | Native epi. AUROC ↑ | Native epi. FPR95 ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 85.49 | 91.53 | 43.94 | 83.78 | — | — |
| Probit OVR | 84.38 ± 0.37 | 91.02 ± 0.24 | 45.81 ± 1.31 | 82.58 ± 0.45 | 11.44 ± 0.00 | 99.74 |
| ReMax moment matching | 71.08 ± 1.03 | 83.99 ± 0.35 | 76.16 ± 3.95 | 73.83 ± 1.43 | 67.22 ± 8.60 | 83.23 ± 5.00 |
| ReMax Laplace (diagonal) | 71.91 ± 2.19 | 84.41 ± 0.98 | 75.38 ± 5.95 | 74.52 ± 0.91 | 66.19 ± 5.25 | 84.52 ± 7.28 |
| Hierarchical classifier (HRC) | 83.79 ± 0.34 | 91.16 ± 0.23 | 47.67 ± 0.62 | 82.38 ± 0.20 | 11.89 ± 0.00 | 99.71 |
| Categorical TAGI-V | 86.05 ± 0.63 | 92.02 ± 0.38 | 43.08 ± 1.34 | 84.23 ± 0.62 | 12.68 ± 0.00 | 99.62 |
| Hierarchical TAGI-V | 85.14 ± 0.83 | 91.91 ± 0.48 | 44.90 ± 1.23 | 83.44 ± 0.75 | 12.23 ± 0.05 | 99.66 ± 0.01 |
| Unit-probit HRC | 82.54 ± 0.35 | 90.44 ± 0.25 | 51.69 ± 0.61 | 80.63 ± 0.26 | 12.62 ± 0.01 | 99.62 ± 0.01 |

Categorical TAGI-V is the only head to improve on the softmax entropy AUROC, reaching 86.05% versus 85.49% and reducing FPR95 from 43.94% to 43.08%. The gain is modest relative to its seed variability. Native epistemic variance is anti-correlated for probit and hierarchical heads; ReMax is positively oriented but much weaker than entropy on CIFAR-100 SVHN.

### CIFAR-100-C shifted classification and calibration

Each result is a macro-average over the 75 corruption/severity conditions.

| Method | Accuracy ↑ | NLL ↓ | Brier ↓ | ECE ↓ | Adaptive ECE ↓ |
| --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 48.11 | 2.452 | 0.691 | 12.86 | 12.85 |
| Probit OVR | 47.51 ± 0.11 | 2.605 ± 0.015 | 0.688 ± 0.003 | 10.24 ± 0.21 | 10.25 ± 0.22 |
| ReMax moment matching | 48.11 ± 0.03 | 3.884 ± 0.413 | 0.720 ± 0.010 | 15.76 ± 2.83 | 15.79 ± 2.85 |
| ReMax Laplace (diagonal) | 47.84 ± 0.33 | 5.102 ± 0.515 | 0.723 ± 0.022 | 15.89 ± 1.90 | 15.92 ± 1.92 |
| Hierarchical classifier (HRC) | 43.84 ± 0.07 | 3.108 ± 0.013 | 0.716 ± 0.001 | 9.05 ± 0.08 | 9.08 ± 0.09 |
| Categorical TAGI-V | 47.83 ± 0.04 | 2.411 ± 0.006 | 0.678 ± 0.001 | 10.29 ± 0.10 | 10.28 ± 0.10 |
| Hierarchical TAGI-V | 44.39 ± 0.06 | 2.816 ± 0.004 | 0.716 ± 0.001 | 12.90 ± 0.08 | 12.90 ± 0.08 |
| Unit-probit HRC | 43.22 ± 0.10 | 3.204 ± 0.023 | 0.730 ± 0.002 | 9.33 ± 0.16 | 9.35 ± 0.16 |

Categorical TAGI-V provides the clearest shifted-data result: versus softmax it gives 47.83% rather than 48.11% accuracy while improving NLL, Brier, ECE, and adaptive ECE. Probit OVR also lowers ECE and Brier but worsens NLL. The hierarchical heads' still-lower ECE comes with 3.7–4.9 points less shifted accuracy and substantially worse NLL, so it is not an unqualified calibration win.

### CIFAR-100 clean-versus-corrupted detection

| Method | Entropy AUROC ↑ | Entropy AUPR-OOD ↑ | Entropy FPR95 ↓ | $1-\max p$ AUROC ↑ | Native epi. AUROC ↑ | Native epi. FPR95 ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 69.39 | 66.60 | 76.65 | 68.82 | — | — |
| Probit OVR | 66.12 ± 0.23 | 63.38 ± 0.19 | 83.45 ± 0.36 | 66.35 ± 0.17 | 34.99 ± 0.00 | 97.10 ± 0.00 |
| ReMax moment matching | 63.43 ± 1.03 | 61.12 ± 1.41 | 86.77 ± 0.63 | 64.08 ± 1.64 | 63.54 ± 4.38 | 87.42 ± 2.90 |
| ReMax Laplace (diagonal) | 62.21 ± 1.73 | 60.16 ± 1.65 | 88.46 ± 1.87 | 63.21 ± 2.02 | 63.64 ± 2.81 | 88.10 ± 2.35 |
| Hierarchical classifier (HRC) | 67.87 ± 0.09 | 64.87 ± 0.14 | 77.59 ± 0.15 | 68.08 ± 0.10 | 34.93 ± 0.00 | 97.11 |
| Categorical TAGI-V | 69.72 ± 0.07 | 66.87 ± 0.07 | 76.37 ± 0.13 | 69.24 ± 0.07 | 35.16 ± 0.00 | 97.04 ± 0.00 |
| Hierarchical TAGI-V | 67.82 ± 0.10 | 64.93 ± 0.13 | 78.41 ± 0.17 | 68.15 ± 0.13 | 34.92 ± 0.01 | 97.09 ± 0.00 |
| Unit-probit HRC | 67.37 ± 0.14 | 64.85 ± 0.18 | 80.07 ± 0.13 | 66.95 ± 0.13 | 35.15 ± 0.00 | 97.05 ± 0.00 |

Categorical TAGI-V again gives the only improvement over softmax entropy AUROC (69.72% versus 69.39%) and FPR95 (76.37% versus 76.65%); the differences are small. ReMax native variance remains correctly oriented but does not beat softmax entropy, while the other native epistemic scores are anti-correlated.

### CIFAR-100-C severity dependence

Corrupted accuracy (%):

| Method | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 |
| --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 62.94 | 54.91 | 48.49 | 41.93 | 32.28 |
| Probit OVR | 62.27 | 54.32 | 47.89 | 41.29 | 31.77 |
| ReMax moment matching | 62.87 | 54.93 | 48.52 | 41.96 | 32.27 |
| ReMax Laplace (diagonal) | 62.50 | 54.61 | 48.24 | 41.70 | 32.12 |
| Hierarchical classifier (HRC) | 58.37 | 50.26 | 43.85 | 37.70 | 28.99 |
| Categorical TAGI-V | 62.66 | 54.65 | 48.20 | 41.64 | 31.98 |
| Hierarchical TAGI-V | 59.06 | 50.85 | 44.45 | 38.20 | 29.41 |
| Unit-probit HRC | 57.58 | 49.56 | 43.30 | 37.21 | 28.46 |

ECE (%):

| Method | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 |
| --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 7.62 | 9.47 | 12.15 | 15.32 | 19.76 |
| Probit OVR | 7.08 | 7.86 | 9.41 | 11.27 | 15.56 |
| ReMax moment matching | 15.25 | 14.44 | 14.86 | 15.99 | 18.26 |
| ReMax Laplace (diagonal) | 14.60 | 14.20 | 15.14 | 16.52 | 19.00 |
| Hierarchical classifier (HRC) | 8.05 | 7.59 | 8.82 | 9.73 | 11.06 |
| Categorical TAGI-V | 8.06 | 8.21 | 9.79 | 11.74 | 13.66 |
| Hierarchical TAGI-V | 14.98 | 12.50 | 12.70 | 12.92 | 11.40 |
| Unit-probit HRC | 5.25 | 6.37 | 8.41 | 10.98 | 15.65 |

Clean-versus-corrupted entropy AUROC (%):

| Method | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 |
| --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 61.44 | 66.66 | 69.94 | 72.59 | 76.34 |
| Probit OVR | 59.44 | 63.95 | 66.61 | 68.68 | 71.91 |
| ReMax moment matching | 57.20 | 60.92 | 63.70 | 65.92 | 69.43 |
| ReMax Laplace (diagonal) | 56.50 | 59.94 | 62.46 | 64.47 | 67.71 |
| Hierarchical classifier (HRC) | 60.76 | 65.51 | 68.24 | 70.69 | 74.13 |
| Categorical TAGI-V | 61.64 | 66.93 | 70.27 | 72.99 | 76.76 |
| Hierarchical TAGI-V | 60.80 | 65.54 | 68.21 | 70.59 | 73.96 |
| Unit-probit HRC | 60.38 | 65.04 | 67.75 | 70.17 | 73.53 |

### CIFAR-100 checkpoint selection and uncertainty dynamics

Clean metrics at validation-selected and epoch-200 checkpoints (accuracy and ECE in %):

| Method | Selected acc. | Epoch-200 acc. | Selected NLL | Epoch-200 NLL | Selected ECE | Epoch-200 ECE |
| --- | --- | --- | --- | --- | --- | --- |
| Probit OVR | 76.08 | 76.08 | 1.091 | 1.091 | 8.41 | 8.41 |
| ReMax moment matching | 76.56 | 74.74 | 1.309 | 2.107 | 18.76 | 14.05 |
| ReMax Laplace (diagonal) | 76.04 | 75.67 | 1.470 | 2.242 | 17.22 | 12.03 |
| Hierarchical classifier (HRC) | 72.65 | 72.65 | 1.304 | 1.304 | 8.59 | 8.59 |
| Categorical TAGI-V | 76.61 | 76.61 | 1.007 | 1.007 | 6.64 | 6.64 |
| Hierarchical TAGI-V | 73.36 | 73.56 | 1.282 | 1.284 | 17.06 | 17.70 |
| Unit-probit HRC | 71.96 | 70.78 | 1.288 | 1.887 | 5.33 | 9.19 |

CIFAR-100-C macro metrics at validation-selected and epoch-200 checkpoints (accuracy and ECE in %):

| Method | Selected acc. | Epoch-200 acc. | Selected NLL | Epoch-200 NLL | Selected ECE | Epoch-200 ECE |
| --- | --- | --- | --- | --- | --- | --- |
| Probit OVR | 47.51 | 47.51 | 2.605 | 2.605 | 10.24 | 10.24 |
| ReMax moment matching | 48.11 | 45.06 | 3.884 | 4.834 | 15.76 | 31.54 |
| ReMax Laplace (diagonal) | 47.84 | 46.57 | 5.102 | 4.946 | 15.89 | 23.61 |
| Hierarchical classifier (HRC) | 43.84 | 43.84 | 3.108 | 3.108 | 9.05 | 9.05 |
| Categorical TAGI-V | 47.83 | 47.83 | 2.411 | 2.411 | 10.29 | 10.29 |
| Hierarchical TAGI-V | 44.39 | 44.54 | 2.816 | 2.803 | 12.90 | 13.29 |
| Unit-probit HRC | 43.22 | 41.86 | 3.204 | 5.252 | 9.33 | 24.60 |

Validation selection is crucial for both ReMax variants and unit-probit HRC. For example, continuing unit-probit HRC from epoch 1 to epoch 200 lowers clean accuracy from 71.96% to 70.78% and raises CIFAR-100-C NLL from 3.204 to 5.252. Categorical TAGI-V, probit OVR, and fixed HRC select epoch 200 and therefore have identical selected/final rows.

Posterior-contraction diagnostics over the 200-epoch confirmation run:

| Method | Required epoch, mean (range) | $U_0$ | $U_{200}$ | $U_{200}/U_0$ | Epoch 170–200 |
| --- | --- | --- | --- | --- | --- |
| Probit OVR | 6.8 (6–7) | 2.250e-03 | 4.496e-06 | 0.002 | still shrinking |
| ReMax moment matching | not reached | 1.255e-04 | 2.600e-03 | 20.830 | rebound or unstable |
| ReMax Laplace (diagonal) | not reached | 1.245e-04 | 1.797e-04 | 1.436 | plateau |
| Hierarchical classifier (HRC) | 11.8 (10–13) | 1.869e-02 | 1.180e-04 | 0.006 | still shrinking |
| Categorical TAGI-V | 167.0 (167–167) | 2.094e-03 | 1.984e-03 | 0.947 | plateau |
| Hierarchical TAGI-V | 11.2 (9–15) | 2.094e-03 | 1.107e-04 | 0.053 | still shrinking |
| Unit-probit HRC | not reached | 1.869e+00 | 1.496e+00 | 0.801 | plateau |

TAGI-V aleatoric-channel diagnostics:

| Method | $A_0$ | $A_{selected}$ | $A_{200}$ | $A_{200}/A_0$ | Epistemic fraction |
| --- | --- | --- | --- | --- | --- |
| Categorical TAGI-V | 1.050e-03 | 1.050e-03 | 1.050e-03 | 1.000 | 64.45 → 63.23 |
| Hierarchical TAGI-V | 1.049e-02 | 1.489e-02 | 1.820e-02 | 1.734 | 16.28 → 0.97 |

### Complete CIFAR-100 metric appendix

Clean secondary classification metrics (%):

| Method | Top-5 acc. ↑ | Mean conf. | Classwise ECE ↓ | AURC ↓ | Risk@80 ↓ | Risk@90 ↓ | Risk@95 ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 93.62 | 78.40 | 0.20 | 6.62 | 13.16 | 17.88 | 20.52 |
| Probit OVR | 90.99 ± 0.10 | 67.67 ± 0.07 | 0.27 ± 0.00 | 7.25 ± 0.03 | 13.58 ± 0.06 | 18.52 ± 0.11 | 21.15 ± 0.13 |
| ReMax moment matching | 92.79 ± 0.18 | 57.83 ± 10.94 | 0.43 ± 0.17 | 7.18 ± 0.51 | 14.03 ± 0.45 | 18.70 ± 0.30 | 20.99 ± 0.17 |
| ReMax Laplace (diagonal) | 92.28 ± 0.52 | 58.98 ± 8.83 | 0.41 ± 0.10 | 7.91 ± 0.97 | 14.64 ± 0.77 | 19.11 ± 0.54 | 21.49 ± 0.48 |
| Hierarchical classifier (HRC) | 87.16 ± 0.10 | 64.07 ± 0.05 | 0.26 ± 0.00 | 8.38 ± 0.04 | 16.54 ± 0.16 | 21.63 ± 0.20 | 24.42 ± 0.18 |
| Categorical TAGI-V | 93.41 ± 0.06 | 71.01 ± 0.01 | 0.24 ± 0.00 | 6.79 ± 0.03 | 13.28 ± 0.16 | 17.99 ± 0.09 | 20.59 ± 0.08 |
| Hierarchical TAGI-V | 88.71 ± 0.14 | 56.30 ± 0.21 | 0.40 ± 0.01 | 8.30 ± 0.04 | 16.11 ± 0.09 | 21.00 ± 0.12 | 23.71 ± 0.15 |
| Unit-probit HRC | 87.90 ± 0.09 | 66.63 ± 0.18 | 0.24 ± 0.00 | 9.13 ± 0.06 | 17.59 ± 0.17 | 22.51 ± 0.12 | 25.19 ± 0.09 |

CIFAR-100-C secondary shifted-classification metrics (%):

| Method | Top-5 acc. ↑ | Mean conf. | Classwise ECE ↓ | AURC ↓ | Risk@80 ↓ | Risk@90 ↓ | Risk@95 ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 70.48 | 58.89 | 0.51 | 32.85 | 44.61 | 48.23 | 50.04 |
| Probit OVR | 67.64 ± 0.09 | 52.20 ± 0.19 | 0.52 ± 0.00 | 33.82 ± 0.17 | 45.28 ± 0.14 | 48.89 ± 0.13 | 50.68 ± 0.12 |
| ReMax moment matching | 69.84 ± 0.19 | 49.02 ± 10.48 | 0.75 ± 0.02 | 34.89 ± 1.58 | 45.22 ± 0.47 | 48.54 ± 0.23 | 50.18 ± 0.12 |
| ReMax Laplace (diagonal) | 69.44 ± 0.43 | 50.68 ± 9.60 | 0.74 ± 0.01 | 35.68 ± 2.07 | 45.61 ± 0.66 | 48.86 ± 0.46 | 50.47 ± 0.37 |
| Hierarchical classifier (HRC) | 60.93 ± 0.04 | 44.50 ± 0.09 | 0.52 ± 0.00 | 35.87 ± 0.08 | 49.02 ± 0.05 | 52.64 ± 0.06 | 54.41 ± 0.06 |
| Categorical TAGI-V | 69.93 ± 0.05 | 48.85 ± 0.09 | 0.48 ± 0.00 | 33.09 ± 0.07 | 44.97 ± 0.05 | 48.54 ± 0.04 | 50.34 ± 0.04 |
| Hierarchical TAGI-V | 62.73 ± 0.03 | 37.87 ± 0.32 | 0.55 ± 0.00 | 35.80 ± 0.10 | 48.53 ± 0.06 | 52.09 ± 0.06 | 53.84 ± 0.06 |
| Unit-probit HRC | 61.93 ± 0.16 | 49.77 ± 0.20 | 0.56 ± 0.01 | 37.20 ± 0.14 | 49.83 ± 0.11 | 53.34 ± 0.12 | 55.04 ± 0.10 |

Complete SVHN OOD metrics (%):

| Method | Score | AUROC ↑ | AUPR-OOD ↑ | AUPR-ID ↑ | FPR95 ↓ |
| --- | --- | --- | --- | --- | --- |
| PyTorch softmax | Entropy | 85.49 | 91.53 | 78.24 | 43.94 |
| PyTorch softmax | $1-\max p$ | 83.78 | 90.61 | 76.33 | 45.33 |
| Probit OVR | Entropy | 84.38 ± 0.37 | 91.02 ± 0.24 | 76.08 ± 0.70 | 45.81 ± 1.31 |
| Probit OVR | $1-\max p$ | 82.58 ± 0.45 | 90.05 ± 0.28 | 73.91 ± 0.78 | 48.79 ± 1.26 |
| Probit OVR | Native epistemic | 11.44 ± 0.00 | 53.11 ± 0.00 | 16.07 ± 0.00 | 99.74 |
| ReMax moment matching | Entropy | 71.08 ± 1.03 | 83.99 ± 0.35 | 51.41 ± 2.47 | 76.16 ± 3.95 |
| ReMax moment matching | $1-\max p$ | 73.83 ± 1.43 | 85.11 ± 0.81 | 57.01 ± 3.06 | 69.20 ± 2.77 |
| ReMax moment matching | Native epistemic | 67.22 ± 8.60 | 82.26 ± 4.58 | 45.45 ± 7.78 | 83.23 ± 5.00 |
| ReMax Laplace (diagonal) | Entropy | 71.91 ± 2.19 | 84.41 ± 0.98 | 52.35 ± 4.40 | 75.38 ± 5.95 |
| ReMax Laplace (diagonal) | $1-\max p$ | 74.52 ± 0.91 | 85.47 ± 0.54 | 57.85 ± 2.52 | 67.94 ± 2.63 |
| ReMax Laplace (diagonal) | Native epistemic | 66.19 ± 5.25 | 81.49 ± 2.14 | 43.39 ± 8.06 | 84.52 ± 7.28 |
| Hierarchical classifier (HRC) | Entropy | 83.79 ± 0.34 | 91.16 ± 0.23 | 75.26 ± 0.42 | 47.67 ± 0.62 |
| Hierarchical classifier (HRC) | $1-\max p$ | 82.38 ± 0.20 | 90.27 ± 0.15 | 73.46 ± 0.27 | 50.30 ± 0.38 |
| Hierarchical classifier (HRC) | Native epistemic | 11.89 ± 0.00 | 53.23 ± 0.00 | 16.12 ± 0.00 | 99.71 |
| Categorical TAGI-V | Entropy | 86.05 ± 0.63 | 92.02 ± 0.38 | 78.65 ± 0.87 | 43.08 ± 1.34 |
| Categorical TAGI-V | $1-\max p$ | 84.23 ± 0.62 | 90.92 ± 0.39 | 76.57 ± 0.83 | 45.69 ± 1.14 |
| Categorical TAGI-V | Native epistemic | 12.68 ± 0.00 | 53.44 ± 0.00 | 16.22 ± 0.00 | 99.62 |
| Hierarchical TAGI-V | Entropy | 85.14 ± 0.83 | 91.91 ± 0.48 | 77.08 ± 1.04 | 44.90 ± 1.23 |
| Hierarchical TAGI-V | $1-\max p$ | 83.44 ± 0.75 | 90.95 ± 0.41 | 74.63 ± 0.97 | 49.03 ± 1.09 |
| Hierarchical TAGI-V | Native epistemic | 12.23 ± 0.05 | 53.32 ± 0.01 | 16.16 ± 0.01 | 99.66 ± 0.01 |
| Unit-probit HRC | Entropy | 82.54 ± 0.35 | 90.44 ± 0.25 | 73.17 ± 0.47 | 51.69 ± 0.61 |
| Unit-probit HRC | $1-\max p$ | 80.63 ± 0.26 | 89.38 ± 0.19 | 70.65 ± 0.42 | 54.70 ± 0.51 |
| Unit-probit HRC | Native epistemic | 12.62 ± 0.01 | 53.42 ± 0.00 | 16.21 ± 0.00 | 99.62 ± 0.01 |

Complete CIFAR-100-C clean-versus-corrupted metrics (%):

| Method | Score | AUROC ↑ | AUPR-OOD ↑ | AUPR-ID ↑ | FPR95 ↓ |
| --- | --- | --- | --- | --- | --- |
| PyTorch softmax | Entropy | 69.39 | 66.60 | 69.71 | 76.65 |
| PyTorch softmax | $1-\max p$ | 68.82 | 66.02 | 69.37 | 76.70 |
| Probit OVR | Entropy | 66.12 ± 0.23 | 63.38 ± 0.19 | 65.48 ± 0.26 | 83.45 ± 0.36 |
| Probit OVR | $1-\max p$ | 66.35 ± 0.17 | 63.78 ± 0.12 | 65.91 ± 0.23 | 82.57 ± 0.40 |
| Probit OVR | Native epistemic | 34.99 ± 0.00 | 40.38 ± 0.00 | 41.68 ± 0.00 | 97.10 ± 0.00 |
| ReMax moment matching | Entropy | 63.43 ± 1.03 | 61.12 ± 1.41 | 62.12 ± 0.78 | 86.77 ± 0.63 |
| ReMax moment matching | $1-\max p$ | 64.08 ± 1.64 | 61.21 ± 1.63 | 63.46 ± 1.70 | 84.63 ± 1.86 |
| ReMax moment matching | Native epistemic | 63.54 ± 4.38 | 63.05 ± 3.04 | 61.64 ± 3.85 | 87.42 ± 2.90 |
| ReMax Laplace (diagonal) | Entropy | 62.21 ± 1.73 | 60.16 ± 1.65 | 60.75 ± 1.86 | 88.46 ± 1.87 |
| ReMax Laplace (diagonal) | $1-\max p$ | 63.21 ± 2.02 | 60.42 ± 1.81 | 62.38 ± 2.36 | 86.19 ± 2.80 |
| ReMax Laplace (diagonal) | Native epistemic | 63.64 ± 2.81 | 63.00 ± 1.59 | 61.37 ± 2.64 | 88.10 ± 2.35 |
| Hierarchical classifier (HRC) | Entropy | 67.87 ± 0.09 | 64.87 ± 0.14 | 68.61 ± 0.08 | 77.59 ± 0.15 |
| Hierarchical classifier (HRC) | $1-\max p$ | 68.08 ± 0.10 | 65.11 ± 0.16 | 68.71 ± 0.08 | 77.47 ± 0.15 |
| Hierarchical classifier (HRC) | Native epistemic | 34.93 ± 0.00 | 40.33 ± 0.00 | 41.65 ± 0.00 | 97.11 |
| Categorical TAGI-V | Entropy | 69.72 ± 0.07 | 66.87 ± 0.07 | 69.99 ± 0.08 | 76.37 ± 0.13 |
| Categorical TAGI-V | $1-\max p$ | 69.24 ± 0.07 | 66.41 ± 0.07 | 69.68 ± 0.07 | 76.47 ± 0.13 |
| Categorical TAGI-V | Native epistemic | 35.16 ± 0.00 | 40.45 ± 0.00 | 41.79 ± 0.00 | 97.04 ± 0.00 |
| Hierarchical TAGI-V | Entropy | 67.82 ± 0.10 | 64.93 ± 0.13 | 68.19 ± 0.09 | 78.41 ± 0.17 |
| Hierarchical TAGI-V | $1-\max p$ | 68.15 ± 0.13 | 65.38 ± 0.21 | 68.39 ± 0.12 | 78.23 ± 0.23 |
| Hierarchical TAGI-V | Native epistemic | 34.92 ± 0.01 | 40.33 ± 0.01 | 41.65 ± 0.01 | 97.09 ± 0.00 |
| Unit-probit HRC | Entropy | 67.37 ± 0.14 | 64.85 ± 0.18 | 67.46 ± 0.09 | 80.07 ± 0.13 |
| Unit-probit HRC | $1-\max p$ | 66.95 ± 0.13 | 64.53 ± 0.18 | 67.12 ± 0.09 | 80.28 ± 0.16 |
| Unit-probit HRC | Native epistemic | 35.15 ± 0.00 | 40.45 ± 0.00 | 41.79 ± 0.00 | 97.05 ± 0.00 |

## CIFAR-10 training duration and epistemic contraction

The diagnostic cohort is the first 2,048 validation features. At every epoch, $U$ is the mean predictive epistemic variance over examples and output dimensions. The "required epoch" is the earliest epoch for which validation NLL remains within 1% of its best value and accuracy remains within 0.5 percentage points of its best value for three consecutive records. The convergence label examines relative changes over epochs 170–180, 180–190, and 190–200.

| Method | Required epoch, mean (range) | $U_0$ | $U_{200}$ | $U_{200}/U_0$ | Epoch 170–200 |
| --- | --- | --- | --- | --- | --- |
| Probit OVR | 6.2 (6–7) | 2.596e-05 | 2.652e-06 | 0.102 | still shrinking |
| ReMax moment matching | not reached | 1.536e-02 | 1.242e-03 | 0.081 | plateau |
| ReMax Laplace (diagonal) | 9.2 (8–11) | 1.655e-02 | 1.617e-03 | 0.099 | plateau |
| Hierarchical classifier (HRC) | 8.2 (6–11) | 5.966e-03 | 6.343e-04 | 0.106 | still shrinking |
| Categorical TAGI-V | 42.2 (42–43) | 6.629e-04 | 6.240e-04 | 0.941 | plateau |
| Hierarchical TAGI-V | 4.8 (4–6) | 6.629e-04 | 2.557e-05 | 0.039 | still shrinking |

![Absolute epistemic uncertainty across epochs](report_assets/epistemic_uncertainty_absolute.png)

The filled regions show one sample standard deviation across five last-layer seeds. White-edged markers show the median validation-selected epoch. Absolute scales differ by orders of magnitude because gains and output transformations define different priors; cross-method comparisons should therefore emphasize contraction ratios rather than raw values.

![Epistemic uncertainty relative to initialization](report_assets/epistemic_uncertainty_relative.png)

Categorical TAGI-V rapidly reaches a high uncertainty floor and retains about 94% of its initial predictive epistemic variance. ReMax contracts to roughly 8–10% and plateaus or slightly rebounds late. Probit and fixed HRC also reach about 10% but are still shrinking by approximately 3% per ten epochs at epoch 200. Hierarchical TAGI-V contracts most strongly, to about 3.9%, and is also still shrinking. Thus some posteriors do not reach a finite empirical floor over the observed horizon.

### TAGI-V aleatoric uncertainty

Here $A$ is the mean positive variance-channel output over the same 2,048 validation examples and all class or tree-node outputs. The epistemic fraction is the mean $S_z/(S_z+\bar v^2)$.

| Method | $A_0$ | $A_{selected}$ | $A_{200}$ | $A_{200}/A_0$ | Epistemic fraction | Observed behavior |
| --- | --- | --- | --- | --- | --- | --- |
| Categorical TAGI-V | 1.050e-03 | 1.050e-03 | 1.050e-03 | 1.000 | 38.50 → 37.08 | constant / inactive channel |
| Hierarchical TAGI-V | 1.499e-03 | 5.903e-03 | 7.006e-03 | 4.673 | 30.52 → 0.77 | increasing; near late plateau |

![TAGI-V aleatoric uncertainty across epochs](report_assets/tagiv_aleatoric_uncertainty.png)

The two TAGI-V variants behave very differently. Hierarchical TAGI-V increases mean aleatoric variance from $1.50\times10^{-3}$ to $7.01\times10^{-3}$ (4.67×) while its epistemic component contracts; by epoch 200 only about 0.77% of total node variance is epistemic. Its aleatoric growth slows below 1% per ten epochs after epoch 170, indicating a near plateau.

Dense categorical TAGI-V does **not** adapt its aleatoric mean in this experiment: it remains exactly $1.049965\times10^{-3}$ at every recorded epoch and seed. Inspection of saved checkpoints confirms that variance-channel weights remain numerically negligible (about $10^{-13}$ mean absolute size at epoch 200) and the variance-channel bias mean remains at initialization. It is therefore more accurate to describe this configuration as having a trainable-but-empirically-inactive aleatoric channel. Its calibration improvement cannot be attributed to learned heteroscedastic variance; it comes from the coupled categorical update and uncertainty-tempered predictive. This is an implementation or parameterization issue to investigate, not evidence that dense TAGI-V has successfully learned aleatoric uncertainty.

![Validation proper-score and calibration dynamics](report_assets/validation_dynamics.png)

Continued contraction is not uniformly beneficial. The next table compares validation-selected and epoch-200 CIFAR-10-C calibration:

| Method | Selected NLL | Epoch-200 NLL | Selected ECE | Epoch-200 ECE |
| --- | --- | --- | --- | --- |
| Probit OVR | 1.275 | 1.321 | 16.80 | 16.84 |
| ReMax moment matching | 1.281 | 2.346 | 15.64 | 23.80 |
| ReMax Laplace (diagonal) | 1.246 | 1.407 | 15.82 | 18.57 |
| Hierarchical classifier (HRC) | 1.128 | 1.128 | 13.16 | 13.16 |
| Categorical TAGI-V | 1.098 | 1.202 | 15.51 | 17.70 |
| Hierarchical TAGI-V | 1.090 | 1.088 | 12.39 | 12.30 |

Moment-matched ReMax is the clearest failure mode: epoch 1 is selected, while continued training drives corrupted NLL from 1.281 to 2.346 and ECE from 15.64% to 23.80% without a meaningful accuracy gain. Diagonal Laplace ReMax and categorical TAGI-V show the same pattern more mildly. HRC is selected at epoch 200, while hierarchical TAGI-V is mixed across seeds. Validation-based checkpointing is therefore essential even when only a tiny last layer is trained.

## Feasibility and conclusion

This research direction is feasible within the deliberately narrow frozen-last-layer scope.

1. **The CIFAR-10 result is strong but does not transfer uniformly.** All six heads preserve CIFAR-10 accuracy and several materially improve calibration. On CIFAR-100, only categorical TAGI-V preserves nearly all clean and corrupted accuracy while improving the principal CIFAR-100-C calibration metrics.
2. **Categorical TAGI-V is the most robust 100-class choice.** It reaches 76.61% clean accuracy, improves CIFAR-100-C NLL/Brier/ECE, and gives the best entropy OOD results among the evaluated heads. Its clean calibration still trails softmax, so the benefit is shift-specific rather than universal.
3. **Hierarchical scaling is the main negative result.** Fixed HRC, hierarchical TAGI-V, and unit-probit HRC lose several accuracy points on CIFAR-100. Their low ECE values partly reflect reduced confidence and must be judged with NLL and accuracy. Unit-probit HRC also overtrains sharply after its epoch-1 selection.
4. **The uncertainty decomposition is not yet generally trustworthy.** Native epistemic variance fails as an OOD ranking score for most heads; ReMax is positively oriented but weaker on CIFAR-100 and its predictive calibration is highly sensitive to overtraining. Hierarchical TAGI-V learns an aleatoric channel, whereas the dense categorical channel remains inactive on both datasets.

The most defensible next step is to investigate why hierarchical output geometry loses accuracy at 100 classes and why dense TAGI-V's variance channel is inactive, while retaining categorical TAGI-V as the strongest frozen-feature CIFAR-100 candidate. Independently retrained backbones are still necessary before making architecture-independent claims.

## Complete metric appendix

The main text emphasizes accuracy, proper scores, marginal calibration, AUROC, AUPR-OOD, and FPR95. The following tables preserve the remaining metrics computed by the study. AURC is area under the risk–coverage curve; Risk@X is the error rate among the X% most confident predictions. Classwise ECE is the mean one-versus-rest calibration error. For OOD metrics, corrupted or SVHN examples are positive for AUPR-OOD and clean CIFAR-10 examples are positive for AUPR-ID.

### Clean CIFAR-10 secondary classification metrics (%)

| Method | Top-5 acc. ↑ | Mean conf. | Classwise ECE ↓ | AURC ↓ | Risk@80 ↓ | Risk@90 ↓ | Risk@95 ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 99.80 | 97.77 | 0.61 | 0.57 | 0.63 | 1.51 | 2.74 |
| Probit OVR | 99.69 ± 0.02 | 95.82 ± 0.05 | 0.36 ± 0.01 | 0.59 ± 0.02 | 0.66 ± 0.03 | 1.64 ± 0.03 | 2.83 ± 0.02 |
| ReMax moment matching | 99.63 ± 0.01 | 95.06 ± 0.05 | 0.41 ± 0.01 | 0.55 ± 0.01 | 0.67 ± 0.02 | 1.61 ± 0.01 | 2.79 ± 0.03 |
| ReMax Laplace (diagonal) | 99.63 ± 0.03 | 95.27 ± 0.73 | 0.41 ± 0.03 | 0.63 ± 0.05 | 0.71 ± 0.04 | 1.64 ± 0.04 | 2.80 ± 0.07 |
| Hierarchical classifier (HRC) | 99.64 ± 0.02 | 95.18 ± 0.01 | 0.32 ± 0.00 | 0.56 ± 0.01 | 0.59 ± 0.01 | 1.58 ± 0.05 | 2.87 ± 0.03 |
| Categorical TAGI-V | 99.78 ± 0.02 | 96.08 ± 0.00 | 0.36 ± 0.01 | 0.58 ± 0.01 | 0.62 ± 0.01 | 1.51 ± 0.01 | 2.75 ± 0.01 |
| Hierarchical TAGI-V | 99.64 ± 0.00 | 94.63 ± 0.06 | 0.33 ± 0.01 | 0.57 ± 0.01 | 0.59 ± 0.02 | 1.60 ± 0.02 | 2.90 ± 0.02 |

### CIFAR-10-C secondary shifted-classification metrics (%)

| Method | Top-5 acc. ↑ | Mean conf. | Classwise ECE ↓ | AURC ↓ | Risk@80 ↓ | Risk@90 ↓ | Risk@95 ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch softmax | 94.67 | 91.79 | 4.41 | 15.34 | 21.22 | 24.16 | 25.86 |
| Probit OVR | 94.24 ± 0.18 | 88.97 ± 0.07 | 4.11 ± 0.04 | 15.75 ± 0.27 | 21.39 ± 0.16 | 24.23 ± 0.15 | 25.86 ± 0.14 |
| ReMax moment matching | 94.35 ± 0.07 | 87.81 ± 0.20 | 4.05 ± 0.01 | 15.58 ± 0.03 | 21.45 ± 0.05 | 24.19 ± 0.05 | 25.78 ± 0.05 |
| ReMax Laplace (diagonal) | 94.05 ± 0.10 | 87.92 ± 1.23 | 4.03 ± 0.07 | 15.80 ± 0.18 | 21.41 ± 0.26 | 24.18 ± 0.24 | 25.80 ± 0.21 |
| Hierarchical classifier (HRC) | 93.57 ± 0.07 | 84.99 ± 0.07 | 3.71 ± 0.01 | 15.10 ± 0.03 | 21.25 ± 0.01 | 24.28 ± 0.01 | 26.04 ± 0.02 |
| Categorical TAGI-V | 94.53 ± 0.09 | 87.62 ± 0.02 | 3.84 ± 0.01 | 15.37 ± 0.06 | 21.13 ± 0.03 | 24.08 ± 0.02 | 25.80 ± 0.02 |
| Hierarchical TAGI-V | 93.78 ± 0.08 | 84.15 ± 0.16 | 3.64 ± 0.03 | 15.11 ± 0.05 | 21.22 ± 0.06 | 24.26 ± 0.07 | 26.01 ± 0.06 |

### Complete SVHN OOD metrics (%)

| Method | Score | AUROC ↑ | AUPR-OOD ↑ | AUPR-ID ↑ | FPR95 ↓ |
| --- | --- | --- | --- | --- | --- |
| PyTorch softmax | Entropy | 91.18 | 95.20 | 86.11 | 24.01 |
| PyTorch softmax | $1-\max p$ | 91.00 | 94.80 | 86.29 | 23.56 |
| Probit OVR | Entropy | 90.39 ± 0.73 | 94.78 ± 0.31 | 84.41 ± 1.89 | 26.08 ± 3.21 |
| Probit OVR | $1-\max p$ | 89.91 ± 0.76 | 94.02 ± 0.34 | 84.58 ± 1.83 | 25.76 ± 2.85 |
| Probit OVR | Native epistemic | 12.32 ± 0.00 | 53.45 ± 0.00 | 16.26 ± 0.00 | 99.93 |
| ReMax moment matching | Entropy | 88.15 ± 0.34 | 93.42 ± 0.14 | 79.54 ± 1.54 | 32.02 ± 1.94 |
| ReMax moment matching | $1-\max p$ | 88.11 ± 0.24 | 92.90 ± 0.13 | 80.71 ± 1.12 | 30.65 ± 1.12 |
| ReMax moment matching | Native epistemic | 90.21 ± 0.48 | 94.92 ± 0.29 | 84.01 ± 0.84 | 28.24 ± 0.72 |
| ReMax Laplace (diagonal) | Entropy | 88.04 ± 0.98 | 93.85 ± 0.33 | 77.92 ± 4.06 | 42.17 ± 12.02 |
| ReMax Laplace (diagonal) | $1-\max p$ | 87.98 ± 0.64 | 93.32 ± 0.27 | 79.02 ± 3.16 | 37.75 ± 8.23 |
| ReMax Laplace (diagonal) | Native epistemic | 89.91 ± 1.15 | 94.66 ± 0.67 | 82.59 ± 2.42 | 28.91 ± 2.30 |
| Hierarchical classifier (HRC) | Entropy | 92.48 ± 0.28 | 95.75 ± 0.20 | 88.80 ± 0.27 | 20.61 ± 0.76 |
| Hierarchical classifier (HRC) | $1-\max p$ | 92.21 ± 0.27 | 95.41 ± 0.20 | 88.73 ± 0.27 | 20.52 ± 0.70 |
| Hierarchical classifier (HRC) | Native epistemic | 23.74 ± 0.00 | 57.63 ± 0.00 | 17.91 ± 0.00 | 99.93 |
| Categorical TAGI-V | Entropy | 91.53 ± 0.13 | 95.56 ± 0.07 | 85.79 ± 0.27 | 24.17 ± 0.54 |
| Categorical TAGI-V | $1-\max p$ | 91.24 ± 0.12 | 95.13 ± 0.08 | 85.89 ± 0.25 | 23.81 ± 0.42 |
| Categorical TAGI-V | Native epistemic | 11.86 ± 0.01 | 53.29 ± 0.00 | 16.21 ± 0.00 | 99.93 |
| Hierarchical TAGI-V | Entropy | 92.24 ± 0.29 | 95.62 ± 0.21 | 88.58 ± 0.42 | 21.52 ± 0.86 |
| Hierarchical TAGI-V | $1-\max p$ | 92.01 ± 0.26 | 95.29 ± 0.18 | 88.59 ± 0.40 | 21.27 ± 0.87 |
| Hierarchical TAGI-V | Native epistemic | 16.48 ± 3.77 | 54.90 ± 1.42 | 16.79 ± 0.53 | 99.95 ± 0.01 |

### Complete CIFAR-10-C clean-versus-corrupted metrics (%)

| Method | Score | AUROC ↑ | AUPR-OOD ↑ | AUPR-ID ↑ | FPR95 ↓ |
| --- | --- | --- | --- | --- | --- |
| PyTorch softmax | Entropy | 70.76 | 69.46 | 68.59 | 80.55 |
| PyTorch softmax | $1-\max p$ | 70.67 | 68.99 | 68.60 | 80.44 |
| Probit OVR | Entropy | 69.16 ± 0.35 | 68.26 ± 0.24 | 66.49 ± 0.62 | 83.74 ± 1.44 |
| Probit OVR | $1-\max p$ | 69.17 ± 0.33 | 67.55 ± 0.21 | 66.76 ± 0.61 | 83.15 ± 1.42 |
| Probit OVR | Native epistemic | 32.46 ± 0.00 | 40.32 ± 0.00 | 39.51 ± 0.00 | 98.48 ± 0.00 |
| ReMax moment matching | Entropy | 69.59 ± 0.29 | 68.54 ± 0.19 | 67.28 ± 0.46 | 82.64 ± 1.24 |
| ReMax moment matching | $1-\max p$ | 69.63 ± 0.25 | 67.88 ± 0.19 | 67.51 ± 0.43 | 82.27 ± 1.18 |
| ReMax moment matching | Native epistemic | 70.49 ± 0.18 | 69.33 ± 0.19 | 68.36 ± 0.25 | 81.18 ± 0.78 |
| ReMax Laplace (diagonal) | Entropy | 68.55 ± 1.27 | 68.11 ± 0.56 | 65.69 ± 1.99 | 84.47 ± 3.64 |
| ReMax Laplace (diagonal) | $1-\max p$ | 68.64 ± 1.23 | 67.54 ± 0.56 | 65.94 ± 1.96 | 84.14 ± 3.61 |
| ReMax Laplace (diagonal) | Native epistemic | 69.95 ± 0.65 | 69.02 ± 0.54 | 67.42 ± 1.26 | 82.46 ± 3.01 |
| Hierarchical classifier (HRC) | Entropy | 71.33 ± 0.10 | 70.11 ± 0.08 | 69.81 ± 0.12 | 78.04 ± 0.30 |
| Hierarchical classifier (HRC) | $1-\max p$ | 71.25 ± 0.10 | 69.65 ± 0.09 | 69.87 ± 0.13 | 77.83 ± 0.32 |
| Hierarchical classifier (HRC) | Native epistemic | 36.88 ± 0.00 | 41.89 ± 0.00 | 41.57 ± 0.00 | 98.32 ± 0.00 |
| Categorical TAGI-V | Entropy | 71.05 ± 0.05 | 69.83 ± 0.05 | 68.99 ± 0.08 | 80.22 ± 0.22 |
| Categorical TAGI-V | $1-\max p$ | 70.93 ± 0.05 | 69.33 ± 0.03 | 68.99 ± 0.08 | 80.11 ± 0.24 |
| Categorical TAGI-V | Native epistemic | 32.49 ± 0.01 | 40.31 ± 0.00 | 39.52 ± 0.00 | 98.47 ± 0.00 |
| Hierarchical TAGI-V | Entropy | 70.99 ± 0.15 | 69.91 ± 0.12 | 69.28 ± 0.19 | 78.77 ± 0.43 |
| Hierarchical TAGI-V | $1-\max p$ | 70.92 ± 0.14 | 69.44 ± 0.13 | 69.36 ± 0.19 | 78.57 ± 0.43 |
| Hierarchical TAGI-V | Native epistemic | 34.74 ± 0.74 | 41.09 ± 0.26 | 40.43 ± 0.35 | 98.49 ± 0.04 |

## Reproducibility and artifacts

- Study manifest: [`study.json`](study.json)
- Runner: [`run_study.py`](run_study.py)
- Report generator: [`generate_report.py`](generate_report.py)
- Per-run machine-readable report: [`../../runs/last_layer/cifar_frozen_last_layer_v2_40k10k/report.csv`](../../runs/last_layer/cifar_frozen_last_layer_v2_40k10k/report.csv)
- Aggregated five-seed report: [`../../runs/last_layer/cifar_frozen_last_layer_v2_40k10k/report_summary.csv`](../../runs/last_layer/cifar_frozen_last_layer_v2_40k10k/report_summary.csv)
- Full JSON report: [`../../runs/last_layer/cifar_frozen_last_layer_v2_40k10k/report.json`](../../runs/last_layer/cifar_frozen_last_layer_v2_40k10k/report.json)

Regenerate this document and its figures with:

```bash
python experiments/last_layer/generate_report.py
```

The entropy histogram regeneration loads the selected TAGI checkpoints and therefore requires the cached clean/SVHN features and a CUDA-capable environment for the Triton heads.

The report combines the completed CIFAR-10 and CIFAR-100 studies. Every aggregate can be traced to the per-run CSV/JSON artifacts above; the CIFAR-100 backbone checkpoint and all validation-selected and epoch-200 last-layer checkpoints are retained under the same study artifact root.
