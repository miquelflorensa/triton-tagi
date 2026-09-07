# AGCI gain, noise scale, and repeated-pass dynamics

## Summary

The behavior of AGCI gain is not an arbitrary dataset effect. For a frozen
Bayesian last layer, gain, observation-noise scale, number of data passes,
class count, and feature geometry combine into an approximately dimensionless
effective-evidence scale

$$
\rho(E,g,\tau)
\approx
E\,
\frac{
g^2\,\beta_K\,N\,\mathbb E\lVert x\rVert^2
}{
D^2\left[
\tau^2+g^2(\mathbb E\lVert x\rVert^2+1)/D
\right]
}.
$$

Here $E$ is the number of repeated passes through the data, $g$ is the prior
gain, $N$ is the number of training examples, $D$ is the feature dimension,
$K$ is the number of classes, and $\beta_K$ is the information carried by one
Gaussian-argmax observation in a class-contrast direction.

Small $\rho$ is prior-dominated and undertrained. Around $\rho\sim 1$, the data
has substantially altered the posterior. Repeatedly driving $\rho$ upward by
reusing the same observations contracts the approximate posterior and can
produce overconfidence without improving accuracy.

In the small-prior-variance regime,

$$
\rho
\approx
E g^2
\frac{
\beta_K N\mathbb E\lVert x\rVert^2
}{D^2\tau^2},
$$

so the leading tradeoff is

$$
\boxed{\rho\propto E g^2/\tau^2.}
$$

This predicts the principal experimental behavior:

- multiplying gain by 10 gives approximately 100 times more evidence per
  pass;
- changing $\tau$ from 1 to 1.25 gives approximately $1/1.25^2=0.64$ as much
  evidence per pass;
- gain 0.1 requires tens to roughly one hundred repeated passes to reach the
  posterior regime reached by gain 1 in about one pass;
- the adaptive parameter cap is inactive at gain 0.1 and changes parameters
  but barely changes predictions at gain 1 under shuffled CIFAR batches.

The derivation below is exact at the symmetric AGCI starting point and becomes
a local Fisher/ADF approximation later in training.

---

## 1. Bayesian last-layer prior and its kernel

For a deterministic frozen feature vector $x\in\mathbb R^D$, the last layer is

$$
z_k(x)=x^\top w_k+b_k.
$$

The implementation initializes independent parameter variances as

$$
S_{W,dk}=\frac{g^2}{D},
\qquad
S_{b,k}=\frac{g^2}{D}.
$$

The parameter means are drawn at the ordinary He scale independently of $g$;
gain changes the prior uncertainty, not the random mean initialization.

The prior utility variance is therefore

$$
s_g^2(x)
=
\operatorname{Var}[z_k(x)]
=
\frac{g^2}{D}\left(\lVert x\rVert^2+1\right).
$$

The same prior induces the linear kernel

$$
k_g(x,x')
=
\operatorname{Cov}[z_k(x),z_k(x')]
=
\frac{g^2}{D}(x^\top x'+1).
$$

For an uncapped TAGI update, the output-mean movement at a new feature $x'$ can
be written as

$$
\Delta m_k(x')
=
\sum_{i\in B}k_g(x',x_i)\,\delta_{ik},
$$

where $B$ is the current batch and $\delta_{ik}$ is the AGCI output
innovation. Thus the last layer is performing a Bayesian kernel update whose
amplitude is proportional to $g^2$.

---

## 2. Exact symmetric AGCI innovation

AGCI observes

$$
Y=c
\quad\Longleftrightarrow\quad
c=\arg\max_j U_j,
\qquad
U_j=Z_j+\epsilon_j,
\qquad
\epsilon_j\sim\mathcal N(0,\tau^2).
$$

At the symmetric starting point, let every latent utility have mean zero and
variance $s_g^2$. Then

$$
U_j\sim\mathcal N(0,q),
\qquad
q=\tau^2+s_g^2.
$$

Define the standard-Gaussian maximum constant

$$
a_K
=
\mathbb E\left[\max_{1\le j\le K}G_j\right],
\qquad
G_j\overset{\mathrm{iid}}\sim\mathcal N(0,1).
$$

Conditioned on class $c$ winning,

$$
\mathbb E[U_c\mid Y=c]=\sqrt q\,a_K.
$$

The argmax event depends only on utility contrasts and is independent of the
common Gaussian location mode. Consequently the conditional expected sum of
the utilities remains zero. By symmetry, every losing utility has conditional
mean

$$
\mathbb E[U_j\mid Y=c]
=
-\frac{\sqrt q\,a_K}{K-1},
\qquad j\ne c.
$$

Fisher's identity, or equivalently the AGCI conditional-moment projection,
then gives the exact initial mean innovations

$$
\delta_c
=
\frac{a_K}{\sqrt q},
\qquad
\delta_{j\ne c}
=
-\frac{a_K}{(K-1)\sqrt q}.
$$

The relevant constants are

| Classes $K$ | $a_K$ | $\beta_K=a_K^2K/(K-1)^2$ |
| ---: | ---: | ---: |
| 10 | 1.53875 | 0.292316 |
| 100 | 2.50759 | 0.064157 |
| 1,000 | 3.24144 | 0.010528 |

The winning-class innovation grows slowly with $K$, but information in each
class-contrast direction decreases approximately as

$$
\beta_K\sim\frac{2\log K}{K}.
$$

Class count therefore does not enter the prior initialization, but it enters
the AGCI likelihood information strongly.

---

## 3. Initial discriminative movement

Consider a balanced dataset with $n=N/K$ examples per class. At the symmetric
point, summing the innovations for output class $k$ gives

$$
\sum_i x_i\delta_{ik}
=
\frac{a_K n}{\sqrt q}
(\mu_k-\mu_{-k}),
$$

where the product on the right is scalar multiplication, $\mu_k$ is the
class-$k$ feature centroid, and $\mu_{-k}$ is the centroid of all other
classes. Substituting into the kernel update gives

$$
\Delta m_k(x')
=
\frac{g^2a_Kn}{D\sqrt{\tau^2+s_g^2}}
x'^\top(\mu_k-\mu_{-k}).
$$

The initial discriminative learning scale is therefore

$$
\boxed{
\frac{g^2}{\sqrt{\tau^2+s_g^2}}
\frac{N}{K}
a_K
x'^\top(\mu_k-\mu_{-k})
}.
$$

This equation shows why feature norm or class count alone cannot determine an
appropriate gain. The relevant dataset quantities include the number of
examples per class and the alignment of each feature with the corresponding
class-centroid contrast.

For the cached training features:

| Quantity | CIFAR-10 | CIFAR-100 |
| --- | ---: | ---: |
| Training examples | 40,000 | 40,000 |
| Feature dimension | 512 | 512 |
| Examples/class | 4,000 | 400 |
| Mean $\lVert x\rVert$ | 5.87 | 11.59 |
| Mean $\lVert x\rVert^2$ | 34.53 | 135.99 |
| Mean correct-class centroid contrast | 26.0 | 88.2 |
| $a_K$ | 1.539 | 2.508 |

At the same gain and noise scale, the ratio of initial correct-class progress
is approximately

$$
\frac{\text{CIFAR-100 progress}}{\text{CIFAR-10 progress}}
\approx
\frac{400}{4000}
\frac{2.508}{1.539}
\frac{88.2}{26.0}
\approx 0.55.
$$

The Gaussian maximum scale that must be overcome is also about
$2.508/1.539=1.63$ times larger at 100 classes. A crude initial prediction is
therefore that CIFAR-100 needs about

$$
1.63/0.55\approx 3
$$

times as much effective training to reach comparable discrimination. In the
observed gain-0.1 trajectories, CIFAR-10 reaches its accuracy plateau after
about one pass, while CIFAR-100 approaches its plateau after roughly three to
five passes.

---

## 4. Fisher information and the effective-evidence coordinate

At the symmetric point, the score vector for an observation of class $c$ is

$$
v_c
=
\frac{a_K}{\sqrt q}
\frac{K}{K-1}
\left(e_c-\frac{1}{K}\mathbf 1\right).
$$

Averaging $v_cv_c^\top$ over uniformly distributed classes gives

$$
F_{\text{class}}
=
\frac{\beta_K}{q}
\left(I-\frac{1}{K}\mathbf1\mathbf1^\top\right),
\qquad
\beta_K=\frac{a_K^2K}{(K-1)^2}.
$$

Ignoring the small bias contribution, the local weight Fisher matrix has the
Kronecker form

$$
\mathcal I_W
\approx
\frac{\beta_K}{q}
\left(I-\frac{1}{K}\mathbf1\mathbf1^\top\right)
\otimes X^\top X.
$$

The dimensionless posterior evidence is prior covariance times likelihood
information. Averaging across feature eigenmodes and accumulating $E$ repeated
passes gives

$$
\rho(E,g,\tau)
\approx
E\,
\frac{
g^2\beta_KN\mathbb E\lVert x\rVert^2
}{
D^2\left[
\tau^2+g^2(\mathbb E\lVert x\rVert^2+1)/D
\right]
}.
$$

This scalar is only an average over the feature Gram spectrum. A more complete
analysis replaces $\mathbb E\lVert x\rVert^2/D$ by the individual eigenvalues
of $X^\top X$. Nevertheless, the average already predicts the main observed
dynamics.

For $\tau=1$:

| Dataset | Gain | $s_g^2/\tau^2$ | Evidence/pass $\rho/E$ | Passes for $\rho\approx1$ |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | 0.1 | 0.000694 | 0.01539 | 65.0 |
| CIFAR-10 | 1.0 | 0.06939 | 1.4401 | 0.69 |
| CIFAR-100 | 0.1 | 0.002676 | 0.01328 | 75.3 |
| CIFAR-100 | 1.0 | 0.26757 | 1.0503 | 0.95 |

Thus gain 0.1 gives small posterior movement over dozens of passes, whereas
gain 1 assimilates roughly one posterior's worth of evidence in one pass.

The CIFAR-100 trajectories provide a particularly direct validation:

| Configuration | Validation accuracy | NLL | Confidence |
| --- | ---: | ---: | ---: |
| gain 1.0, epoch 1 | 75.90% | 0.9466 | 73.22% |
| gain 0.1, epoch 100 | 75.99% | 0.9564 | 73.05% |

These configurations have equal $Eg^2$. Their confidence and accuracy are
nearly identical, as predicted by the small-g effective-time scaling. The
remaining NLL difference is expected because AGCI is nonlinear, its posterior
is projected back to a diagonal family after each observation, and the two
paths do not retain identical variances.

---

## 5. Repeated epochs are likelihood tempering

In exact Bayesian inference, processing the same dataset $E$ times gives

$$
p_E(W\mid\mathcal D)
\propto
p(W)\,p(\mathcal D\mid W)^E.
$$

This is a power posterior rather than the ordinary posterior after observing
the dataset once. AGCI/TAGI's ADF projection and capped batch updates make the
identity approximate, but the repeated-evidence mechanism remains.

Consequently, epoch count acts like an inverse likelihood temperature:

- gain 0.1 plus many epochs uses repeated data to compensate for a highly
  concentrated prior;
- gain 1 moves to the data-dominated regime in one or two epochs;
- continuing gain-1 training raises confidence while accuracy remains flat,
  worsening NLL and calibration;
- posterior contraction across epochs is partly repeated-data contraction,
  not evidence that uncertainty is converging to a statistically valid
  posterior.

If the goal is literal Bayesian updating, every observation should be
assimilated once and gain should represent an actual prior belief. If repeated
epochs are retained as a numerical device, comparisons should be organized by
$Eg^2/\tau^2$ or the fuller $\rho$ above rather than by gain or epoch alone.

---

## 6. Why the adaptive cap has little predictive effect

For batch size 256 the implementation uses cap factor $c=3$. The raw weight
mean update is

$$
\Delta m_{W,dk}
=
\frac{g^2}{D}
\sum_{i\in B}x_{id}\delta_{ik},
$$

while the cap magnitude is

$$
\overline\Delta_{W,dk}
=
\frac{\sqrt{S_W}}{c}
=
\frac{g}{c\sqrt D}.
$$

The cap binds precisely when

$$
\left|\sum_{i\in B}x_{id}\delta_{ik}\right|
>
\frac{\sqrt D}{cg}.
$$

At $D=512$ and $c=3$, the normalized batch-score thresholds are

| Gain | Binding threshold |
| ---: | ---: |
| 0.1 | 75.4 |
| 1.0 | 7.54 |

Gain 0.1 therefore has a threshold ten times larger and did not bind in the
matched CIFAR-10 experiments: capped and uncapped parameter tensors were
bit-for-bit identical after ten epochs.

At gain 1 the cap changes some parameters, but capped and uncapped predictions
remain almost identical on both datasets. For CIFAR-100 at the selected epoch:

| Gain 1, $\tau=1$ | Capped | Uncapped |
| --- | ---: | ---: |
| Selected epoch | 2 | 2 |
| Test accuracy | 76.67% | 76.69% |
| Test NLL | 0.93100 | 0.93143 |
| Test ECE | 3.023% | 3.024% |
| Test Brier | 0.33019 | 0.33013 |

The cap is therefore an emergency coordinate-wise safeguard, not the main
control on the classifier's effective learning rate. It can become important
for pathologically imbalanced batches, where the batch-score cancellation
assumed by the equations above fails. The class-ordered ImageNet shard failure
is an example of exactly that regime.

---

## 7. Why the ImageNet feature-norm heuristic failed

Matching only prior utility standard deviation across datasets ignores the
class-contrast information factor $\beta_K$. Using the observed CIFAR feature
energies and the ImageNet report's mean feature norm gives

| Dataset | Classes | $\beta_K$ | Feature energy | $\beta_K\times$ feature energy |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | 10 | 0.29232 | 34.53 | 10.1 |
| CIFAR-100 | 100 | 0.06416 | 135.99 | 8.7 |
| ImageNet | 1,000 | 0.01053 | approximately $29.59^2$ | approximately 9.2 |

The feature energy rises with class count, but the information in each AGCI
class-contrast direction falls. In these three frozen backbones, the two
effects almost exactly cancel.

This explains why scaling ImageNet gain downward solely to match CIFAR-100's
prior utility standard deviation predicted 0.4--0.5, while the measured
plateau remained near 0.7--1.0. Class count is absent from the prior but not
from the likelihood.

---

## 8. A transferable gain rule

In the small-prior-variance regime, two experiments should have comparable
early AGCI dynamics when

$$
E g^2
\frac{
N\beta_K\mathbb E\lVert x\rVert^2
}{D^2\tau^2}
$$

is held fixed. A gain transferred from experiment 1 to experiment 2 is
therefore approximately

$$
g_2
\approx
g_1\frac{\tau_2}{\tau_1}
\sqrt{
\frac{
E_1N_1\beta_{K_1}\mathbb E\lVert x_1\rVert^2D_2^2
}{
E_2N_2\beta_{K_2}\mathbb E\lVert x_2\rVert^2D_1^2
}
}.
$$

This is not a universal optimum: the desired evidence level still depends on
model misspecification, calibration objectives, the spectrum of the feature
Gram matrix, and class-centroid geometry. It is, however, a principled
one-dimensional scaling law that replaces blind gain sweeps.

A direct falsification test is to rerun several $(E,g,\tau)$ combinations and
plot accuracy, NLL, confidence, and posterior contraction against $\rho$
instead of epoch. If the theory captures the dominant dynamics, curves from
different gains and noise scales should approximately collapse during the
early and middle parts of training.

---

## 9. Scope and limitations

The derivation is exact for the initial symmetric independent-Gaussian AGCI
model. The effective-evidence dynamics after that point are approximate
because:

- the event-conditioned posterior has dense cross-class covariance, while the
  TAGI innovation retains only its diagonal;
- features have an anisotropic Gram spectrum, whereas the scalar $\rho$ uses
  its average eigenvalue;
- feature distributions and labels are correlated through class geometry;
- batch updates sum innovations computed from a shared pre-update posterior;
- the coordinate-wise cap changes large raw updates;
- repeated ADF projection is order-dependent;
- the likelihood curvature changes as class margins grow.

These limitations affect the exact curve and optimum but not the central
scaling mechanism: gain supplies prior kernel amplitude $g^2$, $\tau$ sets the
likelihood scale, repeated passes multiply evidence, and $K$ enters through
Gaussian-maximum class-contrast information.

## Associated results

- `heads/agci/cifar10/result.json`: CIFAR-10, gain 0.1, $\tau=1$.
- `heads/agci/cifar100_tau1_epochs100/result.json`: CIFAR-100, gain 0.1,
  $\tau=1$, 100 epochs.
- `heads/agci/cifar100_tau1_gain1_epochs100/result.json`: CIFAR-100, gain 1,
  $\tau=1$, 100 epochs.
- `heads/agci_cap_ablation/cifar10_gain1_tau1_capped/result.json`: capped
  CIFAR-10 gain-1 ablation.
- `heads/agci_cap_ablation/cifar10_gain1_tau1_uncapped/result.json`: uncapped
  CIFAR-10 gain-1 ablation.
- `heads/agci_cap_ablation/cifar100_gain1_tau1_uncapped/result.json`: uncapped
  CIFAR-100 gain-1 ablation.

All result paths above are relative to
`runs/last_layer/cifar_frozen_last_layer_v2_40k10k/`.
