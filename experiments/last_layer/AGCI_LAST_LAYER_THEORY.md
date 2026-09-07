# AGCI as a Bayesian last-layer model

## Purpose and scope

This note formalizes the AGCI head used on top of frozen deterministic
features. It has three goals:

1. define the probabilistic model represented by the head;
2. state what the TAGI update means in Bayesian terms, including its
   approximations;
3. give a principled initialization procedure for a new last layer and for a
   pretrained warm start.

The focus is the model and its initialization. Empirical results, dataset
comparisons, and claims about a best measured configuration are intentionally
excluded.

---

## 1. Model

Let a frozen feature extractor map an input to

$$
h=f_\theta(x)\in\mathbb R^D,
$$

where $\theta$ is fixed. For $K$ classes, the last layer defines one latent
utility per class:

$$
z_k(h)=h^\top w_k+b_k,
\qquad k=1,\ldots,K.
$$

Collect the parameters into $W=[w_1,\ldots,w_K]\in\mathbb R^{D\times K}$ and
$b\in\mathbb R^K$.

AGCI uses a Gaussian random-utility observation model:

$$
U_k=z_k+\epsilon_k,
\qquad
\epsilon_k\overset{\mathrm{iid}}{\sim}\mathcal N(0,\tau^2),
$$

and observes only the winning class,

$$
Y=c
\quad\Longleftrightarrow\quad
U_c\ge U_j\quad\text{for every }j\ne c.
$$

Thus the categorical likelihood is a multinomial-probit likelihood. The
parameter $\tau>0$ is the fixed decision-noise scale. It is part of the
likelihood and must be the same during training and prediction.

### 1.1 Translation and scale

Only utility contrasts are identified:

$$
\arg\max_k(z_k+a)=\arg\max_k z_k.
$$

A common offset of all utilities therefore has no likelihood information. It is
useful to fix this gauge by centering class-wise parameter means:

$$
\sum_{k=1}^K w_k=0,
\qquad
\sum_{k=1}^K b_k=0.
$$

The likelihood also depends on utility scale only relative to $\tau$. If all
utility means, utility standard deviations, and $\tau$ are multiplied by the
same positive constant, class probabilities are unchanged. Consequently,
$\tau$ should normally define the utility unit, with the prior expressed
relative to it.

### 1.2 The decision-noise law sets the probability tail

Gaussianity of $\epsilon$ is a modeling choice, not a requirement of the
argmax-event derivation. Any i.i.d. law for $\epsilon$ defines a random-utility
likelihood, and the choice controls how fast class probability decays for a
class whose utility trails the winner. Write that margin as

$$
\Delta_c=\max_j z_j-z_c\ge0 .
$$

For Gaussian decision noise the event probability behaves like a Gaussian tail,

$$
\log P(Y=c)\;\asymp\;-\frac{\Delta_c^2}{2\,(s_c+s_{\max}+2\tau^2)},
$$

quadratic in the margin. For Gumbel decision noise with scale $\beta$, the
classical random-utility result gives the multinomial-logit likelihood
conditional on the utilities,

$$
\epsilon_k\overset{\mathrm{iid}}{\sim}\mathrm{Gumbel}(0,\beta)
\quad\Longrightarrow\quad
P\big(Y=c\mid z\big)=\mathrm{softmax}(z/\beta)_c ,
$$

whose tail is only *linear* in the margin,

$$
\log P(Y=c\mid z)\;\asymp\;-\frac{\Delta_c}{\beta} .
$$

This distinction is not a calibration detail. Negative log likelihood is
dominated by the examples whose observed class sits far down the ranking, and
on those examples the Gaussian link charges a penalty quadratic in a margin
where the logit link charges one that is linear. Matching only the noise
variance, as $\tau=\pi/\sqrt6$ does in Section 7.2, equalizes the second
moment and leaves this tail mismatch untouched.

Two consequences follow, and both were tested:

1. Widening $\tau$ cannot repair the tail. It divides the whole quadratic
   exponent by a larger constant, so it flattens the confident bulk at the
   same rate as it lifts the tail, trading calibration for tail mass. This was
   confirmed: sweeping the predictive $\tau$ improved NLL only by degrading
   calibration.
2. Replacing the noise law with Gumbel repairs the tail by construction, at a
   fixed utility posterior. This too was confirmed, but the argument does *not*
   predict the sign of the net effect on proper scoring, and Section 13.5
   records one dataset where the Gaussian link scores better despite the
   thinner tail. The tail governs how large a prior scale the Gaussian link can
   afford; it does not by itself say which link wins.

Substituting $z=\beta u$ shows that $\beta$ is a pure gauge exactly as $\tau$
is: the likelihood $\mathrm{softmax}(z/\beta)$ is invariant and the prior
variance scales as $\beta^2$, so the model depends only on the dimensionless
ratio $\kappa$ of Section 6.2. Fixing $\beta=1$ therefore costs nothing.

#### Conditioning on the observed class under Gumbel noise

The Gumbel link keeps the whole AGCI construction, because the argmax event
still supplies the likelihood and the utilities still carry a Gaussian
posterior. Only the evidence changes. Let $q(z)=\mathcal N(\mu,S)$ with
diagonal $S$ be the prior predictive over utilities and write the log evidence

$$
\log M(\mu,S)
=\log\mathbb E_{z\sim\mathcal N(\mu,S)}
\big[\mathrm{softmax}(z/\beta)_y\big].
$$

For a Gaussian prior, Bonnet's and Price's theorems make the moment-matched
posterior exact in the derivatives of this quantity:

$$
\mathbb E[z_i\mid y]=\mu_i+S_i\,\partial_{\mu_i}\log M,
\qquad
\mathrm{Var}[z_i\mid y]=S_i+S_i^2\,\partial^2_{\mu_i}\log M .
$$

The TAGI output innovations are therefore precisely the first two derivatives
of the log evidence with respect to the output mean. Both are self-normalized
Monte Carlo averages over reparameterized prior draws with analytic softmax
derivatives,

$$
\partial_{z_i}p_y=\beta^{-1}p_y a_i,
\qquad
\partial^2_{z_i}p_y=\beta^{-2}p_y\big(a_i^2-p_i(1-p_i)\big),
\qquad
a_i=[\,i=y\,]-p_i ,
$$

so no competitor quadrature is needed, and in the limit $S\to0$ the update
reduces exactly to the deterministic multinomial-logit ADF step. Sampling is
antithetic and seed-driven, so results are reproducible.

The cost comparison differs between training and prediction, and only the
latter favors the logit link asymptotically. Conditioning on the *observed*
event costs $O(QK)$ under Gaussian noise, because a single candidate class
requires one shifted quadrature over its $K-1$ competitors; the Gumbel update
costs $O(MK)$. These are the same order, with constants $Q=48$ against $M=32$.
Prediction, however, needs every candidate class, so the Gaussian link pays
$O(QK^2)$ against the logit link's unchanged $O(MK)$. Measured wall time for
one pass, as a ratio of Gumbel to Gaussian, follows that quadratic term rather
than any uniform advantage: 1.65 at $K=10$, 0.61 at $K=100$, and 0.23 at
$K=1000$. At small $K$ the Gumbel head is the slower of the two.

---

## 2. Bayesian last-layer prior

The implemented TAGI family is a diagonal Gaussian over last-layer parameters:

$$
q(W,b)
=
\prod_{d,k}\mathcal N(W_{dk};M_{dk},S_{W,dk})
\prod_k\mathcal N(b_k;m_{b,k},S_{b,k}).
$$

At initialization this distribution is the prior $q_0$. After data
assimilation it is the approximate posterior.

Because $h$ is deterministic, the induced latent utility is Gaussian:

$$
Z_k(h)\sim\mathcal N(\mu_k(h),v_k(h)),
$$

with

$$
\mu_k(h)=h^\top M_{\cdot k}+m_{b,k},
$$

and

$$
v_k(h)
=
(h\odot h)^\top S_{W,\cdot k}+S_{b,k}.
$$

The standard implementation keeps the utilities independent before observing
the class event. This is exact under the diagonal parameter family but does
not represent posterior correlations between classes.

The prior also defines a linear covariance kernel. Under an isotropic,
class-symmetric prior,

$$
S_{W,dk}=\frac{g_w^2}{D},
\qquad
S_{b,k}=\frac{g_b^2}{D},
$$

the utility covariance for one class is

$$
k(h,h')
=
\operatorname{Cov}[Z_k(h),Z_k(h')]
=
\frac{g_w^2}{D}h^\top h'
+
\frac{g_b^2}{D}.
$$

This makes the meaning of gain precise: it controls prior covariance and hence
the amplitude of Bayesian updating. It is not merely an optimizer learning
rate.

---

## 3. AGCI class probabilities

Let

$$
r_k^2=v_k+\tau^2.
$$

For diagonal latent utility moments, the probability of class $c$ can be
written as a one-dimensional integral:

$$
p(Y=c\mid h,\mathcal D)
=
\int_{-\infty}^{\infty}
\mathcal N(u;\mu_c,r_c^2)
\prod_{j\ne c}
\Phi\left(\frac{u-\mu_j}{r_j}\right)
\,du.
$$

The apparently $(K-1)$-dimensional comparison event therefore reduces to one
dimension. The implementation evaluates this integral using shifted
Gauss-Hermite quadrature. The shift changes numerical efficiency, not the
model.

The $K$ event probabilities form a partition. Their final normalization only
removes common quadrature error; it is not a replacement for a categorical
likelihood.

At a symmetric prior,

$$
\mu_1=\cdots=\mu_K,
\qquad
v_1=\cdots=v_K,
$$

the predictive class distribution is exactly uniform:

$$
p(Y=k)=\frac1K.
$$

This is one reason a zero-mean, class-symmetric prior is especially natural for
a from-scratch last layer.

---

## 4. Conditioning on an observed class

For every possible class $j$, define

$$
m_j=\mathbb E[Z\mid Y=j,h],
\qquad
M=[m_1,\ldots,m_K].
$$

The categorical observation moments are

$$
\mathbb E[Y]=p,
\qquad
\Sigma_Y=\operatorname{diag}(p)-pp^\top,
$$

and the output-label cross-covariance is

$$
\Sigma_{ZY}
=
[p_1(m_1-\mu),\ldots,p_K(m_K-\mu)].
$$

The minimum-MSE affine categorical gain is

$$
J_{\mathrm{AGCI}}
=\Sigma_{ZY}\Sigma_Y^\dagger.
$$

The pseudoinverse is required because $\Sigma_Y\mathbf 1=0$: a categorical
observation contains only $K-1$ independent contrasts. The implementation may
equivalently work in an orthonormal contrast basis. The current diagonal TAGI
path uses the algebraically equivalent moment identities directly and does not
materialize the pseudoinverse.

For observed class $c$, the affine mean update is

$$
\mu^+
=\mu+J_{\mathrm{AGCI}}(e_c-p)
=m_c
=\mathbb E[Z\mid Y=c,h].
$$

This equality is exact given exact probabilities and conditional means. It
holds because any function of a one-hot categorical observation can be
represented affinely on its $K$ support points.

The full LMMSE-AGCI covariance differs from the covariance of one observed
event:

$$
\Sigma_{Z,\mathrm{AGCI}}^+
=
\operatorname{diag}(v)
-\Sigma_{ZY}\Sigma_Y^\dagger\Sigma_{YZ}
=
\sum_{j=1}^K p_j\operatorname{Cov}(Z\mid Y=j,h).
$$

This class-averaged covariance is independent of the observed class. It is a
useful comparison method, but it is not the covariance used by Event AGCI.

Event AGCI instead uses $\operatorname{Cov}(Z\mid Y=c,h)$. For diagonal output
covariance, define $r_k^2=v_k+\tau^2$ and $\beta_k=v_k/r_k^2$. Its retained
marginal variance is

$$
v_k^+
=v_k+\beta_k^2
\left[\operatorname{Var}(A_k\mid Y=c,h)-r_k^2\right].
$$

The standardized output innovations are

$$
\delta_{\mu,k}
=
\frac{m_{k\mid c}-\mu_k}{v_k},
$$

$$
\delta_{v,k}
=
\frac{v_k^+-v_k}{v_k^2}.
$$

These innovations are passed backward through the linear layer. For a batch
feature matrix $H$, the uncapped updates have the form

$$
\Delta M
=
S_W\odot H^\top\delta_\mu,
$$

$$
\Delta S_W
=
S_W^2\odot (H\odot H)^\top\delta_v,
$$

with analogous bias updates. The observation usually contracts variance, so
$\delta_v$ is normally non-positive.

The `agci` training head computes this observed-event variance directly in
$O(QK)$ work without materializing a dense covariance. The class-averaged
LMMSE routines remain available for explicit comparisons.

---

## 5. What is Bayesian about the head?

Under a prior chosen independently of the fitted data, and when each
observation is assimilated once, the head has the main components of Bayesian
inference:

- a probability distribution over $W$ and $b$;
- a proper categorical likelihood induced by Gaussian random utilities;
- posterior updating based on conditional moments;
- posterior-predictive probabilities that integrate parameter uncertainty;
- sequential updating when new labeled data arrives.

The resulting uncertainty is conditional on the frozen representation. It
does not include uncertainty about $\theta$, feature extraction, architecture,
or dataset support.

### 5.1 What the head brings

Compared with a deterministic linear classifier, the last layer provides:

- explicit prior regularization through $q_0(W,b)$;
- analytic, gradient-free parameter updates;
- a posterior state that can be updated sequentially;
- predictive probabilities that combine fixed decision noise and last-layer
  parameter uncertainty;
- a compact approximation that scales linearly in the number of last-layer
  parameters.

Its strongest interpretation is therefore:

> approximate Bayesian inference for a multinomial-probit linear model,
> conditional on a fixed learned representation.

Calling it a fully Bayesian neural network would be too strong because the
feature extractor is still a point estimate.

### 5.2 What is approximate?

The posterior is not exact because:

1. multiclass probabilities and conditional means use numerical quadrature;
2. the event-conditioned output is represented by Gaussian moments;
3. output cross-class covariance is discarded by the diagonal TAGI family;
4. parameter covariance is diagonal;
5. batch updates combine innovations computed from a shared pre-update state;
6. sequential Gaussian projection is order-dependent;
7. the optional coordinate-wise update cap modifies large moment updates.

These approximations do not remove the Bayesian model, but they mean that
$q(W,b)$ should be called an approximate posterior rather than the posterior.

---

## 6. Initialization from scratch

### 6.1 Use deliberate zero means

For a new last layer with no prior classifier, use

$$
M_0=0,
\qquad
m_{b,0}=0.
$$

This expresses class symmetry and gives uniform initial probabilities.

The default linear-layer initializer draws parameter means at He scale while
`gain_w` and `gain_b` control only the stored variances. Interpreted
Bayesianly, that is a Gaussian prior centered on one arbitrary random
classifier. The random mean is not a draw being integrated out; it becomes the
center of the prior. This is usually undesirable for a frozen Bayesian last
layer.

Random initialization is useful in many neural networks to break hidden-unit
symmetry. It is unnecessary here: class labels immediately create different
updates for different utility columns.

### 6.2 Choose variance using feature energy

Raw gain is not transferable when feature scale changes. Define the mean
per-coordinate feature energy

$$
e_h
=
\frac{1}{D}\mathbb E\lVert h\rVert^2.
$$

Under the implemented isotropic prior,

$$
\mathbb E_h[v_0(h)]
=
g_w^2 e_h+\frac{g_b^2}{D}.
$$

Define a dimensionless prior utility ratio

$$
\kappa^2
=
\frac{\mathbb E_h[v_0(h)]}{\tau^2}.
$$

Then, for a chosen $\kappa$ and bias gain,

$$
\boxed{
g_w
=
\sqrt{
\frac{\kappa^2\tau^2-g_b^2/D}{e_h}
}
}
$$

whenever the numerator is positive.

This is the principled way to initialize gain:

1. compute $e_h$ on training features only;
2. fix the likelihood unit $\tau$;
3. choose the prior utility ratio $\kappa$;
4. derive $g_w$ from the equation above.

A value of $\kappa$ of order one gives prior utility uncertainty comparable to
the decision-noise scale. Smaller $\kappa$ expresses a more concentrated prior
and produces smaller updates; larger $\kappa$ expresses greater uncertainty
and allows stronger movement. There is no dataset-independent optimal raw
gain.

If $\kappa$ is selected using held-out likelihood, that is an empirical-Bayes
hyperparameter choice. It remains valid, but it should be reported as such.

### 6.3 Treat feature preprocessing as part of the model

An isotropic weight prior is meaningful only relative to the feature
coordinates. A clean protocol is:

1. estimate a feature mean and scale using training data;
2. center features;
3. optionally whiten them or apply one global RMS scale;
4. freeze that transform and reuse it for validation and test data;
5. initialize the prior in the transformed coordinates.

Centering reduces confounding between weights and biases. Whitening makes the
isotropic prior closer to equal regularization across feature directions. If
raw backbone features are retained, their energy and anisotropy must be
acknowledged when setting $S_W$.

### 6.4 Biases and class prevalence

For a balanced or deliberately class-neutral prior, initialize all bias means
to zero and use a modest symmetric bias variance.

Known class prevalence can instead be encoded through bias means. Unlike
softmax, however, setting $b_k=\log\pi_k$ does not exactly produce class prior
$\pi_k$ under a Gaussian argmax likelihood. Biases that target a non-uniform
prior should be obtained by solving the AGCI class-probability map, or treated
as an approximation.

---

## 7. Initialization from a pretrained classifier

A pretrained deterministic classifier is a different prior construction. Let
$(W_\star,b_\star)$ be its fitted parameters. A warm-started Bayesian head can
use

$$
M_0=W_\star,
\qquad
m_{b,0}=b_\star,
$$

together with a covariance that represents uncertainty around that fitted
solution.

This is coherent sequential Bayes only when the pretrained state summarizes
information from data that will not be assimilated again. If
$(W_\star,b_\star)$ was fitted on the same dataset and AGCI then processes that
dataset again, the evidence is counted twice. The construction is then an
empirical-Bayes or recycled-likelihood approximation, not an ordinary
posterior update.

### 7.1 Covariance for a warm start

In decreasing order of statistical fidelity, use:

1. a posterior covariance carried from the original training procedure;
2. a Laplace, Fisher, or other local covariance approximation;
3. a declared diagonal covariance chosen from an effective prior strength.

Using pretrained means with the same diffuse variance as a new random layer
mixes two incompatible beliefs: trusted means and untrained uncertainty. A
warm start should normally have less covariance than a from-scratch prior,
unless there is a specific reason to distrust the pretrained solution.

### 7.2 Mapping a softmax classifier to Gaussian utilities

Softmax has an exact random-utility representation with independent
unit-scale Gumbel noise. Its noise standard deviation is

$$
\frac{\pi}{\sqrt6}.
$$

AGCI replaces Gumbel noise with Gaussian noise. Reusing softmax logits with

$$
\tau\approx\frac{\pi}{\sqrt6}
$$

matches noise variance, but not the full noise distribution. Equivalently, if
AGCI fixes $\tau=1$, a first-order scale match rescales the pretrained logits
by

$$
\frac{\sqrt6}{\pi}.
$$

This is a moment match, not an exact equivalence between softmax and
multinomial probit.

Before a warm start, remove the unidentifiable common utility offset by
centering $W_\star$ and $b_\star$ across classes. This leaves all argmax
probabilities unchanged.

---

## 8. Meaning of the main hyperparameters

### `agci_tau`

$\tau$ is the standard deviation of Gaussian decision noise. It sets the
utility unit and the softness of the categorical likelihood. It is not merely
a training-noise knob.

Because utility scale and $\tau$ are jointly non-identifiable, a clean default
is to fix $\tau=1$ and express prior scale through $\kappa$. If $\tau$ is
estimated, the prior scale must be interpreted relative to the fitted value.

### `gain_w` and `gain_b`

The gains parameterize prior standard deviations:

$$
\operatorname{sd}(W_{dk})=\frac{g_w}{\sqrt D},
\qquad
\operatorname{sd}(b_k)=\frac{g_b}{\sqrt D}.
$$

They influence all of the following simultaneously:

- prior predictive variance;
- update magnitude;
- posterior contraction;
- the final uncertainty scale.

They should not be described as ordinary learning rates.

### `agci_num_quad`

The quadrature count controls numerical accuracy of the one-dimensional event
integral. It does not define a different probabilistic model. It should be
raised only until probabilities and conditional moments are numerically
stable.

### `agci_class_chunk_size`

Original categorical AGCI needs conditional means for every possible class.
Its diagonal projection therefore costs $O(K^2)$ arithmetic per example. The
class chunk size bounds working memory by streaming candidate winning classes;
it does not change the model or result, apart from floating-point summation
order. The default chooses a chunk automatically from batch size, $K$, and the
quadrature count.

### Batch size and update cap

Sequential one-example ADF most closely matches the conceptual Bayesian
update. Mini-batches are a computational approximation. They should be
shuffled and class-representative so that one batch does not impose a highly
imbalanced shared update.

The update cap is a numerical safeguard. If it binds frequently, the resulting
state is farther from the stated moment-matching update and should be reported
as a capped approximation.

---

## 9. One pass, repeated passes, and Bayesian evidence

If dataset $\mathcal D$ is observed once, ordinary Bayes gives

$$
p(W,b\mid\mathcal D)
\propto
p(W,b)\,p(\mathcal D\mid W,b).
$$

Processing the same labeled examples for $E$ epochs instead corresponds
ideally to

$$
p_E(W,b\mid\mathcal D)
\propto
p(W,b)\,p(\mathcal D\mid W,b)^E.
$$

This is a power posterior. It is not the posterior for one observation of the
dataset.

AGCI/TAGI is not gradient descent that inherently requires many optimization
epochs. Each pass assimilates evidence. For the cleanest Bayesian
interpretation:

- initialize a prior once;
- shuffle the observations;
- assimilate each observation once;
- stop after one pass;
- use later passes only when deliberately adopting a tempered or
  generalized-Bayes interpretation.

Repeated passes can still be useful computationally, but they must not be
described as repeated refinement of the same ordinary posterior.

---

## 10. Predictive uncertainty: what it does and does not mean

For a test feature $h$, TAGI provides latent epistemic variances $v_k(h)$.
AGCI prediction then combines them with fixed decision noise:

$$
r_k^2(h)=v_k(h)+\tau^2.
$$

The resulting categorical probabilities integrate both sources inside the
argmax event. This is the appropriate posterior-predictive distribution under
the model.

However:

- $v_k(h)$ alone is not an OOD probability;
- mean latent variance is not guaranteed to rank classification errors;
- predictive entropy mixes class ambiguity, fixed decision noise, and
  parameter uncertainty;
- the model cannot represent “none of the above” because one of the $K$
  utilities must win;
- uncertainty is conditional on the frozen feature vector and ignores
  uncertainty in the feature extractor.

A separate feature-support or open-set model is required for a Bayesian claim
about whether an input belongs to the training domain.

---

## 11. Recommended specification

A simple, defensible from-scratch AGCI last layer is:

1. freeze the feature extractor and any feature transform;
2. compute training-only feature centering and energy statistics;
3. use zero parameter means;
4. fix $\tau=1$ as the utility unit;
5. choose a prior utility ratio $\kappa$ and derive $g_w$ from feature energy;
6. use zero bias means and a small, declared symmetric bias variance;
7. assimilate shuffled, representative data once;
8. use the same $\tau$ and quadrature rule for prediction;
9. report the diagonal-Gaussian and ADF approximations explicitly;
10. distinguish posterior predictive uncertainty from OOD evidence.

In compact form, the recommended prior is

$$
w_k\sim
\mathcal N\left(
0,
\frac{g_w^2}{D}I
\right),
\qquad
b_k\sim
\mathcal N\left(
0,
\frac{g_b^2}{D}
\right),
$$

where $g_w$ is selected through

$$
\frac{
g_w^2\,\mathbb E\lVert h\rVert^2/D+g_b^2/D
}{
\tau^2
}
=
\kappa^2.
$$

This separates three roles that are otherwise easy to conflate:

- $\tau$ defines the likelihood scale;
- $\kappa$ defines prior uncertainty relative to that likelihood;
- feature energy converts the dimensionless prior choice into an actual
  parameter variance.

That separation is the central initialization principle.

---

## 12. Claims that can be made carefully

With the specification above, it is reasonable to say:

> The classifier is a multinomial-probit Bayesian linear model on a frozen
> representation. Event AGCI moment-matches the output distribution
> conditioned on the observed winning-class event. TAGI retains its marginal
> variances and projects those moments to a diagonal Gaussian over last-layer
> parameters.

It is also reasonable to say that the model supplies posterior-predictive
probabilities and last-layer epistemic uncertainty.

It would be too strong to claim:

- exact Bayesian inference;
- a Bayesian posterior over the full neural network;
- automatically valid OOD uncertainty;
- an ordinary posterior after repeatedly reusing the training data;
- that a raw gain value transfers across unnormalized feature spaces.

Those boundaries make the theoretical contribution clearer rather than
weaker: the head is a computationally efficient approximate Bayesian
multiclass linear model whose assumptions, scale, and initialization can all
be stated explicitly.

---

## 13. Empirical summary so far

This section records only the findings that materially affect the theoretical
interpretation above. The comparisons use the same frozen features for AGCI
and the deterministic softmax control. They are single-seed results and should
be treated as strong diagnostics, not final multi-seed estimates.

The controlled historical CIFAR reruns in Section 13.1 compare full
class-averaged LMMSE-AGCI with Event AGCI. Their probability model and
conditional mean agree, but their covariance updates differ. The active
`agci` implementation now uses the Event-AGCI rule.

### 13.1 Full categorical AGCI: controlled one-pass rerun

The first full-AGCI experiment held the feature tensors, random seed, random
prior means, gain, $\tau$, quadrature rule, batch order, and batch size fixed.
Only the covariance update changed. Each example was assimilated once.

| Dataset and method | Top-1 % | NLL | ECE % | Brier | Pass wall time s |
| --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10, full AGCI | 94.94 | 0.17801 | 1.9219 | 0.07700 | 9.10 |
| CIFAR-10, predecessor event ADF | 94.94 | 0.17798 | 1.9204 | 0.07698 | 8.90 |
| CIFAR-10, softmax control | 95.00 | 0.19409 | 2.7783 | 0.07995 | -- |
| CIFAR-100, full AGCI | 76.50 | 0.93708 | 4.0786 | 0.33287 | 158.02 |
| CIFAR-100, predecessor event ADF | 76.54 | 0.93710 | 4.0923 | 0.33287 | 9.39 |
| CIFAR-100, softmax control | 76.67 | 0.95811 | 4.8210 | 0.33293 | -- |

At one-pass precision, the two covariance rules are empirically
indistinguishable on both frozen representations. Their mean update is the
same theoretically, and the small difference between the event-conditioned
and class-averaged covariance did not materially change the within-pass
trajectory. Full AGCI retained the proper-scoring and calibration improvements
over softmax, but it did not improve on the predecessor head.

The computational difference was substantial for $K=100$: CIFAR-100 full
AGCI was about 16.8 times slower for the first pass because it requires all 100
conditional mean vectors. CIFAR-10 showed almost no runtime penalty at $K=10$.
This is direct evidence of the $O(K^2)$ all-class-moment cost and is important
for deciding whether the original covariance is practical at ImageNet scale.

### 13.2 Historical clean classification

| Dataset and AGCI state | Top-1 % | NLL | ECE % | Softmax top-1 % | Softmax NLL | Softmax ECE % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10, gain 1, $\tau=1$, one pass | 94.94 | 0.1781 | 1.926 | 95.00 | 0.1941 | 2.778 |
| CIFAR-100, gain 1, $\tau=1$, selected pass 2 | 76.67 | 0.9310 | 3.023 | 76.67 | 0.9581 | 4.821 |
| ImageNet, zero mean, gain 1, $\tau=1.2825$, one pass | 68.64 | 1.3225 | 2.419 | 69.76 | 1.2469 | 2.633 |
| ImageNet, pretrained mean, $\tau=1.2825$, prior variance $\times 0.025$ | 69.96 | 1.4895 | 5.383 | 69.76 | 1.2469 | 2.633 |

On CIFAR-10, a single-pass gain-1 posterior retained essentially all baseline
accuracy while improving NLL, ECE, and Brier score. This is the cleanest
empirical example of the intended Bayesian use: one assimilation of the data
with no need for optimization-style repetition.

On CIFAR-100, the validation-selected gain-1 state at pass 2 exactly matched
the softmax top-1 accuracy, improved NLL and ECE, and slightly improved Brier
score and AURC. The second pass means this state is already better described
as a mild power posterior than as the ordinary one-pass posterior.

ImageNet exposed a sharper tradeoff. The zero-centered, class-symmetric prior
gave the best principled AGCI construction and slightly improved ECE, but lost
about 1.1 top-1 points and did not match softmax NLL. A pretrained-mean warm
start recovered and slightly exceeded softmax accuracy, but its worse NLL and
ECE showed that accuracy preservation alone did not make it a coherent or
well-calibrated posterior.

No ImageNet AGCI configuration obtained the softmax control's combination of
accuracy and proper-scoring quality simultaneously.

### 13.3 Findings that support the scaling theory

The clearest effective-evidence comparison occurred on CIFAR-100:

| Configuration | Validation accuracy % | NLL | Mean confidence % |
| --- | ---: | ---: | ---: |
| gain 1.0, pass 1 | 75.90 | 0.9466 | 73.22 |
| gain 0.1, pass 100 | 75.99 | 0.9564 | 73.05 |

These states have the same leading effective-time factor $E g^2$. Their nearly
identical accuracy and confidence support the interpretation that gain sets
prior covariance and evidence assimilation rate rather than acting as an
ordinary learning-rate knob.

The long gain-1 trajectory also showed why repeated passes require caution.
On CIFAR-100, validation accuracy remained essentially flat between the
selected state and pass 100, while NLL and ECE worsened as the approximate
posterior continued to contract. This is the expected behavior of repeatedly
raising the same likelihood to a higher power.

ImageNet added two initialization lessons:

- zero means removed arbitrary prior logits and improved the clean
  from-scratch AGCI state relative to random prior means;
- matching the Gaussian decision-noise scale to the pretrained softmax
  utility scale improved the link, but did not make a reused softmax
  classifier an automatically coherent Bayesian prior.

Training order was also decisive. Class-ordered ImageNet shards caused a
catastrophic sequential-update failure, whereas globally shuffled shards
restored normal learning. This is direct evidence that the ADF approximation
is order-sensitive and that representative batches are part of the inference
protocol, not merely an implementation detail.

### 13.4 Uncertainty result

The categorical predictive distribution was useful, but the native latent
variance was not a reliable standalone uncertainty score.

For SVHN detection, predictive entropy reached AUROC 0.915 on CIFAR-10 and
0.843 on CIFAR-100, close to the corresponding softmax entropy scores. In
contrast, mean native epistemic variance produced AUROC 0.119 and 0.126,
respectively: it ranked the semantic OOD examples in the wrong direction.

On ImageNet, native output variance also failed to separate classification
errors from correct predictions; error AUROC stayed between 0.413 and 0.423
across the valid AGCI configurations.

The current evidence therefore supports using the full AGCI
posterior-predictive class distribution for closed-set decisions. It does not
support interpreting mean latent variance as a generic error or OOD score.
That limitation is consistent with the theory: last-layer variance is
conditional on a fixed feature vector and contains no model of feature
support.

### 13.5 Decision-noise law: Gaussian versus Gumbel

Section 1.2 argued that the decision-noise law fixes the tail of the class
probability model and predicted that the logit tail should score better. The
prediction holds on ImageNet and fails to generalize. This section records what
replicated, and is deliberate about separating the two.

#### What replicated on ImageNet

Both links were swept over the prior ratio and each bracketed its own optimum:
Gaussian at $\kappa\approx0.5$, Gumbel at $\kappa\approx1.0$. The two links
prefer different priors, so a comparison at one shared $\kappa$ measures prior
mismatch as much as it measures the link and must not be used.

On the 48,000-image held-out split, with epochs selected only on the
2,000-image calibration cohort:

| head | link | selected by | ep | Top-1 % | NLL | ECE | ACE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `logit_site` | logit, prior mean | NLL | 3 | 69.575 | 1.2517 | 0.0292 | 0.0289 |
| `logit_site` | logit, prior mean | ACE | 1 | 69.148 | 1.2587 | 0.0185 | 0.0180 |
| `gumbel_agci` | logit, integrated | NLL | 3 | 69.625 | 1.2548 | 0.0290 | 0.0289 |
| `gumbel_agci` | logit, integrated | ACE | 1 | 69.129 | 1.2586 | 0.0131 | 0.0121 |
| `agci` | Gaussian | NLL | 2 | 69.431 | 1.2941 | 0.0296 | 0.0299 |
| `agci` | Gaussian | ACE | 7 | 69.423 | 1.3231 | 0.0131 | 0.0139 |
| softmax control | -- | -- | -- | 69.727 | 1.2477 | 0.0264 | 0.0263 |
| temperature-scaled softmax | -- | -- | -- | 69.727 | 1.2408 | 0.0186 | 0.0187 |

The held-out NLL deficit against the deterministic control falls from 0.0464 to
0.0040, about 91 per cent of it, and the top-1 deficit from 0.30 to 0.15 points.
The deficit is reduced, not eliminated: temperature-scaled softmax retains the
best held-out NLL, and the earlier cohort reading of a statistical tie with the
control did not survive the transfer. Cohort figures run optimistic for every
head, and slightly more so for the AGCI heads, which use the cohort to select an
epoch while the control has none to select: degradation from cohort to held-out
is 0.019 for the control against 0.023 to 0.024 for the AGCI heads.

Four-seed replication at $\kappa=1$ on the cohort, where the seed drives prior
draws, batch order and the antithetic decision-noise draws:

| head | n | NLL mean | NLL sd | Top-1 mean | selected epochs |
| --- | ---: | ---: | ---: | ---: | --- |
| `logit_site` | 4 | 1.2294 | 0.0012 | 69.72 | 3, 3, 2, 4 |
| `gumbel_agci` | 4 | 1.2320 | 0.0011 | 69.86 | 3, 2, 2, 2 |

The link effect is roughly 0.033 in cohort NLL against a seed standard
deviation near 0.0012, so it is not a sampling artifact. The choice between the
two logit variants is a genuine but much smaller effect, discussed below.

#### What did not generalize

Repeating the same protocol on frozen CIFAR features, sweeping $\kappa$ for
both links and replicating each optimum across four seeds, reverses the
conclusion once and then restores it:

| dataset | $K$ | error % | Gaussian $\kappa^\*$ | Gaussian NLL | Gumbel $\kappa^\*$ | Gumbel NLL | $\Delta$ | $t$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | 10 | 4.7 | 0.4 | 0.1753 | 0.4 | 0.1708 | +0.0046 | +79 |
| CIFAR-100 | 100 | 23.9 | 0.8 | 0.9223 | 1.5 | 0.9505 | -0.0282 | -75 |
| ImageNet | 1000 | 30.0 | 0.5 | 1.2628 | 1.0 | 1.2316 | +0.033 | +27 |

A positive $\Delta$ favors the logit link. **The CIFAR-10 row is superseded.**
Section 13.6 extends this grid downward and finds that both CIFAR-10 optima lie
below $\kappa=0.4$, at 0.08 and 0.10, where the sign reverses to $-0.0060$ in
favor of the Gaussian link. The CIFAR-100 and ImageNet rows are unaffected;
their optima are interior to the grids searched. Read the rest of this
subsection with the CIFAR-10 sign flipped.

CIFAR seed standard deviations are
0.0001 to 0.0007, an order of magnitude *below* ImageNet's, because those last
layers hold 5,120 and 51,200 weights against ImageNet's 512,000 and are
correspondingly better determined. Both CIFAR effects therefore replicate at
$|t|\approx75$ and neither can be dismissed as noise.

With the CIFAR-10 correction the logit link wins only on ImageNet and loses on
both CIFAR datasets, so the sign is monotone in $K$ over the three points
available. Three points do not establish a trend, and no mechanism offered here
predicts one; the operative conclusion is unchanged, that the sign must be
measured per representation.
The one qualitative difference worth following is that on ImageNet the two
$\kappa$ curves are near-parallel with the logit link uniformly better, whereas
on CIFAR-100 they cross, Gaussian dominating below $\kappa=1.0$ and Gumbel
above $\kappa=1.5$. That is a different shape, not merely a different sum, and
it points at something specific to that representation rather than a smooth
trend. It is a hypothesis, not a result.

#### The one consistent empirical pattern

Stratifying validation NLL by the rank the control softmax assigns the observed
class gives the same four signs on both datasets decomposed, with each link at
its own optimum:

| stratum | ImageNet $n$ | ImageNet net | CIFAR-100 $n$ | CIFAR-100 net |
| --- | ---: | ---: | ---: | ---: |
| rank 0 | 1411 | +0.0493 | 7636 | +0.0054 |
| rank 1--4 | 376 | -0.0362 | 1682 | -0.0380 |
| rank 5--24 | 141 | -0.0156 | 540 | -0.0021 |
| rank 25 and beyond | 72 | +0.0341 | 142 | +0.0073 |
| total | | +0.0316 | | -0.0274 |

A positive entry favors the logit link. In both cases the logit link is better
on the easy bulk and on the far tail, and the Gaussian link is better on
near-misses; only the magnitudes differ, overwhelmingly in the rank-0 stratum.
The net therefore depends on how a problem distributes mass across these
strata, which is measurable after the fact but was not predicted in advance
here.

Note what this decomposition does *not* say. The far tail is not where most of
the ImageNet gain lies: rank 0 contributes +0.0493 against the far tail's
+0.0341. The tail argument of Section 1.2 correctly identifies a constraint on
how large $\kappa$ the Gaussian link can use -- its ImageNet optimum sits at
$-2.81$ points of confidence gap, underconfident -- but the resulting cost is
paid mostly on easy examples, not on tail examples. An explanation built on the
tail alone attributes the effect to the wrong stratum.

#### Cost

The two links have the same order of cost for training, $O(QK)$ against
$O(MK)$, and differ only in prediction, where the Gaussian link pays $O(QK^2)$
for its per-candidate quadrature. Measured one-pass wall time as a ratio of
logit to Gaussian is 1.65 at $K=10$, 0.61 at $K=100$ and 0.23 at $K=1000$. The
logit link is the slower of the two at small $K$.

#### Choosing between the two logit variants

`gumbel_agci` integrates the link over the prior predictive by antithetic
sampling; `logit_site` takes the probabilities from the prior means alone and
applies an ordinary Gaussian site, which is the $S\to0$ limit of the former up
to the second-order difference between moment matching and precision addition.
The prior-mean variant is better on NLL by 0.0026 in cohort mean with
$t\approx3.3$, and is roughly four times cheaper. The integrated variant is
markedly better calibrated on its calibration-selected checkpoint, ACE 0.0121
against 0.0180 at indistinguishable NLL, and that 0.0121 is the best calibration
of anything tested, including temperature scaling's 0.0187.

The two are therefore specialized rather than ranked. Integrating over posterior
logit variance widens the predictive, which buys nothing when sharpness is the
objective and a third of the calibration error when it is not. Draw count is not
critical: 8, 32 and 128 draws give cohort NLL 1.2451, 1.2410 and 1.2397.

### 13.6 The Core-Tail link: separating link geometry from update dynamics

Section 13.5 leaves a specific question unanswered. The Gaussian link is better
on near-misses and the logit link is better on the easy bulk and the far tail,
consistently and on both datasets decomposed. Two explanations fit that equally
well. Either the Gaussian advantage is a property of the *static probability
shape*, in which case a link with Gaussian sensitivity near a tie and softmax
sensitivity elsewhere should capture it, or it is a property of the
*sequential event conditioning*, in which case no static reshaping will. The
Core-Tail link was constructed to separate the two, and the prediction was
registered before the measurement.

#### The construction

The link is the convex choice model

$$
\bm p^\star(\bm z)
=
\arg\max_{\bm p\in\Delta_C}
\Big\{\bm z^\mathsf T\bm p-\Omega_\star(\bm p)\Big\},
\qquad
\Omega_\star(\bm p)=\sum_k\Big[p_k\log p_k-a_\star p_k^2(1-p_k)^2\Big],
$$

whose interior correction vanishes identically at $p=0$ and $p=1$. Softmax is
the $a_\star=0$ member. Because the correction is supported strictly inside the
simplex, the link keeps the exact softmax tail including its leading constant,
$-\log p_c\to\Delta$, and modifies only the competitive interior.

The coefficient is derived rather than fitted. The Gumbel utility difference is
standard logistic with variance $\pi^2/3$, so the variance-matched Gaussian
decision model is $\Phi(\sqrt3 d/\pi)$ with slope $\sqrt3/(\pi\sqrt{2\pi})$ at a
tie, while the binary Core-Tail slope is $1/(4+2a_\star)$. Equating them gives

$$
a_\star=\tfrac12\Big(\tfrac{\pi\sqrt{2\pi}}{\sqrt3}-4\Big)=0.2732603854486113 .
$$

Stationarity is the fixed point $\bm p=\operatorname{softmax}(\bm z+2a_\star\bm
p\odot(1-\bm p)\odot(1-2\bm p))$, which a Picard iteration contracts at rate
$a_\star$ and solves below float32 resolution in six $O(C)$ softmaxes.
Differentiating it gives $\bm J=\operatorname{diag}(\bm w)-\bm w\bm w^\mathsf
T/W$ with $w_k=1/\phi''(p_k)$ and
$\phi''(p)=1/p-a_\star(2-12p+12p^2)\ge1-2a_\star>0$, hence the exact score and
the exact diagonal of the Fisher curvature $\bm J^\mathsf
T\operatorname{diag}(1/\bm p)\bm J$, both $O(C)$ and both reducing to $y-p$ and
$p(1-p)$ at $a_\star=0$. The site is formed at the prior means, so `ct_agci`
and `logit_site` differ in the link and in nothing else, and `ct_agci` at
$a_\star=0$ reproduces `logit_site` to four decimals at every prior tested.

#### A correction to the CIFAR-10 entry of Section 13.5

Rerunning the comparison with the prior grid extended downward shows that the
CIFAR-10 optima in Section 13.5 sit at the edge of the grid that was searched
rather than at a stationary point. Validation NLL keeps falling below
$\kappa=0.4$ for every link:

| $\kappa$ | `agci` | `gumbel_agci` | `logit_site` | `ct_agci` |
|---|---|---|---|---|
| 0.05 | 0.1800 | 0.2612 | 0.2612 | 0.2627 |
| 0.08 | **0.1539** | 0.1613 | 0.1613 | 0.1607 |
| 0.10 | 0.1545 | **0.1605** | **0.1606** | **0.1600** |
| 0.15 | 0.1557 | 0.1610 | 0.1610 | 0.1604 |
| 0.20 | 0.1601 | 0.1608 | 0.1609 | 0.1603 |
| 0.40 | 0.1753 | 0.1707 | 0.1713 | 0.1705 |
| 1.00 | 0.1997 | 0.1921 | 0.1949 | 0.1940 |

Every curve now turns inside the grid, and the recorded CIFAR-10 reading
reverses: at the true optima the Gaussian link wins by 0.0060 rather than
losing by 0.0046. This is a grid artifact of the earlier sweep, not a new
effect. It strengthens rather than weakens Section 13.5's conclusion that the
sign of the link effect is not predicted by any rule established here, since
the logit link now wins only on ImageNet. The CIFAR-10 row of the three-dataset
table should not be quoted without this correction. The CIFAR-100 grid is
unaffected: both optima there are interior and reproduce to four decimals.

#### What the Core-Tail link buys

Four seeds at each link's own optimum, validation-only. CIFAR-10:

| link | $\kappa$ | NLL mean | NLL sd | top-1 | ACE | conf gap pp |
|---|---|---|---|---|---|---|
| `agci` | 0.08 | 0.15398 | 0.000026 | 95.34% | 0.0070 | +0.61 |
| `gumbel_agci` | 0.10 | 0.16056 | 0.000018 | 95.34% | 0.0122 | +0.71 |
| `logit_site` | 0.10 | 0.16059 | 0.000018 | 95.34% | 0.0122 | +0.71 |
| `ct_agci` | 0.10 | 0.16001 | 0.000019 | 95.34% | 0.0111 | +0.63 |

and CIFAR-100:

| link | $\kappa$ | NLL mean | NLL sd | top-1 | ACE | ECE | Brier | conf gap pp |
|---|---|---|---|---|---|---|---|---|
| `agci` | 0.8 | 0.9223 | 0.00074 | 76.17% | 0.0252 | 0.0253 | 0.3304 | +1.58 |
| `gumbel_agci` | 1.5 | 0.9490 | 0.00143 | 76.16% | 0.0363 | 0.0370 | 0.3331 | +2.34 |
| `logit_site` | 0.8 | 0.9715 | 0.00034 | 76.22% | 0.0459 | 0.0467 | 0.3372 | +1.94 |
| `ct_agci` | 0.8 | 0.9687 | 0.00031 | 76.21% | 0.0428 | 0.0442 | 0.3364 | +1.82 |

Against its controlled reference the Core-Tail link improves every proper
scoring and calibration metric on both datasets and moves accuracy on neither:

| metric | CIFAR-10 $\Delta$ | $t$ | CIFAR-100 $\Delta$ | $t$ |
|---|---|---|---|---|
| NLL | $+0.00058$ | $+44.5$ | $+0.00274$ | $+11.9$ |
| top-1 % | $+0.000$ | -- | $+0.015$ | $+0.6$ |
| ACE | $+0.00108$ | $+201$ | $+0.00307$ | $+11.6$ |
| ECE | $+0.00087$ | $+9.3$ | $+0.00253$ | $+8.4$ |
| Brier | $+0.00019$ | $+93.0$ | $+0.00076$ | $+8.0$ |
| conf gap pp | $+0.084$ | $+1232$ | $+0.119$ | $+4.0$ |

Positive favors `ct_agci`. Accuracy is flat by construction: the correction is
monotone within a row, so it cannot reorder classes, and on CIFAR-10 all four
links return the identical prediction on all 10,000 examples. The confidence
gap shrinks on both datasets, which is the same effect as the NLL gain read
through a different statistic. A probit core flattens the map exactly where
classes compete, so contested examples become less confident while confident
ones are untouched because the correction vanishes as $p\to1$. On CIFAR-10 the
resulting ECE of 0.0069 is the best of the four links, better than the Gaussian
link's 0.0073, although the Gaussian link still wins ACE there and dominates
every calibration metric on CIFAR-100.

The cost is 1.4 times `logit_site` and a quarter of `agci`: five CIFAR-10
epochs take 0.7 s against 0.5 s, 2.6 s and 6.1 s for `logit_site`, `agci` and
`gumbel_agci`. On CIFAR-100 the Core-Tail link beats `logit_site` at every one
of the seven prior ratios swept, by 0.0027 to 0.0035.

The rank decomposition shows the gain arrives exactly where the design
predicted, with the same signature on both datasets. Positive favors `ct_agci`:

| stratum | CIFAR-10 $n$ | vs `logit_site` | CIFAR-100 $n$ | vs `logit_site` |
|---|---|---|---|---|
| rank 0 | 9539 | $-0.0008$ | 7636 | $-0.0007$ |
| rank 1--4 | 445 | $+0.0014$ | 1682 | $+0.0039$ |
| rank 5--24 | 16 | $-0.0000$ | 540 | $-0.0003$ |
| rank 25 and beyond | -- | -- | 142 | $-0.0002$ |
| total | 10000 | $+0.0006$ | 10000 | $+0.0027$ |

The entire gain is the near-miss stratum, the cost is a smaller charge on the
easy bulk, and the far tail is untouched, which is what a correction supported
strictly inside the simplex must do. CIFAR-100 carries 3.8 times the near-miss
mass of CIFAR-10, 16.8 per cent of examples against 4.5 per cent, and the
effect is 4.5 times larger. That scaling is the mechanism's own prediction and
it holds.

#### What it does not buy

The same decomposition falsifies the stronger hypothesis. The near-miss gap
between `logit_site` and `agci` is $0.0080$ on CIFAR-10 and $0.0603$ on
CIFAR-100. Matching the central slope recovers $0.0014$ and $0.0039$ of them,
18 and 6.5 per cent. The recovered fraction is small on both datasets and
*smaller* on the one where near-misses carry more mass. Four fifths to
nineteen twentieths of the Gaussian near-miss advantage survives a link whose
sensitivity at a tie is exactly the variance-matched Gaussian's.

A coefficient sweep sharpens this. Validation NLL is monotone decreasing in
$a_\star$ over the entire admissible range $[0,1/2)$, with no interior optimum
on CIFAR-10:

| $a_\star$ | $\kappa=0.08$ | 0.10 | 0.15 | 0.20 | 0.40 |
|---|---|---|---|---|---|
| 0.0000 | 0.1613 | 0.1606 | 0.1610 | 0.1609 | 0.1713 |
| 0.1000 | 0.1611 | 0.1603 | 0.1608 | 0.1607 | 0.1710 |
| 0.2000 | 0.1609 | 0.1601 | 0.1606 | 0.1605 | 0.1707 |
| 0.2733 (derived) | 0.1607 | 0.1600 | 0.1604 | 0.1603 | 0.1705 |
| 0.3500 | 0.1606 | 0.1598 | 0.1603 | 0.1602 | 0.1703 |
| 0.4500 | 0.1604 | **0.1596** | 0.1601 | 0.1600 | 0.1701 |

CIFAR-100 gives the same monotone picture:

| $a_\star$ | $\kappa=0.6$ | 0.8 | 1.0 | 1.5 |
|---|---|---|---|---|
| 0.0000 | 0.9719 | 0.9714 | 0.9721 | 0.9735 |
| 0.1000 | 0.9708 | 0.9704 | 0.9710 | 0.9725 |
| 0.2000 | 0.9698 | 0.9694 | 0.9700 | 0.9715 |
| 0.2733 (derived) | 0.9690 | 0.9687 | 0.9692 | 0.9708 |
| 0.3500 | 0.9683 | 0.9680 | 0.9685 | 0.9700 |
| 0.4500 | 0.9673 | **0.9670** | 0.9675 | 0.9691 |

The $a_\star=0$ row reproduces `logit_site` to four decimals at every prior on
both datasets, which is the intended exact reduction and an end-to-end check on
the implementation. The derived coefficient captures 60 per cent of the
improvement available inside the convex regime on CIFAR-10 and 61 per cent on
CIFAR-100, and the link stays numerically well behaved to $a_\star=0.499$, with
a stationarity residual of $5\times10^{-15}$, so the monotonicity is not the
solver degenerating. The direction of the variance-matching derivation is
right; its magnitude is not a stationary point of the objective. Even at the
boundary the link reaches 0.1596 and 0.9670, still 0.0056 and 0.0447 short of
the Gaussian link.

#### Reading

The proposal was posed as falsifiable and it falsified cleanly in one
direction. Static link geometry is a real but small part of the near-miss
effect. Reshaping the competitive core toward the probit reliably buys
near-miss NLL, monotonically in how much reshaping is applied, replicating far
outside seed noise, with the same signature on two datasets and a magnitude
that tracks near-miss mass. It is not the mechanism. Most of the Gaussian
near-miss advantage remains after the central slope is matched exactly, which
points at the sequential event conditioning rather than at the shape of the
probability map, and is consistent with Section 13.4's finding that what the
Gaussian link does to the posterior trajectory is not reproducible by any
prediction-time map.

What the link is good for is narrower and still real. It is a strict
improvement on `logit_site` on both datasets and at every prior ratio tested,
at 1.4 times its cost, with no new hyperparameter, exact reduction to the
current head at $a_\star=0$, and better calibration. Where the logit family is
the right family, it is the better member of it. Whether it is the right family
is still decided by sweeping against the Gaussian link, exactly as Section 13.5
concluded.

### 13.7 Current empirical reading

Taken together, the results support six restrained conclusions:

1. AGCI can preserve deterministic last-layer accuracy while improving proper
   scoring and calibration on frozen CIFAR representations.
2. Prior scale, likelihood scale, and repeated passes interact through an
   effective-evidence coordinate close to $E g^2/\tau^2$.
3. Zero-centered initialization is the coherent from-scratch construction;
   warm starts require both link-scale matching and an honest account of
   reused evidence. Under Gumbel noise the link scale needs no matching at
   all, which removes one of those two obstacles but not the other.
4. The Bayesian value currently lies in regularized posterior-predictive
   classification and sequential updating, not in treating posterior
   contraction or mean output variance as automatic OOD awareness.
5. The decision-noise law is a first-class modeling choice that the
   argmax-event derivation is indifferent to, and on ImageNet it governed the
   proper-scoring gap. Which law scores better is dataset-dependent and is not
   predicted by any rule established here.
6. The near-miss advantage of the Gaussian link is not a property of the
   static probability map. A link with the exact variance-matched probit slope
   at a tie and the exact softmax tail recovers only 6.5 to 18 per cent of it,
   with the smaller fraction on the dataset carrying more near-miss mass.

The scaling question this document previously left open -- whether a better
prior or a richer posterior covariance could close the remaining ImageNet
proper-scoring gap -- was posed about the wrong object. On ImageNet the gap was
governed by the likelihood, not the prior or the covariance: changing only the
decision-noise law removed about 91 per cent of the held-out deficit, with the
coarsest available posterior, a diagonal one-pass ADF state, and at lower cost.

Three things remain open, in descending order of how much they should temper
the result.

First, the direction does not generalize. Under the same protocol with both
links swept and four seeds each, the Gaussian link scores better on both CIFAR
datasets while the logit link wins on ImageNet. Section 13.6 revises the
CIFAR-10 entry: its optima lie below the grid Section 13.5 searched, and at the
true optima the Gaussian link wins there too, by 0.0060. Any recommendation to
prefer one link must currently be made per representation, by sweeping both
over a grid whose interior contains each optimum, and not by appeal to tail
shape.

Second, the ImageNet deficit is reduced rather than closed. A held-out NLL
deficit of 0.0040 and a top-1 deficit of 0.15 points against the softmax control
persist, and temperature-scaled softmax still holds the best held-out NLL.
Whether the residue is the accuracy deficit expressed in NLL, the diagonal
covariance, or the power-posterior contraction visible in the epoch trajectory
is not settled. The warm-start diagnostic indicates it is not intrinsic to the
head: from a prior mean that already carries the control's accuracy, a few logit
passes score strictly better than the control on NLL, Brier, top-5, ECE and ACE.
That diagnostic reuses evidence the backbone already absorbed and is not a
posterior; making it one requires the site-replacement treatment that
`run_full_covariance_adf.py` already applies to repeated passes.

Third, the native output variance remains a failed error and OOD score. Error
AUROC stayed near 0.387 for the logit head, indistinguishable from the Gaussian
head's 0.387 to 0.423. Changing the link did nothing here, exactly as
Section 13.4 predicts, because last-layer variance is conditional on a fixed
feature vector and contains no model of feature support.
