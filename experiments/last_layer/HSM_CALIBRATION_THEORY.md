# Hierarchical probit calibration with a positive uncertain gain

Implementation notes for `triton_tagi/hsm_calibration.py`, which follows
[HSM_calibration_Goulet_Nguyen_Florensa_annotated.tex](HSM_calibration_Goulet_Nguyen_Florensa_annotated.tex).
This document records the notation mapping into the existing code, what is
computed exactly, and where the remaining approximations are.

## Model

On the proper K-leaf tree of `class_to_obs_full`, the frozen head supplies a
Gaussian state `Z_n ~ N(mu_n, v_n)` at every internal node and a deterministic
branch probit `o_n = Phi^-1(pi_n)`. The head also fixes the noise `sigma_v` of
the branch channel it was trained against. Calibration adds one Gaussian
log-gain per sharing group,

    L_r ~ N(lambda_r, q_r),    G_r = exp(L_r) > 0,
    R_n = G_{r(n)} Z_n + sigma_v o_n + eps_n,    eps_n ~ N(0, sigma_v^2),
    P_{c,n} = Pr(s_{c,n} R_n >= 0)
            = Phi(s_{c,n} (G_{r(n)} Z_n / sigma_v + o_n)),
    Q_c = prod_{n in path(c)} P_{c,n},

with `s_{c,n} = +1` on the left branch and `-1` on the right. Positivity is
structural, so no posterior draw reverses the two branches. Because every node
splits its class set exhaustively, `sum_c Q_c = 1` holds pointwise for every
realization of the states and the gains, and no categorical normalizer is
required. The padded tree of `class_to_obs` discards leaves that hold
probability mass and is rejected by the module.

`sigma_v` is fixed and is never inferred; only `L` is. It is carried on
`LogGainPosterior` rather than passed at each call, so a gain fitted against one
channel cannot be read out against another, and `TAGILastLayerClassifier`
refuses a belief whose channel disagrees with the head's.

## The two heads are two channels

    head          channel               lambda = 0, q = 0 reads out as
    hrc           sigma_v as trained    Phi(mu_n / sqrt(sigma_v^2 + v_n))
    hrc_probit    1                     Phi(mu_n / sqrt(1 + v_n))

The base HRC head of `hrc_softmax` observes `+/-1` at each node under a fixed
observation noise, so that noise *is* the channel and `lambda = 0` is the head
exactly as trained. `obs_to_class_probs(ma, Sa, hrc, alpha)` is the same readout
at `alpha = 1 / sigma_v`, and the tests pin the agreement at 1e-15 in float64.

cuTAGI hard-codes `alpha = 3` in that readout whatever noise it trained with,
which is a channel of `1/3` rather than the `0.3` the CIFAR-10 head used. That
is a different model, not a different parameterization of one: the two readouts
differ in class probability, and calibrating either recovers essentially the
same effective scale, since the gain absorbs the convention. `--latent-scale`
in `run_hsm_calibration.py` selects between them and both are recorded.

## Notation against `hrc_probit` and the reparameterization

The gain is the reciprocal of the latent probit scale that `hrc_probit`
carries, measured against the channel,

    G = sigma_v / alpha = sigma_v / tau,    L = log sigma_v - log tau.

The offset is a probit and enters `R_n` scaled by `sigma_v`, so it is
deliberately gain-independent in the numerator of

    a_n(l) = (g(l) mu_n + o_n) / d_n(l),
    g(l)   = exp(l) / sigma_v,
    d_n(l) = sqrt(1 + g(l)^2 v_n),

which at `sigma_v = 1` is `hrc_probit`'s `gamma = (mu + tau o) / sqrt(S +
tau^2)` after multiplying through by `exp(l)`. A zero network output therefore
telescopes to the class prior at any gain and any channel.

Only the ratio `g(l)` enters, so a belief at `sigma_v` is the exact
reparameterization `lambda -> lambda - log sigma_v` of the unit-scale belief:
the channel fixes the anchor, meaning which gain counts as the head as trained
and where the prior is centred, not the family of reachable models. Two things
follow, and both are tested. Re-running the CIFAR-10 base HRC sweep at
`--latent-scale convention` reproduces the records made before the channel
existed to 1e-14, with every fitted log-gain shifted by exactly `log 3`. And
the useful consequence is that `prior_mean = 0` now means the same thing for
both heads, so a group with no calibration data falls back on the head as
trained instead of on a convention carried in the prior mean.

## What is exact

The conditional probability moments are closed forms, not approximations. For
`X ~ N(m, S)` with `a = m / sqrt(1 + S)` and `b = 1 / sqrt(1 + 2 S)`,

    E[Phi(X)]   = Phi(a),
    Var[Phi(X)] = Phi(a) (1 - Phi(a)) - 2 T(a, b),
    Cov(X, Phi(X)) = S phi(a) / sqrt(1 + S),

with `T` Owen's function. The substitution `t = tan(theta)` removes the
`1 / (1 + t^2)` factor from both `T` and the equivalent positive integral for
the variance,

    Var[Phi(X)] = (1 / pi) int_{arctan b}^{pi/4} exp(-a^2 / (2 cos^2 theta)) d theta,

leaving a smooth bounded integrand on a finite interval. The module evaluates
the variance through that positive form, so no two nearly equal numbers are
subtracted, and it consumes the Gauss--Legendre nodes one at a time so that no
tensor of shape `(batch, nodes, order, quadrature)` is ever materialized.
Convergence is reached by order 24 across `|a| <= 20` and `S` in
`[10^-2, 10^8]`; the default order is 48.

The log-gain is integrated by Gauss--Hermite and never folded into the state by
a product-Gaussian approximation. Node moments are

    mu_P = E_L[Phi(a_n(L))],
    v_P  = E_L[Var(U_n | L)] + Var_L(E[U_n | L]),
    Cov(Z_n, U_n) = v_n E_L[g(L) phi(a_n(L)) / d_n(L)],
    Cov(L, U_n)   = E_L[(L - lambda) Phi(a_n(L))],

where the variance is assembled from two sums of nonnegative terms rather than
from `E[U^2] - E[U]^2`. Class moments split on the tree geometry rather than on
the name of the grouping. When no class path visits a group twice, which covers
per-node and per-level gains, the path factors are independent and the products

    mu_Q = prod_n mu_{P_n},
    v_Q  = prod_n (v_{P_n} + mu_{P_n}^2) - prod_n mu_{P_n}^2

are exact. A single global gain makes the factors independent only
conditionally on `L`, so every class moment is a one-dimensional quadrature of
a conditional product; assembling it from marginal node moments would be wrong,
and the module raises rather than doing so for any grouping that repeats within
a path without being global.

`hsm_cross_class_covariance` gives the joint dispersion under per-node or
global gains. Its zero row sums, which follow from `sum_c Q_c = 1` pointwise,
are the sharpest available check on the whole moment machinery and hold to
2e-16 in the tests.

## What is approximate

Three things, all labelled.

1. The states are Gaussian and diagonal within a layer. Those are TAGI
   approximations, inherited unchanged.
2. The calibration module conditions on the frozen forward summaries and treats
   distinct calibration visits as conditionally independent. This is a modular
   or cut approximation, not a joint re-inference of the network posterior.
3. Reading the fitted belief back as a Gaussian is a posterior approximation,
   whether it comes from Laplace or from repeated assumed-density updates. The
   normalized grid fit avoids the Gaussian assumption on the way in but still
   reports two moments.

The Gaussian surrogate `Y = P + E` with `E ~ N(0, R)` and
`R = mu_P (1 - mu_P) - v_P` matches the unconditional mean, variance, and
cross-covariance of the Bernoulli channel, and nothing else. It is not
conditionally equivalent to `B | P ~ Bernoulli(P)`.

## Calibration

The observation is the binary event "this branch was taken", never `P`, `L`,
`G`, or `alpha`. It is the same semantic label the network trained on, expressed
at a tree node, so it belongs on a disjoint calibration split with the network
frozen. Under sign orientation the observed indicator is always one.

Every group is scalar, so the batch fit is order-independent and is the
default. For group `r` with visits `V_r`,

    log p(l_r | D_cal) = -(l_r - lambda_0)^2 / (2 q_0)
                         + sum_{(i,n) in V_r} log Phi(a_{i,n}(l_r; s_{i,n})) + const.

For a global gain that sum is exactly the categorical log-likelihood in `l`,
because conditionally on `l` the path factors are independent and
`E[Q_c | l] = prod_n m_n(l)`. For per-node gains it is the factorization the
independent gains imply. The fit normalizes this on a grid, which needs two
stages: a coarse sweep covering the prior locates the mode, and refined windows
scaled to the local curvature resolve it. A single fixed grid is not enough.
With 10,000 calibration examples the global posterior has `sqrt(q) ~ 0.012`,
below the spacing of any grid wide enough to cover a `q_0 = 4` prior, and an
unrefined grid then reports a spuriously certain gain. The refinement also
keeps a data-free group at its stated prior instead of a truncated version of
it. On a uniform grid the trapezoid error for a near-Gaussian posterior falls
off like `exp(-2 pi^2 q / h^2)`, so one refinement is enough and two is the
default.

`calibrate_hsm_log_gain_adf` is the sequential alternative for a genuine
stream. The mean update `lambda+ = lambda + Cov(L, P) / mu_P` is the exact
one-step Bernoulli identity; the variance update is exact only after averaging
over both labels, so repeating it is assumed-density filtering and its ordering
sensitivity has to be measured, not assumed. The variance decrement is capped
at a fraction of `q` rather than floored at zero, which would freeze the belief.

## Weak identifiability is not just a small-sample effect

The visit count is not sufficient to declare a gain identified. When every
visit to a node agrees, the log-likelihood in `l` is monotone increasing,
because a larger gain sharpens the probit toward whichever branch was taken.
There is then no interior mode: the posterior is the prior tilted, `lambda`
drifts up without bound as visits accumulate, and `q` decays far more slowly
than the Fisher rate. Under a `q_0 = 4` prior, five agreeing visits give
`lambda = 1.27` with `sqrt(q) = 1.46`, and two hundred give `lambda = 2.51`
with `sqrt(q) = 1.05`. Forty times the data buys a 28 percent reduction in the
posterior deviation.

This is the mechanism behind the collapse of the point readouts at small
calibration sizes on CIFAR-10. A deep node with fifteen agreeing visits is
assigned `exp(lambda) ~ 4`, and reading that mean alone as a temperature
multiplies the head's confidence fourfold on the strength of fifteen
observations. Carrying `q` into prediction absorbs it: the same fit costs 0.02
nats over the uncalibrated head instead of 0.17. The visits per group must therefore be reported together
with the posterior deviation, and neither alone is a sufficient diagnostic.

## Interpretation

`v_Q` is the epistemic dispersion of a latent class probability. The variance of
the observed one-versus-rest indicator is `mu_Q (1 - mu_Q)`, and the two are
related by

    mu_Q (1 - mu_Q) = E[Q (1 - Q)] + v_Q,

the same total-variance decomposition as the branch channel. The experiment
reports `v_Q / [mu_Q (1 - mu_Q)]` at the predicted leaf as the epistemic share.
These answer different questions and are not interchangeable.

The method is hierarchical probit temperature calibration. It is not Platt
scaling, which carries an intercept and is usually written with a logistic
link. A global positive gain is the closest analogue of ordinary temperature
scaling; per-level and per-node gains are more flexible and need more
calibration data. The tree itself is part of the statistical problem: different
labellings expose different binary subtasks with different sample sizes at their
internal nodes, so the labelling and the sharing structure are reported with
every result.
