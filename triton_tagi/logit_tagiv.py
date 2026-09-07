"""Logit-space TAGI-V for a last layer trained on teacher logits.

The head regresses centered teacher logits over frozen features with an
input-dependent observation variance. Two affine streams share the interleaved
``2K`` layout the other TAGI-V heads use: even columns carry the latent logit
``Z_k``, odd columns carry a log-variance ``G_k`` whose
:class:`~triton_tagi.layers.EvenExp` image is ``X_k = exp(G_k)``. The positive
observation variance is ``S_k = s_min2 + X_k`` and the observation model is

    Y_k = Z_k + V_k,        V_k | S_k ~ N(0, S_k),

with the ``K`` logit dimensions treated independently, consistent with diagonal
TAGI. Training conditions ``[Z, V]`` on the observed logit, projects the
posterior residual square onto the AGVI moments of ``S``, and maps the ``S``
posterior back to the pre-activation ``G`` through the exact log-normal
cross-covariance ``Cov(G, S) = v_G * mu_X``.

Two deliberate differences from cuTAGI's ``update_delta_z_cuda_heteros``, ported
in :func:`triton_tagi.update.observation.compute_innovation`:

- the latent mean update divides by the total predictive variance
  ``v_Z + mu_S`` rather than by the epistemic variance alone, so the gain is the
  Kalman one and not cuTAGI's "overfit_mu" variant;
- the exponential carries a variance floor ``s_min2``, which keeps the AGVI
  prior variance ``3 v_S + 2 mu_S**2`` bounded away from zero.

Prediction integrates the epistemic logit variance and a post-hoc multiple of
the learned aleatoric variance through a tempered softmax, and splits the
predictive entropy into aleatoric and epistemic parts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

_EPS = 1e-12

#: Default variance floor ``s_min2`` added to ``exp(G)``.
LOGIT_VARIANCE_FLOOR = 1e-6


# ======================================================================
#  Training targets
# ======================================================================


def center_logits(logits: Tensor) -> Tensor:
    """Remove the common logit offset that a softmax cannot identify.

    Args:
        logits: Teacher logits, shape (..., K).

    Returns:
        Logits with a zero mean over the last dimension.
    """

    if logits.dim() < 1 or logits.shape[-1] < 2:
        raise ValueError("logits must have a class dimension of at least two")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite floating-point values")
    return logits - logits.mean(dim=-1, keepdim=True)


def logit_target_scale(targets: Tensor) -> float:
    """Return the root-mean-square scale of centered logit targets.

    The post-hoc temperature absorbs this normalization; dividing by it only
    makes the variance-head initialization dimensionless.

    Args:
        targets: Centered logits, shape (N, K).

    Returns:
        The scalar ``sqrt(mean(t**2))`` over all entries.
    """

    if targets.numel() == 0:
        raise ValueError("targets must be non-empty")
    scale = float(targets.detach().double().square().mean().sqrt())
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("logit targets must have a finite positive scale")
    return scale


def prepare_logit_targets(logits: Tensor, *, scale: float | None = None) -> tuple[Tensor, float]:
    """Center teacher logits and divide them by a global training scale.

    Args:
        logits: Teacher logits, shape (N, K).
        scale: Normalization computed on training data. ``None`` derives it
            from ``logits`` themselves, which is only correct for the training
            split.

    Returns:
        targets: Centered and normalized logits, shape (N, K).
        scale: The scale that was applied.
    """

    centered = center_logits(logits)
    resolved = logit_target_scale(centered) if scale is None else float(scale)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("scale must be finite and positive")
    return centered / resolved, resolved


# ======================================================================
#  Variance head: moments and initialization
# ======================================================================


def logit_variance_head_prior(
    *,
    aleatoric_init: float,
    variance_floor: float = LOGIT_VARIANCE_FLOOR,
    coefficient_of_variation: float = 0.5,
) -> tuple[float, float]:
    """Return the ``(mu_G, v_G)`` prior giving a target initial noise variance.

    With ``v_G = log(1 + c**2)`` the log-normal ``X = exp(G)`` has coefficient
    of variation ``c``, and ``mu_G = log(x0) - v_G / 2`` makes ``E[X] = x0``
    where ``x0 = aleatoric_init - variance_floor``. The prior variance must not
    collapse: ``v_G = 0`` gives ``v_S = 0`` and an AGVI gain of zero, which
    freezes the variance head.

    Args:
        aleatoric_init: Target initial ``E[S] = s_min2 + E[exp(G)]``.
        variance_floor: The floor ``s_min2``.
        coefficient_of_variation: Prior coefficient of variation of ``exp(G)``.

    Returns:
        mu_G: Prior mean of the log-variance pre-activation.
        v_G: Prior variance of the log-variance pre-activation.
    """

    if not math.isfinite(aleatoric_init) or not math.isfinite(variance_floor):
        raise ValueError("aleatoric_init and variance_floor must be finite")
    if variance_floor < 0.0:
        raise ValueError("variance_floor must be nonnegative")
    if aleatoric_init <= variance_floor:
        raise ValueError("aleatoric_init must exceed variance_floor")
    if not math.isfinite(coefficient_of_variation) or coefficient_of_variation <= 0.0:
        raise ValueError("coefficient_of_variation must be finite and positive")
    v_g = math.log1p(coefficient_of_variation**2)
    mu_g = math.log(aleatoric_init - variance_floor) - 0.5 * v_g
    return mu_g, v_g


def gaussian_shrinkage_(mean: Tensor, variance: Tensor, rate: float) -> None:
    """Multiply a Gaussian posterior by a zero-mean Gaussian prior, in place.

    The product of ``N(m, v)`` with ``exp(-rate * w**2 / 2)`` is Gaussian with

        v+ = (1 / v + rate)**-1 = v / (1 + rate * v),
        m+ = m / (1 + rate * v),

    so both moments scale by the same factor and their ratio ``m / v`` is exactly
    preserved: the mean can only shrink in proportion to the variance, and the
    variance decays as ``v_0 / (1 + rate * v_0 * steps)``. The regularizer
    therefore self-attenuates as the head becomes confident, which bounds how
    much spread control a single rate can deliver late in training.

    Applied to the variance head's
    weights after each batch this is a persistent shrinkage: an initialization
    prior is eventually overwhelmed by repeated evidence, whereas this is
    reapplied every step and therefore keeps bounding how much across-input
    spread the log-variance is allowed to develop. Leave the bias out of it, so
    the level stays free while the spread is controlled.

    The per-step factor is ``1 / (1 + rate * v)``, so a useful rate is of order
    ``1 / v``: below that the update pushing the weights back out each step
    dominates and the equilibrium barely moves.

    Args:
        mean: Weight means, modified in place.
        variance: Weight variances, modified in place.
        rate: The prior precision ``lambda_g``; zero is a no-op.
    """

    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError("shrinkage rate must be finite and nonnegative")
    if mean.shape != variance.shape:
        raise ValueError("weight moments must have matching shapes")
    if rate == 0.0:
        return
    scale = 1.0 / (1.0 + rate * variance)
    mean.mul_(scale)
    variance.mul_(scale)


def power_compress_variance(
    aleatoric: Tensor,
    *,
    gamma: float,
    reference: float,
    variance_floor: float = LOGIT_VARIANCE_FLOOR,
) -> Tensor:
    """Compress the spread of a learned variance while fixing one anchor.

    ``a_gamma = s_min2 + a_ref * ((a - s_min2) / a_ref)**gamma`` leaves
    ``a = a_ref`` unchanged and pulls everything else toward it, so ``gamma < 1``
    reduces ``sd(log a)`` by exactly that factor. The map is strictly
    increasing, so every ranking the variance induces is preserved: it can fix a
    level or a predictive integration but cannot manufacture a signal that the
    ordering does not already contain.

    Args:
        aleatoric: Learned variances, shape (..., K).
        gamma: Compression exponent in (0, 1]; one is the identity.
        reference: The anchor ``a_ref``, typically a calibration-set median.
        variance_floor: The floor ``s_min2``.

    Returns:
        The compressed variances, same shape.
    """

    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference must be finite and positive")
    if gamma == 1.0:
        return aleatoric
    excess = (aleatoric - variance_floor).clamp_min(0.0)
    return variance_floor + reference * (excess / reference).pow(gamma)


def replicate_log_variance_offset(repeats: int) -> float:
    """Return the bias of ``log R`` as an estimator of ``log S``.

    For ``M`` Gaussian replicates ``(M - 1) R / S`` is chi-square with
    ``k = M - 1`` degrees of freedom, so ``E[log R] = log S + psi(k / 2) -
    log(k / 2)``. Adding this offset back makes ``log R`` unbiased, which
    matters when the selection criterion is a squared error in log space.

    Args:
        repeats: The replicate count ``M``, at least two.

    Returns:
        The offset to add to ``log R``.
    """

    if repeats < 2:
        raise ValueError("repeats must be at least two")
    half = 0.5 * (repeats - 1)
    return math.log(half) - float(torch.special.digamma(torch.tensor(half)))


def replicate_log_variance_noise(repeats: int) -> float:
    """Return ``var(log R)``, the irreducible floor of a log-space fit.

    Args:
        repeats: The replicate count ``M``, at least two.

    Returns:
        The trigamma variance ``psi'((M - 1) / 2)``.
    """

    if repeats < 2:
        raise ValueError("repeats must be at least two")
    return float(torch.special.polygamma(1, torch.tensor(0.5 * (repeats - 1))))


def logit_feature_energy(features: Tensor) -> float:
    """Return the mean squared feature norm used to scale the variance prior.

    The log-variance pre-activation has prior variance
    ``v_G = Sw * sum_d h_d**2 + Sb``, so the weight variance that gives the
    features a chosen share of ``v_G`` depends on this energy. Compute it on the
    features as the head sees them, after any centering or rescaling.

    Args:
        features: Frozen features, shape (N, D).

    Returns:
        The scalar ``mean(sum_d h_d**2)``.
    """

    if features.dim() != 2 or features.shape[0] == 0:
        raise ValueError("features must have non-empty shape (samples, dimensions)")
    energy = float(features.detach().double().square().sum(dim=1).mean())
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError("features must have finite positive energy")
    return energy


def logit_variance_prior_split(
    prior_variance: float, *, feature_energy: float, weight_share: float
) -> tuple[float, float]:
    """Split the log-variance prior between the weights and the bias.

    A variance head whose weight prior is negligible against its bias prior can
    only learn a global noise level: the feature-dependent term
    ``Sw * sum_d h_d**2`` never moves. Giving the weights a share ``rho`` of
    ``v_G`` at the average feature energy is what makes the head
    heteroscedastic at all.

    Args:
        prior_variance: The target ``v_G`` at the average feature energy.
        feature_energy: ``mean(sum_d h_d**2)`` from :func:`logit_feature_energy`.
        weight_share: Fraction ``rho`` in [0, 1) carried by the weights; zero
            gives a head that starts, and stays, effectively homoscedastic.

    Returns:
        weight_variance: Per-weight prior variance for the log-variance stream.
        bias_variance: Prior variance of its bias.
    """

    if not math.isfinite(prior_variance) or prior_variance <= 0.0:
        raise ValueError("prior_variance must be finite and positive")
    if not math.isfinite(feature_energy) or feature_energy <= 0.0:
        raise ValueError("feature_energy must be finite and positive")
    if not math.isfinite(weight_share) or not 0.0 <= weight_share < 1.0:
        raise ValueError("weight_share must lie in [0, 1)")
    return weight_share * prior_variance / feature_energy, (1.0 - weight_share) * prior_variance


def logit_exp_variance_moments(
    mz: Tensor, Sz: Tensor, *, variance_floor: float = LOGIT_VARIANCE_FLOOR
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the moments of ``S = s_min2 + exp(G)`` for ``G ~ N(mz, Sz)``.

    The log-normal moments are exact::

        mu_X = exp(mz + Sz / 2)
        v_X  = mu_X**2 * expm1(Sz)
        Cov(G, S) = Sz * mu_X

    ``expm1`` avoids cancellation between two nearly equal exponentials when
    ``Sz`` is small.

    Args:
        mz: Pre-activation means of the log-variance head, shape (..., K).
        Sz: Pre-activation variances of the log-variance head, shape (..., K).
        variance_floor: The floor ``s_min2``.

    Returns:
        mu_S: Mean observation variance, shape (..., K).
        v_S: Variance of the observation variance, shape (..., K).
        cov_GS: Cross-covariance ``Cov(G, S)``, shape (..., K).
    """

    if mz.shape != Sz.shape:
        raise ValueError("log-variance moments must have matching shapes")
    if variance_floor < 0.0 or not math.isfinite(variance_floor):
        raise ValueError("variance_floor must be finite and nonnegative")
    variance = Sz.clamp_min(0.0)
    mu_x = torch.exp(mz + 0.5 * variance)
    v_x = mu_x.square() * torch.expm1(variance)
    return variance_floor + mu_x, v_x, variance * mu_x


def split_logit_tagiv_outputs(
    ma: Tensor, Sa: Tensor, *, variance_floor: float = LOGIT_VARIANCE_FLOOR
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split interleaved ``2K`` post-EvenExp moments into logit and noise parts.

    Args:
        ma: Output means, shape (..., 2K), layout ``[Z_0, X_0, Z_1, X_1, ...]``.
        Sa: Output variances, shape (..., 2K).
        variance_floor: The floor ``s_min2`` added to ``E[exp(G)]``.

    Returns:
        mu_Z: Latent logit means, shape (..., K).
        v_Z: Epistemic logit variances, shape (..., K).
        mu_S: Mean observation variances ``s_min2 + E[exp(G)]``, shape (..., K).
        v_S: Variances of the observation variances, shape (..., K).
    """

    if ma.shape != Sa.shape:
        raise ValueError("logit TAGI-V moments must have matching shapes")
    if ma.dim() < 1 or ma.shape[-1] % 2 != 0:
        raise ValueError("logit TAGI-V outputs must have an even width 2K")
    if variance_floor < 0.0 or not math.isfinite(variance_floor):
        raise ValueError("variance_floor must be finite and nonnegative")
    mu_z = ma[..., 0::2]
    v_z = Sa[..., 0::2].clamp_min(0.0)
    mu_s = variance_floor + ma[..., 1::2].clamp_min(0.0)
    v_s = Sa[..., 1::2].clamp_min(0.0)
    return mu_z, v_z, mu_s, v_s


# ======================================================================
#  Observation update
# ======================================================================


def compute_logit_tagiv_innovation(
    targets: Tensor,
    ma: Tensor,
    Sa: Tensor,
    *,
    variance_floor: float = LOGIT_VARIANCE_FLOOR,
    update_mean: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return the TAGI-V innovation for continuous logit observations.

    Conditioning ``[Z, V]`` on ``Y = y`` with ``q = v_Z + mu_S`` and
    ``e = y - mu_Z`` gives the standardized latent deltas ``e / q`` and
    ``-1 / q``, and the residual posterior ``mu_V = mu_S e / q``,
    ``v_V = mu_S v_Z / q``. The AGVI stage projects ``R = V**2`` onto ``S``
    using ``Cov(S, R) = v_S`` and ``Var(R) = 3 v_S + 2 mu_S**2``. Folding the
    AGVI gain ``J_S = v_S / Var(R)`` with the smoother gain
    ``J_G = Cov(G, S) / v_S`` cancels ``v_S`` exactly, leaving deltas that stay
    finite for a frozen variance head::

        delta_ma_G = mu_X * (mu_R_post - mu_S) / Var(R)
        delta_Sa_G = mu_X**2 * (v_R_post - Var(R)) / Var(R)**2

    Deltas are standardized as TAGI's backward recursion expects,
    ``(mu_post - mu) / v`` and ``(v_post - v) / v**2``, so :class:`EvenExp`
    stays an identity passthrough.

    Args:
        targets: Observed teacher logits, shape (..., K).
        ma: Output means, shape (..., 2K), post-:class:`EvenExp`.
        Sa: Output variances, shape (..., 2K).
        variance_floor: The floor ``s_min2``.
        update_mean: When false the latent stream's deltas are zeroed, which
            freezes the mean head while the variance head keeps learning. Use it
            after a mean warm-up so that the AGVI stage sees residuals around a
            converged mean rather than around an untrained one.

    Returns:
        delta_ma: Mean innovations, shape (..., 2K).
        delta_Sa: Variance innovations, shape (..., 2K).
    """

    mu_z, v_z, mu_s, v_s = split_logit_tagiv_outputs(ma, Sa, variance_floor=variance_floor)
    if targets.shape != mu_z.shape:
        raise ValueError("targets must have shape (..., K) matching the logit stream")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("logit targets must be finite")
    mu_x = (mu_s - variance_floor).clamp_min(0.0)

    # Stage 1 — condition [Z, V] on the observed logit.
    residual = targets.to(mu_z.dtype) - mu_z
    total_variance = (v_z + mu_s).clamp_min(_EPS)
    delta_mu_z = residual / total_variance
    delta_var_z = -1.0 / total_variance
    mu_v = mu_s * residual / total_variance
    var_v = (mu_s * v_z / total_variance).clamp_min(0.0)

    # Stage 2 — AGVI projection of the posterior residual square onto S.
    mu_r_post = mu_v.square() + var_v
    var_r_post = 2.0 * var_v.square() + 4.0 * var_v * mu_v.square()
    var_r = (3.0 * v_s + 2.0 * mu_s.square()).clamp_min(_EPS)

    # Stage 3 — map the S posterior back through the exponential. The AGVI and
    # smoother gains cancel v_S, so no division by the variance-head variance
    # survives.
    delta_mu_g = mu_x * (mu_r_post - mu_s) / var_r
    delta_var_g = mu_x.square() * (var_r_post - var_r) / var_r.square()

    delta_ma = torch.empty_like(ma)
    delta_Sa = torch.empty_like(Sa)
    delta_ma[..., 0::2] = delta_mu_z if update_mean else 0.0
    delta_Sa[..., 0::2] = delta_var_z if update_mean else 0.0
    delta_ma[..., 1::2] = delta_mu_g
    delta_Sa[..., 1::2] = delta_var_g
    return delta_ma, delta_Sa


def compute_logit_mean_innovation(
    targets: Tensor,
    ma: Tensor,
    Sa: Tensor,
    *,
    observation_variance: float,
) -> tuple[Tensor, Tensor]:
    """Return a fixed-noise innovation that trains the latent stream alone.

    This is the mean warm-up: the observation variance is a constant rather
    than a learned quantity, so an early residual on a large-magnitude target
    cannot be recorded as noise and cannot throttle its own channel's mean
    update through the ``1 / (v_Z + mu_S)`` gain.

    Args:
        targets: Observed logits, shape (..., K).
        ma: Output means, shape (..., 2K), post-:class:`EvenExp`.
        Sa: Output variances, shape (..., 2K).
        observation_variance: The fixed ``sigma_v**2``.

    Returns:
        delta_ma: Mean innovations, shape (..., 2K), zero on the variance stream.
        delta_Sa: Variance innovations, shape (..., 2K), zero on the variance stream.
    """

    if ma.shape != Sa.shape or ma.dim() < 1 or ma.shape[-1] % 2 != 0:
        raise ValueError("logit TAGI-V outputs must have a matching even width 2K")
    if not math.isfinite(observation_variance) or observation_variance <= 0.0:
        raise ValueError("observation_variance must be finite and positive")
    mu_z = ma[..., 0::2]
    v_z = Sa[..., 0::2].clamp_min(0.0)
    if targets.shape != mu_z.shape:
        raise ValueError("targets must have shape (..., K) matching the logit stream")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("logit targets must be finite")

    total_variance = (v_z + observation_variance).clamp_min(_EPS)
    delta_ma = torch.zeros_like(ma)
    delta_Sa = torch.zeros_like(Sa)
    delta_ma[..., 0::2] = (targets.to(mu_z.dtype) - mu_z) / total_variance
    delta_Sa[..., 0::2] = -1.0 / total_variance
    return delta_ma, delta_Sa


def logit_replicate_variance(repeated: Tensor, *, scale: float = 1.0) -> tuple[Tensor, Tensor]:
    """Return the replicate mean and unbiased sample variance of centered logits.

    Args:
        repeated: Teacher logits, shape (N, M, K).
        scale: Normalization applied to the centered logits.

    Returns:
        mean: The replicate mean ``ybar``, shape (N, K).
        variance: The unbiased sample variance ``R``, shape (N, K).
    """

    if repeated.dim() != 3 or repeated.shape[1] < 2:
        raise ValueError("repeated logits must have shape (samples, repeats >= 2, classes)")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    centered = (repeated - repeated.mean(dim=-1, keepdim=True)) / scale
    return centered.mean(dim=1), centered.var(dim=1, unbiased=True)


def compute_logit_replicate_innovation(
    residual_variance: Tensor,
    ma: Tensor,
    Sa: Tensor,
    *,
    repeats: int,
    variance_floor: float = LOGIT_VARIANCE_FLOOR,
) -> tuple[Tensor, Tensor]:
    """Return the AGVI innovation from a replicated sample variance.

    With ``M`` repeated observations of the same input the unbiased sample
    variance ``R`` is an observation of ``S`` that never passes through the
    latent mean, so a mean-head error cannot be recorded as noise. For Gaussian
    replicates ``E[R | S] = S`` and ``var(R | S) = c_M**2 S**2`` with
    ``c_M**2 = 2 / (M - 1)``, which gives the prior predictive moments

        mu_R = mu_S,
        v_R  = (1 + c_M**2) v_S + c_M**2 mu_S**2,
        Cov(S, R) = v_S,

    and the exact Kalman update ``mu_S+ = mu_S + J (R - mu_S)``,
    ``v_S+ = v_S - v_S**2 / v_R`` with ``J = v_S / v_R``. Composing that gain
    with the smoother gain ``Cov(G, S) / v_S`` cancels ``v_S`` exactly and
    leaves the standardized deltas

        delta_ma_G = mu_X * (R - mu_S) / v_R,
        delta_Sa_G = -mu_X**2 / v_R.

    The single-observation AGVI in :func:`compute_logit_tagiv_innovation` is the
    ``c_M**2 = 2`` case of the same algebra, differing only in that it must
    infer the residual from ``Y - Z`` instead of observing ``R`` directly. The
    variance update here always contracts, which the single-observation form
    does not.

    Args:
        residual_variance: Unbiased sample variance ``R`` over the repeats,
            shape (..., K).
        ma: Output means, shape (..., 2K), post-:class:`EvenExp`.
        Sa: Output variances, shape (..., 2K).
        repeats: The replicate count ``M``, at least two.
        variance_floor: The floor ``s_min2``.

    Returns:
        delta_ma: Mean innovations, shape (..., 2K), zero on the latent stream.
        delta_Sa: Variance innovations, shape (..., 2K), zero on the latent stream.
    """

    if repeats < 2:
        raise ValueError("replicate-aware AGVI needs at least two repeats")
    _, _, mu_s, v_s = split_logit_tagiv_outputs(ma, Sa, variance_floor=variance_floor)
    if residual_variance.shape != mu_s.shape:
        raise ValueError("residual_variance must have shape (..., K)")
    if not bool(torch.isfinite(residual_variance).all()):
        raise ValueError("residual_variance must be finite")
    if bool((residual_variance < 0.0).any()):
        raise ValueError("residual_variance must be nonnegative")

    mu_x = (mu_s - variance_floor).clamp_min(0.0)
    dispersion = 2.0 / (repeats - 1)
    var_r = ((1.0 + dispersion) * v_s + dispersion * mu_s.square()).clamp_min(_EPS)

    delta_ma = torch.zeros_like(ma)
    delta_Sa = torch.zeros_like(Sa)
    delta_ma[..., 1::2] = mu_x * (residual_variance.to(mu_s.dtype) - mu_s) / var_r
    delta_Sa[..., 1::2] = -mu_x.square() / var_r
    return delta_ma, delta_Sa


# ======================================================================
#  Prediction and calibration
# ======================================================================


def standard_normal_base_samples(
    num_samples: int,
    num_classes: int,
    *,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    method: str = "sobol",
) -> Tensor:
    """Return fixed standard-normal draws shared by every calibration candidate.

    A scrambled Sobol sequence mapped through the normal quantile keeps the
    temperature sweep deterministic and lowers the Monte Carlo error relative to
    plain sampling at the same budget.

    Args:
        num_samples: Number of draws ``M``.
        num_classes: Sample dimension ``K``.
        seed: Sobol scrambling or generator seed.
        device: Device of the returned tensor.
        dtype: Floating-point dtype of the returned tensor.
        method: ``"sobol"`` or ``"gaussian"``.

    Returns:
        Standard-normal draws, shape (M, K).
    """

    if num_samples < 2 or num_classes < 2:
        raise ValueError("num_samples and num_classes must both be at least two")
    if method == "sobol":
        engine = torch.quasirandom.SobolEngine(num_classes, scramble=True, seed=seed)
        uniform = engine.draw(num_samples, dtype=torch.float64)
        samples = torch.special.ndtri(uniform.clamp(1e-12, 1.0 - 1e-12))
    elif method == "gaussian":
        generator = torch.Generator().manual_seed(seed)
        samples = torch.randn(num_samples, num_classes, generator=generator, dtype=torch.float64)
    else:
        raise ValueError("method must be 'sobol' or 'gaussian'")
    return samples.to(device=device, dtype=dtype)


def _validate_predictive_moments(
    mu_Z: Tensor, epistemic_var: Tensor, aleatoric_var: Tensor
) -> None:
    if mu_Z.dim() != 2:
        raise ValueError("predictive moments must have shape (samples, classes)")
    if mu_Z.shape != epistemic_var.shape or mu_Z.shape != aleatoric_var.shape:
        raise ValueError("predictive moments must have matching shapes")
    if not bool(torch.isfinite(mu_Z).all()):
        raise ValueError("logit means must be finite")


def _predictive_scales(
    epistemic_var: Tensor,
    aleatoric_var: Tensor,
    alpha: float,
    epistemic_scale: float,
) -> Tensor:
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    if not math.isfinite(epistemic_scale) or epistemic_scale < 0.0:
        raise ValueError("epistemic_scale must be finite and nonnegative")
    total = epistemic_scale * epistemic_var.clamp_min(0.0)
    total = total + alpha * aleatoric_var.clamp_min(0.0)
    return total.clamp_min(0.0).sqrt()


def logit_tagiv_predictive_probs(
    mu_Z: Tensor,
    epistemic_var: Tensor,
    aleatoric_var: Tensor,
    *,
    temperature: float = 1.0,
    alpha: float = 1.0,
    epistemic_scale: float = 1.0,
    base_samples: Tensor | None = None,
    num_samples: int = 256,
    seed: int = 0,
    chunk_size: int = 4096,
) -> Tensor:
    """Return ``E[softmax(L / T)]`` for ``L ~ N(mu, v_e + alpha * a)``.

    Args:
        mu_Z: Latent logit means, shape (N, K).
        epistemic_var: Epistemic logit variances, shape (N, K).
        aleatoric_var: Learned aleatoric logit variances, shape (N, K).
        temperature: Softmax temperature ``T > 0``.
        alpha: Aleatoric multiplier ``alpha >= 0``.
        epistemic_scale: Multiplier on the epistemic variance, one by default.
        base_samples: Fixed standard-normal draws, shape (M, K). Built from
            ``num_samples`` and ``seed`` when omitted.
        num_samples: Draw count used when ``base_samples`` is omitted.
        seed: Seed used when ``base_samples`` is omitted.
        chunk_size: Rows evaluated per block.

    Returns:
        Predictive class probabilities, shape (N, K).
    """

    _validate_predictive_moments(mu_Z, epistemic_var, aleatoric_var)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    num_classes = mu_Z.shape[-1]
    if base_samples is None:
        base_samples = standard_normal_base_samples(
            num_samples,
            num_classes,
            seed=seed,
            device=mu_Z.device,
            dtype=mu_Z.dtype,
        )
    elif base_samples.dim() != 2 or base_samples.shape[-1] != num_classes:
        raise ValueError("base_samples must have shape (samples, classes)")
    draws = base_samples.to(device=mu_Z.device, dtype=mu_Z.dtype)
    scale = _predictive_scales(epistemic_var, aleatoric_var, alpha, epistemic_scale)

    parts = []
    for start in range(0, mu_Z.shape[0], chunk_size):
        stop = start + chunk_size
        perturbed = mu_Z[start:stop, None, :] + scale[start:stop, None, :] * draws
        parts.append(torch.softmax(perturbed / temperature, dim=-1).mean(dim=1))
    return torch.cat(parts) if len(parts) > 1 else parts[0]


@dataclass(frozen=True)
class LogitUncertainty:
    """Predictive probabilities with their entropy decomposition."""

    probabilities: Tensor
    total_entropy: Tensor
    aleatoric_entropy: Tensor
    epistemic_entropy: Tensor


def _entropy(probabilities: Tensor) -> Tensor:
    return -(probabilities * probabilities.clamp_min(_EPS).log()).sum(dim=-1)


def logit_tagiv_uncertainty(
    mu_Z: Tensor,
    epistemic_var: Tensor,
    aleatoric_var: Tensor,
    *,
    temperature: float = 1.0,
    alpha: float = 1.0,
    epistemic_scale: float = 1.0,
    epistemic_samples: int = 64,
    aleatoric_samples: int = 64,
    seed: int = 0,
    chunk_size: int = 256,
) -> LogitUncertainty:
    """Split the predictive entropy into aleatoric and epistemic parts.

    Each epistemic logit draw ``Z ~ N(mu, epistemic_scale * v_e)`` is integrated
    over the aleatoric noise ``N(0, alpha * a)`` to give a per-draw simplex point
    ``pi``. The mean of ``H(pi)`` is the aleatoric term and the mutual
    information ``H(mean pi) - mean H(pi)`` is the epistemic term.

    Args:
        mu_Z: Latent logit means, shape (N, K).
        epistemic_var: Epistemic logit variances, shape (N, K).
        aleatoric_var: Learned aleatoric logit variances, shape (N, K).
        temperature: Softmax temperature ``T > 0``.
        alpha: Aleatoric multiplier ``alpha >= 0``.
        epistemic_scale: Multiplier on the epistemic variance.
        epistemic_samples: Outer draw count ``M``.
        aleatoric_samples: Inner draw count per outer draw.
        seed: Seed for both fixed Sobol sets.
        chunk_size: Rows evaluated per block.

    Returns:
        A :class:`LogitUncertainty` whose entropies each have shape (N,).
    """

    _validate_predictive_moments(mu_Z, epistemic_var, aleatoric_var)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    num_classes = mu_Z.shape[-1]
    outer = standard_normal_base_samples(
        epistemic_samples,
        num_classes,
        seed=seed,
        device=mu_Z.device,
        dtype=mu_Z.dtype,
    )
    inner = standard_normal_base_samples(
        aleatoric_samples,
        num_classes,
        seed=seed + 1,
        device=mu_Z.device,
        dtype=mu_Z.dtype,
    )
    epistemic_sd = _predictive_scales(
        epistemic_var, torch.zeros_like(aleatoric_var), 0.0, epistemic_scale
    )
    aleatoric_sd = _predictive_scales(torch.zeros_like(epistemic_var), aleatoric_var, alpha, 0.0)

    probability_parts, total_parts, aleatoric_parts = [], [], []
    for start in range(0, mu_Z.shape[0], chunk_size):
        stop = start + chunk_size
        latent = mu_Z[start:stop, None, :] + epistemic_sd[start:stop, None, :] * outer
        perturbed = latent[:, :, None, :] + aleatoric_sd[start:stop, None, None, :] * inner
        conditional = torch.softmax(perturbed / temperature, dim=-1).mean(dim=2)
        mean_probabilities = conditional.mean(dim=1)
        probability_parts.append(mean_probabilities)
        total_parts.append(_entropy(mean_probabilities))
        aleatoric_parts.append(_entropy(conditional).mean(dim=1))
    probabilities = torch.cat(probability_parts)
    total = torch.cat(total_parts)
    aleatoric = torch.cat(aleatoric_parts)
    return LogitUncertainty(probabilities, total, aleatoric, total - aleatoric)


@dataclass(frozen=True)
class LogitCalibration:
    """Post-hoc temperature and aleatoric multiplier selected by validation NLL."""

    temperature: float
    alpha: float
    nll: float


def logit_tagiv_predictive_nll(
    mu_Z: Tensor,
    epistemic_var: Tensor,
    aleatoric_var: Tensor,
    labels: Tensor,
    *,
    temperature: float = 1.0,
    alpha: float = 1.0,
    epistemic_scale: float = 1.0,
    base_samples: Tensor | None = None,
    num_samples: int = 256,
    seed: int = 0,
    chunk_size: int = 4096,
) -> float:
    """Return the mean negative log predictive probability of the labels."""

    if labels.dim() != 1 or labels.shape[0] != mu_Z.shape[0]:
        raise ValueError("labels must have shape (samples,)")
    probabilities = logit_tagiv_predictive_probs(
        mu_Z,
        epistemic_var,
        aleatoric_var,
        temperature=temperature,
        alpha=alpha,
        epistemic_scale=epistemic_scale,
        base_samples=base_samples,
        num_samples=num_samples,
        seed=seed,
        chunk_size=chunk_size,
    )
    index = labels.to(device=probabilities.device).long().unsqueeze(1)
    if bool(((index < 0) | (index >= probabilities.shape[1])).any()):
        raise ValueError("labels contain an invalid class index")
    selected = probabilities.gather(1, index).squeeze(1).clamp_min(_EPS)
    return float(-selected.log().mean())


def _golden_section(objective, lower: float, upper: float, iterations: int) -> tuple[float, float]:
    """Minimize a unimodal scalar objective on a bracket without derivatives."""

    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    loss_left, loss_right = objective(left), objective(right)
    for _ in range(iterations):
        if loss_left <= loss_right:
            upper, right, loss_right = right, left, loss_left
            left = upper - ratio * (upper - lower)
            loss_left = objective(left)
        else:
            lower, left, loss_left = left, right, loss_right
            right = lower + ratio * (upper - lower)
            loss_right = objective(right)
    best = 0.5 * (lower + upper)
    return best, objective(best)


def fit_logit_calibration(
    mu_Z: Tensor,
    epistemic_var: Tensor,
    aleatoric_var: Tensor,
    labels: Tensor,
    *,
    mode: str = "joint",
    alpha: float = 1.0,
    alpha_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0),
    alpha_bounds: tuple[float, float] = (0.0, 16.0),
    log_temperature_bounds: tuple[float, float] = (-3.0, 3.0),
    epistemic_scale: float = 1.0,
    iterations: int = 32,
    rounds: int = 2,
    base_samples: Tensor | None = None,
    num_samples: int = 256,
    seed: int = 0,
    chunk_size: int = 4096,
) -> LogitCalibration:
    """Select ``(T, alpha)`` by minimizing validation NLL on shared draws.

    ``mode="temperature"`` holds the aleatoric multiplier at ``alpha`` and fits
    only the temperature, the one-parameter baseline. ``mode="joint"`` screens
    ``alpha_grid`` — which should contain ``0`` so that discarding the learned
    aleatoric variance stays a candidate — and then alternates golden-section
    searches over ``alpha`` and ``log T``. Every candidate reuses the same base
    samples, so the sweep is deterministic.

    Args:
        mu_Z: Latent logit means, shape (N, K).
        epistemic_var: Epistemic logit variances, shape (N, K).
        aleatoric_var: Learned aleatoric logit variances, shape (N, K).
        labels: Calibration labels, shape (N,).
        mode: ``"temperature"`` or ``"joint"``.
        alpha: Fixed multiplier for ``mode="temperature"`` and the starting
            point of the joint search.
        alpha_grid: Multipliers screened before the joint refinement.
        alpha_bounds: Bracket for the joint ``alpha`` search. The upper end is
            deliberately loose: a head whose residuals nearly vanish learns a
            tiny aleatoric variance that the multiplier has to scale up before
            it can contribute.
        log_temperature_bounds: Bracket for the ``log T`` search.
        epistemic_scale: Multiplier on the epistemic variance, held fixed.
        iterations: Golden-section iterations per one-dimensional search.
        rounds: Coordinate-descent rounds after the grid screen.
        base_samples: Fixed standard-normal draws shared by all candidates.
        num_samples: Draw count used when ``base_samples`` is omitted.
        seed: Seed used when ``base_samples`` is omitted.
        chunk_size: Rows evaluated per block.

    Returns:
        The selected :class:`LogitCalibration`.
    """

    _validate_predictive_moments(mu_Z, epistemic_var, aleatoric_var)
    if mode not in {"temperature", "joint"}:
        raise ValueError("mode must be 'temperature' or 'joint'")
    if iterations < 1 or rounds < 1:
        raise ValueError("iterations and rounds must be positive")
    lower_log, upper_log = log_temperature_bounds
    if not math.isfinite(lower_log) or not math.isfinite(upper_log) or lower_log >= upper_log:
        raise ValueError("log_temperature_bounds must be finite and increasing")
    lower_alpha, upper_alpha = alpha_bounds
    if not math.isfinite(lower_alpha) or not math.isfinite(upper_alpha):
        raise ValueError("alpha_bounds must be finite")
    if lower_alpha < 0.0 or lower_alpha >= upper_alpha:
        raise ValueError("alpha_bounds must be nonnegative and increasing")
    if base_samples is None:
        base_samples = standard_normal_base_samples(
            num_samples,
            mu_Z.shape[-1],
            seed=seed,
            device=mu_Z.device,
            dtype=mu_Z.dtype,
        )

    def nll(temperature: float, multiplier: float) -> float:
        return logit_tagiv_predictive_nll(
            mu_Z,
            epistemic_var,
            aleatoric_var,
            labels,
            temperature=temperature,
            alpha=multiplier,
            epistemic_scale=epistemic_scale,
            base_samples=base_samples,
            chunk_size=chunk_size,
        )

    def fit_temperature(multiplier: float) -> tuple[float, float]:
        log_temperature, loss = _golden_section(
            lambda value: nll(math.exp(value), multiplier),
            lower_log,
            upper_log,
            iterations,
        )
        return math.exp(log_temperature), loss

    if mode == "temperature":
        temperature, loss = fit_temperature(alpha)
        return LogitCalibration(temperature, float(alpha), loss)

    candidates = tuple(dict.fromkeys((*alpha_grid, float(alpha))))
    best_alpha, best_temperature, best_loss = float(alpha), 1.0, math.inf
    for candidate in candidates:
        if not math.isfinite(candidate) or candidate < 0.0:
            raise ValueError("alpha_grid entries must be finite and nonnegative")
        temperature, loss = fit_temperature(candidate)
        if loss < best_loss:
            best_alpha, best_temperature, best_loss = candidate, temperature, loss
    for _ in range(rounds):
        multiplier, loss = _golden_section(
            lambda value, fixed=best_temperature: nll(fixed, value),
            lower_alpha,
            upper_alpha,
            iterations,
        )
        if loss < best_loss:
            best_alpha, best_loss = multiplier, loss
        temperature, loss = fit_temperature(best_alpha)
        if loss < best_loss:
            best_temperature, best_loss = temperature, loss
    return LogitCalibration(best_temperature, best_alpha, best_loss)
