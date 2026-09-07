"""AGCI with Gumbel decision noise: the argmax-event logit link.

AGCI conditions Gaussian utilities ``Z`` on the observed argmax event
``argmax(Z + E) == y``.  The law of the i.i.d. decision noise ``E`` is a
modeling choice that fixes the tail of the class-probability model.  With
``E_i ~ N(0, tau**2)`` the event log-probability of a class whose utility
trails the winner by ``Delta`` decays like ``-Delta**2 / (2 tau**2)``.  With
``E_i ~ Gumbel(0, beta)`` it decays like ``-Delta / beta``, and the event
probability conditional on ``Z`` is available in closed form:

    P(argmax(Z + E) == c | Z) = softmax(Z / beta)_c .

Gumbel noise therefore makes AGCI an exact Bayesian multinomial logit over the
utility posterior, with the same argmax-event derivation and the same
declared-scale role for ``beta`` that ``tau`` plays in the Gaussian case.

Only the log evidence

    log M(mu, S) = log E_{Z ~ N(mu, S)}[ softmax(Z / beta)_y ]

is needed.  For a Gaussian prior with diagonal ``S``, Bonnet's and Price's
theorems make the moment-matched (ADF) posterior exact in terms of its
derivatives,

    E[Z_i | y]   = mu_i + S_i  * d log M / d mu_i ,
    Var[Z_i | y] = S_i + S_i**2 * d^2 log M / d mu_i**2 ,

so the TAGI innovations the last layer consumes are precisely the first and
second derivatives of the log evidence with respect to the output mean.  Both
are unbiased self-normalized Monte Carlo averages over reparameterized prior
samples, using analytic softmax derivatives

    d p_y / d Z_i     = beta**-1  * p_y * a_i ,          a_i = [i == y] - p_i ,
    d^2 p_y / d Z_i^2 = beta**-2 * p_y * (a_i**2 - p_i (1 - p_i)) .

Sampling is antithetic and driven by an explicit seed, so every result is
reproducible.  Cost is ``O(num_samples * K)`` per example, without the
``O(num_quad * K)`` competitor quadrature the Gaussian link requires.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_EPS = 1e-8


def _validate(mean: Tensor, variance: Tensor, beta: float, num_samples: int) -> None:
    if mean.shape != variance.shape:
        raise ValueError("Gumbel AGCI expects matching diagonal output moments")
    if mean.dim() != 2:
        raise ValueError("Gumbel AGCI expects two-dimensional output moments")
    if mean.shape[-1] < 2:
        raise ValueError("multiclass AGCI requires at least two utilities")
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    if num_samples < 2 or num_samples % 2 != 0:
        raise ValueError("num_samples must be an even integer of at least two")


def _antithetic_normal(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    pair_index: int,
) -> Tensor:
    """Return one reproducible standard-normal draw for an antithetic pair."""

    generator = torch.Generator(device=device)
    # Deriving each pair's stream from (seed, pair_index) keeps the draws
    # independent of batch traversal order and of how many pairs are requested.
    generator.manual_seed((int(seed) * 1_000_003 + int(pair_index)) % (2**63 - 1))
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def gumbel_agci_log_evidence_derivatives(
    labels: Tensor,
    output_mean: Tensor,
    output_variance: Tensor,
    *,
    beta: float = 1.0,
    num_samples: int = 32,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``log M`` and its first two derivatives in the output mean."""

    _validate(output_mean, output_variance, beta, num_samples)
    num_classes = output_mean.shape[-1]
    labels = labels.to(device=output_mean.device, dtype=torch.long).reshape(-1)
    if labels.numel() != output_mean.shape[0]:
        raise ValueError("labels leading shape must match output moments")
    if bool(((labels < 0) | (labels >= num_classes)).any()):
        raise ValueError("labels contain an invalid class index")

    scale = output_variance.clamp_min(0.0).sqrt()
    rows = torch.arange(output_mean.shape[0], device=output_mean.device)
    num_pairs = num_samples // 2
    shape = tuple(output_mean.shape)
    signs = (1.0, -1.0)

    # Pass one: the per-sample log likelihood only, which is all the
    # self-normalized weights need and is cheap to recompute.
    log_weights = torch.empty(
        output_mean.shape[0],
        num_samples,
        device=output_mean.device,
        dtype=output_mean.dtype,
    )
    for pair_index in range(num_pairs):
        noise = _antithetic_normal(
            shape,
            device=output_mean.device,
            dtype=output_mean.dtype,
            seed=seed,
            pair_index=pair_index,
        )
        for sign_index, sign in enumerate(signs):
            utility = (output_mean + sign * scale * noise) / beta
            log_weights[:, 2 * pair_index + sign_index] = utility[
                rows, labels
            ] - torch.logsumexp(utility, dim=-1)

    log_evidence = torch.logsumexp(log_weights, dim=1) - math.log(num_samples)
    normalized = torch.softmax(log_weights, dim=1)

    # Pass two: accumulate the weighted analytic softmax derivatives.  The
    # same seed reproduces the identical draws, so nothing per-sample and
    # class-shaped has to be held in memory.
    first = torch.zeros_like(output_mean)
    second = torch.zeros_like(output_mean)
    for pair_index in range(num_pairs):
        noise = _antithetic_normal(
            shape,
            device=output_mean.device,
            dtype=output_mean.dtype,
            seed=seed,
            pair_index=pair_index,
        )
        for sign_index, sign in enumerate(signs):
            weight = normalized[:, 2 * pair_index + sign_index].unsqueeze(-1)
            utility = (output_mean + sign * scale * noise) / beta
            probabilities = torch.softmax(utility, dim=-1)
            residual = -probabilities
            residual[rows, labels] += 1.0
            first += weight * residual
            second += weight * (
                residual.square() - probabilities * (1.0 - probabilities)
            )

    first = first / beta
    # d^2 log M / d mu^2 = E_w[d^2 p / d Z^2] / p - (d log M / d mu)^2
    second = second / beta**2 - first.square()
    return log_evidence, first, second


def compute_gumbel_agci_innovation(
    labels: Tensor,
    output_mean: Tensor,
    output_variance: Tensor,
    *,
    beta: float = 1.0,
    num_samples: int = 32,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Project the Gumbel-noise argmax event to TAGI output innovations.

    The returned pair is ``(delta_mean, delta_variance)`` in the same
    normalized convention as :func:`triton_tagi.agci.compute_agci_innovation`:
    ``delta_mean = (posterior_mean - prior_mean) / S`` and
    ``delta_variance = (posterior_variance - S) / S**2``.
    """

    _, first, second = gumbel_agci_log_evidence_derivatives(
        labels,
        output_mean,
        output_variance,
        beta=beta,
        num_samples=num_samples,
        seed=seed,
    )
    active = output_variance > _EPS
    zero = torch.zeros((), device=output_mean.device, dtype=output_mean.dtype)
    delta_mean = torch.where(active, first, zero)
    # Price's identity can overshoot under Monte Carlo error; clamping keeps
    # the implied posterior variance positive without biasing the mean update.
    floor = -1.0 / output_variance.clamp_min(_EPS)
    delta_variance = torch.where(active, second.clamp_min(floor), zero)
    return delta_mean, delta_variance


def gumbel_agci_predictive_probs(
    output_mean: Tensor,
    output_variance: Tensor,
    *,
    beta: float = 1.0,
    num_samples: int = 32,
    seed: int = 0,
) -> Tensor:
    """Return ``E_{Z ~ N(mu, S)}[softmax(Z / beta)]`` by antithetic sampling."""

    _validate(output_mean, output_variance, beta, num_samples)
    scale = output_variance.clamp_min(0.0).sqrt()
    total = torch.zeros_like(output_mean)
    for pair_index in range(num_samples // 2):
        noise = _antithetic_normal(
            tuple(output_mean.shape),
            device=output_mean.device,
            dtype=output_mean.dtype,
            seed=seed,
            pair_index=pair_index,
        )
        for sign in (1.0, -1.0):
            total += torch.softmax((output_mean + sign * scale * noise) / beta, dim=-1)
    return total / num_samples


def compute_logit_site_innovation(
    labels: Tensor,
    output_mean: Tensor,
    output_variance: Tensor,
    *,
    beta: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Minimal Gumbel-AGCI site update from the prior means alone.

    This is the smallest change that swaps the Gaussian event link for the
    logit link. It deliberately omits the integration over the prior
    predictive, taking the class probabilities from the prior means,

        p = softmax(mu / beta),

    and forming the categorical score and diagonal curvature

        s = (y - p) / beta,
        h = p (1 - p) / beta**2 .

    The output moments are then updated by adding the site precision,

        v_post = 1 / (1 / v + h),
        mu_post = mu + v_post * s,

    which is the ordinary Gaussian-site form rather than the moment-matched
    form of :func:`compute_gumbel_agci_innovation`. The two agree to first
    order in the output variance and differ at second order, so this function
    isolates the effect of the likelihood tail from the effect of integrating
    the link over the parameter posterior.

    The returned pair is ``(delta_mean, delta_variance)`` in the normalized
    convention the TAGI backward pass consumes: ``(mu_post - mu) / v`` and
    ``(v_post - v) / v**2``.
    """

    if output_mean.shape != output_variance.shape:
        raise ValueError("Gumbel AGCI expects matching diagonal output moments")
    if output_mean.dim() != 2:
        raise ValueError("Gumbel AGCI expects two-dimensional output moments")
    num_classes = output_mean.shape[-1]
    if num_classes < 2:
        raise ValueError("multiclass AGCI requires at least two utilities")
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    labels = labels.to(device=output_mean.device, dtype=torch.long).reshape(-1)
    if labels.numel() != output_mean.shape[0]:
        raise ValueError("labels leading shape must match output moments")
    if bool(((labels < 0) | (labels >= num_classes)).any()):
        raise ValueError("labels contain an invalid class index")

    log_probabilities = torch.log_softmax(output_mean / beta, dim=-1)
    probabilities = log_probabilities.exp()
    score = -probabilities
    score[torch.arange(output_mean.shape[0], device=output_mean.device), labels] += 1.0
    score = score / beta
    curvature = probabilities * (1.0 - probabilities) / beta**2

    prior_variance = output_variance.clamp_min(_EPS)
    posterior_variance = 1.0 / (1.0 / prior_variance + curvature)
    delta_mean_absolute = posterior_variance * score
    active = output_variance > _EPS
    zero = torch.zeros((), device=output_mean.device, dtype=output_mean.dtype)
    delta_mean = torch.where(active, delta_mean_absolute / prior_variance, zero)
    delta_variance = torch.where(
        active,
        (posterior_variance - prior_variance) / prior_variance.square(),
        zero,
    )
    return delta_mean, delta_variance


def logit_predictive_probs(output_mean: Tensor, *, beta: float = 1.0) -> Tensor:
    """Return ``softmax(mu / beta)``, ignoring posterior logit variance."""

    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    return torch.softmax(output_mean / beta, dim=-1)
