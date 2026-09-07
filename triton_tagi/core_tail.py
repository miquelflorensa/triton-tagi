"""The Core-Tail categorical link and its TAGI site update (CT-AGCI).

The Gaussian and Gumbel decision-noise links disagree in two separate places.
Near a tie the Gaussian link is the more sensitive of the two, and on the far
tail the Gumbel link decays linearly in the utility margin while the Gaussian
link decays quadratically.  Section 13.5 of
``experiments/last_layer/AGCI_LAST_LAYER_THEORY.md`` measures both effects on
the same features: the logit link is uniformly better on the easy bulk and on
the far tail, and uniformly worse on near-misses.  Neither link is better in
both regimes, so the choice between them is a dataset-dependent compromise.

The Core-Tail link removes the compromise by construction.  It is the convex
choice model

    p*(z) = argmax_{p in simplex} { z' p - Omega*(p) } ,

    Omega*(p) = sum_k [ p_k log p_k - a* p_k^2 (1 - p_k)^2 ] ,

whose regularizer is Shannon negative entropy plus an interior correction that
vanishes identically at ``p = 0`` and ``p = 1``.  Softmax is the ``a* = 0``
member of the family.  Because the correction is supported strictly inside the
simplex, the link keeps the exact softmax tail, including its leading constant,
and modifies only the competitive interior.

The coefficient is not fitted.  Requiring that the binary Core-Tail link have
the same slope at a tie as a Gaussian decision model of matched noise variance
gives

    1 / (4 + 2 a*) = sqrt(3) / (pi sqrt(2 pi))   =>
    a* = (pi sqrt(2 pi) / sqrt(3) - 4) / 2 ,

since the Gumbel utility difference is standard logistic with variance
``pi^2 / 3``.  So the link has probit sensitivity in the core, softmax
sensitivity in the tail, and zero new tunable hyperparameters.

Stationarity of the convex program is the fixed point

    p = softmax(z + 2 a* p (1 - p) (1 - 2 p)) ,

which a plain Picard iteration solves to machine precision in a handful of
``O(C)`` softmaxes.  Differentiating stationarity gives the probability
Jacobian in closed form,

    J = diag(w) - w w' / W ,   w_k = 1 / phi''(p_k) ,   W = sum_j w_j ,

with ``phi''(p) = 1/p - a* (2 - 12 p + 12 p^2) >= 1 - 2 a* > 0`` on ``(0, 1]``,
so ``Omega*`` is strictly convex and ``p*`` is unique.  The categorical score
and the diagonal of the Fisher curvature ``J' diag(1/p) J`` follow, both in
``O(C)``, and both reduce exactly to the Gumbel site's ``y - p`` and
``p (1 - p)`` at ``a* = 0``.

Everything here is written in terms of

    u_k = 1 / (1 - a* p_k (2 - 12 p_k + 12 p_k^2)) ,   w_k = p_k u_k ,

rather than in terms of ``1 / p_k`` directly.  The denominator is bounded below
by ``1 - 2 a* ~ 0.4535`` on the whole simplex, so ``u`` is bounded in
``[1, 2.205]`` and no quantity in the update ever divides by a probability.
That matters at ``C = 1000``, where trailing probabilities underflow.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# a* = (pi sqrt(2 pi) / sqrt(3) - 4) / 2, the unique coefficient that matches
# the binary Core-Tail slope at a tie to the variance-matched probit slope.
A_STAR = 0.5 * (math.pi * math.sqrt(2.0 * math.pi) / math.sqrt(3.0) - 4.0)

# Picard iterations for the stationarity fixed point.  The contraction factor
# is at most ``a* ~ 0.273``, so the residual falls by roughly one decade per
# iteration; six iterations sit below float32 resolution for every logit scale
# measured in ``tests/unit/test_core_tail.py``.
DEFAULT_NUM_ITERATIONS = 6

_EPS = 1e-8


def _validate_logits(logits: Tensor, a_star: float, num_iterations: int) -> None:
    if logits.dim() != 2:
        raise ValueError("the Core-Tail link expects two-dimensional logits")
    if logits.shape[-1] < 2:
        raise ValueError("the Core-Tail link requires at least two classes")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating point")
    if not math.isfinite(a_star) or a_star < 0.0 or a_star >= 0.5:
        # phi'' >= 1 - 2 a_star on (0, 1], so a_star < 1/2 is exactly the
        # condition under which Omega* is strictly convex.
        raise ValueError("a_star must lie in [0, 0.5) for a strictly convex link")
    if num_iterations < 1:
        raise ValueError("num_iterations must be positive")


def _validate_labels(labels: Tensor, logits: Tensor) -> Tensor:
    num_classes = logits.shape[-1]
    labels = labels.to(device=logits.device, dtype=torch.long).reshape(-1)
    if labels.numel() != logits.shape[0]:
        raise ValueError("labels leading shape must match the logits")
    if bool(((labels < 0) | (labels >= num_classes)).any()):
        raise ValueError("labels contain an invalid class index")
    return labels


def core_tail_log_probabilities(
    logits: Tensor,
    *,
    a_star: float = A_STAR,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
) -> Tensor:
    """Return ``log p*(logits)`` by Picard iteration on the fixed point."""

    _validate_logits(logits, a_star, num_iterations)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    for _ in range(num_iterations):
        probabilities = log_probabilities.exp()
        correction = (
            2.0
            * a_star
            * probabilities
            * (1.0 - probabilities)
            * (1.0 - 2.0 * probabilities)
        )
        log_probabilities = torch.log_softmax(logits + correction, dim=-1)
    return log_probabilities


def core_tail_probabilities(
    logits: Tensor,
    *,
    a_star: float = A_STAR,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
) -> Tensor:
    """Return the Core-Tail probabilities ``p*(logits)``."""

    return core_tail_log_probabilities(
        logits, a_star=a_star, num_iterations=num_iterations
    ).exp()


def core_tail_stationarity_residual(
    logits: Tensor,
    probabilities: Tensor,
    *,
    a_star: float = A_STAR,
) -> Tensor:
    """Return the per-example sup-norm violation of the fixed point.

    Stationarity holds up to a shared additive constant per row, so the
    residual is measured after removing each row's mean.
    """

    correction = (
        2.0 * a_star * probabilities * (1.0 - probabilities) * (1.0 - 2.0 * probabilities)
    )
    residual = probabilities.clamp_min(1e-300).log() - (logits + correction)
    residual = residual - residual.mean(dim=-1, keepdim=True)
    return residual.abs().amax(dim=-1)


def _inverse_curvature_ratio(probabilities: Tensor, a_star: float) -> Tensor:
    """Return ``u = 1 / (p phi''(p)) = 1 / (1 - a* p (2 - 12 p + 12 p^2))``.

    This is the only place the interior correction enters the site, and it is
    bounded in ``[1, 1 / (1 - 2 a*)]`` because ``p (2 - 12 p + 12 p^2) <= 2``
    on ``[0, 1]``.  Writing ``w = p u`` therefore keeps every downstream
    quantity free of division by a probability.
    """

    denominator = 1.0 - a_star * probabilities * (
        2.0 - 12.0 * probabilities + 12.0 * probabilities.square()
    )
    return 1.0 / denominator.clamp_min(1.0 - 2.0 * a_star)


def core_tail_jacobian(
    probabilities: Tensor,
    *,
    a_star: float = A_STAR,
) -> Tensor:
    """Return the dense probability Jacobian ``dp / dz = diag(w) - w w' / W``."""

    weights = probabilities * _inverse_curvature_ratio(probabilities, a_star)
    total = weights.sum(dim=-1, keepdim=True)
    return torch.diag_embed(weights) - weights.unsqueeze(-1) * weights.unsqueeze(
        -2
    ) / total.unsqueeze(-1)


def core_tail_score_and_curvature(
    labels: Tensor,
    probabilities: Tensor,
    *,
    a_star: float = A_STAR,
) -> tuple[Tensor, Tensor]:
    """Return ``d log p_y / dz`` and the diagonal of ``J' diag(1/p) J``.

    Both are exact for the Core-Tail link and cost ``O(C)``.  At ``a_star = 0``
    they collapse to the Gumbel site's ``y - p`` and ``p (1 - p)``.
    """

    labels = _validate_labels(labels, probabilities)
    rows = torch.arange(probabilities.shape[0], device=probabilities.device)

    ratio = _inverse_curvature_ratio(probabilities, a_star)
    weights = probabilities * ratio
    total = weights.sum(dim=-1, keepdim=True)
    share = weights / total

    # g_k = (w_y / p_y) (1[k == y] - w_k / W), and w_y / p_y is exactly u_y.
    score = -share * ratio[rows, labels].unsqueeze(-1)
    score[rows, labels] += ratio[rows, labels]

    # h_k = (w_k^2 / p_k) (1 - 2 w_k / W) + (w_k^2 / W^2) sum_j w_j^2 / p_j,
    # with w_k^2 / p_k = p_k u_k^2.
    scaled = probabilities * ratio.square()
    curvature = scaled * (1.0 - 2.0 * share) + share.square() * scaled.sum(
        dim=-1, keepdim=True
    )
    # The exact value is a weighted sum of squares and cannot be negative; the
    # clamp only absorbs cancellation in the two-term rearrangement above.
    return score, curvature.clamp_min(0.0)


def compute_core_tail_site_innovation(
    labels: Tensor,
    output_mean: Tensor,
    output_variance: Tensor,
    *,
    beta: float = 1.0,
    a_star: float = A_STAR,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
) -> tuple[Tensor, Tensor]:
    """Project the Core-Tail categorical likelihood to TAGI innovations.

    The site is formed at the prior means, exactly as
    :func:`triton_tagi.gumbel_agci.compute_logit_site_innovation` forms the
    logit site, so that a comparison between the two isolates the link from
    the integration over the prior predictive.  With
    ``p = p*(mu / beta)``, the score ``g`` and the diagonal Fisher curvature
    ``h`` give the Gaussian site

        v_post = 1 / (1 / v + h) ,   mu_post = mu + v_post g ,

    returned in the normalized convention the TAGI backward pass consumes,
    ``(mu_post - mu) / v`` and ``(v_post - v) / v**2``.

    ``a_star = 0`` reproduces :func:`compute_logit_site_innovation` exactly.
    """

    if output_mean.shape != output_variance.shape:
        raise ValueError("the Core-Tail site expects matching diagonal output moments")
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    _validate_logits(output_mean, a_star, num_iterations)
    labels = _validate_labels(labels, output_mean)

    probabilities = core_tail_probabilities(
        output_mean / beta, a_star=a_star, num_iterations=num_iterations
    )
    score, curvature = core_tail_score_and_curvature(
        labels, probabilities, a_star=a_star
    )
    score = score / beta
    curvature = curvature / beta**2

    prior_variance = output_variance.clamp_min(_EPS)
    posterior_variance = 1.0 / (1.0 / prior_variance + curvature)
    active = output_variance > _EPS
    zero = torch.zeros((), device=output_mean.device, dtype=output_mean.dtype)
    delta_mean = torch.where(active, posterior_variance * score / prior_variance, zero)
    delta_variance = torch.where(
        active,
        (posterior_variance - prior_variance) / prior_variance.square(),
        zero,
    )
    return delta_mean, delta_variance


def core_tail_predictive_probs(
    output_mean: Tensor,
    *,
    beta: float = 1.0,
    a_star: float = A_STAR,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
) -> Tensor:
    """Return ``p*(mu / beta)``, ignoring posterior logit variance."""

    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    return core_tail_probabilities(
        output_mean / beta, a_star=a_star, num_iterations=num_iterations
    )
