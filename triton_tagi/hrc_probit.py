"""Hierarchical-probit classification for TAGI.

The observation model has no free scale. Each decision node carries a latent
variable with unit probit noise,

    Z_j | x, D ~ N(mu_j, S_j),   R_j = Z_j + o_j + eps_j,   eps_j ~ N(0, 1),

so for branch sign ``s_cj`` in {-1, +1} the marginal branch probability and the
class probability are::

    p(s_cj | x, D) = Phi( s_cj (mu_j + o_j) / sqrt(S_j + 1) )
    p(c    | x, D) = prod_{j in path(c)} p(s_cj | x, D)

On the full K-leaf tree of :func:`triton_tagi.hrc_softmax.class_to_obs_full`
these sum to one with no normalization, and the negative log-likelihood is
ordinary categorical cross-entropy, directly comparable with a softmax
classifier in nats.

The unit noise variance is an identifiability convention, not a tuned
hyperparameter: ``(mu, S, tau) -> (a mu, a^2 S, a tau)`` leaves every
probability unchanged, so a probit model must fix its latent unit, exactly as
standard probit regression does. The branch offsets are deterministic too,
``o_j = Phi^-1(pi_j)`` for the left-subtree share ``pi_j`` of the node's prior
mass, which is zero at every node of a power-of-two balanced tree.

Everything in this module that takes a ``log_tau``, and the categorical
normalizer that the padded tree of :func:`class_to_obs` needs, exists for
research ablations against that model. They are not part of it: the proposed
head predicts and trains at ``tau = 1`` with no calibration step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .hrc_softmax import HierarchicalSoftmax

DEFAULT_LOG_TAU: float = 0.0
_VARIANCE_FLOOR: float = 1e-12


def _resolve_normalize(hrc: HierarchicalSoftmax, normalize: bool | None) -> bool:
    """Normalize only where the tree topology makes it necessary.

    A full tree needs no normalizer, so the proposed model contains none. The
    padded tree of :func:`class_to_obs` discards leaves that hold probability
    mass, so its ablations cannot skip one.
    """

    if normalize is None:
        return not hrc.is_full
    return bool(normalize)


def log_tau_from_alpha(alpha: float) -> float:
    """Convert an ``obs_to_class_probs`` scale to the latent probit log-scale."""

    if alpha <= 0.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    return -math.log(alpha)


def alpha_from_log_tau(log_tau: float) -> float:
    """Convert a latent probit log-scale to the ``obs_to_class_probs`` scale."""

    if not math.isfinite(log_tau):
        raise ValueError("log_tau must be finite")
    return math.exp(-log_tau)


# ──────────────────────────────────────────────────────────────────────────────
#  Log-space class probabilities
# ──────────────────────────────────────────────────────────────────────────────


def _check_node_moments(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
) -> None:
    if output_mean.shape != output_variance.shape:
        raise ValueError("node means and variances must have the same shape")
    if output_mean.dim() != 2 or output_mean.shape[1] != hrc.len:
        raise ValueError("HRC node moments must have shape (batch, hrc.len)")
    if not bool(torch.isfinite(output_mean).all()):
        raise ValueError("node means must be finite")
    if not bool(torch.isfinite(output_variance).all()) or bool((output_variance < 0).any()):
        raise ValueError("node variances must be finite and nonnegative")


def _path_selector(hrc: HierarchicalSoftmax, device: torch.device) -> tuple[Tensor, Tensor]:
    """Return the flat left/right column selector and the padding mask.

    The selector indexes a ``(batch, 2 * hrc.len)`` concatenation of the
    left-branch and right-branch node log-probabilities, so one advanced index
    replaces a ``(batch, n_classes, hrc.len)`` expansion.
    """

    signs = hrc.obs.to(device).reshape(-1)
    node_index = hrc.idx.to(device).long().reshape(-1) - 1
    if bool(((node_index < 0) | (node_index >= hrc.len)).any()):
        raise ValueError("HRC node index is out of range for the tree width")
    selector = node_index + hrc.len * (signs < 0).long()
    return selector, hrc.path_mask(device)


def hrc_branch_log_probabilities(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
    *,
    log_tau: float | Tensor = DEFAULT_LOG_TAU,
) -> tuple[Tensor, Tensor]:
    """Return per-node ``log Phi`` for the left and right branch events.

    Args:
        output_mean: Node means, shape (batch, hrc.len).
        output_variance: Node variances, shape (batch, hrc.len).
        hrc: Tree structure.
        log_tau: Log latent probit standard deviation. A tensor keeps the
            result differentiable in the scale.

    Returns:
        log_left: ``log Phi(gamma)`` for the left branch, shape (batch, hrc.len).
        log_right: ``log Phi(-gamma)`` for the right branch, same shape.
    """

    _check_node_moments(output_mean, output_variance, hrc)
    mean = output_mean.double()
    variance = output_variance.double()
    scale = torch.as_tensor(log_tau, dtype=torch.float64, device=mean.device)
    if scale.dim() != 0:
        raise ValueError("log_tau must be a scalar")
    tau = scale.exp()
    offset = hrc.node_offset(mean.device).double()
    # The offset enters the latent variable as tau * o_j so that the whole
    # model stays invariant under (mu, S, tau) -> (a mu, a^2 S, a tau). At the
    # unit scale of the proposed head this is exactly R_j = Z_j + o_j + eps_j.
    gamma = (mean + tau * offset) / torch.sqrt(variance + tau.square())
    return torch.special.log_ndtr(gamma), torch.special.log_ndtr(-gamma)


def hrc_path_log_scores(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
    *,
    log_tau: float | Tensor = DEFAULT_LOG_TAU,
) -> Tensor:
    """Return unnormalized path log-scores ``log q_c``, shape (batch, n_classes).

    On a full tree ``logsumexp(log q) == 0`` up to floating-point error, which
    is a useful invariant check on the tree topology.
    """

    log_left, log_right = hrc_branch_log_probabilities(
        output_mean, output_variance, hrc, log_tau=log_tau
    )
    selector, mask = _path_selector(hrc, log_left.device)
    combined = torch.cat((log_left, log_right), dim=1)
    factors = combined[:, selector].reshape(-1, hrc.n_classes, hrc.n_obs)
    factors = torch.where(mask.double() > 0.0, factors, factors.new_zeros(()))
    return factors.sum(dim=-1)


def hrc_log_probs(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
    *,
    log_tau: float | Tensor = DEFAULT_LOG_TAU,
    normalize: bool | None = None,
) -> Tensor:
    """Return categorical class log-probabilities, shape (batch, n_classes).

    Args:
        output_mean: Node means, shape (batch, hrc.len).
        output_variance: Node variances, shape (batch, hrc.len).
        hrc: Tree structure.
        log_tau: Log latent probit standard deviation.
        normalize: Subtract the categorical ``logsumexp``. ``None`` resolves it
            from the tree: a full tree needs no normalizer, so the proposed
            model contains none, while the padded tree of
            :func:`class_to_obs` discards leaves that hold probability mass and
            cannot go without one.

    Returns:
        Class log-probabilities in float64.
    """

    log_scores = hrc_path_log_scores(output_mean, output_variance, hrc, log_tau=log_tau)
    if not _resolve_normalize(hrc, normalize):
        return log_scores
    return log_scores - torch.logsumexp(log_scores, dim=-1, keepdim=True)


def hrc_class_probabilities(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
    *,
    log_tau: float | Tensor = DEFAULT_LOG_TAU,
    normalize: bool | None = None,
) -> Tensor:
    """Return categorical class probabilities in the dtype of ``output_mean``."""

    log_probabilities = hrc_log_probs(
        output_mean, output_variance, hrc, log_tau=log_tau, normalize=normalize
    )
    return log_probabilities.exp().to(output_mean.dtype)


def hrc_negative_log_likelihood(
    output_mean: Tensor,
    output_variance: Tensor,
    labels: Tensor,
    hrc: HierarchicalSoftmax,
    *,
    log_tau: float | Tensor = DEFAULT_LOG_TAU,
    normalize: bool | None = None,
) -> Tensor:
    """Return mean categorical cross-entropy in nats for the hierarchical model.

    Args:
        output_mean: Node means, shape (batch, hrc.len).
        output_variance: Node variances, shape (batch, hrc.len).
        labels: Integer class labels, shape (batch,).
        hrc: Tree structure.
        log_tau: Log latent probit standard deviation.
        normalize: Include the categorical normalizer; ``None`` resolves it
            from the tree topology.

    Returns:
        Scalar float64 tensor, differentiable in ``log_tau`` when it is a tensor.
    """

    if labels.dim() != 1 or labels.shape[0] != output_mean.shape[0]:
        raise ValueError("labels must have shape (batch,) matching the predictions")
    target = labels.to(output_mean.device).long()
    if bool(((target < 0) | (target >= hrc.n_classes)).any()):
        raise ValueError("labels contain an invalid class index")
    log_probabilities = hrc_log_probs(
        output_mean, output_variance, hrc, log_tau=log_tau, normalize=normalize
    )
    return -log_probabilities.gather(1, target[:, None]).squeeze(1).mean()


def hrc_log_partition_deviation(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
    *,
    log_tau: float | Tensor = DEFAULT_LOG_TAU,
) -> float:
    """Return ``max_n |sum_c q_nc - 1|`` for the unnormalized path scores.

    Zero for a full tree; on the padded tree it measures the probability mass
    that the discarded leaves absorb.
    """

    log_scores = hrc_path_log_scores(output_mean, output_variance, hrc, log_tau=log_tau)
    return float((torch.logsumexp(log_scores, dim=-1).exp() - 1.0).abs().max())


# ──────────────────────────────────────────────────────────────────────────────
#  Post-hoc scalar calibration
# ──────────────────────────────────────────────────────────────────────────────


def fit_hrc_log_tau(
    output_mean: Tensor,
    output_variance: Tensor,
    labels: Tensor,
    hrc: HierarchicalSoftmax,
    *,
    log_tau_bounds: tuple[float, float] = (-5.0, 5.0),
    grid_size: int = 33,
    iterations: int = 60,
    normalize: bool | None = None,
) -> float:
    """Fit one latent probit log-scale by held-out categorical NLL.

    A coarse grid brackets the minimum before a golden-section refinement, so
    the result is deterministic and does not assume the objective is unimodal
    over the whole interval. This is the hierarchical analogue of temperature
    scaling; call it on calibration data with the network frozen.

    Args:
        output_mean: Cached node means on the calibration split.
        output_variance: Cached node variances on the calibration split.
        labels: Integer class labels for the calibration split.
        hrc: Tree structure.
        log_tau_bounds: Search interval for ``log_tau``.
        grid_size: Number of grid points used to bracket the minimum.
        iterations: Golden-section refinement steps.
        normalize: Include the categorical normalizer; ``None`` resolves it
            from the tree topology.

    Returns:
        The fitted ``log_tau``.
    """

    lower, upper = log_tau_bounds
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("log_tau_bounds must be finite and increasing")
    if grid_size < 3 or iterations < 1:
        raise ValueError("grid_size must be at least three and iterations positive")

    def objective(value: float) -> float:
        return float(
            hrc_negative_log_likelihood(
                output_mean,
                output_variance,
                labels,
                hrc,
                log_tau=value,
                normalize=normalize,
            )
        )

    grid = [lower + (upper - lower) * step / (grid_size - 1) for step in range(grid_size)]
    losses = [objective(value) for value in grid]
    best = min(range(grid_size), key=lambda index: losses[index])
    left = grid[max(best - 1, 0)]
    right = grid[min(best + 1, grid_size - 1)]

    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    probe_left = right - ratio * (right - left)
    probe_right = left + ratio * (right - left)
    loss_left, loss_right = objective(probe_left), objective(probe_right)
    for _ in range(iterations):
        if loss_left <= loss_right:
            right, probe_right, loss_right = probe_right, probe_left, loss_left
            probe_left = right - ratio * (right - left)
            loss_left = objective(probe_left)
        else:
            left, probe_left, loss_left = probe_left, probe_right, loss_right
            probe_right = left + ratio * (right - left)
            loss_right = objective(probe_right)
    return 0.5 * (left + right)


# ──────────────────────────────────────────────────────────────────────────────
#  Score and curvature in the log-scale
# ──────────────────────────────────────────────────────────────────────────────


def probit_log_tau_factor_score_curvature(
    gamma: Tensor,
    rho: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the closed-form ``log_tau`` score and curvature of one factor.

    For ``l = log Phi(gamma)``, ``lam = phi(gamma) / Phi(gamma)`` and
    ``rho = tau^2 / (S + tau^2)`` with a zero branch offset::

        dl/deta   = -lam * gamma * rho
        d2l/deta2 = lam * gamma * rho * (3 * rho - 2)
                    - lam * (gamma + lam) * gamma^2 * rho^2

    Offsets break these expressions, so they cover the zero-offset case only;
    :func:`hrc_log_tau_score_curvature` differentiates the exact objective and
    is what callers should use.
    """

    work = gamma.double()
    density_ratio = torch.exp(
        -0.5 * work.square() - 0.5 * math.log(2.0 * math.pi) - torch.special.log_ndtr(work)
    )
    fraction = rho.double()
    score = -density_ratio * work * fraction
    curvature = (
        density_ratio * work * fraction * (3.0 * fraction - 2.0)
        - density_ratio * (work + density_ratio) * work.square() * fraction.square()
    )
    return score, curvature


def hrc_log_tau_score_curvature(
    output_mean: Tensor,
    output_variance: Tensor,
    labels: Tensor,
    hrc: HierarchicalSoftmax,
    *,
    log_tau: float = DEFAULT_LOG_TAU,
    normalize: bool | None = None,
) -> tuple[float, float]:
    """Return the summed log-likelihood score and curvature in ``log_tau``.

    Differentiating the exact objective covers branch offsets and the
    categorical normalizer, both of which the per-factor closed form omits.

    Returns:
        score: ``d/d eta sum_n log p(c_n | eta)``.
        curvature: ``d^2/d eta^2 sum_n log p(c_n | eta)``.
    """

    # Enabled explicitly so the derivatives survive an ambient torch.no_grad,
    # which every evaluation loop runs under.
    with torch.enable_grad():
        parameter = torch.tensor(float(log_tau), dtype=torch.float64, requires_grad=True)
        total = -hrc_negative_log_likelihood(
            output_mean.detach(),
            output_variance.detach(),
            labels,
            hrc,
            log_tau=parameter,
            normalize=normalize,
        ) * float(output_mean.shape[0])
        (score,) = torch.autograd.grad(total, parameter, create_graph=True)
        (curvature,) = torch.autograd.grad(score, parameter)
    return float(score.detach()), float(curvature.detach())


# ──────────────────────────────────────────────────────────────────────────────
#  Bayesian scalar scale
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LogTauPosterior:
    """Gaussian belief over ``eta = log tau``."""

    mean: float
    variance: float

    @property
    def tau(self) -> float:
        """Posterior-mean latent probit standard deviation."""

        return math.exp(self.mean)

    @property
    def alpha_impl(self) -> float:
        """The equivalent ``obs_to_class_probs`` scale ``alpha = 1 / tau``."""

        return math.exp(-self.mean)


def update_log_tau_gaussian(
    posterior: LogTauPosterior,
    score: float,
    curvature: float,
    *,
    damping: float = 1.0,
    variance_floor: float = 1e-4,
    max_mean_step: float = 0.5,
) -> LogTauPosterior:
    """Apply one TAGI-style local Gaussian update to the log-scale belief.

    ``m+ = m + S * g`` and ``S+ = S + S^2 * h``, with the damping, variance
    floor, and step cap that TAGI already uses elsewhere. The likelihood need
    not be concave in ``eta``, so a positive curvature is clipped away rather
    than allowed to inflate the variance.
    """

    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0, 1]")
    if variance_floor <= 0.0 or max_mean_step <= 0.0:
        raise ValueError("variance_floor and max_mean_step must be positive")
    if not math.isfinite(score) or not math.isfinite(curvature):
        raise ValueError("score and curvature must be finite")

    variance = posterior.variance
    step = damping * variance * score
    step = max(-max_mean_step, min(max_mean_step, step))
    updated_variance = variance + damping * variance * variance * min(curvature, 0.0)
    return LogTauPosterior(
        mean=posterior.mean + step,
        variance=max(variance_floor, min(variance, updated_variance)),
    )


def fit_hrc_log_tau_laplace(
    output_mean: Tensor,
    output_variance: Tensor,
    labels: Tensor,
    hrc: HierarchicalSoftmax,
    *,
    prior_mean: float = DEFAULT_LOG_TAU,
    prior_variance: float = 1.0,
    log_tau_bounds: tuple[float, float] = (-5.0, 5.0),
    iterations: int = 40,
    tolerance: float = 1e-8,
    normalize: bool | None = None,
) -> LogTauPosterior:
    """Fit a Gaussian posterior over ``log tau`` by a batch Laplace approximation.

    The MAP is located by a safeguarded Newton iteration on the penalized
    log-likelihood, and the posterior variance is the inverse negative
    curvature there. A batch fit is safer than a fully online update because
    the likelihood is not globally concave in ``eta``.

    Args:
        output_mean: Cached node means on the calibration split.
        output_variance: Cached node variances on the calibration split.
        labels: Integer class labels for the calibration split.
        hrc: Tree structure.
        prior_mean: Prior mean of ``eta``.
        prior_variance: Prior variance of ``eta``.
        log_tau_bounds: Interval the iterate is confined to.
        iterations: Maximum Newton steps.
        tolerance: Absolute step size at which the iteration stops.
        normalize: Include the categorical normalizer; ``None`` resolves it
            from the tree topology.

    Returns:
        The Laplace posterior over ``eta = log tau``.
    """

    if prior_variance <= 0.0 or not math.isfinite(prior_variance):
        raise ValueError("prior_variance must be finite and positive")
    lower, upper = log_tau_bounds
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("log_tau_bounds must be finite and increasing")

    def penalized(value: float) -> tuple[float, float, float]:
        score, curvature = hrc_log_tau_score_curvature(
            output_mean,
            output_variance,
            labels,
            hrc,
            log_tau=value,
            normalize=normalize,
        )
        objective = (
            -float(
                hrc_negative_log_likelihood(
                    output_mean,
                    output_variance,
                    labels,
                    hrc,
                    log_tau=value,
                    normalize=normalize,
                )
            )
            * float(output_mean.shape[0])
            - 0.5 * (value - prior_mean) ** 2 / prior_variance
        )
        return (
            objective,
            score - (value - prior_mean) / prior_variance,
            curvature - 1.0 / prior_variance,
        )

    # Start from the profile-likelihood optimum so the Newton step begins in a
    # locally concave region even when the likelihood is not globally concave.
    current = fit_hrc_log_tau(
        output_mean,
        output_variance,
        labels,
        hrc,
        log_tau_bounds=log_tau_bounds,
        normalize=normalize,
    )
    objective, score, curvature = penalized(current)
    for _ in range(iterations):
        if curvature >= -_VARIANCE_FLOOR:
            break
        step = max(-1.0, min(1.0, -score / curvature))
        if abs(step) <= tolerance:
            break
        candidate = max(lower, min(upper, current + step))
        candidate_objective, candidate_score, candidate_curvature = penalized(candidate)
        if candidate_objective < objective:
            break
        current, objective = candidate, candidate_objective
        score, curvature = candidate_score, candidate_curvature

    variance = prior_variance if curvature >= -_VARIANCE_FLOOR else -1.0 / curvature
    return LogTauPosterior(mean=current, variance=min(variance, prior_variance))
