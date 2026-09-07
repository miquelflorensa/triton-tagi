"""Fixed-noise categorical AGCI moments for diagonal TAGI outputs.

The multiclass event is ``argmax(Z + E) == c`` with independent
``E_i ~ N(0, tau**2)``.  Standard TAGI supplies diagonal output covariance.
For that case, conditioning on the winning noisy utility reduces the nominal
``C - 1`` dimensional truncated-normal integral to one dimension.  Shifted
Gauss-Hermite quadrature evaluates that integral and its first two moments.

``agci_event`` returns the dense event-conditioned moments for diagnostics and
checks.  Diagonal TAGI training uses the same observed-event projection but
computes only its marginal variances, preserving the O(QK) algorithm in the
formulation.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

_EPS = 1e-8


def agci_weight_gain_from_kappa(
    feature_energy: float,
    input_dim: int,
    kappa: float,
    *,
    tau: float = 1.0,
    bias_gain: float = 1.0,
) -> float:
    """Derive the isotropic weight gain from the prior utility ratio.

    ``Linear`` stores weight and bias variances as ``gain**2 / input_dim``.
    Consequently, the mean prior utility variance over the transformed
    training features is ``gain_w**2 * feature_energy + gain_b**2 / D``.
    """

    values = (feature_energy, kappa, tau, bias_gain)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("AGCI prior scales must be finite")
    if feature_energy <= 0.0:
        raise ValueError("feature_energy must be positive")
    if input_dim < 1:
        raise ValueError("input_dim must be positive")
    if kappa <= 0.0 or tau <= 0.0 or bias_gain < 0.0:
        raise ValueError("kappa and tau must be positive and bias_gain nonnegative")
    target_variance = (kappa * tau) ** 2
    bias_variance = bias_gain**2 / input_dim
    if target_variance <= bias_variance:
        raise ValueError(
            "kappa**2 * tau**2 must exceed the fan-in-scaled bias variance"
        )
    return math.sqrt((target_variance - bias_variance) / feature_energy)


@lru_cache(maxsize=None)
def _normal_quadrature(num_quad: int) -> tuple[Tensor, Tensor]:
    """Return standard-normal nodes and log weights on the CPU in float64."""

    if num_quad < 8:
        raise ValueError("num_quad must be at least 8")
    nodes, weights = np.polynomial.hermite.hermgauss(num_quad)
    nodes = torch.from_numpy(nodes * math.sqrt(2.0))
    log_weights = torch.from_numpy(np.log(weights) - 0.5 * math.log(math.pi))
    return nodes, log_weights


def _inverse_mills_ratio(value: Tensor) -> Tensor:
    """Return ``phi(value) / Phi(value)`` without left-tail underflow."""

    original_dtype = value.dtype
    work = (
        value.double()
        if value.dtype in (torch.float16, torch.bfloat16, torch.float32)
        else value
    )
    left = math.sqrt(2.0 / math.pi) / torch.special.erfcx(
        -work / math.sqrt(2.0)
    )
    right = torch.exp(
        -0.5 * work.square()
        - 0.5 * math.log(2.0 * math.pi)
        - torch.special.log_ndtr(work)
    )
    return torch.where(work <= 0.0, left, right).to(original_dtype)


def _diagonal_output_variance(mean: Tensor, covariance: Tensor) -> Tensor:
    """Validate output moments and return their diagonal variance."""

    if mean.dim() < 1 or not mean.is_floating_point():
        raise ValueError("output means must be floating point with a class dimension")
    if not torch.isfinite(mean).all():
        raise ValueError("output means must be finite")
    if covariance.shape == mean.shape:
        variance = covariance
    elif covariance.shape == (*mean.shape, mean.shape[-1]):
        if not torch.allclose(
            covariance,
            torch.diag_embed(covariance.diagonal(dim1=-2, dim2=-1)),
            rtol=1e-5,
            atol=1e-7,
        ):
            raise ValueError(
                "AGCI currently requires diagonal TAGI output covariance"
            )
        variance = covariance.diagonal(dim1=-2, dim2=-1)
    else:
        raise ValueError(
            "output covariance must contain diagonal variances or dense C x C blocks"
        )
    if not torch.isfinite(variance).all() or bool((variance < 0.0).any()):
        raise ValueError("output variances must be finite and nonnegative")
    return variance


def binary_probit_agci_event(
    output_mean: Tensor,
    output_variance: Tensor,
    labels: Tensor,
    *,
    tau: float = 1.0,
    eps: float = _EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return exact scalar binary-probit event mass, mean, and variance."""

    if output_mean.shape != output_variance.shape:
        raise ValueError("binary output mean and variance must have matching shapes")
    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    if not output_mean.is_floating_point() or not torch.isfinite(output_mean).all():
        raise ValueError("binary output means must be finite floating-point values")
    if not torch.isfinite(output_variance).all() or bool((output_variance < 0).any()):
        raise ValueError("binary output variances must be finite and nonnegative")

    labels = labels.to(device=output_mean.device).reshape(output_mean.shape)
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("binary labels must be zero or one")
    sign = labels.to(output_mean.dtype).mul(2.0).sub(1.0)
    scale2 = output_variance + tau**2
    scale = torch.sqrt(scale2.clamp_min(eps))
    signed_standardized_mean = sign * output_mean / scale
    log_mass = torch.special.log_ndtr(signed_standardized_mean)
    mills = _inverse_mills_ratio(signed_standardized_mean)
    posterior_mean = (
        output_mean + sign * output_variance / scale * mills
    )
    contraction = mills * (mills + signed_standardized_mean)
    posterior_variance = output_variance - (
        output_variance.square() / scale2.clamp_min(eps) * contraction
    )
    posterior_variance = posterior_variance.clamp(min=0.0)
    return log_mass, posterior_mean, posterior_variance


def binary_probit_agci_predictive_probs(
    output_mean: Tensor,
    output_variance: Tensor,
    *,
    tau: float = 1.0,
    eps: float = _EPS,
) -> Tensor:
    """Return ``[P(Y=0), P(Y=1)]`` for the scalar binary model."""

    if output_mean.shape != output_variance.shape:
        raise ValueError("binary output mean and variance must have matching shapes")
    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    scale = torch.sqrt((output_variance + tau**2).clamp_min(eps))
    probability_one = torch.special.ndtr(output_mean / scale)
    return torch.stack((1.0 - probability_one, probability_one), dim=-1)


def binary_probit_agci_posterior(
    output_mean: Tensor,
    output_variance: Tensor,
    labels: Tensor,
    *,
    tau: float = 1.0,
    eps: float = _EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the original scalar binary AGCI mass, mean, and covariance.

    The mean is event-conditioned. The covariance is the minimum-MSE affine
    residual covariance and therefore does not depend on the observed label.
    """

    log_mass, posterior_mean, _ = binary_probit_agci_event(
        output_mean, output_variance, labels, tau=tau, eps=eps
    )
    probabilities = binary_probit_agci_predictive_probs(
        output_mean, output_variance, tau=tau, eps=eps
    )
    zero_labels = torch.zeros_like(labels)
    one_labels = torch.ones_like(labels)
    _, _, variance_zero = binary_probit_agci_event(
        output_mean, output_variance, zero_labels, tau=tau, eps=eps
    )
    _, _, variance_one = binary_probit_agci_event(
        output_mean, output_variance, one_labels, tau=tau, eps=eps
    )
    posterior_variance = (
        probabilities[..., 0] * variance_zero
        + probabilities[..., 1] * variance_one
    )
    return log_mass, posterior_mean, posterior_variance


def _agci_diagonal_event(
    output_mean: Tensor,
    output_variance: Tensor,
    labels: Tensor,
    *,
    tau: float,
    num_quad: int,
    eps: float,
    compute_moments: bool,
    compute_covariance: bool = True,
    diagonal_covariance: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Evaluate independent-utility class events by shifted 1-D quadrature."""

    num_classes = output_mean.shape[-1]
    if num_classes < 2:
        raise ValueError("multiclass AGCI requires at least two utilities")
    leading_shape = output_mean.shape[:-1]
    flat_mean = output_mean.reshape(-1, num_classes)
    flat_variance = output_variance.reshape(-1, num_classes)
    flat_labels = labels.to(device=output_mean.device, dtype=torch.long).reshape(-1)
    if flat_labels.numel() != flat_mean.shape[0]:
        raise ValueError("labels leading shape must match output moments")
    if bool(((flat_labels < 0) | (flat_labels >= num_classes)).any()):
        raise ValueError("labels contain an invalid class index")

    batch_size = flat_mean.shape[0]
    rows = torch.arange(batch_size, device=output_mean.device)
    classes = torch.arange(num_classes, device=output_mean.device).expand(
        batch_size, -1
    )
    competitors = classes[classes != flat_labels[:, None]].reshape(
        batch_size, num_classes - 1
    )
    winner_mean = flat_mean[rows, flat_labels]
    competitor_mean = flat_mean.gather(1, competitors)
    utility_variance = flat_variance + tau**2
    utility_scale = torch.sqrt(utility_variance.clamp_min(eps))
    winner_scale = utility_scale[rows, flat_labels]
    competitor_scale = utility_scale.gather(1, competitors)

    # Shift the standard-normal proposal to the event-integrand mode.  This
    # keeps rare-class masses accurate when the winning utility must enter a
    # far tail; the importance correction below leaves the integral unchanged.
    shift = torch.zeros_like(winner_mean)
    scale_ratio = winner_scale[:, None] / competitor_scale
    for _ in range(10):
        standardized = (
            winner_mean[:, None]
            + winner_scale[:, None] * shift[:, None]
            - competitor_mean
        ) / competitor_scale
        mills = _inverse_mills_ratio(standardized)
        gradient = -shift + (scale_ratio * mills).sum(dim=1)
        contraction = mills * (mills + standardized)
        hessian = -1.0 - (scale_ratio.square() * contraction).sum(dim=1)
        shift = (shift - gradient / hessian).clamp(-50.0, 50.0)

    cpu_nodes, cpu_log_weights = _normal_quadrature(num_quad)
    nodes = cpu_nodes.to(device=output_mean.device, dtype=output_mean.dtype)
    log_weights = cpu_log_weights.to(
        device=output_mean.device, dtype=output_mean.dtype
    )
    standard_normal = shift[:, None] + nodes[None, :]
    winner_utility = winner_mean[:, None] + winner_scale[:, None] * standard_normal
    standardized = (
        winner_utility[:, :, None] - competitor_mean[:, None, :]
    ) / competitor_scale[:, None, :]
    log_importance_ratio = (
        -shift[:, None] * standard_normal + 0.5 * shift[:, None].square()
    )
    log_terms = (
        log_weights[None, :]
        + log_importance_ratio
        + torch.special.log_ndtr(standardized).sum(dim=-1)
    )
    log_mass = torch.logsumexp(log_terms, dim=1).clamp_max(0.0)
    if not compute_moments:
        return log_mass.reshape(leading_shape), None, None

    normalized_weights = torch.softmax(log_terms, dim=1)
    mills = _inverse_mills_ratio(standardized)
    competitor_conditional_mean = (
        competitor_mean[:, None, :] - competitor_scale[:, None, :] * mills
    )
    conditional_mean_u = torch.empty(
        batch_size,
        num_quad,
        num_classes,
        device=output_mean.device,
        dtype=output_mean.dtype,
    )
    conditional_mean_u.scatter_(
        2,
        competitors[:, None, :].expand(-1, num_quad, -1),
        competitor_conditional_mean,
    )
    conditional_mean_u.scatter_(
        2,
        flat_labels[:, None, None].expand(-1, num_quad, 1),
        winner_utility[:, :, None],
    )
    mean_u = torch.einsum("bq,bqi->bi", normalized_weights, conditional_mean_u)
    gain = flat_variance / utility_variance.clamp_min(eps)
    posterior_mean = flat_mean + gain * (mean_u - flat_mean)
    if not compute_covariance:
        return (
            log_mass.reshape(leading_shape),
            posterior_mean.reshape(*leading_shape, num_classes),
            None,
        )

    competitor_conditional_variance = utility_variance.gather(1, competitors)[
        :, None, :
    ] * (1.0 - standardized * mills - mills.square()).clamp_min(0.0)
    conditional_variance_u = torch.zeros_like(conditional_mean_u)
    conditional_variance_u.scatter_(
        2,
        competitors[:, None, :].expand(-1, num_quad, -1),
        competitor_conditional_variance,
    )
    mean_conditional_variance = torch.einsum(
        "bq,bqi->bi", normalized_weights, conditional_variance_u
    )
    residual_variance = flat_variance - gain * flat_variance
    if diagonal_covariance:
        second_diagonal_u = torch.einsum(
            "bq,bqi->bi", normalized_weights, conditional_mean_u.square()
        ) + mean_conditional_variance
        variance_u = (second_diagonal_u - mean_u.square()).clamp_min(0.0)
        posterior_variance = residual_variance + gain.square() * variance_u
        return (
            log_mass.reshape(leading_shape),
            posterior_mean.reshape(*leading_shape, num_classes),
            posterior_variance.reshape(*leading_shape, num_classes),
        )

    second_u = torch.einsum(
        "bq,bqi,bqj->bij",
        normalized_weights,
        conditional_mean_u,
        conditional_mean_u,
    )
    second_u = second_u + torch.diag_embed(mean_conditional_variance)
    covariance_u = second_u - mean_u.unsqueeze(-1) * mean_u.unsqueeze(-2)
    covariance_u = 0.5 * (covariance_u + covariance_u.transpose(-1, -2))

    posterior_covariance = (
        torch.diag_embed(residual_variance)
        + gain.unsqueeze(-1) * covariance_u * gain.unsqueeze(-2)
    )
    posterior_covariance = 0.5 * (
        posterior_covariance + posterior_covariance.transpose(-1, -2)
    )
    return (
        log_mass.reshape(leading_shape),
        posterior_mean.reshape(*leading_shape, num_classes),
        posterior_covariance.reshape(*leading_shape, num_classes, num_classes),
    )


def _resolve_class_chunk_size(
    output_mean: Tensor,
    num_quad: int,
    class_chunk_size: int | None,
) -> int:
    """Choose a candidate-class chunk that bounds quadrature working memory."""

    num_classes = output_mean.shape[-1]
    if class_chunk_size is not None:
        if class_chunk_size < 1:
            raise ValueError("class_chunk_size must be positive")
        return min(class_chunk_size, num_classes)
    batch_size = output_mean.numel() // num_classes
    # The main work tensors have B x chunk x Q x (K - 1) entries. Keeping a
    # single such tensor near two million elements makes the default safe for
    # large-class heads while allowing all classes at once on small problems.
    elements_per_candidate = max(batch_size * num_quad * num_classes, 1)
    return max(1, min(num_classes, 2_000_000 // elements_per_candidate))


def _agci_candidate_chunks(
    output_mean: Tensor,
    output_variance: Tensor,
    *,
    tau: float,
    num_quad: int,
    eps: float,
    class_chunk_size: int | None,
    compute_means: bool,
):
    """Yield event masses and optional means for consecutive candidate classes."""

    num_classes = output_mean.shape[-1]
    leading_shape = output_mean.shape[:-1]
    chunk_size = _resolve_class_chunk_size(output_mean, num_quad, class_chunk_size)
    for start in range(0, num_classes, chunk_size):
        stop = min(start + chunk_size, num_classes)
        candidates = stop - start
        candidate_mean = output_mean.unsqueeze(-2).expand(
            *leading_shape, candidates, num_classes
        )
        candidate_variance = output_variance.unsqueeze(-2).expand_as(candidate_mean)
        labels = torch.arange(
            start, stop, device=output_mean.device, dtype=torch.long
        ).expand(*leading_shape, candidates)
        log_mass, conditional_mean, _ = _agci_diagonal_event(
            candidate_mean,
            candidate_variance,
            labels,
            tau=tau,
            num_quad=num_quad,
            eps=eps,
            compute_moments=compute_means,
            compute_covariance=False,
        )
        yield start, stop, log_mass, conditional_mean


def agci_event(
    output_mean: Tensor,
    output_covariance: Tensor,
    labels: Tensor,
    *,
    tau: float = 1.0,
    num_quad: int = 48,
    eps: float = _EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Condition TAGI utilities on an observed argmax class.

    The returned mean and dense covariance are the class-event conditional
    moments.  The numerical integration is one-dimensional because standard
    TAGI output covariance is diagonal.
    """

    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    variance = _diagonal_output_variance(output_mean, output_covariance)
    log_mass, posterior_mean, posterior_covariance = _agci_diagonal_event(
        output_mean,
        variance,
        labels,
        tau=tau,
        num_quad=num_quad,
        eps=eps,
        compute_moments=True,
    )
    assert posterior_mean is not None and posterior_covariance is not None
    return log_mass, posterior_mean, posterior_covariance


def agci_event_diagonal(
    output_mean: Tensor,
    output_covariance: Tensor,
    labels: Tensor,
    *,
    tau: float = 1.0,
    num_quad: int = 48,
    eps: float = _EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Condition on the observed class and retain marginal output variances.

    This is the O(QK) Event-AGCI path used by diagonal TAGI.  It evaluates only
    the labeled class event and does not materialize a K x K covariance.
    """

    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    variance = _diagonal_output_variance(output_mean, output_covariance)
    log_mass, posterior_mean, posterior_variance = _agci_diagonal_event(
        output_mean,
        variance,
        labels,
        tau=tau,
        num_quad=num_quad,
        eps=eps,
        compute_moments=True,
        diagonal_covariance=True,
    )
    assert posterior_mean is not None and posterior_variance is not None
    return log_mass, posterior_mean, posterior_variance


def agci_categorical_moments(
    output_mean: Tensor,
    output_covariance: Tensor,
    *,
    tau: float = 1.0,
    num_quad: int = 48,
    eps: float = _EPS,
    class_chunk_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Return all class probabilities and conditional utility means.

    Conditional means have shape (..., C, C): the penultimate axis is the
    candidate observed class and the final axis is the latent utility. This
    explicit representation is intended for theory checks and small C. Large
    classification heads use the streaming diagonal projection below.
    """

    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    variance = _diagonal_output_variance(output_mean, output_covariance)
    log_masses: list[Tensor] = []
    conditional_means: list[Tensor] = []
    for _, _, log_mass, conditional_mean in _agci_candidate_chunks(
        output_mean,
        variance,
        tau=tau,
        num_quad=num_quad,
        eps=eps,
        class_chunk_size=class_chunk_size,
        compute_means=True,
    ):
        assert conditional_mean is not None
        log_masses.append(log_mass)
        conditional_means.append(conditional_mean)
    return (
        torch.softmax(torch.cat(log_masses, dim=-1), dim=-1),
        torch.cat(conditional_means, dim=-2),
    )


def agci_categorical_posterior(
    output_mean: Tensor,
    output_covariance: Tensor,
    labels: Tensor,
    *,
    tau: float = 1.0,
    num_quad: int = 48,
    eps: float = _EPS,
    class_chunk_size: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return original categorical AGCI probabilities, mean, and covariance.

    The class-averaged covariance is the optimal affine residual covariance.
    Materializing it costs O(C^2) storage and its contraction costs O(C^3)
    work; diagonal TAGI training uses the exact streaming projection below.
    """

    variance = _diagonal_output_variance(output_mean, output_covariance)
    probabilities, conditional_means = agci_categorical_moments(
        output_mean,
        variance,
        tau=tau,
        num_quad=num_quad,
        eps=eps,
        class_chunk_size=class_chunk_size,
    )
    num_classes = output_mean.shape[-1]
    leading_shape = output_mean.shape[:-1]
    labels = labels.to(device=output_mean.device, dtype=torch.long)
    if labels.shape != leading_shape:
        raise ValueError("labels leading shape must match output moments")
    if bool(((labels < 0) | (labels >= num_classes)).any()):
        raise ValueError("labels contain an invalid class index")
    gather_index = labels[..., None, None].expand(*leading_shape, 1, num_classes)
    posterior_mean = conditional_means.gather(-2, gather_index).squeeze(-2)
    deviations = conditional_means - output_mean.unsqueeze(-2)
    covariance_reduction = torch.einsum(
        "...c,...ci,...cj->...ij", probabilities, deviations, deviations
    )
    posterior_covariance = torch.diag_embed(variance) - covariance_reduction
    posterior_covariance = 0.5 * (
        posterior_covariance + posterior_covariance.transpose(-1, -2)
    )
    return probabilities, posterior_mean, posterior_covariance


def agci_categorical_posterior_diagonal(
    output_mean: Tensor,
    output_covariance: Tensor,
    labels: Tensor,
    *,
    tau: float = 1.0,
    num_quad: int = 48,
    eps: float = _EPS,
    class_chunk_size: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return original AGCI moments with an exact streaming diagonal projection."""

    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    variance = _diagonal_output_variance(output_mean, output_covariance)
    num_classes = output_mean.shape[-1]
    leading_shape = output_mean.shape[:-1]
    flat_mean = output_mean.reshape(-1, num_classes)
    flat_variance = variance.reshape(-1, num_classes)
    flat_labels = labels.to(device=output_mean.device, dtype=torch.long).reshape(-1)
    if flat_labels.numel() != flat_mean.shape[0]:
        raise ValueError("labels leading shape must match output moments")
    if bool(((flat_labels < 0) | (flat_labels >= num_classes)).any()):
        raise ValueError("labels contain an invalid class index")

    batch_size = flat_mean.shape[0]
    log_masses: list[Tensor] = []
    raw_mass_sum = torch.zeros(
        batch_size, 1, device=output_mean.device, dtype=output_mean.dtype
    )
    covariance_reduction = torch.zeros_like(flat_mean)
    posterior_mean = torch.empty_like(flat_mean)
    rows = torch.arange(batch_size, device=output_mean.device)

    for start, stop, log_mass, conditional_mean in _agci_candidate_chunks(
        flat_mean,
        flat_variance,
        tau=tau,
        num_quad=num_quad,
        eps=eps,
        class_chunk_size=class_chunk_size,
        compute_means=True,
    ):
        assert conditional_mean is not None
        flat_log_mass = log_mass.reshape(batch_size, stop - start)
        flat_conditional_mean = conditional_mean.reshape(
            batch_size, stop - start, num_classes
        )
        raw_mass = flat_log_mass.exp()
        deviations = flat_conditional_mean - flat_mean[:, None, :]
        raw_mass_sum.add_(raw_mass.sum(dim=1, keepdim=True))
        covariance_reduction.add_(
            torch.einsum("bc,bck->bk", raw_mass, deviations.square())
        )
        selected = (flat_labels >= start) & (flat_labels < stop)
        if bool(selected.any()):
            selected_rows = rows[selected]
            posterior_mean[selected_rows] = flat_conditional_mean[
                selected_rows, flat_labels[selected] - start
            ]
        log_masses.append(flat_log_mass)

    covariance_reduction = covariance_reduction / raw_mass_sum.clamp_min(eps)
    posterior_variance = flat_variance - covariance_reduction
    posterior_variance = torch.minimum(
        posterior_variance.clamp_min(0.0), flat_variance
    )
    probabilities = torch.softmax(torch.cat(log_masses, dim=-1), dim=-1)
    return (
        probabilities.reshape(*leading_shape, num_classes),
        posterior_mean.reshape(*leading_shape, num_classes),
        posterior_variance.reshape(*leading_shape, num_classes),
    )


def agci_predictive_probs(
    output_mean: Tensor,
    output_covariance: Tensor,
    *,
    tau: float = 1.0,
    num_quad: int = 48,
    eps: float = _EPS,
    class_chunk_size: int | None = None,
) -> Tensor:
    """Return categorical argmax probabilities under fixed Gaussian noise."""

    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    variance = _diagonal_output_variance(output_mean, output_covariance)
    log_masses = [
        log_mass
        for _, _, log_mass, _ in _agci_candidate_chunks(
            output_mean,
            variance,
            tau=tau,
            num_quad=num_quad,
            eps=eps,
            class_chunk_size=class_chunk_size,
            compute_means=False,
        )
    ]
    # The exact event masses already form a partition. Renormalization only
    # removes small quadrature error; it is not a componentwise simplex map.
    return torch.softmax(torch.cat(log_masses, dim=-1), dim=-1)


def compute_agci_innovation(
    labels: Tensor,
    output_mean: Tensor,
    output_variance: Tensor,
    *,
    tau: float = 1.0,
    num_quad: int = 48,
    eps: float = _EPS,
) -> tuple[Tensor, Tensor]:
    """Project observed-event posterior moments to TAGI innovations."""

    if output_mean.shape != output_variance.shape:
        raise ValueError("TAGI AGCI training expects matching diagonal output moments")
    _, posterior_mean, posterior_variance = agci_event_diagonal(
        output_mean,
        output_variance,
        labels,
        tau=tau,
        num_quad=num_quad,
        eps=eps,
    )
    active = output_variance > eps
    prior_variance = output_variance.clamp_min(eps)
    delta_mean = torch.where(
        active, (posterior_mean - output_mean) / prior_variance, 0.0
    )
    delta_variance = torch.where(
        active,
        (posterior_variance - output_variance) / prior_variance.square(),
        0.0,
    )
    return delta_mean, delta_variance
