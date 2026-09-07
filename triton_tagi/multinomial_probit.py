"""Dense multinomial-probit ADF observation updates for TAGI outputs."""

from __future__ import annotations

import math

import torch
from torch import Tensor

_EPS = 1e-8


def _inverse_mills_ratio(value: Tensor) -> Tensor:
    """Return phi(value) / Phi(value) without left-tail underflow."""

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


def _dense_output_covariance(
    mean: Tensor,
    covariance: Tensor,
) -> Tensor:
    """Validate moments and expand diagonal variances to dense blocks."""

    if mean.dim() < 1 or not mean.is_floating_point():
        raise ValueError("output means must be floating point with a class dimension")
    if not torch.isfinite(mean).all():
        raise ValueError("output means must be finite")
    if covariance.shape == mean.shape:
        if not torch.isfinite(covariance).all() or bool((covariance < 0).any()):
            raise ValueError("output variances must be finite and nonnegative")
        return torch.diag_embed(covariance)
    expected = (*mean.shape, mean.shape[-1])
    if covariance.shape != expected:
        raise ValueError(
            "output covariance must contain diagonal variances or dense K x K blocks"
        )
    if not torch.isfinite(covariance).all():
        raise ValueError("output covariance must be finite")
    if not torch.allclose(
        covariance, covariance.transpose(-1, -2), rtol=1e-5, atol=1e-7
    ):
        raise ValueError("dense output covariance must be symmetric")
    if bool((covariance.diagonal(dim1=-2, dim2=-1) < 0).any()):
        raise ValueError("output covariance diagonal must be nonnegative")
    return covariance


def multinomial_probit_adf_event(
    output_mean: Tensor,
    output_covariance: Tensor,
    labels: Tensor,
    *,
    probit_tau2: float = 1.0,
    eps: float = _EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Condition output moments on the argmax(Z + E) class event.

    The K - 1 pairwise half-space constraints are applied by assumed density
    filtering. Dense ZZ, ZU, and UU blocks are retained throughout, and the
    currently smallest standardized margin is processed first per sample.
    """

    if probit_tau2 < 0.0 or not math.isfinite(probit_tau2):
        raise ValueError("probit_tau2 must be finite and nonnegative")
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")

    dense_covariance = _dense_output_covariance(output_mean, output_covariance)
    num_classes = output_mean.shape[-1]
    if num_classes < 2:
        raise ValueError("multinomial probit requires at least two classes")
    leading_shape = output_mean.shape[:-1]
    flat_mean = output_mean.reshape(-1, num_classes)
    flat_covariance = dense_covariance.reshape(-1, num_classes, num_classes)
    flat_labels = labels.to(device=output_mean.device, dtype=torch.long).reshape(-1)
    if flat_labels.numel() != flat_mean.shape[0]:
        raise ValueError("labels leading shape must match output moments")
    if bool(((flat_labels < 0) | (flat_labels >= num_classes)).any()):
        raise ValueError("labels contain an invalid class index")

    batch_size = flat_mean.shape[0]
    rows = torch.arange(batch_size, device=output_mean.device)
    class_indicator = torch.nn.functional.one_hot(
        flat_labels, num_classes=num_classes
    ).to(flat_mean.dtype)
    identity = torch.eye(
        num_classes, device=output_mean.device, dtype=flat_mean.dtype
    ).expand(batch_size, -1, -1)

    mean_z = flat_mean.clone()
    mean_u = flat_mean.clone()
    covariance_zz = flat_covariance.clone()
    covariance_zu = flat_covariance.clone()
    covariance_uu = flat_covariance + float(probit_tau2) * identity
    log_mass = torch.zeros(
        batch_size, device=output_mean.device, dtype=flat_mean.dtype
    )
    remaining = torch.ones(
        batch_size, num_classes, device=output_mean.device, dtype=torch.bool
    )
    remaining[rows, flat_labels] = False

    for _ in range(num_classes - 1):
        class_mean = mean_u.gather(1, flat_labels[:, None])
        margin_mean = class_mean - mean_u
        diagonal_uu = covariance_uu.diagonal(dim1=-2, dim2=-1)
        class_variance = diagonal_uu.gather(1, flat_labels[:, None])
        class_covariance = covariance_uu[rows, flat_labels, :]
        margin_variance = (
            class_variance + diagonal_uu - 2.0 * class_covariance
        ).clamp_min(eps)
        standardized = margin_mean / torch.sqrt(margin_variance)
        competitor = standardized.masked_fill(~remaining, torch.inf).argmin(dim=1)

        comparison = class_indicator - torch.nn.functional.one_hot(
            competitor, num_classes=num_classes
        ).to(flat_mean.dtype)
        covariance_zu_a = torch.bmm(
            covariance_zu, comparison.unsqueeze(-1)
        ).squeeze(-1)
        covariance_uu_a = torch.bmm(
            covariance_uu, comparison.unsqueeze(-1)
        ).squeeze(-1)
        selected_variance = (
            comparison * covariance_uu_a
        ).sum(dim=1).clamp_min(eps)
        selected_scale = torch.sqrt(selected_variance)
        t = (comparison * mean_u).sum(dim=1) / selected_scale
        log_mass = log_mass + torch.special.log_ndtr(t)

        mills = _inverse_mills_ratio(t)
        contraction = (mills * (mills + t)).clamp(0.0, 1.0)
        gain_z = covariance_zu_a / selected_scale[:, None]
        gain_u = covariance_uu_a / selected_scale[:, None]
        mean_z = mean_z + gain_z * mills[:, None]
        mean_u = mean_u + gain_u * mills[:, None]
        covariance_zz = covariance_zz - contraction[:, None, None] * (
            gain_z.unsqueeze(-1) * gain_z.unsqueeze(-2)
        )
        covariance_zu = covariance_zu - contraction[:, None, None] * (
            gain_z.unsqueeze(-1) * gain_u.unsqueeze(-2)
        )
        covariance_uu = covariance_uu - contraction[:, None, None] * (
            gain_u.unsqueeze(-1) * gain_u.unsqueeze(-2)
        )
        covariance_zz = 0.5 * (
            covariance_zz + covariance_zz.transpose(-1, -2)
        )
        covariance_uu = 0.5 * (
            covariance_uu + covariance_uu.transpose(-1, -2)
        )
        remaining[rows, competitor] = False

    posterior_diagonal = covariance_zz.diagonal(dim1=-2, dim2=-1)
    covariance_zz = covariance_zz + torch.diag_embed(
        posterior_diagonal.clamp_min(eps) - posterior_diagonal
    )
    return (
        log_mass.reshape(leading_shape),
        mean_z.reshape(*leading_shape, num_classes),
        covariance_zz.reshape(*leading_shape, num_classes, num_classes),
    )


def multinomial_probit_adf_predictive_probs(
    output_mean: Tensor,
    output_covariance: Tensor,
    *,
    probit_tau2: float = 1.0,
    eps: float = _EPS,
) -> Tensor:
    """Run every candidate event independently and normalize its ADF mass."""

    dense_covariance = _dense_output_covariance(output_mean, output_covariance)
    num_classes = output_mean.shape[-1]
    leading_shape = output_mean.shape[:-1]
    candidate_mean = output_mean.unsqueeze(-2).expand(
        *leading_shape, num_classes, num_classes
    )
    candidate_covariance = dense_covariance.unsqueeze(-3).expand(
        *leading_shape, num_classes, num_classes, num_classes
    )
    labels = torch.arange(num_classes, device=output_mean.device).expand(
        *leading_shape, num_classes
    )
    log_mass, _, _ = multinomial_probit_adf_event(
        candidate_mean,
        candidate_covariance,
        labels,
        probit_tau2=probit_tau2,
        eps=eps,
    )
    return torch.softmax(log_mass, dim=-1)


def compute_multinomial_probit_adf_innovation(
    labels: Tensor,
    output_mean: Tensor,
    output_variance: Tensor,
    *,
    probit_tau2: float = 1.0,
    eps: float = _EPS,
) -> tuple[Tensor, Tensor]:
    """Project the dense event posterior to diagonal TAGI innovations."""

    if output_mean.shape != output_variance.shape:
        raise ValueError("TAGI ADF training expects matching diagonal output moments")
    _, posterior_mean, posterior_covariance = multinomial_probit_adf_event(
        output_mean,
        output_variance,
        labels,
        probit_tau2=probit_tau2,
        eps=eps,
    )
    active = output_variance > eps
    prior_variance = output_variance.clamp_min(eps)
    posterior_variance = posterior_covariance.diagonal(dim1=-2, dim2=-1)
    delta_mean = torch.where(
        active, (posterior_mean - output_mean) / prior_variance, 0.0
    )
    delta_variance = torch.where(
        active,
        (posterior_variance - output_variance) / prior_variance.square(),
        0.0,
    )
    return delta_mean, delta_variance
