"""Literal implementation of ``hierarchical_softmax_calibration.tex``.

This module intentionally implements the formulation in
``experiments/last_layer/HSM_calibration_Goulet_Nguyen_Florensa_annotated.tex``
rather than the positive-log-gain reformulation in :mod:`hsm_calibration`.

The uncertain gain itself is Gaussian, ``G ~ N(mu_G, v_G)``; ``G Z`` is
projected with the Gaussian multiplicative approximation; and calibration is
the sequential branch-wise ADF recursion from Eq. (update). There is no
prediction-time ``sigma_v`` in this formulation.

For a non-power-of-two tree the TeX prescribes normalizing class means, but it
does not derive moments of the normalized random vector. Consequently
``tex_hsm_class_moments`` returns the literal raw Eq. (class) moments, while
``tex_hsm_class_probabilities`` applies the stated normalization to the means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .hrc_softmax import HierarchicalSoftmax
from .hsm_calibration import GainGroups, node_depths, probit_gaussian_moments


TEX_GAIN_SHARING: tuple[str, ...] = ("global", "level", "node")
TEX_PROBABILITY_EPSILON: float = 1e-6
TEX_GAIN_FLOOR: float = 1e-12


def _path_arrays(
    hrc: HierarchicalSoftmax, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    node_index = hrc.idx.to(device).long() - 1
    if bool(((node_index < 0) | (node_index >= hrc.len)).any()):
        raise ValueError("HRC node index is out of range for the tree width")
    return node_index, hrc.obs.to(device).double(), hrc.path_mask(device).double()


def tex_gain_groups(
    hrc: HierarchicalSoftmax,
    sharing: str = "node",
    *,
    device: torch.device | str | None = None,
) -> GainGroups:
    """Return the global, per-level, or per-node sharing from the TeX."""

    resolved = torch.device(device) if device is not None else hrc.obs.device
    if sharing == "global":
        node_group = torch.zeros(hrc.len, dtype=torch.long, device=resolved)
    elif sharing == "level":
        node_group = node_depths(hrc, resolved)
    elif sharing == "node":
        node_group = torch.arange(hrc.len, dtype=torch.long, device=resolved)
    else:
        raise ValueError(f"sharing must be one of {TEX_GAIN_SHARING}")

    node_index, _, mask = _path_arrays(hrc, resolved)
    path_groups = node_group[node_index]
    repeats = False
    largest = 0
    for row, keep in zip(path_groups.tolist(), (mask > 0.0).tolist()):
        visited = [group for group, on_path in zip(row, keep) if on_path]
        repeats = repeats or len(set(visited)) != len(visited)
        largest = max(largest, len(set(visited)))
    return GainGroups(
        sharing=sharing,
        node_group=node_group,
        n_groups=int(node_group.max()) + 1,
        path_repeats=repeats,
        path_group_count=largest,
    )


@dataclass(frozen=True)
class GaussianGainPosterior:
    """Gaussian belief over the gain ``G`` in the TeX formulation."""

    mean: Tensor
    variance: Tensor
    groups: GainGroups
    visits: Tensor | None = None

    def __post_init__(self) -> None:
        if self.mean.shape != (self.groups.n_groups,):
            raise ValueError("mean must have one entry per gain group")
        if self.variance.shape != self.mean.shape:
            raise ValueError("variance must have the same shape as mean")
        if not bool(torch.isfinite(self.mean).all()):
            raise ValueError("gain means must be finite")
        if not bool(torch.isfinite(self.variance).all()) or bool((self.variance < 0).any()):
            raise ValueError("gain variances must be finite and nonnegative")

    @classmethod
    def prior(
        cls,
        groups: GainGroups,
        *,
        mean: float = 0.3,
        variance: float = 1.0,
    ) -> "GaussianGainPosterior":
        if not math.isfinite(mean):
            raise ValueError("mean must be finite")
        if variance < 0.0 or not math.isfinite(variance):
            raise ValueError("variance must be finite and nonnegative")
        device = groups.node_group.device
        return cls(
            mean=torch.full((groups.n_groups,), mean, dtype=torch.float64, device=device),
            variance=torch.full(
                (groups.n_groups,), variance, dtype=torch.float64, device=device
            ),
            groups=groups,
        )

    def node_mean(self, device: torch.device | str | None = None) -> Tensor:
        values = self.mean if device is None else self.mean.to(device)
        return self.groups.expand(values)

    def node_variance(self, device: torch.device | str | None = None) -> Tensor:
        values = self.variance if device is None else self.variance.to(device)
        return self.groups.expand(values)

    def deterministic(self) -> "GaussianGainPosterior":
        return replace(self, variance=torch.zeros_like(self.variance))


@dataclass(frozen=True)
class GaussianGainNodeMoments:
    """Literal moments in Eqs. (cond) and (node), for the left branch."""

    mean: Tensor
    variance: Tensor
    cov_state: Tensor
    cov_gain: Tensor
    log_mean: Tensor
    log_complement: Tensor


@dataclass(frozen=True)
class GaussianGainClassMoments:
    """Unnormalized class moments from Eq. (class)."""

    mean: Tensor
    variance: Tensor


def _check_node_inputs(mean: Tensor, variance: Tensor, hrc: HierarchicalSoftmax) -> None:
    if mean.shape != variance.shape or mean.dim() != 2 or mean.shape[1] != hrc.len:
        raise ValueError("node means and variances must have shape (batch, hrc.len)")
    if not bool(torch.isfinite(mean).all()):
        raise ValueError("node means must be finite")
    if not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
        raise ValueError("node variances must be finite and nonnegative")


def tex_hsm_node_moments(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
    posterior: GaussianGainPosterior,
) -> GaussianGainNodeMoments:
    """Evaluate Eqs. (S), (a), (cond), and (node) literally."""

    _check_node_inputs(output_mean, output_variance, hrc)
    mean_z = output_mean.double()
    variance_z = output_variance.double()
    mean_g = posterior.node_mean(output_mean.device)[None]
    variance_g = posterior.node_variance(output_mean.device)[None]
    mean_s = mean_g * mean_z
    variance_s = (
        variance_g * variance_z
        + variance_g * mean_z.square()
        + mean_g.square() * variance_z
    )
    probability_mean, probability_variance = probit_gaussian_moments(mean_s, variance_s)
    denominator = (1.0 + variance_s).pow(1.5)
    argument = mean_s / (1.0 + variance_s).sqrt()
    density = torch.exp(-0.5 * argument.square()) / math.sqrt(2.0 * math.pi)
    cov_state = (
        variance_z * mean_g * (1.0 + variance_s - variance_g * mean_z.square())
        / denominator * density
    )
    cov_gain = (
        variance_g * mean_z * (1.0 + variance_s - mean_g.square() * variance_z)
        / denominator * density
    )
    return GaussianGainNodeMoments(
        mean=probability_mean,
        variance=probability_variance,
        cov_state=cov_state,
        cov_gain=cov_gain,
        log_mean=torch.special.log_ndtr(argument),
        log_complement=torch.special.log_ndtr(-argument),
    )


def _path_products(left: Tensor, right: Tensor, hrc: HierarchicalSoftmax) -> Tensor:
    node_index, sign, mask = _path_arrays(hrc, left.device)
    total = left.new_ones((left.shape[0], hrc.n_classes))
    for position in range(hrc.n_obs):
        column = node_index[:, position]
        factor = torch.where(sign[:, position] > 0.0, left[:, column], right[:, column])
        total = total * torch.where(mask[:, position] > 0.0, factor, torch.ones_like(factor))
    return total


def tex_hsm_class_moments(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
    posterior: GaussianGainPosterior,
) -> GaussianGainClassMoments:
    """Return the literal, pre-normalization moments from Eq. (class)."""

    nodes = tex_hsm_node_moments(output_mean, output_variance, hrc, posterior)
    first = _path_products(nodes.mean, 1.0 - nodes.mean, hrc)
    second = _path_products(
        nodes.variance + nodes.mean.square(),
        nodes.variance + (1.0 - nodes.mean).square(),
        hrc,
    )
    return GaussianGainClassMoments(
        mean=first, variance=(second - first.square()).clamp_min(0.0)
    )


def tex_hsm_class_probabilities(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
    posterior: GaussianGainPosterior,
) -> Tensor:
    """Return class means with the TeX's non-power-of-two normalization."""

    mean = tex_hsm_class_moments(output_mean, output_variance, hrc, posterior).mean
    return mean / mean.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(mean.dtype).tiny)


def tex_hsm_adf_projection(
    mean: float,
    variance: float,
    branch_mean: float,
    cov_gain: float,
    *,
    probability_epsilon: float = TEX_PROBABILITY_EPSILON,
    gain_floor: float = TEX_GAIN_FLOOR,
) -> tuple[float, float]:
    """Apply Eq. (update), followed only by the TeX's numerical guards."""

    if not 0.0 < probability_epsilon < 0.5:
        raise ValueError("probability_epsilon must lie in (0, 0.5)")
    if gain_floor <= 0.0 or not math.isfinite(gain_floor):
        raise ValueError("gain_floor must be finite and positive")
    probability = min(max(branch_mean, probability_epsilon), 1.0 - probability_epsilon)
    updated_mean = max(mean + cov_gain / probability, gain_floor)
    updated_variance = max(
        variance - cov_gain * cov_gain / (probability * (1.0 - probability)), 0.0
    )
    return updated_mean, updated_variance


def _oriented_scalar_moments(
    node_mean: float,
    node_variance: float,
    sign: float,
    gain_mean: float,
    gain_variance: float,
) -> tuple[float, float]:
    variance_s = (
        gain_variance * node_variance
        + gain_variance * node_mean * node_mean
        + gain_mean * gain_mean * node_variance
    )
    argument = sign * gain_mean * node_mean / math.sqrt(1.0 + variance_s)
    probability = 0.5 * math.erfc(-argument / math.sqrt(2.0))
    density = math.exp(-0.5 * argument * argument) / math.sqrt(2.0 * math.pi)
    covariance = (
        sign * gain_variance * node_mean
        * (1.0 + variance_s - gain_mean * gain_mean * node_variance)
        / (1.0 + variance_s) ** 1.5 * density
    )
    return probability, covariance


def calibrate_tex_hsm_gain_adf(
    output_mean: Tensor,
    output_variance: Tensor,
    labels: Tensor,
    hrc: HierarchicalSoftmax,
    *,
    sharing: str = "node",
    prior_mean: float = 0.3,
    prior_variance: float = 1.0,
    initial_mean: Tensor | None = None,
    probability_epsilon: float = TEX_PROBABILITY_EPSILON,
    gain_floor: float = TEX_GAIN_FLOOR,
    visit_order: Tensor | None = None,
) -> GaussianGainPosterior:
    """Run Algorithm 1's sequential validation calibration pass.

    ``initial_mean`` implements "keep the mean, re-initialize the variance":
    each call starts at ``prior_variance`` while optionally retaining the mean
    from the previous validation pass.
    """

    _check_node_inputs(output_mean, output_variance, hrc)
    target = labels.reshape(-1).long().to(output_mean.device)
    if target.shape[0] != output_mean.shape[0]:
        raise ValueError("labels must have one entry per row")
    if bool(((target < 0) | (target >= hrc.n_classes)).any()):
        raise ValueError("labels contain an invalid class index")
    groups = tex_gain_groups(hrc, sharing, device=output_mean.device)
    if initial_mean is None:
        means = [float(prior_mean)] * groups.n_groups
    else:
        if initial_mean.shape != (groups.n_groups,):
            raise ValueError("initial_mean must have one entry per gain group")
        means = initial_mean.double().tolist()
    if prior_variance < 0.0 or not math.isfinite(prior_variance):
        raise ValueError("prior_variance must be finite and nonnegative")
    variances = [float(prior_variance)] * groups.n_groups
    visits = [0] * groups.n_groups
    node_index, sign, mask = _path_arrays(hrc, output_mean.device)
    sequence = (
        torch.arange(target.shape[0], device=target.device)
        if visit_order is None
        else visit_order.reshape(-1).long().to(target.device)
    )
    if sequence.shape[0] != target.shape[0]:
        raise ValueError("visit_order must contain one entry per calibration row")

    means_z = output_mean.double().tolist()
    variances_z = output_variance.double().tolist()
    node_columns, signs, masks = node_index.tolist(), sign.tolist(), mask.tolist()
    group_of = groups.node_group.tolist()
    for row in sequence.tolist():
        label = int(target[row])
        for position in range(hrc.n_obs):
            if masks[label][position] <= 0.0:
                continue
            node = int(node_columns[label][position])
            group = int(group_of[node])
            probability, covariance = _oriented_scalar_moments(
                means_z[row][node], variances_z[row][node], signs[label][position],
                means[group], variances[group]
            )
            means[group], variances[group] = tex_hsm_adf_projection(
                means[group], variances[group], probability, covariance,
                probability_epsilon=probability_epsilon, gain_floor=gain_floor
            )
            visits[group] += 1

    device = output_mean.device
    return GaussianGainPosterior(
        mean=torch.tensor(means, dtype=torch.float64, device=device),
        variance=torch.tensor(variances, dtype=torch.float64, device=device),
        groups=groups,
        visits=torch.tensor(visits, dtype=torch.long, device=device),
    )
