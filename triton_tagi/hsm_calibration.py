"""Hierarchical probit calibration for TAGI classification.

Implements the calibration module of Goulet, Nguyen and Florensa-Montilla,
*Hierarchical Probit Calibration for Tractable Approximate Gaussian
Inference*. On a proper binary tree with exactly ``K`` leaves, a frozen
hierarchical head carries a Gaussian output state ``Z_n ~ N(mu_n, v_n)`` at
every internal node, and calibration is one Gaussian log-gain per sharing
group::

    L_r ~ N(lambda_r, q_r),   G_r = exp(L_r) > 0
    R_n = G_{r(n)} Z_n + sigma_v o_n + eps_n,   eps_n ~ N(0, sigma_v^2)
    P_{c,n} = Pr(s_{c,n} R_n >= 0)
            = Phi(s_{c,n} (G_{r(n)} Z_n / sigma_v + o_n))
    Q_c = prod_{n in path(c)} P_{c,n}

with ``s_{c,n} = +1`` on the left branch and ``-1`` on the right, and ``o_n``
the deterministic branch probit of
:func:`triton_tagi.hrc_softmax.class_to_obs_full`. Positivity is structural: no
posterior draw can reverse the two branches. A single global gain is the
positive-gain analogue of temperature scaling; the per-level and per-node
groupings are progressively more flexible and need correspondingly more
calibration data.

``sigma_v`` is the **fixed** noise of the branch channel the head was trained
against, carried by :class:`LogGainPosterior` so that a belief can never be fitted
at one scale and read at another. It is not a free parameter and is never
inferred here: the calibration infers ``L`` only.

    * The base HRC head of :mod:`triton_tagi.hrc_softmax` observes ``+/-1`` at
      each node under a fixed observation noise, so ``sigma_v`` is that noise
      and ``lambda = 0`` is the head exactly as trained. Marginalizing the state
      gives ``Phi(mu_n / sqrt(sigma_v^2 + v_n))``, which is
      :func:`triton_tagi.hrc_softmax.obs_to_class_probs` at ``alpha = 1 /
      sigma_v``. cuTAGI reads out at a hard-coded ``alpha = 3`` whatever noise it
      trained with; passing the trained ``sigma_v`` here is what makes the
      calibrated model and the training channel the same model.
    * The hierarchical-probit head of :mod:`triton_tagi.hrc_probit` fixes its
      latent unit instead, which is ``sigma_v = 1``, the default.

Since only ``G / sigma_v`` enters, a belief at ``sigma_v`` is the exact
reparameterization ``lambda -> lambda - log sigma_v`` of the unit-scale belief;
what the scale fixes is the anchor, meaning which gain counts as "the head as
trained" and where the prior is centred, not the family of reachable models.

Because every node splits its class set exhaustively, ``sum_c Q_c = 1`` holds
pointwise for every state and gain realization, so ``sum_c E[Q_c] = 1`` with no
post-hoc normalizer. The tree must therefore be proper; the padded tree of
:func:`triton_tagi.hrc_softmax.class_to_obs` discards leaves that hold
probability mass and is rejected here.

Every moment in this module is exact for Gaussian ``Z_n`` and ``L_r`` up to
one-dimensional quadrature. The gain is not folded into the state by a
product-Gaussian approximation: node moments integrate the scalar log-gain by
Gauss--Hermite, and the conditional probability moments themselves are the
closed forms

    E[Phi(X)]   = Phi(a),                     a = m / sqrt(1 + S)
    Var[Phi(X)] = Phi(a) (1 - Phi(a)) - 2 T(a, b),   b = 1 / sqrt(1 + 2 S)

for ``X ~ N(m, S)`` and Owen's ``T``. The variance is evaluated through the
equivalent positive integral so that no two nearly equal numbers are
subtracted.

The gain is the reciprocal of the latent probit scale that
:mod:`triton_tagi.hrc_probit` already carries, ``G = sigma_v / tau`` and
``L = log sigma_v - log tau``, so at ``sigma_v = 1`` and ``q = 0`` this
reproduces ``hrc_log_probs(..., log_tau=-lambda)`` to double precision. At any
scale ``lambda = 0`` with ``q = 0`` is the uncalibrated head, whose effective
probit scale is ``alpha = sigma_v exp(-lambda)``.

Calibration itself uses the held-out Bernoulli channel. The observation is the
binary event "this branch was taken", never ``P``, ``L``, or ``G``, and it is
the same semantic label the network already trained on, so it belongs on a
disjoint calibration split with the network frozen. Since every gain group is
scalar, the order-independent batch fit of :func:`fit_hsm_log_gain` is
preferred; :func:`calibrate_hsm_log_gain_adf` provides the sequential
assumed-density alternative for a genuine stream.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .hrc_softmax import HierarchicalSoftmax

DEFAULT_GAIN_ORDER: int = 20
DEFAULT_OWEN_ORDER: int = 48
DEFAULT_GRID_HALF_WIDTH: float = 6.0
DEFAULT_GRID_SIZE: int = 513
DEFAULT_GRID_REFINEMENTS: int = 2
# Blocks are sized by element count, not row count: the working tensors are
# (rows, nodes, order), so a fixed row budget is simultaneously too small to
# amortize the kernel launches on a ten-leaf tree and too large to fit a
# thousand-leaf one. Two million float64 elements is sixteen megabytes per
# working tensor.
DEFAULT_CHUNK_ELEMENTS: int = 2_000_000

GAIN_SHARING: tuple[str, ...] = ("global", "level", "node")

_WINDOW_REACH: float = 16.0
_LOG_TWO_PI: float = math.log(2.0 * math.pi)
_INV_SQRT_TWO: float = 1.0 / math.sqrt(2.0)
_QUARTER_PI: float = 0.25 * math.pi
_PROBABILITY_FLOOR: float = 1e-300
_CURVATURE_FLOOR: float = 1e-12


# ──────────────────────────────────────────────────────────────────────────────
#  Quadrature rules
# ──────────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=None)
def _hermite_rule(order: int) -> tuple[Tensor, Tensor]:
    """Return standard-normal nodes and weights summing to one, CPU float64."""

    if order < 4:
        raise ValueError("quadrature order must be at least four")
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    return (
        torch.from_numpy(np.ascontiguousarray(nodes)),
        torch.from_numpy(np.ascontiguousarray(weights / math.sqrt(2.0 * math.pi))),
    )


@functools.lru_cache(maxsize=None)
def _legendre_rule(order: int) -> tuple[Tensor, Tensor]:
    """Return Gauss--Legendre nodes and weights on ``[-1, 1]``, CPU float64."""

    if order < 4:
        raise ValueError("quadrature order must be at least four")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (
        torch.from_numpy(np.ascontiguousarray(nodes)),
        torch.from_numpy(np.ascontiguousarray(weights)),
    )


def _resolve_chunk_size(
    hrc: HierarchicalSoftmax,
    order: int,
    chunk_size: int | None,
) -> int:
    """Return the block height, derived from the tree width when unset."""

    if chunk_size is not None:
        return max(1, int(chunk_size))
    return max(1, DEFAULT_CHUNK_ELEMENTS // max(1, hrc.len * order))


def _placed_rule(
    rule: tuple[Tensor, Tensor],
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    nodes, weights = rule
    return nodes.to(reference.device), weights.to(reference.device)


# ──────────────────────────────────────────────────────────────────────────────
#  Exact moments of Phi(X) for Gaussian X
# ──────────────────────────────────────────────────────────────────────────────


def _angular_integral(argument: Tensor, lower: Tensor, upper: Tensor, order: int) -> Tensor:
    """Return ``int_lower^upper exp(-argument^2 / (2 cos^2 theta)) d theta``.

    The substitution ``t = tan theta`` removes the ``1 / (1 + t^2)`` factor from
    both Owen's ``T`` and the positive variance integral, leaving a smooth
    bounded integrand on a finite interval. The Legendre nodes are consumed one
    at a time so that no tensor of shape ``(..., order)`` is materialized.
    """

    half = 0.5 * (upper - lower)
    centre = 0.5 * (upper + lower)
    squared = argument.square()
    nodes, weights = _placed_rule(_legendre_rule(order), argument)
    total = torch.zeros_like(argument)
    for node, weight in zip(nodes.tolist(), weights.tolist()):
        cosine = torch.cos(centre + half * node)
        total = total + weight * torch.exp(-0.5 * squared / cosine.square())
    return half * total


def owens_t(
    h: Tensor | float,
    c: Tensor | float,
    *,
    order: int = DEFAULT_OWEN_ORDER,
) -> Tensor:
    """Return Owen's T function.

    ``T(h, c) = (1 / 2 pi) int_0^c exp[-h^2 (1 + t^2) / 2] / (1 + t^2) dt``,
    which is even in ``h`` and odd in ``c``.

    Args:
        h: First argument, any broadcastable shape.
        c: Second argument, any broadcastable shape.
        order: Gauss--Legendre order for the angular integral.

    Returns:
        ``T(h, c)`` in float64, broadcast to the common shape.
    """

    first = torch.as_tensor(h, dtype=torch.float64)
    second = torch.as_tensor(c, dtype=torch.float64).to(first.device)
    first, second = torch.broadcast_tensors(first, second.double())
    upper = second.abs().atan()
    value = _angular_integral(first, torch.zeros_like(upper), upper, order) / (2.0 * math.pi)
    return torch.where(second < 0.0, -value, value)


def probit_gaussian_moments(
    mean: Tensor,
    variance: Tensor,
    *,
    order: int = DEFAULT_OWEN_ORDER,
) -> tuple[Tensor, Tensor]:
    """Return ``E[Phi(X)]`` and ``Var[Phi(X)]`` for ``X ~ N(mean, variance)``.

    The mean is ``Phi(a)`` with ``a = mean / sqrt(1 + variance)``. The variance
    is evaluated as the positive integral

        ``Var[Phi(X)] = (1 / pi) int_b^1 exp[-a^2 (1 + t^2) / 2] / (1 + t^2) dt``

    with ``b = 1 / sqrt(1 + 2 variance)``, which equals
    ``Phi(a) (1 - Phi(a)) - 2 T(a, b)`` without subtracting nearly equal
    numbers.

    Args:
        mean: Mean of ``X``.
        variance: Variance of ``X``, nonnegative.
        order: Gauss--Legendre order for the variance integral.

    Returns:
        first: ``E[Phi(X)]``.
        second: ``Var[Phi(X)]``, clamped into ``[0, first (1 - first)]``.
    """

    work_mean = mean.double()
    work_variance = variance.double().clamp_min(0.0)
    argument = work_mean / (1.0 + work_variance).sqrt()
    lower = (1.0 / (1.0 + 2.0 * work_variance).sqrt()).atan()
    upper = torch.full_like(lower, _QUARTER_PI)
    first = torch.special.ndtr(argument)
    second = _angular_integral(argument, lower, upper, order) / math.pi
    return first, second.clamp_min(0.0).minimum(first * (1.0 - first))


def expected_bernoulli_variance(mean: Tensor, variance: Tensor) -> Tensor:
    """Return ``R = E[P (1 - P)] = mean (1 - mean) - variance``.

    The law of total variance splits the variance of the observed branch
    indicator into ``R``, the average label randomness left conditionally on the
    latent probability, and ``variance``, the epistemic dispersion of that
    probability. Their sum is ``mean (1 - mean)``.
    """

    return (mean * (1.0 - mean) - variance).clamp_min(0.0)


# ──────────────────────────────────────────────────────────────────────────────
#  Tree geometry and gain-sharing groups
# ──────────────────────────────────────────────────────────────────────────────


def _require_proper_tree(hrc: HierarchicalSoftmax, caller: str) -> None:
    if not hrc.is_full:
        raise ValueError(
            f"{caller} requires a proper K-leaf tree from class_to_obs_full; the padded "
            "tree of class_to_obs discards leaves that hold probability mass"
        )


def _path_arrays(
    hrc: HierarchicalSoftmax,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the per-class node indices, branch signs, and padding mask."""

    node_index = hrc.idx.to(device).long() - 1
    if bool(((node_index < 0) | (node_index >= hrc.len)).any()):
        raise ValueError("HRC node index is out of range for the tree width")
    return node_index, hrc.obs.to(device).double(), hrc.path_mask(device).double()


def branch_codes(hrc: HierarchicalSoftmax, device: torch.device | str | None = None) -> Tensor:
    """Return the ``(n_classes, len)`` branch code of every class path.

    Entry ``(c, n)`` is ``+1`` when class ``c`` takes the left branch at node
    ``n``, ``-1`` for the right branch, and ``0`` when node ``n`` is not on the
    path of class ``c``.
    """

    resolved = torch.device(device) if device is not None else hrc.obs.device
    node_index, sign, mask = _path_arrays(hrc, resolved)
    codes = torch.zeros(hrc.n_classes, hrc.len, dtype=torch.float64, device=resolved)
    return codes.scatter_add_(1, node_index, sign * mask)


def node_depths(hrc: HierarchicalSoftmax, device: torch.device | str | None = None) -> Tensor:
    """Return the ``(len,)`` depth of every internal node, root at zero."""

    resolved = torch.device(device) if device is not None else hrc.obs.device
    node_index, _, mask = _path_arrays(hrc, resolved)
    positions = torch.arange(hrc.n_obs, device=resolved).expand(hrc.n_classes, hrc.n_obs)
    keep = mask.reshape(-1) > 0.0
    depths = torch.zeros(hrc.len, dtype=torch.long, device=resolved)
    return depths.scatter_(0, node_index.reshape(-1)[keep], positions.reshape(-1)[keep])


@dataclass(frozen=True)
class GainGroups:
    """Assignment of tree nodes to gain-sharing groups.

    Attributes:
        sharing: Name of the grouping, one of :data:`GAIN_SHARING` or ``custom``.
        node_group: Group index of every node, shape (len,), dtype int64.
        n_groups: Number of groups.
        path_repeats: True when some class path visits one group twice, which
            makes the path factors dependent even between distinct nodes.
        path_group_count: Largest number of distinct groups on a single path.
    """

    sharing: str
    node_group: Tensor
    n_groups: int
    path_repeats: bool
    path_group_count: int

    @property
    def is_global(self) -> bool:
        """True when one gain governs every node of the tree."""

        return self.n_groups == 1

    def expand(self, values: Tensor) -> Tensor:
        """Broadcast one value per group to one value per node."""

        if values.shape[-1] != self.n_groups:
            raise ValueError("values must have one entry per gain group")
        return values.index_select(-1, self.node_group.to(values.device))

    def to(self, device: torch.device | str) -> GainGroups:
        """Return the same grouping with its index tensor on ``device``."""

        return replace(self, node_group=self.node_group.to(device))


def gain_groups(
    hrc: HierarchicalSoftmax,
    sharing: str = "global",
    *,
    device: torch.device | str | None = None,
) -> GainGroups:
    """Build a gain-sharing grouping over the internal nodes of a tree.

    Args:
        hrc: Proper K-leaf tree structure.
        sharing: ``global`` for one gain, ``level`` for one gain per depth, or
            ``node`` for one gain per internal node.
        device: Device for the index tensor; defaults to the tree's device.

    Returns:
        The grouping, with the path-dependence flags the class moments need.
    """

    _require_proper_tree(hrc, "gain_groups")
    resolved = torch.device(device) if device is not None else hrc.obs.device
    if sharing == "global":
        node_group = torch.zeros(hrc.len, dtype=torch.long, device=resolved)
    elif sharing == "level":
        node_group = node_depths(hrc, resolved)
    elif sharing == "node":
        node_group = torch.arange(hrc.len, dtype=torch.long, device=resolved)
    else:
        raise ValueError(f"sharing must be one of {GAIN_SHARING}")
    return custom_gain_groups(hrc, node_group, name=sharing)


def custom_gain_groups(
    hrc: HierarchicalSoftmax,
    node_group: Tensor,
    *,
    name: str = "custom",
) -> GainGroups:
    """Wrap an explicit node-to-group assignment, measuring path dependence."""

    _require_proper_tree(hrc, "custom_gain_groups")
    groups = node_group.reshape(-1).long()
    if groups.shape[0] != hrc.len:
        raise ValueError("node_group must have one entry per internal node")
    if bool((groups < 0).any()):
        raise ValueError("node_group entries must be nonnegative")
    n_groups = int(groups.max()) + 1
    node_index, _, mask = _path_arrays(hrc, groups.device)
    path_groups = groups[node_index]
    repeats = False
    largest = 0
    for row, keep in zip(path_groups.tolist(), (mask > 0.0).tolist()):
        visited = [group for group, on_path in zip(row, keep) if on_path]
        distinct = set(visited)
        repeats = repeats or len(distinct) != len(visited)
        largest = max(largest, len(distinct))
    return GainGroups(
        sharing=name,
        node_group=groups,
        n_groups=n_groups,
        path_repeats=repeats,
        path_group_count=largest,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Log-gain belief
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LogGainPosterior:
    """Gaussian belief over the log-gain of every sharing group.

    Attributes:
        mean: Group log-gain means ``lambda_r``, shape (n_groups,).
        variance: Group log-gain variances ``q_r``, shape (n_groups,).
        groups: The node-to-group assignment these parameters refer to.
        visits: Calibration node-visits per group, shape (n_groups,), or None.
            The paper asks for this alongside any empirical result: deep
            per-node gains can be weakly identified even when the calibration
            set is large.
        sigma_v: Fixed noise of the branch channel the gain is measured
            against. One is the unit-probit convention of ``hrc_probit``; the
            base HRC head passes the observation noise it was trained with, and
            then ``lambda = 0`` is that head unchanged. It travels with the
            belief so that a gain fitted at one scale cannot be read at another.
    """

    mean: Tensor
    variance: Tensor
    groups: GainGroups
    visits: Tensor | None = None
    sigma_v: float = 1.0

    def __post_init__(self) -> None:
        if self.mean.shape != (self.groups.n_groups,):
            raise ValueError("mean must have one entry per gain group")
        if self.variance.shape != self.mean.shape:
            raise ValueError("variance must have the same shape as mean")
        if not bool(torch.isfinite(self.mean).all()):
            raise ValueError("log-gain means must be finite")
        if not bool(torch.isfinite(self.variance).all()) or bool((self.variance < 0).any()):
            raise ValueError("log-gain variances must be finite and nonnegative")
        if not math.isfinite(self.sigma_v) or self.sigma_v <= 0.0:
            raise ValueError("sigma_v must be finite and positive")

    @classmethod
    def prior(
        cls,
        groups: GainGroups,
        *,
        mean: float = 0.0,
        variance: float = 1.0,
        sigma_v: float = 1.0,
    ) -> LogGainPosterior:
        """Return an isotropic prior; ``mean = 0`` is the head as trained."""

        if variance < 0.0 or not math.isfinite(variance):
            raise ValueError("variance must be finite and nonnegative")
        device = groups.node_group.device
        return cls(
            mean=torch.full((groups.n_groups,), float(mean), dtype=torch.float64, device=device),
            variance=torch.full(
                (groups.n_groups,), float(variance), dtype=torch.float64, device=device
            ),
            groups=groups,
            sigma_v=float(sigma_v),
        )

    @classmethod
    def unit(cls, hrc: HierarchicalSoftmax, sigma_v: float = 1.0) -> LogGainPosterior:
        """Return the deterministic unit gain, which is the head as trained.

        At ``sigma_v = 1`` this is the uncalibrated unit-probit head. For the
        base HRC head, pass the observation noise it was trained with and the
        result reads out as ``Phi(mu_n / sqrt(sigma_v^2 + v_n))``, which is
        :func:`triton_tagi.hrc_softmax.obs_to_class_probs` at
        ``alpha = 1 / sigma_v``.
        """

        return cls.prior(gain_groups(hrc, "global"), mean=0.0, variance=0.0, sigma_v=sigma_v)

    def node_mean(self, device: torch.device | str | None = None) -> Tensor:
        """Return ``lambda_{r(n)}`` for every node, shape (len,)."""

        values = self.mean if device is None else self.mean.to(device)
        return self.groups.expand(values)

    def node_variance(self, device: torch.device | str | None = None) -> Tensor:
        """Return ``q_{r(n)}`` for every node, shape (len,)."""

        values = self.variance if device is None else self.variance.to(device)
        return self.groups.expand(values)

    @property
    def gain_median(self) -> Tensor:
        """Posterior median gain ``exp(lambda)``."""

        return self.mean.exp()

    @property
    def gain_mean(self) -> Tensor:
        """Posterior mean gain ``exp(lambda + q / 2)``."""

        return (self.mean + 0.5 * self.variance).exp()

    @property
    def alpha_median(self) -> Tensor:
        """Median of ``alpha = sigma_v / G``, the effective latent probit scale.

        This is the scale the head reads out at: the calibrated node factor is
        ``Phi(mu_n / sqrt(alpha^2 + v_n))`` once the gain is fixed at its
        median. The posterior mean of ``alpha`` is ``sigma_v exp(-lambda +
        q / 2)`` instead; neither should be reported as ``sigma_v / E[G]``.
        """

        return self.sigma_v * (-self.mean).exp()

    def deterministic(self) -> LogGainPosterior:
        """Return the same means with every variance set to zero.

        Reading the fitted gain as a point value is the positive-gain analogue
        of temperature scaling and isolates what the gain uncertainty buys.
        """

        return replace(self, variance=torch.zeros_like(self.variance))

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly record of the belief and its identifiability."""

        record: dict[str, Any] = {
            "sharing": self.groups.sharing,
            "n_groups": self.groups.n_groups,
            "sigma_v": self.sigma_v,
            "log_gain_mean": self.mean.tolist(),
            "log_gain_variance": self.variance.tolist(),
            "gain_median": self.gain_median.tolist(),
            "alpha_median": self.alpha_median.tolist(),
        }
        if self.visits is not None:
            record["visits"] = self.visits.tolist()
        return record


# ──────────────────────────────────────────────────────────────────────────────
#  Node moments with a positive uncertain gain
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HsmNodeMoments:
    """Left-branch node moments of ``U_n = Phi(exp(L) Z_n + o_n)``.

    Reversing the branch sign replaces ``U_n`` by ``1 - U_n``: the mean is
    complemented, the variance is unchanged, and both covariances change sign.

    Both orientations are also carried in the log domain. A saturated node has
    ``E[U_n]`` indistinguishable from one in double precision, so the
    complement of the mean cannot resolve the branch that was not taken;
    ``log_complement`` is a separate log-sum-exp over the same quadrature and
    stays accurate into the far tail, which is where a confident head puts the
    probability of a misclassified example.

    Attributes:
        mean: ``E[U_n]``, shape (batch, len).
        variance: ``Var(U_n)``, shape (batch, len).
        cov_state: ``Cov(Z_n, U_n)``, shape (batch, len).
        cov_log_gain: ``Cov(L_{r(n)}, U_n)``, shape (batch, len).
        log_mean: ``log E[U_n]``, shape (batch, len).
        log_complement: ``log E[1 - U_n]``, shape (batch, len).
    """

    mean: Tensor
    variance: Tensor
    cov_state: Tensor
    cov_log_gain: Tensor
    log_mean: Tensor
    log_complement: Tensor

    @property
    def expected_bernoulli_variance(self) -> Tensor:
        """``R = E[U_n (1 - U_n)]``, the residual label randomness per node."""

        return expected_bernoulli_variance(self.mean, self.variance)


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


def _conditional_node_terms(
    mean: Tensor,
    variance: Tensor,
    offset: Tensor,
    log_gain: Tensor,
    sigma_v: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``a_n(l)``, the conditional latent variance, and ``g(l) / d_n(l)``.

    Only the ratio ``g(l) = exp(l) / sigma_v`` enters, because the branch event
    ``s (G Z_n + sigma_v o_n + eps_n) >= 0`` is invariant to dividing through by
    the fixed channel noise. The offset is a probit and is therefore already in
    those units, which is what makes a zero network output telescope to the
    branch prior at any gain and any ``sigma_v``.

    Args:
        mean: Node means, shape (batch, len).
        variance: Node variances, shape (batch, len).
        offset: Branch offsets, shape (len,).
        log_gain: Log-gain quadrature nodes, shape (len, order) or (order,).
        sigma_v: Fixed channel noise, positive.

    Returns:
        argument: ``a_n(l) = (g(l) mu_n + o_n) / d_n(l)``, shape (batch, len, order).
        latent_variance: ``g(l)^2 v_n``, same shape.
        scaled_gain: ``g(l) / d_n(l)``, same shape.
    """

    gain = log_gain.exp() / sigma_v
    latent_variance = gain.square() * variance[..., None]
    scale = (1.0 + latent_variance).sqrt()
    argument = (gain * mean[..., None] + offset[:, None]) / scale
    return argument, latent_variance, gain / scale


def _node_moment_block(
    mean: Tensor,
    variance: Tensor,
    offset: Tensor,
    log_gain_mean: Tensor,
    log_gain_variance: Tensor,
    nodes: Tensor,
    weights: Tensor,
    owen_order: int,
    sigma_v: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    log_gain = log_gain_mean[:, None] + log_gain_variance.sqrt()[:, None] * nodes
    argument, latent_variance, scaled_gain = _conditional_node_terms(
        mean, variance, offset, log_gain, sigma_v
    )
    conditional_mean = torch.special.ndtr(argument)
    lower = (1.0 / (1.0 + 2.0 * latent_variance).sqrt()).atan()
    upper = torch.full_like(lower, _QUARTER_PI)
    conditional_variance = _angular_integral(argument, lower, upper, owen_order) / math.pi
    density = torch.exp(-0.5 * argument.square() - 0.5 * _LOG_TWO_PI)

    probability_mean = (weights * conditional_mean).sum(-1)
    residual = conditional_mean - probability_mean[..., None]
    # E[Var(U | L)] + Var(E[U | L]): both parts are sums of nonnegative terms,
    # so the total never relies on cancelling two nearly equal numbers.
    probability_variance = (
        (weights * conditional_variance.clamp_min(0.0)).sum(-1)
        + (weights * residual.square()).sum(-1)
    ).clamp_min(0.0)
    cov_state = variance * (weights * scaled_gain * density).sum(-1)
    cov_log_gain = (weights * (log_gain - log_gain_mean[:, None]) * conditional_mean).sum(-1)
    log_weights = weights.log()
    log_mean = torch.logsumexp(log_weights + torch.special.log_ndtr(argument), dim=-1)
    log_complement = torch.logsumexp(log_weights + torch.special.log_ndtr(-argument), dim=-1)
    return (
        probability_mean,
        probability_variance,
        cov_state,
        cov_log_gain,
        log_mean,
        log_complement,
    )


def hsm_node_moments(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
    posterior: LogGainPosterior,
    *,
    order: int = DEFAULT_GAIN_ORDER,
    owen_order: int = DEFAULT_OWEN_ORDER,
    chunk_size: int | None = None,
) -> HsmNodeMoments:
    """Return the left-branch node moments of the calibrated probit model.

    Integrating only the scalar log-gain gives the exact identities

        ``mu_P = E_L[Phi(a_n(L))]``
        ``v_P = E_L[Var(U_n | L)] + Var_L(E[U_n | L])``
        ``Cov(Z_n, U_n) = v_n E_L[exp(L) phi(a_n(L)) / d_n(L)]``
        ``Cov(L, U_n) = E_L[(L - lambda) Phi(a_n(L))]``

    with ``g(l) = exp(l) / sigma_v``, ``a_n(l) = (g(l) mu_n + o_n) / d_n(l)``
    and ``d_n(l) = sqrt(1 + g(l)^2 v_n)``, reading ``sigma_v`` off the belief.
    Setting ``q = 0`` recovers the deterministic positive-gain head exactly, and
    ``lambda = 0`` with ``q = 0`` is the frozen head at its own channel noise.

    Args:
        output_mean: Frozen node means, shape (batch, hrc.len).
        output_variance: Frozen node variances, shape (batch, hrc.len).
        hrc: Proper K-leaf tree structure.
        posterior: Gaussian belief over the group log-gains.
        order: Gauss--Hermite order for the log-gain integral. Raise it until
            every reported moment is stable at the target precision.
        owen_order: Gauss--Legendre order for the conditional variance.
        chunk_size: Rows per block; ``None`` derives one from the tree width
            and :data:`DEFAULT_CHUNK_ELEMENTS`.

    Returns:
        The node moments in float64.
    """

    _require_proper_tree(hrc, "hsm_node_moments")
    _check_node_moments(output_mean, output_variance, hrc)
    device = output_mean.device
    offset = hrc.node_offset(device).double()
    log_gain_mean = posterior.node_mean(device)
    log_gain_variance = posterior.node_variance(device)
    nodes, weights = _placed_rule(_hermite_rule(order), output_mean)
    rows = _resolve_chunk_size(hrc, order, chunk_size)

    if output_mean.shape[0] == 0:
        empty = output_mean.new_zeros((0, hrc.len), dtype=torch.float64)
        return HsmNodeMoments(*(empty.clone() for _ in range(6)))

    parts: list[tuple[Tensor, ...]] = []
    for start in range(0, output_mean.shape[0], rows):
        parts.append(
            _node_moment_block(
                output_mean[start : start + rows].double(),
                output_variance[start : start + rows].double().clamp_min(0.0),
                offset,
                log_gain_mean,
                log_gain_variance,
                nodes,
                weights,
                owen_order,
                posterior.sigma_v,
            )
        )
    return HsmNodeMoments(*(torch.cat(block) for block in zip(*parts)))


# ──────────────────────────────────────────────────────────────────────────────
#  Class moments
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HsmClassMoments:
    """Moments of the latent class probabilities ``Q_c``.

    ``mean`` is the posterior predictive class probability and sums to one
    across classes. ``variance`` is the epistemic dispersion of the latent
    conditional probability, not the variance of the observed class indicator,
    which is ``mean (1 - mean)``.
    """

    mean: Tensor
    variance: Tensor

    @property
    def expected_bernoulli_variance(self) -> Tensor:
        """``E[Q_c (1 - Q_c)]``, the residual randomness of the class indicator."""

        return expected_bernoulli_variance(self.mean, self.variance)


def _oriented_log_terms(
    log_left: Tensor,
    log_right: Tensor,
    variance: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return log first and second moments for both branch orientations.

    The second moments are assembled with ``logaddexp`` so that a branch
    probability far below the square root of machine epsilon still contributes
    its own magnitude rather than a floor.
    """

    log_variance = variance.clamp_min(0.0).log()
    return (
        log_left,
        log_right,
        torch.logaddexp(log_variance, 2.0 * log_left),
        torch.logaddexp(log_variance, 2.0 * log_right),
    )


def _accumulate_path_logs(
    log_left: Tensor,
    log_right: Tensor,
    node_index: Tensor,
    sign: Tensor,
    mask: Tensor,
) -> Tensor:
    """Sum oriented per-node logs along every class path.

    Args:
        log_left: Left-orientation logs, shape (batch, len) or (batch, len, order).
        log_right: Right-orientation logs, same shape.
        node_index: Path node indices, shape (n_classes, depth).
        sign: Path branch signs, shape (n_classes, depth).
        mask: Path padding mask, shape (n_classes, depth).

    Returns:
        Summed logs, shape (batch, n_classes) or (batch, n_classes, order).
    """

    n_classes, depth = node_index.shape
    trailing = log_left.shape[2:]
    total = log_left.new_zeros((log_left.shape[0], n_classes, *trailing))
    for position in range(depth):
        column = node_index[:, position]
        take_left = (sign[:, position] > 0.0).reshape(-1, *(1,) * len(trailing))
        factor = torch.where(take_left, log_left[:, column], log_right[:, column])
        total = total + mask[:, position].reshape(-1, *(1,) * len(trailing)) * factor
    return total


def _class_moments_independent(
    node_moments: HsmNodeMoments,
    hrc: HierarchicalSoftmax,
) -> HsmClassMoments:
    node_index, sign, mask = _path_arrays(hrc, node_moments.mean.device)
    log_left, log_right, log_second_left, log_second_right = _oriented_log_terms(
        node_moments.log_mean, node_moments.log_complement, node_moments.variance
    )
    log_mean = _accumulate_path_logs(log_left, log_right, node_index, sign, mask)
    log_second = _accumulate_path_logs(
        log_second_left, log_second_right, node_index, sign, mask
    )
    mean = log_mean.exp()
    return HsmClassMoments(mean=mean, variance=(log_second.exp() - mean.square()).clamp_min(0.0))


def _global_conditional_terms(
    mean: Tensor,
    variance: Tensor,
    offset: Tensor,
    log_gain: Tensor,
    owen_order: int,
    sigma_v: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return conditional log moments of both orientations for a single gain."""

    argument, latent_variance, _ = _conditional_node_terms(
        mean, variance, offset, log_gain, sigma_v
    )
    lower = (1.0 / (1.0 + 2.0 * latent_variance).sqrt()).atan()
    upper = torch.full_like(lower, _QUARTER_PI)
    conditional_variance = (
        _angular_integral(argument, lower, upper, owen_order) / math.pi
    ).clamp_min(0.0)
    log_left = torch.special.log_ndtr(argument)
    log_right = torch.special.log_ndtr(-argument)
    return _oriented_log_terms(log_left, log_right, conditional_variance)


def _class_moments_global(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
    posterior: LogGainPosterior,
    order: int,
    owen_order: int,
    chunk_size: int,
) -> HsmClassMoments:
    device = output_mean.device
    offset = hrc.node_offset(device).double()
    node_index, sign, mask = _path_arrays(hrc, device)
    nodes, weights = _placed_rule(_hermite_rule(order), output_mean)
    log_gain = posterior.mean[0].to(device) + posterior.variance[0].to(device).sqrt() * nodes

    if output_mean.shape[0] == 0:
        empty = output_mean.new_zeros((0, hrc.n_classes), dtype=torch.float64)
        return HsmClassMoments(mean=empty, variance=empty.clone())

    means: list[Tensor] = []
    variances: list[Tensor] = []
    for start in range(0, output_mean.shape[0], chunk_size):
        block_mean = output_mean[start : start + chunk_size].double()
        block_variance = output_variance[start : start + chunk_size].double().clamp_min(0.0)
        log_left, log_right, log_second_left, log_second_right = _global_conditional_terms(
            block_mean, block_variance, offset, log_gain, owen_order, posterior.sigma_v
        )
        log_path = _accumulate_path_logs(log_left, log_right, node_index, sign, mask)
        log_path_second = _accumulate_path_logs(
            log_second_left, log_second_right, node_index, sign, mask
        )
        mean = (weights * log_path.exp()).sum(-1)
        second = (weights * log_path_second.exp()).sum(-1)
        means.append(mean)
        variances.append((second - mean.square()).clamp_min(0.0))
    return HsmClassMoments(mean=torch.cat(means), variance=torch.cat(variances))


def hsm_class_moments(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
    posterior: LogGainPosterior,
    *,
    order: int = DEFAULT_GAIN_ORDER,
    owen_order: int = DEFAULT_OWEN_ORDER,
    chunk_size: int | None = None,
) -> HsmClassMoments:
    """Return the class-probability moments of the calibrated probit model.

    When no class path visits a gain group twice, which covers per-node and
    per-level gains, the path factors are independent and the class moments are
    the exact products

        ``mu_Q = prod_n mu_{P_n}``
        ``v_Q = prod_n (v_{P_n} + mu_{P_n}^2) - prod_n mu_{P_n}^2``

    A single global gain makes the factors independent only conditionally on
    ``L``, so every class moment is one one-dimensional quadrature of a
    conditional product and must not be assembled from marginal node moments.
    Any grouping that repeats within a path without being global would need
    integration over several shared log-gains and is rejected.

    Args:
        output_mean: Frozen node means, shape (batch, hrc.len).
        output_variance: Frozen node variances, shape (batch, hrc.len).
        hrc: Proper K-leaf tree structure.
        posterior: Gaussian belief over the group log-gains.
        order: Gauss--Hermite order for the log-gain integral.
        owen_order: Gauss--Legendre order for the conditional variance.
        chunk_size: Rows per block; ``None`` derives one from the tree width
            and :data:`DEFAULT_CHUNK_ELEMENTS`.

    Returns:
        Class moments in float64, with ``mean`` summing to one per row.
    """

    _require_proper_tree(hrc, "hsm_class_moments")
    _check_node_moments(output_mean, output_variance, hrc)
    rows = _resolve_chunk_size(hrc, order, chunk_size)
    groups = posterior.groups
    if not groups.path_repeats:
        node_moments = hsm_node_moments(
            output_mean,
            output_variance,
            hrc,
            posterior,
            order=order,
            owen_order=owen_order,
            chunk_size=rows,
        )
        return _class_moments_independent(node_moments, hrc)
    if groups.is_global:
        return _class_moments_global(
            output_mean, output_variance, hrc, posterior, order, owen_order, rows
        )
    raise ValueError(
        "a gain group that repeats within a class path without being global needs "
        "integration over several shared log-gains, which this module does not do"
    )


def hsm_class_probabilities(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
    posterior: LogGainPosterior,
    *,
    order: int = DEFAULT_GAIN_ORDER,
    owen_order: int = DEFAULT_OWEN_ORDER,
    chunk_size: int | None = None,
) -> Tensor:
    """Return ``E[Q_c]`` in the dtype of ``output_mean``, shape (batch, n_classes)."""

    moments = hsm_class_moments(
        output_mean,
        output_variance,
        hrc,
        posterior,
        order=order,
        owen_order=owen_order,
        chunk_size=chunk_size,
    )
    return moments.mean.to(output_mean.dtype)


# ──────────────────────────────────────────────────────────────────────────────
#  Cross-class covariance
# ──────────────────────────────────────────────────────────────────────────────


def _cross_class_masks(hrc: HierarchicalSoftmax, device: torch.device) -> tuple[Tensor, ...]:
    """Return the five ``(n_classes^2, len)`` case masks of a class pair.

    At a node shared by two paths the pair is either on the same branch or on
    opposite branches; a node on exactly one of the two paths contributes that
    path's own branch mean.
    """

    codes = branch_codes(hrc, device)
    on_left = (codes > 0.0).double()
    on_right = (codes < 0.0).double()
    left_c, left_d = on_left[:, None, :], on_left[None, :, :]
    right_c, right_d = on_right[:, None, :], on_right[None, :, :]
    on_c, on_d = left_c + right_c, left_d + right_d
    masks = (
        left_c * left_d,
        right_c * right_d,
        left_c * right_d + right_c * left_d,
        left_c * (1.0 - on_d) + left_d * (1.0 - on_c),
        right_c * (1.0 - on_d) + right_d * (1.0 - on_c),
    )
    return tuple(mask.reshape(-1, hrc.len) for mask in masks)


def _log_second_moment_terms(
    log_left: Tensor,
    log_right: Tensor,
    variance: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return the five per-node log terms a class pair can contribute."""

    _, _, log_second_left, log_second_right = _oriented_log_terms(log_left, log_right, variance)
    # E[U (1 - U)] is a difference by construction, so this one term cannot be
    # written without a subtraction; it is clamped rather than floored blindly.
    opposite = (log_left + log_right).exp() - variance.clamp_min(0.0)
    log_opposite = opposite.clamp_min(_PROBABILITY_FLOOR).log()
    return log_second_left, log_second_right, log_opposite, log_left, log_right


def hsm_cross_class_covariance(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
    posterior: LogGainPosterior,
    *,
    order: int = DEFAULT_GAIN_ORDER,
    owen_order: int = DEFAULT_OWEN_ORDER,
    chunk_size: int | None = None,
) -> Tensor:
    """Return ``Cov(Q_c, Q_d)``, shape (batch, n_classes, n_classes).

    Because the class probabilities sum to one pointwise, every row of the
    result sums to zero up to quadrature and roundoff, which is the sharpest
    available check on the implementation. The cost is quadratic in the class
    count, so this is a diagnostic for small trees rather than a prediction
    path.

    Args:
        output_mean: Frozen node means, shape (batch, hrc.len).
        output_variance: Frozen node variances, shape (batch, hrc.len).
        hrc: Proper K-leaf tree structure.
        posterior: Gaussian belief over the group log-gains. Per-node gains use
            the node-independent formula; one global gain conditions on the
            shared log-gain and integrates it. Per-level gains create
            dependence between distinct nodes of the same depth and are
            rejected.
        order: Gauss--Hermite order for the log-gain integral.
        owen_order: Gauss--Legendre order for the conditional variance.
        chunk_size: Rows per block; ``None`` derives one from the tree width
            and :data:`DEFAULT_CHUNK_ELEMENTS`.

    Returns:
        The covariance matrices in float64.
    """

    _require_proper_tree(hrc, "hsm_cross_class_covariance")
    _check_node_moments(output_mean, output_variance, hrc)
    groups = posterior.groups
    if groups.path_repeats and not groups.is_global:
        raise ValueError("cross-class covariance needs per-node or global gain sharing")
    if not groups.path_repeats and int(groups.node_group.unique().numel()) != hrc.len:
        raise ValueError(
            "cross-class covariance under shared gains needs per-node groups: distinct "
            "nodes of one group are dependent, which the product formula ignores"
        )

    device = output_mean.device
    n_classes = hrc.n_classes
    masks = torch.stack(_cross_class_masks(hrc, device))
    rows = _resolve_chunk_size(hrc, order, chunk_size)
    class_mean = hsm_class_moments(
        output_mean,
        output_variance,
        hrc,
        posterior,
        order=order,
        owen_order=owen_order,
        chunk_size=rows,
    ).mean

    offset = hrc.node_offset(device).double()
    nodes, weights = _placed_rule(_hermite_rule(order), output_mean)
    log_gain = posterior.mean[0].to(device) + posterior.variance[0].to(device).sqrt() * nodes
    blocks: list[Tensor] = []
    for start in range(0, output_mean.shape[0], rows):
        block_mean = output_mean[start : start + rows].double()
        block_variance = output_variance[start : start + rows].double().clamp_min(0.0)
        if groups.is_global:
            argument, latent_variance, _ = _conditional_node_terms(
                block_mean, block_variance, offset, log_gain, posterior.sigma_v
            )
            lower = (1.0 / (1.0 + 2.0 * latent_variance).sqrt()).atan()
            upper = torch.full_like(lower, _QUARTER_PI)
            conditional_variance = (
                _angular_integral(argument, lower, upper, owen_order) / math.pi
            ).clamp_min(0.0)
            terms = torch.stack(
                _log_second_moment_terms(
                    torch.special.log_ndtr(argument),
                    torch.special.log_ndtr(-argument),
                    conditional_variance,
                )
            )
            # (5, batch, len, order) -> (batch * order, len) per case.
            flat = terms.permute(0, 1, 3, 2).reshape(5, -1, hrc.len)
            joint = torch.einsum("cbn,cpn->bp", flat, masks)
            joint = joint.reshape(block_mean.shape[0], -1, n_classes, n_classes)
            second = (weights[:, None, None] * joint.exp()).sum(1)
        else:
            node_moments = hsm_node_moments(
                block_mean,
                block_variance,
                hrc,
                posterior,
                order=order,
                owen_order=owen_order,
                chunk_size=rows,
            )
            terms = torch.stack(
                _log_second_moment_terms(
                    node_moments.log_mean,
                    node_moments.log_complement,
                    node_moments.variance,
                )
            )
            joint = torch.einsum("cbn,cpn->bp", terms, masks)
            second = joint.reshape(-1, n_classes, n_classes).exp()
        blocks.append(second)

    second = torch.cat(blocks)
    return second - class_mean[:, :, None] * class_mean[:, None, :]


# ──────────────────────────────────────────────────────────────────────────────
#  Held-out batch calibration
# ──────────────────────────────────────────────────────────────────────────────


def _label_paths(labels: Tensor, hrc: HierarchicalSoftmax) -> tuple[Tensor, Tensor]:
    """Return per-sample branch signs and visit indicators, shape (batch, len)."""

    target = labels.reshape(-1).long()
    if bool(((target < 0) | (target >= hrc.n_classes)).any()):
        raise ValueError("labels contain an invalid class index")
    codes = branch_codes(hrc, target.device)[target]
    return codes, codes.abs()


def hsm_calibration_visits(
    labels: Tensor,
    hrc: HierarchicalSoftmax,
    groups: GainGroups,
) -> Tensor:
    """Return the number of calibration node-visits assigned to each gain group."""

    _, visited = _label_paths(labels, hrc)
    per_node = visited.sum(dim=0)
    counts = torch.zeros(groups.n_groups, dtype=torch.float64, device=per_node.device)
    counts.index_add_(0, groups.node_group.to(per_node.device), per_node)
    return counts.round().long()


def hsm_log_posterior_on_grid(
    output_mean: Tensor,
    output_variance: Tensor,
    labels: Tensor,
    hrc: HierarchicalSoftmax,
    groups: GainGroups,
    *,
    prior_mean: float = 0.0,
    prior_variance: float = 1.0,
    sigma_v: float = 1.0,
    grid: Tensor | None = None,
    grid_bounds: tuple[float, float] | None = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    chunk_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate the scalar log posterior of every gain group on a log-gain grid.

    For group ``r`` with calibration node-visits ``V_r`` and sign-oriented
    observations, the log posterior is

        ``-(l - lambda_0)^2 / (2 q_0) + sum_{(i, n) in V_r} log Phi(a_{i,n}(l; s))``

    up to a constant. Each visit enters through the exact node mean, so the sum
    is the true label log-likelihood in ``l``: for a global gain it is the
    categorical log-likelihood, and for per-node gains it is the factorization
    the independent gains imply.

    Args:
        output_mean: Frozen calibration node means, shape (batch, hrc.len).
        output_variance: Frozen calibration node variances, same shape.
        labels: Integer leaf labels for the calibration split, shape (batch,).
        hrc: Proper K-leaf tree structure.
        groups: Gain-sharing groups.
        prior_mean: Prior mean of every group log-gain.
        prior_variance: Prior variance of every group log-gain.
        sigma_v: Fixed noise of the branch channel the gain is measured
            against. The likelihood is evaluated at that scale, so a grid fitted
            at one ``sigma_v`` is not comparable with one fitted at another.
        grid: Explicit grid, shape (grid_size,) shared by every group or
            (n_groups, grid_size) with one window per group. ``None`` builds a
            uniform grid from ``grid_bounds``.
        grid_bounds: Inclusive log-gain interval for the built grid. ``None``
            covers the prior out to :data:`DEFAULT_GRID_HALF_WIDTH` standard
            deviations, and never less than that many nats.
        grid_size: Number of grid points, at least five.
        chunk_size: Rows per block; ``None`` derives one from the tree width
            and :data:`DEFAULT_CHUNK_ELEMENTS`.

    Returns:
        grid: The grid the values were evaluated on.
        log_posterior: Unnormalized values, shape (n_groups, grid_size).
    """

    _require_proper_tree(hrc, "hsm_log_posterior_on_grid")
    _check_node_moments(output_mean, output_variance, hrc)
    if prior_variance <= 0.0 or not math.isfinite(prior_variance):
        raise ValueError("prior_variance must be finite and positive")
    if not math.isfinite(sigma_v) or sigma_v <= 0.0:
        raise ValueError("sigma_v must be finite and positive")

    device = output_mean.device
    if grid is None:
        if grid_size < 5:
            raise ValueError("grid_size must be at least five")
        lower, upper = grid_bounds or _prior_grid_bounds(prior_mean, prior_variance)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("grid_bounds must be finite and increasing")
        grid = torch.linspace(lower, upper, grid_size, dtype=torch.float64, device=device)
    grid = grid.to(device=device, dtype=torch.float64)
    if grid.dim() not in (1, 2) or grid.shape[-1] < 5:
        raise ValueError("grid must be (grid_size,) or (n_groups, grid_size) with five points")
    if grid.dim() == 2 and grid.shape[0] != groups.n_groups:
        raise ValueError("a two-dimensional grid must have one row per gain group")
    grid_size = int(grid.shape[-1])
    node_grid = grid if grid.dim() == 1 else grid.index_select(0, groups.node_group.to(device))
    offset = hrc.node_offset(device).double()
    sign, visited = _label_paths(labels.to(device), hrc)
    node_group = groups.node_group.to(device)
    rows = _resolve_chunk_size(hrc, grid_size, chunk_size)

    per_node = torch.zeros(hrc.len, grid_size, dtype=torch.float64, device=device)
    for start in range(0, output_mean.shape[0], rows):
        stop = start + rows
        argument, _, _ = _conditional_node_terms(
            output_mean[start:stop].double(),
            output_variance[start:stop].double().clamp_min(0.0),
            offset,
            node_grid.expand(hrc.len, grid_size),
            sigma_v,
        )
        oriented = sign[start:stop, :, None] * argument
        contribution = visited[start:stop, :, None] * torch.special.log_ndtr(oriented)
        per_node += contribution.sum(dim=0)

    log_likelihood = torch.zeros(groups.n_groups, grid_size, dtype=torch.float64, device=device)
    log_likelihood.index_add_(0, node_group, per_node)
    penalty = -0.5 * (grid - prior_mean).square() / prior_variance
    return grid, log_likelihood + penalty


def _prior_grid_bounds(prior_mean: float, prior_variance: float) -> tuple[float, float]:
    """Return a coarse interval that covers both the prior and any sane gain."""

    reach = DEFAULT_GRID_HALF_WIDTH * math.sqrt(prior_variance)
    return (
        min(prior_mean - reach, -DEFAULT_GRID_HALF_WIDTH),
        max(prior_mean + reach, DEFAULT_GRID_HALF_WIDTH),
    )


def _grid_moments(grid: Tensor, log_posterior: Tensor) -> tuple[Tensor, Tensor]:
    """Return the normalized grid mean and variance of a scalar log posterior.

    A uniform grid makes this a trapezoid rule whose error on a near-Gaussian
    posterior falls off like ``exp(-2 pi^2 q / h^2)``, so it is exact to double
    precision once the spacing ``h`` is comfortably inside the posterior
    standard deviation. That is what the window refinement guarantees.
    """

    weights = torch.softmax(log_posterior, dim=-1)
    mean = (weights * grid).sum(-1)
    variance = (weights * (grid - mean[..., None]).square()).sum(-1)
    return mean, variance


def _laplace_moments(grid: Tensor, log_posterior: Tensor) -> tuple[Tensor, Tensor]:
    """Return the mode and inverse negative curvature of a gridded log posterior.

    The variance is infinite where the gridded posterior has no interior
    maximum, which tells the caller to fall back on the prior scale.
    """

    expanded = grid.expand_as(log_posterior)
    spacing = expanded[:, 1] - expanded[:, 0]
    mode_index = log_posterior[:, 1:-1].argmax(dim=-1) + 1
    rows = torch.arange(log_posterior.shape[0], device=log_posterior.device)
    left = log_posterior[rows, mode_index - 1]
    centre = log_posterior[rows, mode_index]
    right = log_posterior[rows, mode_index + 1]
    curvature = (right - 2.0 * centre + left) / spacing.square()
    concave = curvature < -_CURVATURE_FLOOR
    # A three-point parabola through the grid maximum refines the mode to well
    # inside one spacing without another likelihood evaluation.
    slope = 0.5 * (right - left) / spacing
    shift = torch.where(concave, (slope / curvature).clamp(-spacing, spacing), slope * 0.0)
    mean = expanded[rows, mode_index] - shift
    variance = torch.where(
        concave, -1.0 / curvature, torch.full_like(curvature, float("inf"))
    )
    return mean, variance


def _refined_grid(mean: Tensor, variance: Tensor, fallback: float, size: int) -> Tensor:
    """Return one window per group, centred on the mode and scaled to its width.

    The reach is generous because the cost is only in resolution, and a uniform
    grid has resolution to spare: at sixteen deviations either side of the mode
    the spacing is still a sixteenth of one deviation, while a monotone
    likelihood leaves a posterior skewed enough that a tighter window would
    truncate its heavy tail. Widening the reach from eight to sixteen improves
    the fitted deviation from four to thirteen digits on such a posterior.
    """

    default = torch.full_like(variance, fallback)
    finite = variance.clamp_min(0.0).sqrt()
    deviation = torch.where(torch.isfinite(variance), finite, default).clamp_min(1e-6)
    half = _WINDOW_REACH * deviation
    steps = torch.linspace(-1.0, 1.0, size, dtype=mean.dtype, device=mean.device)
    return mean[:, None] + half[:, None] * steps


def fit_hsm_log_gain(
    output_mean: Tensor,
    output_variance: Tensor,
    labels: Tensor,
    hrc: HierarchicalSoftmax,
    *,
    sharing: str | GainGroups = "global",
    prior_mean: float = 0.0,
    prior_variance: float = 1.0,
    sigma_v: float = 1.0,
    method: str = "grid",
    grid_bounds: tuple[float, float] | None = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    refinements: int = DEFAULT_GRID_REFINEMENTS,
    chunk_size: int | None = None,
) -> LogGainPosterior:
    """Fit the log-gain posterior of every group on a held-out calibration split.

    Every gain group is scalar, so the batch calculation is order-independent
    and is preferred over repeated assumed-density updates. The module
    conditions on the frozen TAGI forward summaries and treats distinct
    calibration visits as conditionally independent: this is a cut
    approximation, not a joint re-inference of the network posterior.

    Call this once with the network frozen. Refitting from the same stated
    prior after every epoch is fine; replaying the same labels while carrying
    the previous posterior precision forward would count the data twice.

    Args:
        output_mean: Frozen calibration node means, shape (batch, hrc.len).
        output_variance: Frozen calibration node variances, same shape.
        labels: Integer leaf labels for the calibration split.
        hrc: Proper K-leaf tree structure.
        sharing: A name from :data:`GAIN_SHARING` or an explicit grouping.
        prior_mean: Prior mean of every group log-gain. Zero is the head as
            trained, which is the natural centre once ``sigma_v`` is the
            channel noise the head actually used.
        prior_variance: Prior variance of every group log-gain.
        sigma_v: Fixed noise of the branch channel; stamped onto the returned
            belief, which is what keeps prediction on the same scale as the fit.
        method: ``grid`` for normalized grid moments, or ``laplace`` for the
            mode and its inverse negative curvature.
        grid_bounds: Inclusive log-gain interval for the coarse sweep. ``None``
            covers the prior, which keeps a data-free group at its stated
            variance instead of truncating it.
        grid_size: Number of grid points per sweep.
        refinements: Sweeps that re-centre and rescale the grid on the current
            mode. A sharp posterior needs them: a grid whose spacing exceeds
            the posterior standard deviation cannot resolve it, and that is the
            regime a large calibration split lands in.
        chunk_size: Rows per block.

    Returns:
        The fitted posterior, carrying the visit count of every group.
    """

    groups = (
        sharing
        if isinstance(sharing, GainGroups)
        else gain_groups(hrc, sharing, device=output_mean.device)
    )
    if method not in ("grid", "laplace"):
        raise ValueError("method must be 'grid' or 'laplace'")
    if refinements < 0:
        raise ValueError("refinements must be nonnegative")

    def sweep(grid: Tensor | None) -> tuple[Tensor, Tensor]:
        return hsm_log_posterior_on_grid(
            output_mean,
            output_variance,
            labels,
            hrc,
            groups,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            sigma_v=sigma_v,
            grid=grid,
            grid_bounds=grid_bounds,
            grid_size=grid_size,
            chunk_size=chunk_size,
        )

    grid, log_posterior = sweep(None)
    prior_deviation = math.sqrt(prior_variance)
    for _ in range(refinements):
        mode, curvature_variance = _laplace_moments(grid, log_posterior)
        window = _refined_grid(mode, curvature_variance, prior_deviation, grid_size)
        grid, log_posterior = sweep(window)
    if method == "grid":
        mean, variance = _grid_moments(grid, log_posterior)
    else:
        mean, variance = _laplace_moments(grid, log_posterior)
        variance = variance.clamp(max=prior_variance)
    return LogGainPosterior(
        mean=mean,
        variance=variance.clamp_min(0.0),
        groups=groups.to(output_mean.device),
        visits=hsm_calibration_visits(labels.to(output_mean.device), hrc, groups),
        sigma_v=float(sigma_v),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Sequential assumed-density calibration
# ──────────────────────────────────────────────────────────────────────────────


def hsm_adf_projection(
    mean: float,
    variance: float,
    branch_mean: float,
    cov_log_gain: float,
    observation: float = 1.0,
    *,
    variance_decrement_cap: float = 0.999,
) -> tuple[float, float]:
    """Project one Bernoulli branch observation onto the log-gain belief.

    Conditioning the moment-matched Gaussian on ``B = b`` gives

        ``lambda+ = lambda + Cov(L, P) (b - mu_P) / [mu_P (1 - mu_P)]``
        ``q+ = q - Cov(L, P)^2 / [mu_P (1 - mu_P)]``

    where ``mu_P (1 - mu_P)`` is the total variance of the observed indicator,
    the sum of ``v_P`` and the expected Bernoulli variance. The mean update is
    the exact one-step Bernoulli identity ``E[L | B = 1] = lambda + Cov(L, P) /
    mu_P``; the variance update is exact only after averaging over both labels,
    so repeating it is assumed-density filtering and its ordering sensitivity
    has to be checked against the batch fit.

    Args:
        mean: Current ``lambda``.
        variance: Current ``q``.
        branch_mean: ``mu_P`` of the observed branch.
        cov_log_gain: ``Cov(L, P)`` of the observed branch.
        observation: The branch indicator, one under sign orientation.
        variance_decrement_cap: Largest fraction of ``q`` one observation may
            remove. Capping keeps the belief from freezing, which flooring a
            negative result at exactly zero would do.

    Returns:
        The updated ``lambda`` and ``q``.
    """

    if not 0.0 < variance_decrement_cap < 1.0:
        raise ValueError("variance_decrement_cap must lie in (0, 1)")
    if not math.isfinite(branch_mean) or not 0.0 < branch_mean < 1.0:
        return mean, variance
    total = branch_mean * (1.0 - branch_mean)
    updated_mean = mean + cov_log_gain * (observation - branch_mean) / total
    decrement = min(cov_log_gain * cov_log_gain / total, variance_decrement_cap * variance)
    return updated_mean, variance - decrement


def _scalar_branch_moments(
    node_mean: float,
    node_variance: float,
    offset: float,
    sign: float,
    log_gain_mean: float,
    log_gain_variance: float,
    rule: tuple[tuple[float, ...], tuple[float, ...]],
    sigma_v: float,
) -> tuple[float, float]:
    """Return ``mu_P`` and ``Cov(L, P)`` of one oriented node visit."""

    nodes, weights = rule
    deviation = math.sqrt(max(log_gain_variance, 0.0))
    branch_mean = 0.0
    covariance = 0.0
    for node, weight in zip(nodes, weights):
        shift = deviation * node
        gain = math.exp(log_gain_mean + shift) / sigma_v
        scale = math.sqrt(1.0 + gain * gain * node_variance)
        argument = sign * (gain * node_mean + offset) / scale
        probability = 0.5 * math.erfc(-argument * _INV_SQRT_TWO)
        branch_mean += weight * probability
        covariance += weight * shift * probability
    return branch_mean, covariance


def calibrate_hsm_log_gain_adf(
    output_mean: Tensor,
    output_variance: Tensor,
    labels: Tensor,
    hrc: HierarchicalSoftmax,
    *,
    sharing: str | GainGroups = "global",
    prior_mean: float = 0.0,
    prior_variance: float = 1.0,
    sigma_v: float = 1.0,
    order: int = DEFAULT_GAIN_ORDER,
    process_variance: float = 0.0,
    variance_decrement_cap: float = 0.999,
    visit_order: Tensor | None = None,
) -> LogGainPosterior:
    """Calibrate the log-gain by sequential assumed-density filtering.

    Provided for a genuine stream in which every observation is used once, and
    as the ordering-sensitivity check the assumed-density approximation
    requires. A batch fit with :func:`fit_hsm_log_gain` is order-independent
    and is the recommended default whenever the whole calibration split is
    available. Do not replay a calibration set through this while treating its
    outcomes as new evidence.

    Args:
        output_mean: Frozen calibration node means, shape (batch, hrc.len).
        output_variance: Frozen calibration node variances, same shape.
        labels: Integer leaf labels for the calibration split.
        hrc: Proper K-leaf tree structure.
        sharing: A name from :data:`GAIN_SHARING` or an explicit grouping.
        prior_mean: Prior mean of every group log-gain.
        prior_variance: Prior variance of every group log-gain.
        sigma_v: Fixed noise of the branch channel; stamped onto the returned
            belief exactly as in :func:`fit_hsm_log_gain`.
        order: Gauss--Hermite order for the per-visit node moments.
        process_variance: Added to ``q`` before every update, for a state model
            in which the gain drifts.
        variance_decrement_cap: Largest fraction of ``q`` one visit may remove.
        visit_order: Optional permutation of the sample index, so the same data
            can be replayed in a different order to measure the sensitivity.

    Returns:
        The filtered posterior, carrying the visit count of every group.
    """

    _require_proper_tree(hrc, "calibrate_hsm_log_gain_adf")
    _check_node_moments(output_mean, output_variance, hrc)
    if process_variance < 0.0 or not math.isfinite(process_variance):
        raise ValueError("process_variance must be finite and nonnegative")
    if not math.isfinite(sigma_v) or sigma_v <= 0.0:
        raise ValueError("sigma_v must be finite and positive")
    groups = (
        sharing
        if isinstance(sharing, GainGroups)
        else gain_groups(hrc, sharing, device=output_mean.device)
    )

    node_index, sign_table, mask_table = _path_arrays(hrc, output_mean.device)
    target = labels.reshape(-1).long().to(output_mean.device)
    if bool(((target < 0) | (target >= hrc.n_classes)).any()):
        raise ValueError("labels contain an invalid class index")
    sequence = (
        torch.arange(target.shape[0], device=target.device)
        if visit_order is None
        else visit_order.reshape(-1).long().to(target.device)
    )
    if sequence.shape[0] != target.shape[0]:
        raise ValueError("visit_order must be a permutation of the calibration index")

    nodes, weights = _hermite_rule(order)
    rule = (tuple(nodes.tolist()), tuple(weights.tolist()))
    offsets = hrc.node_offset(output_mean.device).double().tolist()
    group_of = groups.node_group.tolist()
    means = [float(prior_mean)] * groups.n_groups
    variances = [float(prior_variance)] * groups.n_groups
    visits = [0] * groups.n_groups

    node_columns = node_index.tolist()
    sign_columns = sign_table.tolist()
    mask_columns = mask_table.tolist()
    all_mean = output_mean.double().tolist()
    all_variance = output_variance.double().clamp_min(0.0).tolist()

    for row in sequence.tolist():
        label = int(target[row])
        for position, on_path in enumerate(mask_columns[label]):
            if on_path <= 0.0:
                continue
            node = int(node_columns[label][position])
            group = group_of[node]
            variances[group] += process_variance
            branch_mean, covariance = _scalar_branch_moments(
                all_mean[row][node],
                all_variance[row][node],
                offsets[node],
                sign_columns[label][position],
                means[group],
                variances[group],
                rule,
                sigma_v,
            )
            means[group], variances[group] = hsm_adf_projection(
                means[group],
                variances[group],
                branch_mean,
                covariance,
                1.0,
                variance_decrement_cap=variance_decrement_cap,
            )
            visits[group] += 1

    device = output_mean.device
    return LogGainPosterior(
        mean=torch.tensor(means, dtype=torch.float64, device=device),
        variance=torch.tensor(variances, dtype=torch.float64, device=device),
        groups=groups.to(device),
        visits=torch.tensor(visits, dtype=torch.long, device=device),
        sigma_v=float(sigma_v),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Scoring
# ──────────────────────────────────────────────────────────────────────────────


def hsm_negative_log_likelihood(
    output_mean: Tensor,
    output_variance: Tensor,
    labels: Tensor,
    hrc: HierarchicalSoftmax,
    posterior: LogGainPosterior,
    *,
    order: int = DEFAULT_GAIN_ORDER,
    owen_order: int = DEFAULT_OWEN_ORDER,
    chunk_size: int | None = None,
) -> Tensor:
    """Return the mean categorical cross-entropy in nats of ``E[Q_c]``.

    The predictive class probabilities already sum to one on a proper tree, so
    this is directly comparable with a softmax classifier.
    """

    target = labels.reshape(-1).long().to(output_mean.device)
    if target.shape[0] != output_mean.shape[0]:
        raise ValueError("labels must have shape (batch,) matching the predictions")
    probabilities = hsm_class_moments(
        output_mean,
        output_variance,
        hrc,
        posterior,
        order=order,
        owen_order=owen_order,
        chunk_size=chunk_size,
    ).mean
    picked = probabilities.gather(1, target[:, None]).squeeze(1)
    return -picked.clamp_min(_PROBABILITY_FLOOR).log().mean()


def hsm_partition_deviation(
    output_mean: Tensor,
    output_variance: Tensor,
    hrc: HierarchicalSoftmax,
    posterior: LogGainPosterior,
    *,
    order: int = DEFAULT_GAIN_ORDER,
    owen_order: int = DEFAULT_OWEN_ORDER,
    chunk_size: int | None = None,
) -> float:
    """Return ``max_i |sum_c E[Q_c] - 1|``, which a proper tree drives to roundoff."""

    probabilities = hsm_class_moments(
        output_mean,
        output_variance,
        hrc,
        posterior,
        order=order,
        owen_order=owen_order,
        chunk_size=chunk_size,
    ).mean
    return float((probabilities.sum(dim=-1) - 1.0).abs().max())
