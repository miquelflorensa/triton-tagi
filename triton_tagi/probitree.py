"""Exact-K hierarchical probit tree with direct branch observations.

``ProbiTree`` is a balanced binary classification tree whose ``K - 1``
internal nodes are Gaussian TAGI outputs.  Training observes only the signed
branches on a label's path and projects each probit posterior back to a
Gaussian by exact moment matching.  Prediction analytically marginalizes the
latent Gaussian score at every gate.

This module deliberately has no encoded ``+/-1`` Gaussian-regression update,
softmax normalizer, learned noise head, or deterministic gate offset.  The
fixed positive ``r`` arguments are observation-noise *variances*.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import torch
from torch import Tensor

SQRT2 = math.sqrt(2.0)
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_TAIL_CUTOFF = -35.0
_KAPPA_TOLERANCE = 1e-9


def probit_stats(a: float) -> tuple[float, float, float]:
    """Return ``log Phi(a)``, the inverse Mills ratio, and probit curvature.

    The left-tail expansion keeps both ``lambda`` and
    ``kappa = lambda * (lambda + a)`` finite when the CDF underflows.  Values
    around and below a standardized margin of ``-1000`` are supported in
    float64.
    """

    if not math.isfinite(a):
        raise ValueError("the standardized margin must be finite")
    if a < _TAIL_CUTOFF:
        t = -a
        inv = 1.0 / t
        u = inv * inv
        lam = t + inv * (1.0 + u * (-2.0 + u * (10.0 + u * (-74.0 + 706.0 * u))))
        kappa = 1.0 + u * (-1.0 + u * (6.0 + u * (-50.0 + u * (518.0 - 6354.0 * u))))
        squared = a * a
        if not math.isfinite(squared):
            raise ValueError("the standardized margin is outside the float64 working range")
        logp = -0.5 * squared - LOG_SQRT_2PI - math.log(lam)
    else:
        if a >= 0.0:
            logp = math.log1p(-0.5 * math.erfc(a / SQRT2))
        else:
            logp = math.log(0.5 * math.erfc(-a / SQRT2))
        lam = math.exp(-0.5 * a * a - LOG_SQRT_2PI - logp)
        kappa = lam * (lam + a)
    if not (-_KAPPA_TOLERANCE <= kappa <= 1.0 + _KAPPA_TOLERANCE):
        raise ArithmeticError("invalid probit curvature")
    return logp, lam, min(1.0, max(0.0, kappa))


def gate_update(m: float, v: float, b: int, r: float = 1.0) -> dict[str, float]:
    """Return exact local posterior moments and TAGI messages for one gate."""

    if b not in (-1, 1):
        raise ValueError("b must be -1 or +1")
    if not all(math.isfinite(value) for value in (m, v, r)) or v < 0.0 or r <= 0.0:
        raise ValueError("require finite m, v >= 0, and r > 0")
    s2 = r + v
    if not math.isfinite(s2):
        raise ValueError("r + v must be representable")
    s = math.sqrt(s2)
    logp, lam, kappa = probit_stats(b * m / s)
    g = b * lam / s
    hess = -kappa / s2
    dm = v * g
    # This form avoids cancellation for a highly surprising branch.
    v_post = v * (r / s2 + (v / s2) * (1.0 - kappa))
    dv = v_post - v
    return {
        "m_post": m + dm,
        "v_post": v_post,
        "dm": dm,
        "dv": dv,
        "g": g,
        "hess": hess,
        "logp": logp,
    }


@dataclass(frozen=True, init=False)
class ProbiTree:
    """Balanced binary probit tree with exactly ``K`` class leaves.

    Class IDs are ``0, ..., K - 1``.  Internal gates are numbered in preorder
    from zero.  Nonnegative child IDs denote gates and ``-(class_id + 1)``
    denotes a leaf.  The complete mapping is serializable with :meth:`to_dict`.
    """

    num_classes: int
    left: tuple[int, ...]
    right: tuple[int, ...]
    paths: tuple[tuple[tuple[int, int], ...], ...]
    left_fraction: tuple[float, ...]
    # Derived from the topology, so it takes no part in equality or hashing.
    _path_tensor_cache: dict[torch.device, tuple[Tensor, Tensor, Tensor]] = field(
        compare=False, repr=False
    )

    def __init__(self, num_classes: int) -> None:
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("require an integer K >= 2")
        left: list[int | None] = []
        right: list[int | None] = []
        fractions: list[float] = []
        paths: list[tuple[tuple[int, int], ...] | None] = [None] * num_classes

        def build(classes: list[int], path: tuple[tuple[int, int], ...]) -> int:
            if len(classes) == 1:
                class_id = classes[0]
                paths[class_id] = path
                return -(class_id + 1)
            node = len(left)
            left.append(None)
            right.append(None)
            cut = len(classes) // 2
            fractions.append(cut / len(classes))
            left[node] = build(classes[:cut], path + ((node, 1),))
            right[node] = build(classes[cut:], path + ((node, -1),))
            return node

        if build(list(range(num_classes)), ()) != 0:
            raise AssertionError("the root gate must have index zero")
        if len(left) != num_classes - 1 or any(path is None for path in paths):
            raise AssertionError("invalid full binary tree construction")
        object.__setattr__(self, "num_classes", num_classes)
        object.__setattr__(self, "left", tuple(int(value) for value in left))
        object.__setattr__(self, "right", tuple(int(value) for value in right))
        object.__setattr__(self, "paths", tuple(path for path in paths if path is not None))
        object.__setattr__(self, "left_fraction", tuple(fractions))
        object.__setattr__(self, "_path_tensor_cache", {})

    @property
    def num_gates(self) -> int:
        return len(self.left)

    @property
    def max_depth(self) -> int:
        return max(len(path) for path in self.paths)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete checkpoint-safe topology and class mapping."""

        return {
            "num_classes": self.num_classes,
            "left": list(self.left),
            "right": list(self.right),
            "paths": [[list(step) for step in path] for path in self.paths],
            "left_fraction": list(self.left_fraction),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> ProbiTree:
        """Restore and validate a serialized canonical ``ProbiTree`` mapping."""

        if not isinstance(state, dict) or "num_classes" not in state:
            raise ValueError("invalid ProbiTree state")
        tree = cls(state["num_classes"])
        if tree.to_dict() != state:
            raise ValueError("checkpoint ProbiTree mapping is not the canonical saved topology")
        return tree

    def path_tensors(self, device: torch.device | str) -> tuple[Tensor, Tensor, Tensor]:
        """Return padded zero-indexed nodes, signs, and path mask tensors.

        The topology is frozen, so the result is cached per device. Filling the
        tensors element by element on an accelerator would issue one host copy
        per path step on every call; they are built on the host and moved once.
        """

        key = torch.device(device)
        cached = self._path_tensor_cache.get(key)
        if cached is not None:
            return cached
        nodes = torch.zeros((self.num_classes, self.max_depth), dtype=torch.long)
        signs = torch.ones((self.num_classes, self.max_depth), dtype=torch.float64)
        mask = torch.zeros((self.num_classes, self.max_depth), dtype=torch.bool)
        for class_id, path in enumerate(self.paths):
            for depth, (node, sign) in enumerate(path):
                nodes[class_id, depth] = node
                signs[class_id, depth] = sign
                mask[class_id, depth] = True
        moved = (nodes.to(key), signs.to(key), mask.to(key))
        self._path_tensor_cache[key] = moved
        return moved


def build_tree(num_classes: int) -> ProbiTree:
    """Build the canonical exact-K balanced tree."""

    return ProbiTree(num_classes)


def _check_scalar_heads(
    tree: ProbiTree,
    means: Sequence[float],
    variances: Sequence[float],
    r: float,
) -> None:
    if len(means) != tree.num_gates or len(variances) != tree.num_gates:
        raise ValueError("expected K - 1 gate means and variances")
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError("require a finite positive fixed noise variance")
    if any(not math.isfinite(m) for m in means):
        raise ValueError("means must be finite")
    if any(not math.isfinite(v) or v < 0.0 for v in variances):
        raise ValueError("variances must be finite and nonnegative")
    if any(not math.isfinite(r + v) for v in variances):
        raise ValueError("r + variance must be representable")


def predict_log_probs(
    tree: ProbiTree,
    means: Sequence[float],
    variances: Sequence[float],
    r: float = 1.0,
) -> list[float]:
    """Return all scalar-reference class log probabilities in ``O(K)``."""

    _check_scalar_heads(tree, means, variances, r)
    out: list[float | None] = [None] * tree.num_classes
    stack = [(0, 0.0)]
    while stack:
        node, log_mass = stack.pop()
        if node < 0:
            out[-node - 1] = log_mass
            continue
        a = means[node] / math.sqrt(r + variances[node])
        log_left = probit_stats(a)[0]
        log_right = probit_stats(-a)[0]
        stack.append((tree.left[node], log_mass + log_left))
        stack.append((tree.right[node], log_mass + log_right))
    if any(value is None for value in out):
        raise AssertionError("tree traversal did not reach every class")
    return [float(value) for value in out if value is not None]


def label_updates(
    tree: ProbiTree,
    means: Sequence[float],
    variances: Sequence[float],
    label: int,
    r: float = 1.0,
) -> tuple[dict[int, dict[str, float]], float]:
    """Return active local updates, all computed from one forward prior."""

    _check_scalar_heads(tree, means, variances, r)
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < tree.num_classes:
        raise ValueError("invalid class ID")
    result: dict[int, dict[str, float]] = {}
    log_evidence = 0.0
    for node, sign in tree.paths[label]:
        update = gate_update(means[node], variances[node], sign, r)
        result[node] = update
        log_evidence += update["logp"]
    return result, log_evidence


def uniform_reference_means(
    tree: ProbiTree,
    reference_variances: Sequence[float],
    r: float = 1.0,
) -> list[float]:
    """Return gate means giving uniform leaves at the supplied variances."""

    _check_scalar_heads(tree, [0.0] * tree.num_gates, reference_variances, r)
    normal = NormalDist()
    return [
        math.sqrt(r + variance) * normal.inv_cdf(fraction)
        for fraction, variance in zip(tree.left_fraction, reference_variances, strict=True)
    ]


def _check_tensor_heads(tree: ProbiTree, means: Tensor, variances: Tensor, r: float) -> None:
    if means.shape != variances.shape or means.dim() < 1 or means.shape[-1] != tree.num_gates:
        raise ValueError("gate moments must have matching shapes ending in K - 1")
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError("require a finite positive fixed noise variance")
    if not bool(torch.isfinite(means).all()):
        raise ValueError("gate means must be finite")
    if not bool(torch.isfinite(variances).all()) or bool((variances < 0.0).any()):
        raise ValueError("gate variances must be finite and nonnegative")
    total_variance = variances.double() + r
    if not bool(torch.isfinite(total_variance).all()):
        raise ValueError("r + gate variance must be representable")


def _tensor_probit_stats(a: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Float64 tensor counterpart of :func:`probit_stats`."""

    if not bool(torch.isfinite(a).all()):
        raise ValueError("standardized margins must be finite")
    work = a.double()
    squared = work.square()
    if not bool(torch.isfinite(squared).all()):
        raise ValueError("standardized margins are outside the float64 working range")

    logp_regular = torch.special.log_ndtr(work)
    lam_regular = torch.exp(-0.5 * squared - LOG_SQRT_2PI - logp_regular)
    kappa_regular = lam_regular * (lam_regular + work)

    t = -work
    inv = torch.where(t != 0.0, t.reciprocal(), torch.zeros_like(t))
    u = inv.square()
    lam_tail = t + inv * (1.0 + u * (-2.0 + u * (10.0 + u * (-74.0 + 706.0 * u))))
    kappa_tail = 1.0 + u * (-1.0 + u * (6.0 + u * (-50.0 + u * (518.0 - 6354.0 * u))))
    # lam_tail is positive on the selected left-tail branch.
    logp_tail = -0.5 * squared - LOG_SQRT_2PI - torch.log(lam_tail.clamp_min(1e-300))
    tail = work < _TAIL_CUTOFF
    logp = torch.where(tail, logp_tail, logp_regular)
    lam = torch.where(tail, lam_tail, lam_regular)
    kappa = torch.where(tail, kappa_tail, kappa_regular)
    if not bool(torch.isfinite(logp).all()) or not bool(torch.isfinite(lam).all()):
        raise ArithmeticError("non-finite probit statistics")
    if bool(((kappa < -_KAPPA_TOLERANCE) | (kappa > 1.0 + _KAPPA_TOLERANCE)).any()):
        raise ArithmeticError("invalid probit curvature")
    return logp, lam, kappa.clamp(0.0, 1.0)


def probitree_log_probs(
    tree: ProbiTree,
    means: Tensor,
    variances: Tensor,
    r: float = 1.0,
) -> Tensor:
    """Return tensor class log probabilities via one ``O(K)`` traversal.

    Inputs may have arbitrary common leading dimensions.  The result is
    float64 and has the same leading dimensions followed by ``K`` classes.
    """

    _check_tensor_heads(tree, means, variances, r)
    work_mean = means.double()
    work_variance = variances.double()
    output = work_mean.new_empty((*means.shape[:-1], tree.num_classes))
    zero_mass = work_mean.new_zeros(means.shape[:-1])
    stack: list[tuple[int, Tensor]] = [(0, zero_mass)]
    while stack:
        node, log_mass = stack.pop()
        if node < 0:
            output[..., -node - 1] = log_mass
            continue
        margin = work_mean[..., node] / torch.sqrt(r + work_variance[..., node])
        log_left = _tensor_probit_stats(margin)[0]
        log_right = _tensor_probit_stats(-margin)[0]
        stack.append((tree.left[node], log_mass + log_left))
        stack.append((tree.right[node], log_mass + log_right))
    return output


@dataclass(frozen=True)
class ProbiTreeUpdate:
    """One assembled direct-probit output update from a shared forward prior."""

    posterior_mean: Tensor
    posterior_variance: Tensor
    delta_mean: Tensor
    delta_variance: Tensor
    g: Tensor
    h: Tensor
    log_evidence: Tensor


def probitree_label_update(
    tree: ProbiTree,
    means: Tensor,
    variances: Tensor,
    labels: Tensor,
    r: float = 1.0,
) -> ProbiTreeUpdate:
    """Assemble sparse path moments/messages for a batch of forward priors.

    This function only assembles local output changes; it does not define a
    TAGI batch approximation.  :meth:`triton_tagi.network.Sequential.step_probitree`
    therefore restricts the first backend integration to one labeled sample.
    """

    _check_tensor_heads(tree, means, variances, r)
    if means.dim() != 2:
        raise ValueError("ProbiTree label updates require moments shaped (batch, K - 1)")
    if labels.dim() != 1 or labels.shape[0] != means.shape[0]:
        raise ValueError("labels must have shape (batch,) matching the moments")
    targets = labels.to(device=means.device, dtype=torch.long)
    if bool(((targets < 0) | (targets >= tree.num_classes)).any()):
        raise ValueError("labels contain an invalid class ID")

    nodes_all, signs_all, mask_all = tree.path_tensors(means.device)
    nodes = nodes_all[targets]
    signs = signs_all[targets]
    mask = mask_all[targets]
    selected_mean = torch.gather(means.double(), 1, nodes)
    selected_variance = torch.gather(variances.double(), 1, nodes)
    s2 = selected_variance + r
    margins = signs * selected_mean / torch.sqrt(s2)
    logp, lam, kappa = _tensor_probit_stats(margins)
    keep = mask.double()

    selected_g = keep * signs * lam / torch.sqrt(s2)
    selected_h = keep * -kappa / s2
    selected_dm = selected_variance * selected_g
    selected_v_post = selected_variance * (r / s2 + (selected_variance / s2) * (1.0 - kappa))
    selected_dv = keep * (selected_v_post - selected_variance)

    g = torch.zeros_like(means, dtype=torch.float64)
    h = torch.zeros_like(variances, dtype=torch.float64)
    dm = torch.zeros_like(means, dtype=torch.float64)
    dv = torch.zeros_like(variances, dtype=torch.float64)
    g.scatter_add_(1, nodes, selected_g)
    h.scatter_add_(1, nodes, selected_h)
    dm.scatter_add_(1, nodes, selected_dm)
    dv.scatter_add_(1, nodes, selected_dv)
    posterior_mean = means.double() + dm
    posterior_variance = variances.double() + dv
    log_evidence = (keep * logp).sum(dim=1)
    return ProbiTreeUpdate(
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
        delta_mean=dm,
        delta_variance=dv,
        g=g,
        h=h,
        log_evidence=log_evidence,
    )


def probitree_uniform_reference_means(
    tree: ProbiTree,
    reference_variances: Tensor,
    r: float = 1.0,
) -> Tensor:
    """Tensor gate means giving uniform leaves at reference variances."""

    if reference_variances.dim() != 1 or reference_variances.numel() != tree.num_gates:
        raise ValueError("reference_variances must have shape (K - 1,)")
    _check_tensor_heads(tree, torch.zeros_like(reference_variances), reference_variances, r)
    fractions = torch.tensor(
        tree.left_fraction, dtype=torch.float64, device=reference_variances.device
    )
    return torch.sqrt(reference_variances.double() + r) * torch.special.ndtri(fractions)
