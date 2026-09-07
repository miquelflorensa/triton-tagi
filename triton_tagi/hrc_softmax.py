"""
Hierarchical Softmax (HRCSoftmax) for TAGI classification.

Each class is encoded as a binary codeword of length L = ceil(log2(n_classes)).
The output layer has ``len`` neurons (binary-tree nodes), fewer than n_classes.

During **training**, only L nodes on each class's binary path receive an update
(sparse innovation) instead of all n_classes outputs.

During **inference**, class probabilities are products of Gaussian CDF values
along each class's path through the tree.

This replicates cuTAGI's ``class_to_obs`` / ``obs_to_class`` from
``src/cost.cpp`` and the ``compute_selected_delta_z_output`` logic from
``src/base_output_updater.cpp``.

Reference: https://building-babylon.net/2017/08/01/hierarchical-softmax
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
#  Data structure
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class HierarchicalSoftmax:
    """Binary-tree encoding for TAGI classification.

    Attributes:
        obs:   Float tensor (n_classes, n_obs) of ±1 encoded observations.
               +1 means bit = 0 (left branch); −1 means bit = 1 (right branch).
        idx:   Int tensor (n_classes, n_obs) of 1-indexed output node positions.
        n_obs: Number of bits per class = ceil(log2(n_classes)) for the padded
               fixed-depth tree, or the deepest path length for a full tree.
        len:   Total number of unique nodes in the tree (= output layer width).
        mask:  Optional float tensor (n_classes, n_obs) that is one on real path
               factors and zero on padding. ``None`` means every factor is real,
               which is the case for a fixed-depth tree.
        offset: Optional float tensor (len,) of left-branch prior probits. Node
               ``j`` shifts its latent variable by ``tau * offset[j]``, so a
               zero network output reproduces the branch prior ``Phi(offset[j])``.
               ``None`` means every offset is zero.
    """

    obs: Tensor  # (n_classes, n_obs)  float32, values ∈ {+1, −1}
    idx: Tensor  # (n_classes, n_obs)  int32,   1-indexed
    n_obs: int
    len: int
    mask: Tensor | None = None  # (n_classes, n_obs)  float32, values ∈ {0, 1}
    offset: Tensor | None = None  # (len,)               float32

    @property
    def n_classes(self) -> int:
        """Number of class leaves."""

        return int(self.obs.shape[0])

    @property
    def is_full(self) -> bool:
        """True when the tree has exactly ``n_classes - 1`` decision nodes.

        A full tree needs no categorical normalizer: its leaf probabilities
        already sum to one.
        """

        return self.len == self.n_classes - 1

    def path_mask(self, device: torch.device | str | None = None) -> Tensor:
        """Return the (n_classes, n_obs) padding mask, materializing ones."""

        if self.mask is None:
            return torch.ones_like(self.obs, device=device)
        return self.mask.to(device) if device is not None else self.mask

    def node_offset(self, device: torch.device | str | None = None) -> Tensor:
        """Return the (len,) left-branch prior probits, materializing zeros."""

        if self.offset is None:
            return torch.zeros(self.len, dtype=self.obs.dtype, device=device or self.obs.device)
        return self.offset.to(device) if device is not None else self.offset


# ──────────────────────────────────────────────────────────────────────────────
#  Tree construction
# ──────────────────────────────────────────────────────────────────────────────


def _dec_to_bi(num: int, n_bits: int) -> list[int]:
    """Integer → MSB-first binary list of length n_bits."""
    bits = []
    for _ in range(n_bits):
        bits.append(num % 2)
        num //= 2
    bits.reverse()
    return bits


def _bi_to_dec(bits: list[int]) -> int:
    """MSB-first binary list → integer."""
    result = 0
    for b in bits:
        result = result * 2 + b
    return result


def class_to_obs(n_classes: int) -> HierarchicalSoftmax:
    """Build the binary-tree hierarchical softmax structure.

    Replicates cuTAGI's ``class_to_obs()`` from ``src/cost.cpp`` exactly,
    including the 1-indexed node numbering convention.

    For 10 classes: n_obs = 4, len = 11  (matches cuTAGI's FNN example
    which uses ``Linear(hidden, 11)`` as the output layer).

    Args:
        n_classes: Number of output classes.

    Returns:
        HierarchicalSoftmax with obs (n_classes, L), idx (n_classes, L),
        n_obs = L, len = total tree nodes.
    """
    L = math.ceil(math.log2(n_classes))

    # Binary codes and ±1 observations for each class
    C: list[list[int]] = []
    obs: list[list[float]] = []
    for r in range(n_classes):
        bits = _dec_to_bi(r, L)
        C.append(bits)
        obs.append([(-1.0) ** b for b in bits])  # 0 → +1, 1 → −1

    # C_sum: number of nodes at each depth (computed from leaves to root)
    C_sum = [0] * (L + 1)
    C_sum[L] = n_classes
    for l in range(L - 1, -1, -1):
        C_sum[l] = math.ceil(C_sum[l + 1] / 2.0)

    # Convert to cumulative sum (as in cuTAGI) and add 1-offset
    for l in range(1, L + 1):
        C_sum[l] = C_sum[l - 1] + C_sum[l]
    for l in range(L + 1):
        C_sum[l] += 1

    # 1-indexed node positions: idx[r][c] is the tree node for bit c of class r
    idx: list[list[int]] = [[1] * L for _ in range(n_classes)]
    for r in range(n_classes):
        for c in range(L - 1):
            idx[r][c + 1] = _bi_to_dec(C[r][: c + 1]) + C_sum[c]

    tree_len = max(idx[r][c] for r in range(n_classes) for c in range(L))

    return HierarchicalSoftmax(
        obs=torch.tensor(obs, dtype=torch.float32),  # (n_classes, L)
        idx=torch.tensor(idx, dtype=torch.int32),  # (n_classes, L)
        n_obs=L,
        len=tree_len,
    )


def class_to_obs_full(
    n_classes: int,
    *,
    class_priors: Tensor | list[float] | None = None,
    use_prior_offsets: bool = True,
) -> HierarchicalSoftmax:
    """Build a full binary tree with exactly ``n_classes`` leaves.

    A full binary tree over K classes has exactly K - 1 decision nodes, and its
    leaf probabilities sum to one without a categorical normalizer::

        sum_c prod_{j in path(c)} p(s_cj | x) = 1

    The classes are split recursively into subsets of sizes ``floor(K/2)`` and
    ``ceil(K/2)``, so path lengths differ by at most one. For K = 10 this gives
    six leaves at depth three, four at depth four, and nine decision nodes,
    against eleven nodes and six discarded leaves for :func:`class_to_obs`.

    Unequal subtree sizes are absorbed by a deterministic branch offset. Node
    ``j`` with left-subtree prior mass ``pi_j`` stores ``offset[j] =
    Phi^-1(pi_j)`` and shifts its latent variable by ``tau * offset[j]``, so a
    zero network output telescopes to the class prior rather than to a
    depth-dependent power of one half.

    Args:
        n_classes: Number of output classes, at least two.
        class_priors: Optional nonnegative class weights, shape (n_classes,).
            Defaults to a uniform prior.
        use_prior_offsets: When False, all branch offsets are zero. Use it to
            ablate the prior correction while keeping the full-tree topology.

    Returns:
        HierarchicalSoftmax with ``len = n_classes - 1``, a padding ``mask``,
        and node ``offset`` values.
    """

    if n_classes < 2:
        raise ValueError("n_classes must be at least two")
    if class_priors is None:
        priors = torch.full((n_classes,), 1.0 / n_classes, dtype=torch.float64)
    else:
        priors = torch.as_tensor(class_priors, dtype=torch.float64).reshape(-1)
        if priors.shape[0] != n_classes:
            raise ValueError("class_priors must have one weight per class")
        if not bool(torch.isfinite(priors).all()) or bool((priors < 0).any()):
            raise ValueError("class_priors must be finite and nonnegative")
        total = float(priors.sum())
        if total <= 0.0:
            raise ValueError("class_priors must have positive total mass")
        priors = priors / total

    paths: list[list[tuple[int, float]]] = [[] for _ in range(n_classes)]
    offsets: list[float] = []

    def split(members: list[int]) -> None:
        if len(members) == 1:
            return
        node = len(offsets)  # 0-indexed; the 1-indexed position is node + 1
        offsets.append(0.0)
        half = len(members) // 2
        left, right = members[:half], members[half:]
        left_mass = float(priors[left].sum())
        total_mass = left_mass + float(priors[right].sum())
        fraction = left_mass / total_mass if total_mass > 0.0 else 0.5
        if use_prior_offsets:
            clamped = min(max(fraction, 1e-6), 1.0 - 1e-6)
            offsets[node] = float(torch.special.ndtri(torch.tensor(clamped, dtype=torch.float64)))
        for member in left:
            paths[member].append((node + 1, 1.0))
        for member in right:
            paths[member].append((node + 1, -1.0))
        split(left)
        split(right)

    split(list(range(n_classes)))
    if len(offsets) != n_classes - 1:
        raise AssertionError("a full binary tree must have n_classes - 1 decision nodes")

    depth = max(len(path) for path in paths)
    obs = torch.ones(n_classes, depth, dtype=torch.float32)
    idx = torch.ones(n_classes, depth, dtype=torch.int32)
    mask = torch.zeros(n_classes, depth, dtype=torch.float32)
    for class_index, path in enumerate(paths):
        for position, (node_index, sign) in enumerate(path):
            obs[class_index, position] = sign
            idx[class_index, position] = node_index
            mask[class_index, position] = 1.0

    return HierarchicalSoftmax(
        obs=obs,
        idx=idx,
        n_obs=depth,
        len=n_classes - 1,
        mask=mask,
        offset=torch.tensor(offsets, dtype=torch.float32),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Label encoding
# ──────────────────────────────────────────────────────────────────────────────


def labels_to_hrc(
    labels: Tensor,
    hrc: HierarchicalSoftmax,
) -> tuple[Tensor, Tensor]:
    """Map integer class labels to HRC observations and node indices.

    Args:
        labels: Integer class labels, shape (B,).
        hrc:    HierarchicalSoftmax from :func:`class_to_obs`.

    Returns:
        y_obs: Float tensor (B, n_obs) of ±1 encoded observations.
        y_idx: Int tensor (B, n_obs) of 1-indexed output node positions.
    """
    device = labels.device
    y_obs = hrc.obs.to(device)[labels.long()]  # (B, n_obs)
    y_idx = hrc.idx.to(device)[labels.long()]  # (B, n_obs)
    return y_obs, y_idx


def labels_to_hrc_mask(
    labels: Tensor,
    hrc: HierarchicalSoftmax,
) -> Tensor | None:
    """Return the per-observation padding mask for a batch of labels.

    Args:
        labels: Integer class labels, shape (B,).
        hrc:    HierarchicalSoftmax with variable-depth paths.

    Returns:
        Float tensor (B, n_obs) that is one on real path factors and zero on
        padding, or None when the tree has a fixed depth.
    """

    if hrc.mask is None:
        return None
    return hrc.mask.to(labels.device)[labels.long()]


# ──────────────────────────────────────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────────────────────────────────────

_INV_SQRT2: float = 1.0 / math.sqrt(2.0)


def _require_fixed_depth(hrc: HierarchicalSoftmax, caller: str) -> None:
    """Reject trees whose factors these fixed-depth helpers would misread."""

    if hrc.mask is not None and not bool((hrc.mask == 1).all()):
        raise ValueError(
            f"{caller} cannot read a variable-depth tree; use "
            "triton_tagi.hrc_probit.hrc_class_probabilities"
        )
    if hrc.offset is not None and bool((hrc.offset != 0).any()):
        raise ValueError(
            f"{caller} ignores branch priors; use triton_tagi.hrc_probit.hrc_class_probabilities"
        )


def obs_to_class_probs(
    ma: Tensor,
    Sa: Tensor,
    hrc: HierarchicalSoftmax,
    alpha: float = 3.0,
) -> Tensor:
    """Convert output layer Gaussians to class probabilities.

    For each tree node i::

        P_z[i] = Phi(ma[i] / sqrt((1/alpha)^2 + Sa[i]))

    For each class r, the probability is the product along its binary path::

        P[r] = prod_c { P_z[idx[r,c]-1]      if obs[r,c] == +1
                      { 1 - P_z[idx[r,c]-1]  if obs[r,c] == −1

    Replicates cuTAGI's ``obs_to_class()`` from ``src/cost.cpp`` with alpha=3.

    Args:
        ma:    Output means, shape (B, hrc.len).
        Sa:    Output variances, shape (B, hrc.len).
        hrc:   HierarchicalSoftmax from :func:`class_to_obs`.
        alpha: Scaling factor matching cuTAGI's default of 3.

    Returns:
        Class probabilities (unnormalised), shape (B, n_classes).

    Raises:
        ValueError: If ``hrc`` has variable-depth paths or branch offsets. Use
            :func:`triton_tagi.hrc_probit.hrc_class_probabilities`, which
            handles both and returns a normalized distribution.
    """
    _require_fixed_depth(hrc, "obs_to_class_probs")
    B = ma.shape[0]
    n_classes = hrc.obs.shape[0]
    device = ma.device

    # Per-node CDF: Phi(ma[i] / sqrt((1/alpha)^2 + Sa[i]))
    sigma = torch.sqrt((1.0 / alpha) ** 2 + Sa)  # (B, hrc.len)
    P_z = 0.5 * (1.0 + torch.erf(ma / sigma * _INV_SQRT2))  # (B, hrc.len)

    # Gather P_z at the required node indices for all n_classes × n_obs combos
    idx_0 = hrc.idx.to(device).long() - 1  # (n_classes, L), 0-indexed
    obs_t = hrc.obs.to(device)  # (n_classes, L)

    # Expand to (B, n_classes, L) for vectorised gather
    idx_exp = idx_0.unsqueeze(0).expand(B, -1, -1)  # (B, n_classes, L)
    node_P = torch.gather(
        P_z.unsqueeze(1).expand(-1, n_classes, -1), 2, idx_exp
    )  # (B, n_classes, L)

    # obs == +1 → factor = P_z;  obs == −1 → factor = 1 − P_z
    obs_exp = obs_t.unsqueeze(0).expand(B, -1, -1)  # (B, n_classes, L)
    factors = torch.where(obs_exp > 0, node_P, 1.0 - node_P)  # (B, n_classes, L)

    return factors.prod(dim=2)  # (B, n_classes)


def obs_to_class_probs_probit(
    ma: Tensor,
    Sa: Tensor,
    hrc: HierarchicalSoftmax,
) -> Tensor:
    """Return HRC path probabilities under the fixed unit-probit model.

    The link variance is structurally fixed at one: it is neither an inferred
    observation variance nor a tunable prediction-time temperature.
    """

    if ma.shape != Sa.shape or ma.dim() != 2 or ma.shape[1] != hrc.len:
        raise ValueError("HRC output moments must have shape (batch, hrc.len)")
    _require_fixed_depth(hrc, "obs_to_class_probs_probit")

    node_scale = torch.sqrt(Sa.clamp_min(0.0) + 1.0)
    positive_probability = 0.5 * (1.0 + torch.erf(ma / node_scale.clamp_min(1e-12) * _INV_SQRT2))
    batch_size = ma.shape[0]
    num_classes = hrc.obs.shape[0]
    node_idx = hrc.idx.to(ma.device).long() - 1
    expanded_idx = node_idx.unsqueeze(0).expand(batch_size, -1, -1)
    path_probability = torch.gather(
        positive_probability.unsqueeze(1).expand(-1, num_classes, -1),
        2,
        expanded_idx,
    )
    signs = hrc.obs.to(ma.device).unsqueeze(0)
    factors = torch.where(signs > 0, path_probability, 1.0 - path_probability)
    return factors.prod(dim=2)


def obs_to_class_probs_tagiv(
    ma: Tensor,
    Sa: Tensor,
    hrc: HierarchicalSoftmax,
    alpha: float = 3.0,
) -> Tensor:
    """Convert interleaved HRC TAGI-V node moments to class probabilities."""

    if ma.shape != Sa.shape or ma.shape[-1] != 2 * hrc.len:
        raise ValueError("HRC TAGI-V output must have width 2 * hrc.len")
    node_mean = ma[..., 0::2]
    node_epistemic = Sa[..., 0::2].clamp_min(0.0)
    node_aleatoric = ma[..., 1::2].clamp_min(0.0)
    return obs_to_class_probs(
        node_mean,
        node_epistemic + node_aleatoric,
        hrc,
        alpha=alpha,
    )


def get_predicted_labels(
    ma: Tensor,
    Sa: Tensor,
    hrc: HierarchicalSoftmax,
    alpha: float = 3.0,
) -> Tensor:
    """Return the predicted class index for each sample.

    Args:
        ma:    Output means, shape (B, hrc.len).
        Sa:    Output variances, shape (B, hrc.len).
        hrc:   HierarchicalSoftmax from :func:`class_to_obs`.
        alpha: Scaling factor (default 3.0).

    Returns:
        Predicted class indices, shape (B,), dtype int64.
    """
    return obs_to_class_probs(ma, Sa, hrc, alpha).argmax(dim=1)


def project_classes_to_nodes(
    hrc: HierarchicalSoftmax,
    weight: Tensor,
    bias: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Project a per-class linear classifier onto the tree's decision nodes.

    A class-indexed classifier cannot be copied onto an HRC head, because the
    head scores tree nodes rather than classes. Node ``j`` decides between the
    classes reachable through its ``+1`` branch and those through its ``-1``
    branch, so the projection that preserves what the node is being asked is
    the contrast between the two branch means::

        w_node = mean(w_class for classes on the +1 branch)
               - mean(w_class for classes on the -1 branch)

    This is invariant to adding a constant vector to every class, which is the
    right gauge here: the tree's latent variables are only defined up to that
    shift. A node with an empty branch (reachable only in the padded tree)
    contributes zero from the missing side.

    Args:
        hrc: Tree structure from :func:`class_to_obs` or
            :func:`class_to_obs_full`.
        weight: Per-class weight, shape (n_classes, in_features), as
            :class:`torch.nn.Linear` stores it.
        bias: Per-class bias, shape (n_classes,). ``None`` means zero.

    Returns:
        ``(node_weight, node_bias)`` with shapes (hrc.len, in_features) and
        (hrc.len,), on the dtype and device of ``weight``.
    """

    if weight.dim() != 2:
        raise ValueError("weight must have shape (n_classes, in_features)")
    if weight.shape[0] != hrc.n_classes:
        raise ValueError(
            f"weight has {weight.shape[0]} classes but the tree encodes {hrc.n_classes}"
        )
    if bias is not None and bias.shape != (hrc.n_classes,):
        raise ValueError("bias must have shape (n_classes,)")

    device, dtype = weight.device, weight.dtype
    resolved_bias = (
        torch.zeros(hrc.n_classes, device=device, dtype=dtype)
        if bias is None
        else bias.to(device=device, dtype=dtype)
    )

    obs = hrc.obs.to(device=device, dtype=dtype)
    idx = hrc.idx.to(device=device, dtype=torch.long) - 1  # stored 1-indexed
    mask = hrc.path_mask(device=device).to(dtype=dtype)

    node_weight = torch.zeros(hrc.len, weight.shape[1], device=device, dtype=dtype)
    node_bias = torch.zeros(hrc.len, device=device, dtype=dtype)
    for sign, target in ((1.0, 1.0), (-1.0, -1.0)):
        # Membership of each (class, level) factor in this branch of each node.
        selected = mask * (obs == sign).to(dtype)  # (n_classes, n_obs)
        counts = torch.zeros(hrc.len, device=device, dtype=dtype)
        counts.index_add_(0, idx.reshape(-1), selected.reshape(-1))
        # Each class contributes its own row to every node whose branch it is on.
        flat_weight = weight.repeat_interleave(hrc.n_obs, dim=0)
        flat_bias = resolved_bias.repeat_interleave(hrc.n_obs)
        sums = torch.zeros_like(node_weight)
        sums.index_add_(0, idx.reshape(-1), selected.reshape(-1, 1) * flat_weight)
        bias_sums = torch.zeros(hrc.len, device=device, dtype=dtype)
        bias_sums.index_add_(0, idx.reshape(-1), selected.reshape(-1) * flat_bias)
        # An empty branch leaves counts at zero; clamping makes it contribute zero.
        safe = counts.clamp_min(1.0)
        node_weight = node_weight + target * sums / safe.unsqueeze(1)
        node_bias = node_bias + target * bias_sums / safe
    return node_weight, node_bias
