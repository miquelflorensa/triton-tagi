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
        n_obs: Number of bits per class = ceil(log2(n_classes)).
        len:   Total number of unique nodes in the tree (= output layer width).
    """

    obs: Tensor  # (n_classes, n_obs)  float32, values ∈ {+1, −1}
    idx: Tensor  # (n_classes, n_obs)  int32,   1-indexed
    n_obs: int
    len: int


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
        idx=torch.tensor(idx, dtype=torch.int32),    # (n_classes, L)
        n_obs=L,
        len=tree_len,
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
    y_obs = hrc.obs.to(device)[labels.long()]   # (B, n_obs)
    y_idx = hrc.idx.to(device)[labels.long()]   # (B, n_obs)
    return y_obs, y_idx


# ──────────────────────────────────────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────────────────────────────────────

_INV_SQRT2: float = 1.0 / math.sqrt(2.0)


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
    """
    B = ma.shape[0]
    n_classes = hrc.obs.shape[0]
    device = ma.device

    # Per-node CDF: Phi(ma[i] / sqrt((1/alpha)^2 + Sa[i]))
    sigma = torch.sqrt((1.0 / alpha) ** 2 + Sa)      # (B, hrc.len)
    P_z = 0.5 * (1.0 + torch.erf(ma / sigma * _INV_SQRT2))  # (B, hrc.len)

    # Gather P_z at the required node indices for all n_classes × n_obs combos
    idx_0 = hrc.idx.to(device).long() - 1             # (n_classes, L), 0-indexed
    obs_t = hrc.obs.to(device)                         # (n_classes, L)

    # Expand to (B, n_classes, L) for vectorised gather
    idx_exp = idx_0.unsqueeze(0).expand(B, -1, -1)     # (B, n_classes, L)
    node_P = torch.gather(
        P_z.unsqueeze(1).expand(-1, n_classes, -1), 2, idx_exp
    )                                                  # (B, n_classes, L)

    # obs == +1 → factor = P_z;  obs == −1 → factor = 1 − P_z
    obs_exp = obs_t.unsqueeze(0).expand(B, -1, -1)     # (B, n_classes, L)
    factors = torch.where(obs_exp > 0, node_P, 1.0 - node_P)  # (B, n_classes, L)

    return factors.prod(dim=2)                         # (B, n_classes)


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
