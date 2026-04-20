"""
Bernoulli (max-indicator) activation layer for TAGI.

Computes P(Y_i = 1) = P(Z_i = max_j Z_j) using Gauss-Hermite quadrature.

Given logits Z ~ N(μ_z, S_z), the probability that class i has the
largest logit is:
    P_i = E[ Π_{j≠i} Φ((z - μ_j) / σ_j) ]

integrated over z ~ N(μ_i, σ_i²) using Gauss-Hermite nodes.

Outputs:
    P   : probability of being the max class   (sum to 1)
    V   : Bernoulli variance  P_i · (1 − P_i)
    C   : covariance  cov(Z_i, A_i)         (for backward Jacobian)

Complexity per sample:  O(K² · n_gh)  where K = num classes.
For large K, use top_m truncation for  O(K + M² · n_gh).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from numpy.polynomial.hermite import hermgauss
from torch import Tensor

from ..base import Layer

# ======================================================================
#  Gauss-Hermite node cache
# ======================================================================

_gh_cache = {}


def _get_gh(n_gh, device, dtype=torch.float32):
    """Get cached Gauss-Hermite nodes and weights on the target device."""
    key = (n_gh, str(device), dtype)
    if key not in _gh_cache:
        t, w = hermgauss(n_gh)
        _gh_cache[key] = (
            torch.tensor(t, dtype=dtype, device=device),
            torch.tensor(w, dtype=dtype, device=device),
        )
    return _gh_cache[key]


# ======================================================================
#  Triton kernel — Gauss-Hermite max-indicator moments
# ======================================================================


@triton.jit
def _hermite_kernel(
    mu_ptr,
    sig_ptr,  # (B, K)  contiguous
    nodes_ptr,
    weights_ptr,  # (n_gh,)
    P_ptr,
    EXf_ptr,  # (B, K)  output
    K,  # number of classes   [runtime]
    n_gh,  # number of GH nodes  [runtime]
    stride_b,  # row stride = K for contiguous
    N_GH: tl.constexpr,  # >= n_gh, power-of-2
):
    """One program ↔ one (batch_item, class_i)."""

    pid = tl.program_id(0)
    b_idx = pid // K
    i = pid % K
    base = b_idx * stride_b

    # ── pivot class ──
    mu_i = tl.load(mu_ptr + base + i)
    sig_i = tl.load(sig_ptr + base + i)
    sig_i = tl.maximum(sig_i, 1e-12)

    # ── GH nodes & weights into registers ──
    gh = tl.arange(0, N_GH)
    mask = gh < n_gh
    t = tl.load(nodes_ptr + gh, mask=mask, other=0.0)
    w = tl.load(weights_ptr + gh, mask=mask, other=0.0)

    # ── evaluation points  x_k = √2 σ_i t_k + μ_i ──
    SQRT2: tl.constexpr = 1.4142135623730951
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = SQRT2 * sig_i * t + mu_i  # (N_GH,)

    # ── accumulate  Σ_j log Φ_j(x_k)  over ALL j (including self) ──
    acc = tl.zeros((N_GH,), dtype=tl.float32)

    for j in range(0, K):
        mu_j = tl.load(mu_ptr + base + j)
        sig_j = tl.load(sig_ptr + base + j)
        sig_j = tl.maximum(sig_j, 1e-12)
        z = (x - mu_j) / sig_j * INV_SQRT2
        cdf = 0.5 + 0.5 * tl.math.erf(z)
        cdf = tl.maximum(cdf, 1e-30)
        acc += tl.log(cdf)

    # ── subtract self-term  j == i ──
    z_s = (x - mu_i) / sig_i * INV_SQRT2
    cdf_s = 0.5 + 0.5 * tl.math.erf(z_s)
    cdf_s = tl.maximum(cdf_s, 1e-12)
    acc -= tl.log(cdf_s)

    # ── mask padding ──
    acc = tl.where(mask, acc, 0.0)
    pe = tl.exp(acc)

    # ── weighted sums  ( ÷ √π ) ──
    INV_SQRT_PI: tl.constexpr = 0.5641895835477563
    ws = w * INV_SQRT_PI

    P_val = tl.sum(ws * pe)
    EXf_val = tl.sum(ws * x * pe)

    tl.store(P_ptr + base + i, P_val)
    tl.store(EXf_ptr + base + i, EXf_val)


# ======================================================================
#  Python wrapper — full computation
# ======================================================================


def triton_bernoulli_moments(mu_z, sigma_z_sq, n_gh=32):
    """
    Compute Bernoulli max-indicator moments using Gauss-Hermite quadrature.

    Parameters
    ----------
    mu_z       : Tensor (B, K) or (K,)  logit means
    sigma_z_sq : Tensor (B, K) or (K,)  logit variances
    n_gh       : int                     quadrature order (default 32)

    Returns
    -------
    P   : Tensor  probability of being the max class (sum ≈ 1)
    V   : Tensor  Bernoulli variance  P·(1−P)
    C   : Tensor  covariance  cov(Z_i, A_i)
    """
    squeeze = mu_z.dim() == 1
    if squeeze:
        mu_z = mu_z.unsqueeze(0)
        sigma_z_sq = sigma_z_sq.unsqueeze(0)

    mu_z = mu_z.contiguous()
    sigma_z = torch.sqrt(torch.clamp(sigma_z_sq, min=1e-12)).contiguous()
    B, K = mu_z.shape

    nodes, weights = _get_gh(n_gh, mu_z.device, mu_z.dtype)

    P = torch.empty_like(mu_z)
    EXf = torch.empty_like(mu_z)

    N_GH = triton.next_power_of_2(n_gh)
    _hermite_kernel[(B * K,)](
        mu_z,
        sigma_z,
        nodes,
        weights,
        P,
        EXf,
        K,
        n_gh,
        mu_z.stride(0),
        N_GH=N_GH,
    )

    # normalise + derive variance & covariance
    P = P / P.sum(dim=-1, keepdim=True).clamp(min=1e-7)
    V = (P * (1.0 - P)).clamp(min=1e-7)
    C = EXf - mu_z * P

    if squeeze:
        return P.squeeze(0), V.squeeze(0), C.squeeze(0)
    return P, V, C


# ======================================================================
#  Python wrapper — fast with Top-M truncation
# ======================================================================


def triton_bernoulli_moments_fast(mu_z, sigma_z_sq, n_gh=16, top_m=None):
    """
    Fast Bernoulli moments with optional Top-M truncation.

    Classes with low means have P_i ≈ 0 to machine precision.
    We select the M most competitive classes (by mean), run the exact
    kernel on just those M, and assign P=V=C=0 to the rest.

    Complexity
    ----------
    Without top_m :  O(B · K² · n_gh)
    With top_m=M  :  O(B · (K + M² · n_gh))   [linear in K for fixed M]

    Parameters
    ----------
    mu_z       : Tensor (B, K) or (K,)
    sigma_z_sq : Tensor (B, K) or (K,)
    n_gh       : int   (default 16)
    top_m      : int or None  (if set, only compute for top M classes)
    """
    squeeze = mu_z.dim() == 1
    if squeeze:
        mu_z = mu_z.unsqueeze(0)
        sigma_z_sq = sigma_z_sq.unsqueeze(0)

    B, K = mu_z.shape

    if top_m is None or top_m >= K:
        result = triton_bernoulli_moments(mu_z, sigma_z_sq, n_gh=n_gh)
        if squeeze:
            return tuple(r.squeeze(0) for r in result)
        return result

    M = top_m
    sigma_z = torch.sqrt(torch.clamp(sigma_z_sq, min=1e-12))

    # Step 1: select top-M classes by mean
    _, top_idx = torch.topk(mu_z, M, dim=-1)

    # Step 2: gather their mu and sigma
    mu_sub = torch.gather(mu_z, 1, top_idx).contiguous()
    sig_sub = torch.gather(sigma_z, 1, top_idx).contiguous()

    # Step 3: run exact kernel on (B, M)
    nodes, weights = _get_gh(n_gh, mu_z.device, mu_z.dtype)
    P_sub = torch.empty_like(mu_sub)
    EXf_sub = torch.empty_like(mu_sub)

    N_GH = triton.next_power_of_2(n_gh)
    _hermite_kernel[(B * M,)](
        mu_sub,
        sig_sub,
        nodes,
        weights,
        P_sub,
        EXf_sub,
        M,
        n_gh,
        mu_sub.stride(0),
        N_GH=N_GH,
    )

    P_sub = P_sub / P_sub.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    V_sub = P_sub * (1.0 - P_sub)
    C_sub = EXf_sub - mu_sub * P_sub

    # Step 4: scatter back to full shape
    P = torch.zeros(B, K, device=mu_z.device, dtype=mu_z.dtype)
    V = torch.zeros(B, K, device=mu_z.device, dtype=mu_z.dtype)
    C = torch.zeros(B, K, device=mu_z.device, dtype=mu_z.dtype)
    P.scatter_(1, top_idx, P_sub)
    V.scatter_(1, top_idx, V_sub)
    C.scatter_(1, top_idx, C_sub)

    if squeeze:
        return P.squeeze(0), V.squeeze(0), C.squeeze(0)
    return P, V, C


# ======================================================================
#  Bernoulli Layer
# ======================================================================


class Bernoulli(Layer):
    """
    Bernoulli (max-indicator) activation layer for TAGI.

    Computes P(Z_i = max) using Gauss-Hermite quadrature and propagates
    deltas backward using J_k = cov(z_k, a_k) / S_z_k.

    Parameters
    ----------
    n_gh  : int   Gauss-Hermite quadrature order (default 32)
    top_m : int or None  if set, use Top-M truncation for speed
    """

    def __init__(self, n_gh: int = 32, top_m: int | None = None) -> None:
        self.n_gh = n_gh
        self.top_m = top_m
        self.J = None  # stored Jacobian

    def forward(self, mz: Tensor, Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute Bernoulli output moments.

        Parameters
        ----------
        mz : Tensor (B, K)  logit means
        Sz : Tensor (B, K)  logit variances

        Returns
        -------
        P  : Tensor (B, K)  probability of being max class
        V  : Tensor (B, K)  Bernoulli variance  P·(1−P)
        """
        if self.top_m is not None:
            P, V, C = triton_bernoulli_moments_fast(mz, Sz, n_gh=self.n_gh, top_m=self.top_m)
        else:
            P, V, C = triton_bernoulli_moments(mz, Sz, n_gh=self.n_gh)

        # Apply Cauchy–Schwarz inequality to cov_z_a
        C = torch.clamp(C, -torch.sqrt(Sz * V), torch.sqrt(Sz * V))

        # Jacobian: J_k = cov(z_k, a_k) / Sz_k
        self.J = C / torch.clamp(Sz, min=1e-7)

        return P, V

    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Propagate deltas from probability space back to logit space.

        Parameters
        ----------
        delta_ma : Tensor (B, K)  mean delta in probability space
        delta_Sa : Tensor (B, K)  variance delta in probability space

        Returns
        -------
        delta_mz : Tensor (B, K)  mean delta in logit space
        delta_Sz : Tensor (B, K)  variance delta in logit space
        """
        J = self.J
        return delta_ma * J, delta_Sa * J * J

    def __repr__(self):
        extra = f", top_m={self.top_m}" if self.top_m else ""
        return f"Bernoulli(n_gh={self.n_gh}{extra})"
