"""
Remax activation layer for TAGI.

Remax = rectified-Gaussian softmax alternative for Bayesian networks.

Given logits Z ~ N(μ_z, S_z), computes output probabilities A = M / Σ M
where M_k = max(0, Z_k) is the rectified Gaussian.

Forward (cuTAGI parity — L. Alric, 2024):
    1. MixtureReLU moments of M = max(0, Z)
    2. Log-normal moments of M and M̃ = Σ M_k
    3. cov(ln M, ln M̃) via log-normal identity
    4. A = exp(ln M - ln M̃); renormalize so Σ μ_A = 1
    5. cov(A, M) via log-normal identity
    6. cov(A, Z) = cov(A, M) / Φ(α)   (capped by Cauchy–Schwarz)
    7. Jacobian  J = cov(A, Z) / var(Z)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from ..base import Layer

# ======================================================================
#  Triton kernel — Remax (MixtureReLU + log-normal path, cuTAGI parity)
# ======================================================================


@triton.jit
def _remax_kernel(
    mu_z_ptr,
    var_z_ptr,            # (B, K) inputs
    mu_a_ptr,
    var_a_ptr,            # (B, K) outputs
    J_ptr,                # (B, K) Jacobian cov(A,Z)/var(Z)
    K,                    # number of classes (runtime)
    stride_b,
    BLOCK_K: tl.constexpr,
):
    """One program ↔ one batch item.  All K classes in registers."""
    b = tl.program_id(0)
    base = b * stride_b
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    EPS: tl.constexpr = 1e-6              # match cuTAGI's floor
    INV_SQRT_2PI: tl.constexpr = 0.3989422804014327
    INV_SQRT_2:   tl.constexpr = 0.7071067811865475

    # ── load Z moments ──
    mu_z = tl.load(mu_z_ptr + base + offs, mask=mask, other=0.0)
    # other=1.0 for var_z so masked lanes don't pollute sums / divisions
    var_z = tl.load(var_z_ptr + base + offs, mask=mask, other=1.0)

    # ── 1. MixtureReLU moments  M = max(0, Z)  (L. Alric, 2024) ──
    std_z = tl.sqrt(var_z)
    alpha = mu_z / std_z
    pdf_alpha = INV_SQRT_2PI * tl.exp(-0.5 * alpha * alpha)
    cdf_alpha = 0.5 + 0.5 * tl.math.erf(alpha * INV_SQRT_2)

    mu_m = mu_z * cdf_alpha + std_z * pdf_alpha
    mu_m = tl.maximum(mu_m, EPS)

    var_m = (
        -mu_m * mu_m
        + 2.0 * mu_m * mu_z
        - mu_z * std_z * pdf_alpha
        + (var_z - mu_z * mu_z) * cdf_alpha
    )
    var_m = tl.maximum(var_m, EPS)
    jcb_m = cdf_alpha                      # = Φ(α), used later

    # ── 2. Sum moments M̃ = Σ M_k ──
    mu_mt  = tl.maximum(tl.sum(tl.where(mask, mu_m,  0.0)), EPS)
    var_mt = tl.maximum(tl.sum(tl.where(mask, var_m, 0.0)), EPS)

    # ── 3. Log-normal moments of M and M̃ ──
    var_log_m  = tl.log(1.0 + var_m  / (mu_m  * mu_m))
    mu_log_m   = tl.log(mu_m)  - 0.5 * var_log_m
    var_log_mt = tl.log(1.0 + var_mt / (mu_mt * mu_mt))
    mu_log_mt  = tl.log(mu_mt) - 0.5 * var_log_mt

    # ── 4. cov(ln M_k, ln M̃) ──
    cov_log_m_mt = tl.log(1.0 + var_m / (mu_m * mu_mt))

    # ── 5. Moments of ln(A) ──
    mu_log_a  = mu_log_m - mu_log_mt
    var_log_a = tl.maximum(var_log_m + var_log_mt - 2.0 * cov_log_m_mt, 0.0)

    # ── 6. Remax probabilities (renormalized) ──
    mu_a_raw = tl.maximum(tl.exp(mu_log_a + 0.5 * var_log_a), EPS)
    sum_mu_a = tl.maximum(tl.sum(tl.where(mask, mu_a_raw, 0.0)), EPS)
    mu_a     = mu_a_raw / sum_mu_a
    var_a    = (tl.exp(var_log_a) - 1.0) * mu_a * mu_a

    # ── 7. Jacobian J = cov(A, Z) / var(Z) ──
    # cov(ln A, ln M) = var_log_m - cov_log_m_mt
    # cov(A, M) via log-normal identity: (exp(·) - 1) · μ_A · μ_M
    # cov(A, Z) = cov(A, M) / Φ(α), capped by Cauchy–Schwarz bound
    cov_log_a_log_m = var_log_m - cov_log_m_mt
    cov_a_m = (tl.exp(cov_log_a_log_m) - 1.0) * mu_a * mu_m

    cs_bound = tl.sqrt(var_a * var_z)
    cov_a_z  = tl.minimum(cs_bound, cov_a_m / tl.maximum(jcb_m, EPS))
    J        = cov_a_z / var_z

    # ── store ──
    tl.store(mu_a_ptr + base + offs, mu_a, mask=mask)
    tl.store(var_a_ptr + base + offs, var_a, mask=mask)
    tl.store(J_ptr     + base + offs, J,    mask=mask)


# ======================================================================
#  Python wrapper
# ======================================================================


def triton_remax(mu_z: Tensor, var_z: Tensor):
    """
    Compute Remax moments and Jacobian using fused Triton kernel.

    Parameters
    ----------
    mu_z  : Tensor (B, K)  logit means
    var_z : Tensor (B, K)  logit variances

    Returns
    -------
    mu_a  : Tensor (B, K)  probability means (Σ μ_a ≈ 1)
    var_a : Tensor (B, K)  probability variances
    J     : Tensor (B, K)  Jacobian  J = cov(A, Z) / var(Z)
    """
    squeeze = mu_z.dim() == 1
    if squeeze:
        mu_z = mu_z.unsqueeze(0)
        var_z = var_z.unsqueeze(0)

    mu_z = mu_z.contiguous()
    var_z = var_z.contiguous()
    B, K = mu_z.shape

    mu_a  = torch.empty_like(mu_z)
    var_a = torch.empty_like(mu_z)
    J     = torch.empty_like(mu_z)

    BLOCK_K = triton.next_power_of_2(K)
    _remax_kernel[(B,)](
        mu_z, var_z, mu_a, var_a, J,
        K, mu_z.stride(0),
        BLOCK_K=BLOCK_K,
    )

    if squeeze:
        return mu_a.squeeze(0), var_a.squeeze(0), J.squeeze(0)
    return mu_a, var_a, J


# ======================================================================
#  Remax Layer
# ======================================================================


class Remax(Layer):
    """
    Remax activation layer (softmax alternative for Bayesian networks).

    Forward:  (μ_z, S_z) → (μ_a, S_a)  with Σ μ_a ≈ 1
    Backward: uses J = cov(Z, A) / var(Z) as Jacobian

    Stores J from the forward pass for use during backward.
    """

    def __init__(self) -> None:
        self.J: Tensor | None = None  # stored Jacobian

    def forward(self, mz: Tensor, Sz: Tensor) -> tuple[Tensor, Tensor]:
        mu_a, Sa, J = triton_remax(mz, Sz)
        self.J = J
        return mu_a, Sa

    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        """Propagate deltas through a_k ≈ μ_a_k + J_k · (z_k − μ_z_k)."""
        J = self.J
        return delta_ma * J, delta_Sa * J * J

    def __repr__(self) -> str:
        return "Remax()"
