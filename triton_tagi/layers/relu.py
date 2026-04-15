"""
Bayesian ReLU activation layer for TAGI.

Computes the exact moments of a = max(0, z) where z ~ N(μ_z, S_z).

Using α = μ_z / σ_z :
    μ_a = σ_z · φ(α)  +  μ_z · Φ(α)
    S_a = −μ_a² + 2μ_a·μ_z − μ_z·σ_z·φ(α) + (S_z − μ_z²)·Φ(α)
    J   = Φ(α)     (Jacobian = CDF, used during backward)

where φ and Φ are the standard normal PDF and CDF respectively.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from ..base import Layer

BLOCK = 1024


# ======================================================================
#  Triton kernel
# ======================================================================


@triton.jit
def _bayesian_relu_kernel(
    mz_ptr,
    Sz_ptr,
    ma_ptr,
    Sa_ptr,
    J_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    INV_SQRT_2PI: tl.constexpr = 0.3989422804014327  # 1/√(2π)
    INV_SQRT_2: tl.constexpr = 0.7071067811865476  # 1/√2

    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    mz = tl.load(mz_ptr + offs, mask=valid, other=0.0)
    Sz = tl.load(Sz_ptr + offs, mask=valid, other=0.0)

    Sz_safe = tl.maximum(Sz, 1e-12)
    sigma_z = tl.sqrt(Sz_safe)
    alpha = mz / sigma_z

    pdf = tl.exp(-0.5 * alpha * alpha) * INV_SQRT_2PI
    cdf = 0.5 * (1.0 + tl.math.erf(alpha * INV_SQRT_2))

    mu_a = tl.maximum(sigma_z * pdf + mz * cdf, 1e-7)
    var_a = -mu_a * mu_a + 2.0 * mu_a * mz - mz * sigma_z * pdf + (Sz_safe - mz * mz) * cdf
    var_a = tl.maximum(var_a, 1e-7)

    tl.store(ma_ptr + offs, mu_a, mask=valid)
    tl.store(Sa_ptr + offs, var_a, mask=valid)
    tl.store(J_ptr + offs, cdf, mask=valid)


# ======================================================================
#  Python API
# ======================================================================


def bayesian_relu(mz, Sz):
    """
    Compute exact moments of a rectified Gaussian.

    Parameters
    ----------
    mz : Tensor  pre-activation means   (any shape, processed flat)
    Sz : Tensor  pre-activation variances

    Returns
    -------
    ma : Tensor  post-activation means
    Sa : Tensor  post-activation variances
    J  : Tensor  Jacobian = Φ(α), used during backward
    """
    n = mz.numel()
    ma = torch.empty_like(mz)
    Sa = torch.empty_like(Sz)
    J = torch.empty_like(mz)
    _bayesian_relu_kernel[(triton.cdiv(n, BLOCK),)](
        mz,
        Sz,
        ma,
        Sa,
        J,
        n,
        BLOCK=BLOCK,
    )
    return ma, Sa, J


class ReLU(Layer):
    """
    Bayesian ReLU activation layer.

    Stores the Jacobian from the forward pass for use during backward.
    """

    def __init__(self) -> None:
        self.J = None  # stored Jacobian

    def forward(self, mz: Tensor, Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        mz : Tensor (B, D)  pre-activation means
        Sz : Tensor (B, D)  pre-activation variances

        Returns
        -------
        ma : Tensor (B, D)  post-activation means
        Sa : Tensor (B, D)  post-activation variances
        """
        original_shape = mz.shape
        Sz_flat = Sz.reshape(-1)
        ma, Sa, J = bayesian_relu(mz.reshape(-1), Sz_flat)

        # Cauchy–Schwarz: |Cov(Z,A)| = |J·Sz| ≤ √(Sz·Sa)  →  J ≤ √(Sa/Sz)
        cs_bound = torch.sqrt(Sa / torch.clamp(Sz_flat, min=1e-12))
        J = torch.minimum(J, cs_bound)

        self.J = J.view(original_shape)
        return ma.view(original_shape), Sa.view(original_shape)

    def backward(self, delta_mz: Tensor, delta_Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply stored Jacobian to the deltas.

        J for mean, J² for variance (chain rule on Gaussian moments).

        Parameters
        ----------
        delta_mz : Tensor (B, D)  mean delta
        delta_Sz : Tensor (B, D)  variance delta

        Returns
        -------
        delta_ma : Tensor (B, D)  mean delta to propagate
        delta_Sa : Tensor (B, D)  variance delta to propagate
        """
        J = self.J
        return delta_mz * J, delta_Sz * J * J

    def __repr__(self):
        return "ReLU()"
