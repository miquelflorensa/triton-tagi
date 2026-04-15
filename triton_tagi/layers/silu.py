"""
Bayesian SiLU (Swish) activation layer for TAGI.

SiLU(z) = z · σ(z)  where σ is the logistic sigmoid.

Using the sigmoid moment approximation from cuTAGI:
    σ(μ_z) = 1 / (1 + exp(-μ_z))

For the product z · σ(z) with z ~ N(μ_z, S_z):
    μ_a = μ_z · sig  +  sig · (1 - sig) · S_z
    J   = sig  +  μ_z · sig · (1 - sig)
    S_a = J² · S_z
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
def _bayesian_silu_kernel(
    mz_ptr,
    Sz_ptr,
    ma_ptr,
    Sa_ptr,
    J_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    mz = tl.load(mz_ptr + offs, mask=valid, other=0.0)
    Sz = tl.load(Sz_ptr + offs, mask=valid, other=0.0)

    # Sigmoid moments (from cuTAGI approximation)
    sig = 1.0 / (1.0 + tl.exp(-mz))
    sig_prime = sig * (1.0 - sig)

    # SiLU mean: E[z · σ(z)] ≈ μ_z · σ(μ_z) + σ'(μ_z) · S_z
    mu_a = mz * sig + sig_prime * Sz

    # SiLU Jacobian: d/dz [z · σ(z)] = σ(z) + z · σ'(z)
    J = sig + mz * sig_prime

    # SiLU variance: J² · S_z
    var_a = J * J * Sz
    var_a = tl.maximum(var_a, 1e-12)

    tl.store(ma_ptr + offs, mu_a, mask=valid)
    tl.store(Sa_ptr + offs, var_a, mask=valid)
    tl.store(J_ptr + offs, J, mask=valid)


# ======================================================================
#  Python API
# ======================================================================


def bayesian_silu(mz, Sz):
    """
    Compute approximate moments of SiLU-transformed Gaussian.

    Parameters
    ----------
    mz : Tensor  pre-activation means   (any shape, processed flat)
    Sz : Tensor  pre-activation variances

    Returns
    -------
    ma : Tensor  post-activation means
    Sa : Tensor  post-activation variances
    J  : Tensor  Jacobian, used during backward
    """
    n = mz.numel()
    ma = torch.empty_like(mz)
    Sa = torch.empty_like(Sz)
    J = torch.empty_like(mz)
    _bayesian_silu_kernel[(triton.cdiv(n, BLOCK),)](
        mz,
        Sz,
        ma,
        Sa,
        J,
        n,
        BLOCK=BLOCK,
    )
    return ma, Sa, J


class SiLU(Layer):
    """
    Bayesian SiLU (Swish) activation layer.

    Stores the Jacobian from the forward pass for use during backward.
    """

    def __init__(self) -> None:
        self.J = None

    def forward(self, mz: Tensor, Sz: Tensor) -> tuple[Tensor, Tensor]:
        original_shape = mz.shape
        Sz_flat = Sz.reshape(-1)
        ma, Sa, J = bayesian_silu(mz.reshape(-1), Sz_flat)

        # Cauchy–Schwarz bound: |J| ≤ √(Sa / Sz)
        cs_bound = torch.sqrt(Sa / torch.clamp(Sz_flat, min=1e-12))
        J = torch.clamp(J, -cs_bound, cs_bound)

        self.J = J.view(original_shape)
        return ma.view(original_shape), Sa.view(original_shape)

    def backward(self, delta_mz: Tensor, delta_Sz: Tensor) -> tuple[Tensor, Tensor]:
        J = self.J
        return delta_mz * J, delta_Sz * J * J

    def __repr__(self):
        return "SiLU()"
