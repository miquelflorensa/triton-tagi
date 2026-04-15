"""
Leaky ReLU activation layer for TAGI.

Follows the approximated logic where propagation depends on the mean:
    If μ_z > 0:
        μ_a = μ_z
        J = 1
        S_a = S_z
    Else:
        μ_a = α * μ_z
        J = α
        S_a = α² * S_z
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from ..base import Layer

BLOCK = 1024


@triton.jit
def _leaky_relu_kernel(
    mz_ptr,
    Sz_ptr,
    ma_ptr,
    Sa_ptr,
    J_ptr,
    alpha,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    mz = tl.load(mz_ptr + offs, mask=valid, other=0.0)
    Sz = tl.load(Sz_ptr + offs, mask=valid, other=0.0)

    is_positive = mz > 0.0

    mu_a = tl.where(is_positive, mz, alpha * mz)
    jcb = tl.where(is_positive, 1.0, alpha)
    var_a = tl.where(is_positive, Sz, alpha * alpha * Sz)

    tl.store(ma_ptr + offs, mu_a, mask=valid)
    tl.store(Sa_ptr + offs, var_a, mask=valid)
    tl.store(J_ptr + offs, jcb, mask=valid)


def leaky_relu(mz, Sz, alpha):
    """
    Compute moments of leaky rectified linear unit using approximated mean-based logic.

    Parameters
    ----------
    mz : Tensor  pre-activation means   (any shape, processed flat)
    Sz : Tensor  pre-activation variances
    alpha : float negative slope

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

    _leaky_relu_kernel[(triton.cdiv(n, BLOCK),)](
        mz,
        Sz,
        ma,
        Sa,
        J,
        alpha,
        n,
        BLOCK=BLOCK,
    )
    return ma, Sa, J


class LeakyReLU(Layer):
    """
    Leaky ReLU activation layer.

    Stores the Jacobian from the forward pass for use during backward.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
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
        ma, Sa, J = leaky_relu(mz.reshape(-1), Sz.reshape(-1), self.alpha)
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
        return f"LeakyReLU(alpha={self.alpha})"
