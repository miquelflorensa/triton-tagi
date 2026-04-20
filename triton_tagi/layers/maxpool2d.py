"""
Bayesian MaxPool2D for TAGI.

cuTAGI uses a hard-argmax approximation: the output mean is the maximum mean
in each pooling window, and the output variance is the variance at that same
input position.  The Jacobian is 1 at the argmax position and 0 everywhere
else, so backward simply routes the incoming delta to the winner.

Forward:
    argmax[n,c,h_out,w_out] = argmax_{(dh,dw) in window} mu_a[n,c,...]
    mu_z[n,c,h_out,w_out]   = mu_a[n,c, argmax]
    var_z[n,c,h_out,w_out]  = var_a[n,c, argmax]

Backward (jcb = 1 at argmax, 0 elsewhere — matches cuTAGI):
    delta_ma[argmax] += delta_mz   (accumulate for overlapping pools)
    delta_Sa[argmax] += delta_Sz
    delta_ma[other]   = 0
    delta_Sa[other]   = 0
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..base import Layer


class MaxPool2D(Layer):
    """
    Bayesian max-pooling layer (hard-argmax approximation).

    Passes the maximum-mean input activation and its paired variance through
    each pooling window.  Delta propagation routes the signal back to the
    winning position only (Jacobian = 1 at argmax, 0 elsewhere).

    Parameters
    ----------
    kernel_size : int   pooling window size (square)
    stride      : int   stride (default = kernel_size, i.e. non-overlapping)
    padding     : int   zero-padding on each side (default 0)
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

        # Cache populated in forward
        self.pool_idx: Tensor | None = None     # (N, C, H_out, W_out) argmax indices
        self.input_shape: tuple | None = None   # (N, C, H, W)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        ma : Tensor (N, C, H, W)   input activation means
        Sa : Tensor (N, C, H, W)   input activation variances

        Returns
        -------
        mz : Tensor (N, C, H_out, W_out)  pooled output means
        Sz : Tensor (N, C, H_out, W_out)  pooled output variances
        """
        N, C, H, W = ma.shape
        self.input_shape = (N, C, H, W)

        # ── Argmax of means via PyTorch MaxPool ──
        mz, pool_idx = F.max_pool2d_with_indices(
            ma, self.kernel_size, stride=self.stride, padding=self.padding
        )
        # pool_idx: (N, C, H_out, W_out) — flat indices into the H×W plane

        # ── Gather variance at the argmax positions ──
        H_out, W_out = mz.shape[2], mz.shape[3]
        Sa_flat = Sa.view(N, C, H * W)
        idx_flat = pool_idx.view(N, C, H_out * W_out)
        Sz = Sa_flat.gather(2, idx_flat).view(N, C, H_out, W_out)

        # ── Cache for backward ──
        self.pool_idx = pool_idx

        return mz, Sz

    # ------------------------------------------------------------------
    #  Backward
    # ------------------------------------------------------------------
    def backward(self, delta_mz: Tensor, delta_Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Route incoming deltas to the argmax input positions.

        The Jacobian is 1 at the selected position and 0 elsewhere (hard
        argmax), so delta_ma = delta_mz at the argmax and 0 everywhere else.
        For overlapping pools, multiple output positions may map to the same
        input, so deltas are accumulated (scatter_add).

        Parameters
        ----------
        delta_mz : Tensor (N, C, H_out, W_out)  mean delta from next layer
        delta_Sz : Tensor (N, C, H_out, W_out)  variance delta from next layer

        Returns
        -------
        delta_ma : Tensor (N, C, H, W)  mean delta to propagate
        delta_Sa : Tensor (N, C, H, W)  variance delta to propagate
        """
        N, C, H, W = self.input_shape
        H_out, W_out = delta_mz.shape[2], delta_mz.shape[3]

        delta_ma = torch.zeros(N, C, H * W, device=delta_mz.device, dtype=delta_mz.dtype)
        delta_Sa = torch.zeros_like(delta_ma)

        idx_flat = self.pool_idx.view(N, C, H_out * W_out)
        delta_ma.scatter_add_(2, idx_flat, delta_mz.view(N, C, H_out * W_out))
        delta_Sa.scatter_add_(2, idx_flat, delta_Sz.view(N, C, H_out * W_out))

        return delta_ma.view(N, C, H, W), delta_Sa.view(N, C, H, W)

    def __repr__(self):
        return (
            f"MaxPool2D(kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding})"
        )
