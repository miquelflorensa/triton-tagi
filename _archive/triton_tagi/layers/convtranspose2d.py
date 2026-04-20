"""
Bayesian ConvTranspose2D (transposed convolution / deconvolution) for TAGI.

Forward (moment propagation):
    mz = F.conv_transpose2d(ma, mw_4d, bias=mb)
    Sz = F.conv_transpose2d(Sa, mw²+Sw) + F.conv_transpose2d(ma², Sw) + Sb

Backward (delta propagation):
    delta_ma = F.conv2d(delta_mz, mw_4d)          ← transpose of the forward map
    delta_Sa = F.conv2d(delta_Sz, mw_4d²)

    grad_mw via F.unfold(delta_mz) @ ma_flat.T    ← cross-correlation formula

Weight layout (triton convention):
    mw, Sw : (K, C_out)  where K = C_in · kH · kW
    mb, Sb : (1, C_out)

Conversion to PyTorch conv_transpose2d weight (C_in, C_out, kH, kW):
    mw.view(C_in, kH, kW, C_out).permute(0, 3, 1, 2)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..base import LearnableLayer
from ..param_init import init_weight_bias_conv2d
from ..update.parameters import update_parameters


class ConvTranspose2D(LearnableLayer):
    """
    Bayesian transposed Conv2D layer with Gaussian weight distributions.

    Parameters
    ----------
    C_in        : int   input channels
    C_out       : int   output channels
    kernel_size : int   square kernel size
    stride      : int   (default 1)
    padding     : int   removed from each side of output (default 0)
    device      : str or torch.device
    init_method : str   "He" or "Xavier" (default "He")
    gain_w      : float gain multiplier for weight variance (default 1.0)
    gain_b      : float gain multiplier for bias variance (default 1.0)
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        device: str = "cuda",
        init_method: str = "He",
        gain_w: float = 1.0,
        gain_b: float = 1.0,
    ) -> None:
        self.C_in = C_in
        self.C_out = C_out
        self.kH = self.kW = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = torch.device(device)

        self.mw, self.Sw, self.mb, self.Sb = init_weight_bias_conv2d(
            kernel_size,
            C_in,
            C_out,
            init_method=init_method,
            gain_w=gain_w,
            gain_b=gain_b,
            device=self.device,
        )
        self.has_bias = True

        # Stored for backward pass
        self.ma_in: Tensor | None = None
        self.input_shape: tuple | None = None

        # Parameter deltas (set by backward, consumed by update)
        self.delta_mw: Tensor | None = None
        self.delta_Sw: Tensor | None = None
        self.delta_mb: Tensor | None = None
        self.delta_Sb: Tensor | None = None

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        ma : Tensor (N, C_in, H, W)   activation means
        Sa : Tensor (N, C_in, H, W)   activation variances

        Returns
        -------
        mz : Tensor (N, C_out, H_out, W_out)   pre-activation means
        Sz : Tensor (N, C_out, H_out, W_out)   pre-activation variances

        H_out = (H - 1) * stride - 2 * padding + kH
        """
        N, C_in, H, W = ma.shape
        self.ma_in = ma
        self.input_shape = (N, C_in, H, W)

        # triton (K, C_out) → PyTorch conv_transpose2d (C_in, C_out, kH, kW)
        mw_4d = self.mw.view(C_in, self.kH, self.kW, self.C_out).permute(0, 3, 1, 2).contiguous()
        Sw_4d = self.Sw.view(C_in, self.kH, self.kW, self.C_out).permute(0, 3, 1, 2).contiguous()

        # ── Mean ──
        mz = F.conv_transpose2d(
            ma, mw_4d, bias=self.mb.view(self.C_out), stride=self.stride, padding=self.padding
        )

        # ── Variance ──
        # Sz = (mw²+Sw)*Sa  +  Sw*ma²  +  Sb    (summed over contributing positions)
        mw2_Sw_4d = mw_4d**2 + Sw_4d
        Sz = (
            F.conv_transpose2d(Sa, mw2_Sw_4d, stride=self.stride, padding=self.padding)
            + F.conv_transpose2d(ma**2, Sw_4d, stride=self.stride, padding=self.padding)
            + self.Sb.view(1, self.C_out, 1, 1)
        )

        return mz, Sz

    # ------------------------------------------------------------------
    #  Backward (compute deltas only — NO parameter update)
    # ------------------------------------------------------------------
    def backward(self, delta_mz: Tensor, delta_Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute parameter deltas and propagate delta to the previous layer.

        Parameters
        ----------
        delta_mz : Tensor (N, C_out, H_out, W_out)
        delta_Sz : Tensor (N, C_out, H_out, W_out)

        Returns
        -------
        d_ma : Tensor (N, C_in, H, W)
        d_Sa : Tensor (N, C_in, H, W)
        """
        N, C_in, H, W = self.input_shape
        k = self.kH

        # triton (K, C_out) → PyTorch (C_in, C_out, k, k)
        # conv2d weight convention: (out_channels, in_channels, kH, kW)
        # Here out=C_in, in=C_out — so mw_4d (C_in, C_out, k, k) is correct as-is
        mw_4d = self.mw.view(C_in, k, k, self.C_out).permute(0, 3, 1, 2).contiguous()

        # ── Delta propagation (transpose of the forward conv) ──
        delta_ma = F.conv2d(delta_mz, mw_4d, stride=self.stride, padding=self.padding)
        delta_Sa = F.conv2d(delta_Sz, mw_4d**2, stride=self.stride, padding=self.padding)

        # ── Weight gradient ──
        # grad_mw[c_in, c_out, ki, kj]
        #   = sum_{n,h,w} ma[n,c_in,h,w] * delta_mz[n,c_out, h*s+ki-p, w*s+kj-p]
        # Efficient: unfold delta_mz to extract kernel patches, then batch-matmul with ma_flat
        L = H * W
        ma_flat = self.ma_in.view(N, C_in, L)                                       # (N, C_in, L)
        gy_unf = F.unfold(delta_mz, k, stride=self.stride, padding=self.padding)    # (N, C_out*k*k, L)
        # (N, C_in, L) @ (N, L, C_out*k*k) → (N, C_in, C_out*k*k) → sum over N
        grad_mw = (ma_flat @ gy_unf.permute(0, 2, 1)).sum(0)                        # (C_in, C_out*k*k)
        # Reshape (C_in, C_out, k, k) then to triton layout (K, C_out)
        grad_mw = grad_mw.view(C_in, self.C_out, k, k).permute(0, 2, 3, 1).reshape(-1, self.C_out)

        gy2_unf = F.unfold(delta_Sz, k, stride=self.stride, padding=self.padding)
        ma2_flat = (self.ma_in**2).view(N, C_in, L)
        grad_Sw = (ma2_flat @ gy2_unf.permute(0, 2, 1)).sum(0)
        grad_Sw = grad_Sw.view(C_in, self.C_out, k, k).permute(0, 2, 3, 1).reshape(-1, self.C_out)

        self.delta_mw = self.Sw * grad_mw
        self.delta_Sw = self.Sw**2 * grad_Sw

        # ── Bias gradient ──
        grad_mb = delta_mz.sum(dim=(0, 2, 3)).view(1, self.C_out)
        grad_Sb = delta_Sz.sum(dim=(0, 2, 3)).view(1, self.C_out)
        self.delta_mb = self.Sb * grad_mb
        self.delta_Sb = self.Sb**2 * grad_Sb

        return delta_ma, delta_Sa

    # ------------------------------------------------------------------
    #  Update (apply capped deltas — called by the network)
    # ------------------------------------------------------------------
    def update(self, cap_factor: float) -> None:
        """Apply stored parameter deltas with cuTAGI-style capping."""
        update_parameters(self.mw, self.Sw, self.delta_mw, self.delta_Sw, cap_factor)
        update_parameters(self.mb, self.Sb, self.delta_mb, self.delta_Sb, cap_factor)

    @property
    def num_parameters(self) -> int:
        return 2 * (self.mw.numel() + self.mb.numel())

    def __repr__(self):
        return (
            f"ConvTranspose2D({self.C_in}, {self.C_out}, kernel={self.kH}, "
            f"stride={self.stride}, pad={self.padding})"
        )
