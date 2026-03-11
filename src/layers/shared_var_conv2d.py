"""
Shared-Variance Conv2D layer for TAGI.

Same im2col-based approach as the standard Conv2D, but with scalar sw/sb
instead of per-parameter Sw (K, C_out) and Sb (1, C_out).

Forward variance (after im2col flattening):
    Sz = sw · row_sum(patches_ma²) + patches_Sa @ mw² + sw · row_sum(patches_Sa) + sb
"""

import torch
import triton
import triton.language as tl

from ..kernels.common import triton_fused_backward_delta
from ..update.shared_var_parameters import update_shared_variance, update_shared_mean
from ..param_init import init_weight_bias_conv2d
from .conv2d import _triton_im2col, _triton_col2im


class SharedVarConv2D:
    """
    Bayesian Conv2D layer with shared (scalar) variance per layer.

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    kernel_size  : int
    stride       : int
    padding      : int
    device       : str
    gain_w       : float  gain multiplier for weight variance
    gain_b       : float  gain multiplier for bias variance
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, device="cuda",
                 gain_w=1.0, gain_b=1.0):
        self.C_in = in_channels
        self.C_out = out_channels
        self.kH = self.kW = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = torch.device(device)

        # Get standard init
        mw, Sw, mb, Sb = init_weight_bias_conv2d(
            kernel_size, in_channels, out_channels,
            init_method="He",
            gain_w=gain_w, gain_b=gain_b,
            device=self.device,
        )

        # Means: same as standard
        self.mw = mw     # (K, C_out)  where K = C_in * kH * kW
        self.mb = mb     # (1, C_out)

        # Shared scalar variances
        self.sw = torch.tensor(Sw[0, 0].item(), device=self.device,
                               dtype=torch.float32)
        self.sb = torch.tensor(Sb[0, 0].item(), device=self.device,
                               dtype=torch.float32)

        self.has_bias = True

        # Cached for backward
        self.patches_ma = None
        self.input_shape = None
        self.spatial = None

        # Parameter deltas
        self.delta_mw = None
        self.delta_mb = None

    # ------------------------------------------------------------------
    #  Properties for Sequential / checkpoint compatibility
    # ------------------------------------------------------------------
    @property
    def Sw(self):
        K = self.C_in * self.kH * self.kW
        return self.sw.expand(K, self.C_out)

    @property
    def Sb(self):
        return self.sb.expand(1, self.C_out)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, ma, Sa):
        """
        Parameters
        ----------
        ma : Tensor (N, C_in, H, W)
        Sa : Tensor (N, C_in, H, W)

        Returns
        -------
        mz : Tensor (N, C_out, H_out, W_out)
        Sz : Tensor (N, C_out, H_out, W_out)
        """
        N, C, H, W = ma.shape
        self.input_shape = (N, C, H, W)
        H_out = (H + 2 * self.padding - self.kH) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kW) // self.stride + 1
        self.spatial = (H_out, W_out)

        # im2col: (N, C_in, H, W) → (N·L, K)
        patches_ma = _triton_im2col(ma, self.kH, self.kW, self.stride,
                                    self.padding)
        patches_Sa = _triton_im2col(Sa, self.kH, self.kW, self.stride,
                                    self.padding)
        self.patches_ma = patches_ma

        # Mean: standard matmul + bias
        mz_flat = torch.matmul(patches_ma, self.mw) + self.mb   # (NL, C_out)

        # Variance with shared scalars:
        sw = self.sw.item()
        sb = self.sb.item()
        Sa_mw2 = torch.matmul(patches_Sa, self.mw * self.mw)    # (NL, C_out)
        row_ma2 = (patches_ma * patches_ma).sum(dim=1, keepdim=True)  # (NL, 1)
        row_Sa  = patches_Sa.sum(dim=1, keepdim=True)                 # (NL, 1)
        Sz_flat = Sa_mw2 + sw * (row_ma2 + row_Sa) + sb              # (NL, C_out)

        # Reshape (N·L, C_out) → (N, C_out, H_out, W_out)
        mz = mz_flat.view(N, H_out, W_out, self.C_out) \
                     .permute(0, 3, 1, 2).contiguous()
        Sz = Sz_flat.view(N, H_out, W_out, self.C_out) \
                     .permute(0, 3, 1, 2).contiguous()
        return mz, Sz

    # ------------------------------------------------------------------
    #  Backward (compute deltas only — NO parameter update)
    # ------------------------------------------------------------------
    def backward(self, delta_mz, delta_Sz):
        """
        Compute parameter deltas and propagate to the previous layer.

        Parameters
        ----------
        delta_mz : Tensor (N, C_out, H_out, W_out)
        delta_Sz : Tensor (N, C_out, H_out, W_out)

        Returns
        -------
        d_ma : Tensor (N, C_in, H, W)
        d_Sa : Tensor (N, C_in, H, W)
        """
        N = delta_mz.shape[0]

        # Flatten (N, C_out, H, W) → (N·L, C_out)
        dmz = delta_mz.permute(0, 2, 3, 1).reshape(-1, self.C_out).contiguous()
        dSz = delta_Sz.permute(0, 2, 3, 1).reshape(-1, self.C_out).contiguous()

        # ── Raw gradients ──
        grad_mw = torch.matmul(self.patches_ma.T, dmz)             # (K, C_out)
        grad_mb = dmz.sum(0, keepdim=True)                         # (1, C_out)
        grad_Sw = torch.matmul((self.patches_ma ** 2).T, dSz)      # (K, C_out)
        grad_Sb = dSz.sum(0, keepdim=True)                         # (1, C_out)

        # ── Parameter deltas (scalar sw broadcast) ──
        self.delta_mw = self.sw.item() * grad_mw
        self.delta_mb = self.sb.item() * grad_mb

        # Store variance grads for update()
        self._grad_Sw = grad_Sw
        self._grad_Sb = grad_Sb

        # ── Delta propagation: Triton fused ──
        dp_ma, dp_Sa = triton_fused_backward_delta(dmz, dSz, self.mw)

        # ── col2im: (N·L, K) → (N, C_in, H, W) ──
        _, C, H, W = self.input_shape
        d_ma = _triton_col2im(dp_ma, N, C, H, W,
                              self.kH, self.kW, self.stride, self.padding)
        d_Sa = _triton_col2im(dp_Sa, N, C, H, W,
                              self.kH, self.kW, self.stride, self.padding)
        return d_ma, d_Sa

    # ------------------------------------------------------------------
    #  Update (apply capped deltas — called by the network)
    # ------------------------------------------------------------------
    def update(self, cap_factor):
        """Apply stored parameter deltas with capping + scalar variance update."""
        update_shared_mean(self.mw, self.delta_mw, self.sw.item(), cap_factor)
        update_shared_mean(self.mb, self.delta_mb, self.sb.item(), cap_factor)

        update_shared_variance(self.sw, self._grad_Sw)
        update_shared_variance(self.sb, self._grad_Sb)

    def __repr__(self):
        return (f"SharedVarConv2D({self.C_in}, {self.C_out}, kernel={self.kH}, "
                f"stride={self.stride}, pad={self.padding})")
