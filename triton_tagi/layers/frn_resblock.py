"""
TAGI-compatible Residual Block using FRN + TLU instead of ReLU + BatchNorm.

Replaces all BatchNorm2D → FRN2D and ReLU → TLU following the paper:
"Filter Response Normalization Layer: Eliminating Batch Dependence in the
Training of Deep Neural Networks" (Singh & Davis, 2020).

Architecture:
    Main path:      Conv2D(3×3, stride) → FRN2D → TLU → Conv2D(3×3) → FRN2D → TLU
    Shortcut path:  Conv2D(2×2, stride=2) → FRN2D → TLU   (if projection)
    Merge:          element-wise addition of moments (no post-activation)
"""

from __future__ import annotations

from torch import Tensor

from ..base import LearnableLayer
from .conv2d import Conv2D
from .frn import FRN2D
from .resblock import triton_add_shortcut, triton_delta_merge
from .tlu import TLU


class FRNResBlock(LearnableLayer):
    """
    TAGI Residual Block with FRN + TLU (no batch dependence).

    Architecture:
        Main path:  Conv2D(3×3) → FRN → TLU → Conv2D(3×3) → FRN → TLU
        Shortcut:   Identity  OR  Conv2D(2×2, stride=2) → FRN → TLU
        Merge:      element-wise addition (no post-activation)

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    stride       : int  (default 1)
    device       : str  (default "cuda")
    gain_w, gain_b : float  (default 1.0)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        device: str = "cuda",
        gain_w: float = 1.0,
        gain_b: float = 1.0,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.device = device
        self.training = True

        # ── Main path: Conv→FRN→TLU→Conv→FRN→TLU ──
        self.conv1 = Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            device=device,
            gain_w=gain_w,
            gain_b=gain_b,
        )
        self.frn1 = FRN2D(out_channels, device=device, gain_w=gain_w, gain_b=gain_b)
        self.tlu1 = TLU(out_channels, device=device)

        self.conv2 = Conv2D(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            device=device,
            gain_w=gain_w,
            gain_b=gain_b,
        )
        self.frn2 = FRN2D(out_channels, device=device, gain_w=gain_w, gain_b=gain_b)
        self.tlu2 = TLU(out_channels, device=device)

        # Ordered sub-layer list for main path
        self._main_layers = [self.conv1, self.frn1, self.tlu1, self.conv2, self.frn2, self.tlu2]

        # ── Shortcut path ──
        self.use_projection = (stride != 1) or (in_channels != out_channels)
        if self.use_projection:
            self.proj_conv = Conv2D(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=stride,
                padding=0,
                device=device,
                gain_w=gain_w,
                gain_b=gain_b,
            )
            self.proj_frn = FRN2D(out_channels, device=device, gain_w=gain_w, gain_b=gain_b)
            self.proj_tlu = TLU(out_channels, device=device)
            self._proj_layers = [self.proj_conv, self.proj_frn, self.proj_tlu]
        else:
            self.proj_conv = None
            self.proj_frn = None
            self.proj_tlu = None
            self._proj_layers = []

        # ── All learnable sub-layers ──
        self._learnable = [self.conv1, self.frn1, self.tlu1, self.conv2, self.frn2, self.tlu2]
        if self.use_projection:
            self._learnable.extend([self.proj_conv, self.proj_frn, self.proj_tlu])

    # ------------------------------------------------------------------
    #  Train / Eval (FRN has no batch stats, so these are mostly no-ops)
    # ------------------------------------------------------------------
    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, mu_in: Tensor, var_in: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through the residual block."""
        mu_skip = mu_in.clone()
        var_skip = var_in.clone()

        # ── Main path: Conv→FRN→TLU→Conv→FRN→TLU ──
        mu_z, var_z = mu_in, var_in
        for layer in self._main_layers:
            mu_z, var_z = layer.forward(mu_z, var_z)

        # ── Shortcut path ──
        if self.use_projection:
            mu_x, var_x = mu_skip, var_skip
            for layer in self._proj_layers:
                mu_x, var_x = layer.forward(mu_x, var_x)
        else:
            mu_x, var_x = mu_skip, var_skip

        # ── Merge ──
        triton_add_shortcut(mu_x, var_x, mu_z, var_z)
        return mu_z, var_z

    # ------------------------------------------------------------------
    #  Backward
    # ------------------------------------------------------------------
    def backward(self, delta_mu: Tensor, delta_var: Tensor) -> tuple[Tensor, Tensor]:
        """Backward pass through the residual block."""
        d_mu_main = delta_mu.clone()
        d_var_main = delta_var.clone()

        d_mu_skip = delta_mu.clone()
        d_var_skip = delta_var.clone()

        # ── Main path backward (reversed) ──
        for layer in reversed(self._main_layers):
            d_mu_main, d_var_main = layer.backward(d_mu_main, d_var_main)

        # ── Shortcut path backward ──
        if self.use_projection:
            for layer in reversed(self._proj_layers):
                d_mu_skip, d_var_skip = layer.backward(d_mu_skip, d_var_skip)

        # ── Delta merge ──
        triton_delta_merge(d_mu_skip, d_var_skip, d_mu_main, d_var_main)
        return d_mu_main, d_var_main

    # ------------------------------------------------------------------
    #  Update
    # ------------------------------------------------------------------
    def update(self, cap_factor: float) -> None:
        """Apply capped parameter updates to all learnable sub-layers."""
        for layer in self._learnable:
            layer.update(cap_factor)

    # ------------------------------------------------------------------
    #  Properties for Sequential compatibility
    # ------------------------------------------------------------------
    @property
    def mw(self):
        return self.conv1.mw

    @mw.setter
    def mw(self, value):
        self.conv1.mw = value

    @property
    def Sw(self):
        return self.conv1.Sw

    @property
    def mb(self):
        return self.conv1.mb

    @property
    def Sb(self):
        return self.conv1.Sb

    @property
    def num_parameters(self) -> int:
        """Total learnable scalars across all sub-layers (means + variances)."""
        return sum(layer.num_parameters for layer in self._learnable)

    def __repr__(self):
        proj = "projection" if self.use_projection else "identity"
        return (
            f"FRNResBlock({self.in_channels}→{self.out_channels}, "
            f"stride={self.stride}, skip={proj})"
        )
