"""
Shared-Variance Residual Block for TAGI.

Exact same architecture as the standard ResBlock, but all sub-layers
(Conv2D, BatchNorm2D) use shared scalar variances instead of per-parameter
variances.

Main path:   SharedVarConv2D → ReLU → SharedVarBatchNorm2D
           → SharedVarConv2D → ReLU → SharedVarBatchNorm2D

Shortcut:    Identity  OR  SharedVarConv2D(k=2,s=2) → ReLU → SharedVarBatchNorm2D
"""

from __future__ import annotations

from torch import Tensor

from ..base import LearnableLayer
from .relu import ReLU
from .resblock import triton_add_shortcut, triton_delta_merge
from .shared_var_batchnorm2d import SharedVarBatchNorm2D
from .shared_var_conv2d import SharedVarConv2D


class SharedVarResBlock(LearnableLayer):
    """
    TAGI Residual Block with shared-variance sub-layers.

    Architecture (identical to cuTAGI ResNetBlock):
    ─────────────────────────────────────────────────
    Main path:
        SharedVarConv2D(in_ch, out_ch, 3×3, stride, pad=1)
        → ReLU → SharedVarBatchNorm2D(out_ch)
        → SharedVarConv2D(out_ch, out_ch, 3×3, stride=1, pad=1)
        → ReLU → SharedVarBatchNorm2D(out_ch)

    Shortcut (projection, when stride>1 or ch mismatch):
        SharedVarConv2D(in_ch, out_ch, 2×2, stride=2, pad=0)
        → ReLU → SharedVarBatchNorm2D(out_ch)

    Merge: output = main_output + shortcut_output (no post-activation)

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    stride       : int
    device       : str
    gain_w, gain_b : float
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

        # ── Main path ──
        self.conv1 = SharedVarConv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            device=device,
            gain_w=gain_w,
            gain_b=gain_b,
        )
        self.relu1 = ReLU()
        self.bn1 = SharedVarBatchNorm2D(
            out_channels, device=device, gain_w=gain_w, gain_b=gain_b, preserve_var=False
        )

        self.conv2 = SharedVarConv2D(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            device=device,
            gain_w=gain_w,
            gain_b=gain_b,
        )
        self.relu2 = ReLU()
        self.bn2 = SharedVarBatchNorm2D(
            out_channels, device=device, gain_w=gain_w, gain_b=gain_b, preserve_var=False
        )

        self._main_layers = [self.conv1, self.relu1, self.bn1, self.conv2, self.relu2, self.bn2]

        # ── Shortcut path ──
        self.use_projection = (stride != 1) or (in_channels != out_channels)
        if self.use_projection:
            self.proj_conv = SharedVarConv2D(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=stride,
                padding=0,
                device=device,
                gain_w=gain_w,
                gain_b=gain_b,
            )
            self.proj_relu = ReLU()
            self.proj_bn = SharedVarBatchNorm2D(
                out_channels, device=device, gain_w=gain_w, gain_b=gain_b, preserve_var=False
            )
            self._proj_layers = [self.proj_conv, self.proj_relu, self.proj_bn]
        else:
            self.proj_conv = None
            self.proj_relu = None
            self.proj_bn = None
            self._proj_layers = []

        # ── All learnable sub-layers ──
        self._learnable = [self.conv1, self.bn1, self.conv2, self.bn2]
        if self.use_projection:
            self._learnable.extend([self.proj_conv, self.proj_bn])

    # ------------------------------------------------------------------
    #  Train / Eval
    # ------------------------------------------------------------------
    def train(self) -> None:
        self.training = True
        for layer in self._learnable:
            if hasattr(layer, "train"):
                layer.train()

    def eval(self) -> None:
        self.training = False
        for layer in self._learnable:
            if hasattr(layer, "eval"):
                layer.eval()

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, mu_in: Tensor, var_in: Tensor) -> tuple[Tensor, Tensor]:
        mu_skip = mu_in.clone()
        var_skip = var_in.clone()

        # Main path
        mu_z, var_z = mu_in, var_in
        for layer in self._main_layers:
            mu_z, var_z = layer.forward(mu_z, var_z)

        # Shortcut path
        if self.use_projection:
            mu_x, var_x = mu_skip, var_skip
            for layer in self._proj_layers:
                mu_x, var_x = layer.forward(mu_x, var_x)
        else:
            mu_x, var_x = mu_skip, var_skip

        # Merge
        triton_add_shortcut(mu_x, var_x, mu_z, var_z)
        return mu_z, var_z

    # ------------------------------------------------------------------
    #  Backward
    # ------------------------------------------------------------------
    def backward(self, delta_mu: Tensor, delta_var: Tensor) -> tuple[Tensor, Tensor]:
        d_mu_main = delta_mu.clone()
        d_var_main = delta_var.clone()
        d_mu_skip = delta_mu.clone()
        d_var_skip = delta_var.clone()

        # Main path backward
        for layer in reversed(self._main_layers):
            d_mu_main, d_var_main = layer.backward(d_mu_main, d_var_main)

        # Shortcut backward
        if self.use_projection:
            for layer in reversed(self._proj_layers):
                d_mu_skip, d_var_skip = layer.backward(d_mu_skip, d_var_skip)

        # Delta merge
        triton_delta_merge(d_mu_skip, d_var_skip, d_mu_main, d_var_main)
        return d_mu_main, d_var_main

    # ------------------------------------------------------------------
    #  Update
    # ------------------------------------------------------------------
    def update(self, cap_factor: float) -> None:
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

    def count_variance_params(self):
        """Count the number of scalar variance parameters in this block."""
        count = 0
        for layer in self._learnable:
            if hasattr(layer, "Sw"):
                count += 1  # Sw
            if hasattr(layer, "Sb"):
                count += 1  # Sb
        return count

    def __repr__(self):
        proj = "projection" if self.use_projection else "identity"
        return (
            f"SharedVarResBlock({self.in_channels}→{self.out_channels}, "
            f"stride={self.stride}, skip={proj})"
        )
