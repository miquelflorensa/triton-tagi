"""
Bayesian Residual Block (BasicBlock) for TAGI.

Implements the standard ResNet BasicBlock with skip connections:

    Main path:  Conv3×3 → BN → ReLU → Conv3×3 → BN
    Skip path:  identity  (or Conv1×1 → BN if dims change)
    Output:     μ = μ_main + μ_skip,  S = S_main + S_skip  →  ReLU

The sum of independent Gaussians gives:
    μ_out = μ_main + μ_skip
    S_out = S_main + S_skip

The *external* ReLU after the residual add is included inside the block
to keep the API clean (forward returns post-activation moments).
"""

import torch

from .conv2d import Conv2D
from .batchnorm2d import BatchNorm2D
from .relu import ReLU


class ResBlock:
    """
    TAGI BasicBlock with residual skip connection.

    Parameters
    ----------
    C_in    : int   input channels
    C_out   : int   output channels
    stride  : int   stride for the first conv (1 or 2)
    device  : str or torch.device
    """

    def __init__(self, C_in, C_out, stride=1, device="cuda", gain_w=1.0, gain_b=1.0):
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.device = torch.device(device)

        # ── Main path ──
        self.conv1 = Conv2D(C_in, C_out, 3, stride=stride, padding=1, device=device, gain_w=gain_w, gain_b=gain_b)
        self.bn1   = BatchNorm2D(C_out, device=device)
        self.relu1 = ReLU()

        self.conv2 = Conv2D(C_out, C_out, 3, stride=1, padding=1, device=device, gain_w=gain_w, gain_b=gain_b)
        self.bn2   = BatchNorm2D(C_out, device=device)

        # ── Skip path (projection if dimensions change) ──
        self.has_proj = (stride != 1 or C_in != C_out)
        if self.has_proj:
            self.proj_conv = Conv2D(C_in, C_out, 1, stride=stride, padding=0,
                                    device=device, gain_w=gain_w, gain_b=gain_b)
            self.proj_bn = BatchNorm2D(C_out, device=device)
        else:
            self.proj_conv = None
            self.proj_bn = None

        # ── Post-add ReLU ──
        self.relu_out = ReLU()

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, ma, Sa):
        """
        Parameters
        ----------
        ma : Tensor (N, C_in, H, W)   activation means
        Sa : Tensor (N, C_in, H, W)   activation variances

        Returns
        -------
        ma_out : Tensor (N, C_out, H', W')  post-ReLU means
        Sa_out : Tensor (N, C_out, H', W')  post-ReLU variances
        """
        # ── Main path ──
        m1, S1 = self.conv1.forward(ma, Sa)
        m1, S1 = self.bn1.forward(m1, S1)
        m1, S1 = self.relu1.forward(m1, S1)

        m2, S2 = self.conv2.forward(m1, S1)
        m2, S2 = self.bn2.forward(m2, S2)

        # ── Skip path ──
        if self.has_proj:
            ms, Ss = self.proj_conv.forward(ma, Sa)
            ms, Ss = self.proj_bn.forward(ms, Ss)
        else:
            ms, Ss = ma, Sa

        # ── Residual addition (sum of independent Gaussians) ──
        m_add = m2 + ms
        S_add = S2 + Ss

        # ── Post-add ReLU ──
        ma_out, Sa_out = self.relu_out.forward(m_add, S_add)

        return ma_out, Sa_out

    # ------------------------------------------------------------------
    #  Backward
    # ------------------------------------------------------------------
    def backward(self, delta_ma, delta_Sa):
        """
        Parameters
        ----------
        delta_ma : Tensor (N, C_out, H', W')  mean delta
        delta_Sa : Tensor (N, C_out, H', W')  variance delta

        Returns
        -------
        delta_m_in : Tensor (N, C_in, H, W)   mean delta to propagate
        delta_S_in : Tensor (N, C_in, H, W)   variance delta to propagate
        """
        # ── Post-add ReLU backward ──
        dm, dS = self.relu_out.backward(delta_ma, delta_Sa)

        # ── Main path backward ──
        # The add splits deltas equally to both branches
        dm_main, dS_main = dm, dS

        # BN2 backward
        dm_main, dS_main = self.bn2.backward(dm_main, dS_main)
        # Conv2 backward
        dm_main, dS_main = self.conv2.backward(dm_main, dS_main)
        # ReLU1 backward
        dm_main, dS_main = self.relu1.backward(dm_main, dS_main)
        # BN1 backward
        dm_main, dS_main = self.bn1.backward(dm_main, dS_main)
        # Conv1 backward
        dm_main, dS_main = self.conv1.backward(dm_main, dS_main)

        # ── Skip path backward ──
        dm_skip, dS_skip = dm, dS

        if self.has_proj:
            dm_skip, dS_skip = self.proj_bn.backward(dm_skip, dS_skip)
            dm_skip, dS_skip = self.proj_conv.backward(dm_skip, dS_skip)

        # ── Combine (sum of deltas from both paths) ──
        return dm_main + dm_skip, dS_main + dS_skip

    # ------------------------------------------------------------------
    #  Update
    # ------------------------------------------------------------------
    def update(self, cap_factor):
        """Apply capped parameter updates to all learnable sub-layers."""
        self.conv1.update(cap_factor)
        self.bn1.update(cap_factor)
        self.conv2.update(cap_factor)
        self.bn2.update(cap_factor)
        if self.has_proj:
            self.proj_conv.update(cap_factor)
            self.proj_bn.update(cap_factor)

    # ------------------------------------------------------------------
    #  Train / Eval mode
    # ------------------------------------------------------------------
    def train(self):
        """Set all BatchNorm layers to training mode."""
        self.bn1.train()
        self.bn2.train()
        if self.has_proj and self.proj_bn is not None:
            self.proj_bn.train()

    def eval(self):
        """Set all BatchNorm layers to evaluation mode."""
        self.bn1.eval()
        self.bn2.eval()
        if self.has_proj and self.proj_bn is not None:
            self.proj_bn.eval()

    def __repr__(self):
        proj_str = " + proj" if self.has_proj else ""
        return (f"ResBlock({self.C_in}→{self.C_out}, "
                f"stride={self.stride}{proj_str})")
