"""
ResNet-18 for TAGI — Bayesian ResNet-18 adapted for CIFAR-10 (32×32).

Architecture (CIFAR-10 variant — no 7×7 stem / max-pool):
    Stem:    Conv(3→64, 3×3, s=1, p=1) → BN → ReLU          → 32×32
    Layer1:  BasicBlock(64→64,   s=1) × 2                     → 32×32
    Layer2:  BasicBlock(64→128,  s=2) + BasicBlock(128, s=1)  → 16×16
    Layer3:  BasicBlock(128→256, s=2) + BasicBlock(256, s=1)  →  8×8
    Layer4:  BasicBlock(256→512, s=2) + BasicBlock(512, s=1)  →  4×4
    Head:    AvgPool(4) → Flatten → FC(512→num_classes)

Total: ~11.2M parameters (means + variances).

Each BasicBlock contains:
    Conv3×3 → BN → ReLU → Conv3×3 → BN → (+skip) → ReLU

Uses the cuTAGI-style training loop:
    1. Forward (propagate moments)
    2. Compute output innovation
    3. Backward (compute & store deltas — NO parameter update)
    4. Capped parameter update
"""

import torch

from .layers.conv2d import Conv2D
from .layers.batchnorm2d import BatchNorm2D
from .layers.relu import ReLU
from .layers.resblock import ResBlock
from .layers.avgpool2d import AvgPool2D
from .layers.flatten import Flatten
from .layers.linear import Linear
from .update.observation import compute_innovation
from .update.parameters import get_cap_factor


class ResNet18:
    """
    Bayesian ResNet-18 for CIFAR-10 using TAGI.

    Parameters
    ----------
    num_classes : int   number of output classes (default 10)
    device      : str or torch.device
    """

    def __init__(self, num_classes=10, device="cuda"):
        self.device = torch.device(device)
        self.num_classes = num_classes

        channels = [64, 64, 128, 256, 512]
        gain_w = 0.1
        gain_b = 0.1

        # ── Stem ──
        self.stem_conv = Conv2D(3, channels[0], 3, stride=1, padding=1,
                                device=device, gain_w=gain_w, gain_b=gain_b)
        self.stem_bn   = BatchNorm2D(channels[0], device=device)
        self.stem_relu = ReLU()

        # ── Residual layers ──
        # Layer 1: 64 → 64, spatial 32×32
        self.layer1 = [
            ResBlock(channels[0], channels[1], stride=1, device=device, gain_w=gain_w, gain_b=gain_b),
            ResBlock(channels[1], channels[1], stride=1, device=device, gain_w=gain_w, gain_b=gain_b),
        ]
        # Layer 2: 64 → 128, spatial 32→16
        self.layer2 = [
            ResBlock(channels[1], channels[2], stride=2, device=device, gain_w=gain_w, gain_b=gain_b),
            ResBlock(channels[2], channels[2], stride=1, device=device, gain_w=gain_w, gain_b=gain_b),
        ]
        # Layer 3: 128 → 256, spatial 16→8
        self.layer3 = [
            ResBlock(channels[2], channels[3], stride=2, device=device, gain_w=gain_w, gain_b=gain_b),
            ResBlock(channels[3], channels[3], stride=1, device=device, gain_w=gain_w, gain_b=gain_b),
        ]
        # Layer 4: 256 → 512, spatial 8→4
        self.layer4 = [
            ResBlock(channels[3], channels[4], stride=2, device=device, gain_w=gain_w, gain_b=gain_b),
            ResBlock(channels[4], channels[4], stride=1, device=device, gain_w=gain_w, gain_b=gain_b),
        ]

        self.all_blocks = self.layer1 + self.layer2 + self.layer3 + self.layer4

        # ── Head ──
        self.avgpool = AvgPool2D(4)  # 4×4 → 1×1
        self.flatten = Flatten()
        self.fc = Linear(channels[4], num_classes, device=device, gain_mean=gain_w, gain_var=gain_b)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (N, 3, 32, 32)  input images

        Returns
        -------
        mu  : Tensor (N, num_classes)  predicted output means
        var : Tensor (N, num_classes)  predicted output variances
        """
        ma = x
        Sa = torch.zeros_like(x)

        # ── Stem ──
        ma, Sa = self.stem_conv.forward(ma, Sa)
        ma, Sa = self.stem_bn.forward(ma, Sa)
        ma, Sa = self.stem_relu.forward(ma, Sa)

        # ── Residual blocks ──
        for block in self.all_blocks:
            ma, Sa = block.forward(ma, Sa)

        # ── Head ──
        ma, Sa = self.avgpool.forward(ma, Sa)
        ma, Sa = self.flatten.forward(ma, Sa)
        ma, Sa = self.fc.forward(ma, Sa)

        return ma, Sa

    # ------------------------------------------------------------------
    #  Training step
    # ------------------------------------------------------------------
    def step(self, x_batch, y_batch, sigma_v):
        """
        Perform one forward + backward + capped-update TAGI step.

        Parameters
        ----------
        x_batch : Tensor  input mini-batch  (N, 3, 32, 32)
        y_batch : Tensor  target mini-batch (N, num_classes)
        sigma_v : float   observation noise std

        Returns
        -------
        y_pred_mu  : Tensor  predicted means  (before update)
        y_pred_var : Tensor  predicted variances (before update)
        """
        batch_size = x_batch.shape[0]

        # ── 1. Forward ──
        y_pred_mu, y_pred_var = self.forward(x_batch)

        # ── 2. Output innovation ──
        delta_mu, delta_var = compute_innovation(
            y_batch, y_pred_mu, y_pred_var, sigma_v
        )

        # ── 3. Backward ──
        # FC
        delta_mu, delta_var = self.fc.backward(delta_mu, delta_var)
        # Flatten
        delta_mu, delta_var = self.flatten.backward(delta_mu, delta_var)
        # AvgPool
        delta_mu, delta_var = self.avgpool.backward(delta_mu, delta_var)

        # Residual blocks (reversed)
        for block in reversed(self.all_blocks):
            delta_mu, delta_var = block.backward(delta_mu, delta_var)

        # Stem (reversed)
        delta_mu, delta_var = self.stem_relu.backward(delta_mu, delta_var)
        delta_mu, delta_var = self.stem_bn.backward(delta_mu, delta_var)
        self.stem_conv.backward(delta_mu, delta_var)

        # ── 4. Capped parameter update ──
        cap_factor = get_cap_factor(batch_size)
        self._update_all(cap_factor)

        return y_pred_mu, y_pred_var

    # ------------------------------------------------------------------
    #  Parameter update
    # ------------------------------------------------------------------
    def _update_all(self, cap_factor):
        """Apply capped updates to all learnable parameters."""
        self.stem_conv.update(cap_factor)
        self.stem_bn.update(cap_factor)
        for block in self.all_blocks:
            block.update(cap_factor)
        self.fc.update(cap_factor)

    # ------------------------------------------------------------------
    #  Train / Eval mode
    # ------------------------------------------------------------------
    def train(self):
        """Set all layers to training mode."""
        self.stem_bn.train()
        for block in self.all_blocks:
            block.train()

    def eval(self):
        """Set all layers to evaluation mode."""
        self.stem_bn.eval()
        for block in self.all_blocks:
            block.eval()

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    def num_parameters(self):
        """Return total number of learnable scalars (means + variances)."""
        total = 0

        # Stem
        total += self.stem_conv.mw.numel() + self.stem_conv.mb.numel()
        total += self.stem_bn.mw.numel() + self.stem_bn.mb.numel()

        # Blocks
        for block in self.all_blocks:
            total += block.conv1.mw.numel() + block.conv1.mb.numel()
            total += block.bn1.mw.numel() + block.bn1.mb.numel()
            total += block.conv2.mw.numel() + block.conv2.mb.numel()
            total += block.bn2.mw.numel() + block.bn2.mb.numel()
            if block.has_proj:
                total += block.proj_conv.mw.numel() + block.proj_conv.mb.numel()
                total += block.proj_bn.mw.numel() + block.proj_bn.mb.numel()

        # FC
        total += self.fc.mw.numel() + self.fc.mb.numel()

        return total * 2  # means + variances

    def __repr__(self):
        lines = [
            "ResNet18(",
            f"  Stem: Conv2D(3→64, 3×3) → BN(64) → ReLU",
        ]
        layer_names = ["Layer1", "Layer2", "Layer3", "Layer4"]
        for name, blocks in zip(layer_names,
                                [self.layer1, self.layer2,
                                 self.layer3, self.layer4]):
            for i, blk in enumerate(blocks):
                lines.append(f"  {name}[{i}]: {blk}")
        lines.append(f"  Head: AvgPool(4) → Flatten → Linear(512→{self.num_classes})")
        lines.append(")")
        return "\n".join(lines)
