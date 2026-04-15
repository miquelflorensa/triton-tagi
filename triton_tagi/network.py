"""
Network builder — a Sequential container for TAGI layers.

Supports both MLP and CNN architectures:

    # MLP
    net = Sequential([
        Linear(784, 256), ReLU(),
        Linear(256, 10),  Remax(),
    ])

    # CNN
    net = Sequential([
        Conv2D(1, 32, 5, padding=2), ReLU(), AvgPool2D(2),
        Conv2D(32, 64, 5, padding=2), ReLU(), AvgPool2D(2),
        Flatten(),
        Linear(3136, 256), ReLU(),
        Linear(256, 10),   Remax(),
    ])

The step() method follows cuTAGI's architecture:
    1. Forward pass — propagate moments
    2. Compute output innovation
    3. Backward pass — compute and store deltas on each layer (NO update)
    4. Update — apply capped deltas to all learnable layers
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Layer, LearnableLayer
from .layers.frn_resblock import FRNResBlock
from .layers.resblock import ResBlock
from .layers.shared_var_resblock import SharedVarResBlock
from .update.observation import compute_innovation
from .update.parameters import get_cap_factor


class Sequential:
    """
    Sequential container for TAGI Bayesian neural networks.

    Parameters
    ----------
    layers : list of layer objects
    device : str or torch.device  (default "cuda")
    """

    def __init__(self, layers: list, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.layers = layers

        # Move learnable layers to the target device
        for layer in self.layers:
            if isinstance(layer, (ResBlock, SharedVarResBlock, FRNResBlock)):
                # These blocks manage their own sub-layers
                layer.device = self.device
                for sub in layer._learnable:
                    self._move_layer_to_device(sub)
            elif isinstance(layer, LearnableLayer):
                self._move_layer_to_device(layer)
            # Move BatchNorm running stats
            if hasattr(layer, "running_mean"):
                layer.running_mean = layer.running_mean.to(self.device)
                layer.running_var = layer.running_var.to(self.device)

    def _move_layer_to_device(self, layer):
        """Move a single layer's parameters to self.device."""
        if not hasattr(layer, "mw"):
            return
        layer.device = self.device
        layer.mw = layer.mw.to(self.device)
        if hasattr(layer, "Sw"):
            layer.Sw = layer.Sw.to(self.device)
        if hasattr(layer, "mb"):
            layer.mb = layer.mb.to(self.device)
            if hasattr(layer, "Sb"):
                layer.Sb = layer.Sb.to(self.device)
        if hasattr(layer, "running_mean"):
            layer.running_mean = layer.running_mean.to(self.device)
            layer.running_var = layer.running_var.to(self.device)

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass through the entire network.

        Parameters
        ----------
        x : Tensor  input data (flat or spatial)

        Returns
        -------
        mu  : Tensor  predicted output means
        var : Tensor  predicted output variances
        """
        ma = x
        Sa = torch.zeros_like(x)

        for layer in self.layers:
            if isinstance(layer, Layer):
                ma, Sa = layer.forward(ma, Sa)
            else:
                raise TypeError(f"Unknown layer type: {type(layer)}")

        return ma, Sa

    # ------------------------------------------------------------------
    #  Single training step (cuTAGI-style: backward + capped update)
    # ------------------------------------------------------------------
    def step(self, x_batch: Tensor, y_batch: Tensor, sigma_v: float) -> tuple[Tensor, Tensor]:
        """
        Perform one forward + backward + capped-update TAGI step.

        Parameters
        ----------
        x_batch : Tensor  input mini-batch
        y_batch : Tensor  target mini-batch
        sigma_v : float   observation noise std

        Returns
        -------
        y_pred_mu  : Tensor  predicted means (before update)
        y_pred_var : Tensor  predicted variances (before update)
        """
        batch_size = x_batch.shape[0]

        # ── 1. Forward ──
        y_pred_mu, y_pred_var = self.forward(x_batch)

        # ── 2. Output innovation ──
        delta_mu, delta_var = compute_innovation(y_batch, y_pred_mu, y_pred_var, sigma_v)

        # ── 3. Backward (compute + store deltas, NO param update) ──
        for layer in reversed(self.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)

        # ── 4. Capped parameter update (cuTAGI-style) ──
        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                layer.update(cap_factor)

        return y_pred_mu, y_pred_var

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Set all layers to training mode (affects BatchNorm, etc.)."""
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.train()

    def eval(self) -> None:
        """Set all layers to evaluation mode (affects BatchNorm, etc.)."""
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.eval()

    def __repr__(self):
        lines = ["Sequential("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {layer}")
        lines.append(")")
        return "\n".join(lines)

    def num_parameters(self) -> int:
        """Return total number of learnable scalars (means + variances)."""
        return sum(
            layer.num_parameters for layer in self.layers if isinstance(layer, LearnableLayer)
        )
