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

import torch

from .layers.linear import Linear
from .layers.relu import ReLU
from .layers.leaky_relu import LeakyReLU
from .layers.remax import Remax
from .layers.bernoulli import Bernoulli
from .layers.conv2d import Conv2D
from .layers.avgpool2d import AvgPool2D
from .layers.batchnorm2d import BatchNorm2D
from .layers.resblock import ResBlock, Add
from .layers.flatten import Flatten
from .layers.even_softplus import EvenSoftplus
from .update.observation import compute_innovation
from .update.parameters import get_cap_factor

# Layers that have .forward() and .backward() but NO learnable parameters
_ACTIVATION_LAYERS = (ReLU, LeakyReLU, Remax, Bernoulli, AvgPool2D, Flatten,
                      EvenSoftplus)

# Layers that have learnable parameters and .update()
_LEARNABLE_LAYERS = (Linear, Conv2D, BatchNorm2D, ResBlock)

# All supported layers
_ALL_LAYERS = _ACTIVATION_LAYERS + _LEARNABLE_LAYERS


class Sequential:
    """
    Sequential container for TAGI Bayesian neural networks.

    Parameters
    ----------
    layers : list of layer objects
    device : str or torch.device  (default "cuda")
    """

    def __init__(self, layers, device="cuda"):
        self.device = torch.device(device)
        self.layers = layers

        # Move learnable layers to the target device
        for layer in self.layers:
            if isinstance(layer, ResBlock):
                # ResBlock handles its own sub-layers; just set device
                layer.device = self.device
                for sub in layer._learnable:
                    if hasattr(sub, 'mw'):
                        sub.device = self.device
                        sub.mw = sub.mw.to(self.device)
                        sub.Sw = sub.Sw.to(self.device)
                        sub.mb = sub.mb.to(self.device)
                        sub.Sb = sub.Sb.to(self.device)
                    if hasattr(sub, 'running_mean'):
                        sub.running_mean = sub.running_mean.to(self.device)
                        sub.running_var  = sub.running_var.to(self.device)
            elif hasattr(layer, 'mw'):
                layer.device = self.device
                layer.mw = layer.mw.to(self.device)
                layer.Sw = layer.Sw.to(self.device)
                layer.mb = layer.mb.to(self.device)
                layer.Sb = layer.Sb.to(self.device)
            # Move BatchNorm running stats
            if hasattr(layer, 'running_mean'):
                layer.running_mean = layer.running_mean.to(self.device)
                layer.running_var  = layer.running_var.to(self.device)

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def forward(self, x):
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
            if isinstance(layer, _ALL_LAYERS):
                ma, Sa = layer.forward(ma, Sa)
            else:
                raise TypeError(f"Unknown layer type: {type(layer)}")

        return ma, Sa

    # ------------------------------------------------------------------
    #  Single training step (cuTAGI-style: backward + capped update)
    # ------------------------------------------------------------------
    def step(self, x_batch, y_batch, sigma_v):
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
        delta_mu, delta_var = compute_innovation(
            y_batch, y_pred_mu, y_pred_var, sigma_v
        )

        # ── 3. Backward (compute + store deltas, NO param update) ──
        for layer in reversed(self.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)

        # ── 4. Capped parameter update (cuTAGI-style) ──
        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, _LEARNABLE_LAYERS):
                layer.update(cap_factor)

        return y_pred_mu, y_pred_var

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    def train(self):
        """Set all layers to training mode (affects BatchNorm, etc.)."""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.train()

    def eval(self):
        """Set all layers to evaluation mode (affects BatchNorm, etc.)."""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.eval()

    def __repr__(self):
        lines = ["Sequential("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {layer}")
        lines.append(")")
        return "\n".join(lines)

    def num_parameters(self):
        """Return total number of learnable scalars (means + variances)."""
        total = 0
        for layer in self.layers:
            if isinstance(layer, ResBlock):
                # ResBlock.num_sub_parameters() already returns means+variances
                total += layer.num_sub_parameters()
            elif isinstance(layer, _LEARNABLE_LAYERS):
                total += (layer.mw.numel() + layer.mb.numel()) * 2
        return total
