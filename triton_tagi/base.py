"""
Abstract base classes for TAGI layers.

Every layer in triton-tagi inherits from one of two ABCs:

- ``Layer``         — stateless layers (activations, pooling, flatten)
- ``LearnableLayer`` — layers with trainable parameters (linear, conv, norm)

Using ABCs replaces the fragile ``_LEARNABLE_LAYERS`` tuple in network.py:
``Sequential`` now dispatches via ``isinstance(layer, LearnableLayer)``,
so adding a new layer type requires only inheriting from the correct base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class Layer(ABC):
    """Base class for all TAGI layers."""

    @abstractmethod
    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """Propagate Gaussian moments forward through the layer.

        Args:
            ma: Activation means.
            Sa: Activation variances (non-negative).

        Returns:
            Output means and variances.
        """
        ...

    @abstractmethod
    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        """Propagate innovation deltas backward through the layer.

        Args:
            delta_ma: Incoming delta on activation means.
            delta_Sa: Incoming delta on activation variances.

        Returns:
            Outgoing deltas for the previous layer.
        """
        ...


class LearnableLayer(Layer):
    """Base class for layers with trainable parameters (weights, biases, etc.)."""

    @abstractmethod
    def update(self, cap_factor: float) -> None:
        """Apply the capped parameter update using deltas stored during backward.

        Args:
            cap_factor: Capping factor computed from the batch size.
        """
        ...

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """Total number of learnable scalars (means + variances combined)."""
        ...
