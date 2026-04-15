"""
Flatten layer — reshapes spatial tensors for fully-connected layers.

Forward:  (N, C, H, W) → (N, C·H·W)
Backward: (N, C·H·W) → (N, C, H, W)

Stores the pre-flatten shape so backward can restore it.
"""

from __future__ import annotations

from torch import Tensor

from ..base import Layer


class Flatten(Layer):
    """
    Flatten spatial dimensions for transition from conv → linear layers.
    """

    def __init__(self) -> None:
        self.shape = None

    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        ma : Tensor (N, C, H, W)
        Sa : Tensor (N, C, H, W)

        Returns
        -------
        ma : Tensor (N, C·H·W)
        Sa : Tensor (N, C·H·W)
        """
        self.shape = ma.shape
        N = ma.shape[0]
        return ma.view(N, -1), Sa.view(N, -1)

    def backward(self, delta_m: Tensor, delta_S: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        delta_m : Tensor (N, C·H·W)
        delta_S : Tensor (N, C·H·W)

        Returns
        -------
        delta_m : Tensor (N, C, H, W)
        delta_S : Tensor (N, C, H, W)
        """
        return delta_m.view(self.shape), delta_S.view(self.shape)

    def __repr__(self):
        return "Flatten()"
