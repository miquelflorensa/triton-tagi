"""
Shared-Variance Linear (fully-connected) layer for TAGI.

Same as the standard Linear layer but replaces per-parameter variances
Sw (K, N) and Sb (1, N) with scalar Sw and Sb respectively.

Forward variance:
    Sz = Sw · row_sum(ma²)·1_N  +  Sa @ mw²  +  Sw · row_sum(Sa)·1_N  +  Sb

Backward:
    delta_mw = Sw · grad_mw        (scalar × full gradient)
    delta_mb = Sb · grad_mb
    Variance update: precision-space scalar update averaging all per-param grads.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..base import LearnableLayer
from ..kernels.common import triton_fused_backward_delta
from ..param_init import init_weight_bias_linear
from ..update.shared_var_parameters import update_shared_mean, update_shared_variance


class SharedVarLinear(LearnableLayer):
    """
    Bayesian fully-connected layer with shared (scalar) variance per layer.

    Parameters
    ----------
    in_features  : int
    out_features : int
    device       : str or torch.device
    init_method  : str   "He" or "Xavier"
    gain_w       : float  gain multiplier for weight variance
    gain_b       : float  gain multiplier for bias variance
    bias         : bool
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: str = "cuda",
        init_method: str = "He",
        gain_w: float = 1.0,
        gain_b: float = 1.0,
        bias: bool = True,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.device = torch.device(device)
        self.has_bias = bias

        # Get standard init (to match scale)
        mw, Sw, mb, Sb = init_weight_bias_linear(
            in_features,
            out_features,
            init_method=init_method,
            gain_w=gain_w,
            gain_b=gain_b,
            bias=bias,
            device=self.device,
        )

        # Means: same as standard
        self.mw = mw  # (K, N)
        self.mb = mb  # (1, N)

        # Shared scalar variances: use the mean of the init variance
        # (all elements are the same constant anyway)
        self.Sw = torch.tensor(Sw[0, 0].item(), device=self.device, dtype=torch.float32)
        self.Sb = torch.tensor(Sb[0, 0].item(), device=self.device, dtype=torch.float32)

        # Saved for backward
        self.ma_in = None

        # Parameter deltas
        self.delta_mw = None
        self.delta_mb = None

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        ma : Tensor (B, K)  activation means
        Sa : Tensor (B, K)  activation variances

        Returns
        -------
        mz : Tensor (B, N)  pre-activation means
        Sz : Tensor (B, N)  pre-activation variances
        """
        self.ma_in = ma

        # Mean: standard matmul
        mz = torch.matmul(ma, self.mw) + self.mb

        # Variance with shared scalars:
        # Sz = Sw * row_sum(ma²) + Sa @ mw² + Sw * row_sum(Sa) + Sb
        Sw_val = self.Sw.item()
        Sb_val = self.Sb.item()

        Sa_mw2 = torch.matmul(Sa, self.mw * self.mw)  # (B, N)
        row_ma2 = (ma * ma).sum(dim=1, keepdim=True)  # (B, 1)
        row_Sa = Sa.sum(dim=1, keepdim=True)  # (B, 1)
        Sz = Sa_mw2 + Sw_val * (row_ma2 + row_Sa) + Sb_val  # broadcasts

        return mz, Sz

    # ------------------------------------------------------------------
    #  Backward (compute deltas only — NO parameter update)
    # ------------------------------------------------------------------
    def backward(self, delta_mz: Tensor, delta_Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute parameter deltas and propagate to the previous layer.

        Parameters
        ----------
        delta_mz : Tensor (B, N)
        delta_Sz : Tensor (B, N)

        Returns
        -------
        delta_ma : Tensor (B, K)
        delta_Sa : Tensor (B, K)
        """
        # ── Raw gradients (sum over batch) ──
        grad_mw = torch.matmul(self.ma_in.T, delta_mz)  # (K, N)
        grad_mb = delta_mz.sum(0, keepdim=True)  # (1, N)
        grad_Sw = torch.matmul((self.ma_in**2).T, delta_Sz)  # (K, N)
        grad_Sb = delta_Sz.sum(0, keepdim=True)  # (1, N)

        # ── Parameter deltas (cuTAGI convention with scalar Sw) ──
        # delta_mw = Sw * grad_mw   (scalar broadcast)
        self.delta_mw = self.Sw.item() * grad_mw
        self.delta_mb = self.Sb.item() * grad_mb

        # ── Store variance gradients for update() ──
        self._grad_Sw = grad_Sw
        self._grad_Sb = grad_Sb

        # ── Propagate deltas to previous layer ──
        delta_ma, delta_Sa = triton_fused_backward_delta(delta_mz, delta_Sz, self.mw)
        return delta_ma, delta_Sa

    # ------------------------------------------------------------------
    #  Update (apply capped deltas — called by the network)
    # ------------------------------------------------------------------
    def update(self, cap_factor: float) -> None:
        """Apply stored parameter deltas with capping + scalar variance update."""
        # ── Mean update (capped, using scalar Sw for delta_bar) ──
        update_shared_mean(self.mw, self.delta_mw, self.Sw.item(), cap_factor)
        if self.has_bias:
            update_shared_mean(self.mb, self.delta_mb, self.Sb.item(), cap_factor)

        # ── Scalar variance update (precision-space) ──
        update_shared_variance(self.Sw, self._grad_Sw)
        if self.has_bias:
            update_shared_variance(self.Sb, self._grad_Sb)

    @property
    def num_parameters(self) -> int:
        """Total learnable scalars: weight/bias means + 2 scalar variances (Sw, Sb)."""
        n_means = self.mw.numel() + (self.mb.numel() if self.has_bias else 0)
        n_vars = 1 + (1 if self.has_bias else 0)
        return n_means + n_vars

    def __repr__(self):
        return (
            f"SharedVarLinear(in={self.in_features}, out={self.out_features}, bias={self.has_bias})"
        )
