"""
Shared-Variance Batch Normalization layer for TAGI (2D, channel-wise).

Same normalisation + affine transform as standard BatchNorm2D, but γ and β
each have a single shared scalar variance instead of per-channel variances.

The forward pass, backward delta propagation, and running stats update are
identical to the standard BatchNorm2D.  Only the parameter variance
representation and update differ.
"""

import torch
import triton
import triton.language as tl

from ..update.shared_var_parameters import update_shared_variance, update_shared_mean
from .batchnorm2d import (
    _batchnorm_fwd_kernel, _batchnorm_bwd_kernel, _channel_mean_kernel,
    BLOCK,
)


class SharedVarBatchNorm2D:
    """
    Bayesian Batch Normalization with shared scalar variance for γ and β.

    Parameters
    ----------
    num_features : int    number of channels (C)
    momentum     : float  EMA momentum for running stats
    eps          : float  numerical stability constant
    device       : str or torch.device
    gain_w       : float  gain for gamma variance
    gain_b       : float  gain for beta variance
    preserve_var : bool   if True, set γ to preserve incoming variance on first pass
    """

    def __init__(self, num_features, momentum=0.1, eps=1e-5, device="cuda",
                 gain_w=1.0, gain_b=1.0, preserve_var=True):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.device = torch.device(device)
        self.training = True
        self.preserve_var = preserve_var
        self._is_initialized = False

        # --- Parameters ---
        # Gamma (weight): mean per-channel, scalar shared variance
        self.mw = torch.ones(num_features, device=self.device)
        # Small initial variance for BN parameters
        init_var = 1e-3
        self.sw = torch.tensor(init_var, device=self.device, dtype=torch.float32)

        # Beta (bias): mean per-channel, scalar shared variance
        self.mb = torch.zeros(num_features, device=self.device)
        self.sb = torch.tensor(init_var, device=self.device, dtype=torch.float32)
        self.has_bias = True

        # --- Running statistics ---
        self.running_mean = torch.zeros(num_features, device=self.device)
        self.running_var  = torch.ones(num_features, device=self.device)

        # --- Saved for backward ---
        self.m_hat = None
        self.S_hat = None
        self.input_shape = None

        # --- Parameter deltas ---
        self.delta_mw = None
        self.delta_mb = None

    # ------------------------------------------------------------------
    #  Properties for Sequential / checkpoint compatibility
    # ------------------------------------------------------------------
    @property
    def Sw(self):
        return self.sw.expand(self.num_features)

    @property
    def Sb(self):
        return self.sb.expand(self.num_features)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def _update_running_stats(self, mz, Sz):
        """Update running mean/var with EMA (identical to standard BN)."""
        N, C, H, W = mz.shape
        HW = H * W
        inv_count = 1.0 / (N * HW)

        batch_mean = torch.empty(C, device=mz.device, dtype=mz.dtype)
        batch_var  = torch.empty(C, device=mz.device, dtype=mz.dtype)

        BLOCK_RED = min(1024, triton.next_power_of_2(N * HW))

        _channel_mean_kernel[(C,)](
            mz.contiguous(), batch_mean,
            N, C, HW, inv_count,
            BLOCK=BLOCK_RED,
        )
        _channel_mean_kernel[(C,)](
            Sz.contiguous(), batch_var,
            N, C, HW, inv_count,
            BLOCK=BLOCK_RED,
        )

        # Total variance = E[S_z] + Var(μ_z)
        mz_sq = mz * mz
        batch_mean_sq = torch.empty(C, device=mz.device, dtype=mz.dtype)
        _channel_mean_kernel[(C,)](
            mz_sq.contiguous(), batch_mean_sq,
            N, C, HW, inv_count,
            BLOCK=BLOCK_RED,
        )
        batch_var = batch_var + (batch_mean_sq - batch_mean * batch_mean)

        if not self._is_initialized:
            self.running_mean = batch_mean.clone()
            self.running_var  = batch_var.clone()

            if self.preserve_var:
                self.mw = torch.sqrt(self.running_var + self.eps).clone()

            self._is_initialized = True
        else:
            alpha = self.momentum
            self.running_mean = (1 - alpha) * self.running_mean + alpha * batch_mean
            self.running_var  = (1 - alpha) * self.running_var  + alpha * batch_var

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, mz, Sz):
        """
        Parameters
        ----------
        mz : Tensor (N, C, H, W)
        Sz : Tensor (N, C, H, W)

        Returns
        -------
        ma : Tensor (N, C, H, W)
        Sa : Tensor (N, C, H, W)
        """
        self.input_shape = mz.shape
        N, C, H, W = mz.shape
        HW = H * W

        if self.training:
            self._update_running_stats(mz, Sz)

        # Allocate outputs + cache
        ma   = torch.empty_like(mz)
        Sa   = torch.empty_like(Sz)
        mhat = torch.empty_like(mz)
        Shat = torch.empty_like(Sz)

        total = N * C * HW
        grid = (triton.cdiv(total, BLOCK),)

        # Expand scalar variances to per-channel for the Triton kernel
        Sw_expanded = self.sw.expand(C).contiguous()
        Sb_expanded = self.sb.expand(C).contiguous()

        _batchnorm_fwd_kernel[grid](
            mz.contiguous(), Sz.contiguous(),
            self.running_mean, self.running_var,
            self.mw, Sw_expanded, self.mb, Sb_expanded,
            ma, Sa, mhat, Shat,
            N, C, HW,
            self.eps,
            BLOCK=BLOCK,
        )

        self.m_hat = mhat
        self.S_hat = Shat

        return ma, Sa

    # ------------------------------------------------------------------
    #  Backward (compute deltas only — NO parameter update)
    # ------------------------------------------------------------------
    def backward(self, delta_ma, delta_Sa):
        """
        Parameters
        ----------
        delta_ma : Tensor (N, C, H, W)
        delta_Sa : Tensor (N, C, H, W)

        Returns
        -------
        delta_mz : Tensor (N, C, H, W)
        delta_Sz : Tensor (N, C, H, W)
        """
        N, C, H, W = self.input_shape
        HW = H * W

        # Per-channel gradients (sum over N, H, W)
        dma_flat  = delta_ma.reshape(N, C, HW)
        dSa_flat  = delta_Sa.reshape(N, C, HW)
        mhat_flat = self.m_hat.reshape(N, C, HW)

        grad_mg = (dma_flat * mhat_flat).sum(dim=(0, 2))           # (C,)
        grad_Sg = (dSa_flat * mhat_flat ** 2).sum(dim=(0, 2))      # (C,)
        grad_mb = dma_flat.sum(dim=(0, 2))                          # (C,)
        grad_Sb = dSa_flat.sum(dim=(0, 2))                          # (C,)

        # Parameter deltas (scalar sw broadcast)
        self.delta_mw = self.sw.item() * grad_mg
        self.delta_mb = self.sb.item() * grad_mb

        # Store variance grads for update()
        self._grad_Sw = grad_Sg
        self._grad_Sb = grad_Sb

        # Propagate deltas to previous layer
        delta_mz = torch.empty_like(delta_ma)
        delta_Sz = torch.empty_like(delta_Sa)

        total = N * C * HW
        grid = (triton.cdiv(total, BLOCK),)

        _batchnorm_bwd_kernel[grid](
            delta_ma.contiguous(), delta_Sa.contiguous(),
            self.m_hat, self.S_hat,
            self.running_var,
            self.mw,
            delta_mz, delta_Sz,
            N, C, HW,
            self.eps,
            BLOCK=BLOCK,
        )

        return delta_mz, delta_Sz

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
        return (f"SharedVarBatchNorm2D(num_features={self.num_features}, "
                f"momentum={self.momentum}, eps={self.eps})")
