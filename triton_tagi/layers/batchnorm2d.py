"""
Bayesian Batch Normalization layer for TAGI (2D, channel-wise).

In TAGI every activation is a Gaussian z ~ N(μ, S).  BatchNorm normalises
the channel means using running (or batch) statistics, then applies a
learnable affine transform with Gaussian parameters γ and β.

Forward pass (per channel c):
    1. Normalise:  μ_hat = (μ_z - μ_run) / √(S_run + ε)
                   S_hat = S_z / (S_run + ε)
    2. Affine:     μ_out = μ_γ · μ_hat  +  μ_β
                   S_out = μ_γ² · S_hat  +  S_γ · μ_hat²  +  S_γ · S_hat  +  S_β

    Running statistics are updated with exponential moving average during
    training:
        μ_run ← (1 − α) · μ_run  +  α · μ_batch
        S_run ← (1 − α) · S_run  +  α · S_batch

Backward pass:
    The Jacobian of the affine transform w.r.t. the normalised input is γ,
    which gives us the same delta-propagation pattern as in the Linear layer
    but per-channel with diagonal weight:

        δ_μ_hat = δ_μ_out · μ_γ           (mean delta through affine)
        δ_S_hat = δ_S_out · μ_γ²          (var  delta through affine)

    Then un-normalise to get deltas in the original activation space:
        δ_μ_z = δ_μ_hat / √(S_run + ε)
        δ_S_z = δ_S_hat / (S_run + ε)

    Parameter deltas (cuTAGI convention):
        Δ_μ_γ = S_γ · Σ (δ_μ_out · μ_hat)
        Δ_S_γ = S_γ² · Σ (δ_S_out · μ_hat²)
        Δ_μ_β = S_β · Σ δ_μ_out
        Δ_S_β = S_β² · Σ δ_S_out

All fused into Triton kernels for maximum GPU throughput.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from ..base import LearnableLayer
from ..param_init import init_weight_bias_norm
from ..update.parameters import update_parameters

BLOCK = 1024


# ======================================================================
#  Triton kernel — forward pass
# ======================================================================


@triton.jit
def _batchnorm_fwd_kernel(
    # Input pointers
    mz_ptr,
    Sz_ptr,
    # Running stats
    run_m_ptr,
    run_s_ptr,
    # Parameters (gamma, beta)
    mg_ptr,
    Sg_ptr,
    mb_ptr,
    Sb_ptr,
    # Output pointers
    ma_ptr,
    Sa_ptr,
    # Normalised cache (for backward)
    mhat_ptr,
    Shat_ptr,
    # Dimensions
    N,
    C,
    HW,
    eps,
    BLOCK: tl.constexpr,
):
    """
    Fused BN forward: normalise + affine for both mean and variance.
    Each thread handles one element of the (N, C, H*W) tensor.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * HW
    valid = offs < total

    # Determine channel index for each element
    c = (offs // HW) % C

    # Load inputs
    mz = tl.load(mz_ptr + offs, mask=valid, other=0.0)
    Sz = tl.load(Sz_ptr + offs, mask=valid, other=0.0)

    # Load running stats (per-channel)
    run_m = tl.load(run_m_ptr + c, mask=valid, other=0.0)
    run_s = tl.load(run_s_ptr + c, mask=valid, other=0.0)

    # Load parameters (per-channel)
    mg = tl.load(mg_ptr + c, mask=valid, other=1.0)
    Sg = tl.load(Sg_ptr + c, mask=valid, other=0.0)
    mb = tl.load(mb_ptr + c, mask=valid, other=0.0)
    Sb = tl.load(Sb_ptr + c, mask=valid, other=0.0)

    # Normalise
    inv_std = 1.0 / tl.sqrt(run_s + eps)
    m_hat = (mz - run_m) * inv_std
    S_hat = Sz / (run_s + eps)

    # Affine transform (propagating Gaussian moments)
    #   μ_out = μ_γ · μ_hat + μ_β
    #   S_out = μ_γ² · S_hat + S_γ · μ_hat² + S_γ · S_hat + S_β
    ma_out = mg * m_hat + mb
    Sa_out = mg * mg * S_hat + Sg * m_hat * m_hat + Sg * S_hat + Sb

    # Store outputs
    tl.store(ma_ptr + offs, ma_out, mask=valid)
    tl.store(Sa_ptr + offs, Sa_out, mask=valid)
    tl.store(mhat_ptr + offs, m_hat, mask=valid)
    tl.store(Shat_ptr + offs, S_hat, mask=valid)


# ======================================================================
#  Triton kernel — backward pass
# ======================================================================


@triton.jit
def _batchnorm_bwd_kernel(
    # Incoming deltas
    dma_ptr,
    dSa_ptr,
    # Cached normalised values
    mhat_ptr,
    Shat_ptr,
    # Running stats
    run_s_ptr,
    # Parameters (gamma)
    mg_ptr,
    # Output deltas (to propagate)
    dmz_ptr,
    dSz_ptr,
    # Dimensions
    N,
    C,
    HW,
    eps,
    BLOCK: tl.constexpr,
):
    """
    BN backward: propagate deltas through affine + normalisation.
    Each thread handles one element.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * HW
    valid = offs < total

    c = (offs // HW) % C

    # Load incoming deltas
    dma = tl.load(dma_ptr + offs, mask=valid, other=0.0)
    dSa = tl.load(dSa_ptr + offs, mask=valid, other=0.0)

    # Load cached normalised values
    # (not needed for delta propagation, but kept for consistency)

    # Load gamma mean and running var
    mg = tl.load(mg_ptr + c, mask=valid, other=1.0)
    run_s = tl.load(run_s_ptr + c, mask=valid, other=1.0)

    # Through affine:  δ_μ_hat = δ_μ_out · μ_γ
    #                  δ_S_hat = δ_S_out · μ_γ²
    d_mhat = dma * mg
    d_Shat = dSa * mg * mg

    # Through normalisation:  δ_μ_z = δ_μ_hat / √(S_run + ε)
    #                         δ_S_z = δ_S_hat / (S_run + ε)
    inv_std = 1.0 / tl.sqrt(run_s + eps)
    inv_var = 1.0 / (run_s + eps)

    dmz_out = d_mhat * inv_std
    dSz_out = d_Shat * inv_var

    tl.store(dmz_ptr + offs, dmz_out, mask=valid)
    tl.store(dSz_ptr + offs, dSz_out, mask=valid)


# ======================================================================
#  Triton kernel — compute batch mean per channel (reduction)
# ======================================================================


@triton.jit
def _channel_mean_kernel(
    inp_ptr,
    out_ptr,
    N,
    C,
    HW,
    inv_count,  # 1.0 / (N * HW)
    BLOCK: tl.constexpr,
):
    """Compute per-channel mean: out[c] = mean over (N, HW) of inp[:, c, :]."""
    c = tl.program_id(0)  # one program per channel
    if c >= C:
        return

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    total = N * HW

    for start in range(0, total, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < total

        n = offs // HW
        hw = offs % HW
        idx = n * C * HW + c * HW + hw

        val = tl.load(inp_ptr + idx, mask=valid, other=0.0)
        acc += val

    result = tl.sum(acc, axis=0) * inv_count
    tl.store(out_ptr + c, result)


# ======================================================================
#  BatchNorm2D Layer
# ======================================================================


class BatchNorm2D(LearnableLayer):
    """
    Bayesian Batch Normalization (2D, channel-wise) for TAGI.

    Normalises each channel using running statistics and applies a
    learnable affine transform with Gaussian parameters γ and β.

    Parameters
    ----------
    num_features : int    number of channels (C)
    momentum     : float  EMA momentum for running stats (default 0.1)
    eps          : float  numerical stability constant (default 1e-5)
    device       : str or torch.device
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
        device: str = "cuda",
        gain_w: float = 1.0,
        gain_b: float = 1.0,
        preserve_var: bool = True,
    ) -> None:
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.device = torch.device(device)
        self.training = True
        self.preserve_var = preserve_var

        # Flag for data-dependent initialization
        self._is_initialized = False

        # --- cuTAGI-matching initialization ---
        # mw=1, mb=0, Sw=Sb=2/(n+n)=1/n  (matches cuTAGI init_weight_bias_norm)
        self.mw, self.Sw, self.mb, self.Sb = init_weight_bias_norm(
            num_features, gain_w=gain_w, gain_b=gain_b, device=self.device
        )
        self.has_bias = True

        # --- Running statistics ---
        self.running_mean = torch.zeros(num_features, device=self.device)
        self.running_var = torch.ones(num_features, device=self.device)

        # --- Saved for backward ---
        self.m_hat = None
        self.S_hat = None
        self.input_shape = None
        self._norm_var = None  # variance used for normalization (batch or running)

        # --- Parameter deltas ---
        self.delta_mw = None
        self.delta_Sw = None
        self.delta_mb = None
        self.delta_Sb = None

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Set the layer to training mode (update running stats)."""
        self.training = True

    def eval(self) -> None:
        """Set the layer to evaluation mode (use running stats only)."""
        self.training = False

    def _compute_batch_stats(self, mz, Sz):
        """
        Compute per-channel batch mean and variance from Gaussian inputs.

        batch_mean[c] = E_{n,h,w}[μ_z]                                 (divided by n)
        batch_var[c]  = (Σ (μ_z - batch_mean)² + Σ S_z) / (n − 1)      (Bessel-corrected)

        The variance uses the Bessel-corrected (n − 1) denominator to match
        cuTAGI's ``batchnorm2d_sample_var``. This matters for parity with
        cuTAGI under identical init + identical batches, since any per-step
        discrepancy in running statistics compounds across training.

        Returns batch_mean (C,), batch_var (C,).
        """
        N, C, H, W = mz.shape
        HW = H * W
        n = N * HW
        inv_count = 1.0 / n

        BLOCK_RED = min(1024, triton.next_power_of_2(n))

        batch_mean = torch.empty(C, device=mz.device, dtype=mz.dtype)
        batch_var_s = torch.empty(C, device=mz.device, dtype=mz.dtype)
        batch_mean_sq = torch.empty(C, device=mz.device, dtype=mz.dtype)

        _channel_mean_kernel[(C,)](mz.contiguous(), batch_mean, N, C, HW, inv_count, BLOCK=BLOCK_RED)
        _channel_mean_kernel[(C,)](Sz.contiguous(), batch_var_s, N, C, HW, inv_count, BLOCK=BLOCK_RED)
        _channel_mean_kernel[(C,)]((mz * mz).contiguous(), batch_mean_sq, N, C, HW, inv_count, BLOCK=BLOCK_RED)

        # Current accumulators divide by n; rescale to (n − 1) for Bessel.
        bessel = n / (n - 1)
        batch_var = (batch_var_s + batch_mean_sq - batch_mean * batch_mean) * bessel
        return batch_mean, batch_var

    def _update_running_stats(self, batch_mean, batch_var):
        """
        Update running mean/var with EMA.  First call initialises directly.
        Also applies preserve_var gamma init on the first call.
        """
        if not self._is_initialized:
            self.running_mean = batch_mean.clone()
            self.running_var = batch_var.clone()
            if self.preserve_var:
                self.mw = torch.sqrt(self.running_var + self.eps).clone()
            self._is_initialized = True
        else:
            alpha = self.momentum
            self.running_mean = (1 - alpha) * self.running_mean + alpha * batch_mean
            self.running_var = (1 - alpha) * self.running_var + alpha * batch_var

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, mz: Tensor, Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Normalise and apply the learnable affine transform (γ, β).

        Normalise per channel using running statistics:
            μ_hat = (μ_z − μ_run) / √(S_run + ε)
            S_hat = S_z / (S_run + ε)

        Affine with Gaussian γ ~ N(μ_γ, S_γ), β ~ N(μ_β, S_β):
            μ_out = μ_γ · μ_hat + μ_β
            S_out = μ_γ² · S_hat  +  S_γ · (μ_hat² + S_hat)  +  S_β

        Parameters
        ----------
        mz : Tensor (N, C, H, W)  pre-activation means
        Sz : Tensor (N, C, H, W)  pre-activation variances

        Returns
        -------
        ma : Tensor (N, C, H, W)  post-BN activation means
        Sa : Tensor (N, C, H, W)  post-BN activation variances
        """
        self.input_shape = mz.shape
        N, C, H, W = mz.shape
        HW = H * W

        # During training: normalize with current batch stats (matching cuTAGI).
        # During eval: normalize with accumulated running stats.
        if self.training:
            batch_mean, batch_var = self._compute_batch_stats(mz, Sz)
            self._update_running_stats(batch_mean, batch_var)
            norm_mean = batch_mean
            norm_var = batch_var
        else:
            norm_mean = self.running_mean
            norm_var = self.running_var
        self._norm_var = norm_var

        # Allocate outputs + cache
        ma = torch.empty_like(mz)
        Sa = torch.empty_like(Sz)
        mhat = torch.empty_like(mz)
        Shat = torch.empty_like(Sz)

        total = N * C * HW
        grid = (triton.cdiv(total, BLOCK),)

        _batchnorm_fwd_kernel[grid](
            mz.contiguous(),
            Sz.contiguous(),
            norm_mean,
            norm_var,
            self.mw,
            self.Sw,
            self.mb,
            self.Sb,
            ma,
            Sa,
            mhat,
            Shat,
            N,
            C,
            HW,
            self.eps,
            BLOCK=BLOCK,
        )

        # Cache for backward
        self.m_hat = mhat
        self.S_hat = Shat

        return ma, Sa

    # ------------------------------------------------------------------
    #  Backward (compute deltas only — NO parameter update)
    # ------------------------------------------------------------------
    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute parameter deltas and back-propagate innovation deltas.

        Through the affine (Jacobian = μ_γ per channel):
            δμ_hat = μ_γ · δμ_out
            δS_hat = μ_γ² · δS_out

        Un-normalise to input space:
            δμ_z = δμ_hat / √(S_run + ε)
            δS_z = δS_hat / (S_run + ε)

        Parameter deltas (cuTAGI convention):
            Δμ_γ = S_γ · Σ(δμ_out · μ_hat)    Δμ_β = S_β · Σ δμ_out
            ΔS_γ = S_γ² · Σ(δS_out · μ_hat²)  ΔS_β = S_β² · Σ δS_out

        Parameters
        ----------
        delta_ma : Tensor (N, C, H, W)  mean delta from next layer
        delta_Sa : Tensor (N, C, H, W)  variance delta from next layer

        Returns
        -------
        delta_mz : Tensor (N, C, H, W)  mean delta to propagate
        delta_Sz : Tensor (N, C, H, W)  variance delta to propagate
        """
        N, C, H, W = self.input_shape
        HW = H * W

        # ── Parameter gradients (sum over N, H, W per channel) ──
        # For gamma:
        #   grad_mg[c] = Σ_{n,h,w} δ_μ_out · μ_hat
        #   grad_Sg[c] = Σ_{n,h,w} δ_S_out · μ_hat²
        # For beta:
        #   grad_mb[c] = Σ_{n,h,w} δ_μ_out
        #   grad_Sb[c] = Σ_{n,h,w} δ_S_out

        # Reshape for per-channel reduction: (N, C, HW)
        dma_flat = delta_ma.reshape(N, C, HW)
        dSa_flat = delta_Sa.reshape(N, C, HW)
        mhat_flat = self.m_hat.reshape(N, C, HW)

        # Per-channel sums  (C,)
        grad_mg = (dma_flat * mhat_flat).sum(dim=(0, 2))  # Σ δ_μ · μ_hat
        grad_Sg = (dSa_flat * mhat_flat**2).sum(dim=(0, 2))  # Σ δ_S · μ_hat²
        grad_mb = dma_flat.sum(dim=(0, 2))  # Σ δ_μ
        grad_Sb = dSa_flat.sum(dim=(0, 2))  # Σ δ_S

        # ── Parameter deltas (cuTAGI convention) ──
        self.delta_mw = self.Sw * grad_mg
        self.delta_Sw = (self.Sw**2) * grad_Sg
        self.delta_mb = self.Sb * grad_mb
        self.delta_Sb = (self.Sb**2) * grad_Sb

        # ── Propagate deltas to previous layer ──
        delta_mz = torch.empty_like(delta_ma)
        delta_Sz = torch.empty_like(delta_Sa)

        total = N * C * HW
        grid = (triton.cdiv(total, BLOCK),)

        _batchnorm_bwd_kernel[grid](
            delta_ma.contiguous(),
            delta_Sa.contiguous(),
            self.m_hat,
            self.S_hat,
            self._norm_var,
            self.mw,
            delta_mz,
            delta_Sz,
            N,
            C,
            HW,
            self.eps,
            BLOCK=BLOCK,
        )

        return delta_mz, delta_Sz

    # ------------------------------------------------------------------
    #  Update (apply capped deltas — called by the network)
    # ------------------------------------------------------------------
    def update(self, cap_factor: float) -> None:
        """
        Apply stored parameter deltas with cuTAGI-style capping.

        Parameters
        ----------
        cap_factor : float  regularisation strength (from get_cap_factor)
        """
        update_parameters(self.mw, self.Sw, self.delta_mw, self.delta_Sw, cap_factor)
        update_parameters(self.mb, self.Sb, self.delta_mb, self.delta_Sb, cap_factor)

    @property
    def num_parameters(self) -> int:
        """Total learnable scalars: 2 × (gamma + beta) means and variances."""
        return 2 * (self.mw.numel() + self.mb.numel())

    def __repr__(self):
        return (
            f"BatchNorm2D(num_features={self.num_features}, "
            f"momentum={self.momentum}, eps={self.eps})"
        )
