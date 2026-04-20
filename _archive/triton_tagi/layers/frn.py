"""
Filter Response Normalization (FRN) layer for TAGI.

From: "Filter Response Normalization Layer: Eliminating Batch Dependence
in the Training of Deep Neural Networks" (Singh & Davis, 2020).

FRN normalises each channel per-sample using the second moment (no batch
statistics at all), then applies a learnable affine transform.

Forward pass (per sample n, channel c):
    1. Compute second moment:
           ν²[n,c] = (1/HW) Σ_{h,w} E[x²_{n,c,h,w}]
                   = (1/HW) Σ_{h,w} (μ² + S)
       Under the TAGI diagonal approximation, ν² is a spatial average
       of many terms, so by CLT its variance is small — we treat it as
       deterministic at its expected value.

    2. Normalise:
           μ_hat = μ_x / √(ν² + ε)
           S_hat = S_x / (ν² + ε)

    3. Affine (Gaussian γ and β, same as BatchNorm):
           μ_out = μ_γ · μ_hat  +  μ_β
           S_out = μ_γ² · S_hat  +  S_γ · μ_hat²  +  S_γ · S_hat  +  S_β

Backward pass:
    Through affine:   δ_μ_hat = δ_μ_out · μ_γ
                      δ_S_hat = δ_S_out · μ_γ²
    Through norm:     δ_μ_x = δ_μ_hat / √(ν² + ε)
                      δ_S_x = δ_S_hat / (ν² + ε)

    Parameter deltas (cuTAGI convention):
        Δ_μ_γ = S_γ · Σ (δ_μ_out · μ_hat)
        Δ_S_γ = S_γ² · Σ (δ_S_out · μ_hat²)
        Δ_μ_β = S_β · Σ δ_μ_out
        Δ_S_β = S_β² · Σ δ_S_out

Key advantage over BatchNorm for TAGI:
    - No batch statistics → no EMA, no train/eval divergence
    - Per-sample normalisation → cleaner moment propagation
    - The "treat ν² as deterministic" approximation is well-justified
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from ..base import LearnableLayer
from ..update.parameters import update_parameters

BLOCK = 1024


# ======================================================================
#  Triton kernel — forward pass
# ======================================================================


@triton.jit
def _frn_fwd_kernel(
    # Input pointers
    mz_ptr,
    Sz_ptr,
    # Per-(n,c) second moment (precomputed)
    nu2_ptr,
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
    Fused FRN forward: normalise by per-sample second moment + affine.
    Each thread handles one element of the (N, C, H*W) tensor.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * HW
    valid = offs < total

    # Determine (n, c) indices
    c = (offs // HW) % C
    n = offs // (C * HW)
    nc = n * C + c  # index into nu2 (N, C)

    # Load inputs
    mz = tl.load(mz_ptr + offs, mask=valid, other=0.0)
    Sz = tl.load(Sz_ptr + offs, mask=valid, other=0.0)

    # Load per-(n,c) second moment
    nu2 = tl.load(nu2_ptr + nc, mask=valid, other=1.0)

    # Load parameters (per-channel)
    mg = tl.load(mg_ptr + c, mask=valid, other=1.0)
    Sg = tl.load(Sg_ptr + c, mask=valid, other=0.0)
    mb = tl.load(mb_ptr + c, mask=valid, other=0.0)
    Sb = tl.load(Sb_ptr + c, mask=valid, other=0.0)

    # Normalise by second moment
    inv_nu = 1.0 / tl.sqrt(nu2 + eps)
    m_hat = mz * inv_nu
    S_hat = Sz * inv_nu * inv_nu

    # Affine transform (propagating Gaussian moments)
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
def _frn_bwd_kernel(
    # Incoming deltas
    dma_ptr,
    dSa_ptr,
    # Per-(n,c) second moment
    nu2_ptr,
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
    FRN backward: propagate deltas through affine + normalisation.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * HW
    valid = offs < total

    c = (offs // HW) % C
    n = offs // (C * HW)
    nc = n * C + c

    # Load incoming deltas
    dma = tl.load(dma_ptr + offs, mask=valid, other=0.0)
    dSa = tl.load(dSa_ptr + offs, mask=valid, other=0.0)

    # Load gamma mean and per-sample second moment
    mg = tl.load(mg_ptr + c, mask=valid, other=1.0)
    nu2 = tl.load(nu2_ptr + nc, mask=valid, other=1.0)

    # Through affine
    d_mhat = dma * mg
    d_Shat = dSa * mg * mg

    # Through normalisation
    inv_nu = 1.0 / tl.sqrt(nu2 + eps)
    inv_nu2 = inv_nu * inv_nu

    dmz_out = d_mhat * inv_nu
    dSz_out = d_Shat * inv_nu2

    tl.store(dmz_ptr + offs, dmz_out, mask=valid)
    tl.store(dSz_ptr + offs, dSz_out, mask=valid)


# ======================================================================
#  FRN2D Layer
# ======================================================================


class FRN2D(LearnableLayer):
    """
    Filter Response Normalization (2D, channel-wise) for TAGI.

    Normalises each channel per-sample using the second moment ν² and
    applies a learnable affine transform with Gaussian parameters γ and β.

    Unlike BatchNorm, FRN has no batch statistics — normalisation is
    entirely per-sample, making it ideal for small-batch or online settings.

    Parameters
    ----------
    num_features : int    number of channels (C)
    eps          : float  numerical stability constant (default 1e-6)
    device       : str or torch.device
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        device: str = "cuda",
        gain_w: float = 1.0,
        gain_b: float = 1.0,
    ) -> None:
        self.num_features = num_features
        self.eps = eps
        self.device = torch.device(device)

        # Learnable gamma and beta (Gaussian parameters)
        self.mw = torch.ones(num_features, device=self.device)
        self.Sw = torch.full((num_features,), 1e-3, device=self.device)
        self.mb = torch.zeros(num_features, device=self.device)
        self.Sb = torch.full((num_features,), 1e-3, device=self.device)
        self.has_bias = True

        # Data-dependent initialization flag
        # On the first forward pass, set β to center the output.
        # Unlike BN which subtracts the running mean, FRN only divides
        # by RMS — so without this, positive bias from TLU accumulates.
        self._is_initialized = False

        # Saved for backward
        self.m_hat = None
        self.S_hat = None
        self.nu2 = None  # (N, C) per-sample second moment
        self.input_shape = None

        # Parameter deltas
        self.delta_mw = None
        self.delta_Sw = None
        self.delta_mb = None
        self.delta_Sb = None

    # ------------------------------------------------------------------
    #  Train / Eval (no-ops — FRN has no batch statistics)
    # ------------------------------------------------------------------
    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, mz: Tensor, Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        mz : Tensor (N, C, H, W)  pre-activation means
        Sz : Tensor (N, C, H, W)  pre-activation variances

        Returns
        -------
        ma : Tensor (N, C, H, W)  post-FRN activation means
        Sa : Tensor (N, C, H, W)  post-FRN activation variances
        """
        self.input_shape = mz.shape
        N, C, H, W = mz.shape
        HW = H * W

        # Compute per-sample, per-channel second moment:
        #   ν²[n,c] = (1/HW) Σ_{h,w} E[x²] = (1/HW) Σ (μ² + S)
        nu2 = (mz**2 + Sz).mean(dim=(2, 3))  # (N, C)
        self.nu2 = nu2

        # Allocate outputs + cache
        ma = torch.empty_like(mz)
        Sa = torch.empty_like(Sz)
        mhat = torch.empty_like(mz)
        Shat = torch.empty_like(Sz)

        total = N * C * HW
        grid = (triton.cdiv(total, BLOCK),)

        _frn_fwd_kernel[grid](
            mz.contiguous(),
            Sz.contiguous(),
            nu2.contiguous(),
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

        # ── Data-dependent init (first pass only) ──
        # FRN doesn't center like BN does, so positive bias from TLU
        # accumulates through the network.  On the first forward pass,
        # set β = −γ · E[μ_hat] so the output is zero-centered.
        if not self._is_initialized:
            channel_mean = mhat.mean(dim=(0, 2, 3))  # (C,)
            self.mb = (-self.mw * channel_mean).clone()
            # Re-run the affine with the corrected β
            ma = torch.empty_like(mz)
            Sa = torch.empty_like(Sz)
            _frn_fwd_kernel[grid](
                mz.contiguous(),
                Sz.contiguous(),
                nu2.contiguous(),
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
            self.m_hat = mhat
            self.S_hat = Shat
            self._is_initialized = True

        return ma, Sa

    # ------------------------------------------------------------------
    #  Backward
    # ------------------------------------------------------------------
    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute parameter deltas and propagate to the previous layer.

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
        dma_flat = delta_ma.reshape(N, C, HW)
        dSa_flat = delta_Sa.reshape(N, C, HW)
        mhat_flat = self.m_hat.reshape(N, C, HW)

        grad_mg = (dma_flat * mhat_flat).sum(dim=(0, 2))
        grad_Sg = (dSa_flat * mhat_flat**2).sum(dim=(0, 2))
        grad_mb = dma_flat.sum(dim=(0, 2))
        grad_Sb = dSa_flat.sum(dim=(0, 2))

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

        _frn_bwd_kernel[grid](
            delta_ma.contiguous(),
            delta_Sa.contiguous(),
            self.nu2.contiguous(),
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
    #  Update
    # ------------------------------------------------------------------
    def update(self, cap_factor: float) -> None:
        """
        Apply stored parameter deltas with cuTAGI-style capping.
        """
        update_parameters(self.mw, self.Sw, self.delta_mw, self.delta_Sw, cap_factor)
        update_parameters(self.mb, self.Sb, self.delta_mb, self.delta_Sb, cap_factor)

    @property
    def num_parameters(self) -> int:
        """Total learnable scalars: 2 × (gamma + beta) means and variances."""
        return 2 * (self.mw.numel() + self.mb.numel())

    def __repr__(self):
        return f"FRN2D(num_features={self.num_features}, eps={self.eps})"
