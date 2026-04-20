"""
Thresholded Linear Unit (TLU) activation layer for TAGI.

From: "Filter Response Normalization Layer: Eliminating Batch Dependence
in the Training of Deep Neural Networks" (Singh & Davis, 2020).

TLU is a shifted ReLU with a learnable per-channel threshold τ:

    z = max(y, τ)

For TAGI, y ~ N(μ_y, S_y) and we compute exact moments:

    α = (μ_y − τ) / σ_y

    μ_z = σ_y · φ(α) + μ_y · Φ(α) + τ · (1 − Φ(α))

    E[z²] = (S_y + μ_y²) · Φ(α) + (μ_y + τ) · σ_y · φ(α) + τ² · (1 − Φ(α))
    S_z   = E[z²] − μ_z²

    J = Φ(α)   (Jacobian w.r.t. input, same as ReLU but shifted)

The threshold τ is learnable with Gaussian parameters (μ_τ, S_τ).
Parameter delta for τ:
    ∂E[z]/∂τ = 1 − Φ(α)   (probability of being at the threshold)

When τ = 0, TLU reduces to ReLU exactly.
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
#  Triton kernel — TLU forward
# ======================================================================


@triton.jit
def _tlu_fwd_kernel(
    mz_ptr,
    Sz_ptr,
    # Per-channel threshold
    tau_ptr,
    # Outputs
    ma_ptr,
    Sa_ptr,
    J_ptr,
    # Per-channel Jacobian w.r.t. tau (for backward parameter update)
    Jtau_ptr,
    # Dimensions
    N,
    C,
    HW,
    BLOCK: tl.constexpr,
):
    """
    TLU forward: compute moments of max(y, τ) where y ~ N(μ, S).
    Each thread handles one element of the (N, C, H*W) tensor.
    """
    INV_SQRT_2PI: tl.constexpr = 0.3989422804014327
    INV_SQRT_2: tl.constexpr = 0.7071067811865476

    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * HW
    valid = offs < total

    c = (offs // HW) % C

    mz = tl.load(mz_ptr + offs, mask=valid, other=0.0)
    Sz = tl.load(Sz_ptr + offs, mask=valid, other=0.0)
    tau = tl.load(tau_ptr + c, mask=valid, other=0.0)

    Sz_safe = tl.maximum(Sz, 1e-12)
    sigma = tl.sqrt(Sz_safe)
    alpha = (mz - tau) / sigma

    pdf = tl.exp(-0.5 * alpha * alpha) * INV_SQRT_2PI
    cdf = 0.5 * (1.0 + tl.math.erf(alpha * INV_SQRT_2))

    # E[max(y, τ)] = σ·φ(α) + μ·Φ(α) + τ·(1−Φ(α))
    mu_a = sigma * pdf + mz * cdf + tau * (1.0 - cdf)
    mu_a = tl.maximum(mu_a, 1e-7)

    # E[max(y, τ)²] = (S + μ²)·Φ(α) + (μ+τ)·σ·φ(α) + τ²·(1−Φ(α))
    E_sq = (Sz_safe + mz * mz) * cdf + (mz + tau) * sigma * pdf + tau * tau * (1.0 - cdf)
    var_a = E_sq - mu_a * mu_a
    var_a = tl.maximum(var_a, 1e-7)

    # Jacobian w.r.t. input = Φ(α)
    J = cdf

    # Jacobian w.r.t. τ = 1 − Φ(α)  (for parameter update)
    Jtau = 1.0 - cdf

    tl.store(ma_ptr + offs, mu_a, mask=valid)
    tl.store(Sa_ptr + offs, var_a, mask=valid)
    tl.store(J_ptr + offs, J, mask=valid)
    tl.store(Jtau_ptr + offs, Jtau, mask=valid)


# ======================================================================
#  TLU Layer
# ======================================================================


class TLU(LearnableLayer):
    """
    Thresholded Linear Unit activation for TAGI.

    Computes z = max(y, τ) with learnable per-channel threshold τ.
    When τ = 0, this is equivalent to ReLU.

    Designed to pair with FRN2D as a replacement for BatchNorm + ReLU.

    Parameters
    ----------
    num_features : int   number of channels (C) — τ is per-channel
    device       : str or torch.device
    """

    def __init__(self, num_features: int, device: str = "cuda") -> None:
        self.num_features = num_features
        self.device = torch.device(device)

        # Learnable threshold τ ~ N(μ_τ, S_τ)
        # Initialise at 0 (equivalent to ReLU)
        self.mw = torch.zeros(num_features, device=self.device)
        self.Sw = torch.full((num_features,), 1e-3, device=self.device)
        self.has_bias = False

        # Saved for backward
        self.J = None  # Φ(α), Jacobian w.r.t. input
        self.Jtau = None  # 1−Φ(α), Jacobian w.r.t. τ
        self.input_shape = None

        # Parameter deltas
        self.delta_mw = None
        self.delta_Sw = None

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
        ma : Tensor (N, C, H, W)  post-activation means
        Sa : Tensor (N, C, H, W)  post-activation variances
        """
        self.input_shape = mz.shape
        N, C, H, W = mz.shape
        HW = H * W

        ma = torch.empty_like(mz)
        Sa = torch.empty_like(Sz)
        J = torch.empty_like(mz)
        Jtau = torch.empty_like(mz)

        total = N * C * HW
        grid = (triton.cdiv(total, BLOCK),)

        _tlu_fwd_kernel[grid](
            mz.contiguous(),
            Sz.contiguous(),
            self.mw,
            ma,
            Sa,
            J,
            Jtau,
            N,
            C,
            HW,
            BLOCK=BLOCK,
        )

        # Cauchy–Schwarz: |Cov(Z,A)| = |J·Sz| ≤ √(Sz·Sa)  →  J ≤ √(Sa/Sz)
        cs_bound = torch.sqrt(Sa / torch.clamp(Sz.contiguous(), min=1e-12))
        J = torch.minimum(J, cs_bound)

        self.J = J
        self.Jtau = Jtau

        return ma, Sa

    # ------------------------------------------------------------------
    #  Backward
    # ------------------------------------------------------------------
    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Propagate deltas back through TLU and compute τ parameter deltas.

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
        J = self.J
        Jtau = self.Jtau

        # ── Propagate deltas through activation ──
        delta_mz = delta_ma * J
        delta_Sz = delta_Sa * J * J

        # ── Parameter deltas for τ (cuTAGI convention) ──
        # ∂E[z]/∂τ = 1 − Φ(α) = Jtau, per element
        # Sum over (N, H, W) per channel
        Jtau_flat = Jtau.reshape(N, C, HW)
        dma_flat = delta_ma.reshape(N, C, HW)
        dSa_flat = delta_Sa.reshape(N, C, HW)

        grad_mw = (dma_flat * Jtau_flat).sum(dim=(0, 2))  # (C,)
        grad_Sw = (dSa_flat * Jtau_flat**2).sum(dim=(0, 2))  # (C,)

        self.delta_mw = self.Sw * grad_mw
        self.delta_Sw = (self.Sw**2) * grad_Sw

        return delta_mz, delta_Sz

    # ------------------------------------------------------------------
    #  Update
    # ------------------------------------------------------------------
    def update(self, cap_factor: float) -> None:
        """Apply stored parameter deltas with cuTAGI-style capping."""
        update_parameters(self.mw, self.Sw, self.delta_mw, self.delta_Sw, cap_factor)

    @property
    def num_parameters(self) -> int:
        """Total learnable scalars: 2 × threshold (mean + variance)."""
        return 2 * self.mw.numel()

    def __repr__(self):
        return f"TLU(num_features={self.num_features})"
