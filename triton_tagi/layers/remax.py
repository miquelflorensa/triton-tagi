"""
Remax activation layer for TAGI.

Remax = ReLU + Normalization (softmax alternative for Bayesian networks).

Given logits Z ~ N(μ_z, S_z), computes output probabilities A = M / Σ M
where M_k = max(0, Z_k) (rectified Gaussian).

The forward pass computes:
    μ_a   : mean probabilities  (sum to 1)
    S_a   : variance of probabilities
    cov_z_a : covariance between logits Z and probabilities A

The backward Jacobian is:  J_k = cov(z_k, a_k) / S_z_k
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from ..base import Layer

# ======================================================================
#  Triton kernel — Remax moments
# ======================================================================


@triton.jit
def _remax_kernel(
    mu_z_ptr,
    sig_z_sq_ptr,  # (B, K) inputs
    mu_a_ptr,
    sig_a_sq_ptr,  # (B, K) outputs
    cov_z_a_ptr,  # (B, K) output
    K,  # number of classes [runtime]
    stride_b,  # row stride
    BLOCK_K: tl.constexpr,  # >= K, power-of-2
):
    """One program ↔ one batch item.  All K classes in registers."""

    b = tl.program_id(0)
    base = b * stride_b
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    EPS: tl.constexpr = 1e-9

    # ── load inputs ──
    mu_z = tl.load(mu_z_ptr + base + offs, mask=mask, other=0.0)
    sig_z_sq = tl.load(sig_z_sq_ptr + base + offs, mask=mask, other=0.0)

    # # ── alpha = mu / sigma,  phi(alpha), Phi(alpha) ──
    # alpha = mu_z / safe_sig
    # phi_a = tl.exp(-0.5 * alpha * alpha) * INV_SQRT_2PI
    # Phi_a = 0.5 + 0.5 * tl.math.erf(alpha * INV_SQRT_2)

    # # ── 1. ReLU moments  M = max(0, Z) ──
    # mu_m     = tl.maximum(sig_z * phi_a + mu_z * Phi_a, EPS)
    # sig_m_sq = tl.maximum(
    #     (mu_z * mu_z + sig_z_sq) * Phi_a + mu_z * sig_z * phi_a - mu_m * mu_m,
    #     EPS)
    # cov_z_m  = sig_z_sq * Phi_a

    # ── 1. Softplus moments M = Softplus(Z) ──
    # Stable Sigmoid (Triton has a built-in for this which handles extremes)
    tmp = tl.sigmoid(mu_z)

    # Stable Softplus: log(1 + exp(x)) is practically just x for large positive x.
    # We switch to linear at mu_z > 20.0 to prevent tl.exp() from overflowing to inf.
    softplus_base = tl.where(mu_z > 20.0, mu_z, tl.log(1.0 + tl.exp(mu_z)))

    # Second-order correction to the mean
    mu_m = softplus_base + 0.5 * sig_z_sq * tmp * (1.0 - tmp)

    # CRITICAL: Hard floor mu_m to EPS. If it hits exactly 0, log(mu_m) becomes -inf
    # and division by (mu_m * mu_m) yields inf.
    mu_m = tl.maximum(mu_m, EPS)

    # Covariance and Variance
    cov_z_m = tmp * sig_z_sq
    sig_m_sq = tl.maximum(tmp * sig_z_sq * tmp, EPS)  # Floor this too

    # Cauchy–Schwarz on intermediate cov(Z, M): |cov_z_m| ≤ √(Sz · Sm)
    cs_bound_zm = tl.sqrt(sig_z_sq * sig_m_sq)
    cov_z_m = tl.minimum(tl.maximum(cov_z_m, -cs_bound_zm), cs_bound_zm)

    # # ── 1. Gaussian CDF (Probit) moments M = Phi(Z) ──
    # # First, we calculate the variance-adjusted mean (alpha).
    # # E[Phi(Z)] = Phi(mu / sqrt(1 + sigma^2))
    # denom = tl.sqrt(1.0 + sig_z_sq)
    # alpha = mu_z / denom

    # # Mean of M:
    # mu_m = 0.5 + 0.5 * tl.math.erf(alpha * INV_SQRT_2)
    # # CRITICAL: CDF can hit exactly 0.0 in float32. We must floor it so log(mu_m) survives.
    # mu_m = tl.maximum(mu_m, EPS)

    # # Covariance cov(Z, M): derived via Stein's Lemma -> E[f'(Z)] * Var(Z)
    # # f'(Z) is the Gaussian PDF (phi)
    # phi_alpha = INV_SQRT_2PI * tl.exp(-0.5 * alpha * alpha)
    # cov_z_m = sig_z_sq * phi_alpha / denom
    # sig_m_sq = tl.maximum((sig_z_sq * phi_alpha * phi_alpha) / (1.0 + sig_z_sq), EPS)

    # ── 2. log-space moments of M ──
    sig_ln_m_sq = tl.log(1.0 + sig_m_sq / (mu_m * mu_m))
    mu_ln_m = tl.log(mu_m) - 0.5 * sig_ln_m_sq

    # ── 3. sum moments  M̃ = Σ M_k ──
    mu_mt = tl.sum(tl.where(mask, mu_m, 0.0))
    sig_mt_sq = tl.sum(tl.where(mask, sig_m_sq, 0.0))

    # ── 4. log-space moments of M̃ ──
    sig_ln_mt_sq = tl.log(1.0 + sig_mt_sq / (mu_mt * mu_mt))
    mu_ln_mt = tl.log(mu_mt) - 0.5 * sig_ln_mt_sq

    # ── 5. cov in log-space ──
    cov_ln = tl.log(1.0 + sig_m_sq / (mu_m * mu_mt))

    # ── 6. moments of ln(A) = ln(M) - ln(M̃) ──
    mu_ln_a = mu_ln_m - mu_ln_mt
    sig_ln_a_sq = tl.maximum(sig_ln_m_sq + sig_ln_mt_sq - 2.0 * cov_ln, EPS)

    # ── 7. final moments of A ──
    mu_a = tl.maximum(tl.exp(mu_ln_a + 0.5 * sig_ln_a_sq), EPS)
    mu_a_sum = tl.maximum(tl.sum(tl.where(mask, mu_a, 0.0)), EPS)
    mu_a = mu_a / mu_a_sum

    sig_a_sq = mu_a * mu_a * (tl.exp(sig_ln_a_sq) - 1.0)

    # ── 8. cov(Z, A) ──
    cov_z_a = mu_a * cov_z_m * (1.0 / mu_m - 1.0 / mu_mt)

    # Cauchy–Schwarz on cov(Z, A): |cov_z_a| ≤ √(Sz · Sa)
    cs_bound_za = tl.sqrt(sig_z_sq * sig_a_sq)
    cov_z_a = tl.minimum(tl.maximum(cov_z_a, -cs_bound_za), cs_bound_za)

    # ── store ──
    tl.store(mu_a_ptr + base + offs, mu_a, mask=mask)
    tl.store(sig_a_sq_ptr + base + offs, sig_a_sq, mask=mask)
    tl.store(cov_z_a_ptr + base + offs, cov_z_a, mask=mask)


# ======================================================================
#  Python wrapper
# ======================================================================


def triton_remax(mu_z, sigma_z_sq):
    """
    Compute Remax moments using fused Triton kernel.

    Parameters
    ----------
    mu_z       : Tensor (B, K)  logit means
    sigma_z_sq : Tensor (B, K)  logit variances

    Returns
    -------
    mu_a       : Tensor (B, K)  probability means (sum to 1)
    sigma_a_sq : Tensor (B, K)  probability variances
    cov_z_a    : Tensor (B, K)  covariance between logits and probabilities
    """
    squeeze = mu_z.dim() == 1
    if squeeze:
        mu_z = mu_z.unsqueeze(0)
        sigma_z_sq = sigma_z_sq.unsqueeze(0)

    mu_z = mu_z.contiguous()
    sigma_z_sq = sigma_z_sq.contiguous()
    B, K = mu_z.shape

    mu_a = torch.empty_like(mu_z)
    sig_a_sq = torch.empty_like(mu_z)
    cov_z_a = torch.empty_like(mu_z)

    BLOCK_K = triton.next_power_of_2(K)
    _remax_kernel[(B,)](
        mu_z,
        sigma_z_sq,
        mu_a,
        sig_a_sq,
        cov_z_a,
        K,
        mu_z.stride(0),
        BLOCK_K=BLOCK_K,
    )

    if squeeze:
        return mu_a.squeeze(0), sig_a_sq.squeeze(0), cov_z_a.squeeze(0)
    return mu_a, sig_a_sq, cov_z_a


# ======================================================================
#  Remax Layer
# ======================================================================


class Remax(Layer):
    """
    Remax activation layer (softmax alternative for Bayesian networks).

    Forward:  (μ_z, S_z) → (μ_a, S_a)  with μ_a ∈ simplex
    Backward: uses J_k = cov(z_k, a_k) / S_z_k as Jacobian

    Stores the Jacobian from the forward pass for use during backward.
    """

    def __init__(self) -> None:
        self.J = None  # stored Jacobian

    def forward(self, mz: Tensor, Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute Remax output moments.

        Parameters
        ----------
        mz : Tensor (B, K)  logit means
        Sz : Tensor (B, K)  logit variances

        Returns
        -------
        mu_a : Tensor (B, K)  probability means   (sum ≈ 1)
        Sa   : Tensor (B, K)  probability variances
        """
        mu_a, Sa, cov_z_a = triton_remax(mz, Sz)

        # Apply Cauchy–Schwarz inequality to cov_z_a
        cov_z_a = torch.clamp(cov_z_a, -torch.sqrt(Sz * Sa), torch.sqrt(Sz * Sa))

        # Jacobian: J_k = cov(z_k, a_k) / Sz_k
        self.J = cov_z_a / torch.clamp(Sz, min=1e-7)

        return mu_a, Sa

    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Propagate deltas from probability space back to logit space.

        Uses the linear approximation: a_k ≈ μ_a_k + J_k · (z_k − μ_z_k)

        Parameters
        ----------
        delta_ma : Tensor (B, K)  mean delta in probability space
        delta_Sa : Tensor (B, K)  variance delta in probability space

        Returns
        -------
        delta_mz : Tensor (B, K)  mean delta in logit space
        delta_Sz : Tensor (B, K)  variance delta in logit space
        """
        J = self.J
        return delta_ma * J, delta_Sa * J * J

    def __repr__(self):
        return "Remax()"
