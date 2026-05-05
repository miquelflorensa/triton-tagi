"""
Remax activation layer (corrected chain-rule projection).

This is a parallel implementation of ``triton_tagi.layers.remax.Remax`` that
replaces the kernel's ``cov_a_z = cov_a_m / Phi(alpha)`` (cuTAGI parity) with
the principled chain-rule projection

    Cov(Z_i, A_i) = Cov(A_i, M_i) * Cov(Z_i, M_i) / Var(M_i)
                  = Cov(A_i, M_i) * sigma_Z^2 * Phi(alpha) / Var(M_i).

Derivation: by the law of total covariance with Z_i independent of M_j (j != i)
and the linearisation A_i ≈ mu_A + (∂A/∂Z) * (Z - mu_Z),

    Cov(Z, A) ≈ Var(Z) * E[∂A/∂Z]
              = Var(Z) * E[∂A/∂M * ∂M/∂Z]
              ≈ Var(Z) * Phi(alpha) * Cov(A, M) / Var(M)
              = Cov(A, M) * Cov(Z, M) / Var(M).

The required projection coefficient is the regression of Z on M
(beta_{Z|M} = Cov(Z, M) / Var(M)), not the regression of M on Z used in
TRTCT.tex eq. 240. The corrected formula has no 1/Phi(alpha) singularity:
for inactive units (Phi(alpha) -> 0), Var(M) vanishes at the same rate as
Cov(Z, M), so their ratio remains O(1) and Cov(Z, A) -> 0 cleanly via the
shrinking Cov(A, M).

No Cauchy-Schwarz clip is applied here — the violation experiment in
experiments/cov_violation_mnist.py measures how often (if ever) the
unclipped corrected formula exceeds the CS bound.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from ..base import Layer


@triton.jit
def _remax_chain_kernel(
    mu_z_ptr,
    var_z_ptr,            # (B, K) inputs
    mu_a_ptr,
    var_a_ptr,            # (B, K) outputs
    J_ptr,                # (B, K) Jacobian cov(A,Z)/var(Z)
    K,
    stride_b,
    BLOCK_K: tl.constexpr,
):
    """One program <-> one batch item. All K classes in registers.

    Identical to ``_remax_kernel`` (remax.py) except for the final two lines:
    the Cov(Z, A) projection uses the corrected chain-rule coefficient
    Cov(Z, M) / Var(M), and there is no CS clip.
    """
    b = tl.program_id(0)
    base = b * stride_b
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    EPS: tl.constexpr = 1e-6
    INV_SQRT_2PI: tl.constexpr = 0.3989422804014327
    INV_SQRT_2:   tl.constexpr = 0.7071067811865475

    mu_z = tl.load(mu_z_ptr + base + offs, mask=mask, other=0.0)
    var_z = tl.load(var_z_ptr + base + offs, mask=mask, other=1.0)

    # MixtureReLU moments of M = max(0, Z)
    std_z = tl.sqrt(var_z)
    alpha = mu_z / std_z
    pdf_alpha = INV_SQRT_2PI * tl.exp(-0.5 * alpha * alpha)
    cdf_alpha = 0.5 + 0.5 * tl.math.erf(alpha * INV_SQRT_2)

    mu_m = mu_z * cdf_alpha + std_z * pdf_alpha
    mu_m = tl.maximum(mu_m, EPS)

    var_m = (
        -mu_m * mu_m
        + 2.0 * mu_m * mu_z
        - mu_z * std_z * pdf_alpha
        + (var_z - mu_z * mu_z) * cdf_alpha
    )
    var_m = tl.maximum(var_m, EPS)
    jcb_m = cdf_alpha                       # = Phi(alpha)

    # Sum moments
    mu_mt  = tl.maximum(tl.sum(tl.where(mask, mu_m,  0.0)), EPS)
    var_mt = tl.maximum(tl.sum(tl.where(mask, var_m, 0.0)), EPS)

    # Log-normal moments
    var_log_m  = tl.log(1.0 + var_m  / (mu_m  * mu_m))
    mu_log_m   = tl.log(mu_m)  - 0.5 * var_log_m
    var_log_mt = tl.log(1.0 + var_mt / (mu_mt * mu_mt))
    mu_log_mt  = tl.log(mu_mt) - 0.5 * var_log_mt

    cov_log_m_mt = tl.log(1.0 + var_m / (mu_m * mu_mt))

    mu_log_a  = mu_log_m - mu_log_mt
    var_log_a = tl.maximum(var_log_m + var_log_mt - 2.0 * cov_log_m_mt, 0.0)

    mu_a_raw = tl.maximum(tl.exp(mu_log_a + 0.5 * var_log_a), EPS)
    sum_mu_a = tl.maximum(tl.sum(tl.where(mask, mu_a_raw, 0.0)), EPS)
    mu_a     = mu_a_raw / sum_mu_a
    var_a    = (tl.exp(var_log_a) - 1.0) * mu_a * mu_a

    # Cov(A, M) via log-normal identity (same as cuTAGI parity kernel)
    cov_log_a_log_m = var_log_m - cov_log_m_mt
    cov_a_m = (tl.exp(cov_log_a_log_m) - 1.0) * mu_a * mu_m

    # ── Corrected chain-rule projection ──────────────────────────────────
    # Cov(Z, A) = Cov(A, M) * Cov(Z, M) / Var(M)
    #           = Cov(A, M) * (var_z * Phi(alpha)) / var_m
    # No CS clip: measure violations directly in the diagnostic.
    cov_z_m = var_z * jcb_m
    cov_a_z = cov_a_m * cov_z_m / var_m
    J       = cov_a_z / var_z                # = cov_a_m * jcb_m / var_m

    tl.store(mu_a_ptr + base + offs, mu_a, mask=mask)
    tl.store(var_a_ptr + base + offs, var_a, mask=mask)
    tl.store(J_ptr     + base + offs, J,    mask=mask)


def triton_remax_chain(mu_z: Tensor, var_z: Tensor):
    """Corrected-chain-rule Remax forward.

    Returns (mu_a, var_a, J) with the same shapes/conventions as
    ``triton_tagi.layers.remax.triton_remax``.
    """
    squeeze = mu_z.dim() == 1
    if squeeze:
        mu_z = mu_z.unsqueeze(0)
        var_z = var_z.unsqueeze(0)

    mu_z = mu_z.contiguous()
    var_z = var_z.contiguous()
    B, K = mu_z.shape

    mu_a  = torch.empty_like(mu_z)
    var_a = torch.empty_like(mu_z)
    J     = torch.empty_like(mu_z)

    BLOCK_K = triton.next_power_of_2(K)
    _remax_chain_kernel[(B,)](
        mu_z, var_z, mu_a, var_a, J,
        K, mu_z.stride(0),
        BLOCK_K=BLOCK_K,
    )

    if squeeze:
        return mu_a.squeeze(0), var_a.squeeze(0), J.squeeze(0)
    return mu_a, var_a, J


class RemaxChain(Layer):
    """Drop-in replacement for ``Remax`` using the corrected chain rule."""

    def __init__(self) -> None:
        self.J: Tensor | None = None

    def forward(self, mz: Tensor, Sz: Tensor) -> tuple[Tensor, Tensor]:
        mu_a, Sa, J = triton_remax_chain(mz, Sz)
        self.J = J
        return mu_a, Sa

    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        J = self.J
        return delta_ma * J, delta_Sa * J * J

    def __repr__(self) -> str:
        return "RemaxChain()"
