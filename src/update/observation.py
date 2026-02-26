"""
Observation update — compute the innovation signal at the output layer.

Given observations y, predicted mean μ_z and predicted variance S_z,
the TAGI update computes:
    δ_μ = (y − μ_z) / (S_z + σ_v²)
    δ_S = −1      / (S_z + σ_v²)

These deltas are then back-propagated through the network.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024


# ======================================================================
#  Triton kernel
# ======================================================================

@triton.jit
def _output_innovation_kernel(
    y_ptr, ym_ptr, yS_ptr, sv_sq,
    dm_ptr, dS_ptr, n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    y  = tl.load(y_ptr  + offs, mask=valid)
    ym = tl.load(ym_ptr + offs, mask=valid)
    yS = tl.load(yS_ptr + offs, mask=valid)

    Sy = yS + sv_sq

    tl.store(dm_ptr + offs, (y - ym) / Sy, mask=valid)
    tl.store(dS_ptr + offs, -1.0 / Sy,     mask=valid)


# ======================================================================
#  Python API
# ======================================================================

def compute_innovation(y, y_pred_mu, y_pred_var, sigma_v):
    """
    Compute the output innovation (update signal) for TAGI.

    Parameters
    ----------
    y          : Tensor (B, D)  observed targets
    y_pred_mu  : Tensor (B, D)  predicted output mean
    y_pred_var : Tensor (B, D)  predicted output variance
    sigma_v    : float          observation noise std-dev

    Returns
    -------
    delta_mu  : Tensor (B, D)  mean innovation
    delta_var : Tensor (B, D)  variance innovation
    """
    n = y.numel()
    delta_mu  = torch.empty_like(y)
    delta_var = torch.empty_like(y)

    _output_innovation_kernel[(triton.cdiv(n, BLOCK),)](
        y, y_pred_mu, y_pred_var,
        sigma_v ** 2,
        delta_mu, delta_var,
        n, BLOCK=BLOCK,
    )
    return delta_mu, delta_var
