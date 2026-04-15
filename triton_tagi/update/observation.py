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
    y_ptr,
    ym_ptr,
    yS_ptr,
    sv_sq,
    dm_ptr,
    dS_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    y = tl.load(y_ptr + offs, mask=valid)
    ym = tl.load(ym_ptr + offs, mask=valid)
    yS = tl.load(yS_ptr + offs, mask=valid)

    Sy = yS + sv_sq

    tl.store(dm_ptr + offs, (y - ym) / Sy, mask=valid)
    tl.store(dS_ptr + offs, -1.0 / Sy, mask=valid)


@triton.jit
def _output_innovation_kernel_heteros(
    y_ptr,
    ym_ptr,
    yS_ptr,
    dm_ptr,
    dS_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    y = tl.load(y_ptr + offs, mask=valid)

    # Output layer has twice the size
    obs_col = offs * 2

    # mean of the Gaussian distribution for the output
    mu_a_col = tl.load(ym_ptr + obs_col, mask=valid)
    var_a_col = tl.load(yS_ptr + obs_col, mask=valid)

    # V2_bar_tilde
    mu_v2_bar_tilde = tl.load(ym_ptr + obs_col + 1, mask=valid)
    var_v2_bar_tilde = tl.load(yS_ptr + obs_col + 1, mask=valid)

    # Compute the prior predictive PDF for v2
    mu_v2 = mu_v2_bar_tilde
    var_v2 = 3.0 * var_v2_bar_tilde + 2.0 * mu_v2_bar_tilde * mu_v2_bar_tilde
    cov_y_v = mu_v2

    # Variance of the output
    var_sum = var_a_col + mu_v2

    # Compute updating quantities for the mean of the output
    tmp = 1.0 / var_sum
    obs_diff = y - mu_a_col
    delta_mu_col = tmp * obs_diff
    delta_var_col = -tmp

    # Compute the posterior mean and variance for V
    mu_v_post = cov_y_v / var_sum * obs_diff
    var_v_post = mu_v2 - cov_y_v / var_sum * cov_y_v

    # Compute the posterior mean and variance for V2
    mu_v2_post = mu_v_post * mu_v_post + var_v_post
    var_v2_post = 2.0 * var_v_post * var_v_post + 4.0 * var_v_post * mu_v_post * mu_v_post

    # Compute the posterior mean and variance for V2_bar_tilde
    tmp_ratio = var_v2_bar_tilde / var_v2
    mu_v2_bar_tilde_post = mu_v2_bar_tilde + tmp_ratio * (mu_v2_post - mu_v2)
    var_v2_bar_tilde_post = var_v2_bar_tilde + tmp_ratio * tmp_ratio * (var_v2_post - var_v2)

    # Compute update for V2_bar
    delta_mu_v2 = mu_v2_bar_tilde_post - mu_v2_bar_tilde
    delta_var_v2 = var_v2_bar_tilde_post - var_v2_bar_tilde

    tl.store(dm_ptr + obs_col, delta_mu_col, mask=valid)
    tl.store(dS_ptr + obs_col, delta_var_col, mask=valid)

    tl.store(dm_ptr + obs_col + 1, delta_mu_v2, mask=valid)
    tl.store(dS_ptr + obs_col + 1, delta_var_v2, mask=valid)


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

    if y_pred_mu.shape[-1] == 2 * y.shape[-1]:
        delta_mu = torch.empty_like(y_pred_mu)
        delta_var = torch.empty_like(y_pred_var)
        _output_innovation_kernel_heteros[(triton.cdiv(n, BLOCK),)](
            y,
            y_pred_mu,
            y_pred_var,
            delta_mu,
            delta_var,
            n,
            BLOCK=BLOCK,
        )
    else:
        delta_mu = torch.empty_like(y)
        delta_var = torch.empty_like(y)
        _output_innovation_kernel[(triton.cdiv(n, BLOCK),)](
            y,
            y_pred_mu,
            y_pred_var,
            sigma_v**2,
            delta_mu,
            delta_var,
            n,
            BLOCK=BLOCK,
        )

    return delta_mu, delta_var
