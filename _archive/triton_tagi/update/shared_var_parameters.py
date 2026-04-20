"""
Shared-variance parameter update for TAGI.

Instead of maintaining a full variance tensor Sw (same shape as mw),
each layer has a single scalar variance sw and sb.  The update rule
aggregates per-parameter variance gradients into a single scalar update.

Update rule (precision-space):
    avg_grad = (1/P) Σ_i grad_S[i]       average over all parameters
    sw_new   = sw / (1 - sw * avg_grad)   precision-space update

The capped mean update reuses the standard `update_parameters` kernel
from update/parameters.py, with the scalar sw broadcast to all parameters
via a temporary expansion.  This keeps the mean update capping identical.
"""

import triton
import triton.language as tl

BLOCK = 1024


# ======================================================================
#  Scalar variance update (pure PyTorch — scalar ops)
# ======================================================================


def update_shared_variance(sw, grad_S_flat):
    """
    Update a scalar shared variance using aggregated per-parameter gradients.

    The effective precision update averages the per-parameter precision
    increments:
        avg_grad = (1/P) Σ_i grad_S[i]       where P = #parameters
        sw_new   = sw / (1 - sw · avg_grad)

    Averaging (rather than summing) prevents collapse when P is large.

    Parameters
    ----------
    sw           : scalar Tensor  (shape ())  — updated in-place
    grad_S_flat  : Tensor (any shape)          — per-parameter variance grads
    """
    n_params = grad_S_flat.numel()
    avg_grad = grad_S_flat.sum().item() / n_params
    sw_old = sw.item()
    denom = 1.0 - sw_old * avg_grad
    denom = max(denom, 0.01)  # prevent negative / explosion
    sw_new = sw_old / denom
    sw_new = max(sw_new, 1e-8)  # floor
    sw.fill_(sw_new)


# ======================================================================
#  Triton kernel — capped mean-only update (shared variance broadcast)
# ======================================================================


@triton.jit
def _capped_mean_update_kernel(
    m_ptr,
    dm_ptr,
    sw,  # scalar float — the shared variance
    cap_factor,
    n_elements,
    BLOCK: tl.constexpr,
):
    """
    Update means with capping, using a scalar variance for delta_bar.

    delta_bar = sqrt(sw) / cap_factor   (same for ALL parameters)
    m_new = m + sign(dm) * min(|dm|, delta_bar)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    m = tl.load(m_ptr + offs, mask=valid)
    dm = tl.load(dm_ptr + offs, mask=valid)

    # Shared delta_bar for the whole layer
    delta_bar = tl.sqrt(tl.maximum(sw, 1e-10)) / cap_factor

    dm_sign = tl.where(dm > 0.0, 1.0, tl.where(dm < 0.0, -1.0, 0.0))
    dm_capped = dm_sign * tl.minimum(tl.abs(dm), delta_bar)
    m_new = m + dm_capped

    tl.store(m_ptr + offs, m_new, mask=valid)


def update_shared_mean(m, delta_m, sw_scalar, cap_factor):
    """
    In-place capped mean update using a shared scalar variance.

    Parameters
    ----------
    m          : Tensor  parameter means    (modified in-place)
    delta_m    : Tensor  mean deltas        (sw * grad_m, with sw already applied)
    sw_scalar  : float   the shared variance value  (for computing delta_bar)
    cap_factor : float   regularisation strength
    """
    n = m.numel()
    _capped_mean_update_kernel[(triton.cdiv(n, BLOCK),)](
        m.view(-1),
        delta_m.view(-1),
        sw_scalar,
        cap_factor,
        n,
        BLOCK=BLOCK,
    )
