"""
Parameter update — capped Bayesian update matching cuTAGI.

The update rule is:
    delta_bar  = √S / cap_factor       (adaptive cap per-parameter)
    m_new = m + sign(Δ_μ) · min(|Δ_μ|, delta_bar)
    S_new = S + sign(Δ_S) · min(|Δ_S|, delta_bar)   (if result > 0)
    S_new = 1e-5                                       (only if result ≤ 0)

Cap factor is a heuristic that regularises updates for larger batches:
    batch == 1:    cap_factor = 0.1
    1 < batch < 256:  cap_factor = 2.0
    batch >= 256:  cap_factor = 3.0

This is a general-purpose function — it works on any parameter tensor
(weights, biases, or any future learnable parameters).
"""

import triton
import triton.language as tl

BLOCK = 1024


# ======================================================================
#  Cap-factor heuristic (matches cuTAGI)
# ======================================================================


def get_cap_factor(batch_size: int) -> float:
    """
    Get the cap factor for regularising parameter updates.

    Based on empirical tuning in cuTAGI — larger batches need stronger
    regularisation to prevent overshooting.

    Parameters
    ----------
    batch_size : int

    Returns
    -------
    cap_factor : float
    """
    if batch_size == 1:
        return 0.1
    elif batch_size < 256:
        return 2.0
    else:
        return 3.0


# ======================================================================
#  Triton kernel — capped parameter update
# ======================================================================


@triton.jit
def _capped_param_update_kernel(
    m_ptr,
    S_ptr,
    dm_ptr,
    dS_ptr,
    cap_factor,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    m = tl.load(m_ptr + offs, mask=valid)
    S = tl.load(S_ptr + offs, mask=valid)
    dm = tl.load(dm_ptr + offs, mask=valid)
    dS = tl.load(dS_ptr + offs, mask=valid)

    # Adaptive cap: delta_bar = sqrt(S) / cap_factor
    delta_bar = tl.sqrt(tl.maximum(S, 1e-10)) / cap_factor

    # ── Capped mean update ──
    dm_sign = tl.where(dm > 0.0, 1.0, tl.where(dm < 0.0, -1.0, 0.0))
    dm_capped = dm_sign * tl.minimum(tl.abs(dm), delta_bar)
    m_new = m + dm_capped

    # ── Capped variance update ──
    # cuTAGI floors S at 1e-5 only when the update would make it non-positive
    # (base_layer.cpp: `if (var_w[i] <= 0.0f) var_w[i] = 1E-5f`).
    # An unconditional floor prevents S from shrinking below 1e-5 when it should
    # reach ~1e-8, causing 7× larger mw updates than cuTAGI and training instability.
    dS_sign = tl.where(dS > 0.0, 1.0, tl.where(dS < 0.0, -1.0, 0.0))
    dS_capped = dS_sign * tl.minimum(tl.abs(dS), delta_bar)
    S_raw = S + dS_capped
    S_new = tl.where(S_raw <= 0.0, 1e-5, S_raw)

    tl.store(m_ptr + offs, m_new, mask=valid)
    tl.store(S_ptr + offs, S_new, mask=valid)


# ======================================================================
#  Python API
# ======================================================================


def update_parameters(m, S, delta_m, delta_S, cap_factor):
    """
    In-place capped Bayesian parameter update (matches cuTAGI).

    Each update is independently capped at delta_bar = √S / cap_factor,
    preventing large updates when the batch is large.

    Parameters
    ----------
    m          : Tensor  parameter means    (modified in-place)
    S          : Tensor  parameter variances (modified in-place)
    delta_m    : Tensor  mean deltas    (Sw * grad_m)
    delta_S    : Tensor  variance deltas (Sw² * grad_S)
    cap_factor : float   regularisation strength
    """
    n = m.numel()
    _capped_param_update_kernel[(triton.cdiv(n, BLOCK),)](
        m.view(-1),
        S.view(-1),
        delta_m.view(-1),
        delta_S.view(-1),
        cap_factor,
        n,
        BLOCK=BLOCK,
    )
