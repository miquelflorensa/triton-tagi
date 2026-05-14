"""
Parameter update — supports multiple TAGI update rules.

Available rules:

    additive
        m_new = m + dm
        S_new = S + dS                 (floored at 1e-5 if non-positive)

    capped_additive  (cuTAGI baseline)
        delta_bar = sqrt(S) / cap_factor
        m_new = m + sign(dm) * min(|dm|, delta_bar)
        S_new = S + sign(dS) * min(|dS|, delta_bar)   (floored at 1e-5 if <= 0)

    precision_normalized   (PN-TAGI, rho = 1)
    tempered_precision_normalized   (PN-TAGI, rho < 1)
        chi_raw   = -dS / max(S, eps)
        chi       = max(chi_raw, 0)
        d         = 1 + rho * chi
        m_new     = m + rho * dm / d
        S_new     = S / d

    capped_precision_normalized   (PN-TAGI variance + capped mean step)
        delta_bar = sqrt(S) / cap_factor
        dm_capped = sign(dm) * min(|dm|, delta_bar)        # cap raw dm
        chi       = max(0, -dS / max(S, eps))
        d         = 1 + rho * chi
        m_new     = m + rho * dm_capped / d                # cap THEN dampen
        S_new     = S / d
    Designed to keep PN-TAGI's variance-contraction geometry while
    inheriting the mean-step safety brake of the cap-factor heuristic.
    Use when σ_v is small enough that PN-TAGI alone overshoots μ on
    the first batch (Stage 3 finding).

Cap factor heuristic (only used by capped_additive):
    batch == 1:        cap_factor = 0.1
    1 < batch < 256:   cap_factor = 2.0
    batch >= 256:      cap_factor = 3.0

Diagnostics: if ``record_chi=True``, the kernel writes ``chi_raw`` (the
posterior-contraction ratio defined in the PN-TAGI plan) into a buffer of
the same shape as ``S``. The buffer is returned so the caller can aggregate
median / p95 / max / fraction>0.1 / fraction>1.0 statistics off the hot path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 1024

# ---------------------------------------------------------------------------
# Update-rule encoding
# ---------------------------------------------------------------------------
# Triton kernels branch on tl.constexpr ints — strings are mapped here once
# per launch so each rule compiles to its own specialised kernel.

_RULE_ADDITIVE = 0
_RULE_CAPPED = 1
_RULE_PRECISION_NORM = 2  # also covers tempered_precision_normalized (just rho < 1)
_RULE_CAPPED_PN = 3       # capped mean step + PN variance contraction

_RULE_TO_INT = {
    "additive": _RULE_ADDITIVE,
    "capped_additive": _RULE_CAPPED,
    "precision_normalized": _RULE_PRECISION_NORM,
    "tempered_precision_normalized": _RULE_PRECISION_NORM,
    "capped_precision_normalized": _RULE_CAPPED_PN,
}

VALID_RULES = tuple(_RULE_TO_INT.keys())

# Small epsilon for the chi denominator (inlined inside the kernel below as a
# literal — Triton only allows ``tl.constexpr(...)`` globals to be referenced
# from @jit functions, and a single literal is simpler). The prior variance
# can legitimately shrink to ~1e-8 under PN-TAGI; 1e-12 keeps the diagnostic
# finite without masking meaningful signal.


# ======================================================================
#  Cap-factor heuristic (matches cuTAGI)
# ======================================================================


def get_cap_factor(batch_size: int) -> float:
    """
    Get the cap factor for regularising parameter updates.

    Only used by the ``capped_additive`` rule. Other rules ignore this value
    but still accept it positionally so callers don't have to branch.

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
#  Triton kernel — unified update with optional chi diagnostic
# ======================================================================


@triton.jit
def _param_update_kernel(
    m_ptr,
    S_ptr,
    dm_ptr,
    dS_ptr,
    chi_ptr,
    cap_factor,
    rho,
    n_elements,
    RULE: tl.constexpr,
    WRITE_CHI: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    m = tl.load(m_ptr + offs, mask=valid)
    S = tl.load(S_ptr + offs, mask=valid)
    dm = tl.load(dm_ptr + offs, mask=valid)
    dS = tl.load(dS_ptr + offs, mask=valid)

    # Raw posterior-contraction ratio: fraction of prior variance the
    # additive update tries to consume. Computed before the update so it
    # reflects the *prior* variance, regardless of which rule fires.
    S_safe = tl.maximum(S, 1e-12)
    raw_chi = -dS / S_safe

    if WRITE_CHI:
        tl.store(chi_ptr + offs, raw_chi, mask=valid)

    if RULE == 0:  # additive
        m_new = m + dm
        S_raw = S + dS
        # Numerical floor only — keeps S strictly positive when the additive
        # update would otherwise land at or below zero (the "consumed > 100%
        # of prior" pathology PN-TAGI is designed to avoid).
        S_new = tl.where(S_raw <= 0.0, 1e-5, S_raw)
    elif RULE == 1:  # capped_additive (cuTAGI baseline)
        delta_bar = tl.sqrt(tl.maximum(S, 1e-10)) / cap_factor

        dm_sign = tl.where(dm > 0.0, 1.0, tl.where(dm < 0.0, -1.0, 0.0))
        dm_capped = dm_sign * tl.minimum(tl.abs(dm), delta_bar)
        m_new = m + dm_capped

        dS_sign = tl.where(dS > 0.0, 1.0, tl.where(dS < 0.0, -1.0, 0.0))
        dS_capped = dS_sign * tl.minimum(tl.abs(dS), delta_bar)
        S_raw = S + dS_capped
        S_new = tl.where(S_raw <= 0.0, 1e-5, S_raw)
    elif RULE == 2:  # precision_normalized / tempered_precision_normalized
        # chi clipped at zero so positive variance increments (dS > 0) do not
        # *inflate* posterior variance via the PN denominator. They are still
        # surfaced through raw_chi so the diagnostic preserves them.
        chi = tl.maximum(raw_chi, 0.0)
        d = 1.0 + rho * chi
        m_new = m + rho * dm / d
        S_new = S / d
    else:  # RULE == 3: capped_precision_normalized
        # cap-factor brake on the raw dm THEN PN dampening. The variance
        # update is identical to plain PN-TAGI. Designed for the regime
        # where PN-TAGI alone overshoots μ on the first batch because
        # chi is small while dm is large (small Sw, small σ_v).
        delta_bar = tl.sqrt(tl.maximum(S, 1e-10)) / cap_factor
        dm_sign = tl.where(dm > 0.0, 1.0, tl.where(dm < 0.0, -1.0, 0.0))
        dm_capped = dm_sign * tl.minimum(tl.abs(dm), delta_bar)
        chi = tl.maximum(raw_chi, 0.0)
        d = 1.0 + rho * chi
        m_new = m + rho * dm_capped / d
        S_new = S / d

    tl.store(m_ptr + offs, m_new, mask=valid)
    tl.store(S_ptr + offs, S_new, mask=valid)


# ======================================================================
#  Python API
# ======================================================================


def update_parameters(
    m,
    S,
    delta_m,
    delta_S,
    cap_factor: float,
    update_rule: str = "capped_additive",
    rho: float = 1.0,
    chi_out=None,
):
    """
    In-place TAGI parameter update.

    Parameters
    ----------
    m, S          : Tensor   parameter means / variances (modified in-place)
    delta_m       : Tensor   mean delta    (Sw * grad_m), same shape as m
    delta_S       : Tensor   variance delta (Sw² * grad_S), same shape as S
    cap_factor    : float    only used when update_rule == "capped_additive"
    update_rule   : str      one of VALID_RULES
    rho           : float    temperature for precision-normalised rules (1.0 default)
    chi_out       : Tensor or None
        Optional same-shape buffer to receive raw_chi = -delta_S / max(S, eps).
        If provided, populated regardless of update_rule. If None, no
        diagnostic is written.
    """
    if update_rule not in _RULE_TO_INT:
        raise ValueError(
            f"update_rule must be one of {VALID_RULES}, got {update_rule!r}"
        )
    rule_int = _RULE_TO_INT[update_rule]
    write_chi = chi_out is not None

    n = m.numel()

    # Triton needs a valid pointer even when WRITE_CHI is False — pass the
    # mean buffer as a dummy; the compiled kernel never touches it because
    # the constexpr guard removes the store.
    chi_ptr = chi_out.view(-1) if write_chi else m.view(-1)

    _param_update_kernel[(triton.cdiv(n, BLOCK),)](
        m.view(-1),
        S.view(-1),
        delta_m.view(-1),
        delta_S.view(-1),
        chi_ptr,
        cap_factor,
        rho,
        n,
        RULE=rule_int,
        WRITE_CHI=write_chi,
        BLOCK=BLOCK,
    )


# ======================================================================
#  Layer-side helper
# ======================================================================


def maybe_chi_buffer(layer, attr_name: str, ref: torch.Tensor) -> torch.Tensor:
    """
    Lazy-allocate (or resize) a same-shape ``chi`` buffer on ``layer``.

    Layers carry ``chi_w`` / ``chi_b`` attributes so PN-TAGI diagnostics
    can be inspected after ``update()``. The first call allocates; later
    calls reuse the buffer in-place. Shape changes (rare — e.g. dynamic
    parameter resizing) trigger reallocation.
    """
    buf = getattr(layer, attr_name, None)
    if buf is None or buf.shape != ref.shape or buf.device != ref.device:
        buf = torch.empty_like(ref)
        setattr(layer, attr_name, buf)
    return buf


# ======================================================================
#  Diagnostic aggregation
# ======================================================================


def chi_stats(chi: torch.Tensor) -> dict:
    """
    Aggregate per-parameter ``raw_chi`` into the scalar summaries listed in
    the PN-TAGI plan: median / p95 / max raw_chi, median / p95 clipped chi,
    and fractions chi > 0.1 / chi > 1.0 / dS > 0 (encoded as raw_chi < 0).

    Returns a dict of Python floats so it is cheap to log and JSON-serialise.
    """
    if chi.numel() == 0:
        return {
            "raw_chi_median": 0.0,
            "raw_chi_p95": 0.0,
            "raw_chi_max": 0.0,
            "chi_clip_median": 0.0,
            "chi_clip_p95": 0.0,
            "frac_chi_gt_0p1": 0.0,
            "frac_chi_gt_1": 0.0,
            "frac_dS_pos": 0.0,
            "count": 0,
        }
    flat = chi.detach().reshape(-1).to(torch.float32)
    clipped = torch.clamp(flat, min=0.0)
    return {
        "raw_chi_median": flat.median().item(),
        "raw_chi_p95": torch.quantile(flat, 0.95).item(),
        "raw_chi_max": flat.max().item(),
        "chi_clip_median": clipped.median().item(),
        "chi_clip_p95": torch.quantile(clipped, 0.95).item(),
        "frac_chi_gt_0p1": (clipped > 0.1).float().mean().item(),
        "frac_chi_gt_1": (clipped > 1.0).float().mean().item(),
        # raw_chi < 0  ⇔  -dS / S < 0  ⇔  dS > 0  (positive variance increment)
        "frac_dS_pos": (flat < 0.0).float().mean().item(),
        "count": int(flat.numel()),
    }
