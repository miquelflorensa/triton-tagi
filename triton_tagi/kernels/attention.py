"""
Fused Triton kernels for the attention variance paths.

The MultiheadAttentionV2 forward and backward each do three structurally
identical variance computations that cuBLAS expresses as three separate
matmuls. This module fuses each triple into a single Triton kernel so the
shared operands (mu_q, mu_k, mu_score, ...) are loaded once per tile.

For every kernel, the forward is expressed in TAGI-product form:
    sum over k:    mean = μ_a · μ_b
                   var  = var_a · var_b + var_a · μ_b² + μ_a² · var_b
                        = var_a · (μ_b² + var_b)  +  μ_a² · var_b

Two kernels are provided:

``bmm_tagi_var``
    Computes the full TAGI-product variance ``var_ab`` for two batched
    Gaussian tensors (both operands have nontrivial variance). Used for
    ``QKᵀ`` and ``Score @ V`` in the forward pass.
    Two ``tl.dot`` calls in the inner loop (matching the grouping in
    ``kernels/common.triton_fused_var_forward``).

``bmm_shared_operand``
    Computes both ``mean = a_mean · b`` and ``var = a_var · b²`` in one
    kernel when the right operand ``b`` is deterministic (no variance).
    The symmetric case (deterministic ``a``) is supported by passing
    transposed strides. Used for all four backward reductions:
    ``δV / δscore / δQ / δK``.

All kernels accept arbitrary batch strides; callers may pass views that
alias ``.transpose(-1, -2)`` by supplying the appropriate (stride_a_m,
stride_a_k) — no ``.contiguous()`` copy is needed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ======================================================================
#  Block sizes are chosen once at launch via ``_pick_blocks`` instead of
#  through ``@triton.autotune`` — autotune adds ~100µs of Python dispatch
#  overhead per call even on cache hits, which dominated the small-shape
#  attention kernels used by reverse_predictor (S=8, D=32).
# ======================================================================


def _pick_blocks(M: int, L: int, K: int) -> tuple[int, int, int, int, int]:
    """Shape-adaptive block sizes: (BLOCK_M, BLOCK_L, BLOCK_K, num_warps, num_stages)."""
    if max(M, L) <= 32:
        return 16, 16, 32, 2, 2
    if max(M, L) <= 64:
        return 32, 32, 32, 4, 3
    return 64, 64, 32, 4, 3


# ======================================================================
#  Kernel 1 — TAGI-product variance, both operands Gaussian
#
#  var_ab[n, m, l] = Σ_k  var_a[n, m, k] · (μ_b² + var_b)[n, k, l]
#                       + μ_a²[n, m, k]  · var_b[n, k, l]
# ======================================================================


@triton.jit
def _bmm_tagi_var_kernel(
    mu_a_ptr, var_a_ptr,
    mu_b_ptr, var_b_ptr,
    var_ab_ptr,
    N, M, L, K,
    stride_an, stride_am, stride_ak,
    stride_bn, stride_bk, stride_bl,
    stride_on, stride_om, stride_ol,
    scale_sq,
    BLOCK_M: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_ml = tl.program_id(1)
    num_l = tl.cdiv(L, BLOCK_L)
    pid_m = pid_ml // num_l
    pid_l = pid_ml % num_l

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_k = tl.arange(0, BLOCK_K)

    a_base = pid_n * stride_an
    b_base = pid_n * stride_bn

    mu_a_ptrs = mu_a_ptr + a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    var_a_ptrs = var_a_ptr + a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    mu_b_ptrs = mu_b_ptr + b_base + offs_k[:, None] * stride_bk + offs_l[None, :] * stride_bl
    var_b_ptrs = var_b_ptr + b_base + offs_k[:, None] * stride_bk + offs_l[None, :] * stride_bl

    acc = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        b_mask = k_mask[:, None] & (offs_l[None, :] < L)

        mu_a = tl.load(mu_a_ptrs, mask=a_mask, other=0.0)
        var_a = tl.load(var_a_ptrs, mask=a_mask, other=0.0)
        mu_b = tl.load(mu_b_ptrs, mask=b_mask, other=0.0)
        var_b = tl.load(var_b_ptrs, mask=b_mask, other=0.0)

        # Two fused dots mirror the cuTAGI-parity grouping in common.py.
        # allow_tf32=False keeps FP32 accumulation so variance matches the
        # cuTAGI reference within validation ATOL (test_mha_forward_mean).
        acc += tl.dot(var_a, mu_b * mu_b + var_b, allow_tf32=False)
        acc += tl.dot(mu_a * mu_a, var_b, allow_tf32=False)

        mu_a_ptrs += BLOCK_K * stride_ak
        var_a_ptrs += BLOCK_K * stride_ak
        mu_b_ptrs += BLOCK_K * stride_bk
        var_b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    acc = acc * scale_sq

    out_mask = (offs_m[:, None] < M) & (offs_l[None, :] < L)
    out_ptrs = (
        var_ab_ptr
        + pid_n * stride_on
        + offs_m[:, None] * stride_om
        + offs_l[None, :] * stride_ol
    )
    tl.store(out_ptrs, acc, mask=out_mask)


def _batch_stride(x: torch.Tensor) -> int:
    """Stride to the next batch item, assuming all leading dims are packed.

    Required assumption: the tensor's axes 0..-3 form a single contiguous
    block — so that iterating the flattened batch index advances memory by
    ``x.stride(-3)`` for every step. ``transpose(-1, -2)`` preserves this
    invariant (it only touches the two innermost dims); arbitrary
    ``permute`` does not.
    """
    if x.dim() <= 2:
        # No batch axis — treat as a single slice.
        return x.numel()
    return x.stride(-3)


def bmm_tagi_var(
    mu_a: torch.Tensor,
    var_a: torch.Tensor,
    mu_b: torch.Tensor,
    var_b: torch.Tensor,
    scale_sq: float = 1.0,
) -> torch.Tensor:
    """TAGI-product variance for batched matmul.

    Computes the last-two-dim matmul variance for every batch slice::

        var_ab[n, m, l] = scale_sq · Σ_k ( var_a · (μ_b² + var_b)
                                           + μ_a² · var_b )

    Leading dims of the four inputs must match and be packed relative to
    each other. Accepts non-contiguous views from ``.transpose(-1, -2)``
    without copying; arbitrary ``permute`` requires ``.contiguous()``.

    Args:
        mu_a, var_a: shape (..., M, K)
        mu_b, var_b: shape (..., K, L)
        scale_sq:    scalar multiplier applied after reduction
                     (e.g. ``1/head_dim`` for scaled dot-product attention)

    Returns:
        var_ab: shape (..., M, L), contiguous.
    """
    batch_shape = mu_a.shape[:-2]
    M, K = mu_a.shape[-2], mu_a.shape[-1]
    Kb, L = mu_b.shape[-2], mu_b.shape[-1]
    if K != Kb:
        raise ValueError(f"Reduction dims disagree: a has {K}, b has {Kb}")
    N = 1
    for d in batch_shape:
        N *= d

    out = torch.empty((*batch_shape, M, L), device=mu_a.device, dtype=mu_a.dtype)

    BLOCK_M, BLOCK_L, BLOCK_K, num_warps, num_stages = _pick_blocks(M, L, K)
    grid = (N, triton.cdiv(M, BLOCK_M) * triton.cdiv(L, BLOCK_L))
    _bmm_tagi_var_kernel[grid](
        mu_a, var_a,
        mu_b, var_b,
        out,
        N, M, L, K,
        _batch_stride(mu_a), mu_a.stride(-2), mu_a.stride(-1),
        _batch_stride(mu_b), mu_b.stride(-2), mu_b.stride(-1),
        _batch_stride(out), out.stride(-2), out.stride(-1),
        scale_sq,
        BLOCK_M=BLOCK_M, BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# ======================================================================
#  Kernel 2 — shared deterministic operand
#
#  mean_out[n, m, l] = Σ_k  a_mean[n, m, k] · b[n, k, l]
#  var_out[n, m, l]  = Σ_k  a_var [n, m, k] · b[n, k, l]²
#
#  Used in the backward for δV, δscore, δQ, δK. In every case one side
#  is deterministic (square it in-register) and the other side has
#  a distinct mean and variance (δmean vs δvar, or scaled_mu vs scaled_var).
# ======================================================================

@triton.jit
def _bmm_shared_b_kernel(
    a_mean_ptr, a_var_ptr,
    b_ptr,
    mean_out_ptr, var_out_ptr,
    N, M, L, K,
    stride_an, stride_am, stride_ak,
    stride_bn, stride_bk, stride_bl,
    stride_on, stride_om, stride_ol,
    scale, scale_sq,
    BLOCK_M: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_ml = tl.program_id(1)
    num_l = tl.cdiv(L, BLOCK_L)
    pid_m = pid_ml // num_l
    pid_l = pid_ml % num_l

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_k = tl.arange(0, BLOCK_K)

    a_base = pid_n * stride_an
    b_base = pid_n * stride_bn

    a_mean_ptrs = a_mean_ptr + a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_var_ptrs = a_var_ptr + a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + b_base + offs_k[:, None] * stride_bk + offs_l[None, :] * stride_bl

    acc_mean = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)
    acc_var = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        b_mask = k_mask[:, None] & (offs_l[None, :] < L)

        a_mean = tl.load(a_mean_ptrs, mask=a_mask, other=0.0)
        a_var = tl.load(a_var_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc_mean += tl.dot(a_mean, b, allow_tf32=False)
        acc_var += tl.dot(a_var, b * b, allow_tf32=False)

        a_mean_ptrs += BLOCK_K * stride_ak
        a_var_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    acc_mean = acc_mean * scale
    acc_var = acc_var * scale_sq

    out_mask = (offs_m[:, None] < M) & (offs_l[None, :] < L)
    mean_ptrs = (
        mean_out_ptr
        + pid_n * stride_on
        + offs_m[:, None] * stride_om
        + offs_l[None, :] * stride_ol
    )
    var_ptrs = (
        var_out_ptr
        + pid_n * stride_on
        + offs_m[:, None] * stride_om
        + offs_l[None, :] * stride_ol
    )
    tl.store(mean_ptrs, acc_mean, mask=out_mask)
    tl.store(var_ptrs, acc_var, mask=out_mask)


def bmm_shared_right(
    a_mean: torch.Tensor,
    a_var: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched matmul with a deterministic right operand.

    Returns both the mean and variance outputs in one kernel pass::

        mean[n, m, l] = scale   · Σ_k a_mean[n, m, k] · b[n, k, l]
        var [n, m, l] = scale²  · Σ_k a_var [n, m, k] · b[n, k, l]²

    Use this for ``δQ`` / ``δK`` where ``mu_k`` / ``mu_q`` is the shared
    deterministic operand; pass transposed strides to get ``δQ`` vs ``δK``.

    Args:
        a_mean, a_var: shape (..., M, K)
        b:             shape (..., K, L)
        scale:         scalar applied to the mean (variance gets ``scale²``)

    Returns:
        (mean, var), each of shape (..., M, L).
    """
    batch_shape = a_mean.shape[:-2]
    M, K = a_mean.shape[-2], a_mean.shape[-1]
    Kb, L = b.shape[-2], b.shape[-1]
    if K != Kb:
        raise ValueError(f"Reduction dims disagree: a has {K}, b has {Kb}")
    N = 1
    for d in batch_shape:
        N *= d

    mean = torch.empty((*batch_shape, M, L), device=a_mean.device, dtype=a_mean.dtype)
    var = torch.empty((*batch_shape, M, L), device=a_mean.device, dtype=a_mean.dtype)

    BLOCK_M, BLOCK_L, BLOCK_K, num_warps, num_stages = _pick_blocks(M, L, K)
    grid = (N, triton.cdiv(M, BLOCK_M) * triton.cdiv(L, BLOCK_L))
    _bmm_shared_b_kernel[grid](
        a_mean, a_var,
        b,
        mean, var,
        N, M, L, K,
        _batch_stride(a_mean), a_mean.stride(-2), a_mean.stride(-1),
        _batch_stride(b), b.stride(-2), b.stride(-1),
        _batch_stride(mean), mean.stride(-2), mean.stride(-1),
        scale, scale * scale,
        BLOCK_M=BLOCK_M, BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return mean, var


def bmm_shared_left(
    a: torch.Tensor,
    b_mean: torch.Tensor,
    b_var: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched matmul with a deterministic left operand.

    Returns both the mean and variance outputs in one kernel pass::

        mean[n, m, l] = scale  · Σ_k a[n, m, k]  · b_mean[n, k, l]
        var [n, m, l] = scale² · Σ_k a[n, m, k]² · b_var [n, k, l]

    Use this for ``δV`` / ``δscore`` where ``score`` / ``V`` is the
    shared deterministic operand (in the backward the squaring moves
    from the right operand to the left).

    Args:
        a:             shape (..., M, K)
        b_mean, b_var: shape (..., K, L)
        scale:         scalar applied to the mean (variance gets ``scale²``)

    Returns:
        (mean, var), each of shape (..., M, L).
    """
    # Transpose algebra: (a · b) = (bᵀ · aᵀ)ᵀ. Reuse the shared-right kernel
    # by passing (b_meanᵀ, b_varᵀ) as the "left" operand and aᵀ as the
    # shared "right" operand; transpose the outputs back. transpose(-1,-2)
    # preserves the leading-dim contiguity invariant used by _batch_stride.
    mean_t, var_t = bmm_shared_right(
        b_mean.transpose(-1, -2),
        b_var.transpose(-1, -2),
        a.transpose(-1, -2),
        scale=scale,
    )
    return mean_t.transpose(-1, -2), var_t.transpose(-1, -2)
