"""
Shared Triton kernels used across the TAGI library.

Contains the fused variance-forward and backward-delta tiled matmul kernels.
These are the computational workhorses behind the Linear layer.

Variance formula
----------------
Sz = ma² @ Sw  +  Sa @ (mw² + Sw)  +  Sb

The two-matmul grouping mirrors cuTAGI's inner loop which accumulates
    sum_var += (mw²+Sw)*Sa + Sw*ma²
per input element k, minimising fp32 accumulation differences vs cuTAGI's
FMA-fused CUDA kernel.
"""

import torch
import triton
import triton.language as tl

# ======================================================================
#  Block-size heuristic (shared)
# ======================================================================


def _pick_block(K: int, N: int):
    """Choose tile sizes for the matmul kernels based on matrix dimensions."""
    if K >= 128 and N >= 128:
        return 64, 64, 32
    elif K >= 64 and N >= 64:
        return 32, 32, 32
    else:
        return 16, 16, 16


# ======================================================================
#  Fused variance forward kernel
#  Computes  Sz = ma² @ Sw  +  Sa @ (mw² + Sw)  +  Sb
#
#  Two-matmul grouping mirrors cuTAGI's inner loop:
#      sum_var += (mw²+Sw)*Sa + Sw*ma²  (per k)
#  which minimises fp32 accumulation divergence vs cuTAGI's FMA CUDA kernel.
# ======================================================================


@triton.jit
def _fused_var_forward_kernel(
    ma_ptr,
    Sa_ptr,
    mw_ptr,
    Sw_ptr,
    Sb_ptr,
    Sz_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    ma_ptrs = ma_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    Sa_ptrs = Sa_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    Sw_ptrs = Sw_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    mw_ptrs = mw_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K
        m_mask = (offs_m[:, None] < M) & k_mask[None, :]
        w_mask = k_mask[:, None] & (offs_n[None, :] < N)

        ma_t = tl.load(ma_ptrs, mask=m_mask, other=0.0)
        Sa_t = tl.load(Sa_ptrs, mask=m_mask, other=0.0)
        Sw_t = tl.load(Sw_ptrs, mask=w_mask, other=0.0)
        mw_t = tl.load(mw_ptrs, mask=w_mask, other=0.0)

        # ma² @ Sw  — full fp32 (no tf32) to match cuTAGI's scalar FMA accuracy
        acc += tl.dot(ma_t * ma_t, Sw_t, allow_tf32=False)
        # Sa @ (mw² + Sw)  — fuse the two Sa-side terms; mirrors cuTAGI grouping
        acc += tl.dot(Sa_t, mw_t * mw_t + Sw_t, allow_tf32=False)

        ma_ptrs += BLOCK_K * stride_ak
        Sa_ptrs += BLOCK_K * stride_ak
        Sw_ptrs += BLOCK_K * stride_wk
        mw_ptrs += BLOCK_K * stride_wk
        offs_k += BLOCK_K

    sb = tl.load(
        Sb_ptr + pid_n * BLOCK_N + tl.arange(0, BLOCK_N),
        mask=(pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) < N,
        other=0.0,
    )
    acc += sb[None, :]

    c_ptrs = Sz_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_fused_var_forward(ma, Sa, mw, Sw, Sb):
    """
    Fused variance forward pass.

    Computes the pre-activation variance:
        Sz = ma² @ Sw  +  Sa @ (mw² + Sw)  +  Sb

    Parameters
    ----------
    ma : Tensor (B, K)   activation means
    Sa : Tensor (B, K)   activation variances
    mw : Tensor (K, N)   weight means
    Sw : Tensor (K, N)   weight variances
    Sb : Tensor (1, N)   bias variances

    Returns
    -------
    Sz : Tensor (B, N)   pre-activation variances
    """
    M, K = ma.shape
    _, N = mw.shape
    Sz = torch.empty(M, N, device=ma.device, dtype=ma.dtype)
    BM, BN, BK = _pick_block(K, N)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _fused_var_forward_kernel[grid](
        ma,
        Sa,
        mw,
        Sw,
        Sb.view(-1),
        Sz,
        M,
        N,
        K,
        ma.stride(0),
        ma.stride(1),
        mw.stride(0),
        mw.stride(1),
        Sz.stride(0),
        Sz.stride(1),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
    )
    return Sz


# ======================================================================
#  Fused backward-delta kernel
#  Computes  d_ma = dmz @ mw^T       (mean delta propagation)
#            d_Sa = dSz @ (mw²)^T    (var  delta propagation)
# ======================================================================


@triton.jit
def _fused_backward_delta_kernel(
    dmz_ptr,
    dSz_ptr,
    mw_ptr,
    d_ma_ptr,
    d_Sa_ptr,
    M,
    K,
    N,
    stride_dm,
    stride_dn,
    stride_wk,
    stride_wn,
    stride_cm,
    stride_ck,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_k = tl.cdiv(K, BLOCK_K)
    pid_m = pid // num_k
    pid_k = pid % num_k

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    acc_ma = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    acc_Sa = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    dmz_ptrs = dmz_ptr + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
    dSz_ptrs = dSz_ptr + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
    mw_ptrs = mw_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    for _ in range(0, tl.cdiv(N, BLOCK_N)):
        n_mask = offs_n < N
        d_mask = (offs_m[:, None] < M) & n_mask[None, :]
        w_mask = n_mask[:, None] & (offs_k[None, :] < K)

        dmz_t = tl.load(dmz_ptrs, mask=d_mask, other=0.0)
        dSz_t = tl.load(dSz_ptrs, mask=d_mask, other=0.0)
        mw_t = tl.load(mw_ptrs, mask=w_mask, other=0.0)

        acc_ma += tl.dot(dmz_t, mw_t, allow_tf32=False)
        acc_Sa += tl.dot(dSz_t, mw_t * mw_t, allow_tf32=False)

        dmz_ptrs += BLOCK_N * stride_dn
        dSz_ptrs += BLOCK_N * stride_dn
        mw_ptrs += BLOCK_N * stride_wn
        offs_n += BLOCK_N

    out_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(
        d_ma_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck, acc_ma, mask=out_mask
    )
    tl.store(
        d_Sa_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck, acc_Sa, mask=out_mask
    )


def triton_fused_backward_delta(dmz, dSz, mw):
    """
    Fused backward delta propagation.

    Computes the input-space deltas by back-projecting through weights:
        d_ma = dmz @ mw^T        (mean delta)
        d_Sa = dSz @ (mw²)^T     (variance delta)

    Parameters
    ----------
    dmz : Tensor (B, N)   mean delta from next layer
    dSz : Tensor (B, N)   variance delta from next layer
    mw  : Tensor (K, N)   weight means

    Returns
    -------
    d_ma : Tensor (B, K)  mean delta to propagate backward
    d_Sa : Tensor (B, K)  variance delta to propagate backward
    """
    M, N = dmz.shape
    K = mw.shape[0]
    d_ma = torch.empty(M, K, device=dmz.device, dtype=dmz.dtype)
    d_Sa = torch.empty(M, K, device=dmz.device, dtype=dmz.dtype)
    BM, BK, BN = _pick_block(K, N)
    grid = (triton.cdiv(M, BM) * triton.cdiv(K, BK),)
    _fused_backward_delta_kernel[grid](
        dmz,
        dSz,
        mw,
        d_ma,
        d_Sa,
        M,
        K,
        N,
        dmz.stride(0),
        dmz.stride(1),
        mw.stride(0),
        mw.stride(1),
        d_ma.stride(0),
        d_ma.stride(1),
        BLOCK_M=BM,
        BLOCK_K=BK,
        BLOCK_N=BN,
    )
    return d_ma, d_Sa
