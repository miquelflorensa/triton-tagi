"""
Bayesian AvgPool2D layer for TAGI.

Average pooling for both means and variances:
    μ_out = (1/k²) · Σ μ_in       (within each k×k window)
    S_out = (1/k⁴) · Σ S_in       (variance scales as 1/k⁴ for avg of k² terms)

All computation done in a single fused Triton kernel per direction.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024


# ======================================================================
#  Triton kernel — forward
# ======================================================================

@triton.jit
def _avg_pool_fwd_kernel(
    ma_ptr, Sa_ptr, ma_out_ptr, Sa_out_ptr,
    N, C, H, W, H_out, W_out,
    k, inv_k2, inv_k4,
    BLOCK: tl.constexpr,
):
    """Fused avg-pool for mean and variance in one kernel."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * H_out * W_out
    valid = offs < total

    n    = offs // (C * H_out * W_out)
    rem  = offs %  (C * H_out * W_out)
    c    = rem  // (H_out * W_out)
    rem2 = rem  %  (H_out * W_out)
    oh   = rem2 // W_out
    ow   = rem2 %  W_out

    sum_m = tl.zeros((BLOCK,), dtype=tl.float32)
    sum_s = tl.zeros((BLOCK,), dtype=tl.float32)

    for kh in range(k):
        for kw in range(k):
            ih = oh * k + kh
            iw = ow * k + kw
            idx = n * (C * H * W) + c * (H * W) + ih * W + iw
            m = tl.load(ma_ptr + idx, mask=valid, other=0.0)
            s = tl.load(Sa_ptr + idx, mask=valid, other=0.0)
            sum_m += m
            sum_s += s

    tl.store(ma_out_ptr + offs, sum_m * inv_k2, mask=valid)
    tl.store(Sa_out_ptr + offs, sum_s * inv_k4, mask=valid)


# ======================================================================
#  Triton kernel — backward
# ======================================================================

@triton.jit
def _avg_pool_bwd_kernel(
    dm_ptr, ds_ptr, dm_out_ptr, ds_out_ptr,
    N, C, H, W, H_out, W_out,
    k, inv_k2, inv_k4,
    BLOCK: tl.constexpr,
):
    """Backward: distribute delta equally into k×k block."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * H * W
    valid = offs < total

    n    = offs // (C * H * W)
    rem  = offs %  (C * H * W)
    c    = rem  // (H * W)
    rem2 = rem  %  (H * W)
    h    = rem2 // W
    w    = rem2 %  W

    oh = h // k
    ow = w // k
    idx = n * (C * H_out * W_out) + c * (H_out * W_out) + oh * W_out + ow
    dm = tl.load(dm_ptr + idx, mask=valid, other=0.0)
    ds = tl.load(ds_ptr + idx, mask=valid, other=0.0)

    tl.store(dm_out_ptr + offs, dm * inv_k2, mask=valid)
    tl.store(ds_out_ptr + offs, ds * inv_k4, mask=valid)


# ======================================================================
#  AvgPool2D Layer
# ======================================================================

class AvgPool2D:
    """
    Bayesian average pooling layer.

    Parameters
    ----------
    kernel_size : int  pooling window size (square)
    """

    def __init__(self, kernel_size):
        self.k = kernel_size
        self.input_shape = None

    def forward(self, ma, Sa):
        """
        Parameters
        ----------
        ma : Tensor (N, C, H, W)  activation means
        Sa : Tensor (N, C, H, W)  activation variances

        Returns
        -------
        ma_out : Tensor (N, C, H//k, W//k)
        Sa_out : Tensor (N, C, H//k, W//k)
        """
        self.input_shape = ma.shape
        N, C, H, W = ma.shape
        k = self.k
        H_out, W_out = H // k, W // k
        total = N * C * H_out * W_out

        ma_out = torch.empty(N, C, H_out, W_out, device=ma.device,
                             dtype=ma.dtype)
        Sa_out = torch.empty_like(ma_out)

        _avg_pool_fwd_kernel[(triton.cdiv(total, BLOCK),)](
            ma, Sa, ma_out, Sa_out,
            N, C, H, W, H_out, W_out,
            k, 1.0 / (k * k), 1.0 / (k ** 4),
            BLOCK=BLOCK,
        )
        return ma_out, Sa_out

    def backward(self, dm, ds):
        """
        Parameters
        ----------
        dm : Tensor (N, C, H_out, W_out)  mean delta
        ds : Tensor (N, C, H_out, W_out)  variance delta

        Returns
        -------
        dm_out : Tensor (N, C, H, W)
        ds_out : Tensor (N, C, H, W)
        """
        N, C, H, W = self.input_shape
        k = self.k
        H_out, W_out = H // k, W // k
        total = N * C * H * W

        dm_out = torch.empty(N, C, H, W, device=dm.device, dtype=dm.dtype)
        ds_out = torch.empty_like(dm_out)

        _avg_pool_bwd_kernel[(triton.cdiv(total, BLOCK),)](
            dm, ds, dm_out, ds_out,
            N, C, H, W, H_out, W_out,
            k, 1.0 / (k * k), 1.0 / (k ** 4),
            BLOCK=BLOCK,
        )
        return dm_out, ds_out

    def __repr__(self):
        return f"AvgPool2D(kernel={self.k})"
