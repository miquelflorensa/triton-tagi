"""
TAGI CNN Layers — Triton Implementation
========================================
Conv2D = im2col (Triton) + fused matmul (reused from tagi_triton.py)
AvgPool2D = Triton fused kernel for mean & variance pooling
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import numpy as np
from tagi_triton import (
    triton_fused_var_forward,
    triton_fused_backward_delta,
    triton_param_update,
    triton_relu_moments,
    triton_output_innovation,
    TritonTAGILayer,          # FC layer (for after flatten)
)
from triton_tagi.layers.even_softplus import even_softplus
from triton_tagi.update.observation import compute_innovation

# ====================================================================
# Triton Kernels: im2col / col2im
# ====================================================================

@triton.jit
def im2col_kernel(
    inp_ptr, out_ptr,
    N, C, H, W,
    kH, kW, stride, padding,
    H_out, W_out, K,          # K = C * kH * kW
    BLOCK_K: tl.constexpr,
):
    """Each program = one patch row. Writes K elements to out[pid, :]."""
    pid = tl.program_id(0)          # 0 .. N*H_out*W_out - 1
    n  = pid // (H_out * W_out)
    rem = pid % (H_out * W_out)
    oh = rem // W_out
    ow = rem % W_out

    offs = tl.arange(0, BLOCK_K)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs
        valid = k < K
        c     = k // (kH * kW)
        rem_k = k %  (kH * kW)
        kh    = rem_k // kW
        kw_v  = rem_k %  kW
        ih = oh * stride - padding + kh
        iw = ow * stride - padding + kw_v
        ok = valid & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        idx = n * (C * H * W) + c * (H * W) + ih * W + iw
        val = tl.load(inp_ptr + idx, mask=ok, other=0.0)
        tl.store(out_ptr + pid * K + k, val, mask=valid)


@triton.jit
def col2im_kernel(
    col_ptr, img_ptr,
    N, C, H, W,
    kH, kW, stride, padding,
    H_out, W_out, K,
    total_pixels,
    BLOCK: tl.constexpr,
    KH_KW: tl.constexpr,      # = kH * kW  (constexpr for loop unroll)
    KW: tl.constexpr,          # kW
):
    """Scatter-add columns back to image. Each thread = one input pixel."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < total_pixels

    n    = offs // (C * H * W)
    rem  = offs %  (C * H * W)
    c    = rem  // (H * W)
    rem2 = rem  %  (H * W)
    h    = rem2 // W
    w    = rem2 %  W

    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for idx in tl.static_range(KH_KW):
        kh   = idx // KW
        kw_v = idx %  KW
        oh_num = h + padding - kh
        ow_num = w + padding - kw_v
        oh = oh_num // stride
        ow = ow_num // stride
        ok = valid & (oh_num % stride == 0) & \
             (oh >= 0) & (oh < H_out) & (ow >= 0) & (ow < W_out)
        col_row = n * (H_out * W_out) + oh * W_out + ow
        col_col = c * (kH * kW) + kh * KW + kw_v
        val = tl.load(col_ptr + col_row * K + col_col, mask=ok, other=0.0)
        acc += val

    tl.store(img_ptr + offs, acc, mask=valid)


# ====================================================================
# Triton Kernels: Average Pooling (fused mean + variance)
# ====================================================================

@triton.jit
def avg_pool_fwd_kernel(
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

    tl.store(ma_out_ptr + offs, sum_m * inv_k2, mask=valid)   # mean / k²
    tl.store(Sa_out_ptr + offs, sum_s * inv_k4, mask=valid)   # var  / k⁴


@triton.jit
def avg_pool_bwd_kernel(
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


# ====================================================================
# Python Wrappers
# ====================================================================

BLOCK_EW = 1024

def triton_im2col(x, kH, kW, stride, padding):
    N, C, H, W = x.shape
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    K = C * kH * kW
    L = H_out * W_out
    out = torch.empty(N * L, K, device=x.device, dtype=x.dtype)
    BLOCK_K = max(16, triton.next_power_of_2(min(K, 1024)))
    im2col_kernel[(N * L,)](
        x, out, N, C, H, W, kH, kW, stride, padding, H_out, W_out, K,
        BLOCK_K=BLOCK_K)
    return out


def triton_col2im(col, N, C, H, W, kH, kW, stride, padding):
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    K = C * kH * kW
    total = N * C * H * W
    img = torch.empty(N, C, H, W, device=col.device, dtype=col.dtype)
    col2im_kernel[(triton.cdiv(total, BLOCK_EW),)](
        col, img, N, C, H, W, kH, kW, stride, padding, H_out, W_out, K,
        total, BLOCK=BLOCK_EW, KH_KW=kH * kW, KW=kW)
    return img


def triton_avg_pool_fwd(ma, Sa, k):
    N, C, H, W = ma.shape
    H_out, W_out = H // k, W // k
    total = N * C * H_out * W_out
    ma_o = torch.empty(N, C, H_out, W_out, device=ma.device, dtype=ma.dtype)
    Sa_o = torch.empty_like(ma_o)
    avg_pool_fwd_kernel[(triton.cdiv(total, BLOCK_EW),)](
        ma, Sa, ma_o, Sa_o, N, C, H, W, H_out, W_out,
        k, 1.0 / (k * k), 1.0 / (k ** 4), BLOCK=BLOCK_EW)
    return ma_o, Sa_o


def triton_avg_pool_bwd(dm, ds, N, C, H, W, k):
    H_out, W_out = H // k, W // k
    total = N * C * H * W
    dm_o = torch.empty(N, C, H, W, device=dm.device, dtype=dm.dtype)
    ds_o = torch.empty_like(dm_o)
    avg_pool_bwd_kernel[(triton.cdiv(total, BLOCK_EW),)](
        dm, ds, dm_o, ds_o, N, C, H, W, H_out, W_out,
        k, 1.0 / (k * k), 1.0 / (k ** 4), BLOCK=BLOCK_EW)
    return dm_o, ds_o


# ====================================================================
# TAGI Conv2D Layer
# ====================================================================

class TritonTAGIConv2D:
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, device="cuda"):
        self.C_in, self.C_out = C_in, C_out
        self.kH = self.kW = kernel_size
        self.stride, self.padding = stride, padding
        K = C_in * self.kH * self.kW

        # He initialization: scale = sqrt(1 / fan_in)
        scale = np.sqrt(1.0 / K)
        self.mw = torch.randn(K, C_out, device=device) * scale
        self.Sw = torch.full((K, C_out), scale ** 2, device=device)
        self.mb = torch.zeros(1, C_out, device=device)
        self.Sb = torch.full((1, C_out), scale ** 2, device=device)

    def forward(self, ma, Sa):
        N, C, H, W = ma.shape
        self.input_shape = (N, C, H, W)
        H_out = (H + 2 * self.padding - self.kH) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kW) // self.stride + 1
        self.spatial = (H_out, W_out)

        # im2col: (N, C_in, H, W) → (N*L, K)
        patches_ma = triton_im2col(ma, self.kH, self.kW, self.stride, self.padding)
        patches_Sa = triton_im2col(Sa, self.kH, self.kW, self.stride, self.padding)
        self.patches_ma = patches_ma

        # Mean: cuBLAS matmul + bias
        mz_flat = torch.matmul(patches_ma, self.mw) + self.mb
        # Variance: Triton fused (3 matmuls → 1)
        Sz_flat = triton_fused_var_forward(patches_ma, patches_Sa,
                                           self.mw, self.Sw, self.Sb)

        # Reshape (N*L, C_out) → (N, C_out, H_out, W_out)
        self.mz = mz_flat.view(N, H_out, W_out, self.C_out).permute(0, 3, 1, 2).contiguous()
        self.Sz = Sz_flat.view(N, H_out, W_out, self.C_out).permute(0, 3, 1, 2).contiguous()
        return self.mz, self.Sz

    def backward(self, delta_mz, delta_Sz):
        N = delta_mz.shape[0]
        # Flatten (N, C_out, H, W) → (N*L, C_out)
        dmz = delta_mz.permute(0, 2, 3, 1).reshape(-1, self.C_out).contiguous()
        dSz = delta_Sz.permute(0, 2, 3, 1).reshape(-1, self.C_out).contiguous()
        NL = dmz.shape[0]

        # Batch-size-invariant scaling (see tagi_triton.py for derivation)
        EFFECTIVE_SAMPLES = 12.0
        inv_NL = 1.0 / NL
        gamma = EFFECTIVE_SAMPLES / NL  # damped variance info absorption

        # Mean gradients: averaged over all patches
        grad_mw = torch.matmul(self.patches_ma.T, dmz) * inv_NL
        grad_mb = dmz.mean(0, keepdim=True)
        # Variance gradients: damped sum (absorb C samples of info)
        grad_Sw = torch.matmul((self.patches_ma ** 2).T, dSz) * gamma
        grad_Sb = dSz.sum(0, keepdim=True) * gamma

        # Fused param update (Triton element-wise)
        triton_param_update(self.mw, self.Sw, grad_mw, grad_Sw)
        triton_param_update(self.mb.view(-1), self.Sb.view(-1),
                            grad_mb.view(-1), grad_Sb.view(-1))

        # Delta propagation: Triton fused (2 matmuls → 1)
        dp_ma, dp_Sa = triton_fused_backward_delta(dmz, dSz, self.mw)

        # col2im: (N*L, K) → (N, C_in, H, W)
        _, C, H, W = self.input_shape
        d_ma = triton_col2im(dp_ma, N, C, H, W,
                             self.kH, self.kW, self.stride, self.padding)
        d_Sa = triton_col2im(dp_Sa, N, C, H, W,
                             self.kH, self.kW, self.stride, self.padding)
        return d_ma, d_Sa


# ====================================================================
# TAGI AvgPool2D Layer
# ====================================================================

class TritonTAGIAvgPool2D:
    def __init__(self, kernel_size):
        self.k = kernel_size

    def forward(self, ma, Sa):
        self.input_shape = ma.shape
        return triton_avg_pool_fwd(ma, Sa, self.k)

    def backward(self, dm, ds):
        N, C, H, W = self.input_shape
        return triton_avg_pool_bwd(dm, ds, N, C, H, W, self.k)


# ====================================================================
# TAGI CNN Network (configurable)
# ====================================================================

class TritonTAGICNN:
    """
    Build a TAGI CNN with conv → relu → pool → ... → flatten → fc → ... → output.
    Accepts a list of layer specs, e.g.:
        [('conv', 1, 32, 5, 1, 2),   # C_in, C_out, k, stride, pad
         ('relu',),
         ('pool', 2),
         ('conv', 32, 64, 5, 1, 2),
         ('relu',),
         ('pool', 2),
         ('flatten',),
         ('fc', 3136, 256),
         ('relu',),
         ('fc', 256, 10)]
    """
    def __init__(self, spec, device="cuda"):
        self.device = device
        self.layers = []
        self.layer_types = []
        for s in spec:
            t = s[0]
            self.layer_types.append(t)
            if t == 'conv':
                _, Ci, Co, k, st, pad = s
                self.layers.append(TritonTAGIConv2D(Ci, Co, k, st, pad, device))
            elif t == 'pool':
                self.layers.append(TritonTAGIAvgPool2D(s[1]))
            elif t == 'fc':
                self.layers.append(TritonTAGILayer(s[1], s[2], device))
            elif t == 'even_softplus':
                # s[1] = half_width (K = number of observation dims)
                self.layers.append(s[1])  # store half_width as the "layer"
            elif t in ('relu', 'flatten'):
                self.layers.append(None)

    def forward(self, x):
        ma = x
        Sa = torch.zeros_like(x)
        self.relu_jacobians = []
        self.relu_indices = []
        self.even_softplus_jacobians = []
        self.even_softplus_indices = []

        for i, (lt, layer) in enumerate(zip(self.layer_types, self.layers)):
            if lt == 'conv':
                ma, Sa = layer.forward(ma, Sa)
            elif lt == 'pool':
                ma, Sa = layer.forward(ma, Sa)
            elif lt == 'relu':
                if ma.dim() == 4:  # spatial
                    N, C, H, W = ma.shape
                    ma_r, Sa_r, J = triton_relu_moments(ma.reshape(-1), Sa.reshape(-1))
                    ma = ma_r.view(N, C, H, W)
                    Sa = Sa_r.view(N, C, H, W)
                    self.relu_jacobians.append(J.view(N, C, H, W))
                else:
                    bs = x.shape[0]
                    ma, Sa, J = triton_relu_moments(
                        ma.reshape(-1), Sa.reshape(-1))
                    ma = ma.view(bs, -1)
                    Sa = Sa.view_as(ma)
                    self.relu_jacobians.append(J.view_as(ma))
                self.relu_indices.append(i)
            elif lt == 'even_softplus':
                half_width = layer  # stored half_width
                original_shape = ma.shape
                ma_flat, Sa_flat, J = even_softplus(
                    ma.reshape(-1), Sa.reshape(-1), half_width
                )
                ma = ma_flat.view(original_shape)
                Sa = Sa_flat.view(original_shape)
                self.even_softplus_jacobians.append(J.view(original_shape))
                self.even_softplus_indices.append(i)
            elif lt == 'flatten':
                self.flatten_shape = ma.shape
                ma = ma.view(ma.shape[0], -1)
                Sa = Sa.view(Sa.shape[0], -1)
            elif lt == 'fc':
                mz, Sz = layer.forward(ma, Sa)
                ma, Sa = mz, Sz
        return ma, Sa

    def step(self, x_batch, y_batch, sigma_v):
        y_pred_m, y_pred_S = self.forward(x_batch)

        # Check if heteroscedastic (output is 2x the observation width)
        if y_pred_m.shape[-1] == 2 * y_batch.shape[-1]:
            delta_mz, delta_Sz = compute_innovation(
                y_batch, y_pred_m, y_pred_S, sigma_v)
        else:
            delta_mz, delta_Sz = triton_output_innovation(
                y_batch, y_pred_m, y_pred_S, sigma_v)

        relu_iter = len(self.relu_jacobians) - 1
        esp_iter = len(self.even_softplus_jacobians) - 1

        for i in reversed(range(len(self.layers))):
            lt = self.layer_types[i]
            layer = self.layers[i]

            if lt == 'fc':
                delta_mz, delta_Sz = layer.backward(delta_mz, delta_Sz)

            elif lt == 'relu':
                J = self.relu_jacobians[relu_iter]
                relu_iter -= 1
                if J.shape != delta_mz.shape:
                    J = J.view(delta_mz.shape)
                delta_mz = delta_mz * J          # J for mean
                delta_Sz = delta_Sz * J * J      # J² for variance

            elif lt == 'even_softplus':
                J = self.even_softplus_jacobians[esp_iter]
                esp_iter -= 1
                if J.shape != delta_mz.shape:
                    J = J.view(delta_mz.shape)
                delta_mz = delta_mz * J
                delta_Sz = delta_Sz * J * J

            elif lt == 'flatten':
                delta_mz = delta_mz.view(self.flatten_shape)
                delta_Sz = delta_Sz.view(self.flatten_shape)

            elif lt == 'pool':
                delta_mz, delta_Sz = layer.backward(delta_mz, delta_Sz)

            elif lt == 'conv':
                delta_mz, delta_Sz = layer.backward(delta_mz, delta_Sz)


# ====================================================================
# PyTorch Reference CNN (for benchmarking)
# ====================================================================

class PTConv2D:
    """PyTorch TAGI Conv2D using F.unfold (im2col) + matmul."""
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, device="cuda"):
        self.C_in, self.C_out = C_in, C_out
        self.kH = self.kW = kernel_size
        self.stride, self.padding = stride, padding
        K = C_in * kernel_size * kernel_size

        # He initialization: scale = sqrt(1 / fan_in)
        scale = np.sqrt(1.0 / K)
        self.mw = torch.randn(K, C_out, device=device) * scale
        self.Sw = torch.full((K, C_out), scale ** 2, device=device)
        self.mb = torch.zeros(1, C_out, device=device)
        self.Sb = torch.full((1, C_out), scale ** 2, device=device)

    def forward(self, ma, Sa):
        N, C, H, W = ma.shape
        self.input_shape = (N, C, H, W)
        H_out = (H + 2 * self.padding - self.kH) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kW) // self.stride + 1

        p_ma = F.unfold(ma, self.kH, padding=self.padding,
                        stride=self.stride).permute(0, 2, 1).reshape(-1, self.mw.shape[0])
        p_Sa = F.unfold(Sa, self.kH, padding=self.padding,
                        stride=self.stride).permute(0, 2, 1).reshape(-1, self.mw.shape[0])
        self.patches_ma = p_ma

        mz_f = p_ma @ self.mw + self.mb
        Sz_f = (p_ma ** 2) @ self.Sw + p_Sa @ (self.mw ** 2) + p_Sa @ self.Sw + self.Sb

        self.mz = mz_f.view(N, H_out, W_out, self.C_out).permute(0, 3, 1, 2).contiguous()
        self.Sz = Sz_f.view(N, H_out, W_out, self.C_out).permute(0, 3, 1, 2).contiguous()
        return self.mz, self.Sz

    def backward(self, delta_mz, delta_Sz):
        N = delta_mz.shape[0]
        dmz = delta_mz.permute(0, 2, 3, 1).reshape(-1, self.C_out).contiguous()
        dSz = delta_Sz.permute(0, 2, 3, 1).reshape(-1, self.C_out).contiguous()
        NL = dmz.shape[0]
        inv = 1.0 / NL

        gm = (self.patches_ma.T @ dmz) * inv
        gbm = dmz.mean(0, keepdim=True)
        # Variance: summed
        gS = (self.patches_ma ** 2).T @ dSz
        gbS = dSz.sum(0, keepdim=True)

        self.mw += self.Sw * gm
        self.mb += self.Sb * gbm
        self.Sw = torch.clamp(self.Sw + self.Sw ** 2 * gS, min=1e-6)
        self.Sb = torch.clamp(self.Sb + self.Sb ** 2 * gbS, min=1e-6)

        dp_ma = dmz @ self.mw.T
        dp_Sa = dSz @ (self.mw ** 2).T

        _, C, H, W = self.input_shape
        K = self.mw.shape[0]
        H_out, W_out = self.mz.shape[2], self.mz.shape[3]
        d_ma = F.fold(dp_ma.view(N, H_out * W_out, K).permute(0, 2, 1),
                      (H, W), self.kH, padding=self.padding, stride=self.stride)
        d_Sa = F.fold(dp_Sa.view(N, H_out * W_out, K).permute(0, 2, 1),
                      (H, W), self.kH, padding=self.padding, stride=self.stride)
        return d_ma, d_Sa


class PTAvgPool2D:
    def __init__(self, k):
        self.k = k
    def forward(self, ma, Sa):
        self.input_shape = ma.shape
        return F.avg_pool2d(ma, self.k), F.avg_pool2d(Sa, self.k) / (self.k ** 2)
    def backward(self, dm, ds):
        N, C, H, W = self.input_shape
        k = self.k
        dm_o = dm.repeat_interleave(k, 2).repeat_interleave(k, 3) / (k * k)
        ds_o = ds.repeat_interleave(k, 2).repeat_interleave(k, 3) / (k ** 4)
        return dm_o, ds_o


class PTFCLayer:
    def __init__(self, inf, outf, dev):
        # He initialization: scale = sqrt(1 / fan_in)
        scale = np.sqrt(1.0 / inf)
        self.mw = torch.randn(inf, outf, device=dev) * scale
        self.Sw = torch.full((inf, outf), scale ** 2, device=dev)
        self.mb = torch.zeros(1, outf, device=dev)
        self.Sb = torch.full((1, outf), scale ** 2, device=dev)
    def forward(self, ma, Sa):
        self.ma_in = ma
        self.mz = ma @ self.mw + self.mb
        self.Sz = (ma**2) @ self.Sw + Sa @ (self.mw**2) + Sa @ self.Sw + self.Sb
        return self.mz, self.Sz
    def backward(self, dm, ds):
        bs = dm.shape[0]
        gm = (self.ma_in.T @ dm) / bs
        gbm = dm.mean(0, keepdim=True)
        # Variance: summed
        gS = (self.ma_in**2).T @ ds
        gbS = ds.sum(0, keepdim=True)
        self.mw += self.Sw * gm;  self.mb += self.Sb * gbm
        self.Sw = torch.clamp(self.Sw + self.Sw**2 * gS, min=1e-6)
        self.Sb = torch.clamp(self.Sb + self.Sb**2 * gbS, min=1e-6)
        return dm @ self.mw.T, ds @ (self.mw**2).T


def _bayesian_relu_pt(mz, Sz):
    """PyTorch Bayesian ReLU: rectified Gaussian moments."""
    Sz_safe = torch.clamp(Sz, min=1e-12)
    sigma_z = torch.sqrt(Sz_safe)
    alpha = mz / sigma_z
    pdf = torch.exp(-0.5 * alpha ** 2) * 0.3989422804014327
    cdf = 0.5 * (1.0 + torch.erf(alpha * 0.7071067811865476))
    mu_m = sigma_z * pdf + mz * cdf
    var_m = torch.clamp(
        -mu_m ** 2 + 2 * mu_m * mz - mz * sigma_z * pdf
        + (Sz_safe - mz ** 2) * cdf, min=1e-12)
    return mu_m, var_m, cdf   # cdf = J = Φ(α)


class PTCNN:
    """PyTorch reference CNN with same architecture."""
    def __init__(self, spec, device):
        self.device = device
        self.layers, self.layer_types = [], []
        for s in spec:
            t = s[0]; self.layer_types.append(t)
            if   t == 'conv':  self.layers.append(PTConv2D(s[1],s[2],s[3],s[4],s[5],device))
            elif t == 'pool':  self.layers.append(PTAvgPool2D(s[1]))
            elif t == 'fc':    self.layers.append(PTFCLayer(s[1], s[2], device))
            else:              self.layers.append(None)

    def forward(self, x):
        ma, Sa = x, torch.zeros_like(x)
        self.relu_jacobians, self.relu_indices = [], []
        for i, (lt, L) in enumerate(zip(self.layer_types, self.layers)):
            if lt in ('conv', 'fc'):
                ma, Sa = L.forward(ma, Sa)
            elif lt == 'pool':
                ma, Sa = L.forward(ma, Sa)
            elif lt == 'relu':
                ma, Sa, J = _bayesian_relu_pt(ma, Sa)
                self.relu_jacobians.append(J); self.relu_indices.append(i)
            elif lt == 'flatten':
                self.flatten_shape = ma.shape
                ma = ma.view(ma.shape[0], -1)
                Sa = Sa.view(Sa.shape[0], -1)
        return ma, Sa

    def step(self, xb, yb, sv):
        ym, yS = self.forward(xb)
        Sy = yS + sv**2
        dm, ds = (yb - ym) / Sy, -1.0 / Sy
        ri = len(self.relu_jacobians) - 1
        for i in reversed(range(len(self.layers))):
            lt, L = self.layer_types[i], self.layers[i]
            if   lt == 'fc':      dm, ds = L.backward(dm, ds)
            elif lt == 'conv':    dm, ds = L.backward(dm, ds)
            elif lt == 'pool':    dm, ds = L.backward(dm, ds)
            elif lt == 'flatten':
                dm = dm.view(self.flatten_shape); ds = ds.view(self.flatten_shape)
            elif lt == 'relu':
                J = self.relu_jacobians[ri]; ri -= 1
                if J.shape != dm.shape: J = J.view(dm.shape)
                dm, ds = dm * J, ds * J * J     # J for mean, J² for var
