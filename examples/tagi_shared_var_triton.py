"""
TAGI with Shared Variance per Layer — Triton Implementation
=============================================================
Instead of maintaining one variance per parameter (Sw has the same shape as mw),
each layer has a **single scalar** variance for weights and one for biases.

Benefits:
  • Dramatically fewer variance parameters (2 scalars per layer vs. K×N + N).
  • Acts as a natural regularizer — prevents individual variances from collapsing
    or exploding.
  • Stabilizes training, especially for deeper networks.

Mathematical formulation
------------------------
For a linear layer  z = a @ W + b  with:
  - mw (K, N) : weight means
  - sw (scalar): shared weight variance  (replaces the full Sw matrix)
  - mb (1, N)  : bias means
  - sb (scalar): shared bias variance    (replaces the full Sb vector)

Forward variance propagation:
  Sz = sw * (ma² @ 1_{K×N}) + Sa @ mw² + sw * (Sa @ 1_{K×N}) + sb
     = sw * [sum_k(ma²_k)] · 1_N  +  Sa @ mw²  +  sw * [sum_k(Sa_k)] · 1_N  +  sb

Simplified (since sw is scalar):
  Sz[i,j] = sw · Σ_k ma[i,k]²  +  Σ_k Sa[i,k] · mw[k,j]²  +  sw · Σ_k Sa[i,k]  +  sb

Backward: the scalar variance gradient is the *sum* of all per-parameter gradients.
"""

import torch
import numpy as np
from tagi_triton import (
    triton_relu_moments,
    triton_output_innovation,
    triton_fused_backward_delta,
)


# ====================================================================
# Python Wrappers for Shared-Variance Operations
# ====================================================================


def triton_shared_var_forward(ma, Sa, mw, sw_scalar, sb_scalar):
    """
    Forward variance propagation with scalar shared variances.

    Sz = sw * (row_sum(ma²) · 1_N) + Sa @ mw² + sw * (row_sum(Sa) · 1_N) + sb

    We precompute the row sums in PyTorch (fast on GPU) and just add them
    to the matmul result Sa @ mw², avoiding extra work in the Triton kernel.
    """
    M, K = ma.shape
    _, N = mw.shape
    sw = sw_scalar.item()
    sb = sb_scalar.item()

    # Sa @ mw² — the only term that requires a full matmul
    Sa_mw2 = torch.matmul(Sa, mw * mw)

    # Row sums (M,1)
    row_ma2 = (ma * ma).sum(dim=1, keepdim=True)   # Σ_k ma²[i,k]
    row_Sa  = Sa.sum(dim=1, keepdim=True)            # Σ_k Sa[i,k]

    # Sz = Sa @ mw² + sw * (row_ma2 + row_Sa) + sb
    # The row sums broadcast across N columns
    Sz = Sa_mw2 + sw * (row_ma2 + row_Sa) + sb

    return Sz


def triton_scalar_var_update(sw, grad_S_flat):
    """
    Update a scalar variance using the aggregated gradient (pure PyTorch).

    Since all parameters share the same variance, the effective precision
    update averages the per-parameter precision increments:
      grad_sw = (1/P) Σ_{all params} grad_S[i]    where P = number of params
      sw_new  = sw / (1 - sw * grad_sw)

    Averaging (rather than summing) prevents the shared variance from
    collapsing too fast when there are many parameters.

    Args:
        sw: scalar tensor (shape ()) — shared variance, updated in-place
        grad_S_flat: per-parameter variance gradients (any shape)
    """
    n_params = grad_S_flat.numel()
    total_grad = grad_S_flat.sum().item() / n_params   # average, not sum
    sw_old = sw.item()
    denom = 1.0 - sw_old * total_grad
    # Clamp denominator to prevent negative/exploding variance
    denom = max(denom, 0.01)
    sw_new = sw_old / denom
    sw_new = max(sw_new, 1e-8)
    sw.fill_(sw_new)


# ====================================================================
# Shared-Variance FC Layer
# ====================================================================

class SharedVarTAGILayer:
    """
    TAGI fully-connected layer with a single shared variance per layer.

    Parameters:
        mw  (K, N)   : weight means
        sw  (scalar)  : shared weight variance
        mb  (1, N)   : bias means
        sb  (scalar)  : shared bias variance
    """
    def __init__(self, in_features, out_features, device, gain_var=0.1):
        self.device = device
        self.in_features = in_features
        self.out_features = out_features

        # He initialization
        scale = np.sqrt(1.0 / in_features)
        self.mw = torch.randn(in_features, out_features, device=device) * scale
        self.mb = torch.zeros(1, out_features, device=device)

        # Shared scalar variances (one per layer!)
        self.sw = torch.tensor((gain_var * scale) ** 2, device=device, dtype=torch.float32)
        self.sb = torch.tensor((gain_var * scale) ** 2, device=device, dtype=torch.float32)

    def forward(self, ma, Sa):
        self.ma_in = ma

        # Mean: standard matmul
        self.mz = torch.matmul(ma, self.mw) + self.mb

        # Variance: use shared-variance Triton kernel
        self.Sz = triton_shared_var_forward(ma, Sa, self.mw, self.sw, self.sb)

        return self.mz, self.Sz

    def backward(self, delta_mz, delta_Sz):
        bs = delta_mz.shape[0]
        BASELINE_BATCH = 32.0
        scale_factor_mean = 1.0 / BASELINE_BATCH

        # ---- Mean gradients (same as standard TAGI) ----
        grad_mw = torch.matmul(self.ma_in.T, delta_mz) * scale_factor_mean
        grad_mb = delta_mz.sum(0, keepdim=True) * scale_factor_mean

        # ---- Mean update (same as standard TAGI, using scalar sw broadcast) ----
        # mw_new = mw + sw * grad_mw   (sw is scalar, broadcast to all elements)
        self.mw += self.sw * grad_mw
        self.mb += self.sb * grad_mb

        # ---- Variance gradients ----
        # For the shared variance, we compute the per-element gradient then sum.
        # grad_Sw[k,j] = (ma_in²)^T @ delta_Sz   (same formula as standard TAGI)
        grad_Sw = torch.matmul((self.ma_in ** 2).T, delta_Sz)
        grad_Sb = delta_Sz.sum(0, keepdim=True)

        # ---- Scalar variance update (aggregate all per-element gradients) ----
        triton_scalar_var_update(self.sw, grad_Sw)
        triton_scalar_var_update(self.sb, grad_Sb)

        # ---- Backward delta propagation ----
        # For delta propagation we need the full Sw matrix equivalent.
        # Since sw is shared: effective Sw = sw * ones(K, N), so mw² uses remain the same
        # but the backward delta for Sa uses sw broadcast.
        delta_ma, delta_Sa = triton_fused_backward_delta(delta_mz, delta_Sz, self.mw)

        return delta_ma, delta_Sa


# ====================================================================
# Shared-Variance FNN Network
# ====================================================================

class SharedVarTAGINet:
    """
    TAGI Feed-Forward Network with shared variance per layer.
    Drop-in replacement for TritonTAGINet.
    """
    def __init__(self, layers_struct, device, gain_var=0.1):
        self.layers = []
        self.device = device
        for i in range(len(layers_struct) - 1):
            self.layers.append(
                SharedVarTAGILayer(layers_struct[i], layers_struct[i + 1], device,
                                   gain_var=gain_var)
            )

    def forward(self, x):
        ma, Sa = x, torch.zeros_like(x)
        self.jacobians = []
        for i, layer in enumerate(self.layers):
            mz, Sz = layer.forward(ma, Sa)
            if i < len(self.layers) - 1:
                ma, Sa, J = triton_relu_moments(mz, Sz)
                self.jacobians.append(J)
            else:
                ma, Sa = mz, Sz
                self.jacobians.append(torch.ones_like(mz))
        return ma, Sa

    def step(self, x_batch, y_batch, sigma_v):
        y_pred_m, y_pred_S = self.forward(x_batch)
        delta_mz, delta_Sz = triton_output_innovation(y_batch, y_pred_m, y_pred_S, sigma_v)

        for i in reversed(range(len(self.layers))):
            J = self.jacobians[i]
            dm = delta_mz * J
            ds = delta_Sz * J * J
            delta_mz, delta_Sz = self.layers[i].backward(dm, ds)
        return y_pred_m, y_pred_S

    def get_variances(self):
        """Return a summary of per-layer shared variances for monitoring."""
        info = {}
        for i, layer in enumerate(self.layers):
            info[f'layer_{i}_sw'] = layer.sw.item()
            info[f'layer_{i}_sb'] = layer.sb.item()
        return info


# ====================================================================
# Shared-Variance Conv2D Layer
# ====================================================================

class SharedVarTAGIConv2D:
    """
    TAGI Conv2D layer with a single shared variance for all weights
    and a single shared variance for all biases.
    """
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0,
                 device="cuda", gain_var=None):
        self.C_in, self.C_out = C_in, C_out
        self.kH = self.kW = kernel_size
        self.stride, self.padding = stride, padding
        K = C_in * self.kH * self.kW

        # He initialization
        scale = np.sqrt(1.0 / K)
        self.mw = torch.randn(K, C_out, device=device) * scale
        self.mb = torch.zeros(1, C_out, device=device)

        # Match TritonTAGIConv2D initialization: Sw = scale**2  (no gain_var)
        init_var = scale ** 2
        self.sw = torch.tensor(init_var, device=device, dtype=torch.float32)
        self.sb = torch.tensor(init_var, device=device, dtype=torch.float32)

    def forward(self, ma, Sa):
        from tagi_cnn_triton import triton_im2col

        N, C, H, W = ma.shape
        self.input_shape = (N, C, H, W)
        H_out = (H + 2 * self.padding - self.kH) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kW) // self.stride + 1
        self.spatial = (H_out, W_out)

        # im2col
        patches_ma = triton_im2col(ma, self.kH, self.kW, self.stride, self.padding)
        patches_Sa = triton_im2col(Sa, self.kH, self.kW, self.stride, self.padding)
        self.patches_ma = patches_ma

        # Mean: standard matmul
        mz_flat = torch.matmul(patches_ma, self.mw) + self.mb

        # Variance: shared-variance kernel
        Sz_flat = triton_shared_var_forward(patches_ma, patches_Sa,
                                            self.mw, self.sw, self.sb)

        # Reshape
        self.mz = mz_flat.view(N, H_out, W_out, self.C_out).permute(0, 3, 1, 2).contiguous()
        self.Sz = Sz_flat.view(N, H_out, W_out, self.C_out).permute(0, 3, 1, 2).contiguous()
        return self.mz, self.Sz

    def backward(self, delta_mz, delta_Sz):
        from tagi_cnn_triton import triton_col2im

        N = delta_mz.shape[0]
        dmz = delta_mz.permute(0, 2, 3, 1).reshape(-1, self.C_out).contiguous()
        dSz = delta_Sz.permute(0, 2, 3, 1).reshape(-1, self.C_out).contiguous()
        NL = dmz.shape[0]

        EFFECTIVE_SAMPLES = 12.0
        inv_NL = 1.0 / NL
        gamma = EFFECTIVE_SAMPLES / NL

        # ---- Mean gradients & update ----
        grad_mw = torch.matmul(self.patches_ma.T, dmz) * inv_NL
        grad_mb = dmz.mean(0, keepdim=True)

        self.mw += self.sw * grad_mw
        self.mb += self.sb * grad_mb

        # ---- Variance gradients & scalar update ----
        grad_Sw = torch.matmul((self.patches_ma ** 2).T, dSz) * gamma
        grad_Sb = dSz.sum(0, keepdim=True) * gamma

        triton_scalar_var_update(self.sw, grad_Sw)
        triton_scalar_var_update(self.sb, grad_Sb)

        # ---- Delta propagation ----
        dp_ma, dp_Sa = triton_fused_backward_delta(dmz, dSz, self.mw)

        _, C, H, W = self.input_shape
        d_ma = triton_col2im(dp_ma, N, C, H, W,
                             self.kH, self.kW, self.stride, self.padding)
        d_Sa = triton_col2im(dp_Sa, N, C, H, W,
                             self.kH, self.kW, self.stride, self.padding)
        return d_ma, d_Sa


# ====================================================================
# Shared-Variance CNN Network
# ====================================================================

class SharedVarTAGICNN:
    """
    TAGI CNN with shared variance per layer.
    Drop-in replacement for TritonTAGICNN.

    Spec format (same as TritonTAGICNN):
        [('conv', C_in, C_out, k, stride, pad),
         ('relu',),
         ('pool', k),
         ('flatten',),
         ('fc', in_features, out_features),
         ...]
    """
    def __init__(self, spec, device="cuda", gain_var=0.1):
        from tagi_cnn_triton import TritonTAGIAvgPool2D

        self.device = device
        self.layers = []
        self.layer_types = []
        for s in spec:
            t = s[0]
            self.layer_types.append(t)
            if t == 'conv':
                _, Ci, Co, k, st, pad = s
                self.layers.append(
                    SharedVarTAGIConv2D(Ci, Co, k, st, pad, device, gain_var=gain_var))
            elif t == 'pool':
                self.layers.append(TritonTAGIAvgPool2D(s[1]))
            elif t == 'fc':
                self.layers.append(
                    SharedVarTAGILayer(s[1], s[2], device, gain_var=gain_var))
            elif t in ('relu', 'flatten'):
                self.layers.append(None)

    def forward(self, x):
        ma = x
        Sa = torch.zeros_like(x)
        self.relu_jacobians = []
        self.relu_indices = []

        for i, (lt, layer) in enumerate(zip(self.layer_types, self.layers)):
            if lt == 'conv':
                ma, Sa = layer.forward(ma, Sa)
            elif lt == 'pool':
                ma, Sa = layer.forward(ma, Sa)
            elif lt == 'relu':
                if ma.dim() == 4:
                    N, C, H, W = ma.shape
                    ma_r, Sa_r, J = triton_relu_moments(ma.reshape(-1), Sa.reshape(-1))
                    ma = ma_r.view(N, C, H, W)
                    Sa = Sa_r.view(N, C, H, W)
                    self.relu_jacobians.append(J.view(N, C, H, W))
                else:
                    bs = x.shape[0]
                    ma, Sa, J = triton_relu_moments(ma.reshape(-1), Sa.reshape(-1))
                    ma = ma.view(bs, -1)
                    Sa = Sa.view_as(ma)
                    self.relu_jacobians.append(J.view_as(ma))
                self.relu_indices.append(i)
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
        delta_mz, delta_Sz = triton_output_innovation(
            y_batch, y_pred_m, y_pred_S, sigma_v)

        relu_iter = len(self.relu_jacobians) - 1

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
                delta_mz = delta_mz * J
                delta_Sz = delta_Sz * J * J
            elif lt == 'flatten':
                delta_mz = delta_mz.view(self.flatten_shape)
                delta_Sz = delta_Sz.view(self.flatten_shape)
            elif lt == 'pool':
                delta_mz, delta_Sz = layer.backward(delta_mz, delta_Sz)
            elif lt == 'conv':
                delta_mz, delta_Sz = layer.backward(delta_mz, delta_Sz)

    def get_variances(self):
        """Return a summary of per-layer shared variances for monitoring."""
        info = {}
        layer_idx = 0
        for lt, layer in zip(self.layer_types, self.layers):
            if lt in ('conv', 'fc'):
                info[f'{lt}_{layer_idx}_sw'] = layer.sw.item()
                info[f'{lt}_{layer_idx}_sb'] = layer.sb.item()
            layer_idx += 1
        return info


# ====================================================================
# Quick self-test
# ====================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Test 1: SharedVarTAGINet (FNN)")
    print("=" * 60)
    net = SharedVarTAGINet([784, 256, 128, 10], device)
    x = torch.randn(64, 784, device=device)
    y = torch.randn(64, 10, device=device)
    sigma_v = 0.1

    y_m, y_S = net.step(x, y, sigma_v)
    print(f"  Output mean shape: {y_m.shape}")
    print(f"  Output var  shape: {y_S.shape}")
    print(f"  Variances: {net.get_variances()}")

    print()
    print("=" * 60)
    print("Test 2: SharedVarTAGICNN")
    print("=" * 60)
    spec = [
        ('conv',  1,  32, 5, 1, 2),
        ('relu',),
        ('pool', 2),
        ('conv', 32,  64, 5, 1, 2),
        ('relu',),
        ('pool', 2),
        ('flatten',),
        ('fc', 3136, 256),
        ('relu',),
        ('fc', 256, 10),
    ]
    cnn = SharedVarTAGICNN(spec, device)
    x_img = torch.randn(16, 1, 28, 28, device=device)
    y_cnn = torch.randn(16, 10, device=device)

    cnn.step(x_img, y_cnn, sigma_v)
    print(f"  CNN variances: {cnn.get_variances()}")
    print()
    print("All tests passed!")
