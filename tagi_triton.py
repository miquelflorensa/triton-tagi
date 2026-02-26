"""
Triton Implementation of TAGI with Exact Batch Processing
========================================================================
This version replaces the naive independence assumption with:
  • Conjugate gradient (CG) to solve S^{-1}v without forming S.
  • Randomized diagonal estimation to compute diag(H^T R^{-1} H).
All operations scale as O(N p) per CG iteration, avoiding the O(N³) inversion.
"""

import torch
import triton
import triton.language as tl
import numpy as np

# ====================================================================
# Triton Kernels (unchanged from the original)
# ====================================================================

@triton.jit
def fused_var_forward_kernel(
    ma_ptr, Sa_ptr, mw_ptr, Sw_ptr, Sb_ptr, Sz_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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

        acc += tl.dot(ma_t * ma_t, Sw_t)    # ma² @ Sw
        acc += tl.dot(Sa_t, mw_t * mw_t)    # Sa  @ mw²
        acc += tl.dot(Sa_t, Sw_t)            # Sa  @ Sw

        ma_ptrs += BLOCK_K * stride_ak
        Sa_ptrs += BLOCK_K * stride_ak
        Sw_ptrs += BLOCK_K * stride_wk
        mw_ptrs += BLOCK_K * stride_wk
        offs_k  += BLOCK_K

    sb = tl.load(Sb_ptr + pid_n * BLOCK_N + tl.arange(0, BLOCK_N),
                 mask=(pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) < N, other=0.0)
    acc += sb[None, :]

    c_ptrs = Sz_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def fused_backward_delta_kernel(
    dmz_ptr, dSz_ptr, mw_ptr,
    d_ma_ptr, d_Sa_ptr,
    M, K, N,
    stride_dm, stride_dn,
    stride_wk, stride_wn,
    stride_cm, stride_ck,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
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
    mw_ptrs  = mw_ptr  + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    for _ in range(0, tl.cdiv(N, BLOCK_N)):
        n_mask = offs_n < N
        d_mask = (offs_m[:, None] < M) & n_mask[None, :]
        w_mask = n_mask[:, None] & (offs_k[None, :] < K)

        dmz_t = tl.load(dmz_ptrs, mask=d_mask, other=0.0)
        dSz_t = tl.load(dSz_ptrs, mask=d_mask, other=0.0)
        mw_t  = tl.load(mw_ptrs,  mask=w_mask, other=0.0)

        acc_ma += tl.dot(dmz_t, mw_t)
        acc_Sa += tl.dot(dSz_t, mw_t * mw_t)

        dmz_ptrs += BLOCK_N * stride_dn
        dSz_ptrs += BLOCK_N * stride_dn
        mw_ptrs  += BLOCK_N * stride_wn
        offs_n   += BLOCK_N

    out_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(d_ma_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck,
             acc_ma, mask=out_mask)
    tl.store(d_Sa_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck,
             acc_Sa, mask=out_mask)


@triton.jit
def bayesian_relu_kernel(mz_ptr, Sz_ptr, ma_ptr, Sa_ptr, J_ptr,
                         n_elements, BLOCK: tl.constexpr):
    INV_SQRT_2PI: tl.constexpr = 0.3989422804014327   # 1/√(2π)
    INV_SQRT_2: tl.constexpr   = 0.7071067811865476   # 1/√2

    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    mz = tl.load(mz_ptr + offs, mask=valid, other=0.0)
    Sz = tl.load(Sz_ptr + offs, mask=valid, other=0.0)

    Sz_safe = tl.maximum(Sz, 1e-12)
    sigma_z = tl.sqrt(Sz_safe)
    alpha   = mz / sigma_z

    pdf = tl.exp(-0.5 * alpha * alpha) * INV_SQRT_2PI
    cdf = 0.5 * (1.0 + tl.math.erf(alpha * INV_SQRT_2))

    mu_m = tl.maximum(sigma_z * pdf + mz * cdf, 1e-7)
    var_m = -mu_m * mu_m + 2.0 * mu_m * mz \
            - mz * sigma_z * pdf + (Sz_safe - mz * mz) * cdf
    var_m = tl.maximum(var_m, 1e-7)

    tl.store(ma_ptr + offs, mu_m, mask=valid)
    tl.store(Sa_ptr + offs, var_m, mask=valid)
    tl.store(J_ptr  + offs, cdf, mask=valid)


@triton.jit
def param_update_kernel(m_ptr, S_ptr, gm_ptr, gS_ptr,
                        n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    m_old = tl.load(m_ptr + offs, mask=valid)
    S_old = tl.load(S_ptr + offs, mask=valid)

    gm = tl.load(gm_ptr + offs, mask=valid)
    gS = tl.load(gS_ptr + offs, mask=valid)

    # Mean update
    m_new = m_old + S_old * gm

    # Variance update (precision‑space)
    S_new = S_old / (1.0 - S_old * gS)

    tl.store(m_ptr + offs, m_new, mask=valid)
    tl.store(S_ptr + offs, tl.maximum(S_new, 1e-6), mask=valid)


@triton.jit
def output_innovation_kernel(y_ptr, ym_ptr, yS_ptr, sv_sq,
                             dm_ptr, dS_ptr, n_elements,
                             BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements
    y  = tl.load(y_ptr  + offs, mask=valid)
    ym = tl.load(ym_ptr + offs, mask=valid)
    yS = tl.load(yS_ptr + offs, mask=valid)
    Sy = yS + sv_sq
    tl.store(dm_ptr + offs, (y - ym) / Sy, mask=valid)
    tl.store(dS_ptr + offs, -1.0 / Sy, mask=valid)


# ====================================================================
# Python wrappers (unchanged)
# ====================================================================

BLOCK_EW = 1024

def _pick_block(K, N):
    if K >= 128 and N >= 128:
        return 64, 64, 32
    elif K >= 64 and N >= 64:
        return 32, 32, 32
    else:
        return 16, 16, 16

def triton_fused_var_forward(ma, Sa, mw, Sw, Sb):
    M, K = ma.shape
    _, N = mw.shape
    Sz = torch.empty(M, N, device=ma.device, dtype=ma.dtype)
    BM, BN, BK = _pick_block(K, N)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    fused_var_forward_kernel[grid](
        ma, Sa, mw, Sw, Sb.view(-1), Sz,
        M, N, K,
        ma.stride(0), ma.stride(1),
        mw.stride(0), mw.stride(1),
        Sz.stride(0), Sz.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    return Sz

def triton_relu_moments(mz, Sz):
    n = mz.numel()
    ma = torch.empty_like(mz)
    Sa = torch.empty_like(Sz)
    J  = torch.empty_like(mz)
    bayesian_relu_kernel[(triton.cdiv(n, BLOCK_EW),)](
        mz, Sz, ma, Sa, J, n, BLOCK=BLOCK_EW)
    return ma, Sa, J

def triton_fused_backward_delta(dmz, dSz, mw):
    M, N = dmz.shape
    K = mw.shape[0]
    d_ma = torch.empty(M, K, device=dmz.device, dtype=dmz.dtype)
    d_Sa = torch.empty(M, K, device=dmz.device, dtype=dmz.dtype)
    BM, BK, BN = _pick_block(K, N)
    grid = (triton.cdiv(M, BM) * triton.cdiv(K, BK),)
    fused_backward_delta_kernel[grid](
        dmz, dSz, mw, d_ma, d_Sa,
        M, K, N,
        dmz.stride(0), dmz.stride(1),
        mw.stride(0), mw.stride(1),
        d_ma.stride(0), d_ma.stride(1),
        BLOCK_M=BM, BLOCK_K=BK, BLOCK_N=BN,
    )
    return d_ma, d_Sa

def triton_param_update(m, S, grad_m, grad_S):
    n = m.numel()
    # Ensure inputs are 1‑dimensional views for the kernel
    param_update_kernel[(triton.cdiv(n, BLOCK_EW),)](
        m.view(-1), S.view(-1),
        grad_m.view(-1), grad_S.view(-1),
        n, BLOCK=BLOCK_EW)

def triton_output_innovation(y, ym, yS, sigma_v):
    n = y.numel()
    dm = torch.empty_like(y)
    dS = torch.empty_like(y)
    output_innovation_kernel[(triton.cdiv(n, BLOCK_EW),)](
        y, ym, yS, sigma_v ** 2, dm, dS, n, BLOCK=BLOCK_EW)
    return dm, dS


# ====================================================================
# Triton TAGI Layer (unchanged)
# ====================================================================

class TritonTAGILayer:
    def __init__(self, in_features, out_features, device, gain_mean=2.0, gain_var=0.1):
        self.device = device
        std_mean = np.sqrt(gain_mean / in_features)
        self.mw = torch.randn(in_features, out_features, device=device) * std_mean
        val_var = gain_var / in_features
        self.Sw = torch.full((in_features, out_features), val_var, device=device)
        self.mb = torch.zeros(1, out_features, device=device)
        self.Sb = torch.full((1, out_features), 1e-3, device=device)

    def forward(self, ma, Sa):
        self.ma_in = ma
        self.mz = torch.matmul(ma, self.mw) + self.mb
        self.Sz = triton_fused_var_forward(ma, Sa, self.mw, self.Sw, self.Sb)
        return self.mz, self.Sz

    def backward(self, delta_mz, delta_Sz):
        bs = delta_mz.shape[0]
        BASELINE_BATCH = 32.0
        scale_factor_mean = 1.0 / BASELINE_BATCH

        grad_mw = torch.matmul(self.ma_in.T, delta_mz) * scale_factor_mean
        grad_mb = delta_mz.sum(0, keepdim=True) * scale_factor_mean

        grad_Sw = torch.matmul((self.ma_in ** 2).T, delta_Sz)
        grad_Sb = delta_Sz.sum(0, keepdim=True)

        triton_param_update(self.mw, self.Sw, grad_mw, grad_Sw)
        triton_param_update(self.mb.view(-1), self.Sb.view(-1),
                            grad_mb.view(-1), grad_Sb.view(-1))

        delta_ma, delta_Sa = triton_fused_backward_delta(delta_mz, delta_Sz, self.mw)
        return delta_ma, delta_Sa


# ====================================================================
# Helper functions for flattening / unflattening parameters
# ====================================================================

def get_param_shapes(layers):
    """Return a list of shapes for all parameters (weights then biases)."""
    shapes = []
    for layer in layers:
        shapes.append(layer.mw.shape)   # weights
        shapes.append(layer.mb.shape)   # biases
    return shapes

def flatten_params(layers):
    """Return a vector of all parameter means (for reference) and the shapes."""
    shapes = get_param_shapes(layers)
    flat = []
    for layer in layers:
        flat.append(layer.mw.reshape(-1))
        flat.append(layer.mb.reshape(-1))
    return torch.cat(flat), shapes

def flatten_vars(layers):
    """Return a vector of all parameter variances."""
    flat = []
    for layer in layers:
        flat.append(layer.Sw.reshape(-1))
        flat.append(layer.Sb.reshape(-1))
    return torch.cat(flat)

def unflatten_grads(grad_flat, shapes):
    """Split a flattened gradient vector into per‑layer tensors with the given shapes."""
    grads = []
    idx = 0
    for shape in shapes:
        numel = np.prod(shape)
        grads.append(grad_flat[idx:idx+numel].reshape(shape))
        idx += numel
    return grads


# ====================================================================
# Conjugate Gradient solver (works with any linear operator)
# ====================================================================

def cg_solve(A_func, b, tol=1e-6, max_iter=None):
    """
    Solve A x = b for symmetric positive‑definite A given by a function A_func.
    Returns x.
    """
    x = torch.zeros_like(b)
    r = b - A_func(x)
    p = r.clone()
    rsold = r @ r
    if max_iter is None:
        max_iter = len(b)
    for i in range(max_iter):
        Ap = A_func(p)
        alpha = rsold / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = r @ r
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


# ====================================================================
# Extended TritonTAGINet with exact batch step
# ====================================================================

class TritonTAGINet:
    def __init__(self, layers_struct, device):
        self.layers = []
        self.device = device
        for i in range(len(layers_struct) - 1):
            self.layers.append(TritonTAGILayer(layers_struct[i], layers_struct[i + 1], device))

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

    # ---------- Jacobian‑vector products (needed for exact step) ----------
    def jvp(self, param_deltas):
        """
        param_deltas: list of tensors with same shapes as parameters (weights then biases)
        Returns output perturbation vector (flattened, length N * out_features).
        """
        # Start with zero perturbation at the input
        delta = torch.zeros_like(self.layers[0].ma_in)

        for i, layer in enumerate(self.layers):
            d_mw, d_mb = param_deltas[2*i], param_deltas[2*i+1]   # weights, biases

            # Linear layer: Δz = Δa @ mw + ma @ Δmw + Δmb
            delta_z = delta @ layer.mw + layer.ma_in @ d_mw + d_mb

            if i < len(self.layers) - 1:
                # ReLU: multiply by stored Jacobian Φ(α)
                J = self.jacobians[i]
                delta = delta_z * J
            else:
                # Output layer: no nonlinearity
                out_pert = delta_z

        return out_pert.reshape(-1)   # flatten to vector

    def vjp(self, out_delta_flat):
        """
        out_delta_flat: flattened vector of length N * out_features.
        Returns flattened gradient vector (length total_params) = H^T * out_delta.
        """
        out_delta = out_delta_flat.reshape(-1, self.layers[-1].mb.shape[1])
        delta = out_delta
        grads_flat = []

        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]
            J = self.jacobians[i]

            # Multiply by ReLU Jacobian (applies to output delta before backprop)
            delta = delta * J

            # Gradients for this layer
            g_mw = layer.ma_in.T @ delta
            g_mb = delta.sum(0, keepdim=True)

            grads_flat.append(g_mw.reshape(-1))
            grads_flat.append(g_mb.reshape(-1))

            # Propagate to previous layer
            delta = delta @ layer.mw.T

        # Reverse to match forward parameter order (weights then biases for each layer)
        return torch.cat(grads_flat[::-1])

    # ---------- Exact batch step ----------
    def exact_step(self, x_batch, y_batch, sigma_v, cg_tol=1e-6, cg_max_iter=None,
                   m_rand=20, use_cg=True):
        """
        Perform an exact batch update using CG for the mean and randomized
        diagonal estimation for the variance.
        """
        # Forward pass (stores activations and Jacobians)
        y_pred_m, y_pred_S = self.forward(x_batch)

        # Flatten the innovation vector v = y - μ_Z
        v = (y_batch - y_pred_m).reshape(-1)

        # Get shapes and flattened variances
        shapes = get_param_shapes(self.layers)
        V_flat = flatten_vars(self.layers)   # diagonal of V

        # ---------- Mean update via CG ----------
        def A_func(x_flat):
            """Matrix‑vector product with S = H V H^T + σ_v² I."""
            # x_flat is a flattened vector of length N*out_features
            # 1. H^T x
            grad = self.vjp(x_flat)                 # shape (p,)
            # 2. Multiply by V (diagonal)
            V_grad = V_flat * grad                   # shape (p,)
            # 3. H (V H^T x)
            V_grad_list = unflatten_grads(V_grad, shapes)
            out = self.jvp(V_grad_list)              # shape (N*out_features,)
            # 4. Add noise term
            return out + sigma_v**2 * x_flat

        if use_cg:
            alpha = cg_solve(A_func, v, tol=cg_tol, max_iter=cg_max_iter)
        else:
            # fallback to the old approximate method (diagonal S)
            # This is just for completeness; normally we would use CG.
            Sy_diag = (y_pred_S + sigma_v**2).reshape(-1)
            alpha = v / Sy_diag

        # Mean gradient: gm = H^T α
        grad_mean_flat = self.vjp(alpha)

        # ---------- Variance update via randomized diagonal estimation ----------
        # Estimate diag(H^T H) using Hutchinson's method
        total_params = len(V_flat)
        sum_zv = torch.zeros(total_params, device=self.device)

        for _ in range(m_rand):
            z = (torch.randint(0, 2, (total_params,), device=self.device) * 2 - 1).float()   # ±1
            z_list = unflatten_grads(z, shapes)
            w = self.jvp(z_list)                     # H z
            v_vec = self.vjp(w)                       # H^T H z
            sum_zv += z * v_vec

        diag_HTH = sum_zv / m_rand
        # Diagonal increment for precision: Δ = diag(H^T R^{-1} H) = diag(H^T H) / σ_v²
        delta_diag = diag_HTH / sigma_v**2
        # gS for the kernel is -Δ (because the kernel adds negative of this to the variance)
        grad_var_flat = -delta_diag

        # ---------- Update parameters with the exact gradients ----------
        grad_mean_list = unflatten_grads(grad_mean_flat, shapes)
        grad_var_list = unflatten_grads(grad_var_flat, shapes)

        for i, layer in enumerate(self.layers):
            # weights
            triton_param_update(layer.mw, layer.Sw,
                                grad_mean_list[2*i], grad_var_list[2*i])
            # biases
            triton_param_update(layer.mb.view(-1), layer.Sb.view(-1),
                                grad_mean_list[2*i+1].view(-1), grad_var_list[2*i+1].view(-1))

        # Return predictions (for monitoring)
        return y_pred_m, y_pred_S

    # (Optional) keep the old step method for comparison
    def step(self, x_batch, y_batch, sigma_v):
        y_pred_m, y_pred_S = self.forward(x_batch)
        delta_mz, delta_Sz = triton_output_innovation(y_batch, y_pred_m, y_pred_S, sigma_v)

        for i in reversed(range(len(self.layers))):
            J = self.jacobians[i]
            dm = delta_mz * J
            ds = delta_Sz * J * J
            delta_mz, delta_Sz = self.layers[i].backward(dm, ds)
        return y_pred_m, y_pred_S


# ====================================================================
# Example usage (commented out)
# ====================================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TritonTAGINet([784, 256, 128, 10], device)
    x = torch.randn(64, 784, device=device)
    y = torch.randn(64, 10, device=device)
    sigma_v = 0.1

    # Exact batch update
    y_pred_m, y_pred_S = net.exact_step(x, y, sigma_v, m_rand=10)
    print("Exact step done. Mean prediction shape:", y_pred_m.shape)