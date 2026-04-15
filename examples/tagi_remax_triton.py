"""
TAGI Remax — Triton GPU kernel for MM-Remax moments
=====================================================

Fused kernel: one program per batch item, all K classes vectorized.
Total complexity: O(B · K)  —  no nested loops, just elementwise + 2 reductions.

Outputs: mu_a (mean probs), sigma_a_sq (variance), cov_z_a (covariance Z↔A)
"""

import torch
import triton
import triton.language as tl
import numpy as np
import time


# ═════════════════════════════════════════════════════════════════════
# 1.  Triton kernel
# ═════════════════════════════════════════════════════════════════════

@triton.jit
def _remax_kernel(
    mu_z_ptr, sig_z_sq_ptr,       # (B, K) inputs
    mu_a_ptr, sig_a_sq_ptr,       # (B, K) outputs
    cov_z_a_ptr,                  # (B, K) output
    K,                            # number of classes [runtime]
    stride_b,                     # row stride
    BLOCK_K: tl.constexpr,        # >= K, power-of-2
):
    """One program ↔ one batch item.  All K classes in registers."""

    b    = tl.program_id(0)
    base = b * stride_b
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    EPS          : tl.constexpr = 1e-7
    INV_SQRT_2PI : tl.constexpr = 0.3989422804014327
    INV_SQRT_2   : tl.constexpr = 0.7071067811865476

    # ── load inputs ──
    mu_z     = tl.load(mu_z_ptr    + base + offs, mask=mask, other=0.0)
    sig_z_sq = tl.load(sig_z_sq_ptr + base + offs, mask=mask, other=0.0)
    sig_z    = tl.sqrt(tl.maximum(sig_z_sq, 0.0))
    safe_sig = tl.maximum(sig_z, EPS)

    # ── alpha = mu / sigma,  phi(alpha), Phi(alpha) ──
    alpha = mu_z / safe_sig
    phi_a = tl.exp(-0.5 * alpha * alpha) * INV_SQRT_2PI         # O(K)  1 erf-equivalent
    Phi_a = 0.5 + 0.5 * tl.math.erf(alpha * INV_SQRT_2)        # O(K)  1 erf

    # ── 1. ReLU moments  M = max(0, Z) ──                        O(K)
    mu_m     = tl.maximum(sig_z * phi_a + mu_z * Phi_a, EPS)
    sig_m_sq = tl.maximum(
        (mu_z * mu_z + sig_z_sq) * Phi_a + mu_z * sig_z * phi_a - mu_m * mu_m,
        EPS)
    cov_z_m  = sig_z_sq * Phi_a

    # ── 2. log-space moments of M ──                              O(K)  K logs
    sig_ln_m_sq = tl.log(1.0 + sig_m_sq / (mu_m * mu_m))
    mu_ln_m     = tl.log(mu_m) - 0.5 * sig_ln_m_sq

    # ── 3. sum moments  M̃ = Σ M_k ──                             O(K)  2 reductions
    mu_mt     = tl.sum(tl.where(mask, mu_m, 0.0))               # scalar
    sig_mt_sq = tl.sum(tl.where(mask, sig_m_sq, 0.0))           # scalar

    # ── 4. log-space moments of M̃ ──                             O(1)
    sig_ln_mt_sq = tl.log(1.0 + sig_mt_sq / (mu_mt * mu_mt))
    mu_ln_mt     = tl.log(mu_mt) - 0.5 * sig_ln_mt_sq

    # ── 5. cov in log-space ──                                    O(K)
    cov_ln = tl.log(1.0 + sig_m_sq / (mu_m * mu_mt))

    # ── 6. moments of ln(A) = ln(M) - ln(M̃) ──                  O(K)
    mu_ln_a     = mu_ln_m - mu_ln_mt
    sig_ln_a_sq = tl.maximum(sig_ln_m_sq + sig_ln_mt_sq - 2.0 * cov_ln, EPS)

    # ── 7. final moments of A ──                                  O(K)  K exps
    mu_a = tl.maximum(tl.exp(mu_ln_a + 0.5 * sig_ln_a_sq), EPS)
    # normalize
    mu_a_sum = tl.maximum(tl.sum(tl.where(mask, mu_a, 0.0)), EPS)
    mu_a     = mu_a / mu_a_sum

    sig_a_sq = mu_a * mu_a * (tl.exp(sig_ln_a_sq) - 1.0)

    # ── 8. cov(Z, A) ──                                           O(K)
    cov_z_a = mu_a * cov_z_m * (1.0 / mu_m - 1.0 / mu_mt)

    # ── store ──
    tl.store(mu_a_ptr     + base + offs, mu_a, mask=mask)
    tl.store(sig_a_sq_ptr + base + offs, sig_a_sq, mask=mask)
    tl.store(cov_z_a_ptr  + base + offs, cov_z_a, mask=mask)


# ═════════════════════════════════════════════════════════════════════
# 2.  Python wrapper
# ═════════════════════════════════════════════════════════════════════

def triton_remax(mu_z, sigma_z_sq, return_all=False):
    """
    GPU MM-Remax moments.

    Parameters
    ----------
    mu_z       : Tensor (B, K) or (K,)  means
    sigma_z_sq : Tensor (B, K) or (K,)  variances

    Returns
    -------
    mu_a, sigma_a_sq, cov_z_a : Tensors, same shape as input
    """
    squeeze = mu_z.dim() == 1
    if squeeze:
        mu_z       = mu_z.unsqueeze(0)
        sigma_z_sq = sigma_z_sq.unsqueeze(0)

    mu_z       = mu_z.contiguous()
    sigma_z_sq = sigma_z_sq.contiguous()
    B, K = mu_z.shape

    mu_a     = torch.empty_like(mu_z)
    sig_a_sq = torch.empty_like(mu_z)
    cov_z_a  = torch.empty_like(mu_z)

    BLOCK_K = triton.next_power_of_2(K)
    _remax_kernel[(B,)](
        mu_z, sigma_z_sq,
        mu_a, sig_a_sq, cov_z_a,
        K, mu_z.stride(0),
        BLOCK_K=BLOCK_K,
    )

    if squeeze:
        return mu_a.squeeze(0), sig_a_sq.squeeze(0), cov_z_a.squeeze(0)
    return mu_a, sig_a_sq, cov_z_a


# ═════════════════════════════════════════════════════════════════════
# 3.  NumPy reference (scipy-free, using math.erf)
# ═════════════════════════════════════════════════════════════════════

import math as _math

def _phi_np(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

def _Phi_np(x):
    return 0.5 * (1.0 + np.vectorize(_math.erf)(x / np.sqrt(2.0)))

def numpy_remax(mu_z, sigma_z_sq):
    """NumPy reference for MM-Remax (scipy-free)."""
    eps = 1e-7
    single = mu_z.ndim == 1
    if single:
        mu_z = mu_z[np.newaxis, :]
        sigma_z_sq = sigma_z_sq[np.newaxis, :]

    sig_z = np.sqrt(np.maximum(sigma_z_sq, 0.0))
    alpha = np.divide(mu_z, sig_z, out=np.zeros_like(mu_z), where=sig_z > eps)

    phi_a = _phi_np(alpha)
    Phi_a = _Phi_np(alpha)

    mu_m = np.maximum(sig_z * phi_a + mu_z * Phi_a, eps)
    sig_m_sq = np.maximum(
        (mu_z**2 + sigma_z_sq) * Phi_a + mu_z * sig_z * phi_a - mu_m**2, eps)
    cov_z_m = sigma_z_sq * Phi_a

    sig_ln_m_sq = np.log(1 + sig_m_sq / mu_m**2)
    mu_ln_m = np.log(mu_m) - 0.5 * sig_ln_m_sq

    mu_mt = np.sum(mu_m, axis=-1, keepdims=True)
    sig_mt_sq = np.sum(sig_m_sq, axis=-1, keepdims=True)

    sig_ln_mt_sq = np.log(1 + sig_mt_sq / mu_mt**2)
    mu_ln_mt = np.log(mu_mt) - 0.5 * sig_ln_mt_sq

    cov_ln = np.log(1 + sig_m_sq / (mu_m * mu_mt))

    mu_ln_a = mu_ln_m - mu_ln_mt
    sig_ln_a_sq = np.maximum(sig_ln_m_sq + sig_ln_mt_sq - 2 * cov_ln, eps)

    mu_a = np.maximum(np.exp(mu_ln_a + 0.5 * sig_ln_a_sq), eps)
    mu_a = mu_a / np.maximum(np.sum(mu_a, axis=-1, keepdims=True), eps)
    sig_a_sq = mu_a**2 * (np.exp(sig_ln_a_sq) - 1)
    cov_z_a = mu_a * cov_z_m * (1/mu_m - 1/mu_mt)

    if single:
        return mu_a.squeeze(0), sig_a_sq.squeeze(0), cov_z_a.squeeze(0)
    return mu_a, sig_a_sq, cov_z_a


# ═════════════════════════════════════════════════════════════════════
# 4.  Validation & Benchmark
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from tagi_bernoulli_triton import triton_hermite_moments

    device = torch.device("cuda")

    # ─── Accuracy validation ───
    print("=" * 72)
    print("  VALIDATION:  Triton Remax  vs  NumPy Remax")
    print("=" * 72)

    tests = [
        (np.array([0.2, 0.0, -0.1, 3.0]),   np.array([5.0, 1.0, 0.01, 5.0])),
        (np.array([3.0, 0.0, -0.1, 3.1]),   np.array([0.01, 1.0, 0.01, 30.0])),
        (np.array([0.0, 0.3, -0.3, 0.1]),   np.array([1.0, 1.0, 1.0, 1.0])),
    ]

    for mu_np, var_np in tests:
        mu_t  = torch.tensor(mu_np, dtype=torch.float32, device=device)
        var_t = torch.tensor(var_np, dtype=torch.float32, device=device)

        mu_a_tr, saq_tr, cza_tr = triton_remax(mu_t, var_t)
        mu_a_np, saq_np, cza_np = numpy_remax(mu_np, var_np)

        mu_a_tr = mu_a_tr.cpu().numpy()
        saq_tr  = saq_tr.cpu().numpy()
        cza_tr  = cza_tr.cpu().numpy()

        print(f"\n  mu_z = {mu_np},  sigma_z² = {var_np}")
        hdr = f"{'i':>3}  {'mu_a_tri':>10}  {'mu_a_np':>10}  {'V_tri':>10}  {'V_np':>10}  {'C_tri':>10}  {'C_np':>10}"
        print(hdr)
        print("-" * len(hdr))
        for i in range(len(mu_np)):
            print(f"{i:>3}  {mu_a_tr[i]:>10.6f}  {mu_a_np[i]:>10.6f}  "
                  f"{saq_tr[i]:>10.6f}  {saq_np[i]:>10.6f}  "
                  f"{cza_tr[i]:>10.6f}  {cza_np[i]:>10.6f}")
        print(f"  Max |Triton-NumPy|:  mu_a={np.max(np.abs(mu_a_tr-mu_a_np)):.2e}  "
              f"var={np.max(np.abs(saq_tr-saq_np)):.2e}  "
              f"cov={np.max(np.abs(cza_tr-cza_np)):.2e}")
        print(f"  Sum mu_a:  Triton={mu_a_tr.sum():.8f}  NumPy={mu_a_np.sum():.8f}")

    # ─── Output comparison: Bernoulli vs Remax ───
    print("\n" + "=" * 72)
    print("  OUTPUT COMPARISON:  Bernoulli P_i  vs  Remax mu_a_i")
    print("  (same inputs, different models)")
    print("=" * 72)

    for mu_np, var_np in tests:
        sig_np = np.sqrt(var_np)
        mu_t  = torch.tensor(mu_np, dtype=torch.float32, device=device)
        sig_t = torch.tensor(sig_np, dtype=torch.float32, device=device)
        var_t = torch.tensor(var_np, dtype=torch.float32, device=device)

        P_b, V_b, C_b = triton_hermite_moments(mu_t, sig_t, n_gh=32)
        mu_r, V_r, C_r = triton_remax(mu_t, var_t)

        P_b = P_b.cpu().numpy(); V_b = V_b.cpu().numpy(); C_b = C_b.cpu().numpy()
        mu_r = mu_r.cpu().numpy(); V_r = V_r.cpu().numpy(); C_r = C_r.cpu().numpy()

        print(f"\n  mu = {mu_np},  sigma = {sig_np}")
        hdr = f"{'i':>3}  {'P_bern':>9}  {'P_remax':>9}  {'V_bern':>9}  {'V_remax':>9}  {'C_bern':>9}  {'C_remax':>9}"
        print(hdr)
        print("-" * len(hdr))
        for i in range(len(mu_np)):
            print(f"{i:>3}  {P_b[i]:>9.5f}  {mu_r[i]:>9.5f}  "
                  f"{V_b[i]:>9.5f}  {V_r[i]:>9.5f}  "
                  f"{C_b[i]:>9.5f}  {C_r[i]:>9.5f}")

    # ─── Benchmark: Bernoulli vs Remax ───
    print("\n" + "=" * 72)
    print("  BENCHMARK:  Triton Bernoulli  vs  Triton Remax")
    print("  (warm-up + 100 timed iterations)")
    print("=" * 72)
    print(f"{'n':>6} {'B':>6}  {'Bernoulli':>12}  {'Remax':>12}  {'Speedup':>8}")
    print("-" * 52)

    for n_cls in [4, 10, 100, 1000]:
        for batch in [1, 128, 1024]:
            mu_b  = torch.randn(batch, n_cls, device=device)
            sig_b = (torch.rand(batch, n_cls, device=device) + 0.1)
            var_b = sig_b ** 2

            # warm up
            for _ in range(5):
                triton_hermite_moments(mu_b, sig_b, n_gh=32)
                triton_remax(mu_b, var_b)
            torch.cuda.synchronize()

            N_ITER = 100

            # Bernoulli
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_ITER):
                triton_hermite_moments(mu_b, sig_b, n_gh=32)
            torch.cuda.synchronize()
            dt_bern = (time.perf_counter() - t0) / N_ITER * 1000

            # Remax
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_ITER):
                triton_remax(mu_b, var_b)
            torch.cuda.synchronize()
            dt_remax = (time.perf_counter() - t0) / N_ITER * 1000

            speedup = dt_bern / dt_remax if dt_remax > 0 else float('inf')
            print(f"{n_cls:>6} {batch:>6}  {dt_bern:>10.3f}ms  {dt_remax:>10.3f}ms  {speedup:>7.1f}×")
