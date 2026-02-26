"""
TAGI Bernoulli — Triton GPU kernel for max-indicator moments
=============================================================

Fused kernel:  no O(n² · n_gh) intermediate tensors.
All computation happens in registers — one program per (batch, class).

Complexity per sample : O(n · n_gh) FLOPs
Output memory         : O(B · n)        [no big intermediates]
No scipy in the hot path — only tl.math.erf for Φ.
"""

import torch
import triton
import triton.language as tl
import numpy as np
from numpy.polynomial.hermite import hermgauss
import time

# ═════════════════════════════════════════════════════════════════════
# 1.  GH node cache  (precomputed once per (n_gh, device) pair)
# ═════════════════════════════════════════════════════════════════════

_gh_cache = {}

def _get_gh(n_gh, device, dtype=torch.float32):
    key = (n_gh, str(device), dtype)
    if key not in _gh_cache:
        t, w = hermgauss(n_gh)
        _gh_cache[key] = (
            torch.tensor(t, dtype=dtype, device=device),
            torch.tensor(w, dtype=dtype, device=device),
        )
    return _gh_cache[key]


# ═════════════════════════════════════════════════════════════════════
# 2.  Triton kernel
# ═════════════════════════════════════════════════════════════════════

@triton.jit
def _hermite_kernel(
    mu_ptr, sig_ptr,            # (B, n)  contiguous
    nodes_ptr, weights_ptr,     # (n_gh,)
    P_ptr, EXf_ptr,             # (B, n)  output
    n,                          # number of classes   [runtime]
    n_gh,                       # number of GH nodes  [runtime]
    stride_b,                   # row stride = n for contiguous
    N_GH: tl.constexpr,        # >= n_gh, power-of-2
):
    """One program  ↔  one (batch_item, class_i)."""

    pid   = tl.program_id(0)
    b_idx = pid // n
    i     = pid %  n
    base  = b_idx * stride_b

    # ── pivot class ──
    mu_i  = tl.load(mu_ptr  + base + i)
    sig_i = tl.load(sig_ptr + base + i)
    sig_i = tl.maximum(sig_i, 1e-12)

    # ── GH nodes & weights into registers ──
    gh   = tl.arange(0, N_GH)
    mask = gh < n_gh
    t    = tl.load(nodes_ptr   + gh, mask=mask, other=0.0)
    w    = tl.load(weights_ptr + gh, mask=mask, other=0.0)

    # ── evaluation points  x_k = √2 σ_i t_k + μ_i ──
    SQRT2     : tl.constexpr = 1.4142135623730951
    INV_SQRT2 : tl.constexpr = 0.7071067811865476
    x = SQRT2 * sig_i * t + mu_i                           # (N_GH,)

    # ── accumulate  Σ_j log Φ_j(x_k)  over ALL j (including self) ──
    acc = tl.zeros((N_GH,), dtype=tl.float32)

    for j in range(0, n):
        mu_j  = tl.load(mu_ptr  + base + j)
        sig_j = tl.load(sig_ptr + base + j)
        sig_j = tl.maximum(sig_j, 1e-12)
        z     = (x - mu_j) / sig_j * INV_SQRT2
        cdf   = 0.5 + 0.5 * tl.math.erf(z)
        cdf   = tl.maximum(cdf, 1e-30)
        acc  += tl.log(cdf)

    # ── subtract self-term  j == i ──
    z_s   = (x - mu_i) / sig_i * INV_SQRT2
    cdf_s = 0.5 + 0.5 * tl.math.erf(z_s)
    cdf_s = tl.maximum(cdf_s, 1e-12)
    acc  -= tl.log(cdf_s)

    # ── mask padding ──
    acc = tl.where(mask, acc, 0.0)
    pe  = tl.exp(acc)

    # ── weighted sums  ( ÷ √π ) ──
    INV_SQRT_PI : tl.constexpr = 0.5641895835477563
    ws = w * INV_SQRT_PI

    P_val   = tl.sum(ws * pe)
    EXf_val = tl.sum(ws * x * pe)

    tl.store(P_ptr   + base + i, P_val)
    tl.store(EXf_ptr + base + i, EXf_val)


# ═════════════════════════════════════════════════════════════════════
# 3.  Python wrapper
# ═════════════════════════════════════════════════════════════════════

def triton_hermite_moments(mu, sigma, n_gh=32):
    """
    GPU Gauss-Hermite moments for the TAGI Bernoulli max-indicator.

    Parameters
    ----------
    mu    : Tensor (B, n) or (n,)   means
    sigma : Tensor (B, n) or (n,)   standard deviations
    n_gh  : int                     quadrature order

    Returns
    -------
    P, V, C : Tensors, same shape as input
    """
    squeeze = mu.dim() == 1
    if squeeze:
        mu    = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)

    mu    = mu.contiguous()
    sigma = sigma.contiguous()
    B, n  = mu.shape

    nodes, weights = _get_gh(n_gh, mu.device, mu.dtype)

    P   = torch.empty_like(mu)
    EXf = torch.empty_like(mu)

    N_GH = triton.next_power_of_2(n_gh)
    _hermite_kernel[(B * n,)](
        mu, sigma, nodes, weights, P, EXf,
        n, n_gh, mu.stride(0),
        N_GH=N_GH,
    )

    # normalize + derive variance & covariance
    P = P / P.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    V = P * (1.0 - P)
    C = EXf - mu * P

    if squeeze:
        return P.squeeze(0), V.squeeze(0), C.squeeze(0)
    return P, V, C


# ═════════════════════════════════════════════════════════════════════
# 4.  Fast wrapper with Top-M truncation
# ═════════════════════════════════════════════════════════════════════

def triton_hermite_moments_fast(mu, sigma, n_gh=16, top_m=None):
    """
    Fast GPU Gauss-Hermite moments with optional Top-M truncation.

    Top-M idea:  classes with low means have P_i ≈ 0 to machine precision.
    We select the M most competitive classes (by mean), run the exact kernel
    on just those M, and assign P=V=C=0 to the rest.

    Complexity
    ----------
    Without top_m :  O(B · n² · n_gh)      [same as triton_hermite_moments]
    With top_m=M  :  O(B · (n + M² · n_gh))  [linear in n for fixed M]

    Parameters
    ----------
    mu    : Tensor (B, n) or (n,)
    sigma : Tensor (B, n) or (n,)
    n_gh  : int   (default 16 — half the original, minimal accuracy loss)
    top_m : int or None  (if set, only compute for top M classes)
    """
    squeeze = mu.dim() == 1
    if squeeze:
        mu    = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)

    mu    = mu.contiguous()
    sigma = sigma.contiguous()
    B, n  = mu.shape

    # If top_m not set, or top_m >= n, fall back to full computation
    if top_m is None or top_m >= n:
        P, V, C = triton_hermite_moments(mu, sigma, n_gh=n_gh)
        if squeeze:
            return P.squeeze(0), V.squeeze(0), C.squeeze(0)
        return P, V, C

    M = top_m

    # ── Step 1: select top-M classes by mean  ──  O(B·n)
    _, top_idx = torch.topk(mu, M, dim=-1)             # (B, M)

    # ── Step 2: gather their mu and sigma    ──  O(B·M)
    mu_sub  = torch.gather(mu,    1, top_idx)           # (B, M)
    sig_sub = torch.gather(sigma, 1, top_idx)           # (B, M)

    # ── Step 3: run exact kernel on (B, M)   ──  O(B · M² · n_gh)
    nodes, weights = _get_gh(n_gh, mu.device, mu.dtype)
    P_sub   = torch.empty_like(mu_sub)
    EXf_sub = torch.empty_like(mu_sub)

    N_GH = triton.next_power_of_2(n_gh)
    _hermite_kernel[(B * M,)](
        mu_sub, sig_sub, nodes, weights, P_sub, EXf_sub,
        M, n_gh, mu_sub.stride(0),
        N_GH=N_GH,
    )

    # Normalize within top-M and derive V, C
    P_sub = P_sub / P_sub.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    V_sub = P_sub * (1.0 - P_sub)
    C_sub = EXf_sub - mu_sub * P_sub

    # ── Step 4: scatter back to full shape   ──  O(B·M)
    P = torch.zeros(B, n, device=mu.device, dtype=mu.dtype)
    V = torch.zeros(B, n, device=mu.device, dtype=mu.dtype)
    C = torch.zeros(B, n, device=mu.device, dtype=mu.dtype)
    P.scatter_(1, top_idx, P_sub)
    V.scatter_(1, top_idx, V_sub)
    C.scatter_(1, top_idx, C_sub)

    if squeeze:
        return P.squeeze(0), V.squeeze(0), C.squeeze(0)
    return P, V, C


# ═════════════════════════════════════════════════════════════════════
# 5.  Validation & Benchmark
# ═════════════════════════════════════════════════════════════════════

def _bench(fn, args, warmup=5, iters=100):
    """Time a GPU function (ms/call), excluding JIT."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


if __name__ == "__main__":
    from TAGI_bernouilli import hermite_moments, mc_moments
    from tagi_remax_triton import triton_remax

    device = torch.device("cuda")

    # ─── A) Accuracy: fast vs exact ───
    print("=" * 72)
    print("  ACCURACY:  Bernoulli exact  vs  fast (top_m=3, n_gh=16)")
    print("  (4-class problems — we keep top 3 to show truncation effect)")
    print("=" * 72)

    tests = [
        (np.array([0.2,  0.0, -0.1,  3.0]),  np.array([5.0, 1.0, 0.01, 5.0])),
        (np.array([3.0,  0.0, -0.1,  3.1]),  np.array([0.01, 1.0, 0.01, 30.0])),
        (np.array([0.0,  0.3, -0.3,  0.1]),  np.array([1.0, 1.0, 1.0, 1.0])),
    ]

    for mu_np, sig_np in tests:
        mu_t  = torch.tensor(mu_np,  dtype=torch.float32, device=device)
        sig_t = torch.tensor(sig_np, dtype=torch.float32, device=device)

        P_ex, V_ex, C_ex = triton_hermite_moments(mu_t, sig_t, n_gh=32)
        P_f,  V_f,  C_f  = triton_hermite_moments_fast(mu_t, sig_t, n_gh=16, top_m=3)

        P_ex = P_ex.cpu().numpy();  P_f = P_f.cpu().numpy()
        V_ex = V_ex.cpu().numpy();  V_f = V_f.cpu().numpy()

        print(f"\n  mu={mu_np}  sigma={sig_np}")
        hdr = f"{'i':>3}  {'P_exact':>9}  {'P_fast':>9}  {'V_exact':>9}  {'V_fast':>9}"
        print(hdr)
        print("-" * len(hdr))
        for i in range(len(mu_np)):
            print(f"{i:>3}  {P_ex[i]:>9.5f}  {P_f[i]:>9.5f}  "
                  f"{V_ex[i]:>9.5f}  {V_f[i]:>9.5f}")
        print(f"  Max |exact - fast|  P: {np.max(np.abs(P_ex-P_f)):.2e}  "
              f"V: {np.max(np.abs(V_ex-V_f)):.2e}")

    # ─── B) Large-scale accuracy: n=100, top_m=20 ───
    print("\n" + "=" * 72)
    print("  ACCURACY:  n=100 classes,  exact vs fast (top_m=20, n_gh=16)")
    print("=" * 72)

    torch.manual_seed(42)
    mu_100  = torch.randn(100, device=device)
    sig_100 = torch.rand(100, device=device) + 0.5

    P_ex, _, _ = triton_hermite_moments(mu_100, sig_100, n_gh=32)
    P_f, _, _  = triton_hermite_moments_fast(mu_100, sig_100, n_gh=16, top_m=20)

    P_ex_np = P_ex.cpu().numpy()
    P_f_np  = P_f.cpu().numpy()

    # Show top-10 classes by exact probability
    top10 = np.argsort(-P_ex_np)[:10]
    print(f"\n  Top-10 classes by P_exact:")
    hdr = f"  {'cls':>4}  {'P_exact':>9}  {'P_fast':>9}  {'|diff|':>9}"
    print(hdr)
    for idx in top10:
        diff = abs(P_ex_np[idx] - P_f_np[idx])
        print(f"  {idx:>4}  {P_ex_np[idx]:>9.6f}  {P_f_np[idx]:>9.6f}  {diff:>9.2e}")

    print(f"\n  Sum P:  exact={P_ex_np.sum():.6f}  fast={P_f_np.sum():.6f}")
    print(f"  Max |diff| over all 100 classes: {np.max(np.abs(P_ex_np-P_f_np)):.2e}")
    print(f"  Classes with P_exact > 1e-4: {np.sum(P_ex_np > 1e-4)}")
    print(f"  Classes with P_exact > 1e-6: {np.sum(P_ex_np > 1e-6)}")

    # ─── C) Benchmark: exact vs fast vs remax ───
    print("\n" + "=" * 72)
    print("  BENCHMARK:  Bernoulli exact  vs  fast (top_m=20)  vs  Remax")
    print("=" * 72)
    print(f"{'n':>6} {'B':>6}  {'Exact(32)':>11}  {'Fast(16,M20)':>13}  "
          f"{'Remax':>9}  {'Fast/Exact':>11}  {'Remax/Exact':>12}")
    print("-" * 82)

    for n_cls in [4, 10, 100, 1000]:
        M = min(20, n_cls)
        for batch in [1, 128, 1024]:
            mu_b  = torch.randn(batch, n_cls, device=device)
            sig_b = torch.rand(batch, n_cls, device=device) + 0.1
            var_b = sig_b ** 2

            dt_ex = _bench(triton_hermite_moments, (mu_b, sig_b, 32))
            dt_f  = _bench(triton_hermite_moments_fast, (mu_b, sig_b, 16, M))
            dt_r  = _bench(triton_remax, (mu_b, var_b))

            sp_f = dt_ex / dt_f if dt_f > 0 else float('inf')
            sp_r = dt_ex / dt_r if dt_r > 0 else float('inf')
            print(f"{n_cls:>6} {batch:>6}  {dt_ex:>9.3f}ms  {dt_f:>11.3f}ms  "
                  f"{dt_r:>7.3f}ms  {sp_f:>10.1f}×  {sp_r:>11.1f}×")
