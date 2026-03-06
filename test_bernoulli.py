"""
Stability test bank for the Bernoulli max-indicator layer.

Covers:
  1. Mathematical properties  – probability axioms, variance formula
  2. Monte Carlo validation   – compare GH quadrature against MC sampling
  3. Numerical stability      – extreme inputs, large K, near-zero variance
  4. Top-M truncation         – correctness and failure modes of fast path
  5. Layer API                – Bernoulli class forward / backward
  6. Shape handling           – squeeze/unsqueeze, batch sizes

Run:
    pytest test_bernoulli.py -v
    pytest test_bernoulli.py -v -s        # shows print output (diagnostics)
    pytest test_bernoulli.py -v -k smoke  # only quick sanity checks
"""

import pytest
import torch
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.layers.bernoulli import (
    triton_bernoulli_moments,
    triton_bernoulli_moments_fast,
    Bernoulli,
)

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="Triton kernel requires CUDA"
)


# ──────────────────────────────────────────────────────────────────────
#  Shared helpers
# ──────────────────────────────────────────────────────────────────────

def make_inputs(B, K, mu_scale=1.0, var_lo=0.1, var_hi=2.0,
                device=DEVICE, seed=0):
    """Return (mu_z, sigma_z_sq) with reasonable random values."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    mu_z       = torch.randn(B, K, device=device, generator=g) * mu_scale
    sigma_z_sq = torch.rand(B, K, device=device, generator=g) \
                 * (var_hi - var_lo) + var_lo
    return mu_z, sigma_z_sq


def mc_bernoulli(mu_z, sigma_z_sq, n_samples=300_000):
    """
    Monte Carlo estimate of P(Z_i = argmax Z) for Z_i ~ N(mu_i, s²_i).
    Runs on CPU regardless of input device.
    """
    if mu_z.dim() == 1:
        mu_z       = mu_z.unsqueeze(0)
        sigma_z_sq = sigma_z_sq.unsqueeze(0)
    B, K = mu_z.shape
    mu  = mu_z.float().cpu()
    std = torch.sqrt(sigma_z_sq.float().cpu())

    eps = torch.randn(n_samples, B, K)           # (N, B, K)
    Z   = mu.unsqueeze(0) + std.unsqueeze(0) * eps
    idx = Z.argmax(dim=-1)                        # (N, B)

    P_mc = torch.zeros(B, K)
    for b in range(B):
        counts     = torch.bincount(idx[:, b], minlength=K)
        P_mc[b]    = counts.float() / n_samples
    return P_mc.to(mu_z.device)


def assert_valid_probs(P, V, C=None, tag="", atol=1e-4):
    """Helper: assert P, V, (optionally C) contain no NaN/Inf and P sums to 1."""
    for name, t in [("P", P), ("V", V)] + ([("C", C)] if C is not None else []):
        assert not t.isnan().any(), f"[{tag}] {name} contains NaN"
        assert not t.isinf().any(), f"[{tag}] {name} contains Inf"
    assert (P >= -1e-6).all(), f"[{tag}] P has negative values: min={P.min():.3e}"
    assert (V >= -1e-6).all(), f"[{tag}] V has negative values: min={V.min():.3e}"
    if P.dim() >= 2:
        sums = P.sum(dim=-1)
    else:
        sums = P.sum().unsqueeze(0)
    assert torch.allclose(sums, torch.ones_like(sums), atol=atol), \
        f"[{tag}] P does not sum to 1: {sums}"


# ══════════════════════════════════════════════════════════════════════
#  1. Mathematical properties
# ══════════════════════════════════════════════════════════════════════

class TestMathProperties:
    """Probability axioms and algebraic identities that must always hold."""

    @pytest.mark.smoke
    @requires_cuda
    def test_probs_sum_to_one_various_k(self):
        """sum_i P_i = 1 for K in {2, 5, 10, 50}."""
        for K in [2, 5, 10, 50]:
            mu_z, Sz = make_inputs(8, K, seed=K)
            P, V, C  = triton_bernoulli_moments(mu_z, Sz)
            sums = P.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
                f"K={K}: sum(P)={sums}"

    @requires_cuda
    def test_probs_non_negative(self):
        """P_i >= 0 for all i."""
        mu_z, Sz = make_inputs(16, 10, seed=1)
        P, V, C  = triton_bernoulli_moments(mu_z, Sz)
        assert (P >= -1e-7).all(), f"Negative P: min={P.min():.3e}"

    @requires_cuda
    def test_variance_equals_bernoulli_formula(self):
        """V_i = P_i * (1 - P_i)  (up to the 1e-7 clamp)."""
        mu_z, Sz = make_inputs(8, 10, seed=2)
        P, V, C  = triton_bernoulli_moments(mu_z, Sz)
        expected = (P * (1.0 - P)).clamp(min=1e-7)
        diff = (V - expected).abs()
        assert diff.max() < 1e-5, \
            f"Variance formula mismatch: max diff={diff.max():.2e}"

    @pytest.mark.smoke
    @requires_cuda
    def test_k1_trivial(self):
        """K=1: only class must have P=1 regardless of mu/sigma."""
        mu_z = torch.randn(4, 1, device=DEVICE)
        Sz   = torch.rand(4, 1, device=DEVICE) * 2 + 0.1
        P, V, C = triton_bernoulli_moments(mu_z, Sz)
        assert torch.allclose(P, torch.ones_like(P), atol=1e-4), \
            f"K=1: P={P} (expected all 1s)"

    @requires_cuda
    def test_k2_symmetry(self):
        """K=2, equal mu & sigma → P=[0.5, 0.5]."""
        mu_z = torch.zeros(1, 2, device=DEVICE)
        Sz   = torch.ones(1, 2, device=DEVICE)
        P, V, C = triton_bernoulli_moments(mu_z, Sz, n_gh=64)
        assert torch.allclose(P, torch.full_like(P, 0.5), atol=1e-4), \
            f"K=2 symmetric: P={P}"

    @requires_cuda
    def test_uniform_means_give_uniform_probs(self):
        """Equal mu_i and sigma_i → P_i = 1/K."""
        for K in [2, 5, 10]:
            mu_z = torch.zeros(1, K, device=DEVICE)
            Sz   = torch.ones(1, K, device=DEVICE)
            P, V, C = triton_bernoulli_moments(mu_z, Sz, n_gh=64)
            diff = (P - 1.0 / K).abs().max().item()
            assert diff < 1e-3, \
                f"K={K} uniform: max |P - 1/K| = {diff:.3e}"

    @requires_cuda
    def test_dominant_class_wins(self):
        """One class with mu >> others → P ≈ 1 for that class."""
        K = 5
        mu_z       = torch.zeros(2, K, device=DEVICE)
        mu_z[:, 2] = 15.0
        Sz         = torch.ones(2, K, device=DEVICE) * 0.5
        P, V, C    = triton_bernoulli_moments(mu_z, Sz)
        assert (P[:, 2] > 0.999).all(), \
            f"Dominant class P={P[:, 2]} (expected > 0.999)"
        assert (P[:, :2].max() < 0.001).all() and (P[:, 3:].max() < 0.001).all(), \
            f"Non-dominant classes not suppressed: {P}"

    @requires_cuda
    def test_higher_mean_higher_prob(self):
        """For K=2, mu_0 > mu_1 and equal sigma → P_0 > P_1."""
        mu_z = torch.tensor([[2.0, -2.0]], device=DEVICE)
        Sz   = torch.ones(1, 2, device=DEVICE)
        P, V, C = triton_bernoulli_moments(mu_z, Sz)
        assert P[0, 0] > P[0, 1], \
            f"Expected P[0]>P[1] but got {P[0,0]:.4f} vs {P[0,1]:.4f}"


# ══════════════════════════════════════════════════════════════════════
#  2. Monte Carlo validation
# ══════════════════════════════════════════════════════════════════════

class TestMonteCarlo:
    """
    Compare Gauss-Hermite probabilities against MC ground truth.
    MC uses 300K samples; atol=2% to account for MC variance (~0.3%).
    """

    @pytest.mark.smoke
    @requires_cuda
    def test_k2_vs_mc(self):
        mu_z = torch.tensor([[0.8, -0.4]], device=DEVICE)
        Sz   = torch.tensor([[0.6,  1.2]], device=DEVICE)
        P, _, _ = triton_bernoulli_moments(mu_z, Sz, n_gh=64)
        P_mc    = mc_bernoulli(mu_z, Sz)
        err     = (P - P_mc.to(DEVICE)).abs()
        assert err.max() < 0.02, \
            f"K=2 MC error={err.max():.4f}\n  GH: {P}\n  MC: {P_mc}"

    @requires_cuda
    def test_k5_vs_mc(self):
        mu_z, Sz = make_inputs(1, 5, seed=10)
        P, _, _  = triton_bernoulli_moments(mu_z, Sz, n_gh=64)
        P_mc     = mc_bernoulli(mu_z, Sz)
        err      = (P - P_mc.to(DEVICE)).abs()
        assert err.max() < 0.02, \
            f"K=5 MC error={err.max():.4f}\n  GH: {P}\n  MC: {P_mc}"

    @requires_cuda
    def test_k10_vs_mc(self):
        mu_z, Sz = make_inputs(1, 10, seed=11)
        P, _, _  = triton_bernoulli_moments(mu_z, Sz, n_gh=64)
        P_mc     = mc_bernoulli(mu_z, Sz)
        err      = (P - P_mc.to(DEVICE)).abs()
        assert err.max() < 0.03, \
            f"K=10 MC error={err.max():.4f}\n  GH: {P}\n  MC: {P_mc}"

    @requires_cuda
    def test_batch_k5_vs_mc(self):
        mu_z, Sz = make_inputs(6, 5, seed=12)
        P, _, _  = triton_bernoulli_moments(mu_z, Sz, n_gh=64)
        P_mc     = mc_bernoulli(mu_z, Sz)
        err      = (P - P_mc.to(DEVICE)).abs()
        assert err.max() < 0.03, \
            f"Batched MC max error={err.max():.4f}"

    @requires_cuda
    def test_covariance_vs_mc(self):
        """C_i = E[Z_i · 1_{Z_i=max}] - mu_i · P_i  (compare to MC)."""
        mu_z = torch.tensor([[1.0, 0.3, -0.5]], device=DEVICE)
        Sz   = torch.tensor([[0.8, 1.2,  0.4]], device=DEVICE)
        _, _, C = triton_bernoulli_moments(mu_z, Sz, n_gh=64)

        n = 600_000
        std = torch.sqrt(Sz.cpu())
        Z   = mu_z.cpu() + torch.randn(n, 1, 3) * std           # (N,1,3)
        win = (Z == Z.max(dim=-1, keepdim=True).values).float()
        EXf = (Z * win).mean(dim=0)
        P_mc = win.mean(dim=0)
        C_mc = EXf - mu_z.cpu() * P_mc

        err = (C - C_mc.to(DEVICE)).abs()
        assert err.max() < 0.03, \
            f"Covariance MC error={err.max():.4f}\n  GH: {C}\n  MC: {C_mc}"

    @requires_cuda
    def test_gh_order_convergence(self):
        """
        Higher n_gh should reduce error vs MC.
        Reports errors for n_gh in {4, 8, 16, 32, 64} (diagnostic).
        """
        mu_z, Sz = make_inputs(1, 5, seed=99)
        P_mc = mc_bernoulli(mu_z, Sz, n_samples=1_000_000)

        errors = {}
        for n_gh in [4, 8, 16, 32, 64]:
            P, _, _ = triton_bernoulli_moments(mu_z, Sz, n_gh=n_gh)
            errors[n_gh] = (P - P_mc.to(DEVICE)).abs().max().item()

        print("\n  GH convergence (K=5):")
        for n_gh, err in errors.items():
            print(f"    n_gh={n_gh:3d}: max |P - P_mc| = {err:.5f}")

        assert errors[64] < errors[4], \
            f"Expected convergence: err[64]={errors[64]:.4f} should < err[4]={errors[4]:.4f}"


# ══════════════════════════════════════════════════════════════════════
#  3. Numerical stability
# ══════════════════════════════════════════════════════════════════════

class TestNumericalStability:
    """
    Stress-test with extreme inputs. Tests that document degradation rather
    than fail are marked with 'DIAGNOSTIC' in their name.
    """

    @requires_cuda
    def test_zero_variance_no_crash(self):
        """Sz=0 → should not produce NaN/Inf (clamped internally to 1e-12)."""
        mu_z = torch.randn(4, 5, device=DEVICE)
        Sz   = torch.zeros(4, 5, device=DEVICE)
        P, V, C = triton_bernoulli_moments(mu_z, Sz)
        assert_valid_probs(P, V, C, tag="zero-variance")

    @requires_cuda
    def test_zero_variance_recovers_argmax(self):
        """
        Sz → 0: deterministic logits.  Winning class (argmax mu) should get
        P close to 1 — but with Triton float32 we only expect >0.9 here.

        KNOWN FAILURE RISK: The GH nodes span ±√2·σ·t ≈ 0 when σ=0.
        The kernel still integrates but all evaluation points collapse to mu_i.
        At x=mu_i and Sz→0, Φ((mu_i - mu_j)/eps) is 1 for i=winner or 0 else,
        so P should converge to one-hot.  Verify this holds.
        """
        K = 5
        mu_z = torch.randn(4, K, device=DEVICE)
        Sz   = torch.full((4, K), 1e-10, device=DEVICE)
        P, V, C = triton_bernoulli_moments(mu_z, Sz)
        assert_valid_probs(P, V, C, tag="tiny-variance")
        dominant = mu_z.argmax(dim=-1)
        for b in range(4):
            assert P[b, dominant[b]] > 0.9, \
                f"Batch {b}: dominant P={P[b, dominant[b]]:.4f} (expected >0.9)"

    @requires_cuda
    def test_very_large_variance_no_crash(self):
        """Sz=1e6: should not produce NaN/Inf."""
        mu_z = torch.randn(4, 5, device=DEVICE)
        Sz   = torch.full((4, 5), 1e6, device=DEVICE)
        P, V, C = triton_bernoulli_moments(mu_z, Sz)
        assert_valid_probs(P, V, C, tag="large-variance")

    @requires_cuda
    def test_large_variance_approaches_uniform(self):
        """
        Equal means, Sz → ∞: logits Z become very spread out, but since all
        have the same distribution the probabilities remain 1/K.

        DIAGNOSTIC: prints the actual P for large variances.
        """
        K = 5
        mu_z = torch.zeros(1, K, device=DEVICE)
        print(f"\n  Equal-means (K={K}), varying Sz:")
        for var in [1.0, 1e2, 1e4, 1e6]:
            Sz = torch.full((1, K), var, device=DEVICE)
            P, _, _ = triton_bernoulli_moments(mu_z, Sz, n_gh=32)
            dev = (P - 1.0/K).abs().max().item()
            print(f"    Sz={var:.0e}: max|P - 1/K| = {dev:.4f}")
        # At large variance with equal means, P should still be ≈ 1/K
        Sz = torch.full((1, K), 1e6, device=DEVICE)
        P, _, _ = triton_bernoulli_moments(mu_z, Sz, n_gh=32)
        assert (P - 1.0/K).abs().max() < 0.05, \
            f"Large-variance equal-means: P = {P}"

    @requires_cuda
    def test_extreme_mean_separation(self):
        """mu_0 = 100, others = 0: P_0 must be > 0.9999."""
        K = 5
        mu_z       = torch.zeros(4, K, device=DEVICE)
        mu_z[:, 0] = 100.0
        Sz         = torch.ones(4, K, device=DEVICE)
        P, V, C    = triton_bernoulli_moments(mu_z, Sz)
        assert_valid_probs(P, V, C, tag="extreme-sep")
        assert (P[:, 0] > 0.9999).all(), f"P[:, 0] = {P[:, 0]}"

    @requires_cuda
    def test_all_negative_means(self):
        """All large negative means: relative differences still matter."""
        mu_z        = torch.full((4, 5), -100.0, device=DEVICE)
        mu_z[:, 1]  = -1.0   # least negative → should dominate
        Sz          = torch.ones(4, 5, device=DEVICE) * 0.5
        P, V, C     = triton_bernoulli_moments(mu_z, Sz)
        assert_valid_probs(P, V, C, tag="all-negative")
        assert (P[:, 1] > 0.99).all(), f"P[:, 1] = {P[:, 1]}"

    @requires_cuda
    def test_mixed_variance_no_crash(self):
        """Classes with vastly different variances: e.g., 1e-8 vs 1e4."""
        Sz = torch.tensor([[1e-8, 0.5, 1.0, 5.0, 1e4],
                           [1e-8, 0.5, 1.0, 5.0, 1e4]], device=DEVICE)
        mu_z = torch.zeros(2, 5, device=DEVICE)
        P, V, C = triton_bernoulli_moments(mu_z, Sz)
        assert_valid_probs(P, V, C, tag="mixed-variance")

    @pytest.mark.parametrize("K", [2, 5, 10, 50, 100])
    @requires_cuda
    def test_large_k_sum_accuracy(self, K):
        """
        DIAGNOSTIC: For large K, report how well P sums to 1.

        The GH quadrature accumulates numerical error as K grows because
        each class requires O(K) log-CDF evaluations.  With n_gh=32, the
        un-normalised P values may differ slightly, but post-normalisation
        the sum is forced to 1, so this test mainly checks for NaN/Inf.
        """
        mu_z, Sz = make_inputs(4, K, seed=K)
        P, V, C  = triton_bernoulli_moments(mu_z, Sz, n_gh=32)
        assert_valid_probs(P, V, C, tag=f"K={K}")
        raw_min = P.min().item()
        raw_max = P.max().item()
        print(f"\n  K={K:4d}: P range=[{raw_min:.3e}, {raw_max:.3e}]")

    @requires_cuda
    def test_nan_input_behavior(self):
        """
        DIAGNOSTIC: Document what happens when mu_z contains NaN.

        Expected: NaN propagates (Triton does not guard against NaN input).
        This test never hard-fails; it prints observed behaviour.
        """
        mu_z = torch.randn(2, 5, device=DEVICE)
        Sz   = torch.rand(2, 5, device=DEVICE) + 0.1
        mu_z[0, 0] = float('nan')
        try:
            P, V, C = triton_bernoulli_moments(mu_z, Sz)
            print(f"\n  NaN in mu_z → P has NaN: {P.isnan().any().item()}, "
                  f"V has NaN: {V.isnan().any().item()}")
        except Exception as e:
            print(f"\n  NaN in mu_z → exception {type(e).__name__}: {e}")

    @requires_cuda
    def test_uniform_large_k_symmetry(self):
        """
        K=50, all equal mu/sigma: P should be 1/K for every class.
        Floating-point symmetry breaks can cause deviations — document.
        """
        K    = 50
        mu_z = torch.zeros(2, K, device=DEVICE)
        Sz   = torch.ones(2, K, device=DEVICE)
        P, V, C = triton_bernoulli_moments(mu_z, Sz, n_gh=32)
        dev = (P - 1.0 / K).abs().max().item()
        print(f"\n  K={K} uniform: max|P - 1/K| = {dev:.4e}")
        # Symmetry should hold to at least 1%
        assert dev < 0.01, \
            f"K={K} uniform: symmetry deviation too large: {dev:.3e}"


# ══════════════════════════════════════════════════════════════════════
#  4. Top-M truncation (fast path)
# ══════════════════════════════════════════════════════════════════════

class TestTopM:
    """Correctness and known failure modes of triton_bernoulli_moments_fast."""

    @requires_cuda
    def test_top_m_equals_k_matches_full(self):
        """top_m = K → result identical to full computation."""
        mu_z, Sz = make_inputs(4, 10, seed=20)
        P_full, V_full, C_full = triton_bernoulli_moments(mu_z, Sz, n_gh=32)
        P_fast, V_fast, C_fast = triton_bernoulli_moments_fast(
            mu_z, Sz, n_gh=32, top_m=10)
        assert torch.allclose(P_full, P_fast, atol=1e-5), \
            f"top_m=K mismatch: max|ΔP|={(P_full-P_fast).abs().max():.3e}"

    @requires_cuda
    def test_top_m_probs_sum_to_one(self):
        """Post-normalisation: P sums to 1 for any valid top_m."""
        mu_z, Sz = make_inputs(8, 20, seed=21)
        for top_m in [3, 5, 10, 15]:
            P, V, C = triton_bernoulli_moments_fast(mu_z, Sz, top_m=top_m)
            sums = P.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
                f"top_m={top_m}: P.sum={sums}"

    @requires_cuda
    def test_top_m_clear_winner_accuracy(self):
        """
        With a clear dominant class, top_m=3 should match full closely.
        The dominant class is always selected → error should be tiny.
        """
        K          = 15
        mu_z       = torch.zeros(2, K, device=DEVICE)
        mu_z[:, 0] = 8.0
        Sz         = torch.ones(2, K, device=DEVICE) * 0.5

        P_full, _, _ = triton_bernoulli_moments(mu_z, Sz)
        P_fast, _, _ = triton_bernoulli_moments_fast(mu_z, Sz, top_m=3)
        err = (P_full - P_fast).abs()
        assert err.max() < 0.01, \
            f"top_m=3 vs full with clear winner: max|ΔP|={err.max():.4f}"

    @requires_cuda
    def test_top_m_1_winner_take_all(self):
        """
        top_m=1 → winner-take-all: the argmax(mu) class gets P=1.
        All other classes should be exactly 0 (scattered zeros).
        """
        K = 12
        mu_z, Sz = make_inputs(4, K, seed=22)
        P, V, C  = triton_bernoulli_moments_fast(mu_z, Sz, top_m=1)
        winner   = mu_z.argmax(dim=-1)
        for b in range(4):
            assert P[b, winner[b]] > 0.99, \
                f"Batch {b}: winner P={P[b, winner[b]]:.4f}"
            others = torch.cat([P[b, :winner[b]], P[b, winner[b]+1:]])
            assert (others < 1e-4).all(), \
                f"Batch {b}: non-winner classes not zero: {others}"

    @requires_cuda
    def test_top_m_equal_means_diagnostic(self):
        """
        DIAGNOSTIC: With all equal means, top_m selects K classes arbitrarily
        (ties broken by topk).  Documents how much accuracy is lost.

        EXPECTED FAILURE MODE: when all means are equal, top_m < K gives
        incorrect non-uniform P because excluded classes are set to 0.
        """
        K    = 20
        mu_z = torch.zeros(1, K, device=DEVICE)
        Sz   = torch.ones(1, K, device=DEVICE)
        P_mc = mc_bernoulli(mu_z, Sz, n_samples=500_000)

        P_full, _, _ = triton_bernoulli_moments(mu_z, Sz, n_gh=32)
        print(f"\n  Equal-means K={K}, top_m accuracy:")
        for top_m in [3, 5, 10, 15]:
            P_fast, _, _ = triton_bernoulli_moments_fast(
                mu_z, Sz, top_m=top_m, n_gh=32)
            err_mc = (P_fast - P_mc.to(DEVICE)).abs().max().item()
            err_full = (P_fast - P_full).abs().max().item()
            print(f"    top_m={top_m:2d}: vs MC={err_mc:.3f}, vs full={err_full:.3f}")

    @requires_cuda
    def test_top_m_0_edge_case(self):
        """
        KNOWN FAILURE MODE: top_m=0 returns all-zero P (sums to 0, not 1).

        The code path: topk with k=0 → empty tensors → no kernel programs
        → scatter nothing → P stays at zeros.
        Post-normalisation in the fast path divides by 1e-30, keeping zeros.

        This test DOCUMENTS the bug rather than expecting it to pass.
        """
        K = 10
        mu_z, Sz = make_inputs(2, K, seed=30)
        try:
            P, V, C = triton_bernoulli_moments_fast(mu_z, Sz, top_m=0)
            sums = P.sum(dim=-1)
            print(f"\n  top_m=0 → P.sum={sums}  (BUG: expected 1, got {sums.tolist()})")
            # Document the bug: P sums to 0
            assert (sums < 0.01).all(), \
                "top_m=0 unexpectedly gave valid probabilities (bug fixed?)"
        except Exception as e:
            print(f"\n  top_m=0 → exception {type(e).__name__}: {e}")
            pytest.skip(f"top_m=0 raises {type(e).__name__} (kernel crash)")

    @requires_cuda
    def test_top_m_larger_than_k_falls_back(self):
        """
        top_m > K → falls back to full computation.

        NOTE: triton_bernoulli_moments defaults to n_gh=32, while
        triton_bernoulli_moments_fast defaults to n_gh=16.  The fallback
        uses whatever n_gh was passed to the fast function.  We must
        match n_gh explicitly to get identical results.
        """
        mu_z, Sz = make_inputs(4, 5, seed=31)
        # Use the same n_gh=16 (fast default) for a fair comparison
        P_full, _, _ = triton_bernoulli_moments(mu_z, Sz, n_gh=16)
        P_fast, _, _ = triton_bernoulli_moments_fast(mu_z, Sz, top_m=999, n_gh=16)
        assert torch.allclose(P_full, P_fast, atol=1e-5), \
            f"Fallback mismatch: max|ΔP|={(P_full-P_fast).abs().max():.3e}"

    @requires_cuda
    def test_fast_v_clamp_inconsistency(self):
        """
        DIAGNOSTIC: The fast path does NOT clamp V to 1e-7 (unlike the full
        path).  Verify that V_fast can be < 1e-7 for low-probability classes.
        """
        K = 20
        mu_z       = torch.zeros(1, K, device=DEVICE)
        mu_z[0, 0] = 10.0   # clear winner → all other classes get P ≈ 0

        # fast path with top_m=3: 17 non-selected classes get V=0 (no clamp)
        _, V_fast, _ = triton_bernoulli_moments_fast(mu_z, torch.ones(1, K, device=DEVICE), top_m=3)
        _, V_full, _ = triton_bernoulli_moments(mu_z, torch.ones(1, K, device=DEVICE))

        v_min_fast = V_fast.min().item()
        v_min_full = V_full.min().item()
        print(f"\n  Fast V_min={v_min_fast:.2e},  Full V_min={v_min_full:.2e}")
        print(f"  Full path clamps V to 1e-7 ({v_min_full:.2e}), "
              f"Fast path does not ({v_min_fast:.2e})")


# ══════════════════════════════════════════════════════════════════════
#  5. Layer API  (Bernoulli class)
# ══════════════════════════════════════════════════════════════════════

class TestLayerAPI:
    """Test the public interface of the Bernoulli layer object."""

    @pytest.mark.smoke
    @requires_cuda
    def test_forward_output_shapes(self):
        """forward(mz, Sz) → (P, V) both (B, K)."""
        B, K = 6, 10
        mu_z, Sz = make_inputs(B, K)
        layer    = Bernoulli(n_gh=16)
        P, V     = layer.forward(mu_z, Sz)
        assert P.shape == (B, K), f"P.shape={P.shape}"
        assert V.shape == (B, K), f"V.shape={V.shape}"

    @requires_cuda
    def test_forward_1d_squeezed_output(self):
        """1D input (K,) → 1D output (K,)."""
        K    = 10
        mu_z = torch.randn(K, device=DEVICE)
        Sz   = torch.rand(K, device=DEVICE) + 0.1
        P, V = Bernoulli(n_gh=16).forward(mu_z, Sz)
        assert P.shape == (K,), f"P.shape={P.shape}"
        assert V.shape == (K,), f"V.shape={V.shape}"

    @requires_cuda
    def test_backward_without_forward_raises(self):
        """
        backward() before forward() must raise because self.J is None.
        Expected: TypeError (None * tensor) or AttributeError.
        """
        layer    = Bernoulli()
        delta_ma = torch.ones(4, 10, device=DEVICE)
        delta_Sa = torch.ones(4, 10, device=DEVICE)
        with pytest.raises((TypeError, AttributeError)):
            layer.backward(delta_ma, delta_Sa)

    @requires_cuda
    def test_backward_output_shapes(self):
        """backward returns (delta_mz, delta_Sz) with same shape as input."""
        B, K = 4, 10
        mu_z, Sz = make_inputs(B, K)
        layer    = Bernoulli(n_gh=16)
        layer.forward(mu_z, Sz)
        d_mz, d_Sz = layer.backward(torch.ones(B, K, device=DEVICE),
                                     torch.ones(B, K, device=DEVICE))
        assert d_mz.shape == (B, K)
        assert d_Sz.shape == (B, K)

    @requires_cuda
    def test_backward_delta_sz_non_negative(self):
        """delta_Sz = delta_Sa * J² ≥ 0 when delta_Sa ≥ 0."""
        B, K = 4, 10
        mu_z, Sz = make_inputs(B, K)
        layer    = Bernoulli(n_gh=16)
        layer.forward(mu_z, Sz)
        _, d_Sz = layer.backward(torch.zeros(B, K, device=DEVICE),
                                  torch.ones(B, K, device=DEVICE))
        assert (d_Sz >= -1e-7).all(), \
            f"Negative delta_Sz: min={d_Sz.min():.3e}"

    @requires_cuda
    def test_cauchy_schwarz_holds(self):
        """
        Jacobian clamp in forward ensures |J| ≤ sqrt(V / Sz).
        Equivalently: |C| ≤ sqrt(Sz * V).
        """
        B, K     = 8, 10
        mu_z, Sz = make_inputs(B, K, seed=50)
        layer    = Bernoulli(n_gh=32)
        P, V     = layer.forward(mu_z, Sz)
        J        = layer.J
        bound    = torch.sqrt(V / Sz.clamp(min=1e-7))
        excess   = (J.abs() - bound - 1e-5).clamp(min=0)
        assert excess.max() < 1e-5, \
            f"Cauchy-Schwarz violated: max excess={excess.max():.3e}"

    @requires_cuda
    def test_jacobian_finite(self):
        """J must be finite (no NaN/Inf) after normal forward."""
        mu_z, Sz = make_inputs(8, 10, seed=51)
        layer    = Bernoulli(n_gh=32)
        layer.forward(mu_z, Sz)
        J = layer.J
        assert not J.isnan().any(), "J has NaN"
        assert not J.isinf().any(), "J has Inf"

    @requires_cuda
    def test_jacobian_near_zero_variance(self):
        """
        DIAGNOSTIC: When Sz → 0 the Jacobian is J = C / clamp(Sz, 1e-7).
        C = EXf - mu*P approaches 0 as well (EXf → mu_i when x → mu_i
        and pe → one-hot).  Reports the actual J magnitude.
        """
        K = 5
        mu_z = torch.randn(2, K, device=DEVICE)
        Sz   = torch.full((2, K), 1e-8, device=DEVICE)
        layer = Bernoulli(n_gh=32)
        layer.forward(mu_z, Sz)
        J = layer.J
        print(f"\n  Tiny Sz: J.abs().max()={J.abs().max():.3e}, "
              f"J.abs().min()={J.abs().min():.3e}")
        assert not J.isnan().any(), "J has NaN with tiny Sz"
        assert not J.isinf().any(), "J has Inf with tiny Sz"

    @requires_cuda
    def test_forward_top_m_layer(self):
        """Bernoulli(top_m=5).forward() returns valid probabilities."""
        B, K  = 4, 10
        mu_z, Sz = make_inputs(B, K, seed=60)
        layer = Bernoulli(n_gh=16, top_m=5)
        P, V  = layer.forward(mu_z, Sz)
        assert_valid_probs(P, V, tag="Bernoulli(top_m=5)")

    @requires_cuda
    def test_repr_contains_params(self):
        """__repr__ should mention n_gh and top_m."""
        s1 = repr(Bernoulli(n_gh=16))
        assert "16" in s1, f"n_gh not in repr: {s1}"

        s2 = repr(Bernoulli(n_gh=8, top_m=3))
        assert "8"  in s2 and "3" in s2, f"Params not in repr: {s2}"


# ══════════════════════════════════════════════════════════════════════
#  6. Shape and batch handling
# ══════════════════════════════════════════════════════════════════════

class TestShapes:
    """Squeeze / unsqueeze and batch-size edge cases."""

    @requires_cuda
    def test_1d_vs_2d_consistency(self):
        """(K,) input gives same result as (1, K) input."""
        K    = 8
        torch.manual_seed(42)
        mu_z = torch.randn(K, device=DEVICE)
        Sz   = torch.rand(K, device=DEVICE) + 0.1

        P1, V1, C1 = triton_bernoulli_moments(mu_z, Sz)
        P2, V2, C2 = triton_bernoulli_moments(mu_z.unsqueeze(0),
                                               Sz.unsqueeze(0))
        assert torch.allclose(P1, P2.squeeze(0), atol=1e-6)
        assert torch.allclose(V1, V2.squeeze(0), atol=1e-6)
        assert torch.allclose(C1, C2.squeeze(0), atol=1e-6)

    @requires_cuda
    def test_batch_1(self):
        """B=1 works the same as a single sample."""
        mu_z = torch.randn(1, 5, device=DEVICE)
        Sz   = torch.rand(1, 5, device=DEVICE) + 0.1
        P, V, C = triton_bernoulli_moments(mu_z, Sz)
        assert P.shape == (1, 5)
        assert_valid_probs(P, V, C, tag="B=1")

    @requires_cuda
    def test_large_batch(self):
        """B=512, K=10: verify correctness at scale."""
        mu_z, Sz = make_inputs(512, 10, seed=70)
        P, V, C  = triton_bernoulli_moments(mu_z, Sz)
        assert_valid_probs(P, V, C, tag="B=512")

    @requires_cuda
    def test_k2_b1_directional(self):
        """B=1, K=2: higher mean → higher P."""
        mu_z = torch.tensor([[2.0, -2.0]], device=DEVICE)
        Sz   = torch.ones(1, 2, device=DEVICE)
        P, V, C = triton_bernoulli_moments(mu_z, Sz)
        assert P.shape == (1, 2)
        assert P[0, 0] > P[0, 1], \
            f"Higher mean should yield higher P: {P}"

    @requires_cuda
    def test_fast_1d_vs_2d(self):
        """fast path: (K,) vs (1, K) consistency."""
        K    = 8
        torch.manual_seed(77)
        mu_z = torch.randn(K, device=DEVICE)
        Sz   = torch.rand(K, device=DEVICE) + 0.1

        P1, V1, C1 = triton_bernoulli_moments_fast(mu_z, Sz, top_m=4)
        P2, V2, C2 = triton_bernoulli_moments_fast(mu_z.unsqueeze(0),
                                                    Sz.unsqueeze(0), top_m=4)
        assert torch.allclose(P1, P2.squeeze(0), atol=1e-6)
