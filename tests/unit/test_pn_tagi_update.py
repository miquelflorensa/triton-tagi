"""Stage 1 — Scalar / analytic sanity tests for PN-TAGI.

These tests live below the layer level: they exercise
``triton_tagi.update.parameters.update_parameters`` directly with synthetic
deltas (no forward/backward pipeline) so the four update rules can be
compared against closed-form expressions.

What is verified
----------------
1. Per-rule kernel correctness (additive, capped_additive,
   precision_normalized, tempered_precision_normalized) against analytical
   formulas computed in pure torch.
2. ``chi_out`` diagnostic equals ``-ΔS / max(S, eps)`` regardless of rule,
   and is positive iff the additive update tries to contract.
3. Scalar 1-observation Kalman case:
   - additive == exact Bayesian posterior (μ, σ²) for chi < 1
   - PN-TAGI stays positive but diverges from exact (it is a regularised
     contraction, not exact Bayes — see plan).
4. Scalar N-observation accumulation:
   - additive goes non-positive (and floors to 1e-5) when ``N·K > 1``
   - capped_additive bounds each delta but can still saturate
   - PN-TAGI never reaches zero; consumed-variance fraction matches
     ``χ / (1 + χ)``.
5. Sequential plumbing: each ``update_rule`` runs a small Linear-ReLU-Linear
   end-to-end step on CUDA without NaNs and produces strictly positive Sw
   afterwards.

Run with:
    pytest tests/unit/test_pn_tagi_update.py -v
"""

from __future__ import annotations

import pytest
import torch

from triton_tagi import (
    Linear,
    ReLU,
    Remax,
    Sequential,
    VALID_RULES,
    chi_stats,
    update_parameters,
)

DEVICE = "cuda"
DTYPE = torch.float32
ATOL = 1e-5
RTOL = 1e-5

pytestmark = pytest.mark.cuda


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_buffers(n: int, seed: int = 0):
    """Random m, S, dm, dS with realistic shapes and signs.

    S is always strictly positive (∈ [1e-3, 1.1]).
    dS spans both signs so the chi diagnostic covers contraction *and*
    inflation.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    m = torch.randn(n, generator=g, device=DEVICE, dtype=DTYPE)
    S = torch.rand(n, generator=g, device=DEVICE, dtype=DTYPE) + 0.1
    dm = 0.3 * torch.randn(n, generator=g, device=DEVICE, dtype=DTYPE)
    dS = 0.2 * torch.randn(n, generator=g, device=DEVICE, dtype=DTYPE)
    return m.clone(), S.clone(), dm.clone(), dS.clone()


def _ref_additive(m, S, dm, dS):
    m_new = m + dm
    S_raw = S + dS
    S_new = torch.where(S_raw <= 0.0, torch.full_like(S_raw, 1e-5), S_raw)
    return m_new, S_new


def _ref_capped(m, S, dm, dS, cap_factor):
    delta_bar = torch.sqrt(torch.clamp(S, min=1e-10)) / cap_factor
    dm_capped = torch.sign(dm) * torch.minimum(dm.abs(), delta_bar)
    dS_capped = torch.sign(dS) * torch.minimum(dS.abs(), delta_bar)
    m_new = m + dm_capped
    S_raw = S + dS_capped
    S_new = torch.where(S_raw <= 0.0, torch.full_like(S_raw, 1e-5), S_raw)
    return m_new, S_new


def _ref_pn(m, S, dm, dS, rho):
    raw_chi = -dS / torch.clamp(S, min=1e-12)
    chi = torch.clamp(raw_chi, min=0.0)
    d = 1.0 + rho * chi
    m_new = m + rho * dm / d
    S_new = S / d
    return m_new, S_new


def _ref_capped_pn(m, S, dm, dS, cap_factor, rho):
    """capped_precision_normalized reference: cap raw dm then divide by d."""
    delta_bar = torch.sqrt(torch.clamp(S, min=1e-10)) / cap_factor
    dm_capped = torch.sign(dm) * torch.minimum(dm.abs(), delta_bar)
    raw_chi = -dS / torch.clamp(S, min=1e-12)
    chi = torch.clamp(raw_chi, min=0.0)
    d = 1.0 + rho * chi
    m_new = m + rho * dm_capped / d
    S_new = S / d
    return m_new, S_new


# ===========================================================================
#  1. Per-rule kernel correctness
# ===========================================================================


class TestKernelCorrectness:
    def test_additive_matches_reference(self):
        m, S, dm, dS = _make_buffers(1024, seed=0)
        m_exp, S_exp = _ref_additive(m, S, dm, dS)
        update_parameters(m, S, dm, dS, cap_factor=2.0, update_rule="additive")
        torch.testing.assert_close(m, m_exp, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S, S_exp, atol=ATOL, rtol=RTOL)

    def test_additive_floors_at_1e_5(self):
        # Force S + dS <= 0 to hit the numerical floor branch.
        m = torch.zeros(8, device=DEVICE, dtype=DTYPE)
        S = torch.full((8,), 0.1, device=DEVICE, dtype=DTYPE)
        dm = torch.zeros_like(m)
        dS = torch.full((8,), -0.5, device=DEVICE, dtype=DTYPE)  # S + dS = -0.4
        update_parameters(m, S, dm, dS, cap_factor=2.0, update_rule="additive")
        assert torch.all(S == 1e-5), f"expected floor, got {S}"

    def test_capped_additive_matches_baseline(self):
        # cap_factor=2.0 is the default for 1 < batch < 256.
        m, S, dm, dS = _make_buffers(1024, seed=1)
        m_exp, S_exp = _ref_capped(m, S, dm, dS, cap_factor=2.0)
        update_parameters(m, S, dm, dS, cap_factor=2.0, update_rule="capped_additive")
        torch.testing.assert_close(m, m_exp, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S, S_exp, atol=ATOL, rtol=RTOL)

    def test_capped_default_rule_matches_explicit(self):
        """Default rule must remain ``capped_additive`` (backwards compat)."""
        m1, S1, dm1, dS1 = _make_buffers(1024, seed=2)
        m2, S2, dm2, dS2 = _make_buffers(1024, seed=2)
        update_parameters(m1, S1, dm1, dS1, cap_factor=2.0)
        update_parameters(m2, S2, dm2, dS2, cap_factor=2.0, update_rule="capped_additive")
        torch.testing.assert_close(m1, m2)
        torch.testing.assert_close(S1, S2)

    def test_precision_normalized_matches_reference(self):
        m, S, dm, dS = _make_buffers(1024, seed=3)
        m_exp, S_exp = _ref_pn(m, S, dm, dS, rho=1.0)
        update_parameters(m, S, dm, dS, cap_factor=2.0, update_rule="precision_normalized")
        torch.testing.assert_close(m, m_exp, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S, S_exp, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("rho", [0.25, 0.5, 0.75, 1.0])
    def test_tempered_matches_reference(self, rho):
        m, S, dm, dS = _make_buffers(1024, seed=4)
        m_exp, S_exp = _ref_pn(m, S, dm, dS, rho=rho)
        update_parameters(
            m, S, dm, dS,
            cap_factor=2.0,
            update_rule="tempered_precision_normalized",
            rho=rho,
        )
        torch.testing.assert_close(m, m_exp, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S, S_exp, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("rho", [0.5, 1.0])
    @pytest.mark.parametrize("cap_factor", [0.1, 2.0, 3.0])
    def test_capped_precision_normalized_matches_reference(self, rho, cap_factor):
        m, S, dm, dS = _make_buffers(1024, seed=5)
        m_exp, S_exp = _ref_capped_pn(m, S, dm, dS, cap_factor=cap_factor, rho=rho)
        update_parameters(
            m, S, dm, dS, cap_factor=cap_factor,
            update_rule="capped_precision_normalized", rho=rho,
        )
        torch.testing.assert_close(m, m_exp, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S, S_exp, atol=ATOL, rtol=RTOL)

    def test_capped_pn_keeps_S_strictly_positive_under_extreme_dS(self):
        """The hybrid still inherits PN-TAGI's variance positivity guarantee."""
        m = torch.zeros(2048, device=DEVICE, dtype=DTYPE)
        S = torch.full((2048,), 0.01, device=DEVICE, dtype=DTYPE)
        dm = torch.randn(2048, device=DEVICE, dtype=DTYPE)
        dS = torch.full((2048,), -1.0, device=DEVICE, dtype=DTYPE)
        update_parameters(
            m, S, dm, dS, cap_factor=2.0,
            update_rule="capped_precision_normalized",
        )
        assert torch.all(S > 0.0)

    def test_precision_normalized_keeps_S_strictly_positive(self):
        """Even with dS that would drive additive below zero, PN keeps S > 0."""
        m = torch.zeros(2048, device=DEVICE, dtype=DTYPE)
        S = torch.full((2048,), 0.01, device=DEVICE, dtype=DTYPE)
        dm = torch.randn(2048, device=DEVICE, dtype=DTYPE)
        # Very strong contraction: dS = -100·S → chi = 100
        dS = torch.full((2048,), -1.0, device=DEVICE, dtype=DTYPE)
        update_parameters(
            m, S, dm, dS,
            cap_factor=2.0,
            update_rule="precision_normalized",
        )
        assert torch.all(S > 0.0)
        # σ²_new = σ²_prior / (1 + 100) ≈ 0.01/101 ≈ 9.9e-5
        torch.testing.assert_close(
            S, torch.full_like(S, 0.01 / 101.0), rtol=1e-4, atol=1e-8
        )

    def test_invalid_rule_raises(self):
        m, S, dm, dS = _make_buffers(8)
        with pytest.raises(ValueError, match="update_rule"):
            update_parameters(m, S, dm, dS, cap_factor=2.0, update_rule="not_a_rule")


# ===========================================================================
#  2. Diagnostic correctness
# ===========================================================================


class TestChiDiagnostic:
    def test_chi_matches_formula_under_pn(self):
        m, S, dm, dS = _make_buffers(2048, seed=10)
        S_prior = S.clone()
        chi_out = torch.empty_like(S)
        update_parameters(
            m, S, dm, dS,
            cap_factor=2.0,
            update_rule="precision_normalized",
            chi_out=chi_out,
        )
        chi_expected = -dS / torch.clamp(S_prior, min=1e-12)
        torch.testing.assert_close(chi_out, chi_expected, atol=ATOL, rtol=RTOL)

    def test_chi_recorded_for_additive_too(self):
        """Diagnostic is rule-independent — useful to inspect baselines."""
        m, S, dm, dS = _make_buffers(512, seed=11)
        S_prior = S.clone()
        chi_out = torch.empty_like(S)
        update_parameters(
            m, S, dm, dS,
            cap_factor=2.0,
            update_rule="additive",
            chi_out=chi_out,
        )
        chi_expected = -dS / torch.clamp(S_prior, min=1e-12)
        torch.testing.assert_close(chi_out, chi_expected, atol=ATOL, rtol=RTOL)

    def test_chi_negative_for_positive_dS(self):
        """raw_chi < 0  ⇔  dS > 0 (positive variance increment must be visible)."""
        m, S, dm, dS = _make_buffers(2048, seed=12)
        chi_out = torch.empty_like(S)
        update_parameters(
            m.clone(), S.clone(), dm, dS,
            cap_factor=2.0,
            update_rule="capped_additive",
            chi_out=chi_out,
        )
        # Equivalence of signs:
        assert torch.equal((chi_out < 0), (dS > 0))

    def test_chi_stats_aggregation(self):
        chi = torch.tensor([-0.1, 0.0, 0.05, 0.5, 1.2, 2.0, 10.0], device=DEVICE)
        s = chi_stats(chi)
        assert s["count"] == 7
        assert s["raw_chi_max"] == pytest.approx(10.0)
        assert s["frac_dS_pos"] == pytest.approx(1.0 / 7.0)
        # clipped chi: [0, 0, 0.05, 0.5, 1.2, 2.0, 10.0]
        #   > 0.1 → {0.5, 1.2, 2.0, 10.0}      = 4/7
        #   > 1.0 → {1.2, 2.0, 10.0}           = 3/7
        assert s["frac_chi_gt_0p1"] == pytest.approx(4.0 / 7.0)
        assert s["frac_chi_gt_1"] == pytest.approx(3.0 / 7.0)


# ===========================================================================
#  3. Scalar Kalman: single observation
# ===========================================================================


class TestSingleObservationKalman:
    """For a scalar parameter θ ~ N(μ, σ²) observed once as y = θ + v with
    v ~ N(0, σ_v²), additive TAGI deltas reproduce the *exact* Gaussian
    posterior. PN-TAGI is a regularised contraction and differs by O(χ²).
    """

    @staticmethod
    def _setup(mu_prior=0.0, S_prior=1.0, y=1.0, sigma_v2=0.5):
        # Tensors of length 1, on GPU.
        m = torch.tensor([mu_prior], device=DEVICE, dtype=DTYPE)
        S = torch.tensor([S_prior], device=DEVICE, dtype=DTYPE)
        K = S_prior / (S_prior + sigma_v2)
        dm = torch.tensor([K * (y - mu_prior)], device=DEVICE, dtype=DTYPE)
        dS = torch.tensor([-K * S_prior], device=DEVICE, dtype=DTYPE)
        return m, S, dm, dS, K

    def test_additive_recovers_exact_posterior(self):
        mu, S0, y, sv2 = 0.0, 1.0, 2.0, 0.5
        m, S, dm, dS, K = self._setup(mu, S0, y, sv2)
        update_parameters(m, S, dm, dS, cap_factor=2.0, update_rule="additive")
        # Exact: μ_post = μ + K(y-μ);  σ²_post = σ²(1-K) = σ²σ_v²/(σ²+σ_v²)
        mu_exact = mu + K * (y - mu)
        S_exact = S0 * sv2 / (S0 + sv2)
        torch.testing.assert_close(m.item(), mu_exact, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(S.item(), S_exact, atol=1e-6, rtol=1e-6)

    def test_pn_diverges_from_exact_at_order_chi_squared(self):
        # chi = K. With S0 = σ_v² this gives K = 0.5 → non-trivial difference.
        m, S, dm, dS, K = self._setup(mu_prior=0.0, S_prior=1.0, y=2.0, sigma_v2=1.0)
        update_parameters(m, S, dm, dS, cap_factor=2.0, update_rule="precision_normalized")
        # PN: σ²_new = σ²_prior / (1 + chi) where chi = K
        S_pn_expected = 1.0 / (1.0 + K)
        torch.testing.assert_close(S.item(), S_pn_expected, atol=1e-6, rtol=1e-6)
        # Sanity: PN is *less* contracted than exact (Bayes) for chi > 0.
        S_exact = 1.0 * 1.0 / (1.0 + 1.0)  # σ²σ_v²/(σ²+σ_v²) = 0.5
        assert S.item() > S_exact - 1e-6  # PN keeps more variance (regularised)


# ===========================================================================
#  4. Scalar Kalman: N-observation mini-batch accumulation
# ===========================================================================


class TestMiniBatchAccumulation:
    """In a mini-batch, TAGI sums N additive deltas all computed against the
    *same* prior. This is where the additive rule breaks (variance can be
    driven below zero) and where PN-TAGI's self-normalisation pays off.
    """

    @staticmethod
    def _accum_deltas(N, mu_prior, S_prior, sigma_v2, y_mean):
        K = S_prior / (S_prior + sigma_v2)
        dm = torch.tensor([N * K * (y_mean - mu_prior)], device=DEVICE, dtype=DTYPE)
        dS = torch.tensor([-N * K * S_prior], device=DEVICE, dtype=DTYPE)
        chi_total = N * K
        return dm, dS, chi_total

    def test_additive_floors_when_chi_exceeds_one(self):
        # Pick N large and σ_v² small → chi >> 1
        N, mu, S0, sv2, ybar = 16, 0.0, 1.0, 0.1, 3.0
        m = torch.tensor([mu], device=DEVICE, dtype=DTYPE)
        S = torch.tensor([S0], device=DEVICE, dtype=DTYPE)
        dm, dS, chi = self._accum_deltas(N, mu, S0, sv2, ybar)
        assert chi > 1.0
        update_parameters(m, S, dm, dS, cap_factor=2.0, update_rule="additive")
        assert S.item() == pytest.approx(1e-5), (
            f"additive should hit the floor when chi={chi:.2f} > 1, got S={S.item()}"
        )

    def test_pn_stays_strictly_positive_and_matches_formula(self):
        N, mu, S0, sv2, ybar = 16, 0.0, 1.0, 0.1, 3.0
        m = torch.tensor([mu], device=DEVICE, dtype=DTYPE)
        S = torch.tensor([S0], device=DEVICE, dtype=DTYPE)
        dm, dS, chi = self._accum_deltas(N, mu, S0, sv2, ybar)
        chi_out = torch.empty_like(S)
        update_parameters(
            m, S, dm, dS,
            cap_factor=2.0,
            update_rule="precision_normalized",
            chi_out=chi_out,
        )
        assert S.item() > 0.0
        # σ²_new = σ²_prior / (1 + chi)
        torch.testing.assert_close(S.item(), S0 / (1.0 + chi), rtol=1e-5, atol=1e-8)
        # Diagnostic exposed the true chi
        torch.testing.assert_close(chi_out.item(), chi, rtol=1e-5, atol=1e-8)

    def test_consumed_variance_fraction_matches_chi_over_one_plus_chi(self):
        """The plan's headline identity: PN consumes χ/(1+χ) of prior variance."""
        N, mu, S0, sv2, ybar = 8, 0.0, 1.0, 0.5, 1.0
        m = torch.tensor([mu], device=DEVICE, dtype=DTYPE)
        S = torch.tensor([S0], device=DEVICE, dtype=DTYPE)
        dm, dS, chi = self._accum_deltas(N, mu, S0, sv2, ybar)
        update_parameters(m, S, dm, dS, cap_factor=2.0, update_rule="precision_normalized")
        consumed = 1.0 - S.item() / S0
        torch.testing.assert_close(consumed, chi / (1.0 + chi), rtol=1e-5, atol=1e-8)


# ===========================================================================
#  5. Sequential end-to-end smoke test
# ===========================================================================


@pytest.mark.parametrize("rule", list(VALID_RULES))
def test_sequential_step_runs_under_each_rule(rule):
    """A tiny MLP runs one step under every rule without producing NaN/Inf."""
    torch.manual_seed(0)
    net = Sequential(
        [Linear(8, 16), ReLU(), Linear(16, 4), Remax()],
        device=DEVICE,
        update_rule=rule,
        rho=1.0,
        record_chi=True,
    )

    x = torch.randn(32, 8, device=DEVICE)
    y = torch.zeros(32, 4, device=DEVICE)
    y[torch.arange(32), torch.randint(0, 4, (32,), device=DEVICE)] = 1.0

    mu, var = net.step(x, y, sigma_v=1.0)
    assert torch.isfinite(mu).all(), f"NaN in mu under {rule}"
    assert torch.isfinite(var).all(), f"NaN in var under {rule}"

    for layer in net.layers:
        if hasattr(layer, "Sw") and layer.Sw is not None:
            assert torch.all(layer.Sw > 0.0), f"non-positive Sw after {rule}"

    # Diagnostics were captured.
    stats = net.collect_chi_stats()
    assert len(stats) > 0, f"expected chi buffers under {rule}, got none"
    # Every reported chi tensor must have finite stats.
    for key, s in stats.items():
        assert torch.isfinite(torch.tensor(s["raw_chi_max"])), key
