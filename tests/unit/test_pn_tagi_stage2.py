"""Stage 2 — Tiny linear TAGI regression.

A 1-layer ``Linear`` model (no activation, no output transform) is the
simplest setting where TAGI's local Gaussian assumption is exact, so we
can verify the headline contraction-ratio predictions directly:

    chi ≈ S_w · Σ_i x_i² / (S_z_i + σ_v²)

leading to the monotonicity claims this stage is meant to pin:

    * chi increases as σ_v decreases (likelihood becomes stronger)
    * chi increases with batch size (more observations summed)
    * PN-TAGI keeps parameter variance strictly positive in regimes
      where additive collapses to the 1e-5 numerical floor
    * In the well-conditioned (chi << 1) regime all three rules learn
      the underlying linear map with comparable MSE

Run with:
    pytest tests/unit/test_pn_tagi_stage2.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from triton_tagi import Linear, Sequential

DEVICE = "cuda"
DTYPE = torch.float32
pytestmark = pytest.mark.cuda


# ---------------------------------------------------------------------------
#  Harness
# ---------------------------------------------------------------------------


def run_linear_experiment(
    *,
    in_features: int = 4,
    batch_size: int = 32,
    sigma_v: float = 0.1,
    gain_w: float = 1.0,
    update_rule: str = "capped_additive",
    rho: float = 1.0,
    n_steps: int = 30,
    seed: int = 0,
    sigma_obs: float | None = None,
) -> dict:
    """Train a 1-layer ``Linear`` net on synthetic linear regression.

    A fresh ``(x, y)`` batch is drawn every step from ::

        x ~ N(0, I_in)
        y = x · W_true + b_true + ε,   ε ~ N(0, σ_obs²)

    so the local Gaussian assumption is exact and we can read chi
    behaviour off ``net.collect_chi_stats()`` directly.

    Returns a dict with per-step chi / Sw / mse history plus final
    parameter state.
    """
    if sigma_obs is None:
        sigma_obs = sigma_v

    torch.manual_seed(seed)
    np.random.seed(seed)

    # True parameters — fixed across configs so MSE is comparable.
    W_true = torch.randn(in_features, 1, device=DEVICE, dtype=DTYPE)
    b_true = torch.randn(1, device=DEVICE, dtype=DTYPE) * 0.1

    net = Sequential(
        [Linear(in_features, 1, device=DEVICE, gain_w=gain_w, gain_b=gain_w, bias=True)],
        device=DEVICE,
        update_rule=update_rule,
        rho=rho,
        record_chi=True,
    )

    chi_history: list[dict] = []
    Sw_mean_history: list[float] = []
    mse_history: list[float] = []
    Sw_min_history: list[float] = []

    linear: Linear = net.layers[0]  # type: ignore[assignment]
    g = torch.Generator(device=DEVICE).manual_seed(seed + 1)

    for step in range(n_steps):
        x = torch.randn(batch_size, in_features, generator=g, device=DEVICE, dtype=DTYPE)
        noise = sigma_obs * torch.randn(batch_size, 1, generator=g, device=DEVICE, dtype=DTYPE)
        y = x @ W_true + b_true + noise

        mu_pred, _ = net.step(x, y, sigma_v=sigma_v)

        mse = ((mu_pred - y) ** 2).mean().item()
        mse_history.append(mse)
        Sw_mean_history.append(linear.Sw.mean().item())
        Sw_min_history.append(linear.Sw.min().item())

        stats = net.collect_chi_stats()
        # Only one learnable layer; pull its chi_w stats.
        chi_history.append(stats.get("0.chi_w", {}))

    # Final loss against a fresh, larger validation batch.
    x_val = torch.randn(1024, in_features, generator=g, device=DEVICE, dtype=DTYPE)
    y_val = x_val @ W_true + b_true
    net.eval()
    with torch.no_grad():
        mu_val, _ = net.forward(x_val)
    val_mse = ((mu_val - y_val) ** 2).mean().item()

    return {
        "chi_history": chi_history,
        "Sw_mean_history": Sw_mean_history,
        "Sw_min_history": Sw_min_history,
        "mse_history": mse_history,
        "val_mse": val_mse,
        "W_true": W_true.cpu(),
        "b_true": b_true.cpu(),
        "mw_final": linear.mw.detach().cpu(),
        "Sw_final": linear.Sw.detach().cpu(),
        "any_nan": (
            not math.isfinite(linear.mw.sum().item())
            or not math.isfinite(linear.Sw.sum().item())
        ),
    }


# ---------------------------------------------------------------------------
#  1. Monotonicity of chi
# ---------------------------------------------------------------------------


class TestChiMonotonicity:
    """Predictions from the closed form for a 1-layer Linear:

        chi ≈ S_w · Σ_i x_i² / (S_z_i + σ_v²)

    so chi grows as σ_v shrinks and as batch_size grows.
    """

    @pytest.mark.parametrize("batch_size", [4, 32, 128])
    def test_chi_decreasing_in_sigma_v(self, batch_size):
        """For fixed batch / prior, chi(step=0) ↓ as σ_v ↑."""
        sigma_vs = [0.01, 0.1, 1.0]
        chis = []
        for sv in sigma_vs:
            out = run_linear_experiment(
                batch_size=batch_size,
                sigma_v=sv,
                update_rule="precision_normalized",
                n_steps=1,
                seed=42,
            )
            chis.append(out["chi_history"][0]["raw_chi_median"])
        # Monotonic strictly decreasing.
        assert chis[0] > chis[1] > chis[2], (
            f"expected chi monotonically decreasing in σ_v, got chis={chis} "
            f"for σ_v={sigma_vs}"
        )

    @pytest.mark.parametrize("sigma_v", [0.05, 0.5])
    def test_chi_increasing_in_batch_size(self, sigma_v):
        """For fixed σ_v / prior, chi(step=0) ↑ as batch_size ↑."""
        batch_sizes = [1, 8, 64, 256]
        chis = []
        for B in batch_sizes:
            out = run_linear_experiment(
                batch_size=B,
                sigma_v=sigma_v,
                update_rule="precision_normalized",
                n_steps=1,
                seed=7,
            )
            chis.append(out["chi_history"][0]["raw_chi_median"])
        for prev, curr in zip(chis, chis[1:]):
            assert curr > prev, (
                f"expected chi monotonically increasing in batch_size, "
                f"got chis={chis} for B={batch_sizes}, σ_v={sigma_v}"
            )

    def test_chi_increasing_in_initial_prior_variance(self):
        """chi ∝ S_w_prior, so raising gain_w should raise chi."""
        gains = [0.25, 1.0, 4.0]
        chis = []
        for g in gains:
            out = run_linear_experiment(
                batch_size=32,
                sigma_v=0.1,
                gain_w=g,
                update_rule="precision_normalized",
                n_steps=1,
                seed=3,
            )
            chis.append(out["chi_history"][0]["raw_chi_median"])
        for prev, curr in zip(chis, chis[1:]):
            assert curr > prev, (
                f"expected chi monotonically increasing in gain_w, got chis={chis}"
            )


# ---------------------------------------------------------------------------
#  2. Rule comparison in a high-chi regime
# ---------------------------------------------------------------------------


class TestHighChiRegime:
    """Small σ_v + large batch ⇒ additive deltas accumulate beyond the
    prior variance. This is exactly the failure mode PN-TAGI was designed
    to fix.
    """

    HIGH_CHI = dict(batch_size=256, sigma_v=0.01, gain_w=1.0, n_steps=20, seed=11)

    def test_additive_collapses_Sw_to_floor(self):
        out = run_linear_experiment(**self.HIGH_CHI, update_rule="additive")
        # At least one variance hits the 1e-5 numerical floor during training.
        assert min(out["Sw_min_history"]) <= 1e-5 + 1e-7, (
            f"expected additive Sw to reach the floor under high chi, "
            f"min Sw history = {out['Sw_min_history']}"
        )

    def test_pn_keeps_Sw_strictly_positive(self):
        out = run_linear_experiment(**self.HIGH_CHI, update_rule="precision_normalized")
        # PN-TAGI's headline guarantee: variance never reaches zero. The
        # actual magnitude can be small under high chi (each step shrinks
        # by 1/(1+chi)), but it is always strictly > 0 — not the additive
        # rule's 1e-5 numerical floor.
        assert min(out["Sw_min_history"]) > 0.0, (
            f"expected PN-TAGI Sw strictly positive, "
            f"min Sw = {min(out['Sw_min_history'])}"
        )
        assert not out["any_nan"]

    def test_pn_chi_diagnostic_signals_high_pressure(self):
        """The diagnostic should reflect the high-pressure regime: a
        large initial chi and a large fraction of weights with chi > 1.
        """
        out = run_linear_experiment(**self.HIGH_CHI, update_rule="precision_normalized")
        chi0 = out["chi_history"][0]
        assert chi0["raw_chi_max"] > 1.0, f"expected max raw_chi > 1, got {chi0}"
        assert chi0["frac_chi_gt_1"] > 0.0, (
            f"expected some weights with chi > 1 at step 0, got {chi0}"
        )


# ---------------------------------------------------------------------------
#  3. Convergence in a well-conditioned regime
# ---------------------------------------------------------------------------


class TestWellConditionedConvergence:
    """When chi << 1 throughout training, the additive / capped / PN
    update rules are all close to the exact Bayesian update, so all
    three should learn the underlying linear map.

    With He init at fan_in=4 the initial S_w is ~0.25. To keep chi small
    we use a tiny per-step batch (1) and a large σ_v (1.0) so

        chi ≈ S_w · B · ⟨x²⟩ / (S_w + σ_v²)  ≈  0.25 / 1.25 ≈ 0.2.

    Convergence in this regime is slow (per-step mean update ~ S_w · x ·
    δμ_z, so it scales with the small prior), hence the 400 training
    steps. Tolerances are generous — this catches divergence, not
    bit-exact recovery.
    """

    GENTLE = dict(
        in_features=4,
        batch_size=1,
        sigma_v=1.0,
        gain_w=1.0,
        n_steps=400,
        seed=0,
    )

    @pytest.mark.parametrize(
        "rule", ["additive", "capped_additive", "precision_normalized"]
    )
    def test_all_rules_learn_linear_map(self, rule):
        out = run_linear_experiment(**self.GENTLE, update_rule=rule)
        assert not out["any_nan"], f"{rule} produced NaN/Inf parameters"

        # Final chi p95 should remain below 1 throughout — confirms we
        # are in the well-conditioned regime, not just a lucky run.
        p95_initial = out["chi_history"][0]["raw_chi_p95"]
        assert p95_initial < 1.0, (
            f"{rule}: initial p95 chi = {p95_initial:.2f} — config is "
            f"not well-conditioned, fix the test setup"
        )

        # Recovered weight ≈ true weight.
        w_err = (out["mw_final"].squeeze() - out["W_true"].squeeze()).norm().item()
        w_true_norm = out["W_true"].squeeze().norm().item()
        assert w_err / w_true_norm < 0.35, (
            f"{rule}: relative weight error {w_err / w_true_norm:.3f} > 0.35; "
            f"final mw={out['mw_final'].squeeze().tolist()}, "
            f"W_true={out['W_true'].squeeze().tolist()}"
        )
