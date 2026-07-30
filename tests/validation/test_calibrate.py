"""Validation tests: unified parameter-free online calibration (operator T).

These exercise the Triton-backed calibration kernels, so they require a GPU
(``@pytest.mark.cuda``). They do not need cuTAGI/pytagi — they check the
operator's own mathematical invariants:

  (I)   signal = 1 per output            diag(mwᵀ Gᶜ mw) ≈ 1
  (II)  balanced per-parameter gain      K_b / K_w ≈ 1   (uniform across params)
  (III) calibrated GLOBAL output budget  Sz_final ≈ σ_v²  (⇒ output gain J = ½);
        per-layer local budgets c_global·budget_denom_ℓ vary by design
  online re-inflation lifts a collapsed variance toward its target
  the polar projection yields orthonormal columns (signal preservation)

Run with:
    pytest tests/validation/test_calibrate.py -v
"""

from __future__ import annotations

import pytest
import torch

from triton_tagi import Linear, ReLU, Sequential, calibrate
from triton_tagi.calibrate import (
    OnlineCalibration,
    _forward_sz_analytic,
    online_recalibrate,
    polar_orth_columns,
    surprise_lambda,
)

DEVICE = "cuda"
pytestmark = pytest.mark.cuda

SIGMA2 = 0.01


def _build_net() -> Sequential:
    """Small MLP with fan_in >= fan_out at every layer (exact orthonormal columns)."""
    return Sequential(
        [Linear(64, 32, device=DEVICE), ReLU(), Linear(32, 16, device=DEVICE)],
        device=DEVICE,
    )


def _dense_metas(net):
    return [m for m in (getattr(layer, "_calib", None) for layer in net.layers if hasattr(layer, "mw"))
            if m is not None and m["kind"] == "dense"]


def test_signal_is_unit_per_output():
    """(I) Each output column has unit signal variance: v · Σ_i mw[i,o]² ≈ 1."""
    net = _build_net()
    calibrate(net, sigma2_obs=SIGMA2, metric="analytic")
    for layer in net.layers:
        meta = getattr(layer, "_calib", None)
        if meta is None or meta["kind"] != "dense":
            continue
        v = meta["v_analytic"]
        signal = v * (layer.mw**2).sum(dim=0)          # per output column
        torch.testing.assert_close(
            signal, torch.ones_like(signal), atol=1e-4, rtol=0
        )


def test_budget_equals_sigma_v2():
    """(III, global) Output budget Sz_final ≈ σ_v²  ⇒  output Kalman gain J = ½.

    Condition (III) is a GLOBAL output condition: the per-layer LOCAL budgets
    ``c_global·budget_denom_ℓ`` differ across layers, but their forward-amplified
    sum at the network output equals σ_v² (the only quantity that feeds J).
    """
    net = _build_net()
    calibrate(net, sigma2_obs=SIGMA2, metric="analytic")

    c_global = SIGMA2 / net._calib_A
    for m in _dense_metas(net):                          # every layer shares one gain
        assert m["c"] == pytest.approx(c_global, rel=1e-9)

    sz_final = _forward_sz_analytic(net, c=c_global)     # = c_global · A(net)
    assert abs(sz_final - SIGMA2) < 1e-6
    assert abs(sz_final / (sz_final + SIGMA2) - 0.5) < 1e-6


def test_gain_balance_is_unity():
    """(II) Per-parameter Kalman gains are uniform and K_b ≈ K_w (= c)."""
    net = _build_net()
    calibrate(net, sigma2_obs=SIGMA2, metric="analytic")
    for layer in net.layers:
        meta = getattr(layer, "_calib", None)
        if meta is None or meta["kind"] != "dense":
            continue
        g = meta["g_diag"]
        kw = g.sqrt() * layer.Sw[:, 0]                   # √E[a²] · Sw
        kb = float(layer.Sb.flatten()[0].item())
        assert (kw.max() / kw.min()).item() < 1.0 + 1e-4   # gains uniform
        assert abs(kb - float(kw.mean().item())) < 1e-6    # K_b ≈ K_w


def test_online_reinflation_lifts_collapsed_variance():
    """Online T re-inflates a collapsed posterior toward its calibrated target."""
    net = _build_net()
    calibrate(net, sigma2_obs=SIGMA2, metric="analytic")
    # Online must use the SAME global gain c = σ_v²/A(net) that init used.
    cfg = OnlineCalibration(
        mode="const", lam=0.5, project=False, sigma_v2=SIGMA2, calib_A=net._calib_A
    )

    dense = next(layer for layer in net.layers
                 if getattr(layer, "_calib", None) and layer._calib["kind"] == "dense")
    target = dense._calib["c"] / dense._calib["g_diag"][0].sqrt()
    dense.Sw.fill_(1e-8)

    online_recalibrate(net, lam=0.5, cfg=cfg)
    got = float(dense.Sw[0, 0].item())
    # one λ=0.5 step moves halfway from 1e-8 toward the target
    assert got > 1e-8
    assert abs(got - 0.5 * float(target.item())) < 1e-5


def test_polar_orth_columns_are_orthonormal():
    """The exact polar projection returns column-orthonormal matrices (signal=1)."""
    B = torch.randn(40, 16, device=DEVICE) * 3.0 + 1.0
    O = polar_orth_columns(B)
    gram = O.T @ O
    torch.testing.assert_close(gram, torch.eye(16, device=DEVICE), atol=1e-5, rtol=0)


def test_surprise_lambda_monotone():
    """χ²≈1 → λ≈0 (stable); larger surprise → larger λ, capped at λ_max."""
    assert surprise_lambda(1.0, lam_max=0.1, tau=1.0) == pytest.approx(0.0, abs=1e-9)
    mid = surprise_lambda(2.0, lam_max=0.1, tau=1.0)
    hi = surprise_lambda(10.0, lam_max=0.1, tau=1.0)
    assert 0.0 < mid < hi <= 0.1


# ----------------------------------------------------------------------
#  Data-driven (empirical) metric
# ----------------------------------------------------------------------


def test_data_metric_signal_unit_on_real_data():
    """(I) metric='data': realized Var[z] on the calibration batch is ≈ 1 per output.

    Uses a *correlated* input so the empirical whitener differs from the analytic
    ``Gᶜ = v·I`` — the whole point of the data metric.
    """
    torch.manual_seed(0)
    net = _build_net()
    X = torch.randn(1024, 64, device=DEVICE)
    X = X + 0.6 * X.roll(1, dims=1)                     # induce input correlation
    calibrate(net, sigma2_obs=SIGMA2, metric="data", data_batch=X)

    A = X
    for layer in net.layers:
        meta = getattr(layer, "_calib", None)
        if meta is not None and meta["kind"] == "dense":
            Ac = A - A.mean(dim=0)
            Z = Ac @ layer.mw                          # centred pre-activation signal
            signal = (Z * Z).mean(dim=0)               # Var[z] per output
            torch.testing.assert_close(signal, torch.ones_like(signal), atol=2e-2, rtol=0)
            A = A @ layer.mw + layer.mb                 # propagate calibrated mean
        elif isinstance(layer, ReLU):
            A = torch.relu(A)


def test_data_metric_budget_equals_sigma_v2():
    """(III, global) metric='data': output budget Sz_final ≈ σ_v² (J = ½)."""
    torch.manual_seed(0)
    net = _build_net()
    X = torch.randn(1024, 64, device=DEVICE)
    calibrate(net, sigma2_obs=SIGMA2, metric="data", data_batch=X)

    c_global = SIGMA2 / net._calib_A
    for m in _dense_metas(net):
        assert m["c"] == pytest.approx(c_global, rel=1e-9)

    sz_final = _forward_sz_analytic(net, c=c_global)
    assert abs(sz_final - SIGMA2) < 1e-6


def test_data_metric_requires_batch():
    """metric='data' without a data_batch is a clear ValueError, not a silent no-op."""
    net = _build_net()
    with pytest.raises(ValueError, match="requires"):
        calibrate(net, sigma2_obs=SIGMA2, metric="data")


def test_online_config_validates_enums():
    """Misconfigured string enums fail fast instead of silently no-op'ing."""
    with pytest.raises(ValueError, match="mode"):
        OnlineCalibration(mode="surprize")             # typo
    with pytest.raises(ValueError, match="sigma_v_mode"):
        OnlineCalibration(sigma_v_mode="homo")
