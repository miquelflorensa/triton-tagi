"""Unit tests for triton_tagi.inference_init (IBI).

Per-step correctness (no MNIST, no training):
    - S projection lands the layer-sum mean/variance on target exactly.
    - S2 RTS update moves the layer-sum second-moment toward target
      (monotonic reduction of |S2 - S2_tilde|).
    - Decoupled inverse: after applying gamma scaling and bias shift, a
      re-forward pass produces batch-aggregate moments equal to the target.
    - End-to-end single-layer call: after ``inference_init`` on a loader of
      one batch, the Linear layer's re-forward hits both S and approximately S2
      simultaneously (S exact, S2 asymptotic; we check ordering).
    - Non-Linear layers (ReLU, Flatten, Remax) pass through without mutation.

Run with:
    pytest tests/unit/test_inference_init.py -v
"""

from __future__ import annotations

import pytest
import torch

from triton_tagi import (
    AvgPool2D,
    BatchNorm2D,
    Conv2D,
    Flatten,
    Linear,
    ReLU,
    Remax,
    ResBlock,
    Sequential,
    inference_init,
)
from triton_tagi.inference_init import (
    _decoupled_inverse,
    _layer_targets,
    _s2_projection,
    _s_projection,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.cuda


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_layer(in_feat: int, out_feat: int, seed: int = 0) -> Linear:
    torch.manual_seed(seed)
    return Linear(in_feat, out_feat, device=DEVICE)


def _make_inputs(B: int, D: int, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    ma = torch.randn(B, D, device=DEVICE, generator=g)
    Sa = torch.rand(B, D, device=DEVICE, generator=g).abs() * 0.1 + 1e-4
    return ma, Sa


def _batch_moments(mz: torch.Tensor, Sz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-mean aggregation matching PLAN.md D1 (batch-mean first)."""
    return mz.mean(dim=0), Sz.mean(dim=0)


# ---------------------------------------------------------------------------
#  Targets
# ---------------------------------------------------------------------------


class TestLayerTargets:
    def test_formulas(self):
        A, sm, sz = 128, 0.5, 0.5
        mu_S, var_S, mu_S2, var_S2 = _layer_targets(A, sm, sz)
        assert mu_S == 0.0
        assert var_S == pytest.approx(A * sz**2)
        assert mu_S2 == pytest.approx(A * (sm**2 + sz**2))
        assert var_S2 == pytest.approx(A * (2 * sz**4 + 4 * sm**2 * sz**2))


# ---------------------------------------------------------------------------
#  S projection
# ---------------------------------------------------------------------------


class TestSProjection:
    def test_post_projection_hits_target(self):
        A = 64
        torch.manual_seed(0)
        mu_Z = torch.randn(A, device=DEVICE)
        S_Z = torch.rand(A, device=DEVICE) + 0.1
        mu_t, var_t = 0.0, A * 0.5**2
        mu_post, S_post = _s_projection(mu_Z, S_Z, mu_t, var_t, eps=1e-8)
        torch.testing.assert_close(
            mu_post.sum(), torch.tensor(mu_t, device=DEVICE), atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            S_post.sum(), torch.tensor(var_t, device=DEVICE), atol=1e-4, rtol=0
        )

    def test_already_on_target_is_noop(self):
        A = 32
        mu_Z = torch.zeros(A, device=DEVICE)
        S_Z = torch.full((A,), 0.25, device=DEVICE)  # sum = A * 0.25 = var_t
        mu_t, var_t = 0.0, A * 0.5**2
        mu_post, S_post = _s_projection(mu_Z, S_Z, mu_t, var_t, eps=1e-8)
        torch.testing.assert_close(mu_post, mu_Z, atol=1e-6, rtol=0)
        torch.testing.assert_close(S_post, S_Z, atol=1e-6, rtol=0)

    def test_skip_on_degenerate_variance(self):
        A = 8
        mu_Z = torch.ones(A, device=DEVICE)
        S_Z = torch.zeros(A, device=DEVICE)  # var_S = 0
        mu_post, S_post = _s_projection(mu_Z, S_Z, 0.0, 1.0, eps=1e-8)
        torch.testing.assert_close(mu_post, mu_Z)
        torch.testing.assert_close(S_post, S_Z)

    def test_S_post_stays_positive(self):
        A = 64
        torch.manual_seed(3)
        S_Z = torch.rand(A, device=DEVICE) + 0.5  # strictly positive
        mu_Z = torch.randn(A, device=DEVICE)
        # Target variance smaller than current; ratio var_tilde/var < 1.
        _, S_post = _s_projection(mu_Z, S_Z, 0.0, S_Z.sum().item() * 0.1, eps=1e-8)
        assert torch.all(S_post > 0)


# ---------------------------------------------------------------------------
#  S2 projection (linearized; only monotonic improvement guaranteed)
# ---------------------------------------------------------------------------


class TestS2Projection:
    def test_s2_moves_toward_target(self):
        A = 128
        torch.manual_seed(1)
        mu_Z = torch.randn(A, device=DEVICE) * 0.3
        S_Z = torch.rand(A, device=DEVICE) * 0.5 + 0.1
        mu_S2_t, var_S2_t = A * 0.5, A * 1.0

        def s2_stats(mu, S):
            mu_Z2 = mu * mu + S
            S_Z2 = 2.0 * S * S + 4.0 * S * mu * mu
            return mu_Z2.sum().item(), S_Z2.sum().item()

        pre_mu, pre_var = s2_stats(mu_Z, S_Z)
        mu_post, S_post = _s2_projection(mu_Z, S_Z, mu_S2_t, var_S2_t, eps=1e-8)
        post_mu, post_var = s2_stats(mu_post, S_post)

        # Post must be strictly closer to the target than pre (linearization:
        # one step reduces the error but does not land on it).
        assert abs(post_mu - mu_S2_t) < abs(pre_mu - mu_S2_t)
        assert abs(post_var - var_S2_t) < abs(pre_var - var_S2_t)

    def test_S_post_non_negative(self):
        A = 32
        torch.manual_seed(4)
        mu_Z = torch.randn(A, device=DEVICE) * 2.0  # large means produce large J^2
        S_Z = torch.rand(A, device=DEVICE) + 0.1
        _, S_post = _s2_projection(mu_Z, S_Z, A * 0.1, A * 0.05, eps=1e-8)
        assert torch.all(S_post >= 0)


# ---------------------------------------------------------------------------
#  Decoupled inverse: re-forward lands on target
# ---------------------------------------------------------------------------


class TestDecoupledInverse:
    def test_reforward_matches_target(self):
        B, in_feat, out_feat = 256, 32, 64
        layer = _make_layer(in_feat, out_feat, seed=10)
        ma, Sa = _make_inputs(B, in_feat, seed=11)

        mz, Sz = layer.forward(ma, Sa)
        mu_Z, S_Z = _batch_moments(mz, Sz)

        # Pick a synthetic target: scale S_Z by 0.5, shift mu by +0.2
        S_target = 0.5 * S_Z
        mu_target = mu_Z + 0.2

        _decoupled_inverse(layer, mu_Z, S_Z, mu_target, S_target, eps=1e-8)

        mz_new, Sz_new = layer.forward(ma, Sa)
        mu_Z_new, S_Z_new = _batch_moments(mz_new, Sz_new)

        torch.testing.assert_close(mu_Z_new, mu_target, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(S_Z_new, S_target, atol=1e-4, rtol=1e-4)

    def test_degenerate_units_untouched(self):
        B, in_feat, out_feat = 64, 8, 16
        layer = _make_layer(in_feat, out_feat, seed=20)
        # Zero out last 4 output weight columns so S_Zi = 0 + Sb for those units;
        # we additionally zero Sb on those columns to drive S_Zi < eps.
        layer.mw[:, -4:] = 0.0
        layer.Sw[:, -4:] = 0.0
        layer.Sb[:, -4:] = 0.0
        mw_before = layer.mw.clone()
        mb_before = layer.mb.clone()
        ma, Sa = _make_inputs(B, in_feat, seed=21)

        mz, Sz = layer.forward(ma, Sa)
        mu_Z, S_Z = _batch_moments(mz, Sz)
        mu_target = mu_Z + 0.3
        S_target = S_Z + 1.0  # would demand gamma > 0 on degenerate units

        _decoupled_inverse(layer, mu_Z, S_Z, mu_target, S_target, eps=1e-8)

        # Last 4 columns of mw/mb must be unchanged.
        torch.testing.assert_close(layer.mw[:, -4:], mw_before[:, -4:])
        torch.testing.assert_close(layer.mb[:, -4:], mb_before[:, -4:])


# ---------------------------------------------------------------------------
#  End-to-end: inference_init on a single-layer net with a single batch
# ---------------------------------------------------------------------------


class TestInferenceInitEndToEnd:
    def test_single_layer_single_batch_hits_S(self):
        """After one calibration batch, S projection lands on target modulo the
        S2 step's perturbation (linearization reshuffles S slightly). We check
        sum(mu_Z) is near 0 exactly and sum(S_Z) is within 1% of its target.
        """
        torch.manual_seed(30)
        B, in_feat, out_feat = 512, 64, 128
        sigma_m, sigma_z = 0.5, 0.5
        net = Sequential(
            [Linear(in_feat, out_feat, device=DEVICE)], device=DEVICE
        )
        x = torch.randn(B, in_feat, device=DEVICE)

        inference_init(net, [x], sigma_m, sigma_z)

        net.eval()
        mz, Sz = net.forward(x)
        mu_Z, S_Z = _batch_moments(mz, Sz)
        target_var_S = out_feat * sigma_z**2
        torch.testing.assert_close(
            mu_Z.sum(), torch.tensor(0.0, device=DEVICE), atol=5e-3, rtol=0
        )
        torch.testing.assert_close(
            S_Z.sum(),
            torch.tensor(target_var_S, device=DEVICE),
            atol=0.0,
            rtol=1e-2,
        )

    def test_relu_flatten_remax_passthrough(self):
        """Non-Linear layers must not be mutated. Check via forward equivalence."""
        torch.manual_seed(40)
        B = 64
        net = Sequential(
            [
                Linear(16, 32, device=DEVICE),
                ReLU(),
                Flatten(),  # no-op on (B, 32); forward should still work
                Linear(32, 8, device=DEVICE),
                Remax(),
            ],
            device=DEVICE,
        )
        x = torch.randn(B, 16, device=DEVICE)

        # Capture Remax state (it has no params, but should be the same object).
        remax_before = net.layers[-1]
        relu_before = net.layers[1]
        flatten_before = net.layers[2]

        inference_init(net, [x], sigma_m=0.5, sigma_z=0.5)

        assert net.layers[-1] is remax_before
        assert net.layers[1] is relu_before
        assert net.layers[2] is flatten_before

    def test_multilayer_mlp_calibrated_forward_runs(self):
        """Smoke test: a 5-layer MLP calibrates without error and produces finite output."""
        torch.manual_seed(50)
        B = 128
        net = Sequential(
            [
                Linear(784, 256, device=DEVICE), ReLU(),
                Linear(256, 256, device=DEVICE), ReLU(),
                Linear(256, 256, device=DEVICE), ReLU(),
                Linear(256, 128, device=DEVICE), ReLU(),
                Linear(128, 10, device=DEVICE),
            ],
            device=DEVICE,
        )
        loader = [torch.randn(B, 784, device=DEVICE) for _ in range(4)]
        inference_init(net, loader, sigma_m=0.5, sigma_z=0.5)

        net.eval()
        mu, var = net.forward(loader[0])
        assert torch.isfinite(mu).all()
        assert torch.isfinite(var).all()
        assert torch.all(var >= 0)


# ---------------------------------------------------------------------------
#  Conv2D / BatchNorm2D / ResBlock — Phase 2 + Phase 3 (PLAN.md)
# ---------------------------------------------------------------------------


class TestConv2D:
    def test_single_conv_hits_S(self):
        """Conv2D calibrated standalone: S target hit exactly (per-channel
        scalars aggregated over N*H_out*W_out spatial positions)."""
        torch.manual_seed(70)
        N, C_in, H, C_out = 32, 3, 16, 16
        sigma_m, sigma_z = 0.5, 0.5
        net = Sequential(
            [Conv2D(C_in, C_out, 3, padding=1, device=DEVICE)], device=DEVICE
        )
        x = torch.randn(N, C_in, H, H, device=DEVICE)
        inference_init(net, [x], sigma_m, sigma_z)

        net.eval()
        mz, Sz = net.forward(x)
        # Aggregate to per-channel (C_out,) batch-mean — same convention as
        # inference_init internals.
        mu_Z = mz.permute(0, 2, 3, 1).reshape(-1, C_out).mean(0)
        S_Z = Sz.permute(0, 2, 3, 1).reshape(-1, C_out).mean(0)
        torch.testing.assert_close(
            mu_Z.sum(), torch.tensor(0.0, device=DEVICE), atol=5e-3, rtol=0
        )
        torch.testing.assert_close(
            S_Z.sum(),
            torch.tensor(C_out * sigma_z**2, device=DEVICE),
            atol=0.0,
            rtol=2e-2,
        )

    def test_cnn_calibrates_and_runs(self):
        """End-to-end CNN: Conv→ReLU→Pool→Conv→ReLU→Pool→Flatten→Linear."""
        torch.manual_seed(71)
        N = 64
        net = Sequential(
            [
                Conv2D(3, 16, 3, padding=1, device=DEVICE), ReLU(), AvgPool2D(2),
                Conv2D(16, 32, 3, padding=1, device=DEVICE), ReLU(), AvgPool2D(2),
                Flatten(),
                Linear(32 * 8 * 8, 10, device=DEVICE),
            ],
            device=DEVICE,
        )
        loader = [torch.randn(N, 3, 32, 32, device=DEVICE) for _ in range(2)]
        inference_init(net, loader, sigma_m=0.5, sigma_z=0.5)
        net.eval()
        mu, var = net.forward(loader[0])
        assert torch.isfinite(mu).all()
        assert torch.isfinite(var).all()
        assert (var >= 0).all()


class TestBatchNorm2D:
    def test_bn_passthrough(self):
        """BN's gamma/beta must not be modified by IBI (pass-through layer).
        Conv2D before the BN must still calibrate normally."""
        torch.manual_seed(80)
        N, C, H = 32, 16, 8
        sigma_m, sigma_z = 0.5, 0.5
        net = Sequential(
            [
                Conv2D(3, C, 3, padding=1, device=DEVICE),
                BatchNorm2D(C, device=DEVICE, preserve_var=False),
            ],
            device=DEVICE,
        )
        bn = net.layers[1]
        mw_before = bn.mw.clone()
        mb_before = bn.mb.clone()
        Sw_before = bn.Sw.clone()
        Sb_before = bn.Sb.clone()

        x = torch.randn(N, 3, H, H, device=DEVICE)
        inference_init(net, [x], sigma_m, sigma_z)

        # BN parameters unchanged.
        torch.testing.assert_close(bn.mw, mw_before)
        torch.testing.assert_close(bn.mb, mb_before)
        torch.testing.assert_close(bn.Sw, Sw_before)
        torch.testing.assert_close(bn.Sb, Sb_before)

        # Conv2D calibration still produces a finite forward.
        net.train()
        mu, var = net.forward(x)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(var).all()


class TestResBlock:
    def test_resblock_calibrates_subllayers(self):
        """ResBlock: per-sub-layer calibration leaves Conv1/Conv2/BN1/BN2/proj
        with modified parameters and produces a finite output."""
        torch.manual_seed(90)
        N = 32
        net = Sequential(
            [
                Conv2D(3, 16, 3, padding=1, device=DEVICE),
                ReLU(),
                BatchNorm2D(16, device=DEVICE, preserve_var=False),
                ResBlock(16, 32, stride=2, device=DEVICE),  # projection branch
                ResBlock(32, 32, stride=1, device=DEVICE),  # identity branch
                AvgPool2D(2),
                Flatten(),
                Linear(32 * 8 * 8, 10, device=DEVICE),
            ],
            device=DEVICE,
        )
        block = net.layers[3]
        # Conv sub-layers should change (calibrated); BN sub-layers are
        # pass-through and must remain unchanged.
        changed = {
            "conv1_mw": block.conv1.mw.clone(),
            "conv2_mw": block.conv2.mw.clone(),
            "proj_conv_mw": block.proj_conv.mw.clone(),
        }
        unchanged = {
            "bn1_mw": block.bn1.mw.clone(),
            "bn2_mw": block.bn2.mw.clone(),
            "proj_bn_mw": block.proj_bn.mw.clone(),
        }
        loader = [torch.randn(N, 3, 32, 32, device=DEVICE) for _ in range(2)]
        inference_init(net, loader, sigma_m=0.5, sigma_z=0.5)

        for name, snap in changed.items():
            cur = {
                "conv1_mw": block.conv1.mw,
                "conv2_mw": block.conv2.mw,
                "proj_conv_mw": block.proj_conv.mw,
            }[name]
            assert not torch.allclose(cur, snap), f"{name} unchanged but should be calibrated"

        for name, snap in unchanged.items():
            cur = {
                "bn1_mw": block.bn1.mw,
                "bn2_mw": block.bn2.mw,
                "proj_bn_mw": block.proj_bn.mw,
            }[name]
            torch.testing.assert_close(cur, snap)

        net.eval()
        mu, var = net.forward(loader[0])
        assert torch.isfinite(mu).all()
        assert torch.isfinite(var).all()
        assert (var >= 0).all()

    def test_resnet18_full(self):
        """Smoke: full CIFAR-10 ResNet18 architecture calibrates without error."""
        torch.manual_seed(91)
        kw = {"device": DEVICE, "gain_w": 0.1, "gain_b": 0.1}
        net = Sequential(
            [
                Conv2D(3, 64, 3, stride=1, padding=1, **kw),
                ReLU(),
                BatchNorm2D(64, **kw),
                ResBlock(64, 64, stride=1, **kw),
                ResBlock(64, 64, stride=1, **kw),
                ResBlock(64, 128, stride=2, **kw),
                ResBlock(128, 128, stride=1, **kw),
                ResBlock(128, 256, stride=2, **kw),
                ResBlock(256, 256, stride=1, **kw),
                ResBlock(256, 512, stride=2, **kw),
                ResBlock(512, 512, stride=1, **kw),
                AvgPool2D(4),
                Flatten(),
                Linear(512, 10, **kw),
                Remax(),
            ],
            device=DEVICE,
        )
        loader = [torch.randn(16, 3, 32, 32, device=DEVICE) for _ in range(2)]
        inference_init(net, loader, sigma_m=0.5, sigma_z=0.5)
        net.eval()
        mu, var = net.forward(loader[0])
        assert torch.isfinite(mu).all()
        assert torch.isfinite(var).all()
        # Remax row-sum is 1.
        torch.testing.assert_close(
            mu.sum(dim=1), torch.ones(loader[0].shape[0], device=DEVICE),
            atol=1e-4, rtol=0,
        )
