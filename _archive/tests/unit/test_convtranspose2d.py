"""Unit tests for ConvTranspose2D layer.

Covers:
    - Forward output shapes
    - Forward mean matches F.conv_transpose2d with zero-noise weights
    - Forward variance formula (Sa-term, ma^2-term, Sb-term separately)
    - Backward delta shapes match input
    - Backward delta_ma via F.conv2d identity check
    - Weight gradient formula verified against double-loop reference
    - Bias gradient sums
    - CPU execution (no CUDA required for ConvTranspose2D)
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from triton_tagi.layers.convtranspose2d import ConvTranspose2D


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_layer(C_in=3, C_out=5, k=3, stride=1, padding=0, device="cpu"):
    layer = ConvTranspose2D(C_in, C_out, k, stride=stride, padding=padding, device=device)
    return layer


def _h_out(H, k, stride, padding):
    return (H - 1) * stride - 2 * padding + k


# ──────────────────────────────────────────────────────────────────────────────
#  Shape tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1), (1, 1)])
def test_forward_shape(stride, padding):
    N, C_in, C_out, k, H = 2, 3, 5, 3, 6
    layer = _make_layer(C_in, C_out, k, stride=stride, padding=padding)
    ma = torch.randn(N, C_in, H, H)
    Sa = torch.rand(N, C_in, H, H) * 0.1
    mz, Sz = layer.forward(ma, Sa)
    Ho = _h_out(H, k, stride, padding)
    assert mz.shape == (N, C_out, Ho, Ho), f"mz {mz.shape}"
    assert Sz.shape == (N, C_out, Ho, Ho), f"Sz {Sz.shape}"


def test_backward_shape():
    N, C_in, C_out, k, H = 2, 3, 5, 3, 6
    stride, padding = 2, 1
    layer = _make_layer(C_in, C_out, k, stride=stride, padding=padding)
    ma = torch.randn(N, C_in, H, H)
    Sa = torch.rand(N, C_in, H, H) * 0.1
    mz, Sz = layer.forward(ma, Sa)
    delta_mz = torch.randn_like(mz)
    delta_Sz = torch.rand_like(Sz) * 0.1
    d_ma, d_Sa = layer.backward(delta_mz, delta_Sz)
    assert d_ma.shape == (N, C_in, H, H), f"d_ma {d_ma.shape}"
    assert d_Sa.shape == (N, C_in, H, H), f"d_Sa {d_Sa.shape}"


# ──────────────────────────────────────────────────────────────────────────────
#  Forward mean
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 0), (2, 1)])
def test_forward_mean_matches_pytorch(stride, padding):
    """With Sa=0 and Sw→0, mz must equal F.conv_transpose2d(ma, mw_4d, bias)."""
    torch.manual_seed(42)
    N, C_in, C_out, k, H = 2, 3, 4, 3, 6
    layer = _make_layer(C_in, C_out, k, stride=stride, padding=padding)

    # Zero weight variance and input variance so Sz-path doesn't contaminate mz
    layer.Sw = torch.zeros_like(layer.Sw)
    layer.Sb = torch.zeros_like(layer.Sb)

    ma = torch.randn(N, C_in, H, H)
    Sa = torch.zeros(N, C_in, H, H)
    mz, _ = layer.forward(ma, Sa)

    # Reference: pure conv_transpose2d
    mw_4d = layer.mw.view(C_in, k, k, C_out).permute(0, 3, 1, 2).contiguous()
    ref = F.conv_transpose2d(ma, mw_4d, bias=layer.mb.view(C_out), stride=stride, padding=padding)
    torch.testing.assert_close(mz, ref, atol=1e-5, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Forward variance
# ──────────────────────────────────────────────────────────────────────────────


def test_forward_variance_sa_term():
    """With Sw=0, Sz must equal F.conv_transpose2d(Sa, mw^2) + Sb."""
    torch.manual_seed(7)
    N, C_in, C_out, k, H = 2, 3, 4, 3, 5
    layer = _make_layer(C_in, C_out, k)
    layer.Sw = torch.zeros_like(layer.Sw)

    ma = torch.zeros(N, C_in, H, H)
    Sa = torch.rand(N, C_in, H, H) * 0.5 + 0.1
    _, Sz = layer.forward(ma, Sa)

    mw_4d = layer.mw.view(C_in, k, k, C_out).permute(0, 3, 1, 2).contiguous()
    ref = F.conv_transpose2d(Sa, mw_4d**2) + layer.Sb.view(1, C_out, 1, 1)
    torch.testing.assert_close(Sz, ref, atol=1e-5, rtol=0)


def test_forward_variance_ma2_term():
    """With Sa=0, Sz must equal F.conv_transpose2d(ma^2, Sw) + Sb."""
    torch.manual_seed(11)
    N, C_in, C_out, k, H = 2, 3, 4, 3, 5
    layer = _make_layer(C_in, C_out, k)

    ma = torch.randn(N, C_in, H, H)
    Sa = torch.zeros(N, C_in, H, H)
    _, Sz = layer.forward(ma, Sa)

    mw_4d = layer.mw.view(C_in, k, k, C_out).permute(0, 3, 1, 2).contiguous()
    Sw_4d = layer.Sw.view(C_in, k, k, C_out).permute(0, 3, 1, 2).contiguous()
    ref = F.conv_transpose2d(ma**2, Sw_4d) + layer.Sb.view(1, C_out, 1, 1)
    torch.testing.assert_close(Sz, ref, atol=1e-5, rtol=0)


def test_forward_variance_nonnegative():
    """Sz must be ≥ 0 everywhere."""
    torch.manual_seed(3)
    layer = _make_layer(2, 4, 3)
    ma = torch.randn(3, 2, 8, 8)
    Sa = torch.rand(3, 2, 8, 8) * 0.1
    _, Sz = layer.forward(ma, Sa)
    assert (Sz >= 0).all()


# ──────────────────────────────────────────────────────────────────────────────
#  Backward delta propagation
# ──────────────────────────────────────────────────────────────────────────────


def test_backward_delta_ma_formula():
    """delta_ma = F.conv2d(delta_mz, mw_4d) — the transpose of the forward."""
    torch.manual_seed(99)
    N, C_in, C_out, k, H = 2, 3, 4, 3, 6
    stride, padding = 2, 1
    layer = _make_layer(C_in, C_out, k, stride=stride, padding=padding)

    ma = torch.randn(N, C_in, H, H)
    Sa = torch.zeros_like(ma)
    mz, Sz = layer.forward(ma, Sa)

    delta_mz = torch.randn_like(mz)
    delta_Sz = torch.zeros_like(mz)
    d_ma, _ = layer.backward(delta_mz, delta_Sz)

    mw_4d = layer.mw.view(C_in, k, k, C_out).permute(0, 3, 1, 2).contiguous()
    ref = F.conv2d(delta_mz, mw_4d, stride=stride, padding=padding)
    torch.testing.assert_close(d_ma, ref, atol=1e-5, rtol=0)


def test_backward_delta_sa_formula():
    """delta_Sa = F.conv2d(delta_Sz, mw_4d^2)."""
    torch.manual_seed(88)
    N, C_in, C_out, k, H = 2, 3, 4, 3, 6
    layer = _make_layer(C_in, C_out, k)

    ma = torch.randn(N, C_in, H, H)
    Sa = torch.zeros_like(ma)
    mz, Sz = layer.forward(ma, Sa)

    delta_mz = torch.zeros_like(mz)
    delta_Sz = torch.rand_like(Sz) * 0.1
    _, d_Sa = layer.backward(delta_mz, delta_Sz)

    mw_4d = layer.mw.view(C_in, k, k, C_out).permute(0, 3, 1, 2).contiguous()
    ref = F.conv2d(delta_Sz, mw_4d**2)
    torch.testing.assert_close(d_Sa, ref, atol=1e-5, rtol=0)


def test_backward_delta_sa_nonnegative():
    """d_Sa must be ≥ 0 when delta_Sz ≥ 0 (since mw^2 ≥ 0)."""
    torch.manual_seed(5)
    layer = _make_layer(2, 3, 3)
    ma = torch.randn(2, 2, 6, 6)
    Sa = torch.rand_like(ma) * 0.1
    mz, Sz = layer.forward(ma, Sa)
    delta_mz = torch.zeros_like(mz)
    delta_Sz = torch.rand_like(Sz) * 0.5
    _, d_Sa = layer.backward(delta_mz, delta_Sz)
    assert (d_Sa >= 0).all()


# ──────────────────────────────────────────────────────────────────────────────
#  Weight gradient
# ──────────────────────────────────────────────────────────────────────────────


def test_weight_gradient_formula():
    """delta_mw = Sw * grad where grad is verified against PyTorch autograd."""
    torch.manual_seed(17)
    N, C_in, C_out, k, H = 2, 3, 4, 3, 5
    stride, padding = 2, 1
    layer = _make_layer(C_in, C_out, k, stride=stride, padding=padding)

    # Compute reference grad via autograd
    mw_4d_ref = layer.mw.view(C_in, k, k, C_out).permute(0, 3, 1, 2).detach().clone().requires_grad_(True)
    ma = torch.randn(N, C_in, H, H)
    Sa = torch.zeros_like(ma)

    y_ref = F.conv_transpose2d(ma, mw_4d_ref, stride=stride, padding=padding)
    delta_mz = torch.randn_like(y_ref)
    y_ref.backward(delta_mz)
    # autograd gives dL/d(mw_4d) in (C_in, C_out, k, k) layout
    # Convert to triton (K, C_out) layout
    dw_4d_ref = mw_4d_ref.grad  # (C_in, C_out, k, k)
    dw_ref = dw_4d_ref.permute(0, 2, 3, 1).reshape(C_in * k * k, C_out)  # (K, C_out)

    # Layer backward
    mz, Sz = layer.forward(ma, Sa)
    layer.backward(delta_mz, torch.zeros_like(delta_mz))

    expected_delta_mw = layer.Sw * dw_ref
    torch.testing.assert_close(layer.delta_mw, expected_delta_mw, atol=1e-4, rtol=0)


def test_bias_gradient():
    """delta_mb = Sb * sum_over_batch_spatial(delta_mz)."""
    torch.manual_seed(21)
    N, C_in, C_out, k, H = 3, 2, 4, 3, 5
    layer = _make_layer(C_in, C_out, k)

    ma = torch.randn(N, C_in, H, H)
    Sa = torch.zeros_like(ma)
    mz, Sz = layer.forward(ma, Sa)
    delta_mz = torch.randn_like(mz)
    delta_Sz = torch.zeros_like(mz)
    layer.backward(delta_mz, delta_Sz)

    ref_grad_mb = delta_mz.sum(dim=(0, 2, 3)).view(1, C_out)
    expected = layer.Sb * ref_grad_mb
    torch.testing.assert_close(layer.delta_mb, expected, atol=1e-5, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Parameter shapes & counts
# ──────────────────────────────────────────────────────────────────────────────


def test_parameter_shapes():
    C_in, C_out, k = 3, 5, 3
    layer = _make_layer(C_in, C_out, k)
    K = C_in * k * k
    assert layer.mw.shape == (K, C_out)
    assert layer.Sw.shape == (K, C_out)
    assert layer.mb.shape == (1, C_out)
    assert layer.Sb.shape == (1, C_out)


def test_num_parameters():
    C_in, C_out, k = 3, 5, 3
    layer = _make_layer(C_in, C_out, k)
    K = C_in * k * k
    # 2 × (weight params + bias params)
    expected = 2 * (K * C_out + C_out)
    assert layer.num_parameters == expected


# ──────────────────────────────────────────────────────────────────────────────
#  Device
# ──────────────────────────────────────────────────────────────────────────────


def test_runs_on_cpu():
    """ConvTranspose2D must work on CPU without error (no custom Triton kernels)."""
    layer = _make_layer(2, 3, 3, device="cpu")
    ma = torch.randn(2, 2, 6, 6)
    Sa = torch.rand_like(ma) * 0.1
    mz, Sz = layer.forward(ma, Sa)
    delta_mz = torch.randn_like(mz)
    delta_Sz = torch.rand_like(Sz) * 0.1
    d_ma, d_Sa = layer.backward(delta_mz, delta_Sz)
    assert d_ma.shape == (2, 2, 6, 6)


@pytest.mark.cuda
def test_runs_on_cuda():
    layer = _make_layer(2, 3, 3, device="cuda")
    ma = torch.randn(2, 2, 6, 6, device="cuda")
    Sa = torch.rand_like(ma) * 0.1
    mz, Sz = layer.forward(ma, Sa)
    delta_mz = torch.randn_like(mz)
    delta_Sz = torch.rand_like(Sz) * 0.1
    d_ma, d_Sa = layer.backward(delta_mz, delta_Sz)
    assert d_ma.device.type == "cuda"
