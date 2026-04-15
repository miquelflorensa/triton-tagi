"""Validation tests: triton-tagi AvgPool2D against fp64 reference.

pytagi's AvgPool2d crashes with a FPE when run standalone (it needs a preceding
Conv2d to infer spatial dimensions), so the reference is a pure fp64 Python
implementation of the same formulas.

Forward formula (non-overlapping k×k pooling, stride = k):
    μ_out[n,c,oh,ow] = (1/k²) Σ_{kh,kw} μ_in[n,c,oh·k+kh, ow·k+kw]
    S_out[n,c,oh,ow] = (1/k²) Σ_{kh,kw} S_in[n,c,oh·k+kh, ow·k+kw]

    (variance uses the same 1/k² factor as mean — cuTAGI/triton-tagi convention
     for high spatial correlation; this is NOT the independent-samples formula
     1/k⁴).

Backward formula (chain rule through the 1/k² mean):
    δ_μ_in[n,c,h,w] = δ_μ_out[n,c,h//k, w//k] / k²
    δ_S_in[n,c,h,w] = δ_S_out[n,c,h//k, w//k] / k²

Run with:
    pytest tests/validation/test_avgpool2d.py -v
"""

from __future__ import annotations

import pytest
import torch

from triton_tagi.layers.avgpool2d import AvgPool2D as TAvgPool2D

DEVICE = "cuda"
ATOL = 1e-5

pytestmark = pytest.mark.cuda


# ──────────────────────────────────────────────────────────────────────────────
#  fp64 references
# ──────────────────────────────────────────────────────────────────────────────


def _avgpool_forward_ref(ma: torch.Tensor, Sa: torch.Tensor, k: int):
    """AvgPool2D forward in fp64: mean of k×k blocks for both moments."""
    N, C, H, W = ma.shape
    H_out, W_out = H // k, W // k
    # Reshape to expose the k×k tile dimensions, then average
    ma_out = (
        ma.double()
        .reshape(N, C, H_out, k, W_out, k)
        .mean(dim=(3, 5))           # average over (kh, kw) → (N, C, H_out, W_out)
    )
    Sa_out = (
        Sa.double()
        .reshape(N, C, H_out, k, W_out, k)
        .mean(dim=(3, 5))
    )
    return ma_out.float(), Sa_out.float()


def _avgpool_backward_ref(delta_ma: torch.Tensor, delta_Sa: torch.Tensor, k: int):
    """AvgPool2D backward in fp64: distribute delta equally to all k×k inputs."""
    # Nearest-neighbour upsample by k, then scale by 1/k²
    d_ma = (
        delta_ma.double()
        .repeat_interleave(k, dim=2)
        .repeat_interleave(k, dim=3)
        / (k * k)
    )
    d_Sa = (
        delta_Sa.double()
        .repeat_interleave(k, dim=2)
        .repeat_interleave(k, dim=3)
        / (k * k)
    )
    return d_ma.float(), d_Sa.float()


# ──────────────────────────────────────────────────────────────────────────────
#  Level 1: Forward
# ──────────────────────────────────────────────────────────────────────────────


def test_avgpool2d_forward_mean():
    """AvgPool2D forward mean = block average, matches fp64 reference."""
    torch.manual_seed(0)
    N, C, H, W, k = 4, 8, 16, 16, 2
    ma = torch.randn(N, C, H, W, device=DEVICE)
    Sa = torch.rand(N, C, H, W, device=DEVICE).abs() * 0.5 + 1e-4

    layer = TAvgPool2D(k)
    ma_out, _ = layer.forward(ma, Sa)

    ref_ma, _ = _avgpool_forward_ref(ma.cpu(), Sa.cpu(), k)
    torch.testing.assert_close(ma_out.cpu(), ref_ma, atol=ATOL, rtol=0)


def test_avgpool2d_forward_variance():
    """AvgPool2D forward variance = block average of input variances."""
    torch.manual_seed(1)
    N, C, H, W, k = 4, 8, 16, 16, 2
    ma = torch.randn(N, C, H, W, device=DEVICE)
    Sa = torch.rand(N, C, H, W, device=DEVICE).abs() * 0.5 + 1e-4

    layer = TAvgPool2D(k)
    _, Sa_out = layer.forward(ma, Sa)

    _, ref_Sa = _avgpool_forward_ref(ma.cpu(), Sa.cpu(), k)
    torch.testing.assert_close(Sa_out.cpu(), ref_Sa, atol=ATOL, rtol=0)


def test_avgpool2d_forward_k4():
    """AvgPool2D forward mean with k=4 matches fp64 reference."""
    torch.manual_seed(2)
    N, C, H, W, k = 2, 4, 32, 32, 4
    ma = torch.randn(N, C, H, W, device=DEVICE)
    Sa = torch.rand(N, C, H, W, device=DEVICE).abs() * 0.1 + 1e-4

    layer = TAvgPool2D(k)
    ma_out, Sa_out = layer.forward(ma, Sa)

    ref_ma, ref_Sa = _avgpool_forward_ref(ma.cpu(), Sa.cpu(), k)
    torch.testing.assert_close(ma_out.cpu(), ref_ma, atol=ATOL, rtol=0)
    torch.testing.assert_close(Sa_out.cpu(), ref_Sa, atol=ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 2: Backward
# ──────────────────────────────────────────────────────────────────────────────


def test_avgpool2d_backward_delta_ma():
    """AvgPool2D backward δ_μ_in = δ_μ_out / k² broadcast, matches fp64."""
    torch.manual_seed(0)
    N, C, H, W, k = 4, 8, 16, 16, 2
    ma = torch.randn(N, C, H, W, device=DEVICE)
    Sa = torch.rand(N, C, H, W, device=DEVICE).abs() * 0.5 + 1e-4
    delta_ma = torch.randn(N, C, H // k, W // k, device=DEVICE)
    delta_Sa = torch.rand(N, C, H // k, W // k, device=DEVICE).abs() * 0.01

    layer = TAvgPool2D(k)
    layer.forward(ma, Sa)
    d_ma, _ = layer.backward(delta_ma, delta_Sa)

    ref_d_ma, _ = _avgpool_backward_ref(delta_ma.cpu(), delta_Sa.cpu(), k)
    torch.testing.assert_close(d_ma.cpu(), ref_d_ma, atol=ATOL, rtol=0)


def test_avgpool2d_backward_delta_Sa():
    """AvgPool2D backward δ_S_in = δ_S_out / k² broadcast, matches fp64."""
    torch.manual_seed(1)
    N, C, H, W, k = 4, 8, 16, 16, 2
    ma = torch.randn(N, C, H, W, device=DEVICE)
    Sa = torch.rand(N, C, H, W, device=DEVICE).abs() * 0.5 + 1e-4
    delta_ma = torch.randn(N, C, H // k, W // k, device=DEVICE)
    delta_Sa = torch.rand(N, C, H // k, W // k, device=DEVICE).abs() * 0.01

    layer = TAvgPool2D(k)
    layer.forward(ma, Sa)
    _, d_Sa = layer.backward(delta_ma, delta_Sa)

    _, ref_d_Sa = _avgpool_backward_ref(delta_ma.cpu(), delta_Sa.cpu(), k)
    torch.testing.assert_close(d_Sa.cpu(), ref_d_Sa, atol=ATOL, rtol=0)
