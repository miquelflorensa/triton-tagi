"""Validation tests: triton-tagi Conv2D against cuTAGI (pytagi).

Conv2D reduces to the same im2col + fused-matmul pipeline as Linear, so the
same three validation levels apply.

Level 1 — Forward: identical weights produce matching (mz, Sz).
Level 2 — Backward: given identical output deltas, input deltas and parameter
           deltas match the fp64 analytical reference.
Level 3 — Update: one full step produces matching updated mw.

Weight layout translation (same as Linear):
    triton  mw : (K, C_out)  where K = C_in * kH * kW
    pytagi  mu_w : (C_out * K,) row-major  →  mw.T.flatten()

pytagi Conv2d requires in_width / in_height at construction time.

Run with:
    pytest tests/validation/test_conv2d.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytagi.nn import Conv2d as PConv2d
from pytagi.nn import OutputUpdater
from pytagi.nn import Sequential as PSequential

from triton_tagi.layers.conv2d import Conv2D as TConv2D
from triton_tagi.network import Sequential as TSequential

DEVICE = "cuda"
MEAN_ATOL = 1e-4
VAR_ATOL = 1e-4
UPDATE_ATOL = 1e-4

pytestmark = pytest.mark.cuda


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_triton_conv(C_in, C_out, k, padding, mw, Sw, mb, Sb):
    layer = TConv2D(C_in, C_out, k, padding=padding, device=DEVICE)
    layer.mw = mw.clone().to(DEVICE)
    layer.Sw = Sw.clone().to(DEVICE)
    layer.mb = mb.clone().to(DEVICE)
    layer.Sb = Sb.clone().to(DEVICE)
    return layer


def _make_pytagi_conv(C_in, C_out, k, padding, H, W, mw, Sw, mb, Sb):
    """Build a pytagi Conv2d with the same weights as the triton layer.

    pytagi weight layout: (C_out, K) row-major flat = mw.T.flatten()
    """
    net = PSequential(PConv2d(C_in, C_out, k, padding=padding, in_width=W, in_height=H))
    net.preinit_layer()
    key = list(net.state_dict().keys())[0]
    net.load_state_dict(
        {
            key: (
                mw.T.cpu().numpy().flatten().tolist(),
                Sw.T.cpu().numpy().flatten().tolist(),
                mb.squeeze().cpu().numpy().tolist(),
                Sb.squeeze().cpu().numpy().tolist(),
            )
        }
    )
    return net


def _pytagi_conv_weights(net, key):
    """Extract (mw, Sw, mb, Sb) from pytagi, converting back to triton layout."""
    mu_w_flat, var_w_flat, mu_b_flat, var_b_flat = net.state_dict()[key]
    C_out = len(mu_b_flat)
    K = len(mu_w_flat) // C_out
    mw = torch.tensor(mu_w_flat).reshape(C_out, K).T  # (K, C_out)
    Sw = torch.tensor(var_w_flat).reshape(C_out, K).T
    mb = torch.tensor(mu_b_flat).reshape(1, C_out)
    Sb = torch.tensor(var_b_flat).reshape(1, C_out)
    return mw, Sw, mb, Sb


def _random_conv_params(C_in, C_out, k):
    K = C_in * k * k
    mw = torch.randn(K, C_out)
    Sw = torch.rand(K, C_out).abs() * 0.1 + 1e-6
    mb = torch.randn(1, C_out)
    Sb = torch.rand(1, C_out).abs() * 0.1 + 1e-6
    return mw, Sw, mb, Sb


# ──────────────────────────────────────────────────────────────────────────────
#  Level 1: Forward
# ──────────────────────────────────────────────────────────────────────────────


def test_conv2d_forward_mean():
    """Output means match between triton-tagi and cuTAGI."""
    torch.manual_seed(0)
    N, C_in, H, W, C_out, k = 4, 3, 8, 8, 8, 3
    mw, Sw, mb, Sb = _random_conv_params(C_in, C_out, k)
    ma = torch.randn(N, C_in, H, W)

    tri = _make_triton_conv(C_in, C_out, k, 1, mw, Sw, mb, Sb)
    mz_tri, _ = tri.forward(ma.to(DEVICE), torch.zeros_like(ma.to(DEVICE)))

    cut = _make_pytagi_conv(C_in, C_out, k, 1, H, W, mw, Sw, mb, Sb)
    m_flat, _ = cut(ma.numpy().flatten().astype(np.float32))
    mz_cut = torch.tensor(m_flat).reshape(N, C_out, H, W)

    torch.testing.assert_close(mz_tri.cpu(), mz_cut, atol=MEAN_ATOL, rtol=0)


def test_conv2d_forward_variance():
    """Output variances match between triton-tagi and cuTAGI."""
    torch.manual_seed(1)
    N, C_in, H, W, C_out, k = 4, 3, 8, 8, 8, 3
    mw, Sw, mb, Sb = _random_conv_params(C_in, C_out, k)
    ma = torch.randn(N, C_in, H, W)

    tri = _make_triton_conv(C_in, C_out, k, 1, mw, Sw, mb, Sb)
    _, Sz_tri = tri.forward(ma.to(DEVICE), torch.zeros_like(ma.to(DEVICE)))

    cut = _make_pytagi_conv(C_in, C_out, k, 1, H, W, mw, Sw, mb, Sb)
    _, v_flat = cut(ma.numpy().flatten().astype(np.float32))
    Sz_cut = torch.tensor(v_flat).reshape(N, C_out, H, W)

    torch.testing.assert_close(Sz_tri.cpu(), Sz_cut, atol=VAR_ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 2: Backward — input delta propagation and parameter deltas
#
#  After im2col, Conv2D backward is identical to Linear backward on the
#  patch matrix.  Reference formulas:
#    delta_patches_ma = delta_mz_flat @ mw^T         (then col2im)
#    delta_patches_Sa = delta_Sz_flat @ (mw²)^T      (then col2im)
#    delta_mw = Sw * (patches_ma^T @ delta_mz_flat)
#    delta_Sw = Sw² * ((patches_ma²)^T @ delta_Sz_flat)
# ──────────────────────────────────────────────────────────────────────────────


def _setup_conv_backward(seed=0):
    torch.manual_seed(seed)
    N, C_in, H, W, C_out, k = 4, 3, 8, 8, 8, 3
    mw, Sw, mb, Sb = _random_conv_params(C_in, C_out, k)
    ma = torch.randn(N, C_in, H, W)
    delta_mz = torch.randn(N, C_out, H, W)
    delta_Sz = torch.rand(N, C_out, H, W).abs() * 0.01

    tri = _make_triton_conv(C_in, C_out, k, 1, mw, Sw, mb, Sb)
    tri.forward(ma.to(DEVICE), torch.zeros_like(ma.to(DEVICE)))
    return tri, ma, delta_mz, delta_Sz


def test_conv2d_backward_delta_ma():
    """delta_ma (spatial) matches fp64 reference: delta_patches @ mw^T → col2im."""
    tri, ma, delta_mz, delta_Sz = _setup_conv_backward()
    d_ma, _ = tri.backward(delta_mz.to(DEVICE), delta_Sz.to(DEVICE))

    # fp64 reference using the cached patches from forward
    patches = tri.patches_ma.cpu().double()
    dmz_flat = delta_mz.permute(0, 2, 3, 1).reshape(-1, tri.C_out).double()
    ref_patches_d_ma = dmz_flat @ tri.mw.cpu().double().T  # (NL, K)

    # Compare total column sums (col2im is a scatter-add; verify energy is preserved)
    tri_col_sum = d_ma.cpu().double().sum()
    ref_col_sum = ref_patches_d_ma.sum()
    assert abs((tri_col_sum - ref_col_sum).item()) / (abs(ref_col_sum.item()) + 1e-8) < 1e-4

    # Also compare element-wise against fp64 on the flat patches (pre col2im)
    dp_ma_tri = (
        delta_mz.to(DEVICE).permute(0, 2, 3, 1).reshape(-1, tri.C_out).float()
        @ tri.mw.float().T
    )
    torch.testing.assert_close(
        dp_ma_tri.cpu(),
        ref_patches_d_ma.float(),
        atol=MEAN_ATOL,
        rtol=0,
    )


def test_conv2d_backward_delta_mw():
    """delta_mw = Sw * (patches_ma^T @ delta_mz_flat) matches fp64 reference."""
    tri, ma, delta_mz, delta_Sz = _setup_conv_backward()
    tri.backward(delta_mz.to(DEVICE), delta_Sz.to(DEVICE))

    patches = tri.patches_ma.cpu().double()
    dmz_flat = delta_mz.permute(0, 2, 3, 1).reshape(-1, tri.C_out).double()
    ref = (tri.Sw.cpu().double() * (patches.T @ dmz_flat)).float()

    torch.testing.assert_close(tri.delta_mw.cpu(), ref, atol=UPDATE_ATOL, rtol=0)


def test_conv2d_backward_delta_Sw():
    """delta_Sw = Sw² * ((patches_ma²)^T @ delta_Sz_flat) matches fp64 reference."""
    tri, ma, delta_mz, delta_Sz = _setup_conv_backward()
    tri.backward(delta_mz.to(DEVICE), delta_Sz.to(DEVICE))

    patches = tri.patches_ma.cpu().double()
    dSz_flat = delta_Sz.permute(0, 2, 3, 1).reshape(-1, tri.C_out).double()
    Sw = tri.Sw.cpu().double()
    ref = (Sw * Sw * ((patches**2).T @ dSz_flat)).float()

    torch.testing.assert_close(tri.delta_Sw.cpu(), ref, atol=UPDATE_ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 3: Full step (forward + backward + update)
# ──────────────────────────────────────────────────────────────────────────────


def test_conv2d_update_mw():
    """After one training step, updated mw matches cuTAGI."""
    torch.manual_seed(0)
    N, C_in, H, W, C_out, k = 4, 3, 8, 8, 8, 3
    sigma_v = 0.1
    mw, Sw, mb, Sb = _random_conv_params(C_in, C_out, k)
    ma = torch.randn(N, C_in, H, W)
    y = torch.randn(N, C_out, H, W)

    # triton step
    tri = _make_triton_conv(C_in, C_out, k, 1, mw, Sw, mb, Sb)
    net_tri = TSequential([tri], device=DEVICE)
    net_tri.step(ma.to(DEVICE), y.to(DEVICE), sigma_v)

    # pytagi step
    cut = _make_pytagi_conv(C_in, C_out, k, 1, H, W, mw, Sw, mb, Sb)
    updater = OutputUpdater(cut.device)
    cut(ma.numpy().flatten().astype(np.float32))
    var_y = np.full(N * C_out * H * W, sigma_v**2, dtype=np.float32)
    updater.update(
        output_states=cut.output_z_buffer,
        mu_obs=y.numpy().flatten().astype(np.float32),
        var_obs=var_y,
        delta_states=cut.input_delta_z_buffer,
    )
    cut.backward()
    cut.step()

    key = list(cut.state_dict().keys())[0]
    mw_cut, _, _, _ = _pytagi_conv_weights(cut, key)

    torch.testing.assert_close(tri.mw.cpu(), mw_cut, atol=UPDATE_ATOL, rtol=0)
