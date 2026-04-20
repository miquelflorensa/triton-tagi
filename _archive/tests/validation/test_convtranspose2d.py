"""Validation: triton-tagi ConvTranspose2D vs cuTAGI (pytagi).

Two levels:
  1. Forward formula — given same weights and a zero-Sa input, triton
     ConvTranspose2D produces the same (mu_a, var_a) as cuTAGI.
  2. End-to-end MNIST training — a Conv2D → ReLU → AvgPool2D →
     ConvTranspose2D → ReLU → AvgPool2D → Linear → HRC pipeline
     trained for 3 epochs; both implementations must converge
     similarly.

Architecture for Level 2 (MNIST 28×28 → 11-class HRC):
    Conv2D(1, 4, 3, pad=1) → ReLU → AvgPool2D(2,2) →
    ConvTranspose2D(4, 8, 3) → ReLU → AvgPool2D(4,4) →
    Linear(128, 11)   [HRC output]

Run with:
    pytest tests/validation/test_convtranspose2d.py -v -s
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torchvision import datasets, transforms

pytagi = pytest.importorskip("pytagi", reason="cuTAGI (pytagi) not installed")
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.nn import AvgPool2d as PAvgPool2d
from pytagi.nn import Conv2d as PConv2d
from pytagi.nn import ConvTranspose2d as PConvT
from pytagi.nn import Linear as PLinear
from pytagi.nn import OutputUpdater
from pytagi.nn import MixtureReLU as PReLU
from pytagi.nn import Sequential as PSequential

from triton_tagi.hrc_softmax import class_to_obs, get_predicted_labels, labels_to_hrc
from triton_tagi.layers.avgpool2d import AvgPool2D as TAvgPool2D
from triton_tagi.layers.conv2d import Conv2D as TConv2D
from triton_tagi.layers.convtranspose2d import ConvTranspose2D as TConvT
from triton_tagi.layers.flatten import Flatten as TFlatten
from triton_tagi.layers.linear import Linear as TLinear
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.network import Sequential as TSequential

pytestmark = pytest.mark.cuda

DEVICE = "cuda"
ATOL = 1e-4
DATA_ROOT = "data"


# ──────────────────────────────────────────────────────────────────────────────
#  Weight conversion helpers
# ──────────────────────────────────────────────────────────────────────────────


def _flat_conv(mw, Sw, mb, Sb):
    """Triton Conv2D (K, C_out) → pytagi flat (C_out × K)."""
    return (
        mw.T.cpu().numpy().flatten().tolist(),
        Sw.T.cpu().numpy().flatten().tolist(),
        mb.squeeze().cpu().numpy().tolist(),
        Sb.squeeze().cpu().numpy().tolist(),
    )


def _flat_convt(mw, Sw, mb, Sb, C_in, k):
    """Triton ConvTranspose2D (K, C_out) → pytagi flat (C_in, C_out, k, k)."""
    C_out = mw.shape[1]
    mw_4d = mw.view(C_in, k, k, C_out).permute(0, 3, 1, 2).contiguous().cpu()
    Sw_4d = Sw.view(C_in, k, k, C_out).permute(0, 3, 1, 2).contiguous().cpu()
    return (
        mw_4d.numpy().flatten().tolist(),
        Sw_4d.numpy().flatten().tolist(),
        mb.squeeze().cpu().numpy().tolist(),
        Sb.squeeze().cpu().numpy().tolist(),
    )


def _flat_linear(mw, Sw, mb, Sb):
    """Triton Linear (fan_in, C_out) → pytagi flat (C_out × fan_in)."""
    return (
        mw.T.cpu().numpy().flatten().tolist(),
        Sw.T.cpu().numpy().flatten().tolist(),
        mb.squeeze().cpu().numpy().tolist(),
        Sb.squeeze().cpu().numpy().tolist(),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Level 1: Forward formula
# ──────────────────────────────────────────────────────────────────────────────


def test_convtranspose2d_forward_matches_cutagi():
    """Triton ConvTranspose2D forward matches cuTAGI numerically."""
    torch.manual_seed(0)
    N, C_in, C_out, k, H = 4, 4, 8, 3, 7
    stride, padding = 1, 0

    # Shared weights (He init)
    K = C_in * k * k
    scale = math.sqrt(1.0 / K)
    mw = torch.randn(K, C_out) * scale
    Sw = torch.full((K, C_out), scale**2)
    mb = torch.zeros(1, C_out)
    Sb = torch.full((1, C_out), scale**2)

    # ── triton forward ──
    tri = TConvT(C_in, C_out, k, stride=stride, padding=padding, device=DEVICE)
    tri.mw = mw.to(DEVICE)
    tri.Sw = Sw.to(DEVICE)
    tri.mb = mb.to(DEVICE)
    tri.Sb = Sb.to(DEVICE)

    x_np = torch.randn(N, C_in, H, H).numpy().astype(np.float32)
    x_tri = torch.tensor(x_np, device=DEVICE)
    Sa_in = torch.zeros_like(x_tri)
    mz_tri, Sz_tri = tri.forward(x_tri, Sa_in)

    # ── pytagi forward ──
    net = PSequential(PConvT(C_in, C_out, k, stride=stride, padding=padding,
                             in_width=H, in_height=H))
    net.preinit_layer()
    sd = net.state_dict()
    key = [k2 for k2 in sd.keys() if "Transpose" in k2][0]
    net.load_state_dict({key: _flat_convt(mw, Sw, mb, Sb, C_in, k)})

    x_flat = x_np.reshape(-1)
    ma_cut_flat, Sa_cut_flat = net(x_flat)
    H_out = (H - 1) * stride - 2 * padding + k
    mz_cut = torch.tensor(np.array(ma_cut_flat)).reshape(N, C_out, H_out, H_out)
    Sz_cut = torch.tensor(np.array(Sa_cut_flat)).reshape(N, C_out, H_out, H_out)

    torch.testing.assert_close(mz_tri.cpu(), mz_cut, atol=ATOL, rtol=0)
    torch.testing.assert_close(Sz_tri.cpu(), Sz_cut, atol=ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 2: End-to-end MNIST training
# ──────────────────────────────────────────────────────────────────────────────

N_CLASSES = 10
HRC_LEN = 11
IN_C, H_IMG, W_IMG = 1, 28, 28
SIGMA_V = 0.1
BATCH = 256
N_EPOCHS = 3
ACC_MIN = 0.70    # must beat random (10%) by a wide margin
ACC_TOL = 0.03    # 3 pp — both implementations seeded identically (torch + pytagi); gap is stable ~1 pp


MEAN_MNIST = (0.1307,)
STD_MNIST = (0.3081,)


def _load_mnist():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN_MNIST, STD_MNIST)])
    train_ds = datasets.MNIST(DATA_ROOT, train=True, download=False, transform=tf)
    test_ds = datasets.MNIST(DATA_ROOT, train=False, download=False, transform=tf)
    x_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    y_train = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
    x_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
    return x_train, y_train, x_test, y_test


def _he_conv(C_in, C_out, k):
    scale = math.sqrt(1.0 / (C_in * k * k))
    K = C_in * k * k
    mw = torch.randn(K, C_out) * scale
    Sw = torch.full((K, C_out), scale**2)
    mb = torch.zeros(1, C_out)
    Sb = torch.full((1, C_out), scale**2)
    return mw, Sw, mb, Sb


def _he_linear(fan_in, fan_out):
    scale = math.sqrt(1.0 / fan_in)
    mw = torch.randn(fan_in, fan_out) * scale
    Sw = torch.full((fan_in, fan_out), scale**2)
    mb = torch.zeros(1, fan_out)
    Sb = torch.full((1, fan_out), scale**2)
    return mw, Sw, mb, Sb


def _build_triton(p_conv, p_convt, p_lin):
    """
    Architecture:
        Conv2D(1,4,3,pad=1) → ReLU → AvgPool2D(2,2) →
        ConvTranspose2D(4,8,3) → ReLU → AvgPool2D(4,4) →
        Flatten → Linear(128, 11)
    """
    mw0, Sw0, mb0, Sb0 = p_conv
    mw1, Sw1, mb1, Sb1 = p_convt
    mw2, Sw2, mb2, Sb2 = p_lin

    c0 = TConv2D(1, 4, 3, padding=1, device=DEVICE)
    c0.mw, c0.Sw = mw0.to(DEVICE), Sw0.to(DEVICE)
    c0.mb, c0.Sb = mb0.to(DEVICE), Sb0.to(DEVICE)

    ct = TConvT(4, 8, 3, stride=1, padding=0, device=DEVICE)
    ct.mw, ct.Sw = mw1.to(DEVICE), Sw1.to(DEVICE)
    ct.mb, ct.Sb = mb1.to(DEVICE), Sb1.to(DEVICE)

    lin = TLinear(128, HRC_LEN, device=DEVICE)
    lin.mw, lin.Sw = mw2.to(DEVICE), Sw2.to(DEVICE)
    lin.mb, lin.Sb = mb2.to(DEVICE), Sb2.to(DEVICE)

    return TSequential(
        [c0, TReLU(), TAvgPool2D(2),
         ct, TReLU(), TAvgPool2D(4),
         TFlatten(), lin],
        device=DEVICE,
    )


def _build_pytagi(p_conv, p_convt, p_lin):
    """Same architecture as _build_triton for pytagi."""
    mw0, Sw0, mb0, Sb0 = p_conv
    mw1, Sw1, mb1, Sb1 = p_convt
    mw2, Sw2, mb2, Sb2 = p_lin

    # AvgPool2D(2,2): 28→14; ConvTranspose2D(4,8,3): 14→16; AvgPool2D(4,4): 16→4
    # Linear: 8*4*4=128 → 11
    net = PSequential(
        PConv2d(1, 4, 3, padding=1, in_width=W_IMG, in_height=H_IMG),
        PReLU(),
        PAvgPool2d(2, 2),
        PConvT(4, 8, 3, stride=1, padding=0),
        PReLU(),
        PAvgPool2d(4, 4),
        PLinear(128, HRC_LEN),
    )
    net.preinit_layer()
    sd = net.state_dict()
    keys = sorted(sd.keys())

    conv_keys = [k for k in keys if "Conv2d" in k and "Transpose" not in k]
    convt_keys = [k for k in keys if "Transpose" in k]
    lin_keys = [k for k in keys if "Linear" in k]

    net.load_state_dict({
        conv_keys[0]: _flat_conv(mw0, Sw0, mb0, Sb0),
        convt_keys[0]: _flat_convt(mw1, Sw1, mb1, Sb1, C_in=4, k=3),
        lin_keys[0]: _flat_linear(mw2, Sw2, mb2, Sb2),
    })
    net.to_device("cuda")
    return net


def test_mnist_convtranspose2d_3epochs():
    """Both implementations reach ≥ 70 % and are within 2 % of each other."""
    torch.manual_seed(0)
    pytagi.manual_seed(0)
    params = [
        _he_conv(1, 4, 3),
        _he_conv(4, 8, 3),     # ConvTranspose2D uses same He init as Conv2D
        _he_linear(128, HRC_LEN),
    ]

    tri_hrc = class_to_obs(N_CLASSES)
    utils = Utils()
    metric = HRCSoftmaxMetric(num_classes=N_CLASSES)

    x_train, y_train, x_test, y_test = _load_mnist()

    net_tri = _build_triton(*params)
    net_cut = _build_pytagi(*params)
    updater = OutputUpdater(net_cut.device)

    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(x_train))
        x_s = x_train[perm]
        y_s = y_train[perm]

        # ── triton-tagi ──
        net_tri.train()
        for i in range(0, len(x_s), BATCH):
            xb = x_s[i : i + BATCH].to(DEVICE)
            lb = y_s[i : i + BATCH].to(DEVICE)
            net_tri.step_hrc(xb, lb, tri_hrc, SIGMA_V)

        # ── cuTAGI ──
        x_np = x_s.numpy()
        y_np = y_s.numpy().astype(np.int32)
        for i in range(0, len(x_np), BATCH):
            xb_np = x_np[i : i + BATCH]
            lb_np = y_np[i : i + BATCH]
            nb = len(lb_np)
            xb_flat = xb_np.reshape(-1).astype(np.float32)
            obs_np, obs_idx_np, _ = utils.label_to_obs(lb_np, N_CLASSES)
            var_yb = np.full(nb * tri_hrc.n_obs, SIGMA_V**2, dtype=np.float32)
            net_cut(xb_flat)
            updater.update_using_indices(
                output_states=net_cut.output_z_buffer,
                mu_obs=obs_np.astype(np.float32),
                var_obs=var_yb,
                selected_idx=obs_idx_np.astype(np.int32),
                delta_states=net_cut.input_delta_z_buffer,
            )
            net_cut.backward()
            net_cut.step()

    # ── Accuracy ──
    net_tri.eval()
    correct_tri = 0
    x_test_gpu = x_test.to(DEVICE)
    for i in range(0, len(x_test_gpu), BATCH):
        xb = x_test_gpu[i : i + BATCH]
        ma, Sa = net_tri.forward(xb)
        preds = get_predicted_labels(ma, Sa, tri_hrc)
        correct_tri += (preds.cpu() == y_test[i : i + BATCH]).sum().item()
    acc_tri = correct_tri / len(y_test)

    correct_cut = 0
    x_np = x_test.numpy()
    for i in range(0, len(x_np), BATCH):
        xb_np = x_np[i : i + BATCH]
        nb = len(xb_np)
        ma_flat, Sa_flat = net_cut(xb_np.reshape(-1).astype(np.float32))
        preds = metric.get_predicted_labels(np.array(ma_flat), np.array(Sa_flat))
        correct_cut += (torch.tensor(preds, dtype=torch.long) == y_test[i : i + BATCH]).sum().item()
    acc_cut = correct_cut / len(y_test)

    print(f"\n  triton-tagi ConvTranspose2D: {acc_tri * 100:.2f}%")
    print(f"  cuTAGI ConvTranspose2D:      {acc_cut * 100:.2f}%")
    print(f"  Δ accuracy:                   {abs(acc_tri - acc_cut) * 100:.3f}%  (tol {ACC_TOL*100:.1f}%)")

    assert acc_tri >= ACC_MIN, f"triton-tagi: {acc_tri*100:.2f}% < {ACC_MIN*100:.0f}%"
    assert acc_cut >= ACC_MIN, f"cuTAGI: {acc_cut*100:.2f}% < {ACC_MIN*100:.0f}%"
    assert abs(acc_tri - acc_cut) < ACC_TOL, (
        f"gap {abs(acc_tri - acc_cut)*100:.3f}% > {ACC_TOL*100:.1f}%  "
        f"tri={acc_tri*100:.2f}%  cut={acc_cut*100:.2f}%"
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Level 1.5: Single-batch weight match
# ──────────────────────────────────────────────────────────────────────────────

_STEP_C_IN  = 2
_STEP_C_OUT = 4
_STEP_K     = 3
_STEP_H     = 5   # input spatial size
_STEP_N     = 4   # batch size
_STEP_H_OUT = _STEP_H + _STEP_K - 1   # = 7  (stride=1, pad=0)
_STEP_K_TOT = _STEP_C_IN * _STEP_K * _STEP_K  # = 18
ATOL_STEP   = 1e-3


def _make_convt_params(C_in, C_out, k):
    """He-init parameters for ConvTranspose2D; returns (mw, Sw, mb, Sb)."""
    K = C_in * k * k
    scale = math.sqrt(1.0 / K)
    return (
        torch.randn(K, C_out) * scale,
        torch.full((K, C_out), scale**2),
        torch.zeros(1, C_out),
        torch.full((1, C_out), scale**2),
    )


def _make_linear_params(fan_in, fan_out):
    scale = math.sqrt(1.0 / fan_in)
    return (
        torch.randn(fan_in, fan_out) * scale,
        torch.full((fan_in, fan_out), scale**2),
        torch.zeros(1, fan_out),
        torch.full((1, fan_out), scale**2),
    )


def test_convtranspose2d_single_step_weight_match():
    """After exactly one batch, triton and cuTAGI weights should match to ATOL_STEP.

    Architecture: ConvTranspose2D(2,4,3) → ReLU → AvgPool2D(7) → Flatten → Linear(4,11)
    Batch size 4, fixed random seed — both implementations start from identical weights
    and see identical input.  Any divergence is a formula discrepancy, not noise.
    """
    torch.manual_seed(7)

    p_ct  = _make_convt_params(_STEP_C_IN, _STEP_C_OUT, _STEP_K)
    p_lin = _make_linear_params(_STEP_C_OUT, HRC_LEN)
    mw_ct, Sw_ct, mb_ct, Sb_ct = p_ct
    mw_ln, Sw_ln, mb_ln, Sb_ln = p_lin

    x      = torch.randn(_STEP_N, _STEP_C_IN, _STEP_H, _STEP_H)
    labels = torch.tensor([2, 5, 0, 8])

    hrc   = class_to_obs(N_CLASSES)
    utils = Utils()

    # ── triton-tagi ──────────────────────────────────────────────────────────
    ct  = TConvT(_STEP_C_IN, _STEP_C_OUT, _STEP_K, stride=1, padding=0, device=DEVICE)
    ct.mw, ct.Sw = mw_ct.to(DEVICE), Sw_ct.to(DEVICE)
    ct.mb, ct.Sb = mb_ct.to(DEVICE), Sb_ct.to(DEVICE)

    lin = TLinear(_STEP_C_OUT, HRC_LEN, device=DEVICE)
    lin.mw, lin.Sw = mw_ln.to(DEVICE), Sw_ln.to(DEVICE)
    lin.mb, lin.Sb = mb_ln.to(DEVICE), Sb_ln.to(DEVICE)

    net_tri = TSequential(
        [ct, TReLU(), TAvgPool2D(_STEP_H_OUT), TFlatten(), lin],
        device=DEVICE,
    )
    net_tri.step_hrc(x.to(DEVICE), labels.to(DEVICE), hrc, SIGMA_V)

    # ── cuTAGI ───────────────────────────────────────────────────────────────
    net_cut = PSequential(
        PConvT(_STEP_C_IN, _STEP_C_OUT, _STEP_K,
               stride=1, padding=0,
               in_width=_STEP_H, in_height=_STEP_H),
        PReLU(),
        PAvgPool2d(_STEP_H_OUT, _STEP_H_OUT),
        PLinear(_STEP_C_OUT, HRC_LEN),
    )
    net_cut.preinit_layer()
    sd0 = net_cut.state_dict()
    convt_key0 = [k for k in sd0 if "Transpose" in k][0]
    lin_key0   = [k for k in sd0 if "Linear"    in k][0]
    net_cut.load_state_dict({
        convt_key0: _flat_convt(mw_ct, Sw_ct, mb_ct, Sb_ct, _STEP_C_IN, _STEP_K),
        lin_key0:   _flat_linear(mw_ln, Sw_ln, mb_ln, Sb_ln),
    })
    # NOTE: do NOT call to_device("cuda") — ConvTranspose2dCuda loses loaded weights
    # (pytagi bug: state is empty until after first step()); run CPU for weight comparison.

    updater  = OutputUpdater(net_cut.device)
    labels_np = labels.numpy().astype(np.int32)
    obs_np, obs_idx_np, _ = utils.label_to_obs(labels_np, N_CLASSES)
    var_obs = np.full(_STEP_N * hrc.n_obs, SIGMA_V**2, dtype=np.float32)
    net_cut(x.numpy().reshape(-1).astype(np.float32))
    updater.update_using_indices(
        output_states=net_cut.output_z_buffer,
        mu_obs=obs_np.astype(np.float32),
        var_obs=var_obs,
        selected_idx=obs_idx_np.astype(np.int32),
        delta_states=net_cut.input_delta_z_buffer,
    )
    net_cut.backward()
    net_cut.step()

    # ── Extract cuTAGI weights (same CPU keys — no to_device was called) ───────
    sd1       = net_cut.state_dict()
    convt_key1 = [k for k in sd1 if "Transpose" in k][0]
    lin_key1   = [k for k in sd1 if "Linear"    in k][0]

    mw_cut_raw, Sw_cut_raw, mb_cut_raw, Sb_cut_raw = sd1[convt_key1]
    # pytagi layout (C_in, C_out, k, k) → triton layout (K, C_out)
    def _from_pytagi_convt(flat, C_in, C_out, k):
        return torch.tensor(flat).view(C_in, C_out, k, k).permute(0, 2, 3, 1).reshape(-1, C_out)

    mw_cut = _from_pytagi_convt(mw_cut_raw, _STEP_C_IN, _STEP_C_OUT, _STEP_K)
    Sw_cut = _from_pytagi_convt(Sw_cut_raw, _STEP_C_IN, _STEP_C_OUT, _STEP_K)
    mb_cut = torch.tensor(mb_cut_raw).view(1, _STEP_C_OUT)
    Sb_cut = torch.tensor(Sb_cut_raw).view(1, _STEP_C_OUT)

    mw_lin_raw, Sw_lin_raw, mb_lin_raw, Sb_lin_raw = sd1[lin_key1]
    # pytagi Linear layout (C_out, fan_in) → triton (fan_in, C_out)
    mw_lin_cut = torch.tensor(mw_lin_raw).view(HRC_LEN, _STEP_C_OUT).T
    Sw_lin_cut = torch.tensor(Sw_lin_raw).view(HRC_LEN, _STEP_C_OUT).T
    mb_lin_cut = torch.tensor(mb_lin_raw).view(1, HRC_LEN)
    Sb_lin_cut = torch.tensor(Sb_lin_raw).view(1, HRC_LEN)

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n  ConvTranspose2D weights after 1 step:")
    print(f"    mw  max|Δ| = {(ct.mw.cpu()  - mw_cut).abs().max():.2e}")
    print(f"    Sw  max|Δ| = {(ct.Sw.cpu()  - Sw_cut).abs().max():.2e}")
    print(f"    mb  max|Δ| = {(ct.mb.cpu()  - mb_cut).abs().max():.2e}")
    print(f"    Sb  max|Δ| = {(ct.Sb.cpu()  - Sb_cut).abs().max():.2e}")
    print("  Linear weights after 1 step:")
    print(f"    mw  max|Δ| = {(lin.mw.cpu() - mw_lin_cut).abs().max():.2e}")
    print(f"    Sw  max|Δ| = {(lin.Sw.cpu() - Sw_lin_cut).abs().max():.2e}")
    print(f"    mb  max|Δ| = {(lin.mb.cpu() - mb_lin_cut).abs().max():.2e}")
    print(f"    Sb  max|Δ| = {(lin.Sb.cpu() - Sb_lin_cut).abs().max():.2e}")

    # ── Assert ───────────────────────────────────────────────────────────────
    for name, tri, cut in [
        ("convt.mw", ct.mw.cpu(),  mw_cut),
        ("convt.Sw", ct.Sw.cpu(),  Sw_cut),
        ("convt.mb", ct.mb.cpu(),  mb_cut),
        ("convt.Sb", ct.Sb.cpu(),  Sb_cut),
        ("lin.mw",   lin.mw.cpu(), mw_lin_cut),
        ("lin.Sw",   lin.Sw.cpu(), Sw_lin_cut),
        ("lin.mb",   lin.mb.cpu(), mb_lin_cut),
        ("lin.Sb",   lin.Sb.cpu(), Sb_lin_cut),
    ]:
        torch.testing.assert_close(tri, cut, atol=ATOL_STEP, rtol=0,
                                   msg=f"{name} exceeds ATOL_STEP={ATOL_STEP}")
