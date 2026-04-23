"""Diagnostic: compare triton-tagi vs cuTAGI after identical init.

Strategy: since pytagi's CUDA layers don't expose mw/Sw via state_dict after the
device move, we test init parity *indirectly* — by running forward passes and
checking that outputs match to fp32 noise. If init transfer is correct AND every
layer is algebraically equivalent, then:

  * First forward pass should match to ~1e-6 relative precision.
  * After N training steps, divergence should grow only through accumulation
    non-associativity — modest and symmetric (neither side consistently wins).

If we see output drift that is *systematic* (one side consistently higher),
that's a real formula/convention bug, not fp32 noise.

Run:
    python tests/validation/_diag_remax_parity.py
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch

import pytagi
from pytagi.nn import AvgPool2d as PAvgPool2d
from pytagi.nn import BatchNorm2d as PBatchNorm2d
from pytagi.nn import Conv2d as PConv2d
from pytagi.nn import Linear as PLinear
from pytagi.nn import MixtureReLU, OutputUpdater, Remax
from pytagi.nn import Sequential as PSequential

from triton_tagi.layers.avgpool2d import AvgPool2D as TAvgPool2D
from triton_tagi.layers.batchnorm2d import BatchNorm2D as TBatchNorm2D
from triton_tagi.layers.conv2d import Conv2D as TConv2D
from triton_tagi.layers.flatten import Flatten as TFlatten
from triton_tagi.layers.linear import Linear as TLinear
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.layers.remax import Remax as TRemax
from triton_tagi.network import Sequential as TSequential

DEVICE = "cuda"
IN_C, H, W = 3, 32, 32
OUT_F = 10
BATCH = 128
SIGMA_V = 0.05


def _he_conv(C_in, C_out, k):
    fan_in = C_in * k * k
    scale = math.sqrt(1.0 / fan_in)
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


def _bn_init(C):
    scale = 2.0 / (C + C)
    return (torch.ones(C), torch.full((C,), scale), torch.zeros(C), torch.full((C,), scale))


def _flat_conv(mw, Sw, mb, Sb):
    return (
        mw.T.cpu().numpy().flatten().tolist(),
        Sw.T.cpu().numpy().flatten().tolist(),
        mb.squeeze().cpu().numpy().tolist(),
        Sb.squeeze().cpu().numpy().tolist(),
    )


def _flat_lin(mw, Sw, mb, Sb):
    return (
        mw.T.cpu().numpy().flatten().tolist(),
        Sw.T.cpu().numpy().flatten().tolist(),
        mb.squeeze().cpu().numpy().tolist(),
        Sb.squeeze().cpu().numpy().tolist(),
    )


def _flat_bn(mg, Sg, mb, Sb):
    return (
        mg.cpu().numpy().tolist(),
        Sg.cpu().numpy().tolist(),
        mb.cpu().numpy().tolist(),
        Sb.cpu().numpy().tolist(),
    )


def _build_triton(conv_p, bn_p, lin_p):
    def _load_conv(layer, p):
        mw, Sw, mb, Sb = p
        layer.mw, layer.Sw = mw.to(DEVICE), Sw.to(DEVICE)
        layer.mb, layer.Sb = mb.to(DEVICE), Sb.to(DEVICE)
        return layer

    def _load_bn(C, p):
        mg, Sg, mb, Sb = p
        layer = TBatchNorm2D(C, device=DEVICE, preserve_var=False)
        layer.mw, layer.Sw = mg.to(DEVICE), Sg.to(DEVICE)
        layer.mb, layer.Sb = mb.to(DEVICE), Sb.to(DEVICE)
        return layer

    def _load_linear(layer, p):
        mw, Sw, mb, Sb = p
        layer.mw, layer.Sw = mw.to(DEVICE), Sw.to(DEVICE)
        layer.mb, layer.Sb = mb.to(DEVICE), Sb.to(DEVICE)
        return layer

    c0 = _load_conv(TConv2D(IN_C, 32, 5, padding=2, device=DEVICE), conv_p[0])
    c1 = _load_conv(TConv2D(32, 64, 5, padding=2, device=DEVICE), conv_p[1])
    c2 = _load_conv(TConv2D(64, 64, 5, padding=2, device=DEVICE), conv_p[2])
    bn0, bn1, bn2 = (_load_bn(C, p) for C, p in zip((32, 64, 64), bn_p))
    l0 = _load_linear(TLinear(1024, 256, device=DEVICE), lin_p[0])
    l1 = _load_linear(TLinear(256, 10, device=DEVICE), lin_p[1])
    return TSequential(
        [
            c0, TReLU(), bn0, TAvgPool2D(2),
            c1, TReLU(), bn1, TAvgPool2D(2),
            c2, TReLU(), bn2, TAvgPool2D(2),
            TFlatten(), l0, TReLU(), l1, TRemax(),
        ],
        device=DEVICE,
    )


def _build_pytagi(conv_p, bn_p, lin_p):
    net = PSequential(
        PConv2d(IN_C, 32, 5, padding=2, in_width=W, in_height=H),
        MixtureReLU(), PBatchNorm2d(32), PAvgPool2d(2, 2),
        PConv2d(32, 64, 5, padding=2),
        MixtureReLU(), PBatchNorm2d(64), PAvgPool2d(2, 2),
        PConv2d(64, 64, 5, padding=2),
        MixtureReLU(), PBatchNorm2d(64), PAvgPool2d(2, 2),
        PLinear(1024, 256), MixtureReLU(),
        PLinear(256, 10), Remax(),
    )
    net.preinit_layer()

    def _by_idx(prefix):
        return sorted(
            (k for k in net.state_dict() if k.startswith(prefix)),
            key=lambda k: int(k.split(".")[-1]),
        )

    conv_keys = _by_idx("Conv2d")
    bn_keys = _by_idx("BatchNorm2d")
    lin_keys = _by_idx("Linear")
    state = {}
    for k, p in zip(conv_keys, conv_p):
        state[k] = _flat_conv(*p)
    for k, p in zip(bn_keys, bn_p):
        state[k] = _flat_bn(*p)
    for k, p in zip(lin_keys, lin_p):
        state[k] = _flat_lin(*p)
    net.load_state_dict(state)
    net.to_device("cuda")
    return net


def _cmp(label, a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-30
    rel = diff / denom
    bias = (a - b).mean()
    print(
        f"  {label:<30s}  n={a.size:>6d}  "
        f"|Δ|max={diff.max():.3e}  rel_max={rel.max():.3e}  "
        f"bias(a-b)={bias:+.3e}  |a|mean={np.abs(a).mean():.3e}"
    )


def main():
    torch.manual_seed(0)
    pytagi.manual_seed(0)
    conv_p = [_he_conv(IN_C, 32, 5), _he_conv(32, 64, 5), _he_conv(64, 64, 5)]
    bn_p = [_bn_init(32), _bn_init(64), _bn_init(64)]
    lin_p = [_he_linear(1024, 256), _he_linear(256, 10)]

    net_tri = _build_triton(conv_p, bn_p, lin_p)
    net_cut = _build_pytagi(conv_p, bn_p, lin_p)
    updater = OutputUpdater(net_cut.device)

    torch.manual_seed(1)
    xb = torch.randn(BATCH, IN_C, H, W)
    yb = torch.zeros(BATCH, OUT_F)
    yb[torch.arange(BATCH), torch.randint(0, OUT_F, (BATCH,))] = 1.0

    # ── First forward pass (TRAIN mode — pytagi has no eval() switch) ──
    print("\n=== (1) First forward pass — TRAIN mode, no training yet ===")
    x_tri = xb.to(DEVICE)
    mu_tri, var_tri = net_tri.forward(x_tri)
    mu_tri = mu_tri.detach().cpu().numpy()
    var_tri = var_tri.detach().cpu().numpy()

    x_np = xb.numpy().reshape(-1).astype(np.float32)
    mu_flat, var_flat = net_cut(x_np)
    mu_cut = np.array(mu_flat[: BATCH * OUT_F]).reshape(BATCH, OUT_F)
    var_cut = np.array(var_flat[: BATCH * OUT_F]).reshape(BATCH, OUT_F)

    _cmp("output.mu  (eval, t=0)", mu_tri, mu_cut)
    _cmp("output.var (eval, t=0)", var_tri, var_cut)
    agree = (mu_tri.argmax(1) == mu_cut.argmax(1)).mean()
    print(f"  argmax agreement: {agree * 100:.2f}%")

    # ── Train for a few steps and re-check ──
    y_np = yb.numpy().reshape(-1).astype(np.float32)
    var_yb = np.full(BATCH * OUT_F, SIGMA_V ** 2, dtype=np.float32)

    for step in (1, 3, 10, 30):
        # Step both nets the same number of times, same input/target
        net_tri.step(x_tri, yb.to(DEVICE), SIGMA_V)
        net_cut(x_np)
        updater.update(
            output_states=net_cut.output_z_buffer,
            mu_obs=y_np, var_obs=var_yb,
            delta_states=net_cut.input_delta_z_buffer,
        )
        net_cut.backward()
        net_cut.step()

        # train-mode forward on a *fresh* input to isolate weight drift
        torch.manual_seed(42)
        x_eval = torch.randn(BATCH, IN_C, H, W)
        m_t, v_t = net_tri.forward(x_eval.to(DEVICE))
        m_t = m_t.detach().cpu().numpy()
        v_t = v_t.detach().cpu().numpy()

        mf, vf = net_cut(x_eval.numpy().reshape(-1).astype(np.float32))
        m_c = np.array(mf[: BATCH * OUT_F]).reshape(BATCH, OUT_F)
        v_c = np.array(vf[: BATCH * OUT_F]).reshape(BATCH, OUT_F)

        print(f"\n--- after {step} training step(s) ---")
        _cmp(f"output.mu  (eval, t={step})", m_t, m_c)
        _cmp(f"output.var (eval, t={step})", v_t, v_c)
        agree = (m_t.argmax(1) == m_c.argmax(1)).mean()
        print(f"  argmax agreement: {agree * 100:.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main() or 0)
