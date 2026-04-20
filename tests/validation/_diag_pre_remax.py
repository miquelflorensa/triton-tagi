"""Compare triton vs cuTAGI output of the pre-Remax linear layer on CIFAR CNN+BN.

If this matches tightly, the divergence is entirely inside Remax itself.
If it diverges, the problem is upstream (Conv, BN, or the way MixtureReLU vs
TReLU shape the moments flowing into Remax).
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
from pytagi.nn import MixtureReLU
from pytagi.nn import Sequential as PSequential

from triton_tagi.layers.avgpool2d import AvgPool2D as TAvgPool2D
from triton_tagi.layers.batchnorm2d import BatchNorm2D as TBatchNorm2D
from triton_tagi.layers.conv2d import Conv2D as TConv2D
from triton_tagi.layers.flatten import Flatten as TFlatten
from triton_tagi.layers.linear import Linear as TLinear
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.network import Sequential as TSequential

DEVICE = "cuda"
IN_C, H, W = 3, 32, 32
OUT_F = 10
BATCH = 128


def _he_conv(C_in, C_out, k):
    fan_in = C_in * k * k
    s = math.sqrt(1.0 / fan_in)
    K = C_in * k * k
    return (torch.randn(K, C_out) * s, torch.full((K, C_out), s**2),
            torch.zeros(1, C_out), torch.full((1, C_out), s**2))


def _he_linear(fan_in, fan_out):
    s = math.sqrt(1.0 / fan_in)
    return (torch.randn(fan_in, fan_out) * s, torch.full((fan_in, fan_out), s**2),
            torch.zeros(1, fan_out), torch.full((1, fan_out), s**2))


def _bn_init(C):
    s = 1.0 / C
    return (torch.ones(C), torch.full((C,), s), torch.zeros(C), torch.full((C,), s))


def _cmp(label, a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-30
    rel = diff / denom
    print(
        f"  {label:<24s}  n={a.size:>6d}  "
        f"|Δ|max={diff.max():.3e}  rel_max={rel.max():.3e}  "
        f"|a|mean={np.abs(a).mean():.3e}  bias={((a-b).mean()):+.3e}"
    )


def main():
    torch.manual_seed(0); pytagi.manual_seed(0)
    conv_p = [_he_conv(IN_C, 32, 5), _he_conv(32, 64, 5), _he_conv(64, 64, 5)]
    bn_p = [_bn_init(32), _bn_init(64), _bn_init(64)]
    lin_p = [_he_linear(1024, 256), _he_linear(256, OUT_F)]

    # ── triton network WITHOUT Remax ──
    def _lc(layer, p):
        mw, Sw, mb, Sb = p
        layer.mw, layer.Sw = mw.to(DEVICE), Sw.to(DEVICE)
        layer.mb, layer.Sb = mb.to(DEVICE), Sb.to(DEVICE)
        return layer

    def _lbn(C, p):
        mg, Sg, mb, Sb = p
        l = TBatchNorm2D(C, device=DEVICE, preserve_var=False)
        l.mw, l.Sw = mg.to(DEVICE), Sg.to(DEVICE)
        l.mb, l.Sb = mb.to(DEVICE), Sb.to(DEVICE)
        return l

    c0 = _lc(TConv2D(IN_C, 32, 5, padding=2, device=DEVICE), conv_p[0])
    c1 = _lc(TConv2D(32, 64, 5, padding=2, device=DEVICE), conv_p[1])
    c2 = _lc(TConv2D(64, 64, 5, padding=2, device=DEVICE), conv_p[2])
    bn0, bn1, bn2 = (_lbn(C, p) for C, p in zip((32, 64, 64), bn_p))
    l0 = _lc(TLinear(1024, 256, device=DEVICE), lin_p[0])
    l1 = _lc(TLinear(256, OUT_F, device=DEVICE), lin_p[1])

    net_tri = TSequential(
        [
            c0, TReLU(), bn0, TAvgPool2D(2),
            c1, TReLU(), bn1, TAvgPool2D(2),
            c2, TReLU(), bn2, TAvgPool2D(2),
            TFlatten(), l0, TReLU(), l1,
        ],
        device=DEVICE,
    )

    # ── pytagi network WITHOUT Remax ──
    net_cut = PSequential(
        PConv2d(IN_C, 32, 5, padding=2, in_width=W, in_height=H),
        MixtureReLU(), PBatchNorm2d(32), PAvgPool2d(2, 2),
        PConv2d(32, 64, 5, padding=2),
        MixtureReLU(), PBatchNorm2d(64), PAvgPool2d(2, 2),
        PConv2d(64, 64, 5, padding=2),
        MixtureReLU(), PBatchNorm2d(64), PAvgPool2d(2, 2),
        PLinear(1024, 256), MixtureReLU(),
        PLinear(256, OUT_F),
    )
    net_cut.preinit_layer()

    def _by_idx(prefix):
        return sorted(
            (k for k in net_cut.state_dict() if k.startswith(prefix)),
            key=lambda k: int(k.split(".")[-1]),
        )
    state = {}
    for k, p in zip(_by_idx("Conv2d"), conv_p):
        mw, Sw, mb, Sb = p
        state[k] = (
            mw.T.cpu().numpy().flatten().tolist(),
            Sw.T.cpu().numpy().flatten().tolist(),
            mb.squeeze().cpu().numpy().tolist(),
            Sb.squeeze().cpu().numpy().tolist(),
        )
    for k, p in zip(_by_idx("BatchNorm2d"), bn_p):
        mg, Sg, mb, Sb = p
        state[k] = (mg.cpu().numpy().tolist(), Sg.cpu().numpy().tolist(),
                    mb.cpu().numpy().tolist(), Sb.cpu().numpy().tolist())
    for k, p in zip(_by_idx("Linear"), lin_p):
        mw, Sw, mb, Sb = p
        state[k] = (
            mw.T.cpu().numpy().flatten().tolist(),
            Sw.T.cpu().numpy().flatten().tolist(),
            mb.squeeze().cpu().numpy().tolist(),
            Sb.squeeze().cpu().numpy().tolist(),
        )
    net_cut.load_state_dict(state)
    net_cut.to_device("cuda")

    # ── forward both on identical input, TRAIN mode ──
    torch.manual_seed(1)
    xb = torch.randn(BATCH, IN_C, H, W)

    mu_tri, var_tri = net_tri.forward(xb.to(DEVICE))
    mu_tri = mu_tri.detach().cpu().numpy()
    var_tri = var_tri.detach().cpu().numpy()

    x_np = xb.numpy().reshape(-1).astype(np.float32)
    mu_flat, var_flat = net_cut(x_np)
    mu_cut = np.array(mu_flat[: BATCH * OUT_F]).reshape(BATCH, OUT_F)
    var_cut = np.array(var_flat[: BATCH * OUT_F]).reshape(BATCH, OUT_F)

    print("\n=== Pre-Remax (final Linear) output, BATCH=128, train mode, no steps ===")
    _cmp("pre-Remax.mu", mu_tri, mu_cut)
    _cmp("pre-Remax.var", var_tri, var_cut)
    print(f"\n  tri first row : {mu_tri[0]}")
    print(f"  cut first row : {mu_cut[0]}")


if __name__ == "__main__":
    sys.exit(main() or 0)
