"""Progressive layer peel: find which layer first breaks triton↔cuTAGI parity.

Builds a sequence of networks, each one layer deeper, feeds identical input +
identical weights into both libs, and reports max abs / rel diff on the final
Linear's (mu, var) output.

The first config where diffs jump from fp32-noise (~1e-5) to macroscopic
(~1e-1) is the layer where the two libs disagree.
"""
from __future__ import annotations

import math

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
BATCH = 32


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
        f"  {label:<18s}  |Δ|max={diff.max():.3e}  rel_max={rel.max():.3e}  "
        f"|a|mean={np.abs(a).mean():.3e}  bias={((a-b).mean()):+.3e}"
    )


def build_and_compare(tag, triton_layers_fn, pytagi_layer_specs, extra_lin_fan_in):
    """triton_layers_fn: () -> list of triton layers (no final Linear)
       pytagi_layer_specs: list of (layer_constructor, param_pack_or_None, layer_prefix)
       extra_lin_fan_in: flatten size going into the final Linear readout
    """
    torch.manual_seed(0); pytagi.manual_seed(0)

    # One seed draws the readout Linear params
    lin_p = _he_linear(extra_lin_fan_in, OUT_F)
    mw, Sw, mb, Sb = lin_p

    # triton side
    triton_layers = triton_layers_fn()
    l_final = TLinear(extra_lin_fan_in, OUT_F, device=DEVICE)
    l_final.mw, l_final.Sw = mw.to(DEVICE), Sw.to(DEVICE)
    l_final.mb, l_final.Sb = mb.to(DEVICE), Sb.to(DEVICE)
    net_tri = TSequential(triton_layers + [TFlatten(), l_final], device=DEVICE)

    # pytagi side
    py_layers = []
    state_builders = []
    for spec in pytagi_layer_specs:
        ctor, param_pack, prefix = spec
        py_layers.append(ctor())
        state_builders.append((param_pack, prefix))

    py_layers.append(PLinear(extra_lin_fan_in, OUT_F))
    state_builders.append((lin_p, "Linear"))

    net_cut = PSequential(*py_layers)
    net_cut.preinit_layer()

    # Match keys by prefix+numeric index order
    prefix_lists = {}
    for k in net_cut.state_dict():
        pfx = k.split(".")[0]
        prefix_lists.setdefault(pfx, []).append(k)
    for pfx in prefix_lists:
        prefix_lists[pfx].sort(key=lambda x: int(x.split(".")[-1]))
    prefix_iter = {k: iter(v) for k, v in prefix_lists.items()}

    state = {}
    for param_pack, prefix in state_builders:
        if param_pack is None:
            continue
        k = next(prefix_iter[prefix])
        if prefix == "Conv2d":
            mw_, Sw_, mb_, Sb_ = param_pack
            state[k] = (
                mw_.T.cpu().numpy().flatten().tolist(),
                Sw_.T.cpu().numpy().flatten().tolist(),
                mb_.squeeze().cpu().numpy().tolist(),
                Sb_.squeeze().cpu().numpy().tolist(),
            )
        elif prefix == "BatchNorm2d":
            mg_, Sg_, mb_, Sb_ = param_pack
            state[k] = (mg_.cpu().numpy().tolist(), Sg_.cpu().numpy().tolist(),
                        mb_.cpu().numpy().tolist(), Sb_.cpu().numpy().tolist())
        elif prefix == "Linear":
            mw_, Sw_, mb_, Sb_ = param_pack
            state[k] = (
                mw_.T.cpu().numpy().flatten().tolist(),
                Sw_.T.cpu().numpy().flatten().tolist(),
                mb_.squeeze().cpu().numpy().tolist(),
                Sb_.squeeze().cpu().numpy().tolist(),
            )
    net_cut.load_state_dict(state)
    net_cut.to_device("cuda")

    # Forward
    torch.manual_seed(1)
    xb = torch.randn(BATCH, IN_C, H, W)
    mu_tri, var_tri = net_tri.forward(xb.to(DEVICE))
    mu_tri = mu_tri.detach().cpu().numpy()
    var_tri = var_tri.detach().cpu().numpy()

    x_np = xb.numpy().reshape(-1).astype(np.float32)
    mu_flat, var_flat = net_cut(x_np)
    mu_cut = np.array(mu_flat[: BATCH * OUT_F]).reshape(BATCH, OUT_F)
    var_cut = np.array(var_flat[: BATCH * OUT_F]).reshape(BATCH, OUT_F)

    print(f"\n--- {tag} ---")
    _cmp("mu", mu_tri, mu_cut)
    _cmp("var", var_tri, var_cut)


def main():
    # Config 1: Conv only
    conv_p = _he_conv(IN_C, 32, 5)

    def tri_conv():
        mw, Sw, mb, Sb = conv_p
        l = TConv2D(IN_C, 32, 5, padding=2, device=DEVICE)
        l.mw, l.Sw = mw.to(DEVICE), Sw.to(DEVICE)
        l.mb, l.Sb = mb.to(DEVICE), Sb.to(DEVICE)
        return [l]

    build_and_compare(
        "1: Conv only",
        tri_conv,
        [(lambda: PConv2d(IN_C, 32, 5, padding=2, in_width=W, in_height=H), conv_p, "Conv2d")],
        extra_lin_fan_in=32 * 32 * 32,
    )

    # Config 2: Conv + ReLU
    def tri_conv_relu():
        return tri_conv() + [TReLU()]

    build_and_compare(
        "2: Conv + ReLU",
        tri_conv_relu,
        [(lambda: PConv2d(IN_C, 32, 5, padding=2, in_width=W, in_height=H), conv_p, "Conv2d"),
         (MixtureReLU, None, "MixtureReLU")],
        extra_lin_fan_in=32 * 32 * 32,
    )

    # Config 3: Conv + ReLU + BN
    bn_p = _bn_init(32)

    def tri_conv_relu_bn():
        mw, Sw, mb, Sb = bn_p
        bn = TBatchNorm2D(32, device=DEVICE, preserve_var=False)
        bn.mw, bn.Sw = mw.to(DEVICE), Sw.to(DEVICE)
        bn.mb, bn.Sb = mb.to(DEVICE), Sb.to(DEVICE)
        return tri_conv_relu() + [bn]

    build_and_compare(
        "3: Conv+ReLU+BN",
        tri_conv_relu_bn,
        [(lambda: PConv2d(IN_C, 32, 5, padding=2, in_width=W, in_height=H), conv_p, "Conv2d"),
         (MixtureReLU, None, "MixtureReLU"),
         (lambda: PBatchNorm2d(32), bn_p, "BatchNorm2d")],
        extra_lin_fan_in=32 * 32 * 32,
    )

    # Config 4: Conv + ReLU + BN + AvgPool
    def tri_cfb_pool():
        return tri_conv_relu_bn() + [TAvgPool2D(2)]

    build_and_compare(
        "4: CRBN+AvgPool",
        tri_cfb_pool,
        [(lambda: PConv2d(IN_C, 32, 5, padding=2, in_width=W, in_height=H), conv_p, "Conv2d"),
         (MixtureReLU, None, "MixtureReLU"),
         (lambda: PBatchNorm2d(32), bn_p, "BatchNorm2d"),
         (lambda: PAvgPool2d(2, 2), None, "AvgPool2d")],
        extra_lin_fan_in=32 * 16 * 16,
    )


if __name__ == "__main__":
    main()
