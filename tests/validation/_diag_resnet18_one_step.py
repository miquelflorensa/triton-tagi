"""One-step forward+backward+step parity between triton-tagi and cuTAGI.

Sync cuTAGI → triton weights, then run ONE training step on both with the
same batch + target. Compare the RESULTING updated weights per layer.

Walking from the head (output side) to the stem:
  - First layer whose updated weights match  → backward correct through here
  - First layer whose updated weights diverge → its own backward OR the
    incoming delta from the next layer is buggy
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from torchvision import datasets, transforms

import pytagi
from pytagi.nn import AvgPool2d as PAvgPool2d
from pytagi.nn import BatchNorm2d as PBN
from pytagi.nn import Conv2d as PConv2d
from pytagi.nn import LayerBlock
from pytagi.nn import Linear as PLinear
from pytagi.nn import MixtureReLU as PMixReLU
from pytagi.nn import OutputUpdater, Remax as PRemax, ResNetBlock
from pytagi.nn import Sequential as PSequential

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
)

DEVICE = "cuda"
BATCH = 128
N_CLASSES = 10
GAIN_W = 0.1
GAIN_B = 0.1
SIGMA_V = 0.1
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def _main_block(in_c, out_c, stride=1, padding_type=1):
    return LayerBlock(
        PConv2d(in_c, out_c, 3, bias=False, stride=stride, padding=1,
                padding_type=padding_type, gain_weight=GAIN_W),
        PMixReLU(), PBN(out_c),
        PConv2d(out_c, out_c, 3, bias=False, padding=1, gain_weight=GAIN_W),
        PMixReLU(), PBN(out_c),
    )


def _build_pytagi():
    net = PSequential(
        PConv2d(3, 64, 3, bias=True, padding=1, in_width=32, in_height=32, gain_weight=GAIN_W),
        PMixReLU(), PBN(64),
        ResNetBlock(_main_block(64, 64)),
        ResNetBlock(_main_block(64, 64)),
        ResNetBlock(
            _main_block(64, 128, stride=2, padding_type=2),
            LayerBlock(PConv2d(64, 128, 2, bias=False, stride=2, gain_weight=GAIN_W),
                       PMixReLU(), PBN(128)),
        ),
        ResNetBlock(_main_block(128, 128)),
        ResNetBlock(
            _main_block(128, 256, stride=2, padding_type=2),
            LayerBlock(PConv2d(128, 256, 2, bias=False, stride=2, gain_weight=GAIN_W),
                       PMixReLU(), PBN(256)),
        ),
        ResNetBlock(_main_block(256, 256)),
        ResNetBlock(
            _main_block(256, 512, stride=2, padding_type=2),
            LayerBlock(PConv2d(256, 512, 2, bias=False, stride=2, gain_weight=GAIN_W),
                       PMixReLU(), PBN(512)),
        ),
        ResNetBlock(_main_block(512, 512)),
        PAvgPool2d(4),
        PLinear(512, N_CLASSES, gain_weight=GAIN_W, gain_bias=GAIN_B),
        PRemax(),
    )
    net.preinit_layer()
    net.to_device("cuda")
    return net


def _build_triton():
    kw = {"device": DEVICE, "gain_w": GAIN_W, "gain_b": GAIN_B}
    return Sequential(
        [
            Conv2D(3, 64, 3, stride=1, padding=1, **kw), ReLU(),
            BatchNorm2D(64, preserve_var=False, **kw),
            ResBlock(64, 64, stride=1, **kw),
            ResBlock(64, 64, stride=1, **kw),
            ResBlock(64, 128, stride=2, **kw),
            ResBlock(128, 128, stride=1, **kw),
            ResBlock(128, 256, stride=2, **kw),
            ResBlock(256, 256, stride=1, **kw),
            ResBlock(256, 512, stride=2, **kw),
            ResBlock(512, 512, stride=1, **kw),
            AvgPool2D(4), Flatten(),
            Linear(512, N_CLASSES, **kw),
            Remax(),
        ],
        device=DEVICE,
    )


def _to_tensor(a):
    return torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=DEVICE)


def _load_conv(layer, sd, key, has_bias):
    mw = np.array(sd[key][0]); Sw = np.array(sd[key][1])
    K, C_out = layer.mw.shape
    layer.mw = _to_tensor(mw.reshape(C_out, K).T)
    layer.Sw = _to_tensor(Sw.reshape(C_out, K).T)
    if has_bias:
        mb = np.array(sd[key][2]); Sb = np.array(sd[key][3])
        layer.mb = _to_tensor(mb.reshape(1, C_out))
        layer.Sb = _to_tensor(Sb.reshape(1, C_out))
    else:
        layer.mb = torch.zeros_like(layer.mb)
        layer.Sb = torch.zeros_like(layer.Sb)


def _load_bn(layer, sd, key):
    g = np.array(sd[key][0]); Sg = np.array(sd[key][1])
    b = np.array(sd[key][2]); Sb = np.array(sd[key][3])
    C = len(g)
    layer.mw = _to_tensor(g.reshape(1, C))
    layer.Sw = _to_tensor(Sg.reshape(1, C))
    layer.mb = _to_tensor(b.reshape(1, C))
    layer.Sb = _to_tensor(Sb.reshape(1, C))


def _load_linear(layer, sd, key):
    mw = np.array(sd[key][0]); Sw = np.array(sd[key][1])
    mb = np.array(sd[key][2]); Sb = np.array(sd[key][3])
    C_in, C_out = layer.mw.shape
    layer.mw = _to_tensor(mw.reshape(C_out, C_in).T)
    layer.Sw = _to_tensor(Sw.reshape(C_out, C_in).T)
    layer.mb = _to_tensor(mb.reshape(1, C_out))
    layer.Sb = _to_tensor(Sb.reshape(1, C_out))


def _sync(net_tri, net_cut):
    net_cut.params_to_host()
    sd = net_cut.state_dict()
    L = net_tri.layers
    _load_conv(L[0], sd, "Conv2dCuda.0", has_bias=True)
    _load_bn(L[2], sd, "BatchNorm2dCuda.2")
    proj = {5, 7, 9}
    for pos in range(3, 11):
        blk = L[pos]; p = str(pos)
        _load_conv(blk.conv1, sd, f"Conv2dCuda.main.{p}.0", has_bias=False)
        _load_bn(blk.bn1, sd, f"BatchNorm2dCuda.main.{p}.2")
        _load_conv(blk.conv2, sd, f"Conv2dCuda.main.{p}.3", has_bias=False)
        _load_bn(blk.bn2, sd, f"BatchNorm2dCuda.main.{p}.5")
        if pos in proj:
            _load_conv(blk.proj_conv, sd, f"Conv2dCuda.shortcut.{p}.0", has_bias=False)
            _load_bn(blk.proj_bn, sd, f"BatchNorm2dCuda.shortcut.{p}.2")
    _load_linear(L[13], sd, "LinearCuda.12")


def _cut_params(net_cut):
    """Get a snapshot of cuTAGI's params, keyed by layer position."""
    net_cut.params_to_host()
    return net_cut.state_dict()


def _tri_to_cut_layout_conv(mw_tri, K, C_out):
    """Triton (K, C_out) → cuTAGI (C_out, K) flat."""
    return mw_tri.detach().cpu().numpy().T.reshape(-1)


def _tri_to_cut_layout_linear(mw_tri, C_in, C_out):
    """Triton (C_in, C_out) → cuTAGI (C_out, C_in) flat."""
    return mw_tri.detach().cpu().numpy().T.reshape(-1)


def _cmp(tri_arr, cut_arr, label):
    a = np.asarray(tri_arr).astype(np.float64).ravel()
    b = np.asarray(cut_arr).astype(np.float64).ravel()
    d = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-30
    rel = d / denom
    return (d.max(), d.mean(), rel.max(), rel.mean(), a.size, label)


def _print_row(tup):
    dmax, dmean, rmax, rmean, n, label = tup
    print(f"  {label:<36s}  n={n:>8d}  |Δ|max={dmax:.3e}  mean={dmean:.3e}  "
          f"rel.max={rmax:.3e}  rel.mean={rmean:.3e}")


def _compare_conv(blk_or_layer_triton, sd_after, key_pref, has_bias):
    """Compare triton Conv's mw/Sw (and optionally mb/Sb) to cuTAGI's updated state."""
    rows = []
    K, C_out = blk_or_layer_triton.mw.shape
    tri_mw = _tri_to_cut_layout_conv(blk_or_layer_triton.mw, K, C_out)
    tri_Sw = _tri_to_cut_layout_conv(blk_or_layer_triton.Sw, K, C_out)
    cut_mw = np.array(sd_after[key_pref][0])
    cut_Sw = np.array(sd_after[key_pref][1])
    rows.append(_cmp(tri_mw, cut_mw, f"{key_pref}/mw"))
    rows.append(_cmp(tri_Sw, cut_Sw, f"{key_pref}/Sw"))
    if has_bias:
        tri_mb = blk_or_layer_triton.mb.detach().cpu().numpy().ravel()
        tri_Sb = blk_or_layer_triton.Sb.detach().cpu().numpy().ravel()
        cut_mb = np.array(sd_after[key_pref][2])
        cut_Sb = np.array(sd_after[key_pref][3])
        rows.append(_cmp(tri_mb, cut_mb, f"{key_pref}/mb"))
        rows.append(_cmp(tri_Sb, cut_Sb, f"{key_pref}/Sb"))
    return rows


def _compare_bn(layer_tri, sd_after, key):
    rows = []
    tri_mw = layer_tri.mw.detach().cpu().numpy().ravel()
    tri_Sw = layer_tri.Sw.detach().cpu().numpy().ravel()
    tri_mb = layer_tri.mb.detach().cpu().numpy().ravel()
    tri_Sb = layer_tri.Sb.detach().cpu().numpy().ravel()
    cut = sd_after[key]
    rows.append(_cmp(tri_mw, np.array(cut[0]), f"{key}/mγ"))
    rows.append(_cmp(tri_Sw, np.array(cut[1]), f"{key}/Sγ"))
    rows.append(_cmp(tri_mb, np.array(cut[2]), f"{key}/mβ"))
    rows.append(_cmp(tri_Sb, np.array(cut[3]), f"{key}/Sβ"))
    return rows


def _compare_linear(layer_tri, sd_after, key):
    rows = []
    C_in, C_out = layer_tri.mw.shape
    tri_mw = _tri_to_cut_layout_linear(layer_tri.mw, C_in, C_out)
    tri_Sw = _tri_to_cut_layout_linear(layer_tri.Sw, C_in, C_out)
    tri_mb = layer_tri.mb.detach().cpu().numpy().ravel()
    tri_Sb = layer_tri.Sb.detach().cpu().numpy().ravel()
    cut = sd_after[key]
    rows.append(_cmp(tri_mw, np.array(cut[0]), f"{key}/mw"))
    rows.append(_cmp(tri_Sw, np.array(cut[1]), f"{key}/Sw"))
    rows.append(_cmp(tri_mb, np.array(cut[2]), f"{key}/mb"))
    rows.append(_cmp(tri_Sb, np.array(cut[3]), f"{key}/Sb"))
    return rows


def main():
    torch.manual_seed(0); pytagi.manual_seed(0)

    # Data (one batch)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.CIFAR10("data", train=True, download=False, transform=tf)
    xb = torch.stack([ds[i][0] for i in range(BATCH)])
    yb = torch.tensor([ds[i][1] for i in range(BATCH)])
    yb_oh = torch.zeros(BATCH, N_CLASSES).scatter_(1, yb.unsqueeze(1), 1.0)

    # --- Build both networks ---
    net_cut = _build_pytagi()
    net_tri = _build_triton()

    # cuTAGI must forward first so we can read out its params for syncing
    # (params_to_host invalidates cuTAGI's CUDA state; any call after that breaks)
    # So: sync before running the step.
    net_tri.train()
    net_cut.train()

    # Run a dummy cuTAGI forward just to materialize GPU weights, then sync to triton
    xb_np_dummy = xb.numpy().reshape(-1).astype(np.float32)
    _ = net_cut(xb_np_dummy)

    # Sync cuTAGI → triton (params_to_host invalidates cuTAGI; must rebuild cuTAGI)
    _sync(net_tri, net_cut)

    # After _sync, net_cut's CUDA state is broken. Rebuild it, then sync GPU→CPU→GPU to
    # ensure we have a fresh net_cut with the SAME params that were just copied to triton.
    del net_cut
    torch.cuda.empty_cache()
    pytagi.manual_seed(0)  # reset to get identical fresh init
    net_cut = _build_pytagi()
    # Force its params onto GPU (constructor + preinit already did this).
    # Its randomly initialized weights match pytagi seed 0, which matches what
    # was just dumped to triton.

    # --- Run one full step on both ---
    # triton
    net_tri.step(xb.to(DEVICE), yb_oh.to(DEVICE), SIGMA_V)
    torch.cuda.synchronize()

    # cuTAGI
    updater = OutputUpdater(net_cut.device)
    x_np = xb.numpy().reshape(-1).astype(np.float32)
    y_np = yb_oh.numpy().flatten().astype(np.float32)
    var_y = np.full(BATCH * N_CLASSES, SIGMA_V ** 2, dtype=np.float32)
    net_cut(x_np)
    updater.update(
        output_states=net_cut.output_z_buffer,
        mu_obs=y_np, var_obs=var_y,
        delta_states=net_cut.input_delta_z_buffer,
    )
    net_cut.backward()
    net_cut.step()

    # --- Get cuTAGI's updated state dict ---
    net_cut.params_to_host()
    sd_after = net_cut.state_dict()

    # --- Compare per layer, HEAD → STEM ---
    all_rows = []

    # Head Linear (triton layer[13], cuTAGI pos 12)
    all_rows.append(("── HEAD Linear (triton[13] / cuTAGI[12]) ──", None))
    for r in _compare_linear(net_tri.layers[13], sd_after, "LinearCuda.12"):
        all_rows.append((None, r))

    # ResBlocks (triton layers 10 → 3), cuTAGI positions 10 → 3
    proj = {5, 7, 9}
    for pos in range(10, 2, -1):
        all_rows.append((f"── ResBlock @ pos {pos} ──", None))
        blk = net_tri.layers[pos]; p = str(pos)
        for r in _compare_conv(blk.conv1, sd_after, f"Conv2dCuda.main.{p}.0", has_bias=False):
            all_rows.append((None, r))
        for r in _compare_bn(blk.bn1, sd_after, f"BatchNorm2dCuda.main.{p}.2"):
            all_rows.append((None, r))
        for r in _compare_conv(blk.conv2, sd_after, f"Conv2dCuda.main.{p}.3", has_bias=False):
            all_rows.append((None, r))
        for r in _compare_bn(blk.bn2, sd_after, f"BatchNorm2dCuda.main.{p}.5"):
            all_rows.append((None, r))
        if pos in proj:
            for r in _compare_conv(blk.proj_conv, sd_after, f"Conv2dCuda.shortcut.{p}.0", has_bias=False):
                all_rows.append((None, r))
            for r in _compare_bn(blk.proj_bn, sd_after, f"BatchNorm2dCuda.shortcut.{p}.2"):
                all_rows.append((None, r))

    # Stem (triton layer[0] Conv + layer[2] BN)
    all_rows.append(("── STEM ──", None))
    for r in _compare_conv(net_tri.layers[0], sd_after, "Conv2dCuda.0", has_bias=True):
        all_rows.append((None, r))
    for r in _compare_bn(net_tri.layers[2], sd_after, "BatchNorm2dCuda.2"):
        all_rows.append((None, r))

    # Print
    print(f"\n### One-step weight-divergence per layer (triton - cuTAGI) ###")
    print(f"### Same init, same batch, same σ_v={SIGMA_V}, BATCH={BATCH} ###\n")
    for hdr, row in all_rows:
        if hdr:
            print(f"\n{hdr}")
        else:
            _print_row(row)


if __name__ == "__main__":
    sys.exit(main() or 0)
