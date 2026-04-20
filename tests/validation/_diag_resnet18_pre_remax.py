"""Pre-Remax (μ, S) distribution comparison on ResNet-18 — triton vs cuTAGI.

Build both networks with identical weights (cuTAGI → triton sync) and read
the output of the head Linear(512, 10) — i.e. the tensor that Remax would
consume — in TRAIN mode (batch stats, not running stats).

If the S distributions match, Remax is the bug. If they diverge, the bug
is upstream in ResBlock / BN / Conv under train-mode batch statistics.
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
from pytagi.nn import ResNetBlock
from pytagi.nn import Sequential as PSequential

from pytagi.nn import Remax as PRemax

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


def _build_pytagi_with_remax():
    """Build pytagi ResNet-18 with Remax head (for jcb readout)."""
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


def _build_triton_with_remax():
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


def _load_conv(layer, sd, key, has_bias):
    mw_flat = np.array(sd[key][0]); Sw_flat = np.array(sd[key][1])
    K, C_out = layer.mw.shape
    layer.mw = torch.tensor(mw_flat.reshape(C_out, K).T, dtype=torch.float32, device=DEVICE)
    layer.Sw = torch.tensor(Sw_flat.reshape(C_out, K).T, dtype=torch.float32, device=DEVICE)
    if has_bias:
        mb = np.array(sd[key][2]); Sb = np.array(sd[key][3])
        layer.mb = torch.tensor(mb.reshape(1, C_out), dtype=torch.float32, device=DEVICE)
        layer.Sb = torch.tensor(Sb.reshape(1, C_out), dtype=torch.float32, device=DEVICE)
    else:
        layer.mb = torch.zeros_like(layer.mb)
        layer.Sb = torch.zeros_like(layer.Sb)


def _load_bn(layer, sd, key):
    g = np.array(sd[key][0]); Sg = np.array(sd[key][1])
    b = np.array(sd[key][2]); Sb = np.array(sd[key][3])
    C = len(g)
    layer.mw = torch.tensor(g.reshape(1, C), dtype=torch.float32, device=DEVICE)
    layer.Sw = torch.tensor(Sg.reshape(1, C), dtype=torch.float32, device=DEVICE)
    layer.mb = torch.tensor(b.reshape(1, C), dtype=torch.float32, device=DEVICE)
    layer.Sb = torch.tensor(Sb.reshape(1, C), dtype=torch.float32, device=DEVICE)


def _load_linear(layer, sd, key):
    mw = np.array(sd[key][0]); Sw = np.array(sd[key][1])
    mb = np.array(sd[key][2]); Sb = np.array(sd[key][3])
    C_in, C_out = layer.mw.shape
    layer.mw = torch.tensor(mw.reshape(C_out, C_in).T, dtype=torch.float32, device=DEVICE)
    layer.Sw = torch.tensor(Sw.reshape(C_out, C_in).T, dtype=torch.float32, device=DEVICE)
    layer.mb = torch.tensor(mb.reshape(1, C_out), dtype=torch.float32, device=DEVICE)
    layer.Sb = torch.tensor(Sb.reshape(1, C_out), dtype=torch.float32, device=DEVICE)


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


def _stats(label, mu, S):
    mu_np = np.asarray(mu).ravel().astype(np.float64)
    S_np = np.asarray(S).ravel().astype(np.float64)
    print(f"\n  === {label} ===")
    print(f"  μ: min={mu_np.min():+.4f}  max={mu_np.max():+.4f}  "
          f"mean={mu_np.mean():+.4f}  std={mu_np.std():.4f}  |mean|={np.abs(mu_np).mean():.4f}")
    print(f"  S: min={S_np.min():.3e}  max={S_np.max():.3e}  "
          f"mean={S_np.mean():.3e}  std={S_np.std():.3e}")
    print(f"  S percentiles: p50={np.percentile(S_np, 50):.3e}  "
          f"p90={np.percentile(S_np, 90):.3e}  p99={np.percentile(S_np, 99):.3e}")


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(0); pytagi.manual_seed(0)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = datasets.CIFAR10("data", train=True, download=False, transform=tf)
    x_batch = torch.stack([train_ds[i][0] for i in range(BATCH)])
    x_np = x_batch.numpy().reshape(-1).astype(np.float32)

    net_cut = _build_pytagi_with_remax()
    net_tri = _build_triton_with_remax()

    # cuTAGI forward (TRAIN mode) — get post-Remax mu/var AND jcb
    net_cut.train()
    mu_flat, var_flat = net_cut(x_np)
    mu_a_cut = np.array(mu_flat[: BATCH * N_CLASSES]).reshape(BATCH, N_CLASSES)
    S_a_cut = np.array(var_flat[: BATCH * N_CLASSES]).reshape(BATCH, N_CLASSES)
    # output_z_buffer.jcb = cov(A, Z) / var(Z) populated by Remax.forward
    jcb_cut = np.array(net_cut.output_z_buffer.jcb[: BATCH * N_CLASSES]).reshape(BATCH, N_CLASSES)

    # Sync cuTAGI → triton (this invalidates net_cut CUDA)
    _sync(net_tri, net_cut)

    # triton forward (TRAIN mode)
    net_tri.train()
    mu_a_tri, S_a_tri = net_tri.forward(x_batch.to(DEVICE))
    mu_a_tri = mu_a_tri.detach().cpu().numpy()
    S_a_tri = S_a_tri.detach().cpu().numpy()
    # Remax.J is stored on the triton Remax layer after forward
    J_tri = net_tri.layers[14].J.detach().cpu().numpy()

    # ── Post-Remax stats ──
    print("\n### POST-Remax tensor stats (probabilities) ###")
    _stats("cuTAGI μ_A", mu_a_cut, S_a_cut)
    _stats("triton μ_A", mu_a_tri, S_a_tri)
    diff_mu_a = np.abs(mu_a_tri - mu_a_cut)
    diff_S_a = np.abs(S_a_tri - S_a_cut)
    rel_S_a = diff_S_a / (np.maximum(np.abs(S_a_tri), np.abs(S_a_cut)) + 1e-30)
    print(f"\n  === post-Remax diff ===")
    print(f"  |Δμ_A|  : max={diff_mu_a.max():.3e}  mean={diff_mu_a.mean():.3e}")
    print(f"  |ΔS_A|  : max={diff_S_a.max():.3e}   mean={diff_S_a.mean():.3e}")
    print(f"  rel ΔS_A: max={rel_S_a.max():.3e}   mean={rel_S_a.mean():.3e}")

    # ── Jacobian comparison (Remax's J = cov(A,Z)/Var(Z)) ──
    print("\n### Remax Jacobian J = cov(A, Z) / Var(Z) ###")
    print(f"\n  === cuTAGI J ===")
    print(f"  min={jcb_cut.min():+.4e}  max={jcb_cut.max():+.4e}  "
          f"mean={jcb_cut.mean():+.4e}  std={jcb_cut.std():.4e}")
    print(f"  |J|mean={np.abs(jcb_cut).mean():.4e}")
    print(f"  frac. negative: {(jcb_cut < 0).mean():.3f}")
    print(f"\n  === triton J ===")
    print(f"  min={J_tri.min():+.4e}  max={J_tri.max():+.4e}  "
          f"mean={J_tri.mean():+.4e}  std={J_tri.std():.4e}")
    print(f"  |J|mean={np.abs(J_tri).mean():.4e}")
    print(f"  frac. negative: {(J_tri < 0).mean():.3f}")

    diff_J = np.abs(J_tri - jcb_cut)
    rel_J = diff_J / (np.maximum(np.abs(J_tri), np.abs(jcb_cut)) + 1e-30)
    print(f"\n  === J diff (triton - cuTAGI) ===")
    print(f"  |ΔJ|    : max={diff_J.max():.3e}   mean={diff_J.mean():.3e}")
    print(f"  rel ΔJ  : max={rel_J.max():.3e}   mean={rel_J.mean():.3e}")

    # Sign-flip diagnostic
    sign_same = np.sign(J_tri) == np.sign(jcb_cut)
    sign_diff_frac = 1.0 - sign_same.mean()
    print(f"  fraction of J entries with opposite sign: {sign_diff_frac:.3f}")

    # Ratio
    ratio = J_tri / np.where(np.abs(jcb_cut) < 1e-20, 1.0, jcb_cut)
    print(f"  ratio triton/cuTAGI (where |cuTAGI J|>1e-20): "
          f"mean={ratio.mean():+.3f}  median={np.median(ratio):+.3f}  std={ratio.std():.3f}")

    # Side-by-side row 0
    print(f"\n  row 0 side-by-side:")
    print(f"    cuTAGI J: {jcb_cut[0]}")
    print(f"    triton J: {J_tri[0]}")


if __name__ == "__main__":
    sys.exit(main() or 0)
