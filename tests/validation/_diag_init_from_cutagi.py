"""Init-from-cuTAGI parity test.

Instead of drawing weights with torch.randn and copying to both libs, let
pytagi initialize its own weights on CPU, read them back, push them into
triton, then train both with identical mini-batch order and compare.

This eliminates any doubt about whether _he_init / the state_dict format
is semantically correct — whatever layout pytagi produced, we feed the same
bits back into both libraries.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torchvision import datasets, transforms

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
N_EPOCHS = 10

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def _load_cifar10():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = datasets.CIFAR10("data", train=True, download=False, transform=tf)
    test_ds = datasets.CIFAR10("data", train=False, download=False, transform=tf)
    x_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    y_train = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
    x_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
    y_train_oh = torch.zeros(len(y_train), OUT_F)
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 1.0)
    return x_train, y_train_oh, y_train, x_test, y_test


def _build_pytagi_cpu():
    """Build pytagi on CPU and trigger its own init."""
    net = PSequential(
        PConv2d(IN_C, 32, 5, padding=2, in_width=W, in_height=H),
        MixtureReLU(), PBatchNorm2d(32), PAvgPool2d(2, 2),
        PConv2d(32, 64, 5, padding=2),
        MixtureReLU(), PBatchNorm2d(64), PAvgPool2d(2, 2),
        PConv2d(64, 64, 5, padding=2),
        MixtureReLU(), PBatchNorm2d(64), PAvgPool2d(2, 2),
        PLinear(1024, 256), MixtureReLU(),
        PLinear(256, 10),
        Remax(),
    )
    net.preinit_layer()   # initializes weights using pytagi's own RNG
    return net


def _extract_params(net_cpu):
    """Read the freshly-initialized weights out of pytagi's state_dict."""
    sd = net_cpu.state_dict()
    def by_idx(prefix):
        return sorted(
            (k for k in sd if k.startswith(prefix)),
            key=lambda k: int(k.split(".")[-1]),
        )
    out = {
        "conv": [sd[k] for k in by_idx("Conv2d")],
        "bn": [sd[k] for k in by_idx("BatchNorm2d")],
        "lin": [sd[k] for k in by_idx("Linear")],
    }
    return out


def _build_triton_from(params):
    """Build triton with the exact values pytagi produced.

    pytagi layout for Conv/Linear: (C_out, K) row-major.  triton layout is
    (K, C_out).  Reshape + transpose to convert.
    """
    def _conv(C_in, C_out, k, p, padding, in_width=None, in_height=None):
        mw_flat, Sw_flat, mb_flat, Sb_flat = p
        Kdim = C_in * k * k
        mw = torch.tensor(mw_flat, dtype=torch.float32).reshape(C_out, Kdim).T.contiguous()
        Sw = torch.tensor(Sw_flat, dtype=torch.float32).reshape(C_out, Kdim).T.contiguous()
        mb = torch.tensor(mb_flat, dtype=torch.float32).reshape(1, C_out).contiguous()
        Sb = torch.tensor(Sb_flat, dtype=torch.float32).reshape(1, C_out).contiguous()
        layer = TConv2D(C_in, C_out, k, padding=padding, device=DEVICE)
        layer.mw, layer.Sw = mw.to(DEVICE), Sw.to(DEVICE)
        layer.mb, layer.Sb = mb.to(DEVICE), Sb.to(DEVICE)
        return layer

    def _lin(in_f, out_f, p):
        mw_flat, Sw_flat, mb_flat, Sb_flat = p
        mw = torch.tensor(mw_flat, dtype=torch.float32).reshape(out_f, in_f).T.contiguous()
        Sw = torch.tensor(Sw_flat, dtype=torch.float32).reshape(out_f, in_f).T.contiguous()
        mb = torch.tensor(mb_flat, dtype=torch.float32).reshape(1, out_f).contiguous()
        Sb = torch.tensor(Sb_flat, dtype=torch.float32).reshape(1, out_f).contiguous()
        layer = TLinear(in_f, out_f, device=DEVICE)
        layer.mw, layer.Sw = mw.to(DEVICE), Sw.to(DEVICE)
        layer.mb, layer.Sb = mb.to(DEVICE), Sb.to(DEVICE)
        return layer

    def _bn(C, p):
        mg_flat, Sg_flat, mb_flat, Sb_flat = p
        mw = torch.tensor(mg_flat, dtype=torch.float32)
        Sw = torch.tensor(Sg_flat, dtype=torch.float32)
        mb = torch.tensor(mb_flat, dtype=torch.float32)
        Sb = torch.tensor(Sb_flat, dtype=torch.float32)
        layer = TBatchNorm2D(C, device=DEVICE, preserve_var=False)
        layer.mw, layer.Sw = mw.to(DEVICE), Sw.to(DEVICE)
        layer.mb, layer.Sb = mb.to(DEVICE), Sb.to(DEVICE)
        return layer

    c0 = _conv(IN_C, 32, 5, params["conv"][0], padding=2)
    c1 = _conv(32, 64, 5, params["conv"][1], padding=2)
    c2 = _conv(64, 64, 5, params["conv"][2], padding=2)
    bn0 = _bn(32, params["bn"][0])
    bn1 = _bn(64, params["bn"][1])
    bn2 = _bn(64, params["bn"][2])
    l0 = _lin(1024, 256, params["lin"][0])
    l1 = _lin(256, 10, params["lin"][1])
    return TSequential(
        [
            c0, TReLU(), bn0, TAvgPool2D(2),
            c1, TReLU(), bn1, TAvgPool2D(2),
            c2, TReLU(), bn2, TAvgPool2D(2),
            TFlatten(), l0, TReLU(), l1, TRemax(),
        ],
        device=DEVICE,
    )


def _train_triton(net, x_train, y_train_oh, perm):
    x_s = x_train[perm].to(DEVICE)
    y_s = y_train_oh[perm].to(DEVICE)
    for i in range(0, len(x_s) - (len(x_s) % BATCH), BATCH):
        net.step(x_s[i : i + BATCH], y_s[i : i + BATCH], SIGMA_V)


def _train_pytagi(net, updater, x_train, y_train_oh, perm):
    x_np = x_train[perm].numpy()
    y_np = y_train_oh[perm].numpy()
    for i in range(0, len(x_np) - (len(x_np) % BATCH), BATCH):
        xb = x_np[i : i + BATCH].reshape(-1).astype(np.float32)
        yb = y_np[i : i + BATCH].flatten().astype(np.float32)
        var_yb = np.full(BATCH * OUT_F, SIGMA_V ** 2, dtype=np.float32)
        net(xb)
        updater.update(
            output_states=net.output_z_buffer,
            mu_obs=yb, var_obs=var_yb,
            delta_states=net.input_delta_z_buffer,
        )
        net.backward()
        net.step()


def _accuracy_triton(net, x_test, y_labels):
    net.eval()
    correct = 0
    x = x_test.to(DEVICE)
    n = len(x) - (len(x) % BATCH)
    for i in range(0, n, BATCH):
        mu, _ = net.forward(x[i : i + BATCH])
        correct += (mu.argmax(dim=1).cpu() == y_labels[i : i + BATCH]).sum().item()
    net.train()
    return correct / n


def _accuracy_pytagi(net, x_test, y_labels):
    correct = 0
    x_np = x_test.numpy()
    n = len(x_np) - (len(x_np) % BATCH)
    for i in range(0, n, BATCH):
        xb = x_np[i : i + BATCH].reshape(-1).astype(np.float32)
        mu_flat, _ = net(xb)
        mu = torch.tensor(mu_flat[: BATCH * OUT_F]).reshape(BATCH, OUT_F)
        correct += (mu.argmax(dim=1) == y_labels[i : i + BATCH]).sum().item()
    return correct / n


def main():
    torch.manual_seed(0)
    pytagi.manual_seed(0)

    # 1. pytagi inits itself on CPU
    net_cut = _build_pytagi_cpu()
    params = _extract_params(net_cut)

    # Sanity: print a few values to show they're real (not zeros)
    print(f"conv[0] mw[:5] = {params['conv'][0][0][:5]}")
    print(f"bn[0]   mg[:5] = {params['bn'][0][0][:5]}")
    print(f"lin[0]  mw[:5] = {params['lin'][0][0][:5]}")

    # 2. Build triton from the same values
    net_tri = _build_triton_from(params)

    # 3. Push pytagi to CUDA
    net_cut.to_device("cuda")
    updater = OutputUpdater(net_cut.device)

    # 4. Train both for N_EPOCHS on identical mini-batch order
    x_train, y_train_oh, _, x_test, y_test = _load_cifar10()

    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(x_train))
        _train_triton(net_tri, x_train, y_train_oh, perm)
        _train_pytagi(net_cut, updater, x_train, y_train_oh, perm)
        print(f"  epoch {epoch + 1}/{N_EPOCHS} done")

    acc_tri = _accuracy_triton(net_tri, x_test, y_test)
    acc_cut = _accuracy_pytagi(net_cut, x_test, y_test)

    print(f"\n  triton (CUDA) BN-CNN + Remax:  {acc_tri * 100:.2f}%")
    print(f"  cuTAGI (CUDA) BN-CNN + Remax:  {acc_cut * 100:.2f}%")
    print(f"  Δ:                              {abs(acc_tri - acc_cut) * 100:.3f} pp")


if __name__ == "__main__":
    main()
