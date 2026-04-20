"""3-epoch ResNet-18 comparison: triton-tagi vs cuTAGI on CIFAR-10.

Uses HRC softmax output head (same as test_cifar10_resnet18.py).  Independent
random inits (seed 1 vs 2) since cuTAGI's CUDA Conv has a known load-order
bug that makes weight-matched init impossible for GPU training.

Reports train loss + test accuracy after each epoch.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
from torchvision import datasets, transforms

import pytagi
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.nn import AvgPool2d as PAvgPool2d
from pytagi.nn import BatchNorm2d as PBN
from pytagi.nn import Conv2d as PConv2d
from pytagi.nn import LayerBlock
from pytagi.nn import Linear as PLinear
from pytagi.nn import MixtureReLU as PMixReLU
from pytagi.nn import OutputUpdater, ResNetBlock
from pytagi.nn import Sequential as PSequential

from triton_tagi import (
    AvgPool2D,
    BatchNorm2D,
    Conv2D,
    Flatten,
    Linear,
    ReLU,
    ResBlock,
    Sequential,
    class_to_obs,
    get_predicted_labels,
)

DEVICE = "cuda"
BATCH = 128
BATCH_EVAL = 32
N_EPOCHS = 20
SIGMA_V = 0.1
GAIN_W = 0.1
GAIN_B = 0.1
N_CLASSES = 10
HRC_LEN = 11

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
    return x_train, y_train, x_test, y_test


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
        PLinear(512, HRC_LEN, gain_weight=GAIN_W, gain_bias=GAIN_B),
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
            Linear(512, HRC_LEN, **kw),
        ],
        device=DEVICE,
    )


def _eval_triton(net, x_test, y_test, tri_hrc):
    net.eval()
    x = x_test.to(DEVICE); correct = 0
    with torch.no_grad():
        for i in range(0, len(x), BATCH_EVAL):
            mu, Sa = net.forward(x[i : i + BATCH_EVAL])
            preds = get_predicted_labels(mu, Sa, tri_hrc)
            correct += (preds.cpu() == y_test[i : i + BATCH_EVAL]).sum().item()
    net.train()
    return correct / len(y_test)


def _eval_pytagi(net, x_test, y_test, metric):
    x_np = x_test.numpy(); correct = 0
    for i in range(0, len(x_np), BATCH_EVAL):
        xb_np = x_np[i : i + BATCH_EVAL]
        nb = len(xb_np)
        ma_flat, Sa_flat = net(xb_np.reshape(-1).astype(np.float32))
        preds = metric.get_predicted_labels(np.array(ma_flat), np.array(Sa_flat))
        correct += (torch.tensor(preds, dtype=torch.long) == y_test[i : i + nb]).sum().item()
    return correct / len(y_test)


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    x_train, y_train, x_test, y_test = _load_cifar10()
    tri_hrc = class_to_obs(N_CLASSES)
    utils = Utils()
    metric = HRCSoftmaxMetric(num_classes=N_CLASSES)

    # ── triton-tagi ──
    print(f"\n=== triton-tagi ResNet-18 ===")
    torch.manual_seed(1); pytagi.manual_seed(1)
    net_tri = _build_triton()
    print(f"  params: {net_tri.num_parameters():,}")
    for epoch in range(N_EPOCHS):
        t0 = time.time()
        perm = torch.randperm(len(x_train))
        x_s, y_s = x_train[perm], y_train[perm]
        net_tri.train()
        for i in range(0, len(x_s), BATCH):
            net_tri.step_hrc(
                x_s[i : i + BATCH].to(DEVICE), y_s[i : i + BATCH].to(DEVICE),
                tri_hrc, SIGMA_V,
            )
        acc = _eval_triton(net_tri, x_test, y_test, tri_hrc)
        print(f"  epoch {epoch + 1}/{N_EPOCHS}  |  test acc = {acc * 100:.2f}%  |  {time.time() - t0:.1f}s")
    acc_tri = acc

    del net_tri
    torch.cuda.empty_cache()

    # ── cuTAGI ──
    print(f"\n=== cuTAGI ResNet-18 ===")
    torch.manual_seed(2); pytagi.manual_seed(2)
    net_cut = _build_pytagi()
    updater = OutputUpdater(net_cut.device)
    for epoch in range(N_EPOCHS):
        t0 = time.time()
        perm = torch.randperm(len(x_train))
        x_np = x_train[perm].numpy()
        y_np = y_train[perm].numpy().astype(np.int32)
        for i in range(0, len(x_np), BATCH):
            xb_np = x_np[i : i + BATCH]
            lb_np = y_np[i : i + BATCH]
            nb = len(lb_np)
            obs_np, obs_idx_np, _ = utils.label_to_obs(lb_np, N_CLASSES)
            var_yb = np.full(nb * tri_hrc.n_obs, SIGMA_V ** 2, dtype=np.float32)
            net_cut(xb_np.reshape(-1).astype(np.float32))
            updater.update_using_indices(
                output_states=net_cut.output_z_buffer,
                mu_obs=obs_np.astype(np.float32),
                var_obs=var_yb,
                selected_idx=obs_idx_np.astype(np.int32),
                delta_states=net_cut.input_delta_z_buffer,
            )
            net_cut.backward()
            net_cut.step()
        acc = _eval_pytagi(net_cut, x_test, y_test, metric)
        print(f"  epoch {epoch + 1}/{N_EPOCHS}  |  test acc = {acc * 100:.2f}%  |  {time.time() - t0:.1f}s")
    acc_cut = acc

    print(f"\n=== Summary (3 epochs) ===")
    print(f"  triton-tagi: {acc_tri * 100:.2f}%")
    print(f"  cuTAGI:      {acc_cut * 100:.2f}%")
    print(f"  Δ:           {abs(acc_tri - acc_cut) * 100:.3f} pp")


if __name__ == "__main__":
    sys.exit(main() or 0)
