"""Quick smoke: train MNIST FNN with RemaxChain (corrected chain-rule
projection, no CS clip) for a few epochs and report test accuracy.

Pass criterion: reaches at least 95 % at epoch 5 — same network reaches
~98 % under the original cuTAGI-parity Remax in tests/validation/.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torchvision import datasets

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from triton_tagi.layers.linear import Linear as TLinear
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.layers.remax_chain import RemaxChain
from triton_tagi.network import Sequential as TSequential


IN_F, H1, H2, OUT_F = 784, 256, 128, 10
BATCH = 512
SIGMA_V = 0.05
N_EPOCHS = 5
DEVICE = "cuda"
DATA_ROOT = "data"


def he_init(fan_in: int, fan_out: int):
    scale = math.sqrt(1.0 / fan_in)
    mw = torch.randn(fan_in, fan_out) * scale
    Sw = torch.full((fan_in, fan_out), scale ** 2)
    mb = torch.randn(1, fan_out) * scale
    Sb = torch.full((1, fan_out), scale ** 2)
    return mw, Sw, mb, Sb


def main():
    torch.manual_seed(0)
    train_ds = datasets.MNIST(DATA_ROOT, train=True, download=True)
    test_ds = datasets.MNIST(DATA_ROOT, train=False, download=True)
    x_train = train_ds.data.float().view(-1, IN_F) / 255.0
    x_test = test_ds.data.float().view(-1, IN_F) / 255.0
    mu, sigma = x_train.mean(), x_train.std()
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    y_train = torch.zeros(len(train_ds.targets), OUT_F)
    y_train.scatter_(1, train_ds.targets.unsqueeze(1), 1.0)
    y_test_labels = test_ds.targets

    params = [he_init(IN_F, H1), he_init(H1, H2), he_init(H2, OUT_F)]
    layers = []
    for (mw, Sw, mb, Sb), (fi, fo) in zip(params, [(IN_F, H1), (H1, H2), (H2, OUT_F)]):
        l = TLinear(fi, fo, device=DEVICE)
        l.mw, l.Sw, l.mb, l.Sb = mw.to(DEVICE), Sw.to(DEVICE), mb.to(DEVICE), Sb.to(DEVICE)
        layers.append(l)
    net = TSequential(
        [layers[0], TReLU(), layers[1], TReLU(), layers[2], RemaxChain()],
        device=DEVICE,
    )

    for epoch in range(1, N_EPOCHS + 1):
        perm = torch.randperm(len(x_train))
        x_s = x_train[perm].to(DEVICE)
        y_s = y_train[perm].to(DEVICE)
        for i in range(0, len(x_s), BATCH):
            net.step(x_s[i : i + BATCH], y_s[i : i + BATCH], SIGMA_V)

        net.eval()
        correct = 0
        x_te = x_test.to(DEVICE)
        for i in range(0, len(x_te), BATCH):
            mu_a, _ = net.forward(x_te[i : i + BATCH])
            correct += (mu_a.argmax(dim=1).cpu() == y_test_labels[i : i + BATCH]).sum().item()
        net.train()
        acc = correct / len(y_test_labels)
        print(f"  epoch {epoch}/{N_EPOCHS}  test_acc={acc * 100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
