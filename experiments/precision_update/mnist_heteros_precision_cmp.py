"""
Comparison: cap vs precision update modes, with TAGI-V heteroscedastic head.

With heteros, sigma_v is not used — the network learns its own noise variance.
This tests whether precision full becomes stable when observation noise is
inferred rather than manually set.
"""

from __future__ import annotations

import time

import argparse
import torch
from torchvision import datasets

from triton_tagi import EvenSoftplus, Linear, ReLU, Sequential


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

def load_mnist(data_dir="data", device=torch.device("cpu"), C=3.0):
    train_ds = datasets.MNIST(data_dir, train=True,  download=True)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True)

    x_train = train_ds.data.float().view(-1, 784) / 255.0
    x_test  = test_ds.data.float().view(-1, 784)  / 255.0
    mu, sig = x_train.mean(), x_train.std()
    x_train = ((x_train - mu) / sig).to(device)
    x_test  = ((x_test  - mu) / sig).to(device)

    y_tr = train_ds.targets.to(device)
    y_te = test_ds.targets.to(device)

    oh_tr = torch.full((len(y_tr), 10), -C, device=device)
    oh_tr.scatter_(1, y_tr.unsqueeze(1), C)

    return x_train, oh_tr, x_test, y_te


def evaluate(net, x_test, y_labels, batch_size=1024):
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            mu, _ = net.forward(x_test[i : i + batch_size])
            correct += (mu[:, 0::2].argmax(1) == y_labels[i : i + batch_size]).sum().item()
    net.train()
    return correct / len(x_test)


# ---------------------------------------------------------------------------
#  Build / train
# ---------------------------------------------------------------------------

def build_heteros(depth, hidden, device, gain=0.1, update_mode="cap", rho_mode="full", rho=1.0):
    layers, in_f = [], 784
    for _ in range(depth):
        layers.append(Linear(in_f, hidden, device=device, gain_w=gain, gain_b=gain))
        layers.append(ReLU())
        in_f = hidden
    layers.append(Linear(in_f, 20, device=device, gain_w=gain, gain_b=gain))
    layers.append(EvenSoftplus(half_width=10))
    return Sequential(layers, device=device, update_mode=update_mode, rho_mode=rho_mode, rho=rho)


def run_one(label, update_mode, rho_mode, rho,
            depth, hidden, gain, n_epochs, batch_size,
            x_train, y_train, x_test, y_labels, device, seed):
    torch.manual_seed(seed)
    net = build_heteros(depth, hidden, device, gain=gain,
                        update_mode=update_mode, rho_mode=rho_mode, rho=rho)
    best = 0.0
    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(x_train.size(0), device=device)
        xs, ys = x_train[perm], y_train[perm]
        for i in range(0, len(xs), batch_size):
            # sigma_v is passed but ignored by the heteros kernel
            net.step(xs[i : i + batch_size], ys[i : i + batch_size], sigma_v=0.0)
        acc = evaluate(net, x_test, y_labels)
        best = max(best, acc)
    wall = time.perf_counter() - t0
    return best, acc, wall


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(
    depth: int = 5,
    hidden: int = 256,
    gain: float = 0.1,
    n_epochs: int = 20,
    batch_size: int = 256,
    target_scale: float = 3.0,
    data_dir: str = "data",
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    dev = torch.device(device)
    print(f"Device: {device}")
    x_train, y_train, x_test, y_labels = load_mnist(data_dir, dev, C=target_scale)

    configs = [
        ("cap            ", "cap",       "full",       1.0),
        ("precision full ", "precision", "full",       1.0),
        ("precision sqrt ", "precision", "sqrt_batch", 1.0),
        ("precision avg  ", "precision", "batch_avg",  1.0),
    ]

    print()
    print(f"  depth={depth}  hidden={hidden}  gain={gain}  C=±{target_scale}  (heteros: no sigma_v)")
    print(f"  batch={batch_size}  epochs={n_epochs}")
    print()
    print(f"  {'mode':<22}  {'best':>8}  {'final':>8}  {'time':>7}")
    print("  " + "-" * 55)

    for label, um, rm, r in configs:
        best, final, wall = run_one(
            label, um, rm, r,
            depth, hidden, gain, n_epochs, batch_size,
            x_train, y_train, x_test, y_labels, dev, seed,
        )
        print(f"  {label}  {best*100:7.2f}%  {final*100:7.2f}%  {wall:6.1f}s")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth",        type=int,   default=5)
    parser.add_argument("--hidden",       type=int,   default=256)
    parser.add_argument("--gain",         type=float, default=0.1)
    parser.add_argument("--n_epochs",     type=int,   default=20)
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--target_scale", type=float, default=3.0)
    parser.add_argument("--data_dir",     type=str,   default="data")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(**vars(args))
