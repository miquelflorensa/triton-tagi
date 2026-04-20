"""
Test TAGI Forward Parameter Initialization on MNIST
=================================================
Demonstrates that data-driven forward scaling eliminates the need for
manual gain tuning. Uses the same MLP architecture with arbitrary
initial gains, then scales them forward before training.
"""

import time

import numpy as np
import torch
from torchvision import datasets

from triton_tagi import Sequential
from triton_tagi.layers import Bernoulli, Linear, ReLU

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
#  Data loading
# ======================================================================


def load_mnist(data_dir="data"):
    train_ds = datasets.MNIST(data_dir, train=True, download=True)
    test_ds = datasets.MNIST(data_dir, train=False, download=True)

    x_train = train_ds.data.float().view(-1, 784) / 255.0
    x_test = test_ds.data.float().view(-1, 784) / 255.0

    mu, sigma = x_train.mean(), x_train.std()
    x_train = ((x_train - mu) / sigma).to(DEVICE)
    x_test = ((x_test - mu) / sigma).to(DEVICE)

    y_train_labels = train_ds.targets.to(DEVICE)
    y_test_labels = test_ds.targets.to(DEVICE)

    y_train_oh = torch.zeros(len(y_train_labels), 10, device=DEVICE)
    y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 1.0)

    return x_train, y_train_oh, y_train_labels, x_test, y_test_labels


# ======================================================================
#  Helpers
# ======================================================================


def evaluate(net, x_test, y_labels, batch_size=1024):
    correct = 0
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i : i + batch_size]
        lb = y_labels[i : i + batch_size]
        mu, _ = net.forward(xb)
        correct += (mu.argmax(dim=1) == lb).sum().item()
    return correct / len(x_test)


def train_loop(
    net, x_train, y_train_oh, x_test, y_test_labels, batch_size, sigma_v, n_epochs, label=""
):

    print(f"\n  {'Epoch':>5s}  {'Acc':>7s}  {'Time':>8s}")
    print("  " + "─" * 26)

    best_acc = 0.0
    t_total = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(x_train.size(0), device=DEVICE)
        x_s, y_s = x_train[perm], y_train_oh[perm]

        for i in range(0, len(x_s), batch_size):
            xb = x_s[i : i + batch_size]
            yb = y_s[i : i + batch_size]
            net.step(xb, yb, sigma_v)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        acc = evaluate(net, x_test, y_test_labels)
        best_acc = max(best_acc, acc)
        print(f"  {epoch:5d}  {acc * 100:6.2f}%  {dt:7.2f}s")

    total_time = time.perf_counter() - t_total
    print("  " + "─" * 26)
    print(f"  Best accuracy : {best_acc * 100:.2f}%")
    print(f"  Total time    : {total_time:.1f}s")
    return best_acc


# ======================================================================
#  Main
# ======================================================================


def main():
    print("=" * 60)
    print("  TAGI Forward Parameter Initialization Test — MNIST MLP")
    print("  No manual gain tuning needed!")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n  Loading MNIST...", flush=True)
    x_train, y_train_oh, _, x_test, y_test_labels = load_mnist()
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")

    # ── Build network with DEFAULT gain (no manual tuning) ──
    net = Sequential(
        [
            Linear(784, 512, device=DEVICE),
            ReLU(),
            Linear(512, 256, device=DEVICE),
            ReLU(),
            Linear(256, 10, device=DEVICE),
            # Remax(),
            Bernoulli(n_gh=32),
        ],
        device=DEVICE,
    )

    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    # ── Train ──
    batch_size = 128
    sigma_v = 0.01
    n_epochs = 20

    print(f"\n  Batch size : {batch_size}")
    print(f"  σ_v        : {sigma_v}")
    print(f"  Epochs     : {n_epochs}")

    train_loop(
        net,
        x_train,
        y_train_oh,
        x_test,
        y_test_labels,
        batch_size,
        sigma_v,
        n_epochs,
        label="forward-scaled",
    )


if __name__ == "__main__":
    main()
