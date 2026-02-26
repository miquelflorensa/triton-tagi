"""
MNIST Classification with TAGI-Triton CNN
==========================================
Architecture: Conv(1→32,5) → ReLU → Pool(2) → Conv(32→64,5) → ReLU → Pool(2)
              → Flatten → FC(3136→256) → ReLU → FC(256→10) → Remax

Demonstrates the CNN layers on MNIST digit classification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
from torchvision import datasets

from src import Sequential
from src.layers import Linear, ReLU, Conv2D, AvgPool2D, Flatten, Remax, Bernoulli
from src.init import forward_scale_weights, verify_standardization


torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
#  Data loading
# ======================================================================

def load_mnist(data_dir="data"):
    """Load MNIST as (N, 1, 28, 28) tensors on DEVICE."""
    train_ds = datasets.MNIST(data_dir, train=True,  download=True)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True)

    x_train = train_ds.data.float().view(-1, 1, 28, 28) / 255.0
    x_test  = test_ds.data.float().view(-1, 1, 28, 28)  / 255.0

    # Standardise
    mu, sigma = x_train.mean(), x_train.std()
    x_train = ((x_train - mu) / sigma).to(DEVICE)
    x_test  = ((x_test  - mu) / sigma).to(DEVICE)

    y_train_labels = train_ds.targets.to(DEVICE)
    y_test_labels  = test_ds.targets.to(DEVICE)

    # One-hot 0/1 for Remax
    y_train_oh = torch.zeros(len(y_train_labels), 10, device=DEVICE)
    y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 1.0)

    return x_train, y_train_oh, y_train_labels, x_test, y_test_labels


# ======================================================================
#  Evaluation
# ======================================================================

def evaluate(net, x_test, y_labels, batch_size=256):
    correct = 0
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        lb = y_labels[i:i + batch_size]
        mu, _ = net.forward(xb)
        correct += (mu.argmax(dim=1) == lb).sum().item()
    return correct / len(x_test)


# ======================================================================
#  Training
# ======================================================================

def train(net, x_train, y_train_oh, x_test, y_test_labels,
          batch_size, sigma_v, n_epochs):

    print(f"\n  {'Epoch':>5s}  {'Acc':>7s}  {'Time':>8s}")
    print("  " + "─" * 26)

    best_acc = 0.0
    t_total = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()

        perm = torch.randperm(x_train.size(0), device=DEVICE)
        x_s = x_train[perm]
        y_s = y_train_oh[perm]

        for i in range(0, len(x_s), batch_size):
            xb = x_s[i:i + batch_size]
            yb = y_s[i:i + batch_size]
            net.step(xb, yb, sigma_v)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        acc = evaluate(net, x_test, y_test_labels)
        best_acc = max(best_acc, acc)
        print(f"  {epoch:5d}  {acc*100:6.2f}%  {dt:7.2f}s")

    total_time = time.perf_counter() - t_total
    print("  " + "─" * 26)
    print(f"  Best accuracy : {best_acc*100:.2f}%")
    print(f"  Total time    : {total_time:.1f}s")


# ======================================================================
#  Main
# ======================================================================

def main():
    print("=" * 56)
    print("  MNIST Classification — TAGI-Triton CNN")
    print("  Conv(1→32,5) → Pool(2) → Conv(32→64,5) → Pool(2)")
    print("  → FC(3136→256) → FC(256→10) → Remax")
    print("=" * 56)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n  Loading MNIST...", flush=True)
    x_train, y_train_oh, _, x_test, y_test_labels = load_mnist()
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    print(f"  Input shape: {x_train.shape[1:]}")

    # ── Build CNN ──
    net = Sequential([
        Conv2D(1, 32, 5, stride=1, padding=2, device=DEVICE),   # → 32×28×28
        ReLU(),
        AvgPool2D(2),                                             # → 32×14×14
        Conv2D(32, 64, 5, stride=1, padding=2, device=DEVICE),  # → 64×14×14
        ReLU(),
        AvgPool2D(2),                                             # → 64×7×7
        Flatten(),                                                # → 3136
        Linear(3136, 256, device=DEVICE, gain_mean=3.0, gain_var=3.0),
        ReLU(),
        Linear(256, 10, device=DEVICE, gain_mean=3.0, gain_var=3.0),
        Remax(),
        # Bernoulli(n_gh=32),
    ], device=DEVICE)

    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    # ── Forward scaling ──
    forward_scale_weights(net, x_train[:4096//4], target_var=1.0, skip_last_layer=True, verbose=True)

    # ── Verify ──
    verify_standardization(net, x_train[:4096//4], target_var=1.0, skip_last_layer=True, verbose=True)

    # ── Hyperparameters ──
    batch_size = 128
    sigma_v    = 0.01
    n_epochs   = 20

    print(f"\n  Batch size : {batch_size}")
    print(f"  σ_v        : {sigma_v}")
    print(f"  Epochs     : {n_epochs}")

    train(net, x_train, y_train_oh, x_test, y_test_labels,
          batch_size, sigma_v, n_epochs)


if __name__ == "__main__":
    main()
