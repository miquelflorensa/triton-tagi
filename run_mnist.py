"""
MNIST Classification with TAGI-Triton Library
==============================================
Architecture: 784 → 512 → 256 → 10 → Remax (MLP with Bayesian ReLU + Remax output)

Demonstrates the modular TAGI library on MNIST digit classification.
Reports per-epoch accuracy and uncertainty statistics.
"""

import sys
import os

# Ensure the parent directory is on the path so `src` can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
from torchvision import datasets, transforms

from src import Sequential
from src.layers import Linear, ReLU, Remax, Bernoulli

# ── reproducibility ──
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
#  Data loading
# ======================================================================

def load_mnist(data_dir="data"):
    """Load MNIST and return normalised tensors on DEVICE."""
    train_ds = datasets.MNIST(data_dir, train=True,  download=True)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True)

    x_train = train_ds.data.float().view(-1, 784) / 255.0
    x_test  = test_ds.data.float().view(-1, 784)  / 255.0

    # Standardise using training stats
    mu, sigma = x_train.mean(), x_train.std()
    x_train = ((x_train - mu) / sigma).to(DEVICE)
    x_test  = ((x_test  - mu) / sigma).to(DEVICE)

    y_train_labels = train_ds.targets.to(DEVICE)
    y_test_labels  = test_ds.targets.to(DEVICE)

    # One-hot encode with 0/1 (probability targets for Remax)
    y_train_oh = torch.zeros(len(y_train_labels), 10, device=DEVICE)
    y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 1.0)

    return x_train, y_train_oh, y_train_labels, x_test, y_test_labels


# ======================================================================
#  Evaluation
# ======================================================================

def evaluate(net, x_test, y_labels, batch_size=1024):
    """Compute classification accuracy + mean predictive variance."""
    correct = 0
    total_var = 0.0
    n_batches = 0

    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        lb = y_labels[i:i + batch_size]

        mu, var = net.forward(xb)

        correct   += (mu.argmax(dim=1) == lb).sum().item()
        total_var += var.mean().item()
        n_batches += 1

    acc = correct / len(x_test)
    avg_var = total_var / n_batches
    return acc, avg_var


# ======================================================================
#  Training loop
# ======================================================================

def train(net, x_train, y_train_oh, x_test, y_test_labels,
          batch_size, sigma_v, n_epochs):
    """Full training loop with per-epoch reporting."""

    print(f"\n  {'Epoch':>5s}  {'Acc':>7s}  {'Avg Var':>10s}  {'Time':>8s}")
    print("  " + "─" * 38)

    best_acc = 0.0
    t_total = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()

        # Shuffle
        perm = torch.randperm(x_train.size(0), device=DEVICE)
        x_s = x_train[perm]
        y_s = y_train_oh[perm]

        # Mini-batch training
        for i in range(0, len(x_s), batch_size):
            xb = x_s[i:i + batch_size]
            yb = y_s[i:i + batch_size]
            net.step(xb, yb, sigma_v)

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        # Evaluate
        acc, avg_var = evaluate(net, x_test, y_test_labels)
        best_acc = max(best_acc, acc)

        print(f"  {epoch:5d}  {acc*100:6.2f}%  {avg_var:10.4f}  {dt:7.2f}s")

    total_time = time.perf_counter() - t_total
    print("  " + "─" * 38)
    print(f"  Best accuracy : {best_acc*100:.2f}%")
    print(f"  Total time    : {total_time:.1f}s")
    return best_acc


# ======================================================================
#  Main
# ======================================================================

def main():
    print("=" * 56)
    print("  MNIST Classification — TAGI-Triton Library")
    print("  MLP: 784 → 512 → 256 → 10 → Remax")
    print("=" * 56)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠  No GPU found — running on CPU (will be slow)")

    # ── Load data ──
    print("\n  Loading MNIST...", flush=True)
    x_train, y_train_oh, _, x_test, y_test_labels = load_mnist()
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")

    # ── Build network ──
    net = Sequential([
        Linear(784, 512, device=DEVICE, gain_mean=1.0, gain_var=1.0),
        ReLU(),
        Linear(512, 256, device=DEVICE, gain_mean=1.0, gain_var=1.0),
        ReLU(),
        Linear(256, 10,  device=DEVICE, gain_mean=1.0, gain_var=1.0),
        # Remax(),
        Bernoulli(n_gh=32),
    ], device=DEVICE)

    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    # ── Hyperparameters ──
    batch_size = 128
    sigma_v    = 0.1
    n_epochs   = 50

    print(f"\n  Batch size : {batch_size}")
    print(f"  σ_v        : {sigma_v}")
    print(f"  Epochs     : {n_epochs}")

    # ── Train ──
    train(net, x_train, y_train_oh, x_test, y_test_labels,
          batch_size, sigma_v, n_epochs)


if __name__ == "__main__":
    main()
