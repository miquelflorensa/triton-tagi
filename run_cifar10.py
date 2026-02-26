"""
CIFAR-10 Classification with TAGI-Triton 3-Block CNN
=====================================================
Architecture (matches cuTAGI's build_3block_cnn):

  Block 1: Conv(3→64,  3×3, pad=1) → ReLU → Conv(64→64,   4×4, s=2, pad=1) → ReLU
  Block 2: Conv(64→128, 3×3, pad=1) → ReLU → Conv(128→128, 4×4, s=2, pad=1) → ReLU
  Block 3: Conv(128→256,3×3, pad=1) → ReLU → Conv(256→256, 4×4, s=2, pad=1) → ReLU
  Head:    Flatten → FC(4096→512) → ReLU → FC(512→10) → Remax

  Input: 32×32 → 16×16 → 8×8 → 4×4 (stride-2 convolutions replace pooling)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
from torchvision import datasets, transforms

from src import Sequential
from src.layers import Linear, ReLU, Conv2D, Flatten, Remax, Bernoulli, BatchNorm2D, AvgPool2D
from src.init import initialize_deep_network, autotune_decoupled_gains

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
#  Network builder
# ======================================================================

def build_simple_3cnn(num_classes=10, head="remax", device="cuda",
                      gain_w=1.0, gain_b=1.0):
    """
    Classic Simple 3-Conv CNN matching the requested BatchNorm architecture.
    """
    layers = [
        # Block 1
        Conv2D(3, 32, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(32, device=device),
        AvgPool2D(2),  # 32x32 -> 16x16
        
        # Block 2
        Conv2D(32, 64, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(64, device=device),
        AvgPool2D(2),  # 16x16 -> 8x8
        
        # Block 3
        Conv2D(64, 64, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(64, device=device),
        AvgPool2D(2),  # 8x8 -> 4x4
        
        # Classifier
        Flatten(),
        Linear(64 * 4 * 4, 256, device=device,
               gain_mean=gain_w, gain_var=gain_w),
        ReLU(),
        Linear(256, num_classes, device=device,
               gain_mean=gain_w, gain_var=gain_w),
    ]

    # Output head
    if head == "remax":
        layers.append(Remax())
    elif head == "bernoulli":
        layers.append(Bernoulli(n_gh=64))
    elif head in ("none", "logit"):
        pass
    else:
        raise ValueError(f"Unknown head: {head}")

    return Sequential(layers, device=device)


# ======================================================================
#  Data loading
# ======================================================================

def load_cifar10(data_dir="data"):
    """Load CIFAR-10 as (N, 3, 32, 32) tensors on DEVICE."""
    # Standard CIFAR-10 normalisation
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True,
                                transform=transform)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=transform)

    # Stack into tensors
    x_train = torch.stack([img for img, _ in train_ds]).to(DEVICE)
    y_train = torch.tensor([lbl for _, lbl in train_ds], device=DEVICE)

    x_test = torch.stack([img for img, _ in test_ds]).to(DEVICE)
    y_test = torch.tensor([lbl for _, lbl in test_ds], device=DEVICE)

    # One-hot 0/1 for Remax
    y_train_oh = torch.zeros(len(y_train), 10, device=DEVICE)
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 1.0)

    return x_train, y_train_oh, y_train, x_test, y_test


# ======================================================================
#  Evaluation
# ======================================================================

def evaluate(net, x_test, y_labels, batch_size=256):
    net.eval()
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

        net.train()
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
    print("=" * 60)
    print("  CIFAR-10 Classification — TAGI-Triton Simple 3-CNN")
    print("  Block 1: Conv(3→32,5) → BatchNorm → ReLU → Pool(2)")
    print("  Block 2: Conv(32→64,5) → BatchNorm → ReLU → Pool(2)")
    print("  Block 3: Conv(64→64,5) → BatchNorm → ReLU → Pool(2)")
    print("  Head:    FC(1024→256) → FC(256→10) → Remax")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n  Loading CIFAR-10...", flush=True)
    x_train, y_train_oh, _, x_test, y_test_labels = load_cifar10()
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    print(f"  Input shape: {x_train.shape[1:]}")

    # ── Build Simple 3-CNN ──
    net = build_simple_3cnn(
        num_classes=10, head="bernoulli", device=DEVICE,
        gain_w=1.0, gain_b=1.0,
    )

    # initialize_deep_network(net, gain_mu=2.0, gain_sigma=0.083, verbose=True)

    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    # ── Hyperparameters ──
    batch_size = 128
    sigma_v    = 0.01
    n_epochs   = 50

    print(f"\n  Batch size : {batch_size}")
    print(f"  σ_v        : {sigma_v}")
    print(f"  Epochs     : {n_epochs}")

    train(net, x_train, y_train_oh, x_test, y_test_labels,
          batch_size, sigma_v, n_epochs)


if __name__ == "__main__":
    main()
