"""
CIFAR-10 with TAGI-Triton 3-Block CNN (Heteroscedastic)
=====================================================
Model uses the heteroscedastic method which outputs 2 * num_classes
units in the final layer to learn the noise variance.

Architecture:
  Block 1: Conv(3→32,5) → BatchNorm → ReLU → Pool(2)
  Block 2: Conv(32→64,5) → BatchNorm → ReLU → Pool(2)
  Block 3: Conv(64→64,5) → BatchNorm → ReLU → Pool(2)
  Head:    FC(1024→256) → FC(256→20)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import time
from torchvision import datasets, transforms

from src import Sequential
from src.layers import Linear, Conv2D, Flatten, BatchNorm2D, AvgPool2D, LeakyReLU
from src.monitor import TAGIMonitor

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================
#  Network builder
# ======================================================================

def build_simple_3cnn_heteros(num_classes=10, device="cuda", gain_w=0.1, gain_b=0.1):
    """
    Classic Simple 3-Conv CNN matching the requested BatchNorm architecture.
    Outputs 2 * num_classes to trigger heteroscedastic update.
    """
    layers = [
        # Block 1
        Conv2D(3, 32, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        LeakyReLU(alpha=0.01),
        BatchNorm2D(32, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),  # 32x32 -> 16x16
        
        # Block 2
        Conv2D(32, 64, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        LeakyReLU(alpha=0.01),
        BatchNorm2D(64, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),  # 16x16 -> 8x8
        
        # Block 3
        Conv2D(64, 64, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        LeakyReLU(alpha=0.01),
        BatchNorm2D(64, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),  # 8x8 -> 4x4
        
        # Classifier
        Flatten(),
        Linear(64 * 4 * 4, 256, device=device,
               gain_w=gain_w, gain_b=gain_b),
        LeakyReLU(alpha=0.01),
        # Final layer outputs 2 * num_classes for mean and learned variance components
        Linear(256, 2 * num_classes, device=device,
               gain_w=gain_w, gain_b=gain_b),
    ]

    return Sequential(layers, device=device)

# ======================================================================
#  Data loading
# ======================================================================

def load_cifar10(data_dir="data"):
    """Load CIFAR-10 as (N, 3, 32, 32) tensors on DEVICE."""
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

    x_train = torch.stack([img for img, _ in train_ds]).to(DEVICE)
    y_train = torch.tensor([lbl for _, lbl in train_ds], device=DEVICE)

    x_test = torch.stack([img for img, _ in test_ds]).to(DEVICE)
    y_test = torch.tensor([lbl for _, lbl in test_ds], device=DEVICE)

    # One-hot representation for targets (+3 for correct, -3 for incorrect classes)
    num_classes = 10
    y_train_oh = torch.zeros(len(y_train), num_classes, device=DEVICE) - 3.0 
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 3.0)

    return x_train, y_train_oh, y_train, x_test, y_test

# ======================================================================
#  Evaluation
# ======================================================================

def evaluate(net, x_test, y_labels, num_classes=10, batch_size=256):
    net.eval()
    correct = 0
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        lb = y_labels[i:i + batch_size]
        mu, _ = net.forward(xb)
        # Extract only the first `num_classes` means which represent the actual output
        mu_classes = mu[:, :num_classes]
        correct += (mu_classes.argmax(dim=1) == lb).sum().item()
    return correct / len(x_test)

# ======================================================================
#  Training
# ======================================================================

def train(net, x_train, y_train_oh, x_test, y_test_labels,
          batch_size, sigma_v, n_epochs, num_classes=10,
          monitor: TAGIMonitor = None,
          monitor_every: int = 1):
    
    print(f"\n  {'Epoch':>5s}  {'Acc':>7s}  {'Time':>8s}")
    print("  " + "─" * 26)

    best_acc = 0.0
    t_total = time.perf_counter()
    x_probe = x_train[:256]

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()

        net.train()
        perm = torch.randperm(x_train.size(0), device=DEVICE)
        x_s = x_train[perm]
        y_s = y_train_oh[perm]

        for i in range(0, len(x_s), batch_size):
            xb = x_s[i:i + batch_size]
            yb = y_s[i:i + batch_size]
            # sigma_v is passed but unused directly by kernel when outputs=2*targets
            net.step(xb, yb, sigma_v)
            if monitor is not None:
                monitor.count_step()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        acc = evaluate(net, x_test, y_test_labels, num_classes=num_classes)
        best_acc = max(best_acc, acc)
        print(f"  {epoch:5d}  {acc*100:6.2f}%  {dt:7.2f}s")

        if monitor is not None and epoch % monitor_every == 0:
            net.eval()
            monitor.record(epoch, x_probe, tag=f"acc={acc*100:.1f}%")
            monitor.print_report()

    total_time = time.perf_counter() - t_total
    print("  " + "─" * 26)
    print(f"  Best accuracy : {best_acc*100:.2f}%")
    print(f"  Total time    : {total_time:.1f}s")

# ======================================================================
#  Main
# ======================================================================

def main():
    print("=" * 60)
    print("  CIFAR-10 Heteroscedastic — TAGI-Triton Simple 3-CNN")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n  Loading CIFAR-10...", flush=True)
    x_train, y_train_oh, _, x_test, y_test_labels = load_cifar10()
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    print(f"  Input shape: {x_train.shape[1:]}")

    num_classes = 10
    net = build_simple_3cnn_heteros(
        num_classes=num_classes, device=DEVICE,
        gain_w=1.0, gain_b=1.0,
    )

    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    monitor = TAGIMonitor(net, log_dir="run_logs", probe_size=256)
    net.eval()
    monitor.record(epoch=0, x_probe=x_train[:256], tag="init")
    monitor.print_report()

    batch_size    = 256
    sigma_v       = 0.0001
    n_epochs      = 50
    monitor_every = 5

    print(f"\n  Batch size     : {batch_size}")
    print(f"  sigma_v (API)  : {sigma_v}")
    print(f"  Epochs         : {n_epochs}")
    print(f"  Monitor every  : {monitor_every} epoch(s)")

    train(net, x_train, y_train_oh, x_test, y_test_labels,
          batch_size, sigma_v, n_epochs, num_classes=num_classes,
          monitor=monitor, monitor_every=monitor_every)

    monitor.plot("run_logs/monitor_heteros.png")
    monitor.save_csv("run_logs/monitor_heteros.csv")

if __name__ == "__main__":
    main()
