"""
CIFAR-10 Classification with Adam-TAGI Optimizer
=================================================
Same 3-block CNN as run_cifar10.py, but uses AdamTAGI instead of the
standard capped Bayesian update.

Comparison:
    python run_cifar10.py          # standard TAGI (baseline)
    python run_cifar10_adam.py      # Adam-TAGI (this script)
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import time
from torchvision import datasets, transforms

from src import Sequential
from src.layers import (Linear, ReLU, Conv2D, Flatten, Remax,
                        BatchNorm2D, AvgPool2D)
from src.optimizer import AdamTAGI
from src.monitor import TAGIMonitor

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
#  Network builder (identical to run_cifar10.py)
# ======================================================================

def build_simple_3cnn(num_classes=10, device="cuda", gain_w=1.0, gain_b=1.0):
    layers = [
        # Block 1
        Conv2D(3, 32, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(32, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),

        # Block 2
        Conv2D(32, 64, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(64, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),

        # Block 3
        Conv2D(64, 64, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(64, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),

        # Classifier
        Flatten(),
        Linear(64 * 4 * 4, 256, device=device,
               gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        Linear(256, num_classes, device=device,
               gain_w=gain_w, gain_b=gain_b),
        Remax(),
    ]
    return Sequential(layers, device=device)


# ======================================================================
#  Data loading
# ======================================================================

def load_cifar10(data_dir="data"):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True,
                                transform=train_transform)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=test_transform)

    x_train = torch.stack([img for img, _ in train_ds]).to(DEVICE)
    y_train = torch.tensor([lbl for _, lbl in train_ds], device=DEVICE)
    x_test = torch.stack([img for img, _ in test_ds]).to(DEVICE)
    y_test = torch.tensor([lbl for _, lbl in test_ds], device=DEVICE)

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
#  GPU-batched augmentation
# ======================================================================

def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    pad = 4

    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)

    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")

    top  = torch.randint(0, 2 * pad, (B,), device=x.device)
    left = torch.randint(0, 2 * pad, (B,), device=x.device)

    rows = top.unsqueeze(1)  + torch.arange(H, device=x.device).unsqueeze(0)
    cols = left.unsqueeze(1) + torch.arange(W, device=x.device).unsqueeze(0)

    x_crop = x_pad[
        torch.arange(B, device=x.device)[:, None, None, None],
        torch.arange(C, device=x.device)[None, :, None, None],
        rows[:, None, :, None].expand(B, C, H, W),
        cols[:, None, None, :].expand(B, C, H, W),
    ]
    return x_crop


# ======================================================================
#  Training loop (uses AdamTAGI)
# ======================================================================

def train(opt, x_train, y_train_oh, y_train_labels, x_test, y_test_labels,
          batch_size, initial_sigma_v, n_epochs,
          monitor=None, monitor_every=5, anneal_rate=1.05):

    net = opt.net
    print(f"\n  {'Epoch':>5s}  {'Train':>7s}  {'Test':>7s}  {'σ_v':>7s}  {'Time':>8s}")
    print("  " + "─" * 45)

    best_acc = 0.0
    t_total = time.perf_counter()
    x_probe = x_train[:256]
    current_sigma_v = initial_sigma_v

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()

        net.train()
        perm = torch.randperm(x_train.size(0), device=DEVICE)
        x_s = x_train[perm]
        y_s = y_train_oh[perm]

        for i in range(0, len(x_s), batch_size):
            xb = x_s[i:i + batch_size]
            yb = y_s[i:i + batch_size]
            xb = gpu_augment(xb)

            # ── Adam-TAGI step (replaces net.step) ──
            opt.step(xb, yb, current_sigma_v)

            if monitor is not None:
                monitor.count_step()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        train_acc = evaluate(net, x_train, y_train_labels)
        acc = evaluate(net, x_test, y_test_labels)
        best_acc = max(best_acc, acc)

        print(f"  {epoch:5d}  {train_acc*100:6.2f}%  {acc*100:6.2f}%  "
              f"{current_sigma_v:7.3f}  {dt:7.2f}s")

        if monitor is not None and epoch % monitor_every == 0:
            net.eval()
            monitor.record(epoch, x_probe, tag=f"acc={acc*100:.1f}%")
            monitor.print_report()

        if epoch > 20:
            current_sigma_v *= anneal_rate

    total_time = time.perf_counter() - t_total
    print("  " + "─" * 45)
    print(f"  Best accuracy : {best_acc*100:.2f}%")
    print(f"  Total time    : {total_time:.1f}s")


# ======================================================================
#  Main
# ======================================================================

def main():
    print("=" * 60)
    print("  CIFAR-10 — Adam-TAGI Optimizer Experiment")
    print("  Same 3-block CNN, swapping capped update for Adam-TAGI")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n  Loading CIFAR-10...", flush=True)
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels = load_cifar10()
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")

    # ── Build network ──
    net = build_simple_3cnn(num_classes=10, device=DEVICE,
                            gain_w=0.1, gain_b=0.1)
    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    # ── Warm-up BN layers ──
    net.train()
    net.forward(x_train[:32])
    net.eval()

    # ── Create Adam-TAGI optimizer ──
    # Hyperparameters
    beta1  = 0.9      # first moment decay
    beta2  = 0.999    # second moment decay
    eps_Q  = 1e-9     # process noise (plasticity floor)

    opt = AdamTAGI(net, beta1=beta1, beta2=beta2, eps_Q=eps_Q)
    print(f"\n  Optimizer: {opt}")

    # ── Monitor ──
    monitor = TAGIMonitor(net, log_dir="run_logs_adam", probe_size=256)
    net.eval()
    monitor.record(epoch=0, x_probe=x_train[:256], tag="init")
    monitor.print_report()

    # ── Training config ──
    batch_size    = 128
    sigma_v       = 0.05
    n_epochs      = 100
    monitor_every = 10
    anneal_rate   = 0.98

    print(f"\n  Adam-TAGI config:")
    print(f"    β1={beta1}, β2={beta2}, ε_Q={eps_Q}")
    print(f"  Training config:")
    print(f"    batch_size={batch_size}, σ_v={sigma_v}, epochs={n_epochs}")
    print(f"    anneal_rate={anneal_rate} (after epoch 20)")

    # ── Train ──
    train(opt, x_train, y_train_oh, y_train_labels,
          x_test, y_test_labels,
          batch_size, sigma_v, n_epochs,
          monitor=monitor, monitor_every=monitor_every,
          anneal_rate=anneal_rate)

    # ── Save logs ──
    monitor.plot("run_logs_adam/monitor.png")
    monitor.save_csv("run_logs_adam/monitor.csv")


if __name__ == "__main__":
    main()
