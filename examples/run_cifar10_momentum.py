"""
CIFAR-10 with State-Space Momentum — comparison script.

Runs the same 3-block CNN architecture from run_cifar10.py but uses
StateSpaceMomentum as the optimizer instead of standard TAGI updates.

Usage:
    python run_cifar10_momentum.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import time
from torchvision import datasets, transforms

from triton_tagi import Sequential, StateSpaceMomentum
from triton_tagi.layers import (Linear, ReLU, Conv2D, Flatten, Remax,
                         BatchNorm2D, AvgPool2D)
from triton_tagi.monitor import TAGIMonitor

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
#  Network builder (identical to run_cifar10.py)
# ======================================================================

def build_simple_3cnn(num_classes=10, device="cuda", gain_w=1.0, gain_b=1.0):
    layers = [
        Conv2D(3, 32, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(32, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),

        Conv2D(32, 64, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(64, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),

        Conv2D(64, 64, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(64, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),

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
    x_test  = torch.stack([img for img, _ in test_ds]).to(DEVICE)
    y_test  = torch.tensor([lbl for _, lbl in test_ds], device=DEVICE)

    y_train_oh = torch.zeros(len(y_train), 10, device=DEVICE)
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 1.0)

    return x_train, y_train_oh, y_train, x_test, y_test


# ======================================================================
#  GPU augmentation (same as run_cifar10.py)
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
#  Training with StateSpaceMomentum + Monitor
# ======================================================================

def train(opt, net, x_train, y_train_oh, y_train_labels,
          x_test, y_test_labels,
          batch_size, sigma_v, n_epochs,
          monitor=None, monitor_every=5,
          anneal_rate=1.05):

    print(f"\n  {'Epoch':>5s}  {'Train':>7s}  {'Test':>7s}  {'σ_v':>7s}  {'Time':>8s}")
    print("  " + "─" * 45)

    best_acc = 0.0
    t_total = time.perf_counter()
    current_sigma_v = sigma_v
    x_probe = x_train[:256]

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        net.train()

        perm = torch.randperm(x_train.size(0), device=DEVICE)
        x_s = x_train[perm]
        y_s = y_train_oh[perm]

        for i in range(0, len(x_s), batch_size):
            xb = gpu_augment(x_s[i:i + batch_size])
            yb = y_s[i:i + batch_size]
            opt.step(xb, yb, current_sigma_v)

            if monitor is not None:
                monitor.count_step()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        train_acc = evaluate(net, x_train, y_train_labels)
        acc = evaluate(net, x_test, y_test_labels)
        best_acc = max(best_acc, acc)

        print(f"  {epoch:5d}  {train_acc*100:6.2f}%  {acc*100:6.2f}%  "
              f"{current_sigma_v:7.4f}  {elapsed:7.2f}s")

        # ── Monitor snapshot ──
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
    print("  CIFAR-10 — TAGI + State-Space Momentum")
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

    # ── Wrap with state-space momentum ──
    # vel_cap_frac: max velocity contribution per step = frac × sqrt(S)
    #   0.1 → conservative (10% of param std), 0.3 → moderate, 1.0 → uncapped
    opt = StateSpaceMomentum(
        net,
        gamma=0.9,         # velocity decay
        dt=0.5,            # velocity integration step (high dt + cap = safe momentum)
        vel_cap_frac=1.0,  # velocity capped at 10% of sqrt(S) per step
        decouple=True,     # don't inflate S from velocity
        # q_theta, sigma_v0, q_v all auto-scaled from parameter variances
    )
    print(f"\n  Optimizer: {opt}")

    # ── Monitor ──
    monitor = TAGIMonitor(net, log_dir="run_logs_momentum", probe_size=256)
    net.eval()
    monitor.record(epoch=0, x_probe=x_train[:256], tag="init")
    monitor.print_report()

    # ── Hyperparameters ──
    batch_size    = 128
    sigma_v       = 0.05
    n_epochs      = 50
    monitor_every = 5
    anneal_rate   = 1.05

    print(f"\n  Batch size     : {batch_size}")
    print(f"  Initial σ_v    : {sigma_v}")
    print(f"  Anneal Rate    : {anneal_rate} (starts after epoch 20)")
    print(f"  Epochs         : {n_epochs}")
    print(f"  Momentum γ     : {opt.gamma}")
    print(f"  Momentum dt    : {opt.dt}")
    print(f"  Decouple       : {opt.decouple}")
    print(f"  Monitor every  : {monitor_every} epoch(s)")

    train(opt, net, x_train, y_train_oh, y_train_labels,
          x_test, y_test_labels,
          batch_size, sigma_v, n_epochs,
          monitor=monitor, monitor_every=monitor_every,
          anneal_rate=anneal_rate)

    # ── Final plots and CSV ──
    monitor.plot("run_logs_momentum/monitor.png")
    monitor.save_csv("run_logs_momentum/monitor.csv")


if __name__ == "__main__":
    main()
