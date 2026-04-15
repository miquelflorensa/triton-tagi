"""
CIFAR-100 ResNet-18 with State-Space Momentum
==============================================
Same ResNet-18 architecture, adapted for 100 classes.

Usage:
    python run_cifar100_momentum.py
    python run_cifar100_momentum.py --gamma 0.95 --dt 0.5
    python run_cifar100_momentum.py --sigma-v 0.01
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

from triton_tagi import Sequential, StateSpaceMomentum
from triton_tagi.layers import (Linear, ReLU, Conv2D, Flatten, Remax,
                        BatchNorm2D, AvgPool2D, ResBlock)
from triton_tagi.monitor import TAGIMonitor
from triton_tagi.init import init_residual_aware
from run_resnet18 import (
    depth_gain, build_resnet18, evaluate, compute_ece,
    calibrate_sigma_v, gpu_augment, save_checkpoint,
)

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
#  Data loading — CIFAR-100
# ======================================================================

def load_cifar100(data_dir="data"):
    """Load CIFAR-100 as (N, 3, 32, 32) tensors on DEVICE."""
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR100(data_dir, train=True,  download=True,
                                 transform=train_transform)
    test_ds  = datasets.CIFAR100(data_dir, train=False, download=True,
                                 transform=test_transform)

    x_train = torch.stack([img for img, _ in train_ds]).to(DEVICE)
    y_train = torch.tensor([lbl for _, lbl in train_ds], device=DEVICE)

    x_test = torch.stack([img for img, _ in test_ds]).to(DEVICE)
    y_test = torch.tensor([lbl for _, lbl in test_ds], device=DEVICE)

    num_classes = 100
    y_train_oh = torch.zeros(len(y_train), num_classes, device=DEVICE)
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 1.0)

    return x_train, y_train_oh, y_train, x_test, y_test


# ======================================================================
#  Training loop
# ======================================================================

def train(opt, net, x_train, y_train_oh, y_train_labels,
          x_test, y_test_labels,
          batch_size, initial_sigma_v, n_epochs,
          monitor=None, monitor_every=10,
          decay_factor=0.95, min_sigma_v=0.01,
          label_smooth=0.0,
          checkpoint_dir="run_logs_cifar100_momentum/checkpoints",
          start_epoch=1):

    num_classes = y_train_oh.shape[1]
    print(f"\n  {'Epoch':>5s}  {'Train':>7s}  {'Test':>7s}  {'ECE':>6s}  "
          f"{'σ_v':>7s}  {'Time':>8s}")
    print("  " + "─" * 57)

    best_acc = 0.0
    t_total = time.perf_counter()
    x_probe = x_train[:256]
    current_sigma_v = initial_sigma_v

    for epoch in range(start_epoch, start_epoch + n_epochs):
        t0 = time.perf_counter()
        net.train()

        perm = torch.randperm(x_train.size(0), device=DEVICE)
        x_s = x_train[perm]
        y_s = y_train_oh[perm]

        for i in range(0, len(x_s), batch_size):
            xb = x_s[i:i + batch_size]
            yb = y_s[i:i + batch_size]
            if label_smooth > 0.0:
                yb = (1.0 - label_smooth) * yb + label_smooth / num_classes
            xb = gpu_augment(xb)
            opt.step(xb, yb, current_sigma_v)

            if monitor is not None:
                monitor.count_step()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        train_acc = evaluate(net, x_train, y_train_labels)
        acc = evaluate(net, x_test, y_test_labels)
        ece = compute_ece(net, x_test, y_test_labels)
        best_acc = max(best_acc, acc)

        print(f"  {epoch:5d}  {train_acc*100:6.2f}%  {acc*100:6.2f}%  "
              f"{ece:.4f}  {current_sigma_v:7.4f}  {dt:7.2f}s")

        save_checkpoint(net, epoch, checkpoint_dir)

        if monitor is not None and epoch % monitor_every == 0:
            net.eval()
            monitor.record(epoch, x_probe, tag=f"acc={acc*100:.1f}%")
            monitor.print_report()

        if epoch >= start_epoch + 5:
            current_sigma_v = max(current_sigma_v * decay_factor, min_sigma_v)

    total_time = time.perf_counter() - t_total
    print("  " + "─" * 57)
    print(f"  Best accuracy : {best_acc*100:.2f}%")
    print(f"  Total time    : {total_time:.1f}s")
    return best_acc


# ======================================================================
#  Main
# ======================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train TAGI ResNet-18 on CIFAR-100 with State-Space Momentum")
    parser.add_argument("--sigma-v", type=float, default=None,
                        help="Override σ_v instead of auto-calibrating")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Momentum velocity decay (default: 0.95)")
    parser.add_argument("--dt", type=float, default=0.5,
                        help="Velocity integration step (default: 0.5)")
    parser.add_argument("--vel-cap", type=float, default=0.1,
                        help="Velocity cap as fraction of sqrt(S) (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs (default: 50)")
    args = parser.parse_args()

    num_classes = 100

    print("=" * 62)
    print("  CIFAR-100 — TAGI ResNet-18 + State-Space Momentum")
    print("=" * 62)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ──
    print("\n  Loading CIFAR-100...", flush=True)
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels = load_cifar100()
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    print(f"  Classes: {num_classes}")

    # ── Build ResNet-18 for 100 classes ──
    eta = 0.5
    net = build_resnet18(
        num_classes=num_classes, head="remax", device=DEVICE,
        g_min=0.1, g_max=0.1,
    )
    init_residual_aware(net, eta=eta, verbose=True)

    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    # ── BN warmup ──
    print("\n  Initialising BN layers (single forward pass)...", flush=True)
    net.train()
    net.forward(x_train[:32])
    net.eval()
    torch.cuda.empty_cache()

    # ── Wrap with state-space momentum ──
    opt = StateSpaceMomentum(
        net,
        gamma=args.gamma,
        dt=args.dt,
        vel_cap_frac=args.vel_cap,
        decouple=True,
        eps_Q=1e-6,
    )
    print(f"\n  Optimizer: {opt}")

    # ── Monitor ──
    log_dir = "run_logs_cifar100_momentum"
    monitor = TAGIMonitor(net, log_dir=log_dir, probe_size=256)
    net.eval()
    monitor.record(epoch=0, x_probe=x_train[:256], tag="init")
    monitor.print_report()

    # ── Calibrate σ_v ──
    if args.sigma_v is not None:
        sigma_v = args.sigma_v
        print(f"\n  Using provided σ_v = {sigma_v:.6f}")
    else:
        print("\n  Thermodynamic σ_v calibration:")
        torch.cuda.empty_cache()
        sigma_v = calibrate_sigma_v(net, x_train[:256], tau=1.0, batch_size=32)

    # ── Hyperparameters ──
    batch_size     = 128
    n_epochs       = args.epochs
    monitor_every  = 5
    label_smooth   = 0.0
    decay_factor   = 0.95
    min_sigma_v    = 0.10 * sigma_v   # higher floor (10%) to prevent late explosion
    checkpoint_dir = f"{log_dir}/checkpoints"

    print(f"\n  Batch size     : {batch_size}")
    print(f"  Initial σ_v    : {sigma_v:.4f}")
    print(f"  Decay factor   : {decay_factor} (starts after 5 epochs)")
    print(f"  Min σ_v        : {min_sigma_v:.4f}")
    print(f"  Epochs         : {n_epochs}")
    print(f"  Momentum γ     : {opt.gamma}")
    print(f"  Momentum dt    : {opt.dt}")
    print(f"  Vel cap frac   : {opt.vel_cap_frac}")
    print(f"  Decouple       : {opt.decouple}")
    print(f"  Monitor every  : {monitor_every} epoch(s)")

    # ── Train ──
    best_acc = train(
        opt, net, x_train, y_train_oh, y_train_labels,
        x_test, y_test_labels,
        batch_size, sigma_v, n_epochs,
        monitor=monitor, monitor_every=monitor_every,
        decay_factor=decay_factor,
        min_sigma_v=min_sigma_v,
        label_smooth=label_smooth,
        checkpoint_dir=checkpoint_dir,
    )

    # ── Save logs ──
    monitor.plot(f"{log_dir}/monitor.png")
    monitor.save_csv(f"{log_dir}/monitor.csv")


if __name__ == "__main__":
    main()
