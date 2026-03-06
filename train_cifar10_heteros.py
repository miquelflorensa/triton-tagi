"""
CIFAR-10 Classification with TAGI-V (Heteroscedastic Noise Learning)
=====================================================================
Same architecture as run_cifar10.py but with TAGI-V:
  - No σ_v (learned heteroscedastically)
  - No Remax (regression-style with ±3 labels)
  - 2×K outputs: K means + K learned variances (via EvenSoftplus)

Architecture:
  Block 1: Conv(3→32, 5×5, pad=2) → ReLU → BatchNorm → AvgPool(2)
  Block 2: Conv(32→64, 5×5, pad=2) → ReLU → BatchNorm → AvgPool(2)
  Block 3: Conv(64→64, 5×5, pad=2) → ReLU → BatchNorm → AvgPool(2)
  Head:    Flatten → FC(1024→256) → ReLU → FC(256→20) → EvenSoftplus(10)
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
from src.layers import Linear, ReLU, Conv2D, Flatten, BatchNorm2D, AvgPool2D, EvenSoftplus
from src.monitor import TAGIMonitor

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 10


# ======================================================================
#  Network builder
# ======================================================================

def build_simple_3cnn(num_classes=10, device="cuda",
                      gain_w=1.0, gain_b=1.0):
    """
    Classic Simple 3-Conv CNN — TAGI-V variant.
    Output is 2×num_classes: even = means, odd = learned σ_v².
    """
    layers = [
        # Block 1
        Conv2D(3, 32, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(32, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),  # 32x32 -> 16x16

        # Block 2
        Conv2D(32, 64, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(64, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),  # 16x16 -> 8x8

        # Block 3
        Conv2D(64, 64, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        BatchNorm2D(64, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),  # 8x8 -> 4x4

        # Classifier
        Flatten(),
        Linear(64 * 4 * 4, 256, device=device,
               gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        Linear(256, 2 * num_classes, device=device,
               gain_w=gain_w, gain_b=gain_b),

        # TAGI-V: softplus on odd indices (learned variance channels)
        EvenSoftplus(num_classes),
    ]

    return Sequential(layers, device=device)


# ======================================================================
#  Data loading
# ======================================================================

def load_cifar10(data_dir="data"):
    """Load CIFAR-10 as (N, 3, 32, 32) tensors on DEVICE."""
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

    # ±3 encoding: correct class = +3, others = -3
    y_train_oh = torch.full((len(y_train), N_CLASSES), -3.0, device=DEVICE)
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 3.0)

    return x_train, y_train_oh, y_train, x_test, y_test


# ======================================================================
#  Evaluation
# ======================================================================

def evaluate(net, x_test, y_labels, batch_size=256):
    """Evaluate accuracy — predictions from even-indexed outputs (means)."""
    net.eval()
    correct = 0
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        lb = y_labels[i:i + batch_size]
        mu, _ = net.forward(xb)
        # Extract only the mean channels (even indices: 0, 2, 4, ..., 18)
        mu_means = mu[:, 0::2]
        correct += (mu_means.argmax(dim=1) == lb).sum().item()
    return correct / len(x_test)


def evaluate_sigma_v(net, x_test, batch_size=256):
    """Evaluate the learned σ_v² statistics."""
    net.eval()
    all_sigma_v = []
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        mu, _ = net.forward(xb)
        sigma_v_sq = mu[:, 1::2]  # odd indices = learned variance
        all_sigma_v.append(sigma_v_sq.cpu())
    return torch.cat(all_sigma_v, dim=0)


# ======================================================================
#  GPU-batched augmentation (no CPU round-trip)
# ======================================================================

def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    """
    Random horizontal flip + random crop, fully on GPU.
    x : (B, C, H, W) float tensor already on device.
    Returns an augmented tensor of the same shape.
    """
    B, C, H, W = x.shape
    pad = 4

    # ── Random horizontal flip (vectorised, no loop) ──
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)

    # ── Reflect-pad the whole batch in one call: (B, C, H+2p, W+2p) ──
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")

    # ── Sample one (top, left) crop offset per image ──
    top  = torch.randint(0, 2 * pad, (B,), device=x.device)
    left = torch.randint(0, 2 * pad, (B,), device=x.device)

    # ── Vectorised crop via advanced indexing ──
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
#  Training
# ======================================================================

def train(net, x_train, y_train_oh, y_train_labels, x_test, y_test_labels,
          batch_size, n_epochs,
          monitor: TAGIMonitor = None,
          monitor_every: int = 1):
    """
    TAGI-V training loop — no sigma_v needed (learned heteroscedastically).
    """
    print(f"\n  {'Epoch':>5s}  {'Train':>7s}  {'Test':>7s}  {'σ_v² mean':>9s}  {'Time':>8s}")
    print("  " + "─" * 50)

    best_acc = 0.0
    t_total = time.perf_counter()
    x_probe = x_train[:256]

    # sigma_v is unused by the heteroscedastic update, but the API requires it
    dummy_sigma_v = 0.0

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
            net.step(xb, yb, dummy_sigma_v)

            if monitor is not None:
                monitor.count_step()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        train_acc = evaluate(net, x_train, y_train_labels)
        acc = evaluate(net, x_test, y_test_labels)
        best_acc = max(best_acc, acc)

        # Learned σ_v² statistics
        sv_sq = evaluate_sigma_v(net, x_test[:1000])
        sv_mean = sv_sq.mean().item()

        print(f"  {epoch:5d}  {train_acc*100:6.2f}%  {acc*100:6.2f}%  {sv_mean:9.4f}  {dt:7.2f}s")

        # ── monitor snapshot ──
        if monitor is not None and epoch % monitor_every == 0:
            net.eval()
            monitor.record(epoch, x_probe, tag=f"acc={acc*100:.1f}%")
            monitor.print_report()

    total_time = time.perf_counter() - t_total

    # ── Final σ_v statistics ──
    sv_sq = evaluate_sigma_v(net, x_test)
    print("  " + "─" * 50)
    print(f"  Best accuracy : {best_acc*100:.2f}%")
    print(f"  Total time    : {total_time:.1f}s")
    print()
    print("  Learned σ_v² statistics (test set):")
    print(f"    Mean:   {sv_sq.mean().item():.6f}")
    print(f"    Std:    {sv_sq.std().item():.6f}")
    print(f"    Min:    {sv_sq.min().item():.6f}")
    print(f"    Max:    {sv_sq.max().item():.6f}")
    per_class = sv_sq.mean(dim=0)
    print(f"    Per-class: {[f'{v:.4f}' for v in per_class.tolist()]}")


# ======================================================================
#  Main
# ======================================================================

def main():
    print("=" * 60)
    print("  CIFAR-10 — TAGI-V Heteroscedastic (no σ_v, no Remax)")
    print("  Block 1: Conv(3→32,5) → ReLU → BatchNorm → Pool(2)")
    print("  Block 2: Conv(32→64,5) → ReLU → BatchNorm → Pool(2)")
    print("  Block 3: Conv(64→64,5) → ReLU → BatchNorm → Pool(2)")
    print("  Head:    FC(1024→256) → ReLU → FC(256→20) → EvenSoftplus(10)")
    print("  Labels:  ±3 encoding")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n  Loading CIFAR-10...", flush=True)
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels = load_cifar10()
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    print(f"  Input shape: {x_train.shape[1:]}")

    # ── Build Simple 3-CNN with TAGI-V head ──
    net = build_simple_3cnn(
        num_classes=10, device=DEVICE,
        gain_w=0.6, gain_b=0.6,
    )

    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    # ── Monitor: snapshot at init ──
    monitor = TAGIMonitor(net, log_dir="run_logs", probe_size=256)
    net.eval()
    monitor.record(epoch=0, x_probe=x_train[:256], tag="init")
    monitor.print_report()

    # ── Hyperparameters (no sigma_v needed!) ──
    batch_size    = 128
    n_epochs      = 100
    monitor_every = 10

    print(f"\n  Batch size     : {batch_size}")
    print(f"  σ_v            : learned (heteroscedastic)")
    print(f"  Epochs         : {n_epochs}")
    print(f"  Monitor every  : {monitor_every} epoch(s)")

    train(net, x_train, y_train_oh, y_train_labels, x_test, y_test_labels,
          batch_size, n_epochs,
          monitor=monitor, monitor_every=monitor_every)

    # ── Final plots and CSV ──
    monitor.plot("run_logs/monitor.png")
    monitor.save_csv("run_logs/monitor.csv")


if __name__ == "__main__":
    main()
