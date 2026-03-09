"""
CIFAR-100 Classification with TAGI-Triton ResNet-18
====================================================
Architecture — exact replica of cuTAGI's test_resnet.cpp:

  Stem:     Conv(3→64, 3×3, s=1, pad=1) → ReLU → BN(64)
  Layer 1:  ResBlock(64→64)   × 2         (identity shortcuts)
  Layer 2:  ResBlock(64→128, s=2) + ResBlock(128→128)
  Layer 3:  ResBlock(128→256, s=2) + ResBlock(256→256)
  Layer 4:  ResBlock(256→512, s=2) + ResBlock(512→512)
  Head:     AvgPool(4×4) → Flatten → FC(512→100) → Remax

  Each ResBlock main path:    Conv3×3→ReLU→BN→Conv3×3→ReLU→BN
  Projection shortcut:        Conv2×2(stride=2)→ReLU→BN
  NO activation after the residual addition.

  Input: 32×32 → 32×32 → 16×16 → 8×8 → 4×4 → global avg → 100 classes
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
                        BatchNorm2D, AvgPool2D, ResBlock, Bernoulli)
from src.monitor import TAGIMonitor
from src.init import init_residual_aware

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 100


# ======================================================================
#  Depth-scaled gain function (Section 4 of theory doc)
# ======================================================================

def depth_gain(layer_idx: int, total_layers: int,
               g_min: float = 0.1, g_max: float = 0.5) -> float:
    if total_layers <= 1:
        return g_min
    t = layer_idx / (total_layers - 1)
    g = g_min + (g_max - g_min) * 4.0 * t * (1.0 - t)
    return g


# ======================================================================
#  Network builder — ResNet-18 for CIFAR-100
# ======================================================================

def build_resnet18(num_classes=NUM_CLASSES, head="remax", device="cuda",
                   gain_w=1.0, gain_b=1.0,
                   g_min=0.1, g_max=0.5):
    N_SLOTS = 11

    def g(slot):
        return depth_gain(slot, N_SLOTS, g_min, g_max)

    layers = [
        # ── Stem: Conv(3→64) → ReLU → BN  (slot 0, 1) ──
        Conv2D(3, 64, kernel_size=3, stride=1, padding=1,
               device=device, gain_w=g(0), gain_b=g(0)),
        ReLU(),
        BatchNorm2D(64, device=device, gain_w=g(1), gain_b=g(1)),

        # ── Layer 1: 64 → 64, identity shortcuts (slots 2, 3) ──
        ResBlock(64, 64, stride=1,
                 device=device, gain_w=g(2), gain_b=g(2)),
        ResBlock(64, 64, stride=1,
                 device=device, gain_w=g(3), gain_b=g(3)),

        # ── Layer 2: 64 → 128, projection shortcut (slots 4, 5) ──
        ResBlock(64, 128, stride=2,
                 device=device, gain_w=g(4), gain_b=g(4)),
        ResBlock(128, 128, stride=1,
                 device=device, gain_w=g(5), gain_b=g(5)),

        # ── Layer 3: 128 → 256 (slots 6, 7) ──
        ResBlock(128, 256, stride=2,
                 device=device, gain_w=g(6), gain_b=g(6)),
        ResBlock(256, 256, stride=1,
                 device=device, gain_w=g(7), gain_b=g(7)),

        # ── Layer 4: 256 → 512 (slots 8, 9) ──
        ResBlock(256, 512, stride=2,
                 device=device, gain_w=g(8), gain_b=g(8)),
        ResBlock(512, 512, stride=1,
                 device=device, gain_w=g(9), gain_b=g(9)),

        # ── Head: AvgPool(4) → Flatten → FC(512→num_classes) (slot 10) ──
        AvgPool2D(4),
        Flatten(),
        Linear(512, num_classes,
               device=device, gain_w=g(10), gain_b=g(10)),
    ]

    if head == "remax":
        layers.append(Remax())
    elif head in ("bernoulli"):
        layers.append(Bernoulli())
    elif head in ("none", "logit"):
        pass
    else:
        raise ValueError(f"Unknown head: {head}")

    return Sequential(layers, device=device)


# ======================================================================
#  Data loading
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

    # One-hot for Remax
    y_train_oh = torch.zeros(len(y_train), NUM_CLASSES, device=DEVICE)
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 1.0)

    return x_train, y_train_oh, y_train, x_test, y_test


# ======================================================================
#  Evaluation
# ======================================================================

def evaluate(net, x_test, y_labels, batch_size=128):
    net.eval()
    correct = 0
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        lb = y_labels[i:i + batch_size]
        mu, _ = net.forward(xb)
        correct += (mu.argmax(dim=1) == lb).sum().item()
    return correct / len(x_test)


def compute_ece(net, x_test, y_labels, n_bins=15, batch_size=128):
    net.eval()
    all_conf = []
    all_correct = []

    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        lb = y_labels[i:i + batch_size]
        mu, _ = net.forward(xb)
        conf, pred = mu.max(dim=1)
        all_conf.append(conf.cpu())
        all_correct.append((pred == lb).float().cpu())

    all_conf    = torch.cat(all_conf)
    all_correct = torch.cat(all_correct)
    N = len(all_conf)

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (all_conf >= lo) & (all_conf < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = all_correct[mask].mean().item()
        bin_conf = all_conf[mask].mean().item()
        ece += (mask.sum().item() / N) * abs(bin_acc - bin_conf)

    return ece


def save_checkpoint(net, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    state = {}

    def _layer_state(layer):
        d = {}
        if hasattr(layer, "mw"):
            d["mw"] = layer.mw.cpu().clone()
            d["Sw"] = layer.Sw.cpu().clone()
            d["mb"] = layer.mb.cpu().clone()
            d["Sb"] = layer.Sb.cpu().clone()
        if hasattr(layer, "running_mean"):
            d["running_mean"] = layer.running_mean.cpu().clone()
            d["running_var"]  = layer.running_var.cpu().clone()
        return d

    for i, layer in enumerate(net.layers):
        if isinstance(layer, ResBlock):
            for j, sub in enumerate(layer._learnable):
                key = f"layer_{i}_sub_{j}_{type(sub).__name__}"
                state[key] = _layer_state(sub)
        else:
            s = _layer_state(layer)
            if s:
                state[f"layer_{i}_{type(layer).__name__}"] = s

    path = os.path.join(save_dir, f"checkpoint_epoch_{epoch:04d}.pt")
    torch.save({"epoch": epoch, "state": state}, path)
    return path


def load_checkpoint(net, path: str) -> int:
    import glob as _glob

    if path == "latest":
        candidates = sorted(_glob.glob(
            os.path.join("run_logs_cifar100", "checkpoints", "checkpoint_epoch_*.pt")))
        if not candidates:
            raise FileNotFoundError("No checkpoints found in run_logs_cifar100/checkpoints/")
        path = candidates[-1]

    print(f"  Loading checkpoint: {path}")
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ck["state"]

    def _restore(layer, d):
        if "mw" in d:
            layer.mw.data.copy_(d["mw"].to(DEVICE))
            layer.Sw.data.copy_(d["Sw"].to(DEVICE))
            layer.mb.data.copy_(d["mb"].to(DEVICE))
            layer.Sb.data.copy_(d["Sb"].to(DEVICE))
        if "running_mean" in d:
            layer.running_mean.copy_(d["running_mean"].to(DEVICE))
            layer.running_var.copy_(d["running_var"].to(DEVICE))

    for i, layer in enumerate(net.layers):
        if isinstance(layer, ResBlock):
            for j, sub in enumerate(layer._learnable):
                key = f"layer_{i}_sub_{j}_{type(sub).__name__}"
                if key in state:
                    _restore(sub, state[key])
        else:
            key = f"layer_{i}_{type(layer).__name__}"
            if key in state:
                _restore(layer, state[key])

    return ck["epoch"]


# ======================================================================
#  Thermodynamic σ_v calibration (Section 5 of theory doc)
# ======================================================================

def calibrate_sigma_v(net, x_probe, tau=1.0, batch_size=64):
    net.eval()

    Sa_accum = 0.0
    n_total = 0
    for i in range(0, len(x_probe), batch_size):
        xb = x_probe[i:i + batch_size]
        ma, Sa = net.forward(xb)
        Sa_accum += Sa.sum().item()
        n_total += Sa.numel()
        del ma, Sa
    mean_Sa = Sa_accum / max(n_total, 1)
    sigma_v = math.sqrt(max(tau * mean_Sa, 1e-12))
    K = mean_Sa / (mean_Sa + sigma_v ** 2)
    print(f"  σ_v calibration: E[S_a(Remax)]={mean_Sa:.6f}, τ={tau}, "
          f"σ_v*={sigma_v:.6f}, K={K:.3f}")
    return sigma_v


# ======================================================================
#  GPU-batched augmentation (no CPU round-trip)
# ======================================================================

def gpu_augment(x: torch.Tensor, cutout_size: int = 8) -> torch.Tensor:
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

    if cutout_size > 0:
        cy = torch.randint(0, H, (B,), device=x.device)
        cx = torch.randint(0, W, (B,), device=x.device)
        y1 = (cy - cutout_size // 2).clamp(0, H)
        y2 = (cy + cutout_size // 2).clamp(0, H)
        x1 = (cx - cutout_size // 2).clamp(0, W)
        x2 = (cx + cutout_size // 2).clamp(0, W)
        ys = torch.arange(H, device=x.device).unsqueeze(0)
        xs = torch.arange(W, device=x.device).unsqueeze(0)
        mask_y = (ys >= y1[:, None]) & (ys < y2[:, None])
        mask_x = (xs >= x1[:, None]) & (xs < x2[:, None])
        mask = ~(mask_y[:, :, None] & mask_x[:, None, :])
        x_crop = x_crop * mask[:, None, :, :].float()

    return x_crop


# ======================================================================
#  Training loop
# ======================================================================

def train(net, x_train, y_train_oh, y_train_labels, x_test, y_test_labels,
          batch_size, initial_sigma_v, n_epochs,
          monitor: TAGIMonitor = None,
          monitor_every: int = 5,
          decay_factor: float = 0.95,
          min_sigma_v: float = 0.01,
          label_smooth: float = 0.1,
          checkpoint_dir: str = "run_logs_cifar100/checkpoints",
          start_epoch: int = 1):
    num_classes = y_train_oh.shape[1]
    print(f"\n  {'Epoch':>5s}  {'Train':>7s}  {'Test':>7s}  {'ECE':>6s}  {'σ_v':>7s}  {'Time':>8s}")
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
            net.step(xb, yb, current_sigma_v)

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
    parser = argparse.ArgumentParser(description="Train TAGI ResNet-18 on CIFAR-100")
    resume_grp = parser.add_mutually_exclusive_group()
    resume_grp.add_argument("--resume", metavar="CHECKPOINT",
                            help="Resume from a specific .pt checkpoint file")
    resume_grp.add_argument("--resume-latest", action="store_true",
                            help="Resume from the latest checkpoint in run_logs_cifar100/checkpoints/")
    parser.add_argument("--sigma-v", type=float, default=None,
                        help="Override σ_v instead of auto-calibrating (useful when resuming)")
    args = parser.parse_args()

    print("=" * 62)
    print("  CIFAR-100 Classification — TAGI-Triton ResNet-18")
    print("  (exact replica of cuTAGI's ResNetBlock logic)")
    print("=" * 62)
    print("  Stem:     Conv(3→64, 3×3) → ReLU → BN")
    print("  ResBlock: Conv3×3→ReLU→BN→Conv3×3→ReLU→BN + shortcut")
    print("  Layer 1:  ResBlock(64→64) × 2   [identity]")
    print("  Layer 2:  ResBlock(64→128,s=2)  [proj: Conv2×2] + ResBlock(128)")
    print("  Layer 3:  ResBlock(128→256,s=2) [proj: Conv2×2] + ResBlock(256)")
    print("  Layer 4:  ResBlock(256→512,s=2) [proj: Conv2×2] + ResBlock(512)")
    print("  Head:     AvgPool(4) → Flatten → FC(512→100) → Remax")
    print("=" * 62)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ──
    print("\n  Loading CIFAR-100...", flush=True)
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels = load_cifar100()
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    print(f"  Input shape: {x_train.shape[1:]}")

    # ── Build ResNet-18, then apply residual-aware init ──
    eta = 0.125
    net = build_resnet18(
        num_classes=NUM_CLASSES, head="remax", device=DEVICE,
        g_min=0.1, g_max=0.1,
    )
    init_residual_aware(net, eta=eta, verbose=True)

    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    # ── Warm-up pass: trigger BN data-dependent initialization ──
    print("\n  Initialising BN layers (single forward pass)...", flush=True)
    net.train()
    net.forward(x_train[:32])
    net.eval()
    torch.cuda.empty_cache()

    # ── Resume from checkpoint (if requested) ──
    start_epoch = 1
    if args.resume or args.resume_latest:
        ckpt_path = "latest" if args.resume_latest else args.resume
        resumed_epoch = load_checkpoint(net, ckpt_path)
        start_epoch = resumed_epoch + 1
        print(f"  Resuming from epoch {resumed_epoch} → starting epoch {start_epoch}")

    # ── Monitor setup ──
    monitor = TAGIMonitor(net, log_dir="run_logs_cifar100", probe_size=256)
    net.eval()
    monitor.record(epoch=start_epoch - 1, x_probe=x_train[:256],
                   tag="resumed" if start_epoch > 1 else "init")
    monitor.print_report()

    # ── Calibrate σ_v ──
    if args.sigma_v is not None:
        sigma_v = args.sigma_v
        print(f"\n  Using provided σ_v = {sigma_v:.6f}")
    else:
        print("\n  Thermodynamic σ_v calibration (Section 5):")
        torch.cuda.empty_cache()
        sigma_v = calibrate_sigma_v(net, x_train[:256], tau=1.0, batch_size=32)

    # ── Hyperparameters ──
    batch_size      = 128
    n_epochs        = 100
    monitor_every   = 5
    label_smooth    = 0.0
    checkpoint_dir  = "run_logs_cifar100/checkpoints"
    decay_factor    = 1.0
    min_sigma_v     = 0.15 * sigma_v

    print(f"\n  Batch size     : {batch_size}")
    print(f"  Start epoch    : {start_epoch}")
    print(f"  Initial σ_v    : {sigma_v:.4f}")
    print(f"  Decay factor   : {decay_factor} (starts after 5 epochs)")
    print(f"  Min σ_v        : {min_sigma_v:.4f}")
    print(f"  Epochs         : {n_epochs}")
    print(f"  Label smoothing: {label_smooth}")
    print(f"  Monitor every  : {monitor_every} epoch(s)")
    print(f"  Checkpoint dir : {checkpoint_dir}")

    # ── Train ──
    best_acc = train(
        net, x_train, y_train_oh, y_train_labels,
        x_test, y_test_labels,
        batch_size, sigma_v, n_epochs,
        monitor=monitor, monitor_every=monitor_every,
        decay_factor=decay_factor,
        min_sigma_v=min_sigma_v,
        label_smooth=label_smooth,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
    )

    # ── Save logs ──
    monitor.plot("run_logs_cifar100/resnet18_monitor.png")
    monitor.save_csv("run_logs_cifar100/resnet18_monitor.csv")


if __name__ == "__main__":
    main()
