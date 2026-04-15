"""
CIFAR-10 Classification with TAGI-Triton ResNet-18 + FRN
=========================================================
Same architecture as run_resnet18.py but with FRN + TLU replacing
BatchNorm + ReLU throughout, following:

  "Filter Response Normalization Layer: Eliminating Batch Dependence
   in the Training of Deep Neural Networks" (Singh & Davis, 2020)

Architecture:
  Stem:     Conv(3→64, 3×3, s=1, pad=1) → FRN(64) → TLU(64)
  Layer 1:  FRNResBlock(64→64)   × 2         (identity shortcuts)
  Layer 2:  FRNResBlock(64→128, s=2) + FRNResBlock(128→128)
  Layer 3:  FRNResBlock(128→256, s=2) + FRNResBlock(256→256)
  Layer 4:  FRNResBlock(256→512, s=2) + FRNResBlock(512→512)
  Head:     AvgPool(4×4) → Flatten → FC(512→10) → Remax

  Each FRNResBlock main path:  Conv3×3→FRN→TLU→Conv3×3→FRN→TLU
  Projection shortcut:          Conv2×2(stride=2)→FRN→TLU
  NO activation after the residual addition.

Key differences from BatchNorm variant:
  - No batch statistics → no EMA, no train/eval mode divergence
  - No warm-up forward pass needed (FRN has no running stats)
  - TLU has learnable threshold τ (starts at 0, equivalent to ReLU)
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

from triton_tagi import Sequential
from triton_tagi.layers import (Linear, Conv2D, Flatten, Remax,
                        AvgPool2D, Bernoulli,
                        FRN2D, TLU, FRNResBlock)
from triton_tagi.monitor import TAGIMonitor
from triton_tagi.init import init_residual_aware

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
#  Depth-scaled gain function
# ======================================================================

def depth_gain(layer_idx: int, total_layers: int,
               g_min: float = 0.1, g_max: float = 0.5) -> float:
    """
    Compute a depth-dependent gain factor g(l).
    Parabolic profile: g_min at stem/head, g_max at deepest block.
    """
    if total_layers <= 1:
        return g_min
    t = layer_idx / (total_layers - 1)
    g = g_min + (g_max - g_min) * 4.0 * t * (1.0 - t)
    return g


# ======================================================================
#  Network builder — ResNet-18 with FRN + TLU
# ======================================================================

def build_resnet18_frn(num_classes=10, head="remax", device="cuda",
                       gain_w=1.0, gain_b=1.0,
                       g_min=0.1, g_max=0.5):
    """
    Build a ResNet-18 for CIFAR-10 using FRN + TLU instead of BN + ReLU.

    Architecture:
        Stem:     Conv(3→64, 3×3) → FRN(64) → TLU(64)
        Layer 1:  [FRNResBlock(64→64)]   × 2       (identity shortcuts)
        Layer 2:  FRNResBlock(64→128, s=2) + FRNResBlock(128→128)
        Layer 3:  FRNResBlock(128→256, s=2) + FRNResBlock(256→256)
        Layer 4:  FRNResBlock(256→512, s=2) + FRNResBlock(512→512)
        Head:     AvgPool(4) → Flatten → FC(512→num_classes) → Remax
    """
    # stem_conv(0), stem_frn(1), 8 FRNResBlocks(2..9), head_fc(10) = 11 slots
    N_SLOTS = 11

    def g(slot):
        return depth_gain(slot, N_SLOTS, g_min, g_max)

    layers = [
        # ── Stem: Conv(3→64) → FRN → TLU  (slots 0, 1) ──
        Conv2D(3, 64, kernel_size=3, stride=1, padding=1,
               device=device, gain_w=g(0), gain_b=g(0)),
        FRN2D(64, device=device, gain_w=g(1), gain_b=g(1)),
        TLU(64, device=device),

        # ── Layer 1: 64 → 64, identity shortcuts (slots 2, 3) ──
        FRNResBlock(64, 64, stride=1,
                    device=device, gain_w=g(2), gain_b=g(2)),
        FRNResBlock(64, 64, stride=1,
                    device=device, gain_w=g(3), gain_b=g(3)),

        # ── Layer 2: 64 → 128, projection shortcut (slots 4, 5) ──
        FRNResBlock(64, 128, stride=2,
                    device=device, gain_w=g(4), gain_b=g(4)),
        FRNResBlock(128, 128, stride=1,
                    device=device, gain_w=g(5), gain_b=g(5)),

        # ── Layer 3: 128 → 256 (slots 6, 7) ──
        FRNResBlock(128, 256, stride=2,
                    device=device, gain_w=g(6), gain_b=g(6)),
        FRNResBlock(256, 256, stride=1,
                    device=device, gain_w=g(7), gain_b=g(7)),

        # ── Layer 4: 256 → 512 (slots 8, 9) ──
        FRNResBlock(256, 512, stride=2,
                    device=device, gain_w=g(8), gain_b=g(8)),
        FRNResBlock(512, 512, stride=1,
                    device=device, gain_w=g(9), gain_b=g(9)),

        # ── Head: AvgPool(4) → Flatten → FC(512→num_classes) (slot 10) ──
        AvgPool2D(4),
        Flatten(),
        Linear(512, num_classes,
               device=device, gain_w=g(10), gain_b=g(10)),
    ]

    # Output head
    if head == "remax":
        layers.append(Remax())
    elif head == "bernoulli":
        layers.append(Bernoulli())
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

    # One-hot for Remax
    y_train_oh = torch.zeros(len(y_train), 10, device=DEVICE)
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 1.0)

    return x_train, y_train_oh, y_train, x_test, y_test


# ======================================================================
#  Evaluation
# ======================================================================

def evaluate(net, x_test, y_labels, batch_size=128):
    """Evaluate accuracy on a dataset."""
    net.eval()
    correct = 0
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        lb = y_labels[i:i + batch_size]
        mu, _ = net.forward(xb)
        correct += (mu.argmax(dim=1) == lb).sum().item()
    return correct / len(x_test)


def compute_ece(net, x_test, y_labels, n_bins=15, batch_size=128):
    """Expected Calibration Error."""
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
    """Save all learnable parameters."""
    os.makedirs(save_dir, exist_ok=True)
    state = {}

    def _layer_state(layer):
        d = {}
        if hasattr(layer, "mw"):
            d["mw"] = layer.mw.cpu().clone()
            d["Sw"] = layer.Sw.cpu().clone()
            if hasattr(layer, 'has_bias') and layer.has_bias:
                d["mb"] = layer.mb.cpu().clone()
                d["Sb"] = layer.Sb.cpu().clone()
        return d

    for i, layer in enumerate(net.layers):
        if isinstance(layer, FRNResBlock):
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
    """Restore all learnable parameters from a checkpoint."""
    import glob as _glob

    if path == "latest":
        candidates = sorted(_glob.glob(
            os.path.join("run_logs_frn", "checkpoints", "checkpoint_epoch_*.pt")))
        if not candidates:
            raise FileNotFoundError("No checkpoints found in run_logs_frn/checkpoints/")
        path = candidates[-1]

    print(f"  Loading checkpoint: {path}")
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ck["state"]

    def _restore(layer, d):
        if "mw" in d:
            layer.mw.data.copy_(d["mw"].to(DEVICE))
            layer.Sw.data.copy_(d["Sw"].to(DEVICE))
        if "mb" in d:
            layer.mb.data.copy_(d["mb"].to(DEVICE))
            layer.Sb.data.copy_(d["Sb"].to(DEVICE))

    for i, layer in enumerate(net.layers):
        if isinstance(layer, FRNResBlock):
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
#  Thermodynamic σ_v calibration
# ======================================================================

def calibrate_sigma_v(net, x_probe, tau=1.0, batch_size=64):
    """
    Calibrate observation noise from the network's initial predictive variance.
    σ_v* = sqrt(τ · E[S_a])
    """
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
#  GPU-batched augmentation
# ======================================================================

def gpu_augment(x: torch.Tensor, cutout_size: int = 8) -> torch.Tensor:
    """Random horizontal flip + random crop + cutout, fully on GPU."""
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
          checkpoint_dir: str = "run_logs_frn/checkpoints",
          start_epoch: int = 1):
    """TAGI training loop with observation noise annealing."""
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

        # Save checkpoint
        save_checkpoint(net, epoch, checkpoint_dir)

        # Monitor snapshot
        if monitor is not None and epoch % monitor_every == 0:
            net.eval()
            monitor.record(epoch, x_probe, tag=f"acc={acc*100:.1f}%")
            monitor.print_report()

        # Observation noise decay (skip warmup)
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
        description="Train TAGI ResNet-18 with FRN+TLU on CIFAR-10")
    resume_grp = parser.add_mutually_exclusive_group()
    resume_grp.add_argument("--resume", metavar="CHECKPOINT",
                            help="Resume from a specific .pt checkpoint file")
    resume_grp.add_argument("--resume-latest", action="store_true",
                            help="Resume from the latest checkpoint")
    parser.add_argument("--sigma-v", type=float, default=None,
                        help="Override σ_v instead of auto-calibrating")
    args = parser.parse_args()

    print("=" * 62)
    print("  CIFAR-10 Classification — TAGI-Triton ResNet-18 + FRN")
    print("  (FRN + TLU replacing BatchNorm + ReLU)")
    print("=" * 62)
    print("  Stem:          Conv(3→64, 3×3) → FRN → TLU")
    print("  FRNResBlock:   Conv3×3→FRN→TLU→Conv3×3→FRN→TLU + shortcut")
    print("  Layer 1:       FRNResBlock(64→64) × 2   [identity]")
    print("  Layer 2:       FRNResBlock(64→128,s=2) + FRNResBlock(128)")
    print("  Layer 3:       FRNResBlock(128→256,s=2) + FRNResBlock(256)")
    print("  Layer 4:       FRNResBlock(256→512,s=2) + FRNResBlock(512)")
    print("  Head:          AvgPool(4) → Flatten → FC(512→10) → Remax")
    print("=" * 62)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ──
    print("\n  Loading CIFAR-10...", flush=True)
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels = load_cifar10()
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    print(f"  Input shape: {x_train.shape[1:]}")

    # ── Build ResNet-18 with FRN ──
    eta = 0.5
    net = build_resnet18_frn(
        num_classes=10, head="remax", device=DEVICE,
        g_min=0.1, g_max=0.1,
    )
    init_residual_aware(net, eta=eta, verbose=True)

    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    # ── Warm-up pass: trigger FRN data-dependent initialization ──
    # FRN doesn't center like BN, so on the first forward pass each FRN
    # sets β = −γ · E[μ_hat] to center the output and prevent positive
    # bias accumulation from TLU through the ResBlocks.
    print("\n  Initialising FRN layers (single forward pass)...", flush=True)
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
    monitor = TAGIMonitor(net, log_dir="run_logs_frn", probe_size=256)
    net.eval()
    monitor.record(epoch=start_epoch - 1, x_probe=x_train[:256],
                   tag="resumed" if start_epoch > 1 else "init")
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
    batch_size      = 128
    n_epochs        = 100
    monitor_every   = 10
    label_smooth    = 0.0
    checkpoint_dir  = "run_logs_frn/checkpoints"
    decay_factor    = 0.90
    min_sigma_v     = 0.05 * sigma_v

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
    monitor.plot("run_logs_frn/resnet18_frn_monitor.png")
    monitor.save_csv("run_logs_frn/resnet18_frn_monitor.csv")


if __name__ == "__main__":
    main()
