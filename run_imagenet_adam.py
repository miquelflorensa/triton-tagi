"""
ImageNet (ILSVRC-2012) Classification with Adam-TAGI ResNet-18
===============================================================
Architecture — standard ResNet-18 adapted for 224×224 input:

  Stem:     Conv(3→64, 7×7, s=2, pad=3) → ReLU → BN(64) → AvgPool(2)
  Layer 1:  ResBlock(64→64)   × 2         (identity shortcuts)    56×56
  Layer 2:  ResBlock(64→128, s=2) + ResBlock(128→128)             28×28
  Layer 3:  ResBlock(128→256, s=2) + ResBlock(256→256)            14×14
  Layer 4:  ResBlock(256→512, s=2) + ResBlock(512→512)             7×7
  Head:     AvgPool(7) → Flatten → FC(512→1000) → Remax

  Input: 224×224 → 112×112 → 56×56 → 28×28 → 14×14 → 7×7 → 1×1 → 1000

Usage:
    python run_imagenet_adam.py
    python run_imagenet_adam.py --resume-latest
    python run_imagenet_adam.py --sigma-v 0.03 --beta1 0.9 --eps-Q 1e-9
"""

import math
import sys
import os
import multiprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Prevent DataLoader worker segfaults (fork + CUDA = bad)
multiprocessing.set_start_method("spawn", force=True)

import torch
import torch.nn.functional as F
import numpy as np
import time
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src import Sequential
from src.layers import (Linear, ReLU, Conv2D, Flatten, Remax,
                        BatchNorm2D, AvgPool2D, ResBlock)
from src.optimizer import AdamTAGI
from src.monitor import TAGIMonitor
from src.init import init_residual_aware

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 1000
IMAGENET_DIR = "/usr/local/share/imagenet/ILSVRC/Data/CLS-LOC"


# ======================================================================
#  Depth-scaled gain
# ======================================================================

def depth_gain(layer_idx, total_layers, g_min=0.1, g_max=0.5):
    if total_layers <= 1:
        return g_min
    t = layer_idx / (total_layers - 1)
    return g_min + (g_max - g_min) * 4.0 * t * (1.0 - t)


# ======================================================================
#  Network builder — ImageNet ResNet-18
# ======================================================================

def build_resnet18_imagenet(num_classes=NUM_CLASSES, head="remax",
                            device="cuda", g_min=0.1, g_max=0.1):
    """
    ResNet-18 for ImageNet (224×224 input).

    Differences from CIFAR ResNet-18:
        - Stem uses 7×7 conv with stride 2 (not 3×3 stride 1)
        - AvgPool(2) after stem replaces MaxPool (not available in TAGI)
        - AvgPool(7) before head (not AvgPool(4))
        - FC outputs 1000 classes
    """
    # 12 slots: stem_conv(0), stem_bn(1), 8 ResBlocks(2..9), head_fc(10)
    # +1 for the stem pool conceptually, but pool has no params
    N_SLOTS = 11

    def g(slot):
        return depth_gain(slot, N_SLOTS, g_min, g_max)

    layers = [
        # ── Stem: Conv(3→64, 7×7, s=2, p=3) → ReLU → BN → AvgPool(2) ──
        # 224×224 → 112×112 → 56×56
        Conv2D(3, 64, kernel_size=7, stride=2, padding=3,
               device=device, gain_w=g(0), gain_b=g(0)),
        ReLU(),
        BatchNorm2D(64, device=device, gain_w=g(1), gain_b=g(1)),
        AvgPool2D(2),  # replaces MaxPool(3, s=2, p=1) — 112→56

        # ── Layer 1: 64 → 64, identity shortcuts ── 56×56
        ResBlock(64, 64, stride=1,
                 device=device, gain_w=g(2), gain_b=g(2)),
        ResBlock(64, 64, stride=1,
                 device=device, gain_w=g(3), gain_b=g(3)),

        # ── Layer 2: 64 → 128 ── 56→28
        ResBlock(64, 128, stride=2,
                 device=device, gain_w=g(4), gain_b=g(4)),
        ResBlock(128, 128, stride=1,
                 device=device, gain_w=g(5), gain_b=g(5)),

        # ── Layer 3: 128 → 256 ── 28→14
        ResBlock(128, 256, stride=2,
                 device=device, gain_w=g(6), gain_b=g(6)),
        ResBlock(256, 256, stride=1,
                 device=device, gain_w=g(7), gain_b=g(7)),

        # ── Layer 4: 256 → 512 ── 14→7
        ResBlock(256, 512, stride=2,
                 device=device, gain_w=g(8), gain_b=g(8)),
        ResBlock(512, 512, stride=1,
                 device=device, gain_w=g(9), gain_b=g(9)),

        # ── Head: AvgPool(7) → Flatten → FC(512→1000) ──
        AvgPool2D(7),
        Flatten(),
        Linear(512, num_classes,
               device=device, gain_w=g(10), gain_b=g(10)),
    ]

    if head == "remax":
        layers.append(Remax())
    elif head in ("none", "logit"):
        pass
    else:
        raise ValueError(f"Unknown head: {head}")

    return Sequential(layers, device=device)


# ======================================================================
#  Data loading (streaming via DataLoader)
# ======================================================================

def build_dataloaders(data_dir=IMAGENET_DIR, batch_size=64, num_workers=8):
    """Build train and val DataLoaders for ImageNet."""
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=min(num_workers, 4), pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader


# ======================================================================
#  Evaluation
# ======================================================================

def evaluate(net, val_loader):
    """Compute top-1 and top-5 accuracy on validation set."""
    net.eval()
    correct1 = 0
    correct5 = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            mu, _ = net.forward(xb)

            # Top-1
            correct1 += (mu.argmax(dim=1) == yb).sum().item()

            # Top-5
            _, top5_pred = mu.topk(5, dim=1)
            correct5 += (top5_pred == yb.unsqueeze(1)).any(dim=1).sum().item()

            total += yb.size(0)

    return correct1 / total, correct5 / total


# ======================================================================
#  Checkpointing
# ======================================================================

def save_checkpoint(net, opt, epoch, save_dir):
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

    opt_state = {
        't': opt.t,
        'ema': {str(k): {'ema_m': v['ema_m'].cpu().clone(),
                          'ema_v': v['ema_v'].cpu().clone()}
                for k, v in opt._states.items()},
    }

    path = os.path.join(save_dir, f"checkpoint_epoch_{epoch:04d}.pt")
    torch.save({"epoch": epoch, "state": state, "opt_state": opt_state}, path)
    return path


def load_checkpoint(net, opt, path):
    import glob as _glob
    if path == "latest":
        candidates = sorted(_glob.glob(
            os.path.join("run_logs_imagenet", "checkpoints",
                         "checkpoint_epoch_*.pt")))
        if not candidates:
            raise FileNotFoundError("No checkpoints found")
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

    if "opt_state" in ck:
        opt.t = ck["opt_state"]["t"]
        for layer in opt._learnable_list:
            for attr in ('mw', 'mb'):
                if attr == 'mb' and not getattr(layer, 'has_bias', True):
                    continue
                opt._get_state(layer, attr)
        saved_ema = ck["opt_state"]["ema"]
        for k, v in opt._states.items():
            sk = str(k)
            if sk in saved_ema:
                v['ema_m'].copy_(saved_ema[sk]['ema_m'].to(DEVICE))
                v['ema_v'].copy_(saved_ema[sk]['ema_v'].to(DEVICE))

    return ck["epoch"]


# ======================================================================
#  σ_v calibration
# ======================================================================

def calibrate_sigma_v(net, train_loader, tau=1.0, n_batches=8):
    """Calibrate σ_v from a few forward passes on training data."""
    net.eval()
    Sa_accum = 0.0
    n_total = 0
    with torch.no_grad():
        for i, (xb, _) in enumerate(train_loader):
            if i >= n_batches:
                break
            xb = xb.to(DEVICE, non_blocking=True)
            ma, Sa = net.forward(xb)
            Sa_accum += Sa.sum().item()
            n_total += Sa.numel()
            del ma, Sa

    mean_Sa = Sa_accum / max(n_total, 1)
    sigma_v = math.sqrt(max(tau * mean_Sa, 1e-12))
    K = mean_Sa / (mean_Sa + sigma_v ** 2)
    print(f"  σ_v calibration: E[S_a]={mean_Sa:.6f}, τ={tau}, "
          f"σ_v*={sigma_v:.6f}, K={K:.3f}")
    return sigma_v


# ======================================================================
#  Training loop (streaming, uses AdamTAGI)
# ======================================================================

def train(opt, train_loader, val_loader,
          initial_sigma_v, n_epochs,
          monitor=None, monitor_every=5,
          decay_factor=1.0, min_sigma_v=0.01,
          label_smooth=0.1,
          checkpoint_dir="run_logs_imagenet/checkpoints",
          start_epoch=1):

    net = opt.net
    num_classes = NUM_CLASSES
    n_batches = len(train_loader)

    print(f"\n  {'Epoch':>5s}  {'Train':>7s}  {'Top1':>7s}  {'Top5':>7s}  "
          f"{'σ_v':>7s}  {'Time':>8s}")
    print("  " + "─" * 57)

    best_top1 = 0.0
    t_total = time.perf_counter()
    current_sigma_v = initial_sigma_v

    for epoch in range(start_epoch, start_epoch + n_epochs):
        t0 = time.perf_counter()

        # ── Train one epoch ──
        net.train()
        train_correct = 0
        train_total = 0
        log_interval = max(1, n_batches // 20)  # ~20 updates per epoch

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            # One-hot targets with optional label smoothing
            yb_oh = torch.zeros(yb.size(0), num_classes, device=DEVICE)
            yb_oh.scatter_(1, yb.unsqueeze(1), 1.0)
            if label_smooth > 0.0:
                yb_oh = (1.0 - label_smooth) * yb_oh + label_smooth / num_classes

            # Adam-TAGI step
            mu_pred, _ = opt.step(xb, yb_oh, current_sigma_v)

            train_correct += (mu_pred.argmax(dim=1) == yb).sum().item()
            train_total += yb.size(0)

            if monitor is not None:
                monitor.count_step()

            # Progress line with running acc, speed, and ETA
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.perf_counter() - t0
                imgs_sec = train_total / elapsed
                pct = (batch_idx + 1) / n_batches
                eta = elapsed / pct * (1 - pct)
                acc_so_far = train_correct / train_total * 100
                print(f"\r    [{batch_idx+1:5d}/{n_batches}] "
                      f"acc={acc_so_far:5.1f}%  "
                      f"{imgs_sec:5.0f} img/s  "
                      f"ETA {eta:4.0f}s", end="", flush=True)

        print("\r" + " " * 70 + "\r", end="")  # clear progress line

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        train_acc = train_correct / train_total

        # ── Validation ──
        top1, top5 = evaluate(net, val_loader)
        best_top1 = max(best_top1, top1)

        print(f"  {epoch:5d}  {train_acc*100:6.2f}%  {top1*100:6.2f}%  "
              f"{top5*100:6.2f}%  {current_sigma_v:7.4f}  {dt:7.1f}s")

        # ── Checkpoint ──
        save_checkpoint(net, opt, epoch, checkpoint_dir)

        # ── Monitor ──
        if monitor is not None and epoch % monitor_every == 0:
            # Use a fixed probe batch for monitoring
            net.eval()
            probe_iter = iter(train_loader)
            x_probe = next(probe_iter)[0][:64].to(DEVICE)
            monitor.record(epoch, x_probe, tag=f"top1={top1*100:.1f}%")
            monitor.print_report()
            del x_probe

        # ── σ_v decay ──
        if epoch >= start_epoch + 5:
            current_sigma_v = max(current_sigma_v * decay_factor, min_sigma_v)

    total_time = time.perf_counter() - t_total
    print("  " + "─" * 57)
    print(f"  Best top-1     : {best_top1*100:.2f}%")
    print(f"  Total time     : {total_time/3600:.1f}h")
    return best_top1


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Adam-TAGI ResNet-18 on ImageNet")
    resume_grp = parser.add_mutually_exclusive_group()
    resume_grp.add_argument("--resume", metavar="CHECKPOINT")
    resume_grp.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--sigma-v", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps-Q", type=float, default=1e-9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default=IMAGENET_DIR)
    args = parser.parse_args()

    print("=" * 66)
    print("  ImageNet (ILSVRC-2012) — Adam-TAGI ResNet-18")
    print("=" * 66)
    print("  Stem:     Conv(3→64, 7×7, s=2) → ReLU → BN → AvgPool(2)")
    print("  Layer 1:  ResBlock(64→64) × 2    [identity]     56×56")
    print("  Layer 2:  ResBlock(64→128,s=2)  + ResBlock(128)  28×28")
    print("  Layer 3:  ResBlock(128→256,s=2) + ResBlock(256)  14×14")
    print("  Layer 4:  ResBlock(256→512,s=2) + ResBlock(512)   7×7")
    print("  Head:     AvgPool(7) → Flatten → FC(512→1000) → Remax")
    print("=" * 66)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")

    # ── Data loaders ──
    print(f"\n  Loading ImageNet from {args.data_dir}...")
    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    print(f"  Train: {len(train_loader.dataset):,} images "
          f"({len(train_loader)} batches × {args.batch_size})")
    print(f"  Val:   {len(val_loader.dataset):,} images")

    # ── Build ResNet-18 for ImageNet ──
    net = build_resnet18_imagenet(
        num_classes=NUM_CLASSES, head="remax", device=DEVICE,
        g_min=0.1, g_max=0.1,
    )
    init_residual_aware(net, eta=0.5, verbose=True)

    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    # ── Warm-up BN ──
    print("\n  Initialising BN layers (single forward pass)...", flush=True)
    net.train()
    warmup_batch = next(iter(train_loader))[0][:16].to(DEVICE)
    net.forward(warmup_batch)
    del warmup_batch
    net.eval()
    torch.cuda.empty_cache()

    # ── Create Adam-TAGI optimizer ──
    opt = AdamTAGI(net, beta1=args.beta1, beta2=args.beta2, eps_Q=args.eps_Q)
    print(f"\n  Optimizer: {opt}")

    # ── Resume ──
    start_epoch = 1
    if args.resume or args.resume_latest:
        ckpt_path = "latest" if args.resume_latest else args.resume
        resumed_epoch = load_checkpoint(net, opt, ckpt_path)
        start_epoch = resumed_epoch + 1
        print(f"  Resuming from epoch {resumed_epoch} → starting epoch {start_epoch}")

    # ── Monitor ──
    monitor = TAGIMonitor(net, log_dir="run_logs_imagenet", probe_size=64)

    # ── Calibrate σ_v ──
    if args.sigma_v is not None:
        sigma_v = args.sigma_v
        print(f"\n  Using provided σ_v = {sigma_v:.6f}")
    else:
        print("\n  Thermodynamic σ_v calibration:")
        torch.cuda.empty_cache()
        sigma_v = calibrate_sigma_v(net, train_loader, tau=1.0, n_batches=8)

    # ── Hyperparameters ──
    label_smooth    = 0.1
    checkpoint_dir  = "run_logs_imagenet/checkpoints"
    decay_factor    = 1.0     # constant σ_v for ImageNet
    min_sigma_v     = 0.15 * sigma_v
    monitor_every   = 5

    print(f"\n  Adam-TAGI config:")
    print(f"    β1={args.beta1}, β2={args.beta2}, ε_Q={args.eps_Q}")
    print(f"  Training config:")
    print(f"    batch_size={args.batch_size}, σ_v={sigma_v:.4f}")
    print(f"    decay_factor={decay_factor}")
    print(f"    epochs={args.epochs}, label_smooth={label_smooth}")
    print(f"    workers={args.workers}")
    print(f"    checkpoint_dir={checkpoint_dir}")

    # ── Train ──
    best_top1 = train(
        opt, train_loader, val_loader,
        sigma_v, args.epochs,
        monitor=monitor, monitor_every=monitor_every,
        decay_factor=decay_factor,
        min_sigma_v=min_sigma_v,
        label_smooth=label_smooth,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
    )

    # ── Save logs ──
    monitor.plot("run_logs_imagenet/imagenet_adam_monitor.png")
    monitor.save_csv("run_logs_imagenet/imagenet_adam_monitor.csv")


if __name__ == "__main__":
    main()
