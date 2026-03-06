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
from src.layers import Linear, ReLU, Conv2D, Flatten, Remax, Bernoulli, BatchNorm2D, AvgPool2D, LeakyReLU
from src.monitor import TAGIMonitor, sweep_gains

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
        # LeakyReLU(alpha=0.01),
        ReLU(),
        BatchNorm2D(32, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),  # 32x32 -> 16x16
        
        # Block 2
        Conv2D(32, 64, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        # LeakyReLU(alpha=0.01),
        ReLU(),
        BatchNorm2D(64, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),  # 16x16 -> 8x8
        
        # Block 3
        Conv2D(64, 64, 5, stride=1, padding=2, device=device,
               gain_w=gain_w, gain_b=gain_b),
        # LeakyReLU(alpha=0.01),
        ReLU(),
        BatchNorm2D(64, device=device, gain_w=gain_w, gain_b=gain_b),
        AvgPool2D(2),  # 8x8 -> 4x4
        
        # Classifier
        Flatten(),
        Linear(64 * 4 * 4, 256, device=device,
               gain_w=gain_w, gain_b=gain_b),
        # LeakyReLU(alpha=0.01),
        ReLU(),
        Linear(256, num_classes, device=device,
               gain_w=gain_w, gain_b=gain_b),
    ]

    # Output head
    if head == "remax":
        layers.append(Remax())
    elif head == "bernoulli":
        layers.append(Bernoulli(n_gh=32))
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

    # Train: no augmentation here — augmentation is applied on-the-fly per batch
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Test: clean (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True,
                                transform=train_transform)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=test_transform)

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
    flip = torch.rand(B, device=x.device) < 0.5          # (B,)
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)

    # ── Reflect-pad the whole batch in one call: (B, C, H+2p, W+2p) ──
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")

    # ── Sample one (top, left) crop offset per image ──
    top  = torch.randint(0, 2 * pad, (B,), device=x.device)  # shape (B,)
    left = torch.randint(0, 2 * pad, (B,), device=x.device)

    # ── Vectorised crop via advanced indexing ──
    # row/col indices for each image: (B, H) and (B, W)
    rows = top.unsqueeze(1)  + torch.arange(H, device=x.device).unsqueeze(0)   # (B, H)
    cols = left.unsqueeze(1) + torch.arange(W, device=x.device).unsqueeze(0)   # (B, W)

    # Gather rows then cols: x_pad[b, c, rows[b], :] then [:, cols[b]]
    # Use expand + gather for a fully batched, no-loop extract
    x_crop = x_pad[
        torch.arange(B, device=x.device)[:, None, None, None],   # (B,1,1,1)
        torch.arange(C, device=x.device)[None, :, None, None],   # (1,C,1,1)
        rows[:, None, :, None].expand(B, C, H, W),                # (B,C,H,W)
        cols[:, None, None, :].expand(B, C, H, W),                # (B,C,H,W)
    ]
    return x_crop


# ======================================================================
#  Training
# ======================================================================

def train(net, x_train, y_train_oh, y_train_labels, x_test, y_test_labels,
          batch_size, initial_sigma_v, n_epochs,
          monitor: TAGIMonitor = None,
          monitor_every: int = 1,
          anneal_rate: float = 1.05): # Added anneal rate parameter
    """
    monitor_every : record a monitor snapshot every N epochs (default 1).
    anneal_rate   : Multiplier to increase sigma_v per epoch. (e.g., 1.05 = +5%/epoch).
                    This simulates a learning rate decay.
    """
    print(f"\n  {'Epoch':>5s}  {'Train':>7s}  {'Test':>7s}  {'σ_v':>7s}  {'Time':>8s}")
    print("  " + "─" * 45)

    best_acc = 0.0
    t_total = time.perf_counter()
    x_probe = x_train[:256]   # fixed probe batch reused every epoch
    
    # Initialize the dynamic observation noise
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
            
            # Pass the current, dynamically annealed sigma_v to the step
            net.step(xb, yb, current_sigma_v)
            
            if monitor is not None:
                monitor.count_step()

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        train_acc = evaluate(net, x_train, y_train_labels)
        acc = evaluate(net, x_test, y_test_labels)
        best_acc = max(best_acc, acc)
        
        print(f"  {epoch:5d}  {train_acc*100:6.2f}%  {acc*100:6.2f}%  {current_sigma_v:7.3f}  {dt:7.2f}s")

        # ── monitor snapshot ──
        if monitor is not None and epoch % monitor_every == 0:
            net.eval()
            monitor.record(epoch, x_probe, tag=f"acc={acc*100:.1f}%")
            monitor.print_report()

        # ── EXPONENTIAL NOISE ANNEALING ──
        # Slowly increase the observation noise to reduce the Kalman Gain (simulates LR decay)
        # Note: We only start annealing after epoch 20 to allow the network to settle first.
        if epoch > 20:
            current_sigma_v *= anneal_rate

    total_time = time.perf_counter() - t_total
    print("  " + "─" * 45)
    print(f"  Best accuracy : {best_acc*100:.2f}%")
    print(f"  Total time    : {total_time:.1f}s")


# ======================================================================
#  Main
# ======================================================================

def _bayesian_relu(mz, Sz):
    Sz_safe = torch.clamp(Sz, min=1e-12)
    sigma_z = torch.sqrt(Sz_safe)
    alpha = mz / sigma_z
    pdf = torch.exp(-0.5 * alpha ** 2) * 0.3989422804014327
    cdf = 0.5 * (1.0 + torch.erf(alpha * 0.7071067811865476))
    mu_m = sigma_z * pdf + mz * cdf
    var_m = torch.clamp(
        -mu_m ** 2 + 2 * mu_m * mz - mz * sigma_z * pdf
        + (Sz_safe - mz ** 2) * cdf, min=1e-12)
    return mu_m, var_m

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
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels = load_cifar10()
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    print(f"  Input shape: {x_train.shape[1:]}")

    # ── Build Simple 3-CNN ──
    # Note: gain_w and gain_b are completely ignored by the theoretical init, 
    # but we pass 1.0 just to satisfy the class signature.
    net = build_simple_3cnn(
        num_classes=10, head="remax", device=DEVICE,
        gain_w=1.0, gain_b=1.0,
    )

    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    # ── Monitor: snapshot at init ──
    monitor = TAGIMonitor(net, log_dir="run_logs", probe_size=256)
    net.eval()
    monitor.record(epoch=0, x_probe=x_train[:256], tag="init")
    monitor.print_report()

    # ── 2. Auto-Calibrate Observation Noise (sigma_v) ──
    print("\n  Auto-Calibrating Observation Noise...")
    net.eval() # Ensure we don't update batchnorm stats during calibration
    x_calib = x_train[:256]
    
    # Pass data through the network up to the layer BEFORE Remax
    mu_a = x_calib.clone()
    var_a = torch.full_like(x_calib, 1e-4) # Small input uncertainty
    
    # We slice [:-1] to skip the Remax layer and get raw logits
    for layer in net.layers[:-1]: 
        if isinstance(layer, (ReLU, Flatten, AvgPool2D, BatchNorm2D)):
             # These layers don't use weights, handle routing
             if isinstance(layer, ReLU):
                 flat_mu = mu_a.view(mu_a.shape[0], -1)
                 flat_var = var_a.view(var_a.shape[0], -1)
                 mu_a_flat, var_a_flat = _bayesian_relu(flat_mu, flat_var)
                 mu_a = mu_a_flat.view(mu_a.shape)
                 var_a = var_a_flat.view(var_a.shape)
             elif isinstance(layer, Flatten):
                 mu_a = mu_a.view(mu_a.shape[0], -1)
                 var_a = var_a.view(var_a.shape[0], -1)
             elif isinstance(layer, (AvgPool2D, BatchNorm2D)):
                 mu_a, var_a = layer.forward(mu_a, var_a)
        else:
             # Parametrized layers
             mu_a, var_a = layer.forward(mu_a, var_a)

    # Measure the global variance of the logits
    empirical_var = mu_a.var().item()
    
    # Target a Kalman Gain roughly equivalent to a 10% learning rate (alpha_eff = 0.1)
    # 0.1 = var / (var + sigma_v^2) => sigma_v^2 = 9 * var
    optimal_sigma_v = math.sqrt(9.0 * empirical_var)
    
    print("  " + "─" * 45)
    print(f"  Empirical Logit Variance: {empirical_var:.4f}")
    print(f"  Calculated optimal σ_v  : {optimal_sigma_v:.4f}")
    print("  " + "─" * 45)

    # ── Hyperparameters ──
    batch_size    = 128
    # sigma_v       = optimal_sigma_v # Use the math!
    sigma_v = 0.05
    n_epochs      = 50
    monitor_every = 5    
    anneal_rate   = 1.05 # Increase noise by 5% every epoch after epoch 20

    print(f"\n  Batch size     : {batch_size}")
    print(f"  Initial σ_v    : {sigma_v:.4f}")
    print(f"  Anneal Rate    : {anneal_rate} (starts after epoch 20)")
    print(f"  Epochs         : {n_epochs}")
    print(f"  Monitor every  : {monitor_every} epoch(s)")

    train(net, x_train, y_train_oh, y_train_labels, x_test, y_test_labels,
          batch_size, sigma_v, n_epochs,
          monitor=monitor, monitor_every=monitor_every, 
          anneal_rate=anneal_rate)

    # ── Final plots and CSV ──
    monitor.plot("run_logs/monitor.png")
    monitor.save_csv("run_logs/monitor.csv")

if __name__ == "__main__":
    main()