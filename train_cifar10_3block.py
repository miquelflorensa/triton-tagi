"""
CIFAR-10 — 3-Block CNN with TAGI (Triton)
==========================================
Mirrors the cuTAGI build_3block_cnn architecture:

  Block 1:  Conv(3→64,  3×3, pad=1)        → 32×32
            Conv(64→64, 4×4, pad=1, s=2)   → 16×16
  Block 2:  Conv(64→128, 3×3, pad=1)       → 16×16
            Conv(128→128, 4×4, pad=1, s=2) →  8×8
  Block 3:  Conv(128→256, 3×3, pad=1)      →  8×8
            Conv(256→256, 4×4, pad=1, s=2) →  4×4
  Head:     Flatten(4096) → FC(512) → FC(10)

Stride-2 conv replaces max-pooling.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from torchvision import datasets
from tagi_cnn_triton import TritonTAGICNN

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda")


# ====================================================================
# Architecture
# ====================================================================

def build_3block_spec():
    """
    3-block CNN: stride-2 conv (k=4, pad=1) halves spatial dims.
    32 → 16 → 8 → 4   ⟹  256×4×4 = 4096 → FC(512) → FC(10)
    """
    return [
        # Block 1
        ('conv',   3,  64, 3, 1, 1),   # 3→64,   3×3, s=1, p=1  → 32×32
        ('relu',),
        ('conv',  64,  64, 4, 2, 1),   # 64→64,  4×4, s=2, p=1  → 16×16
        ('relu',),
        # Block 2
        ('conv',  64, 128, 3, 1, 1),   # 64→128, 3×3, s=1, p=1  → 16×16
        ('relu',),
        ('conv', 128, 128, 4, 2, 1),   # 128→128, 4×4, s=2, p=1 →  8×8
        ('relu',),
        # Block 3
        ('conv', 128, 256, 3, 1, 1),   # 128→256, 3×3, s=1, p=1 →  8×8
        ('relu',),
        ('conv', 256, 256, 4, 2, 1),   # 256→256, 4×4, s=2, p=1 →  4×4
        ('relu',),
        # Head
        ('flatten',),                   # 256×4×4 = 4096
        ('fc', 4096, 512),
        ('relu',),
        ('fc',  512,  10),
    ]


# ====================================================================
# Helpers
# ====================================================================

def augment_batch(x, pad=4):
    """Random horizontal flip + random crop (pad=4) — all on GPU."""
    N, C, H, W = x.shape
    flip_mask = torch.rand(N, 1, 1, 1, device=x.device) < 0.5
    x = torch.where(flip_mask, x.flip(-1), x)
    x_pad = torch.nn.functional.pad(x, [pad] * 4, mode='reflect')
    off_h = torch.randint(0, 2 * pad + 1, (N,), device=x.device)
    off_w = torch.randint(0, 2 * pad + 1, (N,), device=x.device)
    h_idx = off_h[:, None] + torch.arange(H, device=x.device)[None, :]
    w_idx = off_w[:, None] + torch.arange(W, device=x.device)[None, :]
    n_idx = torch.arange(N, device=x.device)
    x = x_pad[n_idx[:, None, None, None],
              torch.arange(C, device=x.device)[None, :, None, None],
              h_idx[:, None, :, None],
              w_idx[:, None, None, :]]
    return x


def evaluate(net, x_test, y_labels, batch_size=256):
    correct = 0
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        lb = y_labels[i:i + batch_size]
        with torch.no_grad():
            ym, _ = net.forward(xb)
        correct += (ym.argmax(dim=1) == lb).sum().item()
    return correct / len(x_test)


def train_one_epoch(net, x, y_oh, bs, sv, augment=True):
    perm = torch.randperm(x.size(0), device=x.device)
    xs, ys = x[perm], y_oh[perm]
    for i in range(0, len(xs), bs):
        xb = xs[i:i + bs]
        if augment:
            xb = augment_batch(xb)
        with torch.no_grad():
            net.step(xb, ys[i:i + bs], sv)


# ====================================================================
# Main
# ====================================================================

def main():
    SPEC = build_3block_spec()

    print("=" * 66)
    print("  CIFAR-10 — 3-Block CNN with TAGI (Triton)")
    print("  Conv(3→64→64) → Conv(64→128→128) → Conv(128→256→256)")
    print("  → FC(512) → FC(10)    [stride-2 convs replace pooling]")
    print("=" * 66)
    print(f"  GPU: {torch.cuda.get_device_name(0)}\n")

    # --- Load & Normalize CIFAR-10 ---
    print("Loading CIFAR-10...", flush=True)
    train_ds = datasets.CIFAR10("data", train=True,  download=True)
    test_ds  = datasets.CIFAR10("data", train=False, download=True)

    x_train = torch.tensor(train_ds.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    x_test  = torch.tensor(test_ds.data,  dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

    # Canonical CIFAR-10 per-channel normalization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    x_train = ((x_train - mean) / std).to(DEVICE)
    x_test  = ((x_test  - mean) / std).to(DEVICE)

    y_train_labels = torch.tensor(train_ds.targets, device=DEVICE)
    y_test_labels  = torch.tensor(test_ds.targets,  device=DEVICE)

    # One-hot labels in {-1, +1} style (matching cuTAGI convention)
    n_cls = 10
    y_train_oh = torch.full((len(y_train_labels), n_cls), -1.0, device=DEVICE)
    y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 1.0)

    # --- Hyperparams ---
    batch_size = 128
    sigma_v    = 0.01
    n_epochs   = 200

    print(f"  Batch: {batch_size}  |  σ_v: {sigma_v}  |  Epochs: {n_epochs}\n")

    # --- Warm up Triton JIT ---
    print("Compiling Triton kernels...", flush=True)
    torch.manual_seed(42)
    tmp = TritonTAGICNN(SPEC, DEVICE)
    for _ in range(2):
        tmp.step(x_train[:batch_size], y_train_oh[:batch_size], sigma_v)
    torch.cuda.synchronize()
    del tmp

    # --- Train ---
    torch.manual_seed(42)
    net = TritonTAGICNN(SPEC, DEVICE)

    accs, wall = [], []
    t0 = time.perf_counter()
    best_acc = 0.0

    for ep in range(n_epochs):
        ep_t = time.perf_counter()
        train_one_epoch(net, x_train, y_train_oh, batch_size, sigma_v)
        torch.cuda.synchronize()
        ep_time = time.perf_counter() - ep_t
        acc = evaluate(net, x_test, y_test_labels)
        accs.append(acc)
        wall.append(time.perf_counter() - t0)
        best_acc = max(best_acc, acc)
        print(f"  Epoch {ep+1:>3}/{n_epochs}  "
              f"Acc: {acc*100:5.2f}%  Best: {best_acc*100:5.2f}%  ({ep_time:.2f}s)")

    total = time.perf_counter() - t0

    # --- Summary ---
    print()
    print("=" * 66)
    print(f"  Final Accuracy: {accs[-1]*100:.2f}%")
    print(f"  Best Accuracy:  {best_acc*100:.2f}%")
    print(f"  Total Time:     {total:.2f}s  ({total/60:.1f} min)")
    print(f"  Avg Epoch:      {total/n_epochs:.2f}s")
    print("=" * 66)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    epochs = np.arange(1, n_epochs + 1)

    ax1.plot(epochs, [a * 100 for a in accs], "s-",
             color="#ef4444", lw=1.5, ms=3,
             label=f"Triton TAGI 3-Block CNN  (best {best_acc*100:.2f}%)")
    ax1.axhline(best_acc * 100, color="#ef4444", lw=1, ls=":", alpha=0.5)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title("CIFAR-10 — TAGI 3-Block CNN (Triton)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(wall, [a * 100 for a in accs], "s-",
             color="#ef4444", lw=1.5, ms=3, label="Triton TAGI")
    ax2.set_xlabel("Wall-Clock Time (s)", fontsize=12)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy vs Training Time", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "cifar10_tagi_3block_cnn.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out}")


if __name__ == "__main__":
    main()
