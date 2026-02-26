"""
CIFAR-10 Classification with TAGI (Triton)
============================================
3-CNN Architecture:
  Conv(3→32, 3×3, pad=1) → ReLU → AvgPool(2)  → 16×16
  Conv(32→64, 3×3, pad=1) → ReLU → AvgPool(2) →  8×8
  Conv(64→128,3×3, pad=1) → ReLU → AvgPool(2) →  4×4
  Flatten(2048) → FC(10)
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms
from tagi_cnn_triton import TritonTAGICNN

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda")


def evaluate(net, x_test, y_labels, batch_size=256):
    correct = 0
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        lb = y_labels[i:i + batch_size]
        with torch.no_grad():
            ym, _ = net.forward(xb)
        correct += (ym.argmax(dim=1) == lb).sum().item()
    return correct / len(x_test)


def augment_batch(x, pad=4):
    """Random horizontal flip + random crop (pad=4) — all on GPU."""
    N, C, H, W = x.shape
    # Random horizontal flip (per sample)
    flip_mask = torch.rand(N, 1, 1, 1, device=x.device) < 0.5
    x = torch.where(flip_mask, x.flip(-1), x)
    # Random crop: pad with reflect → random crop back to HxW
    x_pad = torch.nn.functional.pad(x, [pad] * 4, mode='reflect')
    # Per-sample random offsets
    off_h = torch.randint(0, 2 * pad + 1, (N,), device=x.device)
    off_w = torch.randint(0, 2 * pad + 1, (N,), device=x.device)
    # Crop using advanced indexing
    # Build (N, H) and (N, W) index grids
    h_idx = off_h[:, None] + torch.arange(H, device=x.device)[None, :]  # (N, H)
    w_idx = off_w[:, None] + torch.arange(W, device=x.device)[None, :]  # (N, W)
    # Index: x_pad[n, :, h_idx[n], :][:, :, w_idx[n]]
    n_idx = torch.arange(N, device=x.device)
    x = x_pad[n_idx[:, None, None, None],
              torch.arange(C, device=x.device)[None, :, None, None],
              h_idx[:, None, :, None],
              w_idx[:, None, None, :]]
    return x


def train_one_epoch(net, x, y_oh, bs, sv, augment=True):
    perm = torch.randperm(x.size(0), device=x.device)
    xs, ys = x[perm], y_oh[perm]
    for i in range(0, len(xs), bs):
        xb = xs[i:i + bs]
        if augment:
            xb = augment_batch(xb)
        with torch.no_grad():
            net.step(xb, ys[i:i + bs], sv)


def main():
    # --- Architecture ---
    SPEC = [
        ('conv',  3,  32, 3, 1, 1),   # 3→32,  3×3, stride=1, pad=1  → 32×32
        ('relu',),
        ('pool', 2),                    # → 16×16
        ('conv', 32,  64, 3, 1, 1),    # 32→64, 3×3                   → 16×16
        ('relu',),
        ('pool', 2),                    # → 8×8
        ('conv', 64, 128, 3, 1, 1),    # 64→128, 3×3                  → 8×8
        ('relu',),
        ('pool', 2),                    # → 4×4
        ('flatten',),                   # 128×4×4 = 2048
        ('fc', 2048, 10),
    ]

    print("=" * 66)
    print("  CIFAR-10 Classification with TAGI (Triton)")
    print("  3-CNN: Conv(3→32)→Conv(32→64)→Conv(64→128)→FC(10)")
    print("=" * 66)
    print(f"  GPU: {torch.cuda.get_device_name(0)}\n")

    # --- Load CIFAR-10 ---
    print("Loading CIFAR-10...", flush=True)
    train_ds = datasets.CIFAR10("data", train=True,  download=True)
    test_ds  = datasets.CIFAR10("data", train=False, download=True)

    # (N, 32, 32, 3) → (N, 3, 32, 32), float [0,1]
    x_train = torch.tensor(train_ds.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    x_test  = torch.tensor(test_ds.data,  dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

    # Standard CIFAR-10 per-channel normalization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    x_train = ((x_train - mean) / std).to(DEVICE)
    x_test  = ((x_test  - mean) / std).to(DEVICE)

    y_train_labels = torch.tensor(train_ds.targets, device=DEVICE)
    y_test_labels  = torch.tensor(test_ds.targets,  device=DEVICE)

    # One-hot encode
    # y_train_oh = torch.zeros(len(y_train_labels), 10, device=DEVICE)
    # y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 1.0)

    # -3 / 3
    y_train_oh = torch.zeros(len(y_train_labels), 10, device=DEVICE) - 3.0
    y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 3.0)

    # --- Hyperparams ---
    batch_size = 256
    sigma_v    = 0.01
    n_epochs   = 500

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

    for ep in range(n_epochs):
        ep_t = time.perf_counter()
        train_one_epoch(net, x_train, y_train_oh, batch_size, sigma_v)
        torch.cuda.synchronize()
        ep_time = time.perf_counter() - ep_t
        acc = evaluate(net, x_test, y_test_labels)
        accs.append(acc)
        wall.append(time.perf_counter() - t0)
        print(f"  Epoch {ep+1:>2}/{n_epochs}  "
              f"Acc: {acc*100:5.2f}%  ({ep_time:.2f}s)")

    total = time.perf_counter() - t0

    # --- Summary ---
    print()
    print("=" * 66)
    print(f"  Final Accuracy: {accs[-1]*100:.2f}%")
    print(f"  Best Accuracy:  {max(accs)*100:.2f}%")
    print(f"  Total Time:     {total:.2f}s")
    print(f"  Avg Epoch:      {total/n_epochs:.2f}s")
    print("=" * 66)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    epochs = np.arange(1, n_epochs + 1)

    ax1.plot(epochs, [a * 100 for a in accs], "s-",
             color="#ef4444", lw=2, ms=4, label=f"Triton TAGI ({total:.1f}s)")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title("CIFAR-10 — TAGI 3-CNN (Triton)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, n_epochs)

    ax2.plot(wall, [a * 100 for a in accs], "s-",
             color="#ef4444", lw=2, ms=4, label="Triton TAGI")
    ax2.set_xlabel("Wall-Clock Time (s)", fontsize=12)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy vs Training Time", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "cifar10_tagi_3cnn.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out}")


if __name__ == "__main__":
    main()
