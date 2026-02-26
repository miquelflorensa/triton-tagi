"""
CIFAR-10 — ResNet-18 with TAGI + BatchNorm (Triton)
=====================================================
ResNet-18 adapted for CIFAR-10 (32×32 input) using the modular src/ layers:

    Stem:    Conv(3→64, 3×3, s=1) → BN → ReLU              → 32×32
    Layer1:  BasicBlock(64→64,   s=1) × 2                    → 32×32
    Layer2:  BasicBlock(64→128,  s=2) + BasicBlock(128, s=1) → 16×16
    Layer3:  BasicBlock(128→256, s=2) + BasicBlock(256, s=1) →  8×8
    Layer4:  BasicBlock(256→512, s=2) + BasicBlock(512, s=1) →  4×4
    Head:    AvgPool(4) → Flatten → FC(512→10)

Data augmentation:
    - Random horizontal flip
    - Random crop (pad=4, reflect padding)

Each BasicBlock:
    Conv3×3 → BN → ReLU → Conv3×3 → BN → (+skip) → ReLU
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from torchvision import datasets

from src.resnet18 import ResNet18

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda")


# ====================================================================
# Data Augmentation (GPU-accelerated)
# ====================================================================

def augment_batch(x, pad=4):
    """
    Random horizontal flip + random crop — all on GPU.

    Parameters
    ----------
    x   : Tensor (N, C, H, W)
    pad : int  zero-padding before crop (default 4)

    Returns
    -------
    x_aug : Tensor (N, C, H, W)
    """
    N, C, H, W = x.shape

    # Random horizontal flip (per sample)
    flip_mask = torch.rand(N, 1, 1, 1, device=x.device) < 0.5
    x = torch.where(flip_mask, x.flip(-1), x)

    # Random crop: pad with reflect → random crop back to H×W
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


# ====================================================================
# Evaluation
# ====================================================================

def evaluate(net, x_test, y_labels, batch_size=128):
    """Compute test accuracy (eval mode — BN uses running stats)."""
    net.eval()
    correct = 0
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        lb = y_labels[i:i + batch_size]
        with torch.no_grad():
            ym, _ = net.forward(xb)
        correct += (ym.argmax(dim=1) == lb).sum().item()
    net.train()
    return correct / len(x_test)


# ====================================================================
# Training
# ====================================================================

def train_one_epoch(net, x, y_oh, bs, sv, augment=True):
    """Train for one epoch with optional data augmentation."""
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
    print("=" * 70)
    print("  CIFAR-10 — ResNet-18 + BatchNorm with TAGI (Triton)")
    print("  Stem → 4×BasicBlock groups (BN inside) → AvgPool → FC(10)")
    print("  Data Aug: Random flip + Random crop (pad=4)")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}\n")

    # --- Load CIFAR-10 ---
    print("Loading CIFAR-10...", flush=True)
    train_ds = datasets.CIFAR10("data", train=True,  download=True)
    test_ds  = datasets.CIFAR10("data", train=False, download=True)

    # (N, 32, 32, 3) → (N, 3, 32, 32), float [0, 1]
    x_train = torch.tensor(train_ds.data, dtype=torch.float32) \
                   .permute(0, 3, 1, 2) / 255.0
    x_test  = torch.tensor(test_ds.data, dtype=torch.float32) \
                   .permute(0, 3, 1, 2) / 255.0

    # Canonical CIFAR-10 per-channel normalization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    x_train = ((x_train - mean) / std).to(DEVICE)
    x_test  = ((x_test  - mean) / std).to(DEVICE)

    y_train_labels = torch.tensor(train_ds.targets, device=DEVICE)
    y_test_labels  = torch.tensor(test_ds.targets,  device=DEVICE)

    # ±1 one-hot labels (matching cuTAGI convention)
    y_train_oh = torch.full((len(y_train_labels), 10), -3.0, device=DEVICE)
    y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 3.0)

    # --- Hyperparams ---
    batch_size = 128
    sigma_v    = 0.01
    n_epochs   = 200

    print(f"  Batch: {batch_size}  |  σ_v: {sigma_v}  |  Epochs: {n_epochs}")

    # --- Build model ---
    print("\nBuilding ResNet-18 + BN...", flush=True)
    torch.manual_seed(42)
    net = ResNet18(num_classes=10, device=DEVICE)
    print(net)
    print(f"\n  Parameters: {net.num_parameters():,} (means + variances)")

    # --- Warm up Triton JIT ---
    print("\nCompiling Triton kernels...", flush=True)
    for _ in range(2):
        with torch.no_grad():
            net.step(x_train[:batch_size], y_train_oh[:batch_size], sigma_v)
    torch.cuda.synchronize()

    # Re-initialise for a clean training run
    torch.manual_seed(42)
    net = ResNet18(num_classes=10, device=DEVICE)
    print("  Done.\n")

    # --- Train ---
    accs, wall = [], []
    t0 = time.perf_counter()
    best_acc = 0.0

    for ep in range(n_epochs):
        ep_t = time.perf_counter()
        train_one_epoch(net, x_train, y_train_oh, batch_size, sigma_v,
                        augment=True)
        torch.cuda.synchronize()
        ep_time = time.perf_counter() - ep_t

        acc = evaluate(net, x_test, y_test_labels)
        accs.append(acc)
        wall.append(time.perf_counter() - t0)
        best_acc = max(best_acc, acc)

        print(f"  Epoch {ep+1:>3}/{n_epochs}  "
              f"Acc: {acc*100:5.2f}%  Best: {best_acc*100:5.2f}%  "
              f"({ep_time:.2f}s)")

    total = time.perf_counter() - t0

    # --- Summary ---
    print()
    print("=" * 70)
    print(f"  Final Accuracy: {accs[-1]*100:.2f}%")
    print(f"  Best Accuracy:  {best_acc*100:.2f}%")
    print(f"  Total Time:     {total:.2f}s  ({total/60:.1f} min)")
    print(f"  Avg Epoch:      {total/n_epochs:.2f}s")
    print("=" * 70)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    epochs = np.arange(1, n_epochs + 1)

    ax1.plot(epochs, [a * 100 for a in accs], "-",
             color="#6366f1", lw=1.5,
             label=f"TAGI ResNet-18+BN  (best {best_acc*100:.2f}%)")
    ax1.axhline(best_acc * 100, color="#6366f1", lw=1, ls=":", alpha=0.5)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title("CIFAR-10 — TAGI ResNet-18 + BatchNorm (Triton)",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, n_epochs)

    ax2.plot(wall, [a * 100 for a in accs], "-",
             color="#6366f1", lw=1.5, label="TAGI ResNet-18+BN")
    ax2.set_xlabel("Wall-Clock Time (s)", fontsize=12)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy vs Training Time", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "cifar10_resnet18_bn.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out}")


if __name__ == "__main__":
    main()
