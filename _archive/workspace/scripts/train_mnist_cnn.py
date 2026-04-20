"""
MNIST CNN Classification with TAGI: PyTorch vs Triton
=====================================================
LeNet-like:  Conv(1→32,5×5) → ReLU → Pool(2) →
             Conv(32→64,5×5) → ReLU → Pool(2) →
             Flatten(3136) → FC(256) → ReLU → FC(10)
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from torchvision import datasets
from tagi_cnn_triton import TritonTAGICNN, PTCNN

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda")

# ====================================================================
# Helpers
# ====================================================================

def evaluate(net, x_test, y_labels, batch_size=512):
    correct = 0
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        lb = y_labels[i:i + batch_size]
        with torch.no_grad():
            ym, _ = net.forward(xb)
        correct += (ym.argmax(dim=1) == lb).sum().item()
    return correct / len(x_test)


def train_one_epoch(net, x, y_oh, bs, sv):
    perm = torch.randperm(x.size(0), device=x.device)
    xs, ys = x[perm], y_oh[perm]
    for i in range(0, len(xs), bs):
        with torch.no_grad():
            net.step(xs[i:i + bs], ys[i:i + bs], sv)


def run(label, NetClass, spec, device, x_tr, y_oh, x_te, y_lb,
        bs, sv, n_ep, warmup=False):
    torch.manual_seed(42)
    net = NetClass(spec, device)

    if warmup:
        tmp = NetClass(spec, device)
        for _ in range(2):
            tmp.step(x_tr[:bs], y_oh[:bs], sv)
        torch.cuda.synchronize()
        del tmp
        torch.manual_seed(42)
        net = NetClass(spec, device)

    accs, wall = [], []
    t0 = time.perf_counter()
    for ep in range(n_ep):
        ep_t = time.perf_counter()
        train_one_epoch(net, x_tr, y_oh, bs, sv)
        torch.cuda.synchronize()
        ep_time = time.perf_counter() - ep_t
        acc = evaluate(net, x_te, y_lb)
        accs.append(acc)
        wall.append(time.perf_counter() - t0)
        print(f"  [{label}] Epoch {ep+1:>2}/{n_ep}  "
              f"Acc: {acc*100:5.2f}%  ({ep_time:.2f}s)")
    total = time.perf_counter() - t0
    return accs, wall, total


# ====================================================================
# Main
# ====================================================================

def main():
    SPEC = [
        ('conv', 1,  32, 5, 1, 2),   # 1→32, 5×5, stride=1, pad=2  → 28×28
        ('relu',),
        ('pool', 2),                   # → 14×14
        ('conv', 32, 64, 5, 1, 2),    # 32→64, 5×5                  → 14×14
        ('relu',),
        ('pool', 2),                   # → 7×7
        ('flatten',),                  # 64*7*7 = 3136
        ('fc', 3136, 256),
        ('relu',),
        ('fc', 256, 10),
    ]

    print("=" * 66)
    print("  MNIST CNN Classification with TAGI")
    print("  Conv(1→32,5²) → Pool → Conv(32→64,5²) → Pool → FC(256) → FC(10)")
    print("=" * 66)
    print(f"  GPU: {torch.cuda.get_device_name(0)}\n")

    # --- Load MNIST ---
    print("Loading MNIST...", flush=True)
    train_ds = datasets.MNIST("data", train=True,  download=True)
    test_ds  = datasets.MNIST("data", train=False, download=True)

    # Keep spatial dims: (N, 1, 28, 28)
    x_train = train_ds.data.float().unsqueeze(1) / 255.0
    x_test  = test_ds.data.float().unsqueeze(1)  / 255.0

    mu, sigma = x_train.mean(), x_train.std()
    x_train = ((x_train - mu) / sigma).to(DEVICE)
    x_test  = ((x_test  - mu) / sigma).to(DEVICE)

    y_train_labels = train_ds.targets.to(DEVICE)
    y_test_labels  = test_ds.targets.to(DEVICE)

    y_train_oh = torch.zeros(len(y_train_labels), 10, device=DEVICE)
    y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 1.0)

    # --- Hyperparams ---
    batch_size = 128
    sigma_v    = 0.01
    n_epochs   = 15

    print(f"  Batch: {batch_size}  |  σ_v: {sigma_v}  |  Epochs: {n_epochs}\n")

    # --- PyTorch CNN ---
    # accs_pt, wall_pt, t_pt = run(
    #     "PyTorch", PTCNN, SPEC, DEVICE,
    #     x_train, y_train_oh, x_test, y_test_labels,
    #     batch_size, sigma_v, n_epochs)
    print()

    # --- Triton CNN ---
    accs_tr, wall_tr, t_tr = run(
        "Triton ", TritonTAGICNN, SPEC, DEVICE,
        x_train, y_train_oh, x_test, y_test_labels,
        batch_size, sigma_v, n_epochs, warmup=True)

    # --- Summary ---
    print()
    print("=" * 66)
    print(f"  {'':30s} {'PyTorch':>12s} {'Triton':>12s}")
    print(f"  {'Final Accuracy':30s} {accs_pt[-1]*100:11.2f}% {accs_tr[-1]*100:11.2f}%")
    print(f"  {'Best Accuracy':30s} {max(accs_pt)*100:11.2f}% {max(accs_tr)*100:11.2f}%")
    print(f"  {'Total Time':30s} {t_pt:11.2f}s {t_tr:11.2f}s")
    print(f"  {'Speedup':30s} {'':>12s} {t_pt/t_tr:11.2f}x")
    print("=" * 66)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    epochs = np.arange(1, n_epochs + 1)

    ax1.plot(epochs, [a*100 for a in accs_pt], "o-",
             color="#3b82f6", lw=2, ms=5, label=f"PyTorch ({t_pt:.1f}s)")
    ax1.plot(epochs, [a*100 for a in accs_tr], "s--",
             color="#ef4444", lw=2, ms=5, label=f"Triton  ({t_tr:.1f}s)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("MNIST CNN — TAGI  [LeNet-like]", fontweight="bold")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(wall_pt, [a*100 for a in accs_pt], "o-", color="#3b82f6", lw=2, ms=5, label="PyTorch")
    ax2.plot(wall_tr, [a*100 for a in accs_tr], "s--", color="#ef4444", lw=2, ms=5, label="Triton")
    ax2.set_xlabel("Wall-Clock Time (s)"); ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("Accuracy vs Training Time", fontweight="bold")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "mnist_cnn_tagi_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out}")


if __name__ == "__main__":
    main()
