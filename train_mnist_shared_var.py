"""
MNIST: Standard TAGI vs Shared-Variance TAGI (CNN)
====================================================
Side-by-side comparison of:
  1. Standard TAGI CNN (per-parameter variances)
  2. Shared-Var TAGI CNN (one scalar variance per layer)

Architecture: Conv(1→32,5)→ReLU→Pool(2)→Conv(32→64,5)→ReLU→Pool(2)→FC(3136→256)→ReLU→FC(256→10)
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from torchvision import datasets
from tagi_cnn_triton import TritonTAGICNN
from tagi_shared_var_triton import SharedVarTAGICNN

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda")

SPEC = [
    ('conv',  1,  32, 5, 1, 2),   # 1→32,  5×5, stride=1, pad=2  → 28×28
    ('relu',),
    ('pool', 2),                    # → 14×14
    ('conv', 32,  64, 5, 1, 2),    # 32→64, 5×5                   → 14×14
    ('relu',),
    ('pool', 2),                    # → 7×7
    ('flatten',),                   # 64×7×7 = 3136
    ('fc', 3136, 256),
    ('relu',),
    ('fc', 256, 10),
]


# ====================================================================
# Helpers
# ====================================================================

def evaluate(net, x_test, y_labels, batch_size=256):
    correct = 0
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        lb = y_labels[i:i + batch_size]
        with torch.no_grad():
            ym, _ = net.forward(xb)
        correct += (ym.argmax(dim=1) == lb).sum().item()
    return correct / len(x_test)


def train_one_epoch(net, x_train, y_train_oh, batch_size, sigma_v):
    perm = torch.randperm(x_train.size(0), device=x_train.device)
    x_s, y_s = x_train[perm], y_train_oh[perm]
    for i in range(0, len(x_s), batch_size):
        xb = x_s[i:i + batch_size]
        yb = y_s[i:i + batch_size]
        with torch.no_grad():
            net.step(xb, yb, sigma_v)


def count_variance_params(net):
    """Count the number of variance parameters."""
    total = 0
    if isinstance(net, TritonTAGICNN):
        for lt, layer in zip(net.layer_types, net.layers):
            if lt == 'conv' or lt == 'fc':
                total += layer.Sw.numel() + layer.Sb.numel()
    elif isinstance(net, SharedVarTAGICNN):
        for lt, layer in zip(net.layer_types, net.layers):
            if lt in ('conv', 'fc'):
                total += 2   # 1 sw + 1 sb
    return total


def run_experiment(label, net, x_train, y_train_oh, x_test, y_test_labels,
                   batch_size, sigma_v, n_epochs):
    accs = []
    t0 = time.perf_counter()

    var_history = []  # track variance evolution

    for epoch in range(n_epochs):
        ep_start = time.perf_counter()
        train_one_epoch(net, x_train, y_train_oh, batch_size, sigma_v)
        torch.cuda.synchronize()
        ep_time = time.perf_counter() - ep_start
        acc = evaluate(net, x_test, y_test_labels)
        accs.append(acc)

        # Track variances
        if hasattr(net, 'get_variances'):
            var_history.append(net.get_variances())

        print(f"  [{label}] Epoch {epoch + 1:>2}/{n_epochs}  "
              f"Acc: {acc * 100:5.2f}%  ({ep_time:.2f}s)")

    total = time.perf_counter() - t0
    return accs, total, var_history


# ====================================================================
# Main
# ====================================================================

def main():
    print("=" * 70)
    print("  MNIST: Standard TAGI CNN vs Shared-Variance TAGI CNN")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}\n")

    # --- Load MNIST ---
    print("Loading MNIST...", flush=True)
    train_ds = datasets.MNIST("data", train=True,  download=True)
    test_ds  = datasets.MNIST("data", train=False, download=True)

    x_train = train_ds.data.float().unsqueeze(1) / 255.0   # (N, 1, 28, 28)
    x_test  = test_ds.data.float().unsqueeze(1)  / 255.0

    mu, sigma = x_train.mean(), x_train.std()
    x_train = ((x_train - mu) / sigma).to(DEVICE)
    x_test  = ((x_test  - mu) / sigma).to(DEVICE)

    y_train_labels = train_ds.targets.to(DEVICE)
    y_test_labels  = test_ds.targets.to(DEVICE)

    y_train_oh = torch.zeros(len(y_train_labels), 10, device=DEVICE) - 3.0
    y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 3.0)

    # --- Hyperparameters ---
    batch_size = 128
    sigma_v    = 0.01
    n_epochs   = 30

    print(f"  Architecture: CNN [Conv32→Conv64→FC256→FC10]")
    print(f"  Batch: {batch_size}  |  σ_v: {sigma_v}  |  Epochs: {n_epochs}\n")

    # --- Standard TAGI CNN ---
    print("-" * 50)
    print("  Standard TAGI CNN (per-parameter variances)")
    print("-" * 50)
    torch.manual_seed(42)
    net_std = TritonTAGICNN(SPEC, DEVICE)
    n_var_std = count_variance_params(net_std)
    print(f"  Variance parameters: {n_var_std:,}")
    accs_std, t_std, _ = run_experiment(
        "Standard", net_std, x_train, y_train_oh, x_test, y_test_labels,
        batch_size, sigma_v, n_epochs)
    print()

    # --- Shared-Variance TAGI CNN ---
    print("-" * 50)
    print("  Shared-Variance TAGI CNN (one variance per layer)")
    print("-" * 50)
    torch.manual_seed(42)
    net_sv = SharedVarTAGICNN(SPEC, DEVICE, gain_var=0.1)
    n_var_sv = count_variance_params(net_sv)
    print(f"  Variance parameters: {n_var_sv}")
    accs_sv, t_sv, var_hist = run_experiment(
        "SharedVar", net_sv, x_train, y_train_oh, x_test, y_test_labels,
        batch_size, sigma_v, n_epochs)

    # --- Summary ---
    print()
    print("=" * 70)
    print(f"  {'':30s} {'Standard':>12s} {'SharedVar':>12s}")
    print(f"  {'Final Accuracy':30s} {accs_std[-1]*100:11.2f}% {accs_sv[-1]*100:11.2f}%")
    print(f"  {'Best Accuracy':30s} {max(accs_std)*100:11.2f}% {max(accs_sv)*100:11.2f}%")
    print(f"  {'Total Time':30s} {t_std:11.2f}s {t_sv:11.2f}s")
    print(f"  {'Variance Params':30s} {n_var_std:>12,} {n_var_sv:>12,}")
    reduction = (1 - n_var_sv / n_var_std) * 100
    print(f"  {'Var Param Reduction':30s} {'':>12s} {reduction:10.1f}%")
    print("=" * 70)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = np.arange(1, n_epochs + 1)

    # 1. Accuracy comparison
    ax = axes[0]
    ax.plot(epochs, [a * 100 for a in accs_std], "o-",
            color="#3b82f6", lw=2, ms=4, label=f"Standard ({n_var_std:,} var params)")
    ax.plot(epochs, [a * 100 for a in accs_sv], "s--",
            color="#ef4444", lw=2, ms=4, label=f"SharedVar ({n_var_sv} var params)")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy: Standard vs SharedVar TAGI", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_epochs)

    # 2. Variance evolution (SharedVar only)
    ax = axes[1]
    if var_hist:
        keys = sorted(var_hist[0].keys())
        for key in keys:
            vals = [vh[key] for vh in var_hist]
            ax.plot(epochs, vals, lw=2, label=key)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Variance value", fontsize=11)
        ax.set_title("SharedVar: Variance Evolution", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    # 3. Parameter count comparison (bar chart)
    ax = axes[2]
    labels = ['Standard', 'SharedVar']
    var_counts = [n_var_std, n_var_sv]
    colors = ['#3b82f6', '#ef4444']
    bars = ax.bar(labels, var_counts, color=colors, width=0.5, edgecolor='black', lw=0.5)
    ax.set_ylabel("Number of Variance Params", fontsize=11)
    ax.set_title("Variance Parameter Count", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    for bar, count in zip(bars, var_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{count:,}', ha='center', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = "shared_var_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")


if __name__ == "__main__":
    main()
