"""
MNIST Classification — 3-layer MLP — triton-tagi example.

Canonical accuracy benchmark: 784 → 256 → 128 → 10, standard TAGI update.
Reaches ≥ 96 % test accuracy in 20 epochs (96.50 % triton vs 96.61 % cuTAGI
at 5 epochs in Phase 1 validation).

Usage:
    python examples/mnist_mlp.py
    python examples/mnist_mlp.py --n_epochs 10 --batch_size 256 --sigma_v 0.05
    python examples/mnist_mlp.py --data_dir /path/to/data
    python examples/mnist_mlp.py --help
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torchvision import datasets

from triton_tagi import Linear, ReLU, Remax, Sequential
from triton_tagi.checkpoint import RunDir


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

def load_mnist(
    data_dir: str = "data",
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and normalise MNIST, return tensors on ``device``.

    Returns:
        x_train (60000, 784), y_train_oh (60000, 10) one-hot,
        y_train_labels (60000,), x_test (10000, 784), y_test_labels (10000,).
    """
    train_ds = datasets.MNIST(data_dir, train=True, download=True)
    test_ds = datasets.MNIST(data_dir, train=False, download=True)

    x_train = train_ds.data.float().view(-1, 784) / 255.0
    x_test = test_ds.data.float().view(-1, 784) / 255.0

    mu, sigma = x_train.mean(), x_train.std()
    x_train = ((x_train - mu) / sigma).to(device)
    x_test = ((x_test - mu) / sigma).to(device)

    y_train_labels = train_ds.targets.to(device)
    y_test_labels = test_ds.targets.to(device)

    y_train_oh = torch.zeros(len(y_train_labels), 10, device=device)
    y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 1.0)

    return x_train, y_train_oh, y_train_labels, x_test, y_test_labels


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    net: Sequential,
    x_test: torch.Tensor,
    y_labels: torch.Tensor,
    batch_size: int = 1024,
) -> tuple[float, float]:
    """Return (accuracy, mean_predictive_variance) on the full test set."""
    net.eval()
    correct = 0
    total_var = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            xb = x_test[i : i + batch_size]
            lb = y_labels[i : i + batch_size]
            mu, var = net.forward(xb)
            correct += (mu.argmax(dim=1) == lb).sum().item()
            total_var += var.mean().item()
            n_batches += 1
    net.train()
    return correct / len(x_test), total_var / n_batches


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train(
    net: Sequential,
    x_train: torch.Tensor,
    y_train_oh: torch.Tensor,
    x_test: torch.Tensor,
    y_test_labels: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    sigma_v: float,
    device: torch.device,
    run: RunDir,
    config: dict,
) -> float:
    """Full training loop. Returns best test accuracy."""
    print(f"\n  {'Epoch':>5}  {'Test Acc':>9}  {'Avg Var':>9}  {'Time':>7}")
    print("  " + "─" * 38)

    best_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(x_train.size(0), device=device)
        x_s, y_s = x_train[perm], y_train_oh[perm]

        for i in range(0, len(x_s), batch_size):
            net.step(x_s[i : i + batch_size], y_s[i : i + batch_size], sigma_v)

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        acc, avg_var = evaluate(net, x_test, y_test_labels)
        best_acc = max(best_acc, acc)
        print(f"  {epoch:5d}  {acc*100:8.2f}%  {avg_var:9.4f}  {wall:6.2f}s")
        run.append_metrics(epoch, test_acc=acc, avg_var=avg_var, wall_s=wall)

        if epoch % config.get("checkpoint_interval", 10) == 0 or epoch == n_epochs:
            run.save_checkpoint(net, epoch, config)

    print("  " + "─" * 38)
    print(f"  Best test accuracy: {best_acc*100:.2f}%")
    return best_acc


# ---------------------------------------------------------------------------
#  Figure
# ---------------------------------------------------------------------------

def save_figure(run: RunDir) -> None:
    """Plot test accuracy vs epoch from metrics.csv."""
    try:
        import csv
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figure (pip install matplotlib)")
        return

    epochs, accs = [], []
    with open(run.metrics_csv) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            accs.append(float(row["test_acc"]) * 100)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, accs, color="C0", linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("MNIST MLP — TAGI test accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(run.figures / f"training_curve.{ext}", dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {run.figures}/")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(
    n_epochs: int = 20,
    batch_size: int = 512,
    sigma_v: float = 0.05,
    hidden1: int = 256,
    hidden2: int = 128,
    gain_w: float = 1.0,
    gain_b: float = 1.0,
    head: str = "none",
    data_dir: str = "data",
    checkpoint_interval: int = 10,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """MNIST MLP benchmark.

    Args:
        head: Output activation — ``"none"`` (plain linear, one-hot targets) or
              ``"remax"`` (Remax Bayesian softmax).
    """
    torch.manual_seed(seed)
    dev = torch.device(device)

    print("=" * 56)
    print("  MNIST Classification — 3-layer MLP — triton-tagi")
    print(f"  Architecture: 784 → {hidden1} → {hidden2} → 10")
    print("=" * 56)
    if device == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    # ── Data ──
    print(f"\n  Loading MNIST from '{data_dir}'...", flush=True)
    x_train, y_train_oh, _, x_test, y_test_labels = load_mnist(data_dir, dev)
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")

    # ── Config ──
    arch_str = f"mlp_{hidden1}_{hidden2}"
    config: dict = {
        "dataset": "mnist",
        "arch": arch_str,
        "optimizer": "tagi",
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "sigma_v": sigma_v,
        "hidden1": hidden1,
        "hidden2": hidden2,
        "head": head,
        "gain_w": gain_w,
        "gain_b": gain_b,
        "checkpoint_interval": checkpoint_interval,
        "seed": seed,
        "device": device,
        "triton_tagi_version": "0.1.0",
    }

    # ── RunDir ──
    run = RunDir("mnist", arch_str, "tagi")
    run.save_config(config)
    print(f"  Run directory: {run.path}")

    # ── Network ──
    layers: list = [
        Linear(784, hidden1, device=dev, gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        Linear(hidden1, hidden2, device=dev, gain_w=gain_w, gain_b=gain_b),
        ReLU(),
        Linear(hidden2, 10, device=dev, gain_w=gain_w, gain_b=gain_b),
    ]
    if head == "remax":
        layers.append(Remax())
    net = Sequential(layers, device=dev)

    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")
    print(f"\n  Epochs: {n_epochs}  |  Batch: {batch_size}  |  σ_v: {sigma_v}  |  head: {head}")

    # ── Train ──
    best_acc = train(
        net, x_train, y_train_oh, x_test, y_test_labels,
        n_epochs, batch_size, sigma_v, dev, run, config,
    )

    # ── Figure ──
    save_figure(run)
    print(f"\n  Results in: {run.path}")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNIST 3-layer MLP benchmark with TAGI"
    )
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--sigma_v", type=float, default=0.05)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--gain_w", type=float, default=1.0)
    parser.add_argument("--gain_b", type=float, default=1.0)
    parser.add_argument("--head", type=str, default="none",
                        choices=["none", "remax"],
                        help="Output activation: 'none' (plain linear) or 'remax'")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(**vars(args))
