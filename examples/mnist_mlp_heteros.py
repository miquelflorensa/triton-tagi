"""
MNIST Classification -- TAGI-V Heteroscedastic Regression with +/-C Labels.

Treats 10-class classification as 10 parallel binary regressions:
  y_k = +C  if sample belongs to class k,  -C  otherwise  (default C=3.0).

The network has 2x10 = 20 outputs, interleaved as
  (mean_0, noise_0, mean_1, noise_1, ..., mean_9, noise_9).
EvenSoftplus(half_width=10) maps the noise pre-activations to positive values
while leaving the mean pre-activations unchanged.

Inference:  predicted class = argmax over the 10 mean outputs (columns 0,2,...,18).
Aleatoric:  mean predicted noise std per digit class, reported after training.

Stability note: gain_w=gain_b=0.1 (small init) is required to prevent noise-head
explosion caused by shared hidden-layer growth under gain=1.0. Target scale C=3.0
converges faster than C=1.0 at equal gain.

Usage:
    python examples/mnist_mlp_heteros.py
    python examples/mnist_mlp_heteros.py --n_epochs 30 --target_scale 1.0
    python examples/mnist_mlp_heteros.py --help
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torchvision import datasets

from triton_tagi import EvenSoftplus, Linear, ReLU, Sequential
from triton_tagi.checkpoint import RunDir


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

def load_mnist(
    data_dir: str = "data",
    device: torch.device = torch.device("cpu"),
    target_scale: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and normalise MNIST, return tensors on ``device``.

    Returns:
        x_train  (60000, 784),
        y_scaled (60000, 10)  +/-C encoded targets,
        x_test   (10000, 784),
        y_labels (10000,)     integer class labels,
        y_pm_te  (10000, 10)  +/-C encoded test targets.
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

    # One-hot -> +/-C:  wrong class -> -C,  true class -> +C
    C = target_scale
    oh_train = torch.full((len(y_train_labels), 10), -C, device=device)
    oh_train.scatter_(1, y_train_labels.unsqueeze(1), C)

    oh_test = torch.full((len(y_test_labels), 10), -C, device=device)
    oh_test.scatter_(1, y_test_labels.unsqueeze(1), C)

    return x_train, oh_train, x_test, y_test_labels, oh_test


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    net: Sequential,
    x_test: torch.Tensor,
    y_labels: torch.Tensor,
    batch_size: int = 1024,
) -> tuple[float, torch.Tensor]:
    """Return (accuracy, per_class_mean_noise_std).

    noise_std shape (10,): mean predicted aleatoric noise std for each digit
    class, evaluated on the subset of test samples that truly belong to it.
    """
    net.eval()
    correct = 0
    noise_sum = torch.zeros(10, device=x_test.device)
    noise_cnt = torch.zeros(10, device=x_test.device)

    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            xb = x_test[i : i + batch_size]
            lb = y_labels[i : i + batch_size]
            mu, _ = net.forward(xb)

            # mean heads: even columns 0, 2, ..., 18  ->  (B, 10)
            mu_cls = mu[:, 0::2]
            # noise-var heads: odd columns 1, 3, ..., 19  ->  (B, 10)
            noise_var = mu[:, 1::2]

            correct += (mu_cls.argmax(dim=1) == lb).sum().item()

            # per-class aleatoric noise (gather true-class noise variance)
            noise_true = noise_var.gather(1, lb.unsqueeze(1)).squeeze(1)
            noise_std = noise_true.clamp(min=1e-8).sqrt()
            for c in range(10):
                mask = lb == c
                if mask.any():
                    noise_sum[c] += noise_std[mask].sum()
                    noise_cnt[c] += mask.sum()

    net.train()
    acc = correct / len(x_test)
    per_class_noise = (noise_sum / noise_cnt.clamp(min=1)).cpu()
    return acc, per_class_noise


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train(
    net: Sequential,
    x_train: torch.Tensor,
    y_scaled: torch.Tensor,
    x_test: torch.Tensor,
    y_labels: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    sigma_v: float,
    device: torch.device,
    run: RunDir,
    config: dict,
) -> float:
    """Full training loop. Returns best test accuracy."""
    print(f"\n  {'Epoch':>5}  {'Test Acc':>9}  {'Time':>7}")
    print("  " + "-" * 30)

    best_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(x_train.size(0), device=device)
        x_s, y_s = x_train[perm], y_scaled[perm]

        for i in range(0, len(x_s), batch_size):
            # net.step auto-selects heteros kernel: output (B,20), target (B,10)
            net.step(x_s[i : i + batch_size], y_s[i : i + batch_size], sigma_v)

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        acc, _ = evaluate(net, x_test, y_labels)
        best_acc = max(best_acc, acc)
        print(f"  {epoch:5d}  {acc*100:8.2f}%  {wall:6.2f}s")
        run.append_metrics(epoch, test_acc=acc, wall_s=wall)

        if epoch % config.get("checkpoint_interval", 10) == 0 or epoch == n_epochs:
            run.save_checkpoint(net, epoch, config)

    print("  " + "-" * 30)
    print(f"  Best test accuracy: {best_acc*100:.2f}%")
    return best_acc


# ---------------------------------------------------------------------------
#  Figure
# ---------------------------------------------------------------------------

def save_figure(run: RunDir, per_class_noise: torch.Tensor) -> None:
    """Plot test accuracy curve and per-class aleatoric noise."""
    try:
        import csv

        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed -- skipping figure")
        return

    epochs, accs = [], []
    with open(run.metrics_csv) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            accs.append(float(row["test_acc"]) * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, accs, color="C0", linewidth=1.5, marker="o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test accuracy (%)")
    ax1.set_title("MNIST TAGI-V -- test accuracy")
    ax1.grid(True, alpha=0.3)

    classes = list(range(10))
    ax2.bar(classes, per_class_noise.numpy(), color="C1", alpha=0.8)
    ax2.set_xlabel("Digit class")
    ax2.set_ylabel("Mean aleatoric noise std")
    ax2.set_title("Learned aleatoric noise per class")
    ax2.set_xticks(classes)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(run.figures / f"results.{ext}", dpi=150)
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
    gain_w: float = 0.1,
    gain_b: float = 0.1,
    target_scale: float = 3.0,
    data_dir: str = "data",
    checkpoint_interval: int = 10,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """MNIST TAGI-V: heteroscedastic regression with +/-C binary targets per class.

    Small gain (0.1) is required for stability -- gain=1.0 causes noise-head
    explosion due to shared hidden-layer growth during mean-head training.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)

    print("=" * 64)
    print("  MNIST TAGI-V -- Heteroscedastic Regression (+/-C labels)")
    print(f"  Architecture: 784 -> {hidden1} -> {hidden2} -> 20 + EvenSoftplus")
    print(f"  target_scale={target_scale}  gain_w={gain_w}  gain_b={gain_b}")
    print("=" * 64)
    if device == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    # -- Data --
    print(f"\n  Loading MNIST from '{data_dir}'...", flush=True)
    x_train, y_scaled, x_test, y_labels, y_pm_te = load_mnist(data_dir, dev, target_scale)
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    print(f"  Targets: +/-{target_scale} per class")

    # -- Config --
    arch_str = f"mlp_{hidden1}_{hidden2}_heteros"
    config: dict = {
        "dataset": "mnist_heteros",
        "arch": arch_str,
        "optimizer": "tagi-v",
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "sigma_v": sigma_v,
        "hidden1": hidden1,
        "hidden2": hidden2,
        "gain_w": gain_w,
        "gain_b": gain_b,
        "target_scale": target_scale,
        "checkpoint_interval": checkpoint_interval,
        "seed": seed,
        "device": device,
        "note": "TAGI-V: 20 outputs = 10x(mean, noise_var); +/-C regression targets",
    }

    # -- RunDir --
    run = RunDir("mnist_heteros", arch_str, "tagi-v")
    run.save_config(config)
    print(f"  Run directory: {run.path}")

    # -- Network --
    # 20 outputs: (mean_0, noise_0, ..., mean_9, noise_9)
    # EvenSoftplus(half_width=10) applies softplus to noise heads (odd columns)
    net = Sequential(
        [
            Linear(784, hidden1, device=dev, gain_w=gain_w, gain_b=gain_b),
            ReLU(),
            Linear(hidden1, hidden2, device=dev, gain_w=gain_w, gain_b=gain_b),
            ReLU(),
            Linear(hidden2, 20, device=dev, gain_w=gain_w, gain_b=gain_b),
            EvenSoftplus(half_width=10),
        ],
        device=dev,
    )
    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")
    print(f"\n  Epochs: {n_epochs}  |  Batch: {batch_size}  |  sigma_v: {sigma_v} (unused by heteros kernel)")

    # -- Train --
    best_acc = train(
        net, x_train, y_scaled, x_test, y_labels,
        n_epochs, batch_size, sigma_v, dev, run, config,
    )

    # -- Per-class aleatoric noise --
    _, per_class_noise = evaluate(net, x_test, y_labels)
    print("\n  Mean aleatoric noise std per digit class:")
    for c, v in enumerate(per_class_noise.tolist()):
        bar = "#" * max(1, int(v * 20))
        print(f"    {c}: {v:.4f}  {bar}")

    # -- Figure --
    save_figure(run, per_class_noise)
    print(f"\n  Results in: {run.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNIST TAGI-V: heteroscedastic regression with +/-C binary targets"
    )
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--sigma_v",
        type=float,
        default=0.05,
        help="Passed to net.step() but unused by the heteros update kernel.",
    )
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--gain_w", type=float, default=0.1)
    parser.add_argument("--gain_b", type=float, default=0.1)
    parser.add_argument(
        "--target_scale",
        type=float,
        default=3.0,
        help="C: targets are +C (true class) / -C (other classes).",
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    main(**vars(args))
