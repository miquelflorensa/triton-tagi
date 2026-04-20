"""
CIFAR-10 Classification — 3-block CNN — triton-tagi example.

Architecture:
    Conv(3→32, 5, pad=2) → ReLU → BN → AvgPool(2)   [32→16]
    Conv(32→64, 5, pad=2) → ReLU → BN → AvgPool(2)  [16→8]
    Conv(64→64, 5, pad=2) → ReLU → BN → AvgPool(2)  [8→4]
    Flatten → FC(1024→256) → ReLU → FC(256→10) → Remax

GPU augmentation (random flip + random crop) is applied each batch.

Usage:
    python examples/cifar10_cnn.py
    python examples/cifar10_cnn.py --n_epochs 20 --sigma_v 0.01
    python examples/cifar10_cnn.py --data_dir /path/to/data --no_augment
    python examples/cifar10_cnn.py --help
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from triton_tagi import (
    AvgPool2D,
    BatchNorm2D,
    Conv2D,
    Flatten,
    Linear,
    ReLU,
    Remax,
    Sequential,
)
from triton_tagi.checkpoint import RunDir


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(
    data_dir: str = "data",
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load CIFAR-10 as (N, 3, 32, 32) tensors on ``device``.

    Returns:
        x_train (50000,3,32,32), y_train_oh (50000,10), y_train_labels (50000,),
        x_test (10000,3,32,32), y_test_labels (10000,).
    """
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)])
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=norm)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=norm)

    x_train = torch.stack([img for img, _ in train_ds]).to(device)
    y_train = torch.tensor([lbl for _, lbl in train_ds], device=device)
    x_test = torch.stack([img for img, _ in test_ds]).to(device)
    y_test = torch.tensor([lbl for _, lbl in test_ds], device=device)

    y_train_oh = torch.zeros(len(y_train), 10, device=device)
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 1.0)

    return x_train, y_train_oh, y_train, x_test, y_test


# ---------------------------------------------------------------------------
#  GPU augmentation (random horizontal flip + random crop, no CPU round-trip)
# ---------------------------------------------------------------------------

def gpu_augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random horizontal flip and random crop applied to a batch on-device."""
    B, C, H, W = x.shape
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    top = torch.randint(0, 2 * pad, (B,), device=x.device)
    left = torch.randint(0, 2 * pad, (B,), device=x.device)
    rows = top.unsqueeze(1) + torch.arange(H, device=x.device).unsqueeze(0)
    cols = left.unsqueeze(1) + torch.arange(W, device=x.device).unsqueeze(0)
    return x_pad[
        torch.arange(B, device=x.device)[:, None, None, None],
        torch.arange(C, device=x.device)[None, :, None, None],
        rows[:, None, :, None].expand(B, C, H, W),
        cols[:, None, None, :].expand(B, C, H, W),
    ]


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def _predict_probs(net: Sequential, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    net.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            mu, _ = net.forward(x[i : i + batch_size])
            chunks.append(mu)
    net.train()
    probs = torch.cat(chunks, dim=0).clamp(min=0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return probs


def evaluate(
    net: Sequential,
    x: torch.Tensor,
    y_labels: torch.Tensor,
    batch_size: int = 256,
    n_bins: int = 15,
) -> tuple[float, float]:
    """Return (accuracy, 15-bin ECE) on the given set."""
    probs = _predict_probs(net, x, batch_size)
    confidences, preds = probs.max(dim=1)
    correct = (preds == y_labels).float()
    acc = correct.mean().item()

    bounds = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = 0.0
    N = len(y_labels)
    for i in range(n_bins):
        lo, hi = bounds[i], bounds[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences <= hi)
        n = mask.sum().item()
        if n > 0:
            ece += (n / N) * abs(
                confidences[mask].mean().item() - correct[mask].mean().item()
            )
    return acc, ece


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train(
    net: Sequential,
    x_train: torch.Tensor,
    y_train_oh: torch.Tensor,
    y_train_labels: torch.Tensor,
    x_test: torch.Tensor,
    y_test_labels: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    sigma_v: float,
    augment: bool,
    device: torch.device,
    run: RunDir,
    config: dict,
) -> float:
    """Training loop with optional GPU augmentation. Returns best test accuracy."""
    header = f"  {'Epoch':>5}  {'Train':>7}  {'Test':>7}  {'ECE':>6}  {'Time':>7}"
    print(f"\n{header}")
    print("  " + "─" * (len(header) - 2))

    best_acc = 0.0
    train_eval_n = config.get("train_eval_n", 5000)

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(x_train.size(0), device=device)
        x_s, y_s = x_train[perm], y_train_oh[perm]

        for i in range(0, len(x_s), batch_size):
            xb = x_s[i : i + batch_size]
            if augment:
                xb = gpu_augment(xb)
            net.step(xb, y_s[i : i + batch_size], sigma_v)

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        idx = torch.randperm(x_train.size(0), device=device)[:train_eval_n]
        train_acc, _ = evaluate(net, x_train[idx], y_train_labels[idx])
        test_acc, test_ece = evaluate(net, x_test, y_test_labels)
        best_acc = max(best_acc, test_acc)

        print(
            f"  {epoch:5d}  {train_acc*100:6.2f}%  {test_acc*100:6.2f}%"
            f"  {test_ece:6.4f}  {wall:6.2f}s"
        )
        run.append_metrics(
            epoch,
            train_acc=train_acc,
            test_acc=test_acc,
            test_ece=test_ece,
            sigma_v=sigma_v,
            wall_s=wall,
        )

        if epoch % config.get("checkpoint_interval", 10) == 0 or epoch == n_epochs:
            run.save_checkpoint(net, epoch, config)

    print("  " + "─" * (len(header) - 2))
    print(f"  Best test accuracy: {best_acc*100:.2f}%")
    return best_acc


# ---------------------------------------------------------------------------
#  Figure
# ---------------------------------------------------------------------------

def save_figure(run: RunDir) -> None:
    try:
        import csv
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figure (pip install matplotlib)")
        return

    epochs, train_accs, test_accs, eces = [], [], [], []
    with open(run.metrics_csv) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_accs.append(float(row["train_acc"]) * 100)
            test_accs.append(float(row["test_acc"]) * 100)
            eces.append(float(row["test_ece"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax1.plot(epochs, train_accs, label="Train", color="C1", linewidth=1.5, marker="o", markersize=3)
    ax1.plot(epochs, test_accs, label="Test", color="C0", linewidth=1.5, marker="o", markersize=3)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("CIFAR-10 3-block CNN — TAGI")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.plot(epochs, eces, color="C3", linewidth=1.5, marker="o", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test ECE")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(run.figures / f"training_curve.{ext}", dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {run.figures}/")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(
    n_epochs: int = 50,
    batch_size: int = 128,
    sigma_v: float = 0.01,
    gain_w: float = 0.1,
    gain_b: float = 0.1,
    pool: str = "avg",
    augment: bool = True,
    data_dir: str = "data",
    checkpoint_interval: int = 10,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """CIFAR-10 3-block CNN benchmark.

    Args:
        sigma_v: Observation noise (fixed).
        pool:    Spatial downsample. "avg" (AvgPool2D k=2) or "stride"
                 (stride-2 conv replaces the block's Conv; no separate pool).
        augment: Apply random flip + crop augmentation each batch.
    """
    if pool not in ("avg", "stride"):
        raise ValueError(f"--pool must be 'avg' or 'stride', got {pool!r}")

    torch.manual_seed(seed)
    dev = torch.device(device)

    print("=" * 60)
    print("  CIFAR-10 Classification — 3-block CNN — triton-tagi")
    print(f"  pool={pool}")
    print("=" * 60)
    if device == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    # ── Data ──
    print(f"\n  Loading CIFAR-10 from '{data_dir}'...", flush=True)
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels = load_cifar10(data_dir, dev)
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    print(f"  Input shape: {tuple(x_train.shape[1:])}")

    arch_tag = f"cnn3_bn_{pool}"

    # ── Config ──
    config: dict = {
        "dataset": "cifar10",
        "arch": arch_tag,
        "optimizer": "tagi",
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "sigma_v": sigma_v,
        "gain_w": gain_w,
        "gain_b": gain_b,
        "pool": pool,
        "augment": augment,
        "checkpoint_interval": checkpoint_interval,
        "seed": seed,
        "device": device,
        "triton_tagi_version": "0.2.0",
    }

    # ── RunDir ──
    run = RunDir("cifar10", arch_tag, "tagi")
    run.save_config(config)
    print(f"  Run directory: {run.path}")

    # ── Network ──
    kw = {"device": dev, "gain_w": gain_w, "gain_b": gain_b}

    def _conv_block(in_c: int, out_c: int) -> list:
        stride = 2 if pool == "stride" else 1
        block = [
            Conv2D(in_c, out_c, 5, stride=stride, padding=2, **kw),
            ReLU(),
            BatchNorm2D(out_c, **kw),
        ]
        if pool == "avg":
            block.append(AvgPool2D(2))
        return block

    net = Sequential(
        [
            *_conv_block(3, 32),    # 32→16
            *_conv_block(32, 64),   # 16→8
            *_conv_block(64, 64),   # 8→4
            Flatten(),              # 64*4*4 = 1024
            Linear(1024, 256, **kw),
            ReLU(),
            Linear(256, 10, **kw),
            Remax(),
        ],
        device=dev,
    )
    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")
    print(
        f"\n  Epochs: {n_epochs}  |  Batch: {batch_size}  |  σ_v: {sigma_v}"
        f"  |  augment: {augment}"
    )

    # ── Train ──
    best_acc = train(
        net, x_train, y_train_oh, y_train_labels, x_test, y_test_labels,
        n_epochs, batch_size, sigma_v, augment, dev, run, config,
    )

    # ── Figure ──
    save_figure(run)
    print(f"\n  Results in: {run.path}")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CIFAR-10 3-block CNN benchmark with TAGI"
    )
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sigma_v", type=float, default=0.01)
    parser.add_argument("--gain_w", type=float, default=0.1)
    parser.add_argument("--gain_b", type=float, default=0.1)
    parser.add_argument("--pool", choices=["avg", "stride"], default="avg",
                        help="Downsample: avg (AvgPool2D k=2) or stride (stride-2 conv)")
    parser.add_argument("--no_augment", dest="augment", action="store_false",
                        help="Disable GPU augmentation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(augment=True)
    args = parser.parse_args()
    main(**vars(args))
