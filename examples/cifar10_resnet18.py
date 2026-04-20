"""
CIFAR-10 Classification — ResNet-18 — triton-tagi example.

Architecture (CIFAR-10 adaptation — 3×3 stem, no max-pool):
    Stem:    Conv(3→64, 3×3, p=1) → ReLU → BN           [32×32]
    Stage 1: ResBlock(64,  64,  s=1) × 2                 [32×32]
    Stage 2: ResBlock(64,  128, s=2) + ResBlock(128, 128) [16×16]
    Stage 3: ResBlock(128, 256, s=2) + ResBlock(256, 256) [8×8]
    Stage 4: ResBlock(256, 512, s=2) + ResBlock(512, 512) [4×4]
    Head:    AvgPool(4) → Flatten → FC(512→10) → Remax

Each ResBlock: Conv(3×3,s) → ReLU → BN → Conv(3×3) → ReLU → BN + shortcut.
Projection shortcut (stride>1 or ch mismatch): Conv(2×2, s=2) → ReLU → BN.

σ_v is decayed each epoch: σ_v(t) = max(σ_v · rate^t, σ_v_min).
Matches cuTAGI's exponential_scheduler (default: 1.0 → 0.3 at ×0.95/ep).

Usage:
    python examples/cifar10_resnet18.py
    python examples/cifar10_resnet18.py --n_epochs 30
    python examples/cifar10_resnet18.py --data_dir /path/to/data --no_augment
    python examples/cifar10_resnet18.py --help
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
    ResBlock,
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

def evaluate(
    net: Sequential,
    x_test: torch.Tensor,
    y_labels: torch.Tensor,
    batch_size: int = 256,
) -> float:
    """Return test accuracy."""
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            mu, _ = net.forward(x_test[i : i + batch_size])
            correct += (mu.argmax(dim=1) == y_labels[i : i + batch_size]).sum().item()
    net.train()
    return correct / len(x_test)


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
    augment: bool,
    device: torch.device,
    run: RunDir,
    config: dict,
) -> float:
    """Training loop with optional GPU augmentation. Returns best test accuracy."""
    print(f"\n  {'Epoch':>5}  {'Test Acc':>9}  {'Time':>7}")
    print("  " + "─" * 26)

    best_acc = 0.0

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

        acc = evaluate(net, x_test, y_test_labels)
        best_acc = max(best_acc, acc)
        print(f"  {epoch:5d}  {acc*100:8.2f}%  {wall:6.2f}s")
        run.append_metrics(epoch, test_acc=acc, sigma_v=sigma_v, wall_s=wall)

        if epoch % config.get("checkpoint_interval", 10) == 0 or epoch == n_epochs:
            run.save_checkpoint(net, epoch, config)

    print("  " + "─" * 34)
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

    epochs, accs = [], []
    with open(run.metrics_csv) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            accs.append(float(row["test_acc"]) * 100)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, accs, color="C0", linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("CIFAR-10 ResNet-18 — TAGI test accuracy")
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
    n_epochs: int = 100,
    batch_size: int = 128,
    sigma_v: float = 0.05,
    gain_w: float = 0.1,
    gain_b: float = 0.1,
    augment: bool = True,
    data_dir: str = "data",
    checkpoint_interval: int = 10,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """CIFAR-10 ResNet-18 benchmark.

    Args:
        sigma_v: Observation noise (fixed).
        augment: Apply random flip + crop augmentation each batch.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)

    print("=" * 60)
    print("  CIFAR-10 Classification — ResNet-18 — triton-tagi")
    print("  Stem+4 stages(64→128→256→512)+GAP → FC(512→10) → Remax")
    print("=" * 60)
    if device == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    # ── Data ──
    print(f"\n  Loading CIFAR-10 from '{data_dir}'...", flush=True)
    x_train, y_train_oh, _, x_test, y_test_labels = load_cifar10(data_dir, dev)
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    print(f"  Input shape: {tuple(x_train.shape[1:])}")

    # ── Config ──
    config: dict = {
        "dataset": "cifar10",
        "arch": "resnet18",
        "optimizer": "tagi",
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "sigma_v": sigma_v,
        "gain_w": gain_w,
        "gain_b": gain_b,
        "augment": augment,
        "checkpoint_interval": checkpoint_interval,
        "seed": seed,
        "device": device,
        "triton_tagi_version": "0.1.0",
    }

    # ── RunDir ──
    run = RunDir("cifar10", "resnet18", "tagi")
    run.save_config(config)
    print(f"  Run directory: {run.path}")

    # ── Network ──
    kw = {"device": dev, "gain_w": gain_w, "gain_b": gain_b}

    net = Sequential(
        [
            # Stem: 32×32
            Conv2D(3, 64, 3, stride=1, padding=1, **kw),
            ReLU(),
            BatchNorm2D(64, **kw),
            # Stage 1: 32×32
            ResBlock(64, 64, stride=1, **kw),
            ResBlock(64, 64, stride=1, **kw),
            # Stage 2: 32→16
            ResBlock(64, 128, stride=2, **kw),
            ResBlock(128, 128, stride=1, **kw),
            # Stage 3: 16→8
            ResBlock(128, 256, stride=2, **kw),
            ResBlock(256, 256, stride=1, **kw),
            # Stage 4: 8→4
            ResBlock(256, 512, stride=2, **kw),
            ResBlock(512, 512, stride=1, **kw),
            # Head
            AvgPool2D(4),           # 4×4 → 1×1
            Flatten(),              # 512
            Linear(512, 10, **kw),
            Remax(),
        ],
        device=dev,
    )
    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")
    print(f"\n  Epochs: {n_epochs}  |  Batch: {batch_size}  |  σ_v: {sigma_v}  |  augment: {augment}")

    # ── Train ──
    best_acc = train(
        net, x_train, y_train_oh, x_test, y_test_labels,
        n_epochs, batch_size, sigma_v, augment, dev, run, config,
    )

    # ── Figure ──
    save_figure(run)
    print(f"\n  Results in: {run.path}")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CIFAR-10 ResNet-18 benchmark with TAGI"
    )
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sigma_v", type=float, default=0.05)
    parser.add_argument("--gain_w", type=float, default=0.1)
    parser.add_argument("--gain_b", type=float, default=0.1)
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
