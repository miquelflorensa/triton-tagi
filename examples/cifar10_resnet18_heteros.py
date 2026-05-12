"""
CIFAR-10 Classification — ResNet-18 — TAGI-V Heteroscedastic Head.

Architecture (CIFAR-10 adaptation — 3×3 stem, no max-pool):
    Stem:    Conv(3→64, 3×3, p=1) → ReLU → BN           [32×32]
    Stage 1: ResBlock(64,  64,  s=1) × 2                 [32×32]
    Stage 2: ResBlock(64,  128, s=2) + ResBlock(128, 128) [16×16]
    Stage 3: ResBlock(128, 256, s=2) + ResBlock(256, 256) [8×8]
    Stage 4: ResBlock(256, 512, s=2) + ResBlock(512, 512) [4×4]
    Head:    AvgPool(4) → Flatten → FC(512→20) → EvenSoftplus(10)

20 outputs = 10×(mean, noise_var) interleaved.
Targets: +C for the true class, -C for all others (default C=3).
Inference: argmax over the 10 mean outputs (even columns).

Usage:
    python examples/cifar10_resnet18_heteros.py
    python examples/cifar10_resnet18_heteros.py --update_mode precision --rho_mode full
    python examples/cifar10_resnet18_heteros.py --n_epochs 50 --help
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
    EvenSoftplus,
    Flatten,
    Linear,
    ReLU,
    ResBlock,
    Sequential,
)
from triton_tagi.checkpoint import RunDir


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2470, 0.2435, 0.2616)


def load_cifar10(
    data_dir: str = "data",
    device: torch.device = torch.device("cpu"),
    target_scale: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load CIFAR-10 as (N, 3, 32, 32) tensors on ``device``.

    Returns:
        x_train (50000,3,32,32), y_train_pm (50000,10) +/-C targets,
        x_test  (10000,3,32,32), y_test_labels (10000,).
    """
    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=norm)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=norm)

    x_train = torch.stack([img for img, _ in train_ds]).to(device)
    y_train = torch.tensor([lbl for _, lbl in train_ds], device=device)
    x_test  = torch.stack([img for img, _ in test_ds]).to(device)
    y_test  = torch.tensor([lbl for _, lbl in test_ds], device=device)

    C = target_scale
    y_train_pm = torch.full((len(y_train), 10), -C, device=device)
    y_train_pm.scatter_(1, y_train.unsqueeze(1), C)

    return x_train, y_train_pm, x_test, y_test


# ---------------------------------------------------------------------------
#  GPU augmentation
# ---------------------------------------------------------------------------

def gpu_augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    B, C, H, W = x.shape
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    top  = torch.randint(0, 2 * pad, (B,), device=x.device)
    left = torch.randint(0, 2 * pad, (B,), device=x.device)
    rows = top.unsqueeze(1)  + torch.arange(H, device=x.device).unsqueeze(0)
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
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            mu, _ = net.forward(x_test[i : i + batch_size])
            correct += (mu[:, 0::2].argmax(dim=1) == y_labels[i : i + batch_size]).sum().item()
    net.train()
    return correct / len(x_test)


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train(
    net: Sequential,
    x_train: torch.Tensor,
    y_train_pm: torch.Tensor,
    x_test: torch.Tensor,
    y_test_labels: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    augment: bool,
    device: torch.device,
    run: RunDir,
    config: dict,
) -> float:
    print(f"\n  {'Epoch':>5}  {'Test Acc':>9}  {'Time':>7}")
    print("  " + "─" * 26)

    best_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(x_train.size(0), device=device)
        x_s, y_s = x_train[perm], y_train_pm[perm]

        for i in range(0, len(x_s), batch_size):
            xb = x_s[i : i + batch_size]
            if augment:
                xb = gpu_augment(xb)
            # sigma_v is passed but ignored by the heteros kernel
            net.step(xb, y_s[i : i + batch_size], sigma_v=0.0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        acc = evaluate(net, x_test, y_test_labels)
        best_acc = max(best_acc, acc)
        print(f"  {epoch:5d}  {acc*100:8.2f}%  {wall:6.2f}s")
        run.append_metrics(epoch, test_acc=acc, wall_s=wall)

        if epoch % config.get("checkpoint_interval", 10) == 0 or epoch == n_epochs:
            run.save_checkpoint(net, epoch, config)

    print("  " + "─" * 34)
    print(f"  Best test accuracy: {best_acc*100:.2f}%")
    return best_acc


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(
    n_epochs: int = 100,
    batch_size: int = 128,
    gain_w: float = 0.1,
    gain_b: float = 0.1,
    target_scale: float = 3.0,
    augment: bool = True,
    update_mode: str = "cap",
    rho_mode: str = "full",
    chi_max: float = 1e9,
    data_dir: str = "data",
    checkpoint_interval: int = 10,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    torch.manual_seed(seed)
    dev = torch.device(device)

    print("=" * 64)
    print("  CIFAR-10 — ResNet-18 — TAGI-V Heteroscedastic Head")
    print("  Stem+4 stages(64→128→256→512)+GAP → FC(512→20) → EvenSoftplus")
    chi_str = f"{chi_max:.0e}" if chi_max < 1e8 else "inf"
    print(f"  update={update_mode}/{rho_mode}  chi_max={chi_str}  gain={gain_w}  C=±{target_scale}")
    print("=" * 64)
    if device == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    print(f"\n  Loading CIFAR-10 from '{data_dir}'...", flush=True)
    x_train, y_train_pm, x_test, y_test_labels = load_cifar10(data_dir, dev, target_scale)
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")

    config: dict = {
        "dataset": "cifar10_heteros",
        "arch": "resnet18_heteros",
        "optimizer": "tagi-v",
        "update_mode": update_mode,
        "rho_mode": rho_mode,
        "chi_max": chi_max,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "gain_w": gain_w,
        "gain_b": gain_b,
        "target_scale": target_scale,
        "augment": augment,
        "checkpoint_interval": checkpoint_interval,
        "seed": seed,
        "device": device,
    }

    run = RunDir("cifar10_heteros", "resnet18_heteros", "tagi-v")
    run.save_config(config)
    print(f"  Run directory: {run.path}")

    kw = {"device": dev, "gain_w": gain_w, "gain_b": gain_b}
    net = Sequential(
        [
            # Stem: 32×32
            Conv2D(3, 64, 3, stride=1, padding=1, **kw),

            ReLU(),
            BatchNorm2D(64, **kw),
            # Stage 1: 32×32
            ResBlock(64,  64,  stride=1, **kw),
            ResBlock(64,  64,  stride=1, **kw),
            # Stage 2: 32→16
            ResBlock(64,  128, stride=2, **kw),
            ResBlock(128, 128, stride=1, **kw),
            # Stage 3: 16→8
            ResBlock(128, 256, stride=2, **kw),
            ResBlock(256, 256, stride=1, **kw),
            # Stage 4: 8→4
            ResBlock(256, 512, stride=2, **kw),
            ResBlock(512, 512, stride=1, **kw),
            # Head: 20 outputs = 10×(mean, noise_var)
            AvgPool2D(4),
            Flatten(),
            Linear(512, 20, **kw),
            EvenSoftplus(half_width=10),
        ],
        device=dev,
        update_mode=update_mode,
        rho_mode=rho_mode,
        chi_max=chi_max,
    )
    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")
    print(f"\n  Epochs: {n_epochs}  |  Batch: {batch_size}  |  augment: {augment}")

    best_acc = train(
        net, x_train, y_train_pm, x_test, y_test_labels,
        n_epochs, batch_size, augment, dev, run, config,
    )

    print(f"\n  Results in: {run.path}")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CIFAR-10 ResNet-18 with TAGI-V heteroscedastic head"
    )
    parser.add_argument("--n_epochs",           type=int,   default=100)
    parser.add_argument("--batch_size",         type=int,   default=128)
    parser.add_argument("--gain_w",             type=float, default=0.1)
    parser.add_argument("--gain_b",             type=float, default=0.1)
    parser.add_argument("--target_scale",       type=float, default=3.0)
    parser.add_argument("--update_mode",        type=str,   default="cap",
                        choices=["cap", "precision"])
    parser.add_argument("--rho_mode",           type=str,   default="full",
                        choices=["full", "sqrt_batch", "batch_avg", "custom"])
    parser.add_argument("--chi_max",            type=float, default=1e9,
                        help="Max per-parameter information ratio (default 1e9≈inf). "
                             "Set to ~1-10 for CNN layers to prevent spatial summation blow-up.")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--data_dir",           type=str,   default="data")
    parser.add_argument("--checkpoint_interval",type=int,   default=10)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--device",             type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(augment=True)
    args = parser.parse_args()
    main(**vars(args))
