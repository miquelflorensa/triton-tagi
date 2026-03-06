"""
OOD Evaluation: ResNet-18 on CIFAR-10 (in-distribution) vs SVHN (OOD)
======================================================================
Loads the latest checkpoint from run_logs/checkpoints/, then produces:
  1. ECE reliability bar plot — confidence vs accuracy per bin (CIFAR-10 only)
  2. Predictive-entropy histogram — CIFAR-10 vs SVHN side-by-side

Usage:
    conda run -n cuTAGI python eval_ood_resnet18.py [--checkpoint PATH]
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from run_resnet18 import build_resnet18
from src.layers import ResBlock

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────
#  Checkpoint loading  (mirrors save_checkpoint in run_resnet18.py)
# ──────────────────────────────────────────────────────────────────────

def load_checkpoint(net, path: str):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ck["state"]

    def _restore(layer, d):
        if "mw" in d:
            layer.mw.data.copy_(d["mw"].to(DEVICE))
            layer.Sw.data.copy_(d["Sw"].to(DEVICE))
            layer.mb.data.copy_(d["mb"].to(DEVICE))
            layer.Sb.data.copy_(d["Sb"].to(DEVICE))
        if "running_mean" in d:
            layer.running_mean.copy_(d["running_mean"].to(DEVICE))
            layer.running_var.copy_(d["running_var"].to(DEVICE))

    for i, layer in enumerate(net.layers):
        if isinstance(layer, ResBlock):
            for j, sub in enumerate(layer._learnable):
                key = f"layer_{i}_sub_{j}_{type(sub).__name__}"
                if key in state:
                    _restore(sub, state[key])
        else:
            key = f"layer_{i}_{type(layer).__name__}"
            if key in state:
                _restore(layer, state[key])

    return ck["epoch"]


def find_latest_checkpoint(ckpt_dir: str) -> str:
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return paths[-1]


# ──────────────────────────────────────────────────────────────────────
#  Data loaders
# ──────────────────────────────────────────────────────────────────────

# CIFAR-10 normalization used during training
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def load_cifar10_test(data_dir="data"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])
    ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    x = torch.stack([img for img, _ in ds]).to(DEVICE)
    y = torch.tensor([lbl for _, lbl in ds], device=DEVICE)
    return x, y


def load_svhn_test(data_dir="data"):
    """Load SVHN test split with the same CIFAR-10 normalization for OOD evaluation."""
    tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])
    ds = datasets.SVHN(data_dir, split="test", download=True, transform=tf)
    x = torch.stack([img for img, _ in ds]).to(DEVICE)
    y = torch.tensor([lbl for _, lbl in ds], device=DEVICE)
    return x, y


# ──────────────────────────────────────────────────────────────────────
#  Inference utilities
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_probs(net, x, batch_size=256):
    """Return predicted probabilities (N, C) by batching through the network."""
    net.eval()
    all_probs = []
    for i in range(0, len(x), batch_size):
        mu, _ = net.forward(x[i:i + batch_size])
        all_probs.append(mu.cpu())
    return torch.cat(all_probs)   # (N, C)


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    """H[p] = -Σ p_c log(p_c), clipped to avoid log(0)."""
    p = probs.clamp(min=1e-12)
    return -(p * p.log()).sum(dim=-1)


# ──────────────────────────────────────────────────────────────────────
#  ECE reliability diagram
# ──────────────────────────────────────────────────────────────────────

def compute_ece_bins(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    """
    Returns per-bin accuracy, confidence, fraction of samples, and overall ECE.
    """
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    N = len(conf)

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    bin_acc   = torch.zeros(n_bins)
    bin_conf  = torch.zeros(n_bins)
    bin_frac  = torch.zeros(n_bins)

    ece = 0.0
    for k, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (conf >= lo) & (conf < hi)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        bin_acc[k]  = correct[mask].mean().item()
        bin_conf[k] = conf[mask].mean().item()
        bin_frac[k] = n_k / N
        ece += bin_frac[k] * abs(bin_acc[k] - bin_conf[k])

    return bin_acc.numpy(), bin_conf.numpy(), bin_frac.numpy(), float(ece)


def plot_ece_bar(bin_acc, bin_conf, bin_frac, ece, n_bins, out_path):
    """
    Reliability diagram: gap bars (|accuracy − confidence|) coloured by
    whether the model is over- or under-confident.
    """
    centers = np.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins)
    width   = 1.0 / n_bins * 0.9

    fig, ax = plt.subplots(figsize=(7, 5))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")

    # Accuracy bars
    ax.bar(centers, bin_acc, width=width, alpha=0.7,
           color="#4c72b0", label="Accuracy", zorder=3)

    # Gap bars (over/under confidence)
    gap = bin_conf - bin_acc   # positive → over-confident
    for k, (c, g) in enumerate(zip(centers, gap)):
        if abs(g) < 1e-6:
            continue
        color = "#dd4949" if g > 0 else "#2ca02c"
        bottom = bin_acc[k] if g > 0 else bin_conf[k]
        ax.bar(c, abs(g), width=width, bottom=bottom,
               alpha=0.55, color=color, zorder=2)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        plt.Line2D([0], [0], color="k", ls="--", lw=1.2, label="Perfect calibration"),
        Patch(facecolor="#4c72b0", alpha=0.7, label="Accuracy"),
        Patch(facecolor="#dd4949", alpha=0.55, label="Over-confident gap"),
        Patch(facecolor="#2ca02c", alpha=0.55, label="Under-confident gap"),
    ], fontsize=9)

    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"Reliability Diagram — CIFAR-10 Test Set\nECE = {ece:.4f}", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", lw=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ECE bar plot saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────
#  Entropy histogram
# ──────────────────────────────────────────────────────────────────────

def plot_entropy_histogram(entropy_id: torch.Tensor, entropy_ood: torch.Tensor,
                           out_path: str):
    """
    Overlapping entropy histograms for in-distribution (CIFAR-10) vs OOD (SVHN).
    """
    h_id  = entropy_id.numpy()
    h_ood = entropy_ood.numpy()

    max_H = np.log(10)  # maximum entropy for 10 classes
    bins  = np.linspace(0, max_H, 51)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(h_id,  bins=bins, density=True, alpha=0.55,
            color="#4c72b0", label=f"CIFAR-10 (ID)  n={len(h_id):,}")
    ax.hist(h_ood, bins=bins, density=True, alpha=0.55,
            color="#dd4949", label=f"SVHN (OOD)     n={len(h_ood):,}")

    ax.axvline(h_id.mean(),  color="#4c72b0", lw=1.8, ls="--",
               label=f"ID mean  = {h_id.mean():.3f}")
    ax.axvline(h_ood.mean(), color="#dd4949", lw=1.8, ls="--",
               label=f"OOD mean = {h_ood.mean():.3f}")

    ax.set_xlabel("Predictive Entropy  H[p]", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Predictive Entropy: CIFAR-10 (ID) vs SVHN (OOD)\nResNet-18 / TAGI-Triton",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", lw=0.5, alpha=0.5)

    # Annotate AUROC (area-under ROC for OOD detection using entropy as score)
    from sklearn.metrics import roc_auc_score
    scores = np.concatenate([h_id,  h_ood])
    labels = np.concatenate([np.zeros(len(h_id)), np.ones(len(h_ood))])
    auroc  = roc_auc_score(labels, scores)
    ax.text(0.98, 0.95, f"AUROC = {auroc:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Entropy histogram saved → {out_path}")
    return auroc


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OOD evaluation for TAGI ResNet-18")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint .pt file (default: latest in run_logs/checkpoints/)")
    parser.add_argument("--ckpt-dir", default="run_logs/checkpoints")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir",  default="run_logs")
    parser.add_argument("--n-bins",   type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Find checkpoint ──
    ckpt_path = args.checkpoint or find_latest_checkpoint(args.ckpt_dir)
    print(f"\n  Checkpoint : {ckpt_path}")

    # ── Build & load model ──
    print("  Building ResNet-18...")
    net = build_resnet18(num_classes=10, head="remax", device=DEVICE,
                         g_min=0.10, g_max=0.10)
    epoch = load_checkpoint(net, ckpt_path)
    print(f"  Loaded epoch {epoch}  |  device={DEVICE}")

    # ── Load datasets ──
    print("\n  Loading CIFAR-10 test set...")
    x_cf, y_cf = load_cifar10_test(args.data_dir)
    print(f"  CIFAR-10: {x_cf.shape[0]:,} samples")

    print("  Loading SVHN test set...")
    x_sv, y_sv = load_svhn_test(args.data_dir)
    print(f"  SVHN    : {x_sv.shape[0]:,} samples")

    # ── Forward passes ──
    print("\n  Running inference on CIFAR-10...")
    probs_cf = get_probs(net, x_cf, args.batch_size)

    print("  Running inference on SVHN...")
    probs_sv = get_probs(net, x_sv, args.batch_size)

    # ── Accuracy ──
    acc_cf = (probs_cf.argmax(1) == y_cf.cpu()).float().mean().item()
    acc_sv = (probs_sv.argmax(1) == y_sv.cpu()).float().mean().item()
    print(f"\n  CIFAR-10 accuracy : {acc_cf*100:.2f}%")
    print(f"  SVHN accuracy     : {acc_sv*100:.2f}%  (expect low — OOD)")

    # ── ECE bar plot (CIFAR-10 only) ──
    print("\n  Computing ECE...")
    bin_acc, bin_conf, bin_frac, ece = compute_ece_bins(
        probs_cf, y_cf.cpu(), n_bins=args.n_bins)
    print(f"  ECE (CIFAR-10) : {ece:.4f}")

    ece_path = os.path.join(args.out_dir, f"resnet18_ece_epoch{epoch:04d}.png")
    plot_ece_bar(bin_acc, bin_conf, bin_frac, ece, args.n_bins, ece_path)

    # ── Entropy histogram ──
    print("\n  Computing predictive entropy...")
    ent_cf = predictive_entropy(probs_cf)
    ent_sv = predictive_entropy(probs_sv)
    print(f"  CIFAR-10 mean entropy : {ent_cf.mean():.4f}")
    print(f"  SVHN     mean entropy : {ent_sv.mean():.4f}")

    ent_path = os.path.join(args.out_dir, f"resnet18_entropy_ood_epoch{epoch:04d}.png")
    auroc = plot_entropy_histogram(ent_cf, ent_sv, ent_path)
    print(f"  AUROC (entropy OOD)   : {auroc:.4f}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
