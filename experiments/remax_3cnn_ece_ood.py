"""
Remax paper rebuttal: 3-block CNN on CIFAR-10 with triton-tagi.

Trains the BN+AvgPool 3CNN + Remax (triton-tagi's standard CIFAR example
architecture) for 50 epochs, then evaluates:
  - Test accuracy + NLL
  - ECE (15 bins) on CIFAR-10 test
  - OOD AUROC (predictive entropy, total variance) for CIFAR-10 -> SVHN

~5 min on one 4070 Ti SUPER.

Usage:
    python experiments/remax_3cnn_ece_ood.py
    python experiments/remax_3cnn_ece_ood.py --n_epochs 30 --sigma_v 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.cifar10_cnn import (
    _CIFAR_MEAN,
    _CIFAR_STD,
    gpu_augment,
    load_cifar10,
)
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
#  Network
# ---------------------------------------------------------------------------
def build_net(device: torch.device, gain_w: float, gain_b: float) -> Sequential:
    """Exact trtc-bnn `build_3block_cnn` architecture (paper Table 1).

    6 conv layers (3×3 wide + 4×4 stride-2) with MixtureReLU, no BN, no pool.
    Channels: 3→64→64→128→128→256→256. Classifier: 4096→512→10.
    """
    kw = {"device": device, "gain_w": gain_w, "gain_b": gain_b}
    return Sequential(
        [
            # Block 1
            Conv2D(3, 64, 3, padding=1, **kw),                 # 32×32×64
            ReLU(),
            Conv2D(64, 64, 4, stride=2, padding=1, **kw),      # 16×16×64
            ReLU(),
            # Block 2
            Conv2D(64, 128, 3, padding=1, **kw),               # 16×16×128
            ReLU(),
            Conv2D(128, 128, 4, stride=2, padding=1, **kw),    # 8×8×128
            ReLU(),
            # Block 3
            Conv2D(128, 256, 3, padding=1, **kw),              # 8×8×256
            ReLU(),
            Conv2D(256, 256, 4, stride=2, padding=1, **kw),    # 4×4×256
            ReLU(),
            # Classifier
            Flatten(),
            Linear(256 * 4 * 4, 512, **kw),
            ReLU(),
            Linear(512, 10, **kw),
            Remax(),
        ],
        device=device,
    )


# ---------------------------------------------------------------------------
#  Inference (preserves variance)
# ---------------------------------------------------------------------------
def predict_probs_var(
    net: Sequential, x: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (mu_probs, var_probs) both (N, K)."""
    net.eval()
    mus, vars_ = [], []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            mu, var = net.forward(x[i : i + batch_size])
            mus.append(mu)
            vars_.append(var)
    net.train()
    mu = torch.cat(mus, dim=0)
    var = torch.cat(vars_, dim=0)
    # Renormalize mean to sum to 1 (MM-Remax scaling step)
    mu_clamped = mu.clamp(min=0)
    s = mu_clamped.sum(dim=1, keepdim=True).clamp(min=1e-12)
    mu_norm = mu_clamped / s
    # Scale variance by the same s² so it lives on the normalized simplex
    var_norm = var / (s**2)
    return mu_norm, var_norm


def _ece_15(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    confidences, preds = probs.max(dim=1)
    correct = (preds == y).float()
    bounds = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = 0.0
    N = len(y)
    for i in range(n_bins):
        lo, hi = bounds[i], bounds[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences <= hi)
        n = mask.sum().item()
        if n > 0:
            ece += (n / N) * abs(
                confidences[mask].mean().item() - correct[mask].mean().item()
            )
    return ece


def _nll(probs: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    p_true = probs[torch.arange(len(y), device=probs.device), y].clamp(min=eps)
    return (-torch.log(p_true)).mean().item()


# ---------------------------------------------------------------------------
#  OOD (SVHN)
# ---------------------------------------------------------------------------
def load_svhn_test(data_dir: str, device: torch.device) -> torch.Tensor:
    """Return SVHN test images (26032, 3, 32, 32), normalized w/ CIFAR stats."""
    norm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)]
    )
    ds = datasets.SVHN(data_dir, split="test", download=True, transform=norm)
    x = torch.stack([img for img, _ in ds]).to(device)
    return x


def _auroc(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
    """AUROC treating OOD as positive class (higher score = OOD)."""
    id_scores = id_scores.cpu().numpy().astype(np.float64)
    ood_scores = ood_scores.cpu().numpy().astype(np.float64)
    scores = np.concatenate([id_scores, ood_scores])
    y = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    order = np.argsort(-scores, kind="stable")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    tpr = np.concatenate([[0.0], tp / y_sorted.sum()])
    fpr = np.concatenate([[0.0], fp / (1.0 - y_sorted).sum()])
    return float(np.trapezoid(tpr, fpr))


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------
def plot_reliability(
    probs: torch.Tensor, y: torch.Tensor, save_path: Path, n_bins: int = 15
) -> None:
    confidences, preds = probs.max(dim=1)
    correct = (preds == y).float()
    bounds = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    bin_conf, bin_acc, bin_cnt = [], [], []
    for i in range(n_bins):
        lo, hi = bounds[i], bounds[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences <= hi)
        n = mask.sum().item()
        if n > 0:
            bin_conf.append(confidences[mask].mean().item())
            bin_acc.append(correct[mask].mean().item())
            bin_cnt.append(n)
        else:
            bin_conf.append(0.0)
            bin_acc.append(0.0)
            bin_cnt.append(0)
    bin_conf = np.asarray(bin_conf)
    bin_acc = np.asarray(bin_acc)
    bin_cnt = np.asarray(bin_cnt)
    centers = np.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins)
    ece = float(np.sum(np.abs(bin_conf - bin_acc) * bin_cnt / max(bin_cnt.sum(), 1)))

    fig, ax = plt.subplots(figsize=(6, 6))
    w = 1.0 / n_bins
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    for c, a, cf, n in zip(centers, bin_acc, bin_conf, bin_cnt):
        if n == 0:
            continue
        ax.bar(c, a, width=w * 0.9, color="#4A90E2", alpha=0.8, edgecolor="k", linewidth=0.3)
        gap = cf - a
        if gap > 0:
            ax.bar(c, gap, width=w * 0.9, bottom=a, color="#E26B6B", alpha=0.6, edgecolor="k", linewidth=0.3)
        else:
            ax.bar(c, -gap, width=w * 0.9, bottom=a + gap, color="#8AD48A", alpha=0.6, edgecolor="k", linewidth=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram — ECE={ece:.4f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_ood_histograms(
    id_ent: torch.Tensor,
    ood_ent: torch.Tensor,
    id_tv: torch.Tensor,
    ood_tv: torch.Tensor,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, id_s, ood_s, title in [
        (axes[0], id_ent, ood_ent, "Predictive entropy"),
        (axes[1], id_tv, ood_tv, "Total probability variance"),
    ]:
        id_np = id_s.cpu().numpy()
        ood_np = ood_s.cpu().numpy()
        ax.hist(id_np, bins=60, density=True, alpha=0.55, color="#4A90E2", label="CIFAR-10 (ID)")
        ax.hist(ood_np, bins=60, density=True, alpha=0.55, color="#E24A4A", label="SVHN (OOD)")
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Train loop (ECE logged per-epoch)
# ---------------------------------------------------------------------------
def train_loop(
    net: Sequential,
    x_tr: torch.Tensor,
    y_tr_oh: torch.Tensor,
    y_tr: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    sigma_v: float,
    augment: bool,
    run: RunDir,
    config: dict,
    eval_batch: int = 256,
) -> dict:
    """Train, selecting the best checkpoint by **val NLL** (not val accuracy).

    NLL rewards well-calibrated models, accuracy rewards overconfident ones.
    For BNNs where probabilities sharpen as training continues, selecting by
    accuracy finds overconfident late-epoch weights; selecting by NLL finds
    earlier checkpoints with better ECE + OOD signal.
    """
    best_val_nll = float("inf")
    best_ck = None
    print(f"  {'ep':>3}  {'train':>6}  {'val':>6}  {'vnll':>6}  {'test':>6}  {'tnll':>6}  {'tece':>6}  {'wall':>6}")
    print("  " + "─" * 66)
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(x_tr.size(0), device=x_tr.device)
        x_s, y_s = x_tr[perm], y_tr_oh[perm]
        for i in range(0, len(x_s), batch_size):
            xb = x_s[i : i + batch_size]
            if augment:
                xb = gpu_augment(xb)
            net.step(xb, y_s[i : i + batch_size], sigma_v)
        if x_tr.device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        # Eval
        tr_idx = torch.randperm(x_tr.size(0), device=x_tr.device)[:5000]
        tr_probs, _ = predict_probs_var(net, x_tr[tr_idx], eval_batch)
        tr_acc = (tr_probs.argmax(1) == y_tr[tr_idx]).float().mean().item()
        val_probs, _ = predict_probs_var(net, x_val, eval_batch)
        val_acc = (val_probs.argmax(1) == y_val).float().mean().item()
        val_nll = _nll(val_probs, y_val)
        te_probs, _ = predict_probs_var(net, x_te, eval_batch)
        te_acc = (te_probs.argmax(1) == y_te).float().mean().item()
        te_ece = _ece_15(te_probs, y_te)
        te_nll = _nll(te_probs, y_te)
        print(
            f"  {epoch:3d}  {tr_acc*100:5.2f}%  {val_acc*100:5.2f}%  "
            f"{val_nll:6.4f}  {te_acc*100:5.2f}%  {te_nll:6.4f}  {te_ece:6.4f}  {wall:5.1f}s"
        )
        run.append_metrics(
            epoch,
            train_acc=tr_acc,
            val_acc=val_acc,
            val_nll=val_nll,
            test_acc=te_acc,
            test_nll=te_nll,
            test_ece=te_ece,
            sigma_v=sigma_v,
            wall_s=wall,
        )
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_ck = run.save_checkpoint(net, epoch, config)
            print(f"         ↳ new best (val NLL {val_nll:.4f})")
    return {
        "best_val_nll": best_val_nll,
        "best_ck_path": str(best_ck) if best_ck else None,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else '—'}")

    # Data: split train into 90% train / 10% val (deterministic by seed)
    print("Loading CIFAR-10 ...")
    x_all, y_all_oh, y_all, x_te, y_te = load_cifar10(args.data_dir, device)
    n_total = len(x_all)
    split_gen = torch.Generator(device="cpu").manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=split_gen).to(device)
    n_val = int(args.val_split * n_total)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    x_tr = x_all[tr_idx]
    y_tr_oh = y_all_oh[tr_idx]
    y_tr = y_all[tr_idx]
    x_val = x_all[val_idx]
    y_val = y_all[val_idx]
    del x_all, y_all_oh, y_all
    print(f"  train {len(x_tr):,}  |  val {len(x_val):,}  |  test {len(x_te):,}")

    # Net
    net = build_net(device, args.gain_w, args.gain_b)
    print(net)
    print(f"  parameters: {net.num_parameters():,}")

    config = {
        "dataset": "cifar10",
        "arch": "3cnn_bn_avgpool_remax",
        "optimizer": "tagi",
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "sigma_v": args.sigma_v,
        "gain_w": args.gain_w,
        "gain_b": args.gain_b,
        "augment": args.augment,
        "seed": args.seed,
        "triton_tagi_version": "0.2.0",
    }
    run = RunDir("cifar10", "3cnn_bn_avg_remax_rebuttal", "tagi")
    run.save_config(config)
    print(f"Run dir: {run.path}")

    # Train — best checkpoint selected by val NLL
    print(f"Training {args.n_epochs} epochs, sigma_v={args.sigma_v}, augment={args.augment}")
    tr_out = train_loop(
        net, x_tr, y_tr_oh, y_tr, x_val, y_val, x_te, y_te,
        args.n_epochs, args.batch_size, args.sigma_v, args.augment,
        run, config,
    )
    print(f"\nBest val NLL: {tr_out['best_val_nll']:.4f}")
    print(f"Best checkpoint: {tr_out['best_ck_path']}")

    # Evaluate on best checkpoint
    if tr_out["best_ck_path"]:
        run.load_checkpoint(net, tr_out["best_ck_path"])
    net.eval()

    # ECE + accuracy + NLL on CIFAR-10 test
    print("\nCIFAR-10 test inference...")
    te_probs, te_var = predict_probs_var(net, x_te, 256)
    te_acc = (te_probs.argmax(1) == y_te).float().mean().item()
    te_ece = _ece_15(te_probs, y_te)
    te_nll = _nll(te_probs, y_te)
    print(f"  accuracy: {te_acc*100:.2f}%")
    print(f"  NLL:      {te_nll:.4f}")
    print(f"  ECE(15):  {te_ece:.4f}")
    plot_reliability(te_probs, y_te, run.figures / "reliability.png")

    # OOD (SVHN)
    print("\nSVHN (OOD) inference...")
    x_svhn = load_svhn_test(args.data_dir, device)
    print(f"  svhn: {len(x_svhn):,} samples")
    sv_probs, sv_var = predict_probs_var(net, x_svhn, 256)

    eps = 1e-12
    id_ent = -(te_probs.clamp(min=eps) * te_probs.clamp(min=eps).log()).sum(dim=1)
    ood_ent = -(sv_probs.clamp(min=eps) * sv_probs.clamp(min=eps).log()).sum(dim=1)
    id_tv = te_var.sum(dim=1)
    ood_tv = sv_var.sum(dim=1)
    id_1mc = 1.0 - te_probs.max(dim=1).values
    ood_1mc = 1.0 - sv_probs.max(dim=1).values

    auroc_ent = _auroc(id_ent, ood_ent)
    auroc_tv = _auroc(id_tv, ood_tv)
    auroc_1mc = _auroc(id_1mc, ood_1mc)

    print(f"  AUROC (entropy):      {auroc_ent:.4f}")
    print(f"  AUROC (total var):    {auroc_tv:.4f}")
    print(f"  AUROC (1 - max prob): {auroc_1mc:.4f}")
    print(f"  ID  entropy mean: {id_ent.mean().item():.4f} ± {id_ent.std().item():.4f}")
    print(f"  OOD entropy mean: {ood_ent.mean().item():.4f} ± {ood_ent.std().item():.4f}")
    print(f"  ID  total-var mean: {id_tv.mean().item():.4f}")
    print(f"  OOD total-var mean: {ood_tv.mean().item():.4f}")

    plot_ood_histograms(id_ent, ood_ent, id_tv, ood_tv, run.figures / "ood_hist.png")

    summary = {
        "checkpoint": tr_out["best_ck_path"],
        "cifar10_accuracy": te_acc,
        "cifar10_nll": te_nll,
        "cifar10_ece_15bins": te_ece,
        "auroc_entropy": auroc_ent,
        "auroc_total_var": auroc_tv,
        "auroc_one_minus_conf": auroc_1mc,
        "id_entropy_mean": float(id_ent.mean()),
        "id_entropy_std": float(id_ent.std()),
        "ood_entropy_mean": float(ood_ent.mean()),
        "ood_entropy_std": float(ood_ent.std()),
        "id_total_var_mean": float(id_tv.mean()),
        "ood_total_var_mean": float(ood_tv.mean()),
        "config": config,
    }
    (run.path / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        run.path / "raw_predictions.npz",
        cifar_probs=te_probs.cpu().numpy(),
        cifar_var=te_var.cpu().numpy(),
        cifar_y=y_te.cpu().numpy(),
        svhn_probs=sv_probs.cpu().numpy(),
        svhn_var=sv_var.cpu().numpy(),
    )
    print(f"\nSaved summary + raw predictions in {run.path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--sigma_v", type=float, default=0.05)
    p.add_argument("--gain_w", type=float, default=0.1)
    p.add_argument("--gain_b", type=float, default=0.1)
    p.add_argument("--no_augment", dest="augment", action="store_false")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.set_defaults(augment=True)
    args = p.parse_args()
    main(args)
