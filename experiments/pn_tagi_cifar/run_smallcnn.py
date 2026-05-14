"""CIFAR-10 small CNN — does the mature MNIST recipe transfer to convolutions?

Frozen recipe (the two confirmed theory anchors):
    update_rule    = capped_precision_normalized       (κ ≤ 1)
    cap_factor     = 1.0
    rho            = 1.0
    gain_w         = 1.0                              (DO NOT TUNE)
    Σ_Y           = σ²_A + μ_A(1 − μ_A) + σ²_floor    (Model C)
    σ_floor       ∈ {0.05, 0.1}

Architecture (matches examples/cifar10_cnn.py, optionally without BN):
    Conv(3→32, 5, pad=2)  → ReLU → [BN] → AvgPool(2)   [32 → 16]
    Conv(32→64, 5, pad=2) → ReLU → [BN] → AvgPool(2)   [16 → 8]
    Conv(64→64, 5, pad=2) → ReLU → [BN] → AvgPool(2)   [8 → 4]
    Flatten → Linear(1024, 256) → ReLU → Linear(256, 10) → Remax

This first run skips BatchNorm. If the model fails to learn, we retry
with BN as a separate config (``--use_bn``).

Augmentation: random horizontal flip + 4-pixel random crop on-device,
identical to the baseline example. Keeps the comparison apples-to-apples.

Per-batch diagnostics (averaged into per-epoch CSV):
    test_acc, test_CE
    ζ_dof  (innovation surprise / (K − 1))
    κ_p95, χ_p95, cap-active% (worst learnable layer)
    dead-ReLU max
    r_argmax (floor activation at the predicted class)
    per-layer activation variance (mean S_a after each layer)

Decision rule:
    no-BN works → recipe transfers, move on to ResNet
    fails by forward activation drift → init / normalisation is the
        next theory pillar
    fails by high ζ or κ → observation/update model needs more work
    σ_floor=0.1 ≫ σ_floor=0.05 → floor likely needs to be learned

Run::

    python experiments/pn_tagi_cifar/run_smallcnn.py
    python experiments/pn_tagi_cifar/run_smallcnn.py --use_bn  # retry with BN
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from triton_tagi import (
    AvgPool2D, BatchNorm2D, Conv2D, Flatten, Linear, ReLU, Remax, Sequential,
)
from triton_tagi.base import LearnableLayer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 10

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------------------
#  Data + augmentation (matches examples/cifar10_cnn.py)
# ---------------------------------------------------------------------------


def load_cifar10(data_dir: str, device: torch.device):
    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=norm)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=norm)
    x_train = torch.stack([img for img, _ in train_ds]).to(device)
    y_train = torch.tensor([lbl for _, lbl in train_ds], device=device)
    x_test = torch.stack([img for img, _ in test_ds]).to(device)
    y_test = torch.tensor([lbl for _, lbl in test_ds], device=device)
    y_train_oh = torch.zeros(len(y_train), 10, device=device)
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 1.0)
    return x_train, y_train_oh, y_train, x_test, y_test


def gpu_augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
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
#  Network — 3-block conv tower + small classifier
# ---------------------------------------------------------------------------


def build_cnn(*, use_bn, gain_w, device):
    layers = []
    for in_c, out_c in [(3, 32), (32, 64), (64, 64)]:
        layers.append(Conv2D(in_c, out_c, 5, stride=1, padding=2,
                             device=device, gain_w=gain_w, gain_b=gain_w))
        layers.append(ReLU())
        if use_bn:
            layers.append(BatchNorm2D(out_c, device=device, gain_w=gain_w, gain_b=gain_w))
        layers.append(AvgPool2D(2))
    layers.append(Flatten())
    layers.append(Linear(1024, 256, device=device, gain_w=gain_w, gain_b=gain_w))
    layers.append(ReLU())
    layers.append(Linear(256, 10, device=device, gain_w=gain_w, gain_b=gain_w))
    layers.append(Remax())
    return Sequential(
        layers, device=device,
        update_rule="capped_precision_normalized", rho=1.0, record_chi=False,
        cap_factor=1.0,
    )


# ---------------------------------------------------------------------------
#  Model C innovation with σ_floor
# ---------------------------------------------------------------------------


def innovation_C(y_oh, mu_Y, S_A, sigma_floor):
    cat = mu_Y * (1.0 - mu_Y)
    floor_sq = sigma_floor ** 2
    sigma_Y_sq = (S_A + cat + floor_sq).clamp_min(1e-8)
    residual = y_oh - mu_Y
    delta_mu_y = residual / sigma_Y_sq
    delta_S_y = -1.0 / sigma_Y_sq
    zeta_raw = ((residual ** 2) / sigma_Y_sq).sum(dim=-1).mean().item()
    zeta_dof = zeta_raw / (N_CLASSES - 1)
    r = floor_sq / (cat + floor_sq).clamp_min(1e-12)
    return delta_mu_y, delta_S_y, zeta_dof, r


# ---------------------------------------------------------------------------
#  Per-batch diagnostics
# ---------------------------------------------------------------------------


def per_batch_diagnostics(net):
    """Aggregate κ, χ, cap-active across all *learnable* layers (Conv + Linear).

    For BatchNorm2D / etc., we include them too — they're learnable, and
    we want a full picture.
    """
    kappa_layers, chi_layers, cap_layers, sw_means = [], [], [], []
    for layer in net.layers:
        if not isinstance(layer, LearnableLayer):
            continue
        # Some learnable layers nest sub-layers (MultiheadAttentionV2,
        # ResBlock). For the conv tower we don't have those.
        if not hasattr(layer, "Sw") or layer.Sw is None or layer.delta_mw is None:
            continue
        Sw = layer.Sw
        dmw = layer.delta_mw
        dSw = layer.delta_Sw
        sigma_w = Sw.sqrt().clamp_min(1e-8)
        kappa = dmw.abs() / sigma_w
        chi = -dSw / Sw.clamp_min(1e-12)
        cap_active = (dmw.abs() > sigma_w).float().mean().item()
        kappa_layers.append(float(kappa.quantile(0.95).item()))
        chi_layers.append(float(chi.quantile(0.95).item()))
        cap_layers.append(cap_active)
        sw_means.append(float(Sw.mean().item()))
    return {
        "kappa_p95_max":   max(kappa_layers) if kappa_layers else math.nan,
        "chi_p95_max":     max(chi_layers) if chi_layers else math.nan,
        "cap_active_mean": float(np.mean(cap_layers)) if cap_layers else math.nan,
        "cap_active_max":  max(cap_layers) if cap_layers else math.nan,
    }


def forward_with_activation_stats(net, x_batch):
    """Forward, returning (mu, var) at output plus dict of per-layer-output
    activation mean / variance and dead-ReLU fractions."""
    ma = x_batch
    Sa = torch.zeros_like(ma)
    layer_stats = []
    dead_relu = []
    for layer in net.layers:
        ma, Sa = layer.forward(ma, Sa)
        layer_stats.append({
            "layer_type": type(layer).__name__,
            "act_mean": float(ma.mean().item()),
            "act_var_pred": float(Sa.mean().item()),
            "act_var_emp": float(ma.var(unbiased=False).item()),
        })
        if isinstance(layer, ReLU):
            dead_relu.append(float((ma.abs() < 1e-6).float().mean().item()))
    return ma, Sa, layer_stats, max(dead_relu) if dead_relu else 0.0


# ---------------------------------------------------------------------------
#  Eval
# ---------------------------------------------------------------------------


def evaluate(net, x, y_labels, batch_size=512):
    net.eval()
    correct = 0
    ce_sum = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i : i + batch_size]
            lb = y_labels[i : i + batch_size]
            mu, _ = net.forward(xb)
            correct += int((mu.argmax(dim=1) == lb).sum().item())
            p_true = mu.clamp_min(1e-8).gather(1, lb.unsqueeze(1)).squeeze(1)
            ce_sum += float(-p_true.log().sum().item())
            n += xb.size(0)
    net.train()
    return correct / max(n, 1), ce_sum / max(n, 1)


def any_nan_or_inf(net):
    for layer in net.layers:
        if not hasattr(layer, "mw") or layer.mw is None:
            continue
        for t in (layer.mw, layer.Sw, getattr(layer, "mb", None), getattr(layer, "Sb", None)):
            if t is None:
                continue
            if not torch.isfinite(t).all():
                return True
    return False


# ---------------------------------------------------------------------------
#  Train one config
# ---------------------------------------------------------------------------


def train_one(*, sigma_floor, use_bn, gain_w, batch_size, n_epochs, seed,
              augment, data, device):
    torch.manual_seed(seed)
    net = build_cnn(use_bn=use_bn, gain_w=gain_w, device=device)
    n_params = net.num_parameters()

    n_train = data["x_train"].size(0)
    rows = []
    test_acc, test_ce = evaluate(net, data["x_test"], data["y_test_labels"])
    rows.append({
        "epoch": 0, "train_CE": math.nan,
        "test_acc": test_acc, "test_CE": test_ce,
        "zeta_dof": math.nan,
        "kappa_p95_max": math.nan, "chi_p95_max": math.nan,
        "cap_active_mean": math.nan, "cap_active_max": math.nan,
        "dead_relu_max": math.nan,
        "r_mean": math.nan, "r_p95": math.nan, "r_at_argmax": math.nan,
        "wall_s": 0.0, "any_nan": False,
    })

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, device=device)
        xs = data["x_train"][perm]
        ys = data["y_train_oh"][perm]
        labels = data["y_train_labels"][perm]

        ce_sum = 0.0
        n_seen = 0
        zeta_acc, kappa_acc, chi_acc, cap_mean_acc, cap_max_acc = [], [], [], [], []
        r_mean_acc, r_p95_acc, r_argmax_acc = [], [], []
        dead_acc = []

        net.train()
        for i in range(0, n_train, batch_size):
            xb = xs[i : i + batch_size]
            yb = ys[i : i + batch_size]
            lb = labels[i : i + batch_size]
            if xb.size(0) == 0:
                continue
            if augment:
                xb = gpu_augment(xb)

            # Forward — also record activation stats and dead-ReLU.
            mu_Y, S_A, layer_stats, dead_max = forward_with_activation_stats(net, xb)
            dead_acc.append(dead_max)

            # Innovation (Model C).
            dmu_y, dS_y, zeta_dof, r = innovation_C(yb, mu_Y, S_A, sigma_floor)

            # Backward.
            cur_dmu, cur_dvar = dmu_y, dS_y
            for layer in reversed(net.layers):
                cur_dmu, cur_dvar = layer.backward(cur_dmu, cur_dvar)

            # Diagnostics pre-update.
            diag = per_batch_diagnostics(net)
            zeta_acc.append(zeta_dof)
            kappa_acc.append(diag["kappa_p95_max"])
            chi_acc.append(diag["chi_p95_max"])
            cap_mean_acc.append(diag["cap_active_mean"])
            cap_max_acc.append(diag["cap_active_max"])
            r_mean_acc.append(float(r.mean().item()))
            r_p95_acc.append(float(r.quantile(0.95).item()))
            am = mu_Y.argmax(dim=1, keepdim=True)
            r_argmax_acc.append(float(r.gather(1, am).mean().item()))

            # CE.
            p_true = mu_Y.clamp_min(1e-8).gather(1, lb.unsqueeze(1)).squeeze(1)
            ce_sum += float(-p_true.log().sum().item())
            n_seen += xb.size(0)

            # Update.
            for layer in net.layers:
                if isinstance(layer, LearnableLayer):
                    layer.update(
                        net.cap_factor_override,
                        update_rule=net.update_rule,
                        rho=net.rho, record_chi=False,
                    )

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        nan = any_nan_or_inf(net)
        if nan:
            test_acc, test_ce = math.nan, math.nan
        else:
            test_acc, test_ce = evaluate(net, data["x_test"], data["y_test_labels"])

        rows.append({
            "epoch": epoch,
            "train_CE": ce_sum / max(n_seen, 1),
            "test_acc": test_acc, "test_CE": test_ce,
            "zeta_dof":         float(np.nanmean(zeta_acc)),
            "kappa_p95_max":    float(np.nanmean(kappa_acc)),
            "chi_p95_max":      float(np.nanmean(chi_acc)),
            "cap_active_mean":  float(np.nanmean(cap_mean_acc)),
            "cap_active_max":   float(np.nanmean(cap_max_acc)),
            "dead_relu_max":    float(np.nanmean(dead_acc)),
            "r_mean":      float(np.nanmean(r_mean_acc)),
            "r_p95":       float(np.nanmean(r_p95_acc)),
            "r_at_argmax": float(np.nanmean(r_argmax_acc)),
            "wall_s": wall, "any_nan": nan,
        })
        if nan:
            print(f"      ✗ NaN at epoch {epoch}; aborting.")
            break
    return rows, n_params


# ---------------------------------------------------------------------------
#  Sweep
# ---------------------------------------------------------------------------


def run_sweep(*, out_dir, floors, use_bn, gain_w, batch_size, n_epochs, seed,
              augment, data, device):
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    results = {}

    for i, floor in enumerate(floors, 1):
        cfg_id = f"floor{floor:g}__{'bn' if use_bn else 'nobn'}"
        print(f"\n  [{i}/{len(floors)}] {cfg_id}")
        rows, n_params = train_one(
            sigma_floor=floor, use_bn=use_bn, gain_w=gain_w,
            batch_size=batch_size, n_epochs=n_epochs, seed=seed,
            augment=augment, data=data, device=device,
        )
        results[cfg_id] = rows
        with (traces_dir / f"{cfg_id}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"      ({n_params:,} learnable scalars)")
        for r in rows:
            ta = (f"{r['test_acc']*100:5.2f}%"
                  if math.isfinite(r['test_acc']) else "  NaN ")
            ce = (f"{r['test_CE']:6.3f}"
                  if math.isfinite(r['test_CE']) else "  NaN ")
            z = (f"{r['zeta_dof']:5.2f}"
                 if math.isfinite(r['zeta_dof']) else "  -  ")
            k = (f"{r['kappa_p95_max']:6.2f}"
                 if math.isfinite(r['kappa_p95_max']) else "  -  ")
            cap = (f"{r['cap_active_mean']*100:5.2f}%"
                   if math.isfinite(r['cap_active_mean']) else "  -  ")
            rA = (f"{r['r_at_argmax']:.3f}"
                  if math.isfinite(r['r_at_argmax']) else "  -  ")
            dead = (f"{r['dead_relu_max']*100:5.1f}%"
                    if math.isfinite(r['dead_relu_max']) else "  -  ")
            print(f"      ep {r['epoch']:2d}  acc={ta}  CE={ce}  "
                  f"ζ_dof={z}  κ_p95={k}  cap_mean={cap}  "
                  f"r_argmax={rA}  dead={dead}  ({r['wall_s']:5.1f}s)")
        final = rows[-1]
        summary.append({
            "config_id": cfg_id, "sigma_floor": floor, "use_bn": use_bn,
            "final_test_acc": final["test_acc"],
            "final_test_CE": final["test_CE"],
            "final_zeta_dof": final["zeta_dof"],
            "final_kappa_p95_max": final["kappa_p95_max"],
            "final_cap_active_mean": final["cap_active_mean"],
            "final_dead_relu_max": final["dead_relu_max"],
            "final_r_at_argmax": final["r_at_argmax"],
            "ever_nan": any(rr["any_nan"] for rr in rows),
        })
    sp = out_dir / "summary.csv"
    with sp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\n  Summary: {sp}")
    return results, summary


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------


def plot_sweep(out_dir, results, floors, use_bn):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    metrics = [
        ("test_acc", "test acc (%)", 100.0, False),
        ("test_CE", "test CE", 1.0, False),
        ("zeta_dof", "ζ_dof  (1 = calibrated)", 1.0, True),
        ("kappa_p95_max", "κ_p95 worst layer", 1.0, True),
        ("cap_active_mean", "cap-active (mean, %)", 100.0, False),
        ("r_at_argmax", "r at argmax", 1.0, False),
        ("dead_relu_max", "dead-ReLU max (%)", 100.0, False),
    ]
    colors = {0.05: "tab:blue", 0.1: "tab:orange", 0.02: "tab:green", 0.0: "tab:red"}
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7, 2.4 * len(metrics)),
                             squeeze=False)
    for ri, (metric, ylabel, scale, log_y) in enumerate(metrics):
        ax = axes[ri, 0]
        for floor in floors:
            cfg_id = f"floor{floor:g}__{'bn' if use_bn else 'nobn'}"
            rows = results[cfg_id]
            xs = [r["epoch"] for r in rows]
            ys = [(r[metric] * scale) if math.isfinite(r[metric]) else math.nan
                  for r in rows]
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5,
                    color=colors.get(floor, "tab:gray"),
                    label=f"σ_floor={floor:g}")
        if log_y:
            ax.set_yscale("log")
        if metric == "zeta_dof":
            ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.7)
        if metric == "kappa_p95_max":
            ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.7)
        ax.set_xlabel("epoch" if ri == len(metrics) - 1 else "")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        if ri == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"CIFAR small CNN ({'with BN' if use_bn else 'no BN'}) — κ-anchored CPN + Model C")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"cifar_traces_{'bn' if use_bn else 'nobn'}.{ext}", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floors", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--use_bn", action="store_true",
                        help="Include BatchNorm2D in each conv block.")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gain_w", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_false", dest="augment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = "bn" if args.use_bn else "nobn"
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("runs") / f"pn_tagi_cifar_smallcnn_{tag}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"  CIFAR small CNN  —  recipe transfer test  ({'BN' if args.use_bn else 'no BN'})")
    print("=" * 64)
    print(f"  floors      : {args.floors}")
    print(f"  use_bn      : {args.use_bn}")
    print(f"  n_epochs    : {args.n_epochs}")
    print(f"  gain_w      : {args.gain_w}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  augment     : {args.augment}")
    print(f"  out_dir     : {out_dir}\n")

    device = torch.device(DEVICE)
    print(f"  Loading CIFAR-10 from '{args.data_dir}'...")
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels = load_cifar10(
        args.data_dir, device,
    )
    print(f"  Train: {x_train.shape[0]:,} | Test: {x_test.shape[0]:,}\n")
    data = dict(x_train=x_train, y_train_oh=y_train_oh, y_train_labels=y_train_labels,
                x_test=x_test, y_test_labels=y_test_labels)

    results, summary = run_sweep(
        out_dir=out_dir, floors=args.floors, use_bn=args.use_bn,
        gain_w=args.gain_w, batch_size=args.batch_size, n_epochs=args.n_epochs,
        seed=args.seed, augment=args.augment, data=data, device=device,
    )
    plot_sweep(out_dir, results, args.floors, args.use_bn)

    print("\n  Final summary:")
    for s in summary:
        ta = (f"{s['final_test_acc']*100:5.2f}%"
              if math.isfinite(s['final_test_acc']) else "  NaN ")
        print(f"    {s['config_id']:<24}  acc={ta}  "
              f"CE={s['final_test_CE']:6.3f}  "
              f"ζ_dof={s['final_zeta_dof']:5.2f}  "
              f"κ_p95={s['final_kappa_p95_max']:5.2f}  "
              f"dead={s['final_dead_relu_max']*100:5.1f}%")
    print(f"\n  Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
