"""Experiment 2 — training comparison of observation-variance models
under the κ-anchored CPN update.

Fixed:
    update_rule  = capped_precision_normalized
    cap_factor   = 1.0     (κ ≤ 1)
    rho          = 1.0
    gain_w       = 1.0
    batch_size   = 512
    sigma_v      = 0.05    (kept for model A and C; ignored by model B)
    depths       = {3, 5, 7}
    epochs       = 3
    hidden       = 256

Observation models compared (only these three — label smoothing was
ruled out at the init audit):

    A. current             σ²_Y_i = σ²_A_i + σ_v²
    B. categorical         σ²_Y_i = σ²_A_i + μ_A_i (1 − μ_A_i)
    C. cat + σ_v           σ²_Y_i = σ²_A_i + μ_A_i (1 − μ_A_i) + σ_v²

Per-batch metrics logged (averaged into per-epoch CSV):
    test_acc, train_CE, test_CE
    ζ_dof  =  Σ_i (y_i − μ_Y_i)² / Σ_Y_i  /  (K − 1)
    κ_p95, χ_p95 (worst layer across the network)
    cap-activation fraction (network mean across Linear layers)
    dead-ReLU fraction (worst layer)
    mean σ²_A and mean μ_A(1 − μ_A) at the output

Headline question: does the calibrated ζ at init *stay* calibrated
during training, and does it reduce dependence on the κ cap?
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
from torchvision import datasets

from triton_tagi import Linear, ReLU, Remax, Sequential
from triton_tagi.base import LearnableLayer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 10


# ---------------------------------------------------------------------------
#  Data + net
# ---------------------------------------------------------------------------


def load_mnist(data_dir: str, device: torch.device):
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


def build_mlp(*, depth, hidden, gain_w, device):
    """κ-anchored CPN MLP with the new κ ≤ 1 cap."""
    layers = [Linear(784, hidden, device=device, gain_w=gain_w, gain_b=gain_w), ReLU()]
    for _ in range(depth - 1):
        layers.append(Linear(hidden, hidden, device=device, gain_w=gain_w, gain_b=gain_w))
        layers.append(ReLU())
    layers.append(Linear(hidden, 10, device=device, gain_w=gain_w, gain_b=gain_w))
    layers.append(Remax())
    return Sequential(
        layers, device=device,
        update_rule="capped_precision_normalized", rho=1.0, record_chi=False,
        cap_factor=1.0,
    )


# ---------------------------------------------------------------------------
#  Custom innovation per observation model
# ---------------------------------------------------------------------------


def innovation(model: str, y_oh, mu_Y, S_A, sigma_v):
    """Return (δμ_y, δS_y, σ²_Y, ζ_dof, mean_S_A, mean_categorical).

    ζ_dof = ζ_raw / (K − 1).
    """
    if model == "A":
        sigma_Y_sq = S_A + sigma_v ** 2
    elif model == "B":
        sigma_Y_sq = S_A + mu_Y * (1.0 - mu_Y)
    elif model == "C":
        sigma_Y_sq = S_A + mu_Y * (1.0 - mu_Y) + sigma_v ** 2
    else:
        raise ValueError(model)
    sigma_Y_sq = sigma_Y_sq.clamp_min(1e-8)

    residual = y_oh - mu_Y
    delta_mu_y = residual / sigma_Y_sq
    delta_S_y = -1.0 / sigma_Y_sq

    zeta_raw = ((residual ** 2) / sigma_Y_sq).sum(dim=-1).mean().item()
    zeta_dof = zeta_raw / (N_CLASSES - 1)
    mean_S_A = float(S_A.mean().item())
    mean_cat = float((mu_Y * (1.0 - mu_Y)).mean().item())
    return delta_mu_y, delta_S_y, sigma_Y_sq, zeta_dof, mean_S_A, mean_cat


# ---------------------------------------------------------------------------
#  Per-batch diagnostics from layer state
# ---------------------------------------------------------------------------


def per_batch_diagnostics(net):
    """Compute κ_p95, χ_p95, cap-activation per Linear; aggregate over net."""
    kappa_p95_layers = []
    chi_p95_layers = []
    cap_active_layers = []
    for layer in net.layers:
        if not isinstance(layer, Linear):
            continue
        Sw = layer.Sw
        dmw = layer.delta_mw
        dSw = layer.delta_Sw
        sigma_w = Sw.sqrt().clamp_min(1e-8)
        kappa = dmw.abs() / sigma_w
        chi = -dSw / Sw.clamp_min(1e-12)
        cap_active = (dmw.abs() > sigma_w).float().mean().item()
        kappa_p95_layers.append(float(kappa.quantile(0.95).item()))
        chi_p95_layers.append(float(chi.quantile(0.95).item()))
        cap_active_layers.append(cap_active)
    return {
        "kappa_p95_max": max(kappa_p95_layers),
        "chi_p95_max":   max(chi_p95_layers),
        "cap_active_mean": float(np.mean(cap_active_layers)),
        "cap_active_max":  max(cap_active_layers),
    }


def dead_relu_fraction(net, x_batch):
    """Forward pass; record fraction of |a| < 1e-6 at each ReLU output."""
    ma = x_batch
    Sa = torch.zeros_like(ma)
    dead = []
    for layer in net.layers:
        ma, Sa = layer.forward(ma, Sa)
        if isinstance(layer, ReLU):
            dead.append(float((ma.abs() < 1e-6).float().mean().item()))
    return max(dead) if dead else 0.0


# ---------------------------------------------------------------------------
#  Training loop with custom innovation
# ---------------------------------------------------------------------------


def train_one_config(
    *, model, depth, hidden, gain_w, sigma_v, batch_size, n_epochs, seed,
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels, device,
):
    torch.manual_seed(seed)
    net = build_mlp(depth=depth, hidden=hidden, gain_w=gain_w, device=device)

    n_train = x_train.size(0)
    epoch_rows = []

    # Epoch 0: pre-training eval.
    test_acc, test_ce = evaluate(net, x_test, y_test_labels)
    epoch_rows.append({
        "epoch": 0, "train_CE": math.nan, "test_acc": test_acc, "test_CE": test_ce,
        "zeta_dof": math.nan,
        "kappa_p95_max": math.nan, "chi_p95_max": math.nan,
        "cap_active_mean": math.nan, "cap_active_max": math.nan,
        "dead_relu_max": math.nan, "mean_S_A": math.nan, "mean_cat": math.nan,
        "wall_s": 0.0, "any_nan": False,
    })

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, device=device)
        xs = x_train[perm]
        ys = y_train_oh[perm]

        # Batch accumulators for the epoch.
        ce_sum = 0.0
        n_seen = 0
        zeta_acc = []
        kappa_p95_acc = []
        chi_p95_acc = []
        cap_active_mean_acc = []
        cap_active_max_acc = []
        dead_acc = []
        S_A_acc = []
        cat_acc = []

        for i in range(0, n_train, batch_size):
            xb = xs[i : i + batch_size]
            yb = ys[i : i + batch_size]
            if xb.size(0) == 0:
                continue

            # 1. Forward (caches activations on each layer).
            mu_Y, S_A = net.forward(xb)

            # 2. Custom innovation per observation model.
            dmu_y, dS_y, _, zeta_dof, mean_S_A, mean_cat = innovation(
                model, yb, mu_Y, S_A, sigma_v,
            )

            # 3. Manual backward through layers in reverse order.
            cur_dmu = dmu_y
            cur_dvar = dS_y
            for layer in reversed(net.layers):
                cur_dmu, cur_dvar = layer.backward(cur_dmu, cur_dvar)

            # 4. Per-batch diagnostics (post-backward, pre-update).
            diag = per_batch_diagnostics(net)
            zeta_acc.append(zeta_dof)
            kappa_p95_acc.append(diag["kappa_p95_max"])
            chi_p95_acc.append(diag["chi_p95_max"])
            cap_active_mean_acc.append(diag["cap_active_mean"])
            cap_active_max_acc.append(diag["cap_active_max"])
            S_A_acc.append(mean_S_A)
            cat_acc.append(mean_cat)

            # CE on this batch (train CE).
            p_true = mu_Y.clamp_min(1e-8).gather(
                1, y_train_labels[perm][i : i + batch_size].unsqueeze(1)
            ).squeeze(1)
            ce_sum += float(-p_true.log().sum().item())
            n_seen += xb.size(0)

            # 5. Apply update (κ-anchored CPN with cap_factor = 1.0).
            cap_factor = net.cap_factor_override  # 1.0 by build_mlp
            for layer in net.layers:
                if isinstance(layer, LearnableLayer):
                    layer.update(
                        cap_factor,
                        update_rule=net.update_rule,
                        rho=net.rho, record_chi=False,
                    )

            # 6. Dead-ReLU after update (cheap; one extra forward).
            dead_acc.append(dead_relu_fraction(net, xb))

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        nan = any_nan_or_inf(net)
        train_ce = ce_sum / max(n_seen, 1)
        if nan:
            test_acc, test_ce = math.nan, math.nan
        else:
            test_acc, test_ce = evaluate(net, x_test, y_test_labels)

        epoch_rows.append({
            "epoch": epoch,
            "train_CE": train_ce,
            "test_acc": test_acc,
            "test_CE": test_ce,
            "zeta_dof":         float(np.mean(zeta_acc)),
            "kappa_p95_max":    float(np.mean(kappa_p95_acc)),
            "chi_p95_max":      float(np.mean(chi_p95_acc)),
            "cap_active_mean":  float(np.mean(cap_active_mean_acc)),
            "cap_active_max":   float(np.mean(cap_active_max_acc)),
            "dead_relu_max":    float(np.mean(dead_acc)),
            "mean_S_A":         float(np.mean(S_A_acc)),
            "mean_cat":         float(np.mean(cat_acc)),
            "wall_s": wall, "any_nan": nan,
        })

        if nan:
            print(f"      ✗ NaN at epoch {epoch}; aborting this config.")
            break

    return epoch_rows


def evaluate(net, x, y_labels, batch_size=1024):
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
        if isinstance(layer, Linear):
            for t in (layer.mw, layer.Sw, layer.mb, layer.Sb):
                if t is None:
                    continue
                if not torch.isfinite(t).all():
                    return True
    return False


# ---------------------------------------------------------------------------
#  Sweep + plot
# ---------------------------------------------------------------------------


def run_sweep(*, out_dir, depths, models, hidden, gain_w, sigma_v, batch_size,
              n_epochs, seed, data, device):
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    results = {}  # cfg_id → list of epoch rows
    n_configs = len(depths) * len(models)
    i = 0

    for model in models:
        for d in depths:
            i += 1
            cfg_id = f"{model}__d{d}"
            print(f"\n  [{i}/{n_configs}] {cfg_id}")
            rows = train_one_config(
                model=model, depth=d, hidden=hidden, gain_w=gain_w,
                sigma_v=sigma_v, batch_size=batch_size, n_epochs=n_epochs,
                seed=seed,
                x_train=data["x_train"], y_train_oh=data["y_train_oh"],
                y_train_labels=data["y_train_labels"],
                x_test=data["x_test"], y_test_labels=data["y_test_labels"],
                device=device,
            )
            results[cfg_id] = rows

            with (traces_dir / f"{cfg_id}.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

            for r in rows:
                acc_s = (f"{r['test_acc']*100:5.2f}%" if math.isfinite(r['test_acc'])
                         else "  NaN ")
                zeta_s = (f"{r['zeta_dof']:5.2f}" if math.isfinite(r['zeta_dof'])
                          else "  -  ")
                cap_s = (f"{r['cap_active_mean']*100:5.2f}%"
                         if math.isfinite(r['cap_active_mean']) else "  -  ")
                dead_s = (f"{r['dead_relu_max']*100:5.1f}%"
                          if math.isfinite(r['dead_relu_max']) else "  -  ")
                print(f"      epoch {r['epoch']:2d}  test_acc={acc_s}  "
                      f"test_CE={r['test_CE']:6.3f}  ζ_dof={zeta_s}  "
                      f"κ_p95={r['kappa_p95_max']:6.2f}  cap_mean={cap_s}  "
                      f"dead_max={dead_s}")

            final = rows[-1]
            summary_rows.append({
                "config_id": cfg_id, "model": model, "depth": d,
                "final_test_acc": final["test_acc"],
                "final_test_CE": final["test_CE"],
                "final_zeta_dof": final["zeta_dof"],
                "final_kappa_p95_max": final["kappa_p95_max"],
                "final_cap_active_mean": final["cap_active_mean"],
                "final_dead_relu_max": final["dead_relu_max"],
                "ever_nan": any(r["any_nan"] for r in rows),
            })

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n  Summary written to {summary_path}")
    return results, summary_rows


def make_plots(out_dir, results, depths, models):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figures")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    metric_panels = [
        ("test_acc", "test accuracy (%)", 100.0, False, False),
        ("test_CE", "test CE (nats)", 1.0, False, False),
        ("zeta_dof", "ζ_dof  (1 = calibrated)", 1.0, True, False),
        ("kappa_p95_max", "κ_p95 worst layer", 1.0, True, False),
        ("cap_active_mean", "cap-activation (mean across Linears, %)", 100.0, False, False),
        ("dead_relu_max", "dead-ReLU worst layer (%)", 100.0, False, False),
    ]

    colors = {"A": "tab:red", "B": "tab:green", "C": "tab:blue"}
    labels = {"A": "A: σ²_A + σ_v²", "B": "B: + μ(1-μ)", "C": "C: + μ(1-μ) + σ_v²"}

    fig, axes = plt.subplots(
        len(metric_panels), len(depths),
        figsize=(4.0 * len(depths), 2.6 * len(metric_panels)), squeeze=False,
    )
    for r_idx, (metric, ylabel, scale, log_y, _) in enumerate(metric_panels):
        for c_idx, d in enumerate(depths):
            ax = axes[r_idx, c_idx]
            for m in models:
                rows = results[f"{m}__d{d}"]
                xs = [r["epoch"] for r in rows]
                ys = [(r[metric] * scale) if math.isfinite(r[metric]) else math.nan
                      for r in rows]
                ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5,
                        color=colors[m], label=labels[m])
            if log_y:
                ax.set_yscale("log")
            if metric == "zeta_dof":
                ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
            if metric == "kappa_p95_max":
                ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)
            ax.set_xlabel("epoch" if r_idx == len(metric_panels) - 1 else "")
            if c_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if r_idx == 0:
                ax.set_title(f"depth = {d}")
            ax.grid(True, alpha=0.3)
            if r_idx == 0 and c_idx == 0:
                ax.legend(fontsize=7)

    fig.suptitle("Experiment 2 — κ-anchored CPN × observation model × depth")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"training_traces.{ext}", dpi=140)
    plt.close(fig)
    print(f"  Figure: {fig_dir}/training_traces.png")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--models", type=str, nargs="+", default=["A", "B", "C"])
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gain_w", type=float, default=1.0)
    parser.add_argument("--sigma_v", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("runs") / f"pn_tagi_zeta_training_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Experiment 2 — categorical covariance training comparison")
    print("=" * 64)
    print(f"  update_rule  : capped_precision_normalized")
    print(f"  cap_factor   : 1.0  (κ ≤ 1)")
    print(f"  rho          : 1.0")
    print(f"  gain_w       : {args.gain_w}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  sigma_v      : {args.sigma_v}  (used by A, ignored by B, present in C)")
    print(f"  depths       : {args.depths}")
    print(f"  models       : {args.models}")
    print(f"  n_epochs     : {args.n_epochs}\n")

    device = torch.device(DEVICE)
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels = load_mnist(
        args.data_dir, device,
    )
    data = dict(x_train=x_train, y_train_oh=y_train_oh, y_train_labels=y_train_labels,
                x_test=x_test, y_test_labels=y_test_labels)

    results, summary = run_sweep(
        out_dir=out_dir, depths=args.depths, models=args.models,
        hidden=args.hidden, gain_w=args.gain_w, sigma_v=args.sigma_v,
        batch_size=args.batch_size, n_epochs=args.n_epochs,
        seed=args.seed, data=data, device=device,
    )

    make_plots(out_dir, results, args.depths, args.models)

    print("\n  Final summary:")
    for s in summary:
        ta = (f"{s['final_test_acc']*100:5.2f}%"
              if math.isfinite(s['final_test_acc']) else "  NaN ")
        z = (f"{s['final_zeta_dof']:5.2f}"
             if math.isfinite(s['final_zeta_dof']) else "  -  ")
        k = (f"{s['final_kappa_p95_max']:6.2f}"
             if math.isfinite(s['final_kappa_p95_max']) else "  -  ")
        cap = (f"{s['final_cap_active_mean']*100:5.2f}%"
               if math.isfinite(s['final_cap_active_mean']) else "  -  ")
        print(f"    {s['config_id']:<12}  acc={ta}  CE={s['final_test_CE']:6.3f}  "
              f"ζ_dof={z}  κ_p95={k}  cap_mean={cap}")
    print(f"\n  Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
