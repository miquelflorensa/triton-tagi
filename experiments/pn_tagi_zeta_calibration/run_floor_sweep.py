"""σ_floor sweep — find the smallest residual-mismatch floor that keeps
Model C stable, and check whether the floor activates "as intended".

Background. Experiment 2 showed that
    σ²_Y_i  =  σ²_A_i  +  μ_A_i(1 − μ_A_i)  +  σ²_floor
is the cleanest Remax observation model: epistemic + intrinsic
categorical + residual mismatch floor. Without the floor (σ_floor = 0)
the categorical term can collapse to 0 when predictions become peaked,
which lets confidently-wrong residuals blow up the innovation. The
question is: how small can σ_floor be before that collapse re-appears?

This sweep varies σ_floor ∈ {0, 0.01, 0.02, 0.05, 0.1} across depths
{3, 5, 7} under the κ-anchored CPN update. Each (σ_floor, depth)
config trains 5 epochs. Per epoch we record the floor-activation ratio

    r_i  =  σ²_floor  /  (μ_A_i(1 − μ_A_i)  +  σ²_floor)

aggregated as mean / p95 / "at-argmax". At init r should be ≈ 0
(categorical term dominates). Once predictions peak, r rises toward 1
— and we want to see exactly when and by how much.

Run::

    python experiments/pn_tagi_zeta_calibration/run_floor_sweep.py
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
#  Model C innovation + diagnostics
# ---------------------------------------------------------------------------


def innovation_C(y_oh, mu_Y, S_A, sigma_floor):
    """δμ, δS for Model C: σ²_Y = σ²_A + μ(1 − μ) + σ²_floor.

    Returns the deltas plus the ζ_dof scalar and the per-element floor
    activation ratio r_i for diagnostics.
    """
    cat = mu_Y * (1.0 - mu_Y)
    floor_sq = sigma_floor ** 2
    sigma_Y_sq = (S_A + cat + floor_sq).clamp_min(1e-8)
    residual = y_oh - mu_Y
    delta_mu_y = residual / sigma_Y_sq
    delta_S_y = -1.0 / sigma_Y_sq
    zeta_raw = ((residual ** 2) / sigma_Y_sq).sum(dim=-1).mean().item()
    zeta_dof = zeta_raw / (N_CLASSES - 1)
    # Floor activation ratio  r_i = σ²_floor / (μ(1-μ) + σ²_floor)
    # (excludes S_A so the metric is "how much of the categorical+floor
    # denominator does the floor account for"). Per-element.
    r = floor_sq / (cat + floor_sq).clamp_min(1e-12)
    return delta_mu_y, delta_S_y, sigma_Y_sq, zeta_dof, r


def per_batch_diagnostics(net):
    kappa_layers = []
    cap_active_layers = []
    for layer in net.layers:
        if not isinstance(layer, Linear):
            continue
        Sw = layer.Sw
        dmw = layer.delta_mw
        sigma_w = Sw.sqrt().clamp_min(1e-8)
        kappa = dmw.abs() / sigma_w
        cap_active = (dmw.abs() > sigma_w).float().mean().item()
        kappa_layers.append(float(kappa.quantile(0.95).item()))
        cap_active_layers.append(cap_active)
    return {
        "kappa_p95_max": max(kappa_layers),
        "cap_active_mean": float(np.mean(cap_active_layers)),
    }


def dead_relu_max(net, x_batch):
    ma = x_batch
    Sa = torch.zeros_like(ma)
    dead = []
    for layer in net.layers:
        ma, Sa = layer.forward(ma, Sa)
        if isinstance(layer, ReLU):
            dead.append(float((ma.abs() < 1e-6).float().mean().item()))
    return max(dead) if dead else 0.0


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
#  Train one (σ_floor, depth) config
# ---------------------------------------------------------------------------


def train_one(*, sigma_floor, depth, hidden, gain_w, batch_size, n_epochs, seed,
              x_train, y_train_oh, y_train_labels, x_test, y_test_labels, device):
    torch.manual_seed(seed)
    net = build_mlp(depth=depth, hidden=hidden, gain_w=gain_w, device=device)

    n_train = x_train.size(0)
    rows = []
    test_acc, test_ce = evaluate(net, x_test, y_test_labels)
    rows.append({
        "epoch": 0,
        "train_CE": math.nan, "test_acc": test_acc, "test_CE": test_ce,
        "zeta_dof": math.nan,
        "kappa_p95_max": math.nan, "cap_active_mean": math.nan,
        "dead_relu_max": math.nan,
        "r_mean": math.nan, "r_p95": math.nan, "r_at_argmax": math.nan,
        "wall_s": 0.0, "any_nan": False,
    })

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, device=device)
        xs = x_train[perm]
        ys = y_train_oh[perm]

        ce_sum = 0.0
        n_seen = 0
        zeta_acc, kappa_acc, cap_acc, dead_acc = [], [], [], []
        r_mean_acc, r_p95_acc, r_argmax_acc = [], [], []

        for i in range(0, n_train, batch_size):
            xb = xs[i : i + batch_size]
            yb = ys[i : i + batch_size]
            if xb.size(0) == 0:
                continue

            mu_Y, S_A = net.forward(xb)
            dmu_y, dS_y, _, zeta_dof, r = innovation_C(yb, mu_Y, S_A, sigma_floor)

            # Manual backward.
            cur_dmu, cur_dvar = dmu_y, dS_y
            for layer in reversed(net.layers):
                cur_dmu, cur_dvar = layer.backward(cur_dmu, cur_dvar)

            # Diagnostics (pre-update).
            diag = per_batch_diagnostics(net)
            zeta_acc.append(zeta_dof)
            kappa_acc.append(diag["kappa_p95_max"])
            cap_acc.append(diag["cap_active_mean"])
            r_mean_acc.append(float(r.mean().item()))
            r_p95_acc.append(float(r.quantile(0.95).item()))
            # r at argmax (the predicted class): how much of denom is floor at the peak.
            am = mu_Y.argmax(dim=1, keepdim=True)
            r_argmax = float(r.gather(1, am).mean().item())
            r_argmax_acc.append(r_argmax)

            # CE on this batch.
            lab = y_train_labels[perm][i : i + batch_size]
            p_true = mu_Y.clamp_min(1e-8).gather(1, lab.unsqueeze(1)).squeeze(1)
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
            dead_acc.append(dead_relu_max(net, xb))

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        nan = any_nan_or_inf(net)
        if nan:
            test_acc, test_ce = math.nan, math.nan
        else:
            test_acc, test_ce = evaluate(net, x_test, y_test_labels)
        rows.append({
            "epoch": epoch,
            "train_CE": ce_sum / max(n_seen, 1),
            "test_acc": test_acc, "test_CE": test_ce,
            "zeta_dof":        float(np.mean(zeta_acc)),
            "kappa_p95_max":   float(np.mean(kappa_acc)),
            "cap_active_mean": float(np.mean(cap_acc)),
            "dead_relu_max":   float(np.mean(dead_acc)),
            "r_mean":      float(np.mean(r_mean_acc)),
            "r_p95":       float(np.mean(r_p95_acc)),
            "r_at_argmax": float(np.mean(r_argmax_acc)),
            "wall_s": wall, "any_nan": nan,
        })
        if nan:
            print(f"      ✗ NaN at epoch {epoch}; aborting.")
            break
    return rows


# ---------------------------------------------------------------------------
#  Sweep + plot
# ---------------------------------------------------------------------------


def run_sweep(*, out_dir, floors, depths, hidden, gain_w, batch_size,
              n_epochs, seed, data, device):
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    results = {}  # cfg_id → rows

    n = len(floors) * len(depths)
    i = 0
    for floor in floors:
        for d in depths:
            i += 1
            cfg_id = f"floor{floor:g}__d{d}"
            print(f"\n  [{i}/{n}] {cfg_id}")
            rows = train_one(
                sigma_floor=floor, depth=d, hidden=hidden, gain_w=gain_w,
                batch_size=batch_size, n_epochs=n_epochs, seed=seed,
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
                ta = (f"{r['test_acc']*100:5.2f}%" if math.isfinite(r['test_acc']) else "  NaN ")
                ce = (f"{r['test_CE']:6.3f}" if math.isfinite(r['test_CE']) else "  NaN ")
                z = (f"{r['zeta_dof']:5.2f}" if math.isfinite(r['zeta_dof']) else "  -  ")
                rm = (f"{r['r_mean']:6.3f}" if math.isfinite(r['r_mean']) else "  -  ")
                rA = (f"{r['r_at_argmax']:6.3f}" if math.isfinite(r['r_at_argmax']) else "  -  ")
                d_max = (f"{r['dead_relu_max']*100:5.1f}%"
                         if math.isfinite(r['dead_relu_max']) else "  -  ")
                print(f"      ep {r['epoch']:2d}  acc={ta}  CE={ce}  "
                      f"ζ_dof={z}  r_mean={rm}  r_argmax={rA}  dead={d_max}")
            final = rows[-1]
            summary.append({
                "config_id": cfg_id, "sigma_floor": floor, "depth": d,
                "final_test_acc": final["test_acc"],
                "final_test_CE": final["test_CE"],
                "final_zeta_dof": final["zeta_dof"],
                "final_r_mean": final["r_mean"],
                "final_r_at_argmax": final["r_at_argmax"],
                "final_cap_active_mean": final["cap_active_mean"],
                "final_dead_relu_max": final["dead_relu_max"],
                "ever_nan": any(rr["any_nan"] for rr in rows),
            })

    sp = out_dir / "summary.csv"
    with sp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\n  Summary: {sp}")
    return results, summary


def plot_sweep(out_dir, results, summary, floors, depths):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed")
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # 1. Heatmap: final acc vs (σ_floor × depth).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    mat = np.full((len(floors), len(depths)), np.nan)
    for r in summary:
        fi = floors.index(r["sigma_floor"])
        di = depths.index(r["depth"])
        mat[fi, di] = r["final_test_acc"] * 100 if math.isfinite(r["final_test_acc"]) else math.nan
    im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="upper")
    ax.set_xticks(range(len(depths)), [str(d) for d in depths])
    ax.set_yticks(range(len(floors)), [f"{f:g}" for f in floors])
    ax.set_xlabel("depth")
    ax.set_ylabel("σ_floor")
    ax.set_title("Final test accuracy (%)")
    for fi in range(len(floors)):
        for di in range(len(depths)):
            v = mat[fi, di]
            if math.isfinite(v):
                ax.text(di, fi, f"{v:.1f}", ha="center", va="center",
                        color="white" if v < 60 else "black", fontsize=8)
            else:
                ax.text(di, fi, "NaN", ha="center", va="center", color="red", fontsize=8)
    plt.colorbar(im, ax=ax)

    # 2. r_at_argmax heatmap.
    ax = axes[1]
    mat_r = np.full((len(floors), len(depths)), np.nan)
    for r in summary:
        fi = floors.index(r["sigma_floor"])
        di = depths.index(r["depth"])
        mat_r[fi, di] = r["final_r_at_argmax"] if math.isfinite(r["final_r_at_argmax"]) else math.nan
    im = ax.imshow(mat_r, aspect="auto", cmap="magma", origin="upper", vmin=0, vmax=1)
    ax.set_xticks(range(len(depths)), [str(d) for d in depths])
    ax.set_yticks(range(len(floors)), [f"{f:g}" for f in floors])
    ax.set_xlabel("depth")
    ax.set_ylabel("σ_floor")
    ax.set_title("r at argmax  (= σ²_floor / (μ(1-μ) + σ²_floor) at predicted class)")
    for fi in range(len(floors)):
        for di in range(len(depths)):
            v = mat_r[fi, di]
            if math.isfinite(v):
                ax.text(di, fi, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax)

    fig.suptitle(f"σ_floor sweep — final values after {summary[0].get('config_id', '?')[-3:] or '?'} epochs")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"floor_sweep_heatmap.{ext}", dpi=140)
    plt.close(fig)

    # 3. Per-epoch traces: test acc + r_at_argmax + dead-ReLU, panels per depth, lines per floor.
    metrics = [
        ("test_acc", "test accuracy (%)", 100.0),
        ("zeta_dof", "ζ_dof", 1.0),
        ("r_at_argmax", "r at argmax", 1.0),
        ("dead_relu_max", "dead-ReLU max (%)", 100.0),
    ]
    fig, axes = plt.subplots(len(metrics), len(depths),
                             figsize=(4.0 * len(depths), 2.6 * len(metrics)),
                             squeeze=False)
    cmap = plt.cm.viridis
    colors = [cmap(0.15 + 0.7 * i / max(len(floors) - 1, 1)) for i in range(len(floors))]
    for ri, (metric, ylabel, scale) in enumerate(metrics):
        for ci, d in enumerate(depths):
            ax = axes[ri, ci]
            for fi, floor in enumerate(floors):
                rows = results[f"floor{floor:g}__d{d}"]
                xs = [r["epoch"] for r in rows]
                ys = [(r[metric] * scale) if math.isfinite(r[metric]) else math.nan
                      for r in rows]
                ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.4,
                        color=colors[fi], label=f"σ_floor={floor:g}")
            if metric == "zeta_dof":
                ax.set_yscale("log")
                ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.7)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=6)
            if ri == 0:
                ax.set_title(f"depth = {d}")
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlabel("epoch" if ri == len(metrics) - 1 else "")
            ax.grid(True, alpha=0.3)
    fig.suptitle("σ_floor sweep — per-epoch traces")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"floor_sweep_traces.{ext}", dpi=140)
    plt.close(fig)

    print(f"  Figures: {fig_dir}/")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floors", type=float, nargs="+",
                        default=[0.0, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gain_w", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("runs") / f"pn_tagi_floor_sweep_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  σ_floor sweep — Model C with κ-anchored CPN")
    print("=" * 64)
    print(f"  floors    : {args.floors}")
    print(f"  depths    : {args.depths}")
    print(f"  n_epochs  : {args.n_epochs}")
    print(f"  hidden    : {args.hidden}")
    print(f"  gain_w    : {args.gain_w}")
    print(f"  batch     : {args.batch_size}\n")

    device = torch.device(DEVICE)
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels = load_mnist(
        args.data_dir, device,
    )
    data = dict(x_train=x_train, y_train_oh=y_train_oh, y_train_labels=y_train_labels,
                x_test=x_test, y_test_labels=y_test_labels)

    results, summary = run_sweep(
        out_dir=out_dir, floors=args.floors, depths=args.depths,
        hidden=args.hidden, gain_w=args.gain_w, batch_size=args.batch_size,
        n_epochs=args.n_epochs, seed=args.seed, data=data, device=device,
    )

    plot_sweep(out_dir, results, summary, args.floors, args.depths)

    print("\n  Final summary:")
    for s in summary:
        ta = (f"{s['final_test_acc']*100:5.2f}%"
              if math.isfinite(s['final_test_acc']) else "  NaN ")
        z = (f"{s['final_zeta_dof']:5.2f}"
             if math.isfinite(s['final_zeta_dof']) else "  -  ")
        rA = (f"{s['final_r_at_argmax']:.3f}"
              if math.isfinite(s['final_r_at_argmax']) else "  -  ")
        print(f"    σ_floor={s['sigma_floor']:<6g}  d={s['depth']}  "
              f"acc={ta}  CE={s['final_test_CE']:6.3f}  ζ_dof={z}  r_argmax={rA}")
    print(f"\n  Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
