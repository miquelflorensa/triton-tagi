"""Stage R follow-up — does small-prior init rescue plain PN-TAGI at depth?

The Stage R depth sweep showed plain ``precision_normalized`` cliff-fails
at depth ≥ 3 even with TAGI-V (V² learned, no σ_v hyperparameter). The
Stage 3 ``gain_w`` axis sweep (MNIST, depth=3) had separately shown that
``gain_w = 0.25`` rescues PN-TAGI at depth=3. This script tests whether
that rescue extends through every depth — i.e. whether the "depth
problem" is fundamentally a prior-variance / initialization problem
that disappears with a small enough initial S_w.

Sweep: ``gain_w ∈ {0.05, 0.1, 0.25, 0.5, 1.0}`` × ``depth ∈ {1, 3, 5, 7}``
× ``rule ∈ {capped_additive, precision_normalized}``. TAGI-V regression
(1-D heteros), 30 epochs each, hidden=50, batch=64. Plot a heatmap of
final test RMSE per rule on (gain × depth) axes.

If plain PN-TAGI fully recovers across all depths at some gain value,
that's the smallest-prior-fix-the-init story. If it still fails at deep
configs at every gain, then the failure is genuinely beyond init scale
and a step-bound mechanism is necessary.

Usage::

    python experiments/pn_tagi_stageR/run_regression_gain_depth.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
from pathlib import Path

import numpy as np
import torch

# Reuse the harness from the sister script via a relative import path. We
# keep this script self-contained to avoid an experimental-package
# __init__.py — just copy the small helpers needed.
from triton_tagi import EvenSoftplus, Linear, ReLU, Sequential


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_data(n_train=800, n_test=500, seed=0):
    rng = np.random.default_rng(seed)
    x_tr = rng.uniform(-4.0, 4.0, n_train).astype(np.float32)
    x_te = np.linspace(-4.0, 4.0, n_test, dtype=np.float32)

    def _sample(x):
        noise_std = 0.05 + 0.3 * np.abs(x)
        return np.sin(x) + rng.normal(0.0, noise_std).astype(np.float32)

    return (x_tr.reshape(-1, 1), _sample(x_tr).reshape(-1, 1),
            x_te.reshape(-1, 1), _sample(x_te).reshape(-1, 1))


def normalise(x_tr, y_tr, x_te, y_te):
    mu_x, sd_x = x_tr.mean(), x_tr.std() + 1e-8
    mu_y, sd_y = y_tr.mean(), y_tr.std() + 1e-8
    return ((x_tr - mu_x) / sd_x, (y_tr - mu_y) / sd_y,
            (x_te - mu_x) / sd_x, (y_te - mu_y) / sd_y)


def build_mlp(*, depth, hidden, gain_w, device, update_rule):
    layers = [Linear(1, hidden, device=device, gain_w=gain_w, gain_b=gain_w), ReLU()]
    for _ in range(depth - 1):
        layers.append(Linear(hidden, hidden, device=device, gain_w=gain_w, gain_b=gain_w))
        layers.append(ReLU())
    layers.append(Linear(hidden, 2, device=device, gain_w=gain_w, gain_b=gain_w))
    layers.append(EvenSoftplus(half_width=1))
    return Sequential(layers, device=device, update_rule=update_rule, rho=1.0, record_chi=False)


def evaluate(net, x, y, batch_size=1024):
    net.eval()
    sq_err = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i+batch_size]
            yb = y[i:i+batch_size]
            mu, _ = net.forward(xb)
            sq_err += float(((mu[:, 0:1] - yb) ** 2).sum().item())
            n += xb.size(0)
    net.train()
    return math.sqrt(sq_err / max(n, 1))


def any_nan(net):
    for layer in net.layers:
        if isinstance(layer, Linear):
            for t in (layer.mw, layer.Sw, layer.mb, layer.Sb):
                if t is None:
                    continue
                if not torch.isfinite(t).all():
                    return True
    return False


def train_one(*, depth, rule, gain_w, hidden, batch_size, n_epochs, seed, data, device):
    torch.manual_seed(seed)
    net = build_mlp(depth=depth, hidden=hidden, gain_w=gain_w,
                    device=device, update_rule=rule)
    x_train, y_train = data["x_train"], data["y_train"]
    n_train = x_train.size(0)
    for _ in range(n_epochs):
        perm = torch.randperm(n_train, device=device)
        xs = x_train[perm]
        ys = y_train[perm]
        for i in range(0, n_train, batch_size):
            net.step(xs[i:i+batch_size], ys[i:i+batch_size], sigma_v=1.0)
        if any_nan(net):
            return math.nan, True
    return evaluate(net, data["x_test"], data["y_test"]), any_nan(net)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gains", type=float, nargs="+",
                        default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 3, 5, 7])
    parser.add_argument("--rules", type=str, nargs="+",
                        default=["capped_additive", "precision_normalized"])
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("runs") / f"pn_tagi_stageR_gain_depth_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)
    x_tr_raw, y_tr_raw, x_te_raw, y_te_raw = generate_data()
    x_tr, y_tr, x_te, y_te = normalise(x_tr_raw, y_tr_raw, x_te_raw, y_te_raw)
    data = dict(
        x_train=torch.from_numpy(x_tr).to(device),
        y_train=torch.from_numpy(y_tr).to(device),
        x_test=torch.from_numpy(x_te).to(device),
        y_test=torch.from_numpy(y_te).to(device),
    )

    print(f"  gains={args.gains}  depths={args.depths}  rules={args.rules}")
    print(f"  n_epochs={args.n_epochs}  hidden={args.hidden}  batch={args.batch_size}\n")

    rows = []
    for rule in args.rules:
        for g in args.gains:
            for d in args.depths:
                rmse, nan = train_one(
                    depth=d, rule=rule, gain_w=g, hidden=args.hidden,
                    batch_size=args.batch_size, n_epochs=args.n_epochs,
                    seed=args.seed, data=data, device=device,
                )
                rows.append({"rule": rule, "gain_w": g, "depth": d,
                             "final_rmse": rmse, "ever_nan": nan})
                rmse_s = (f"{rmse:6.3f}" if math.isfinite(rmse) else "  NaN ")
                print(f"    rule={rule:<22}  gain={g:<5}  depth={d}  "
                      f"rmse={rmse_s}  {'(NaN)' if nan else ''}")

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV: {csv_path}")

    # Heatmap: rows = gains, cols = depths, one panel per rule.
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        print("  matplotlib not installed — skipping figures")
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    G, D = len(args.gains), len(args.depths)
    finite_rmses = [r["final_rmse"] for r in rows if math.isfinite(r["final_rmse"])]
    vmin = min(finite_rmses) if finite_rmses else 0.0
    vmax = max(finite_rmses) if finite_rmses else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, len(args.rules), figsize=(5.5 * len(args.rules), 4),
                             squeeze=False)
    for c, rule in enumerate(args.rules):
        ax = axes[0, c]
        mat = np.full((G, D), np.nan)
        for r in rows:
            if r["rule"] != rule:
                continue
            i = args.gains.index(r["gain_w"])
            j = args.depths.index(r["depth"])
            mat[i, j] = r["final_rmse"]
        im = ax.imshow(mat, cmap="viridis_r", origin="upper", norm=norm, aspect="auto")
        ax.set_xticks(range(D), [str(d) for d in args.depths])
        ax.set_yticks(range(G), [f"{g:g}" for g in args.gains])
        ax.set_xlabel("depth")
        ax.set_ylabel("gain_w (smaller ↑ = smaller init prior)")
        ax.set_title(f"{rule}")
        for i in range(G):
            for j in range(D):
                v = mat[i, j]
                if math.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v > (vmin + vmax) / 2 else "black",
                            fontsize=8)
                else:
                    ax.text(j, i, "NaN", ha="center", va="center",
                            color="red", fontsize=7)
        plt.colorbar(im, ax=ax)
    fig.suptitle("TAGI-V regression: final test RMSE vs (gain_w × depth)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"gain_depth.{ext}", dpi=140)
    plt.close(fig)
    print(f"  Figure: {fig_dir}/gain_depth.png")


if __name__ == "__main__":
    main()
