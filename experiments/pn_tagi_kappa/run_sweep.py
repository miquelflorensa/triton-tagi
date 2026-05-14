"""κ-target sweep — does κ ≤ 1 give us a hyperparameter-free cap?

From the identity
    κ² = χ · ζ,
    κ ≡ |Δμ_θ| / σ_θ,
    χ ≡ −Δσ²_θ / σ²_θ,
    ζ ≡ (y − μ_Y)ᵀ Σ_Y⁻¹ (y − μ_Y) / |B|,

and the existing cuTAGI cap

    |Δμ_w| ≤ √σ²_w / c_B   ⟺   κ ≤ 1 / c_B,

the heuristic ``c_B ∈ {0.1, 2, 3}`` is implicitly picking κ-targets ∈
{10, 0.5, 0.33}. The principled anchor ``κ ≤ 1`` corresponds to
``c_B = 1`` and has the clean Bayesian meaning "a parameter never
moves more than one prior standard deviation per batch".

This sweep tests:
    cap_factor ∈ {0.25, 0.5, 1.0, 2.0, 3.0} ⇔ κ_target ∈ {4, 2, 1, 0.5, 0.33}
    rule       ∈ {capped_additive, capped_precision_normalized}
    depth      ∈ {3, 5, 7}

at the canonical MNIST PN-failure regime (σ_v=0.05, B=512, gain_w=1.0,
hidden=256). We also include plain ``precision_normalized`` (no cap) as
a baseline failure case.

Expected outcome:
- κ_target = 1 (cap_factor=1.0) should sit somewhere on the boundary
  between "no cap" (which fails) and the cuTAGI tight cap (which works).
- If κ_target = 1 works for both capped_additive and CPN across depths,
  we have a principled hyperparameter-free cap replacement.

Usage::

    python experiments/pn_tagi_kappa/run_sweep.py
    python experiments/pn_tagi_kappa/run_sweep.py --depths 3 --n_epochs 5
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
#  Data + net builder
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


def build_mlp(*, depth, hidden, gain_w, device, update_rule, rho, cap_factor):
    layers = [Linear(784, hidden, device=device, gain_w=gain_w, gain_b=gain_w), ReLU()]
    for _ in range(depth - 1):
        layers.append(Linear(hidden, hidden, device=device, gain_w=gain_w, gain_b=gain_w))
        layers.append(ReLU())
    layers.append(Linear(hidden, 10, device=device, gain_w=gain_w, gain_b=gain_w))
    layers.append(Remax())
    return Sequential(layers, device=device, update_rule=update_rule, rho=rho,
                      record_chi=False, cap_factor=cap_factor)


# ---------------------------------------------------------------------------
#  Eval + train
# ---------------------------------------------------------------------------


def evaluate(net, x, y_labels, batch_size=1024):
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            mu, _ = net.forward(x[i : i + batch_size])
            correct += int((mu.argmax(dim=1) == y_labels[i : i + batch_size]).sum().item())
    net.train()
    return correct / len(x)


def any_nan(net):
    for layer in net.layers:
        if isinstance(layer, Linear):
            for t in (layer.mw, layer.Sw, layer.mb, layer.Sb):
                if t is None:
                    continue
                if not torch.isfinite(t).all():
                    return True
    return False


def train_one(*, depth, rule, cap_factor, gain_w, hidden, batch_size, sigma_v,
              n_epochs, seed, x_train, y_train_oh, x_test, y_test_labels, device):
    torch.manual_seed(seed)
    net = build_mlp(depth=depth, hidden=hidden, gain_w=gain_w,
                    device=device, update_rule=rule, rho=1.0, cap_factor=cap_factor)

    n_train = x_train.size(0)
    accs = []
    t0 = time.perf_counter()
    for _ in range(n_epochs):
        perm = torch.randperm(n_train, device=device)
        xs, ys = x_train[perm], y_train_oh[perm]
        for i in range(0, n_train, batch_size):
            net.step(xs[i : i + batch_size], ys[i : i + batch_size], sigma_v)
        if any_nan(net):
            break
        accs.append(evaluate(net, x_test, y_test_labels))
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    return {
        "final_acc": (accs[-1] if accs else math.nan),
        "all_accs": accs,
        "ever_nan": any_nan(net),
        "wall_s": wall,
    }


# ---------------------------------------------------------------------------
#  Sweep
# ---------------------------------------------------------------------------


def run_sweep(*, out_dir, depths, cap_factors, rules, gain_w, hidden, batch_size,
              sigma_v, n_epochs, seed, data, device, include_plain_pn=True):
    rows = []
    n_configs = len(depths) * len(cap_factors) * len(rules) + (len(depths) if include_plain_pn else 0)
    i = 0

    for d in depths:
        # Capped variants × cap_factor.
        for rule in rules:
            for cf in cap_factors:
                i += 1
                kappa_target = 1.0 / cf
                res = train_one(
                    depth=d, rule=rule, cap_factor=cf, gain_w=gain_w,
                    hidden=hidden, batch_size=batch_size, sigma_v=sigma_v,
                    n_epochs=n_epochs, seed=seed,
                    x_train=data["x_train"], y_train_oh=data["y_train_oh"],
                    x_test=data["x_test"], y_test_labels=data["y_test_labels"],
                    device=device,
                )
                row = {
                    "config_id": f"{rule}__cf{cf:g}__d{d}",
                    "rule": rule, "cap_factor": cf, "kappa_target": kappa_target,
                    "depth": d,
                    "final_acc": res["final_acc"],
                    "ever_nan": res["ever_nan"],
                    "wall_s": res["wall_s"],
                    "all_accs": ",".join(f"{a:.4f}" for a in res["all_accs"]),
                }
                rows.append(row)
                ta = res["final_acc"]
                ta_s = (f"{ta*100:5.2f}%" if isinstance(ta, float)
                        and math.isfinite(ta) else "  NaN ")
                print(f"  [{i:>2}/{n_configs}] {row['config_id']:<46} "
                      f"κ≤{kappa_target:g}  final={ta_s}  "
                      f"{'(NaN)' if res['ever_nan'] else ''}")

        # Plain PN baseline (no cap) for this depth.
        if include_plain_pn:
            i += 1
            res = train_one(
                depth=d, rule="precision_normalized", cap_factor=None,  # ignored by PN
                gain_w=gain_w, hidden=hidden, batch_size=batch_size, sigma_v=sigma_v,
                n_epochs=n_epochs, seed=seed,
                x_train=data["x_train"], y_train_oh=data["y_train_oh"],
                x_test=data["x_test"], y_test_labels=data["y_test_labels"],
                device=device,
            )
            row = {
                "config_id": f"precision_normalized__no_cap__d{d}",
                "rule": "precision_normalized", "cap_factor": math.nan,
                "kappa_target": math.inf, "depth": d,
                "final_acc": res["final_acc"],
                "ever_nan": res["ever_nan"],
                "wall_s": res["wall_s"],
                "all_accs": ",".join(f"{a:.4f}" for a in res["all_accs"]),
            }
            rows.append(row)
            ta = res["final_acc"]
            ta_s = (f"{ta*100:5.2f}%" if isinstance(ta, float)
                    and math.isfinite(ta) else "  NaN ")
            print(f"  [{i:>2}/{n_configs}] {row['config_id']:<46} "
                  f"κ≤∞ (no cap)  final={ta_s}  "
                  f"{'(NaN)' if res['ever_nan'] else ''}")

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV: {csv_path}")
    return rows


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------


def plot_kappa_sweep(out_dir: Path, rows: list[dict], depths, cap_factors, rules):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figures")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    kappa_targets = [1.0 / cf for cf in cap_factors]

    fig, axes = plt.subplots(1, len(depths), figsize=(4.5 * len(depths), 4.2),
                             sharey=True, squeeze=False)
    for j, d in enumerate(depths):
        ax = axes[0, j]
        for rule in rules:
            xs = []
            ys = []
            for cf in cap_factors:
                match = next((r for r in rows
                              if r["rule"] == rule and r["cap_factor"] == cf
                              and r["depth"] == d), None)
                if match is None:
                    continue
                xs.append(1.0 / cf)
                ys.append(match["final_acc"] * 100 if math.isfinite(match["final_acc"]) else math.nan)
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.5, label=rule)

        # Plain PN baseline (constant — horizontal line for reference).
        pn_match = next((r for r in rows
                         if r["rule"] == "precision_normalized" and r["depth"] == d
                         and not math.isfinite(r["cap_factor"])), None)
        if pn_match is not None and math.isfinite(pn_match["final_acc"]):
            ax.axhline(pn_match["final_acc"] * 100, color="red", linestyle="--",
                       linewidth=1.0, alpha=0.6, label="PN (no cap)")

        ax.axvline(1.0, color="grey", linestyle=":", linewidth=0.8, label="κ=1 anchor")
        ax.set_xscale("log")
        ax.set_xlabel("κ_target = 1 / cap_factor")
        if j == 0:
            ax.set_ylabel("final test accuracy (%)")
        ax.set_title(f"depth = {d}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)

    fig.suptitle(f"κ-target sweep — does κ ≤ 1 give a principled cap?  "
                 f"(MNIST, σ_v=0.05, B=512, gain_w=1.0)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"kappa_target_sweep.{ext}", dpi=140)
    plt.close(fig)
    print(f"  Figure: {fig_dir}/kappa_target_sweep.png")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--cap_factors", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--rules", type=str, nargs="+",
                        default=["capped_additive", "capped_precision_normalized"])
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--sigma_v", type=float, default=0.05)
    parser.add_argument("--gain_w", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("runs") / f"pn_tagi_kappa_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  PN-TAGI — κ-target sweep on MNIST")
    print("=" * 64)
    print(f"  depths       : {args.depths}")
    print(f"  cap_factors  : {args.cap_factors}  (κ_target = 1/cap_factor)")
    print(f"  rules        : {args.rules}")
    print(f"  hidden       : {args.hidden}")
    print(f"  n_epochs     : {args.n_epochs}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  sigma_v      : {args.sigma_v}")
    print(f"  gain_w       : {args.gain_w}")
    print(f"  out_dir      : {out_dir}\n")

    device = torch.device(DEVICE)
    x_train, y_train_oh, _, x_test, y_test_labels = load_mnist(args.data_dir, device)
    data = dict(x_train=x_train, y_train_oh=y_train_oh,
                x_test=x_test, y_test_labels=y_test_labels)

    rows = run_sweep(
        out_dir=out_dir, depths=args.depths, cap_factors=args.cap_factors,
        rules=args.rules, gain_w=args.gain_w, hidden=args.hidden,
        batch_size=args.batch_size, sigma_v=args.sigma_v,
        n_epochs=args.n_epochs, seed=args.seed, data=data, device=device,
    )
    plot_kappa_sweep(out_dir, rows, args.depths, args.cap_factors, args.rules)


if __name__ == "__main__":
    main()
