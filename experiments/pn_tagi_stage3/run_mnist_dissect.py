"""Stage 3 dissection — per-batch health diagnostics on MNIST MLP.

The Stage 3 sweep ``run_mnist_depths.py`` showed PN-TAGI catastrophically
fails at depth ≥ 3 with σ_v=0.05 while capped_additive trains normally.
The per-epoch logs are too coarse to identify *when* and *how* PN-TAGI
collapses, so this script instruments **every batch of every epoch** with:

    Per Linear layer:
        chi_p95, chi_max, chi_median, frac_chi_gt_1     (post-update)
        Sw_mean, Sw_min                                  (post-update)
        mw_mean_abs, mw_max_abs                          (post-update)

    Per layer interface (after a separate forward pass after the update):
        activation_mean, activation_var
        dead_frac  ≡  fraction of units with ma < 1e-6 after a ReLU

    Per batch (output):
        train_acc, mean_output_entropy
        output_max_prob_mean (peaked? uniform?)

Two configs are run side-by-side (PN-TAGI failure vs capped success) and
the per-batch trace is dumped to CSV plus a 4-panel diagnostic plot.

Usage::

    python experiments/pn_tagi_stage3/run_mnist_dissect.py
    python experiments/pn_tagi_stage3/run_mnist_dissect.py \
        --depth 3 --sigma_v 0.05 --n_batches 30 --batch_size 512
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

from triton_tagi import Linear, ReLU, Remax, Sequential
from triton_tagi.update.parameters import chi_stats


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
#  Data
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


# ---------------------------------------------------------------------------
#  Network builder (same as run_mnist_depths.py)
# ---------------------------------------------------------------------------


def build_mlp(depth, hidden, gain_w, device, update_rule, rho):
    layers = [Linear(784, hidden, device=device, gain_w=gain_w, gain_b=gain_w), ReLU()]
    for _ in range(depth - 1):
        layers.append(Linear(hidden, hidden, device=device, gain_w=gain_w, gain_b=gain_w))
        layers.append(ReLU())
    layers.append(Linear(hidden, 10, device=device, gain_w=gain_w, gain_b=gain_w))
    layers.append(Remax())
    return Sequential(layers, device=device, update_rule=update_rule, rho=rho, record_chi=True)


# ---------------------------------------------------------------------------
#  Per-batch diagnostic harness
# ---------------------------------------------------------------------------


def forward_with_taps(net: Sequential, x: torch.Tensor) -> list[tuple]:
    """Walk ``net.layers`` manually, returning (layer_name, ma, Sa) after
    every layer's forward (no parameter updates fire). Activation tensors
    are detached and kept on GPU for later stats — they are small.
    """
    ma = x
    Sa = torch.zeros_like(x)
    taps: list[tuple[str, torch.Tensor, torch.Tensor]] = [("input", ma, Sa)]
    for i, layer in enumerate(net.layers):
        ma, Sa = layer.forward(ma, Sa)
        taps.append((f"{i}:{type(layer).__name__}", ma.detach(), Sa.detach()))
    return taps


def collect_batch_diagnostics(
    net: Sequential,
    x_batch: torch.Tensor,
    y_labels: torch.Tensor,
) -> dict:
    """Run a *probe* forward pass after the parameter update and read:
    per-layer chi stats (already stored on the layer by net.step), Sw
    stats, mw magnitude, per-interface activation health, and Remax
    output entropy / peak prob.
    """
    taps = forward_with_taps(net, x_batch)

    per_layer: list[dict] = []
    linear_idx = 0
    for i, layer in enumerate(net.layers):
        if isinstance(layer, Linear):
            chi_w = getattr(layer, "chi_w", None)
            cs = chi_stats(chi_w) if chi_w is not None else {}
            per_layer.append({
                "linear_idx": linear_idx,
                "seq_idx": i,
                "Sw_mean": float(layer.Sw.mean().item()),
                "Sw_min": float(layer.Sw.min().item()),
                "Sw_max": float(layer.Sw.max().item()),
                "mw_mean_abs": float(layer.mw.abs().mean().item()),
                "mw_max_abs": float(layer.mw.abs().max().item()),
                "chi_p95": cs.get("raw_chi_p95", math.nan),
                "chi_max": cs.get("raw_chi_max", math.nan),
                "chi_median": cs.get("raw_chi_median", math.nan),
                "frac_chi_gt_1": cs.get("frac_chi_gt_1", math.nan),
            })
            linear_idx += 1

    per_iface: list[dict] = []
    for name, ma, Sa in taps:
        per_iface.append({
            "interface": name,
            "act_mean": float(ma.mean().item()),
            "act_var_pred": float(Sa.mean().item()),   # predicted (TAGI) variance
            "act_var_emp": float(ma.var(unbiased=False).item()),  # empirical
            # Fraction of units with ma below a small ReLU-death threshold.
            "frac_below_1e6": float((ma.abs() < 1e-6).float().mean().item()),
        })

    # Remax output is the last tap.
    out_ma = taps[-1][1]
    # Output is non-negative and approximately sums to 1 — clamp before log.
    p = out_ma.clamp_min(1e-12)
    entropy = float((-p * p.log()).sum(dim=1).mean().item())
    max_prob = float(p.max(dim=1).values.mean().item())
    pred = out_ma.argmax(dim=1)
    train_acc = float((pred == y_labels).float().mean().item())

    return {
        "train_acc": train_acc,
        "out_entropy": entropy,
        "out_max_prob": max_prob,
        "per_layer": per_layer,
        "per_iface": per_iface,
    }


def run_one(
    *,
    rule: str,
    depth: int,
    rho: float,
    hidden: int,
    sigma_v: float,
    batch_size: int,
    n_batches: int,
    gain_w: float,
    seed: int,
    x_train,
    y_train_oh,
    y_train_labels,
) -> dict:
    """Train ``n_batches`` mini-batches with per-batch diagnostics.
    Re-shuffles once at the start; does not loop epochs."""
    torch.manual_seed(seed)
    device = torch.device(DEVICE)
    net = build_mlp(depth=depth, hidden=hidden, gain_w=gain_w,
                    device=device, update_rule=rule, rho=rho)

    perm = torch.randperm(x_train.size(0), device=device)
    x_s = x_train[perm]
    y_s = y_train_oh[perm]
    lab_s = y_train_labels[perm]

    rows: list[dict] = []
    # Step 0: pre-update health (no chi yet — record_chi populates only after
    # the first update fires).
    x_b = x_s[:batch_size]
    lb = lab_s[:batch_size]
    init = collect_batch_diagnostics(net, x_b, lb)
    rows.append({"batch": 0, **init})

    for b in range(1, n_batches + 1):
        i = (b - 1) * batch_size
        xb = x_s[i : i + batch_size]
        yb = y_s[i : i + batch_size]
        lb = lab_s[i : i + batch_size]
        if xb.size(0) == 0:
            break
        net.step(xb, yb, sigma_v=sigma_v)
        diag = collect_batch_diagnostics(net, xb, lb)
        rows.append({"batch": b, **diag})
    return {"rows": rows, "rule": rule, "depth": depth}


# ---------------------------------------------------------------------------
#  CSV writers
# ---------------------------------------------------------------------------


def write_traces(out_dir: Path, cfg_id: str, rows: list[dict]) -> None:
    # Flat per-batch globals.
    flat_path = out_dir / f"{cfg_id}__global.csv"
    with flat_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "train_acc", "out_entropy", "out_max_prob"])
        for r in rows:
            w.writerow([r["batch"], r["train_acc"], r["out_entropy"], r["out_max_prob"]])

    # Per-layer wide.
    layer_path = out_dir / f"{cfg_id}__per_layer.csv"
    if rows and rows[0]["per_layer"]:
        first = rows[0]["per_layer"][0]
        fields = ["batch", "linear_idx"] + [k for k in first.keys() if k != "linear_idx"]
        with layer_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
            for r in rows:
                for pl in r["per_layer"]:
                    w.writerow([r["batch"]] + [pl[k] for k in fields[1:]])

    # Per-interface wide.
    iface_path = out_dir / f"{cfg_id}__per_iface.csv"
    if rows and rows[0]["per_iface"]:
        first = rows[0]["per_iface"][0]
        fields = ["batch"] + list(first.keys())
        with iface_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
            for r in rows:
                for it in r["per_iface"]:
                    w.writerow([r["batch"]] + [it[k] for k in fields[1:]])


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------


def plot_dissect(
    out_dir: Path,
    runs: dict[str, dict],
    *,
    rules: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figure")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # --- Panel grid: rows = metric family, cols = rule ---
    # Metrics: (1) train_acc + out_entropy, (2) per-layer chi_p95,
    # (3) per-layer Sw_min, (4) per-interface dead_frac, (5) per-layer mw_mean_abs.
    n_rules = len(rules)
    fig, axes = plt.subplots(5, n_rules, figsize=(5.5 * n_rules, 13), squeeze=False)

    for c, rule in enumerate(rules):
        rows = runs[rule]["rows"]
        batches = [r["batch"] for r in rows]

        # ---- Row 0: train_acc and output entropy (twin axis) ----
        ax = axes[0, c]
        ax.plot(batches, [r["train_acc"] * 100 for r in rows], color="tab:blue",
                label="train_acc (%)", marker="o", markersize=2)
        ax.set_xlabel("batch index")
        ax.set_ylabel("train acc (%)", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(batches, [r["out_entropy"] for r in rows],
                 color="tab:red", linewidth=1.5, label="out_entropy", linestyle="--")
        ax2.axhline(math.log(10), color="grey", linestyle=":",
                    linewidth=0.7, label="uniform (log 10)")
        ax2.set_ylabel("output entropy (nats)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax.set_title(f"{rule}: train_acc + Remax entropy")

        # ---- Row 1: per-layer chi_p95 (log y) ----
        ax = axes[1, c]
        n_lin = len(rows[1]["per_layer"]) if len(rows) > 1 else 0
        for li in range(n_lin):
            ys = [r["per_layer"][li]["chi_p95"] for r in rows[1:]]
            ax.plot(batches[1:], ys, label=f"L{li}", marker=".", markersize=3)
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.set_xlabel("batch index")
        ax.set_ylabel("per-Linear chi_p95")
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.7)
        ax.legend(fontsize=7, ncol=2)
        ax.set_title("post-update chi_p95")

        # ---- Row 2: per-layer Sw_min (log y) ----
        ax = axes[2, c]
        for li in range(n_lin):
            ys = [r["per_layer"][li]["Sw_min"] for r in rows]
            ax.plot(batches, ys, label=f"L{li}", marker=".", markersize=3)
        ax.set_yscale("log")
        ax.axhline(1e-5, color="grey", linestyle=":", linewidth=0.7,
                   label="additive floor")
        ax.set_xlabel("batch index")
        ax.set_ylabel("per-Linear Sw_min")
        ax.legend(fontsize=7, ncol=2)
        ax.set_title("Sw_min")

        # ---- Row 3: per-interface dead-fraction (ReLU + final layer) ----
        ax = axes[3, c]
        if rows and rows[0]["per_iface"]:
            iface_names = [it["interface"] for it in rows[0]["per_iface"]]
            for k, name in enumerate(iface_names):
                # Plot only ReLU outputs and the final Remax output to keep
                # the panel readable.
                if "ReLU" not in name and "Remax" not in name:
                    continue
                ys = [r["per_iface"][k]["frac_below_1e6"] * 100 for r in rows]
                ax.plot(batches, ys, label=name, marker=".", markersize=3)
        ax.set_xlabel("batch index")
        ax.set_ylabel("frac |a| < 1e-6 (%)")
        ax.set_ylim(-2, 102)
        ax.legend(fontsize=7, ncol=2)
        ax.set_title("dead-unit fraction (ReLU + Remax)")

        # ---- Row 4: per-layer mw mean magnitude (log y) ----
        ax = axes[4, c]
        for li in range(n_lin):
            ys = [r["per_layer"][li]["mw_mean_abs"] for r in rows]
            ax.plot(batches, ys, label=f"L{li}", marker=".", markersize=3)
        ax.set_yscale("log")
        ax.set_xlabel("batch index")
        ax.set_ylabel("per-Linear |μ_w| mean")
        ax.legend(fontsize=7, ncol=2)
        ax.set_title("|μ_w| (mean over weights)")

    fig.suptitle(
        f"Stage 3 dissection — depth={runs[rules[0]]['depth']}  "
        f"first {len(runs[rules[0]]['rows']) - 1} batches"
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"dissect.{ext}", dpi=140)
    plt.close(fig)
    print(f"  Figure saved to {fig_dir}/")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=3,
                        help="MLP hidden depth.")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--sigma_v", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_batches", type=int, default=30,
                        help="Number of mini-batches to log per config.")
    parser.add_argument("--gain_w", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--rules", type=str, nargs="+",
        default=["capped_additive", "precision_normalized"],
    )
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("runs") / f"pn_tagi_stage3_dissect_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  PN-TAGI Stage 3 dissection — per-batch health")
    print("=" * 64)
    print(f"  depth      : {args.depth}")
    print(f"  hidden     : {args.hidden}")
    print(f"  sigma_v    : {args.sigma_v}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  n_batches  : {args.n_batches}")
    print(f"  gain_w     : {args.gain_w}")
    print(f"  rho        : {args.rho}")
    print(f"  rules      : {args.rules}")
    print(f"  out_dir    : {out_dir}\n")

    device = torch.device(DEVICE)
    print(f"  Loading MNIST from '{args.data_dir}'...")
    x_train, y_train_oh, y_train_labels, _, _ = load_mnist(args.data_dir, device)

    runs: dict[str, dict] = {}
    for rule in args.rules:
        print(f"\n  ▶ {rule}")
        runs[rule] = run_one(
            rule=rule, depth=args.depth, rho=args.rho, hidden=args.hidden,
            sigma_v=args.sigma_v, batch_size=args.batch_size,
            n_batches=args.n_batches, gain_w=args.gain_w, seed=args.seed,
            x_train=x_train, y_train_oh=y_train_oh, y_train_labels=y_train_labels,
        )
        write_traces(out_dir, rule, runs[rule]["rows"])
        # Brief console summary every few batches.
        for r in runs[rule]["rows"][::max(1, args.n_batches // 8)]:
            print(
                f"      batch {r['batch']:3d}  "
                f"train_acc={r['train_acc']*100:5.2f}%  "
                f"entropy={r['out_entropy']:.3f}  "
                f"max_prob={r['out_max_prob']:.3f}  "
                f"chi_p95(out)={r['per_layer'][-1]['chi_p95']:.2g}  "
                f"Sw_min(out)={r['per_layer'][-1]['Sw_min']:.2g}"
            )

    plot_dissect(out_dir, runs, rules=args.rules)
    print(f"\n  Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
