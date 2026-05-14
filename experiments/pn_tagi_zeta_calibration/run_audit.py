"""Init-only ζ audit — is the Remax observation model calibrated?

We freeze the update rule at the κ-anchored CPN found in the previous
sweep (cap_factor = 1.0, ρ = 1, never apply update during the audit)
and compare four observation-variance models that change how the
output innovation is computed:

    A. current:               σ²_Y_i = σ²_A_i + σ_v²
    B. categorical diag:      σ²_Y_i = σ²_A_i + μ_A_i (1 − μ_A_i)
    C. categorical + σ_v:     σ²_Y_i = σ²_A_i + μ_A_i (1 − μ_A_i) + σ_v²
    D. label smoothing + cur: σ²_Y_i = σ²_A_i + σ_v²,
                              y → (1 − α) · onehot + α / K

For each fresh init we do ONE forward, then run the layer-wise
backward four times — once per model — without touching the parameters.
That gives a per-model snapshot of (δμ_w, δσ²_w) at every Linear, from
which we compute κ, χ, and the cap-activation fraction under the κ ≤ 1
rule. ζ is read straight from `(y − μ_Y)² / Σ_Y` at the output.

Calibration target: ζ_dof ≡ ζ_raw / (K − 1) ≈ 1, where K is the number
of classes. The (K − 1) normaliser comes from the Remax simplex
constraint ∑_i μ_Y_i = 1.

Usage::

    python experiments/pn_tagi_zeta_calibration/run_audit.py
    python experiments/pn_tagi_zeta_calibration/run_audit.py --depths 3 5 7 --n_batches 10
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 10
LABEL_SMOOTH_ALPHA = 0.1


# ---------------------------------------------------------------------------
#  Data + net
# ---------------------------------------------------------------------------


def load_mnist(data_dir: str, device: torch.device):
    train_ds = datasets.MNIST(data_dir, train=True, download=True)
    x_train = train_ds.data.float().view(-1, 784) / 255.0
    mu, sigma = x_train.mean(), x_train.std()
    x_train = ((x_train - mu) / sigma).to(device)
    y_train_labels = train_ds.targets.to(device)
    y_train_oh = torch.zeros(len(y_train_labels), 10, device=device)
    y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 1.0)
    return x_train, y_train_oh, y_train_labels


def build_mlp(*, depth, hidden, gain_w, device):
    """Same architecture as Stage 3 (κ-anchored CPN rule baked in via
    update_rule + cap_factor, though we never apply the update here)."""
    layers = [Linear(784, hidden, device=device, gain_w=gain_w, gain_b=gain_w), ReLU()]
    for _ in range(depth - 1):
        layers.append(Linear(hidden, hidden, device=device, gain_w=gain_w, gain_b=gain_w))
        layers.append(ReLU())
    layers.append(Linear(hidden, 10, device=device, gain_w=gain_w, gain_b=gain_w))
    layers.append(Remax())
    return Sequential(
        layers, device=device,
        update_rule="capped_precision_normalized", rho=1.0, record_chi=False,
        cap_factor=1.0,  # κ ≤ 1 anchor
    )


# ---------------------------------------------------------------------------
#  Per-model innovation
# ---------------------------------------------------------------------------


def innovation_for_model(model: str, y_oh, mu_Y, S_A, sigma_v):
    """Return (δμ_y, δS_y, σ²_Y, y_used, ζ) for the chosen observation model.

    The compute_innovation kernel reads δμ = (y − μ)/Σ, δS = −1/Σ. We
    duplicate that here in PyTorch so we can plug in arbitrary Σ_Y
    expressions without writing a new kernel.
    """
    if model == "A":
        sigma_Y_sq = S_A + sigma_v ** 2
        y_used = y_oh
    elif model == "B":
        sigma_Y_sq = S_A + mu_Y * (1.0 - mu_Y)
        y_used = y_oh
    elif model == "C":
        sigma_Y_sq = S_A + mu_Y * (1.0 - mu_Y) + sigma_v ** 2
        y_used = y_oh
    elif model == "D":
        # Label-smooth then use the current σ_v model.
        sigma_Y_sq = S_A + sigma_v ** 2
        y_used = (1.0 - LABEL_SMOOTH_ALPHA) * y_oh + LABEL_SMOOTH_ALPHA / N_CLASSES
    else:
        raise ValueError(f"unknown model {model!r}")

    sigma_Y_sq = sigma_Y_sq.clamp_min(1e-8)
    residual = y_used - mu_Y
    delta_mu_y = residual / sigma_Y_sq
    delta_S_y = -1.0 / sigma_Y_sq
    # ζ per sample = Σ_i (y_i - μ_Y_i)² / σ²_Y_i, then mean across batch.
    zeta = ((residual ** 2) / sigma_Y_sq).sum(dim=-1).mean().item()
    return delta_mu_y, delta_S_y, sigma_Y_sq, y_used, zeta


# ---------------------------------------------------------------------------
#  Per-layer diagnostic from the backward deltas
# ---------------------------------------------------------------------------


def per_layer_stats(net):
    """Read δμ_w, δσ²_w from every Linear and produce κ, χ, cap-activation."""
    rows = []
    linear_idx = 0
    for layer in net.layers:
        if not isinstance(layer, Linear):
            continue
        Sw = layer.Sw
        dmw = layer.delta_mw
        dSw = layer.delta_Sw
        sigma_w = Sw.sqrt().clamp_min(1e-8)
        kappa = dmw.abs() / sigma_w
        kappa_sq = (dmw ** 2) / Sw.clamp_min(1e-12)
        chi = -dSw / Sw.clamp_min(1e-12)
        # Cap-activation under κ ≤ 1: fraction of weights with |Δμ_w| > σ_w.
        cap_active = (dmw.abs() > sigma_w).float().mean().item()
        rows.append({
            "linear_idx": linear_idx,
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "kappa_median": float(kappa.median().item()),
            "kappa_p95":   float(kappa.quantile(0.95).item()),
            "kappa_max":   float(kappa.max().item()),
            "kappa_sq_median": float(kappa_sq.median().item()),
            "chi_p95": float(chi.quantile(0.95).item()),
            "chi_max": float(chi.max().item()),
            "cap_activation_frac": cap_active,
        })
        linear_idx += 1
    return rows


# ---------------------------------------------------------------------------
#  Audit one (depth, seed) pair across all four models
# ---------------------------------------------------------------------------


def audit_once(*, depth, hidden, gain_w, sigma_v, batch_size, n_batches,
               seed, x_train, y_train_oh, device):
    """Build a fresh net, average per-model diagnostics over n_batches of
    fresh init forwards.
    """
    torch.manual_seed(seed)
    net = build_mlp(depth=depth, hidden=hidden, gain_w=gain_w, device=device)

    # Sample n_batches fresh mini-batches from the training set.
    g = torch.Generator(device=device).manual_seed(seed + 1)
    n_train = x_train.size(0)

    # Accumulate per-model stats across batches.
    accum = {m: {"zeta": [], "per_layer": []} for m in ("A", "B", "C", "D")}

    for b in range(n_batches):
        # Fresh batch (no shuffling persistence between iters).
        idx = torch.randint(0, n_train, (batch_size,), generator=g, device=device)
        xb = x_train[idx]
        yb = y_train_oh[idx]

        # Forward once; cached activations are reused across the 4 backwards.
        mu_Y, S_A = net.forward(xb)

        for model in ("A", "B", "C", "D"):
            dmu_y, dS_y, _, _, zeta = innovation_for_model(
                model, yb, mu_Y, S_A, sigma_v,
            )
            # Layer-wise backward: pass δμ_y / δS_y through the network in
            # reverse order. Each layer.backward overwrites its δμ_w /
            # δσ²_w buffers, which we then read.
            cur_dmu = dmu_y
            cur_dvar = dS_y
            for layer in reversed(net.layers):
                cur_dmu, cur_dvar = layer.backward(cur_dmu, cur_dvar)

            accum[model]["zeta"].append(zeta)
            accum[model]["per_layer"].append(per_layer_stats(net))

    # Average over batches.
    result = {}
    for m, d in accum.items():
        zeta_avg = float(np.mean(d["zeta"]))
        # Per-layer averages.
        n_lin = len(d["per_layer"][0])
        per_layer_avg = []
        for li in range(n_lin):
            stats = {}
            for key in d["per_layer"][0][li]:
                if isinstance(d["per_layer"][0][li][key], (int, float)):
                    stats[key] = float(np.mean([batch[li][key] for batch in d["per_layer"]]))
                else:
                    stats[key] = d["per_layer"][0][li][key]
            per_layer_avg.append(stats)
        result[m] = {
            "zeta_raw": zeta_avg,
            "zeta_over_K": zeta_avg / N_CLASSES,
            "zeta_dof":   zeta_avg / (N_CLASSES - 1),
            "per_layer": per_layer_avg,
            # Network-wide aggregates.
            "kappa_p95_max":  max(s["kappa_p95"]  for s in per_layer_avg),
            "kappa_median_max": max(s["kappa_median"] for s in per_layer_avg),
            "chi_p95_max":    max(s["chi_p95"]    for s in per_layer_avg),
            "cap_active_max": max(s["cap_activation_frac"] for s in per_layer_avg),
            "cap_active_mean": float(np.mean([s["cap_activation_frac"] for s in per_layer_avg])),
        }
    return result


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--gain_w", type=float, default=1.0)
    parser.add_argument("--sigma_v", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("runs") / f"pn_tagi_zeta_audit_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  ζ audit — Remax observation-model calibration at init")
    print("=" * 64)
    print(f"  depths     : {args.depths}")
    print(f"  hidden     : {args.hidden}")
    print(f"  gain_w     : {args.gain_w}")
    print(f"  sigma_v    : {args.sigma_v}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  n_batches  : {args.n_batches}")
    print(f"  K          : {N_CLASSES}  → ζ_dof normaliser = K-1 = {N_CLASSES - 1}")
    print(f"  label_α    : {LABEL_SMOOTH_ALPHA}  (Model D)\n")

    device = torch.device(DEVICE)
    print(f"  Loading MNIST from '{args.data_dir}'...")
    x_train, y_train_oh, _ = load_mnist(args.data_dir, device)

    summary_rows = []
    for d in args.depths:
        print(f"\n  ── depth = {d} ──")
        res = audit_once(
            depth=d, hidden=args.hidden, gain_w=args.gain_w,
            sigma_v=args.sigma_v, batch_size=args.batch_size,
            n_batches=args.n_batches, seed=args.seed,
            x_train=x_train, y_train_oh=y_train_oh, device=device,
        )
        # Print a quick table.
        header = (f"    {'model':<24} {'ζ_raw':>10} {'ζ/K':>8} {'ζ_dof':>8} "
                  f"{'κ_p95':>8} {'κ_med':>8} {'χ_p95':>10} "
                  f"{'cap%_max':>9} {'cap%_mean':>10}")
        print(header)
        print("    " + "-" * (len(header) - 4))
        for m_key, m_label in [
            ("A", "A: σ²_A + σ_v²"),
            ("B", "B: σ²_A + μ(1-μ)"),
            ("C", "C: σ²_A + μ(1-μ) + σ_v²"),
            ("D", "D: label-smooth + cur"),
        ]:
            r = res[m_key]
            print(f"    {m_label:<24} {r['zeta_raw']:10.3f} {r['zeta_over_K']:8.3f} "
                  f"{r['zeta_dof']:8.3f} {r['kappa_p95_max']:8.3f} "
                  f"{r['kappa_median_max']:8.3f} {r['chi_p95_max']:10.4f} "
                  f"{r['cap_active_max']*100:8.2f}% {r['cap_active_mean']*100:9.2f}%")
            summary_rows.append({
                "depth": d, "model": m_key, "model_label": m_label,
                "zeta_raw": r["zeta_raw"], "zeta_over_K": r["zeta_over_K"],
                "zeta_dof": r["zeta_dof"],
                "kappa_p95_max": r["kappa_p95_max"],
                "kappa_median_max": r["kappa_median_max"],
                "chi_p95_max": r["chi_p95_max"],
                "cap_active_max": r["cap_active_max"],
                "cap_active_mean": r["cap_active_mean"],
            })

        # Per-layer detail CSV per depth.
        per_layer_path = out_dir / f"per_layer_d{d}.csv"
        with per_layer_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "linear_idx", "in_features", "out_features",
                        "kappa_median", "kappa_p95", "kappa_max",
                        "chi_p95", "chi_max", "cap_activation_frac"])
            for m in ("A", "B", "C", "D"):
                for s in res[m]["per_layer"]:
                    w.writerow([m, s["linear_idx"], s["in_features"], s["out_features"],
                                s["kappa_median"], s["kappa_p95"], s["kappa_max"],
                                s["chi_p95"], s["chi_max"], s["cap_activation_frac"]])

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n  Summary CSV: {summary_path}")

    # Plot.
    plot_audit(out_dir, summary_rows, args.depths)
    print(f"\n  Done. Results in {out_dir}")


def plot_audit(out_dir: Path, rows: list[dict], depths):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figures")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    models = ["A", "B", "C", "D"]
    model_labels = {"A": "A: σ²_A + σ_v²", "B": "B: + μ(1-μ)",
                    "C": "C: + μ(1-μ) + σ_v²", "D": "D: label-smooth"}
    colors = {"A": "tab:red", "B": "tab:green", "C": "tab:blue", "D": "tab:purple"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Top-left: ζ_dof per (depth, model).
    ax = axes[0, 0]
    for m in models:
        ys = [next(r["zeta_dof"] for r in rows if r["depth"] == d and r["model"] == m)
              for d in depths]
        ax.plot(depths, ys, marker="o", markersize=6, linewidth=1.5,
                color=colors[m], label=model_labels[m])
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, label="ζ_dof = 1 (calibrated)")
    ax.set_xlabel("depth")
    ax.set_ylabel("ζ_dof  =  ζ_raw / (K − 1)")
    ax.set_yscale("log")
    ax.set_title("ζ_dof — closer to 1 = better calibrated")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Top-right: ζ_raw per (depth, model).
    ax = axes[0, 1]
    for m in models:
        ys = [next(r["zeta_raw"] for r in rows if r["depth"] == d and r["model"] == m)
              for d in depths]
        ax.plot(depths, ys, marker="o", markersize=6, linewidth=1.5,
                color=colors[m], label=model_labels[m])
    ax.axhline(N_CLASSES, color="grey", linestyle=":", linewidth=0.6, label="K = 10")
    ax.axhline(N_CLASSES - 1, color="grey", linestyle="--", linewidth=0.6, label="K - 1 = 9")
    ax.set_xlabel("depth")
    ax.set_ylabel("ζ_raw")
    ax.set_yscale("log")
    ax.set_title("ζ_raw (sum across classes)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Bottom-left: cap-activation fraction (network mean).
    ax = axes[1, 0]
    for m in models:
        ys = [next(r["cap_active_mean"] * 100 for r in rows if r["depth"] == d and r["model"] == m)
              for d in depths]
        ax.plot(depths, ys, marker="o", markersize=6, linewidth=1.5,
                color=colors[m], label=model_labels[m])
    ax.set_xlabel("depth")
    ax.set_ylabel("cap-activation (%)  [mean across layers]")
    ax.set_title("Cap-activation fraction under κ ≤ 1")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Bottom-right: κ_p95 (worst layer).
    ax = axes[1, 1]
    for m in models:
        ys = [next(r["kappa_p95_max"] for r in rows if r["depth"] == d and r["model"] == m)
              for d in depths]
        ax.plot(depths, ys, marker="o", markersize=6, linewidth=1.5,
                color=colors[m], label=model_labels[m])
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, label="κ = 1 anchor")
    ax.set_xlabel("depth")
    ax.set_ylabel("κ_p95  (worst layer)")
    ax.set_yscale("log")
    ax.set_title("κ_p95 across the network at init")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Init-only ζ audit (MNIST, κ ≤ 1 cap, no update applied)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"zeta_audit.{ext}", dpi=140)
    plt.close(fig)
    print(f"  Figure: {fig_dir}/zeta_audit.png")


if __name__ == "__main__":
    main()
