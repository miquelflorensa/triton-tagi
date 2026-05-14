"""Stage 3 — MNIST FNN with Remax across depths {1, 3, 5, 7}.

Trains a feed-forward classifier ``784 -> [hidden]*depth -> 10`` with Remax
output under each of the configured update rules and records per-epoch:

    * train / test accuracy and CE (Remax output → use NLL on the one-hot
      target = -log(p_true))
    * per-layer chi stats: median, p95, max raw_chi; fraction χ > 1
    * per-layer S_w mean / min / max
    * NaN/Inf count across all parameters

Stage 3 questions this script is built to answer (see PLAN.md):

    1. Does instability correlate with high early χ?
    2. Does PN-TAGI prevent posterior variance collapse?
    3. Does accuracy improve, degrade, or simply stabilise across depth?
    4. Does fixed σ_v become usable over more depths under PN-TAGI?

The full default sweep is 4 depths × 2 rules × ``--n_epochs`` epochs and
takes a few minutes on a modern GPU.

Usage
-----

    python experiments/pn_tagi_stage3/run_mnist_depths.py             # default
    python experiments/pn_tagi_stage3/run_mnist_depths.py --smoke     # tiny grid
    python experiments/pn_tagi_stage3/run_mnist_depths.py \
        --depths 1 3 5 7 --rules capped_additive precision_normalized \
        --n_epochs 5 --sigma_v 0.05
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
from triton_tagi.update.parameters import chi_stats


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

DEFAULT_RULES = (
    "capped_additive",
    "precision_normalized",
    "capped_precision_normalized",
)
ALL_RULES = (
    "additive",
    "capped_additive",
    "precision_normalized",
    "tempered_precision_normalized",
    "capped_precision_normalized",
)


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------


def load_mnist(data_dir: str, device: torch.device):
    """Return (x_train, y_train_oh, y_train_labels, x_test, y_test_labels)."""
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
#  Network builder
# ---------------------------------------------------------------------------


def build_mlp(
    *,
    depth: int,
    hidden: int,
    gain_w: float,
    device: torch.device,
    update_rule: str,
    rho: float,
) -> Sequential:
    """Build ``784 -> [hidden]*depth -> 10`` MLP with ReLU activations and
    a Remax output. ``depth`` counts the hidden layers, not the output
    layer; depth=1 ⇒ 1 hidden + 1 output Linear.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    layers: list = [Linear(784, hidden, device=device, gain_w=gain_w, gain_b=gain_w)]
    layers.append(ReLU())
    for _ in range(depth - 1):
        layers.append(Linear(hidden, hidden, device=device, gain_w=gain_w, gain_b=gain_w))
        layers.append(ReLU())
    layers.append(Linear(hidden, 10, device=device, gain_w=gain_w, gain_b=gain_w))
    layers.append(Remax())
    return Sequential(layers, device=device, update_rule=update_rule, rho=rho, record_chi=True)


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------


def evaluate(net: Sequential, x: torch.Tensor, y_labels: torch.Tensor, batch_size: int = 1024):
    """Return (test_accuracy, mean_NLL) over the full eval set."""
    net.eval()
    correct = 0
    nll_sum = 0.0
    n_total = len(x)
    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            xb = x[i : i + batch_size]
            lb = y_labels[i : i + batch_size]
            mu, _ = net.forward(xb)
            pred = mu.argmax(dim=1)
            correct += (pred == lb).sum().item()
            # Remax output is already p ∈ [0, 1] summing to 1; clamp for log.
            p_true = mu.gather(1, lb.unsqueeze(1)).clamp_min(1e-8).squeeze(1)
            nll_sum += float(-torch.log(p_true).sum().item())
    net.train()
    return correct / n_total, nll_sum / n_total


# ---------------------------------------------------------------------------
#  Per-layer diagnostics
# ---------------------------------------------------------------------------


def collect_layer_diagnostics(net: Sequential) -> list[dict]:
    """One row per Linear layer in network order: per-layer chi stats,
    Sw aggregates, and parameter-mean magnitude.

    Skips activation / Remax layers (they have no Sw).
    """
    rows: list[dict] = []
    layer_idx_in_seq = 0
    linear_idx = 0
    for layer in net.layers:
        if isinstance(layer, Linear):
            chi_w = getattr(layer, "chi_w", None)
            cs = chi_stats(chi_w) if chi_w is not None else {}
            rows.append({
                "linear_idx": linear_idx,
                "layer_idx_in_seq": layer_idx_in_seq,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "Sw_mean": float(layer.Sw.mean().item()),
                "Sw_min": float(layer.Sw.min().item()),
                "Sw_max": float(layer.Sw.max().item()),
                "mw_mean_abs": float(layer.mw.abs().mean().item()),
                "chi_p95": cs.get("raw_chi_p95", math.nan),
                "chi_max": cs.get("raw_chi_max", math.nan),
                "chi_median": cs.get("raw_chi_median", math.nan),
                "frac_chi_gt_1": cs.get("frac_chi_gt_1", math.nan),
            })
            linear_idx += 1
        layer_idx_in_seq += 1
    return rows


def any_nan_or_inf(net: Sequential) -> bool:
    for layer in net.layers:
        if isinstance(layer, Linear):
            for t in (layer.mw, layer.Sw, layer.mb, layer.Sb):
                if t is None:
                    continue
                if not torch.isfinite(t).all():
                    return True
    return False


# ---------------------------------------------------------------------------
#  Per-config training run
# ---------------------------------------------------------------------------


def train_one_config(
    *,
    depth: int,
    rule: str,
    rho: float,
    hidden: int,
    sigma_v: float,
    batch_size: int,
    n_epochs: int,
    gain_w: float,
    seed: int,
    x_train: torch.Tensor,
    y_train_oh: torch.Tensor,
    y_train_labels: torch.Tensor,
    x_test: torch.Tensor,
    y_test_labels: torch.Tensor,
    device: torch.device,
) -> dict:
    """Train one (depth, rule) config and return per-epoch traces."""
    torch.manual_seed(seed)
    net = build_mlp(
        depth=depth, hidden=hidden, gain_w=gain_w,
        device=device, update_rule=rule, rho=rho,
    )

    epoch_rows: list[dict] = []
    per_layer_rows: list[dict] = []

    # Epoch 0 — pre-training snapshot, before any update fires.
    # (No chi yet — record_chi only populates on the first update.)
    test_acc, test_nll = evaluate(net, x_test, y_test_labels)
    epoch_rows.append({
        "epoch": 0,
        "train_acc": math.nan,
        "train_nll": math.nan,
        "test_acc": test_acc,
        "test_nll": test_nll,
        "wall_s": 0.0,
        "any_nan": False,
    })

    n_train = x_train.size(0)
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, device=device)
        x_s = x_train[perm]
        y_s = y_train_oh[perm]
        y_lab_s = y_train_labels[perm]

        n_correct_train = 0
        nll_train_sum = 0.0
        for i in range(0, n_train, batch_size):
            xb = x_s[i : i + batch_size]
            yb = y_s[i : i + batch_size]
            lb = y_lab_s[i : i + batch_size]
            mu_pred, _ = net.step(xb, yb, sigma_v)
            pred = mu_pred.argmax(dim=1)
            n_correct_train += int((pred == lb).sum().item())
            p_true = mu_pred.gather(1, lb.unsqueeze(1)).clamp_min(1e-8).squeeze(1)
            nll_train_sum += float(-torch.log(p_true).sum().item())

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        nan = any_nan_or_inf(net)
        train_acc = n_correct_train / n_train
        train_nll = nll_train_sum / n_train

        if nan:
            test_acc, test_nll = math.nan, math.nan
        else:
            test_acc, test_nll = evaluate(net, x_test, y_test_labels)

        epoch_rows.append({
            "epoch": epoch,
            "train_acc": train_acc,
            "train_nll": train_nll,
            "test_acc": test_acc,
            "test_nll": test_nll,
            "wall_s": wall,
            "any_nan": nan,
        })

        # Per-layer diagnostics for this epoch (after last batch of the epoch).
        for layer_row in collect_layer_diagnostics(net):
            per_layer_rows.append({"epoch": epoch, **layer_row})

        if nan:
            # No point training further once parameters explode.
            print(f"      NaN at epoch {epoch}; aborting this config.")
            break

    return {
        "epoch_rows": epoch_rows,
        "per_layer_rows": per_layer_rows,
        "final_test_acc": epoch_rows[-1]["test_acc"],
        "ever_nan": any(r["any_nan"] for r in epoch_rows),
    }


# ---------------------------------------------------------------------------
#  Sweep
# ---------------------------------------------------------------------------


def run_sweep(
    *,
    out_dir: Path,
    depths: list[int],
    rules: list[str],
    rho: float,
    hidden: int,
    sigma_v: float,
    batch_size: int,
    n_epochs: int,
    gain_w: float,
    seed: int,
    data_dir: str,
) -> dict:
    device = torch.device(DEVICE)
    print(f"  Loading MNIST from '{data_dir}'...", flush=True)
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels = load_mnist(data_dir, device)
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")

    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    results: dict[str, dict] = {}

    n_configs = len(depths) * len(rules)
    i = 0
    for rule in rules:
        for depth in depths:
            i += 1
            cfg_id = f"{rule}__d{depth}"
            print(f"\n  [{i}/{n_configs}] {cfg_id}")
            res = train_one_config(
                depth=depth, rule=rule, rho=rho,
                hidden=hidden, sigma_v=sigma_v, batch_size=batch_size,
                n_epochs=n_epochs, gain_w=gain_w, seed=seed,
                x_train=x_train, y_train_oh=y_train_oh, y_train_labels=y_train_labels,
                x_test=x_test, y_test_labels=y_test_labels,
                device=device,
            )
            results[cfg_id] = res

            # Per-epoch CSV.
            ep_path = traces_dir / f"{cfg_id}__epochs.csv"
            with ep_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(res["epoch_rows"][0].keys()))
                w.writeheader()
                w.writerows(res["epoch_rows"])

            # Per-layer CSV.
            if res["per_layer_rows"]:
                pl_path = traces_dir / f"{cfg_id}__layers.csv"
                with pl_path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(res["per_layer_rows"][0].keys()))
                    w.writeheader()
                    w.writerows(res["per_layer_rows"])

            # Per-epoch summary line.
            for r in res["epoch_rows"]:
                ta = r["test_acc"]
                tn = r["test_nll"]
                ta_str = f"{ta*100:5.2f}%" if isinstance(ta, float) and math.isfinite(ta) else "  NaN "
                tn_str = f"{tn:6.3f}" if isinstance(tn, float) and math.isfinite(tn) else "  NaN "
                print(
                    f"      epoch {r['epoch']:2d}  "
                    f"train_acc={r['train_acc']*100 if math.isfinite(r['train_acc']) else float('nan'):5.2f}%  "
                    f"test_acc={ta_str}  test_nll={tn_str}  wall={r['wall_s']:5.1f}s"
                )

            # Build a summary row.
            # Pick the early-chi snapshot from epoch=1 (first post-update reading).
            ep1_layers = [r for r in res["per_layer_rows"] if r["epoch"] == 1]
            early_chi_max = (
                max((r["chi_max"] for r in ep1_layers if math.isfinite(r["chi_max"])),
                    default=math.nan)
            )
            early_chi_p95 = (
                max((r["chi_p95"] for r in ep1_layers if math.isfinite(r["chi_p95"])),
                    default=math.nan)
            )
            summary_rows.append({
                "config_id": cfg_id,
                "rule": rule,
                "depth": depth,
                "final_test_acc": res["final_test_acc"],
                "ever_nan": res["ever_nan"],
                "early_chi_max": early_chi_max,
                "early_chi_p95": early_chi_p95,
            })

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n  Summary written to {summary_path}")
    return {"results": results, "summary_rows": summary_rows}


# ---------------------------------------------------------------------------
#  Plots
# ---------------------------------------------------------------------------


def make_plots(
    *,
    out_dir: Path,
    results: dict[str, dict],
    depths: list[int],
    rules: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figures (pip install matplotlib)")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # --- 1. Test accuracy vs epoch, one panel per depth, line per rule. ---
    fig, axes = plt.subplots(
        1, len(depths),
        figsize=(3.6 * len(depths), 3.6),
        sharey=True, squeeze=False,
    )
    for j, depth in enumerate(depths):
        ax = axes[0, j]
        for rule in rules:
            cfg_id = f"{rule}__d{depth}"
            ep = results[cfg_id]["epoch_rows"]
            xs = [r["epoch"] for r in ep]
            ys = [r["test_acc"] * 100 if math.isfinite(r["test_acc"]) else math.nan for r in ep]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=rule)
        ax.set_title(f"depth = {depth}")
        ax.set_xlabel("epoch")
        if j == 0:
            ax.set_ylabel("test accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Test accuracy across depth × rule")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"accuracy.{ext}", dpi=140)
    plt.close(fig)

    # --- 2. Per-layer chi_p95 heatmap across (linear_idx × epoch), one panel per
    #        (depth × rule) cell. ---
    n_rows, n_cols = len(rules), len(depths)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), squeeze=False
    )
    # Determine global vmin/vmax for shared color scale (log).
    all_chi = []
    for r_row in results.values():
        for pl in r_row["per_layer_rows"]:
            v = pl["chi_p95"]
            if math.isfinite(v) and v > 0:
                all_chi.append(v)
    if all_chi:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=max(min(all_chi), 1e-8), vmax=max(all_chi))
    else:
        norm = None
    cmap = "magma"

    for i, rule in enumerate(rules):
        for j, depth in enumerate(depths):
            ax = axes[i, j]
            cfg_id = f"{rule}__d{depth}"
            pl = results[cfg_id]["per_layer_rows"]
            if not pl:
                ax.set_title(f"{rule} d={depth} (no data)")
                ax.set_axis_off()
                continue
            n_lin = max(r["linear_idx"] for r in pl) + 1
            n_ep = max(r["epoch"] for r in pl)
            mat = np.full((n_lin, n_ep), np.nan)
            for r in pl:
                mat[r["linear_idx"], r["epoch"] - 1] = r["chi_p95"]
            im = ax.imshow(
                mat, aspect="auto", cmap=cmap, norm=norm, origin="upper",
            )
            ax.set_xticks(range(n_ep), [str(e + 1) for e in range(n_ep)])
            ax.set_yticks(range(n_lin))
            ax.set_xlabel("epoch")
            if j == 0:
                ax.set_ylabel(f"{rule}\nLinear idx")
            else:
                ax.set_ylabel("Linear idx")
            ax.set_title(f"depth = {depth}")
            for li in range(n_lin):
                for ep in range(n_ep):
                    v = mat[li, ep]
                    if math.isfinite(v):
                        ax.text(ep, li, f"{v:.1g}", ha="center", va="center",
                                color="white" if v > 1.0 else "black", fontsize=6)
    fig.suptitle("Per-layer p95 raw_chi across (depth × rule)")
    fig.tight_layout()
    if norm is not None:
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cax)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"chi_p95_heatmap.{ext}", dpi=140)
    plt.close(fig)

    # --- 3. Sw_mean per layer over epochs, one panel per (depth × rule). ---
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), squeeze=False
    )
    for i, rule in enumerate(rules):
        for j, depth in enumerate(depths):
            ax = axes[i, j]
            cfg_id = f"{rule}__d{depth}"
            pl = results[cfg_id]["per_layer_rows"]
            if not pl:
                ax.set_title(f"{rule} d={depth} (no data)")
                ax.set_axis_off()
                continue
            n_lin = max(r["linear_idx"] for r in pl) + 1
            for li in range(n_lin):
                xs = sorted({r["epoch"] for r in pl if r["linear_idx"] == li})
                ys = [next(r["Sw_mean"] for r in pl if r["linear_idx"] == li and r["epoch"] == ep) for ep in xs]
                ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.0, label=f"L{li}")
            ax.set_yscale("log")
            ax.axhline(1e-5, color="grey", linestyle=":", linewidth=0.6)
            ax.set_xlabel("epoch")
            if j == 0:
                ax.set_ylabel(f"{rule}\nSw mean")
            else:
                ax.set_ylabel("Sw mean")
            ax.set_title(f"depth = {depth}")
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Per-layer S_w mean over training")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"Sw_trace.{ext}", dpi=140)
    plt.close(fig)

    print(f"  Figures saved to {fig_dir}/")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


_SMOKE = dict(depths=[1], rules=["capped_additive", "precision_normalized"], n_epochs=1)
_DEFAULT = dict(
    depths=[1, 3, 5, 7],
    rules=list(DEFAULT_RULES),
    n_epochs=5,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny grid (depth=1, 1 epoch) — verify the script runs.")
    parser.add_argument("--depths", type=int, nargs="+", default=None,
                        help=f"Default: {_DEFAULT['depths']}")
    parser.add_argument("--rules", type=str, nargs="+", default=None,
                        help=f"Subset of {list(ALL_RULES)}. Default: {_DEFAULT['rules']}")
    parser.add_argument("--n_epochs", type=int, default=None,
                        help=f"Default: {_DEFAULT['n_epochs']}")
    parser.add_argument("--rho", type=float, default=1.0,
                        help="Temperature for precision_normalized variants.")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--sigma_v", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--gain_w", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: runs/pn_tagi_stage3_<timestamp>).")
    args = parser.parse_args()

    defaults = _SMOKE if args.smoke else _DEFAULT
    depths = args.depths or defaults["depths"]
    rules = args.rules or defaults["rules"]
    n_epochs = args.n_epochs or defaults["n_epochs"]

    for r in rules:
        if r not in ALL_RULES:
            raise SystemExit(f"unknown rule: {r!r} (choices: {list(ALL_RULES)})")

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"pn_tagi_stage3_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  PN-TAGI Stage 3 — MNIST MLP across depths")
    print("=" * 64)
    print(f"  device      : {DEVICE}")
    print(f"  depths      : {depths}")
    print(f"  rules       : {rules}")
    print(f"  hidden      : {args.hidden}")
    print(f"  n_epochs    : {n_epochs}")
    print(f"  sigma_v     : {args.sigma_v}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  gain_w      : {args.gain_w}")
    print(f"  rho         : {args.rho}")
    print(f"  out_dir     : {out_dir}\n")

    sweep = run_sweep(
        out_dir=out_dir,
        depths=depths,
        rules=rules,
        rho=args.rho,
        hidden=args.hidden,
        sigma_v=args.sigma_v,
        batch_size=args.batch_size,
        n_epochs=n_epochs,
        gain_w=args.gain_w,
        seed=args.seed,
        data_dir=args.data_dir,
    )

    make_plots(
        out_dir=out_dir,
        results=sweep["results"],
        depths=depths,
        rules=rules,
    )

    print("\n  Final summary:")
    for r in sweep["summary_rows"]:
        ta = r["final_test_acc"]
        ta_str = f"{ta*100:5.2f}%" if isinstance(ta, float) and math.isfinite(ta) else "  NaN "
        print(
            f"    {r['config_id']:<40} "
            f"final={ta_str}  early_chi_p95={r['early_chi_p95']:.2g}  "
            f"{'(NaN)' if r['ever_nan'] else ''}"
        )
    print(f"\n  Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
