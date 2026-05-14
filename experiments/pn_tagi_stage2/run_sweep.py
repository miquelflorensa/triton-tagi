"""Stage 2 — Tiny linear TAGI regression sweep.

A single-layer ``Linear`` (no activation, no output transform) is the
simplest setting where TAGI's local Gaussian assumption is exact, so
contraction behaviour can be observed clearly without confounds from
deeper architectures.

This script sweeps:

    - batch_size   : {1, 8, 32, 128, 256}
    - sigma_v      : {0.01, 0.05, 0.2, 1.0}
    - gain_w       : {0.5, 1.0, 2.0}
    - update_rule  : {additive, capped_additive, precision_normalized}

For each config it trains for ``--n_steps`` steps and records, per step:

    - median / p95 / max raw_chi   (posterior contraction ratio)
    - fraction of weights with chi > 1
    - mean S_w
    - min S_w  (hits the 1e-5 floor when additive collapses)
    - batch train MSE

A 1024-sample validation set is evaluated at the end of each run, and
the per-config summary plus per-step traces are written to ``runs/``
under a timestamped directory along with three figures:

    - heatmap of validation MSE  (batch × sigma_v, one panel per rule)
    - heatmap of initial chi p95 (same axes)
    - line plot of S_w over training for a fixed config across rules

Usage
-----

    python experiments/pn_tagi_stage2/run_sweep.py                    # full sweep
    python experiments/pn_tagi_stage2/run_sweep.py --quick            # tiny grid
    python experiments/pn_tagi_stage2/run_sweep.py --n_steps 100 \
        --batch_sizes 1 8 32 --sigma_vs 0.1 1.0 --gain_ws 1.0

The script is self-contained — no dependency on the test module — so
the harness duplicates the small training loop used by
``tests/unit/test_pn_tagi_stage2.py``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
from pathlib import Path

import numpy as np
import torch

from triton_tagi import Linear, Sequential


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

UPDATE_RULES = ("additive", "capped_additive", "precision_normalized")


# ---------------------------------------------------------------------------
#  Harness
# ---------------------------------------------------------------------------


def run_linear_experiment(
    *,
    in_features: int,
    batch_size: int,
    sigma_v: float,
    gain_w: float,
    update_rule: str,
    n_steps: int,
    seed: int,
    sigma_obs: float | None = None,
) -> dict:
    """One training run with the given config. See module docstring for
    the data model and which traces are recorded.

    Mirrors ``tests/unit/test_pn_tagi_stage2.run_linear_experiment`` but
    is intentionally duplicated so this script has no test dependency.
    """
    if sigma_obs is None:
        sigma_obs = sigma_v

    torch.manual_seed(seed)
    np.random.seed(seed)

    W_true = torch.randn(in_features, 1, device=DEVICE, dtype=DTYPE)
    b_true = torch.randn(1, device=DEVICE, dtype=DTYPE) * 0.1

    net = Sequential(
        [Linear(in_features, 1, device=DEVICE, gain_w=gain_w, gain_b=gain_w, bias=True)],
        device=DEVICE,
        update_rule=update_rule,
        rho=1.0,
        record_chi=True,
    )
    linear: Linear = net.layers[0]  # type: ignore[assignment]

    g = torch.Generator(device=DEVICE).manual_seed(seed + 1)

    chi_p95 = np.empty(n_steps, dtype=np.float32)
    chi_max = np.empty(n_steps, dtype=np.float32)
    chi_med = np.empty(n_steps, dtype=np.float32)
    chi_frac_gt_1 = np.empty(n_steps, dtype=np.float32)
    Sw_mean = np.empty(n_steps, dtype=np.float32)
    Sw_min = np.empty(n_steps, dtype=np.float32)
    mse = np.empty(n_steps, dtype=np.float32)

    for step in range(n_steps):
        x = torch.randn(batch_size, in_features, generator=g, device=DEVICE, dtype=DTYPE)
        noise = sigma_obs * torch.randn(batch_size, 1, generator=g, device=DEVICE, dtype=DTYPE)
        y = x @ W_true + b_true + noise

        mu_pred, _ = net.step(x, y, sigma_v=sigma_v)

        mse[step] = ((mu_pred - y) ** 2).mean().item()
        Sw_mean[step] = linear.Sw.mean().item()
        Sw_min[step] = linear.Sw.min().item()

        s = net.collect_chi_stats().get("0.chi_w", {})
        chi_p95[step] = s.get("raw_chi_p95", math.nan)
        chi_max[step] = s.get("raw_chi_max", math.nan)
        chi_med[step] = s.get("raw_chi_median", math.nan)
        chi_frac_gt_1[step] = s.get("frac_chi_gt_1", math.nan)

    # Validation MSE on a held-out batch.
    x_val = torch.randn(1024, in_features, generator=g, device=DEVICE, dtype=DTYPE)
    y_val = x_val @ W_true + b_true
    net.eval()
    with torch.no_grad():
        mu_val, _ = net.forward(x_val)
    val_mse = float(((mu_val - y_val) ** 2).mean().item())

    any_nan = (
        not math.isfinite(linear.mw.sum().item())
        or not math.isfinite(linear.Sw.sum().item())
    )

    return {
        "chi_p95": chi_p95,
        "chi_max": chi_max,
        "chi_med": chi_med,
        "chi_frac_gt_1": chi_frac_gt_1,
        "Sw_mean": Sw_mean,
        "Sw_min": Sw_min,
        "mse": mse,
        "val_mse": val_mse,
        "any_nan": any_nan,
    }


# ---------------------------------------------------------------------------
#  Sweep + I/O
# ---------------------------------------------------------------------------


def _config_id(rule: str, B: int, sv: float, gw: float) -> str:
    return f"{rule}__B{B}__sv{sv:g}__gw{gw:g}"


def run_sweep(
    *,
    out_dir: Path,
    batch_sizes: list[int],
    sigma_vs: list[float],
    gain_ws: list[float],
    rules: list[str],
    in_features: int,
    n_steps: int,
    seed: int,
) -> dict:
    """Run the full sweep and write per-step traces + a config summary.

    Returns the in-memory ``results`` dict (keyed by config id) so the
    plotting code can reuse it without re-loading from disk.
    """
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    summary_rows: list[dict] = []

    n_configs = len(batch_sizes) * len(sigma_vs) * len(gain_ws) * len(rules)
    print(f"  Running {n_configs} configs ({n_steps} steps each)...")
    i = 0
    for rule in rules:
        for B in batch_sizes:
            for sv in sigma_vs:
                for gw in gain_ws:
                    i += 1
                    cfg_id = _config_id(rule, B, sv, gw)
                    out = run_linear_experiment(
                        in_features=in_features,
                        batch_size=B,
                        sigma_v=sv,
                        gain_w=gw,
                        update_rule=rule,
                        n_steps=n_steps,
                        seed=seed,
                    )
                    results[cfg_id] = out

                    # Per-step trace (one CSV per config).
                    with (traces_dir / f"{cfg_id}.csv").open("w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(
                            ["step", "chi_med", "chi_p95", "chi_max",
                             "chi_frac_gt_1", "Sw_mean", "Sw_min", "mse"]
                        )
                        for s in range(n_steps):
                            w.writerow([
                                s,
                                float(out["chi_med"][s]),
                                float(out["chi_p95"][s]),
                                float(out["chi_max"][s]),
                                float(out["chi_frac_gt_1"][s]),
                                float(out["Sw_mean"][s]),
                                float(out["Sw_min"][s]),
                                float(out["mse"][s]),
                            ])

                    # Only additive / capped_additive have a 1e-5 numerical
                    # safety floor; PN-TAGI's small Sw values are legitimate
                    # contraction, not floor activation. Detect floor hits as
                    # *exactly* 1e-5 within fp tolerance.
                    floored = (
                        rule in ("additive", "capped_additive")
                        and bool((np.abs(out["Sw_min"] - 1e-5) < 1e-8).any())
                    )
                    summary_rows.append({
                        "config_id": cfg_id,
                        "rule": rule,
                        "batch_size": B,
                        "sigma_v": sv,
                        "gain_w": gw,
                        "chi_p95_initial": float(out["chi_p95"][0]),
                        "chi_max_initial": float(out["chi_max"][0]),
                        "Sw_min_final": float(out["Sw_min"][-1]),
                        "Sw_mean_final": float(out["Sw_mean"][-1]),
                        "val_mse": out["val_mse"],
                        "any_nan": out["any_nan"],
                        "ever_hit_floor": floored,
                    })
                    print(
                        f"  [{i:>3}/{n_configs}] {cfg_id:<50} "
                        f"chi_p95(0)={out['chi_p95'][0]:8.2g} "
                        f"val_mse={out['val_mse']:.3f} "
                        f"{'NAN' if out['any_nan'] else ('FLOORED' if floored else 'ok')}"
                    )

    # Summary CSV.
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n  Summary written to {summary_path}")
    return results


# ---------------------------------------------------------------------------
#  Plots
# ---------------------------------------------------------------------------


def _heatmap(
    ax,
    values: np.ndarray,
    *,
    xticks: list,
    yticks: list,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    log_norm: bool = False,
):
    import matplotlib.pyplot as plt  # local import — script is plot-free without it
    from matplotlib.colors import LogNorm, Normalize

    if log_norm:
        # Clip non-positive entries; LogNorm rejects them.
        v = np.where(values > 0, values, np.nan)
        norm = LogNorm(vmin=np.nanmin(v), vmax=np.nanmax(v))
    else:
        norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, origin="lower")
    ax.set_xticks(range(len(xticks)), [f"{t:g}" for t in xticks])
    ax.set_yticks(range(len(yticks)), [f"{t:g}" for t in yticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Annotate cells.
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if math.isfinite(v):
                ax.text(j, i, f"{v:.2g}", ha="center", va="center",
                        color="white" if (not log_norm and v > 0.5 * np.nanmax(values))
                        else "black", fontsize=7)
    plt.colorbar(im, ax=ax)


def plot_heatmaps(
    *,
    out_dir: Path,
    results: dict[str, dict],
    batch_sizes: list[int],
    sigma_vs: list[float],
    gain_ws: list[float],
    rules: list[str],
) -> None:
    """val_mse and initial chi_p95 heatmaps, one panel per rule, gain_w
    averaged out (median across gain_ws to avoid being dominated by
    extreme inits).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figures (pip install matplotlib)")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    def grid_for(metric: str) -> np.ndarray:
        # shape: (rules, sigma_vs, batch_sizes)  — median over gain_ws.
        g = np.full((len(rules), len(sigma_vs), len(batch_sizes)), np.nan, dtype=np.float64)
        for r, rule in enumerate(rules):
            for i, sv in enumerate(sigma_vs):
                for j, B in enumerate(batch_sizes):
                    vals = []
                    for gw in gain_ws:
                        out = results[_config_id(rule, B, sv, gw)]
                        v = out[metric][0] if metric.startswith("chi_") else out[metric]
                        # val_mse is scalar; chi_p95[0] is initial.
                        vals.append(float(v))
                    g[r, i, j] = float(np.nanmedian(vals))
        return g

    val_grid = np.full((len(rules), len(sigma_vs), len(batch_sizes)), np.nan)
    chi_grid = np.full((len(rules), len(sigma_vs), len(batch_sizes)), np.nan)
    for r, rule in enumerate(rules):
        for i, sv in enumerate(sigma_vs):
            for j, B in enumerate(batch_sizes):
                vals_val = []
                vals_chi = []
                for gw in gain_ws:
                    out = results[_config_id(rule, B, sv, gw)]
                    vals_val.append(float(out["val_mse"]))
                    vals_chi.append(float(out["chi_p95"][0]))
                val_grid[r, i, j] = float(np.nanmedian(vals_val))
                chi_grid[r, i, j] = float(np.nanmedian(vals_chi))

    # --- val_mse heatmap ---
    fig, axes = plt.subplots(1, len(rules), figsize=(5 * len(rules), 4), squeeze=False)
    for r, rule in enumerate(rules):
        _heatmap(
            axes[0, r],
            val_grid[r],
            xticks=batch_sizes,
            yticks=sigma_vs,
            title=f"val MSE — {rule}",
            xlabel="batch size",
            ylabel="σ_v",
            cmap="viridis",
            log_norm=True,
        )
    fig.suptitle(f"Validation MSE  (median over gain_w ∈ {gain_ws})")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"val_mse.{ext}", dpi=140)
    plt.close(fig)

    # --- initial chi_p95 heatmap ---
    fig, axes = plt.subplots(1, len(rules), figsize=(5 * len(rules), 4), squeeze=False)
    for r, rule in enumerate(rules):
        _heatmap(
            axes[0, r],
            chi_grid[r],
            xticks=batch_sizes,
            yticks=sigma_vs,
            title=f"initial chi_p95 — {rule}",
            xlabel="batch size",
            ylabel="σ_v",
            cmap="magma",
            log_norm=True,
        )
    fig.suptitle(f"Initial p95 raw_chi  (median over gain_w ∈ {gain_ws})")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"chi_p95_initial.{ext}", dpi=140)
    plt.close(fig)

    # --- S_w trace for a single config across rules ---
    # Pick a high-chi config (middle σ_v, biggest batch, gain_w=1.0) so
    # the rule differences are visible.
    sv = sigma_vs[len(sigma_vs) // 2]
    B = batch_sizes[-1]
    gw = gain_ws[len(gain_ws) // 2]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for rule in rules:
        out = results[_config_id(rule, B, sv, gw)]
        ax.plot(out["Sw_mean"], label=f"{rule} (mean)", linewidth=1.5)
        ax.plot(out["Sw_min"], label=f"{rule} (min)", linewidth=0.8, linestyle="--")
    ax.axhline(1e-5, color="grey", linestyle=":", linewidth=0.8, label="additive floor")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("S_w")
    ax.set_title(f"S_w trace  (B={B}, σ_v={sv}, gain_w={gw})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / "Sw_trace.{ext}".format(ext=ext), dpi=140)
    plt.close(fig)

    print(f"  Figures saved to {fig_dir}/")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


_QUICK = dict(
    batch_sizes=[1, 32, 128],
    sigma_vs=[0.05, 1.0],
    gain_ws=[1.0],
    rules=list(UPDATE_RULES),
    n_steps=100,
)

_FULL = dict(
    batch_sizes=[1, 8, 32, 128, 256],
    sigma_vs=[0.01, 0.05, 0.2, 1.0],
    gain_ws=[0.5, 1.0, 2.0],
    rules=list(UPDATE_RULES),
    n_steps=300,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Run a small grid for smoke-testing (~30 s).")
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--in_features", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--sigma_vs", type=float, nargs="+", default=None)
    parser.add_argument("--gain_ws", type=float, nargs="+", default=None)
    parser.add_argument(
        "--rules", type=str, nargs="+", default=None,
        help=f"Subset of {list(UPDATE_RULES)} (default: all).",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: runs/pn_tagi_stage2_<timestamp>/).",
    )
    args = parser.parse_args()

    defaults = _QUICK if args.quick else _FULL
    batch_sizes = args.batch_sizes or defaults["batch_sizes"]
    sigma_vs = args.sigma_vs or defaults["sigma_vs"]
    gain_ws = args.gain_ws or defaults["gain_ws"]
    rules = args.rules or defaults["rules"]
    n_steps = args.n_steps or defaults["n_steps"]

    for r in rules:
        if r not in UPDATE_RULES:
            raise SystemExit(f"unknown rule: {r!r} (choices: {list(UPDATE_RULES)})")

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"pn_tagi_stage2_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  PN-TAGI Stage 2 — tiny linear regression sweep")
    print("=" * 64)
    print(f"  device      : {DEVICE}")
    print(f"  in_features : {args.in_features}")
    print(f"  n_steps     : {n_steps}")
    print(f"  batch_sizes : {batch_sizes}")
    print(f"  sigma_vs    : {sigma_vs}")
    print(f"  gain_ws     : {gain_ws}")
    print(f"  rules       : {rules}")
    print(f"  out_dir     : {out_dir}\n")

    results = run_sweep(
        out_dir=out_dir,
        batch_sizes=batch_sizes,
        sigma_vs=sigma_vs,
        gain_ws=gain_ws,
        rules=rules,
        in_features=args.in_features,
        n_steps=n_steps,
        seed=args.seed,
    )

    plot_heatmaps(
        out_dir=out_dir,
        results=results,
        batch_sizes=batch_sizes,
        sigma_vs=sigma_vs,
        gain_ws=gain_ws,
        rules=rules,
    )
    print(f"\n  Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
