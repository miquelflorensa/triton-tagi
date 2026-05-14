"""Stage R — TAGI-V regression depth sweep.

We pivot from MNIST/Remax to 1-D heteroscedastic regression with the
TAGI-V (heteroscedastic V²) output head so that ``σ_v`` is **learned**,
not a fixed hyperparameter. Output layer is ``Linear(h, 2) + EvenSoftplus``:
even units predict the mean, odd units predict V² (the noise variance)
through softplus. The compute_innovation kernel auto-routes to the
heteros branch when the output dim is 2× the target dim — no σ_v
plumbing required.

Architectures (varying depth, fixed width=50):

    1 ⟶ [hidden]·depth ⟶ Linear(h, 2) ⟶ EvenSoftplus

Rules compared (all 4):

    additive
    capped_additive               (cuTAGI baseline)
    precision_normalized          (PN-TAGI)
    capped_precision_normalized   (hybrid found in Stage 3)

This is the regression mirror of Stage 3 (MNIST depths). With V²
learned rather than fixed, plain PN-TAGI should be far less susceptible
to the mean-step overshoot identified at depth ≥ 3 in classification —
the V² prior contributes an automatic likelihood-trust cap that the
fixed σ_v=0.05 setup lacked.

Per-epoch logging: test RMSE, test NLL, per-layer chi p95 / Sw_mean /
mw_mean_abs, mean predicted V² across the test set, NaN/Inf flag.

Usage::

    python experiments/pn_tagi_stageR/run_regression_depths.py
    python experiments/pn_tagi_stageR/run_regression_depths.py --smoke
    python experiments/pn_tagi_stageR/run_regression_depths.py \
        --depths 1 3 5 7 --rules capped_additive precision_normalized \
        --n_epochs 50 --batch_size 64
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

from triton_tagi import EvenSoftplus, Linear, ReLU, Sequential
from triton_tagi.update.parameters import chi_stats


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

DEFAULT_RULES = (
    "capped_additive",
    "precision_normalized",
    "capped_precision_normalized",
    "additive",
)
ALL_RULES = DEFAULT_RULES + ("tempered_precision_normalized",)


# ---------------------------------------------------------------------------
#  Data (1-D heteroscedastic, identical to examples/regression_heteros.py)
# ---------------------------------------------------------------------------


def generate_data(n_train=800, n_test=500, seed=0):
    """y = sin(x) + ε(x) with σ_obs(x) = 0.05 + 0.3·|x|, x ∈ [-4, 4]."""
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
            (x_te - mu_x) / sd_x, (y_te - mu_y) / sd_y,
            (mu_x, sd_x, mu_y, sd_y))


# ---------------------------------------------------------------------------
#  Net builder (TAGI-V heteros head)
# ---------------------------------------------------------------------------


def build_mlp(*, depth, hidden, gain_w, device, update_rule, rho):
    """1 → [hidden]·depth → 2 with EvenSoftplus head (TAGI-V)."""
    layers = [Linear(1, hidden, device=device, gain_w=gain_w, gain_b=gain_w), ReLU()]
    for _ in range(depth - 1):
        layers.append(Linear(hidden, hidden, device=device, gain_w=gain_w, gain_b=gain_w))
        layers.append(ReLU())
    layers.append(Linear(hidden, 2, device=device, gain_w=gain_w, gain_b=gain_w))
    layers.append(EvenSoftplus(half_width=1))
    return Sequential(layers, device=device, update_rule=update_rule, rho=rho, record_chi=True)


# ---------------------------------------------------------------------------
#  Eval (TAGI-V: column 0 = mean, column 1 = predicted V²)
# ---------------------------------------------------------------------------


def evaluate(net, x_te, y_te, batch_size=1024):
    """Return (rmse, mean_nll, mean_predicted_V2) over the test set."""
    net.eval()
    sq_err_sum = 0.0
    nll_sum = 0.0
    v2_sum = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(x_te), batch_size):
            xb = x_te[i : i + batch_size]
            yb = y_te[i : i + batch_size]
            mu, Sa = net.forward(xb)
            mu_pred = mu[:, 0:1]
            var_epi = Sa[:, 0:1]
            v2 = mu[:, 1:2]  # learned aleatoric variance (after EvenSoftplus)
            pred_var = (var_epi + v2).clamp_min(1e-8)
            sq_err_sum += float(((mu_pred - yb) ** 2).sum().item())
            nll_sum += float((0.5 * (mu_pred - yb) ** 2 / pred_var
                              + 0.5 * pred_var.log()
                              + 0.5 * math.log(2 * math.pi)).sum().item())
            v2_sum += float(v2.sum().item())
            n += xb.size(0)
    net.train()
    rmse = math.sqrt(sq_err_sum / max(n, 1))
    return rmse, nll_sum / max(n, 1), v2_sum / max(n, 1)


def any_nan(net):
    for layer in net.layers:
        if isinstance(layer, Linear):
            for t in (layer.mw, layer.Sw, layer.mb, layer.Sb):
                if t is None:
                    continue
                if not torch.isfinite(t).all():
                    return True
    return False


# ---------------------------------------------------------------------------
#  Per-layer diagnostics
# ---------------------------------------------------------------------------


def collect_layer_diagnostics(net):
    rows = []
    linear_idx = 0
    for i, layer in enumerate(net.layers):
        if isinstance(layer, Linear):
            chi_w = getattr(layer, "chi_w", None)
            cs = chi_stats(chi_w) if chi_w is not None else {}
            rows.append({
                "linear_idx": linear_idx,
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "Sw_mean": float(layer.Sw.mean().item()),
                "Sw_min": float(layer.Sw.min().item()),
                "mw_mean_abs": float(layer.mw.abs().mean().item()),
                "chi_p95": cs.get("raw_chi_p95", math.nan),
                "chi_max": cs.get("raw_chi_max", math.nan),
                "chi_median": cs.get("raw_chi_median", math.nan),
                "frac_chi_gt_1": cs.get("frac_chi_gt_1", math.nan),
            })
            linear_idx += 1
    return rows


# ---------------------------------------------------------------------------
#  Train one config
# ---------------------------------------------------------------------------


def train_one_config(
    *, depth, rule, rho, hidden, batch_size, n_epochs, gain_w, seed,
    x_train, y_train, x_test, y_test, device,
):
    torch.manual_seed(seed)
    net = build_mlp(depth=depth, hidden=hidden, gain_w=gain_w,
                    device=device, update_rule=rule, rho=rho)

    epoch_rows = []
    per_layer_rows = []

    rmse, nll, v2 = evaluate(net, x_test, y_test)
    epoch_rows.append({
        "epoch": 0, "train_rmse": math.nan,
        "test_rmse": rmse, "test_nll": nll, "test_v2_mean": v2,
        "wall_s": 0.0, "any_nan": False,
    })

    n_train = x_train.size(0)
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, device=device)
        xs = x_train[perm]
        ys = y_train[perm]

        sq_err_sum = 0.0
        n_seen = 0
        for i in range(0, n_train, batch_size):
            xb = xs[i : i + batch_size]
            yb = ys[i : i + batch_size]
            # sigma_v argument is ignored by the heteros kernel.
            mu_pred, _ = net.step(xb, yb, sigma_v=1.0)
            sq_err_sum += float(((mu_pred[:, 0:1] - yb) ** 2).sum().item())
            n_seen += xb.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        nan = any_nan(net)
        train_rmse = math.sqrt(sq_err_sum / max(n_seen, 1))
        if nan:
            test_rmse, test_nll, test_v2 = math.nan, math.nan, math.nan
        else:
            test_rmse, test_nll, test_v2 = evaluate(net, x_test, y_test)

        epoch_rows.append({
            "epoch": epoch,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "test_nll": test_nll,
            "test_v2_mean": test_v2,
            "wall_s": wall,
            "any_nan": nan,
        })
        for lr in collect_layer_diagnostics(net):
            per_layer_rows.append({"epoch": epoch, **lr})

        if nan:
            print(f"      NaN at epoch {epoch}; aborting this config.")
            break

    return {
        "epoch_rows": epoch_rows,
        "per_layer_rows": per_layer_rows,
        "final_rmse": epoch_rows[-1]["test_rmse"],
        "final_nll": epoch_rows[-1]["test_nll"],
        "ever_nan": any(r["any_nan"] for r in epoch_rows),
    }


# ---------------------------------------------------------------------------
#  Sweep
# ---------------------------------------------------------------------------


def run_sweep(
    *, out_dir, depths, rules, rho, hidden, batch_size, n_epochs,
    gain_w, seed, data,
):
    device = torch.device(DEVICE)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    results = {}
    n_configs = len(depths) * len(rules)
    i = 0
    for rule in rules:
        for depth in depths:
            i += 1
            cfg_id = f"{rule}__d{depth}"
            print(f"\n  [{i}/{n_configs}] {cfg_id}")
            res = train_one_config(
                depth=depth, rule=rule, rho=rho, hidden=hidden,
                batch_size=batch_size, n_epochs=n_epochs, gain_w=gain_w,
                seed=seed, device=device,
                x_train=data["x_train"], y_train=data["y_train"],
                x_test=data["x_test"], y_test=data["y_test"],
            )
            results[cfg_id] = res

            with (traces_dir / f"{cfg_id}__epochs.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(res["epoch_rows"][0].keys()))
                w.writeheader()
                w.writerows(res["epoch_rows"])
            if res["per_layer_rows"]:
                with (traces_dir / f"{cfg_id}__layers.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(res["per_layer_rows"][0].keys()))
                    w.writeheader()
                    w.writerows(res["per_layer_rows"])

            # Brief console summary at sparse epochs.
            step = max(1, n_epochs // 5)
            for r in res["epoch_rows"][::step] + [res["epoch_rows"][-1]]:
                rmse_s = (f"{r['test_rmse']:7.4f}" if isinstance(r['test_rmse'], float)
                          and math.isfinite(r['test_rmse']) else "   NaN ")
                nll_s = (f"{r['test_nll']:+7.3f}" if isinstance(r['test_nll'], float)
                         and math.isfinite(r['test_nll']) else "   NaN ")
                v2_s = (f"{r['test_v2_mean']:.3g}" if isinstance(r['test_v2_mean'], float)
                        and math.isfinite(r['test_v2_mean']) else "  NaN ")
                print(f"      epoch {r['epoch']:3d}  "
                      f"rmse={rmse_s}  nll={nll_s}  V²={v2_s}  wall={r['wall_s']:5.1f}s")

            # Summary row.
            ep1 = [r for r in res["per_layer_rows"] if r["epoch"] == 1]
            early_chi_p95 = (max((r["chi_p95"] for r in ep1 if math.isfinite(r["chi_p95"])),
                                 default=math.nan))
            summary_rows.append({
                "config_id": cfg_id, "rule": rule, "depth": depth,
                "final_rmse": res["final_rmse"], "final_nll": res["final_nll"],
                "ever_nan": res["ever_nan"], "early_chi_p95": early_chi_p95,
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


def make_plots(*, out_dir, results, depths, rules):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figures")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # 1. Test RMSE vs epoch, panel per depth.
    fig, axes = plt.subplots(1, len(depths), figsize=(3.6 * len(depths), 3.5),
                             sharey=True, squeeze=False)
    for j, d in enumerate(depths):
        ax = axes[0, j]
        for rule in rules:
            ep = results[f"{rule}__d{d}"]["epoch_rows"]
            ax.plot([r["epoch"] for r in ep],
                    [r["test_rmse"] if math.isfinite(r["test_rmse"]) else math.nan
                     for r in ep],
                    marker="o", markersize=2, linewidth=1.4, label=rule)
        ax.set_title(f"depth = {d}")
        ax.set_xlabel("epoch")
        if j == 0:
            ax.set_ylabel("test RMSE")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("TAGI-V regression: test RMSE")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"rmse.{ext}", dpi=140)
    plt.close(fig)

    # 2. Test NLL vs epoch.
    fig, axes = plt.subplots(1, len(depths), figsize=(3.6 * len(depths), 3.5),
                             sharey=True, squeeze=False)
    for j, d in enumerate(depths):
        ax = axes[0, j]
        for rule in rules:
            ep = results[f"{rule}__d{d}"]["epoch_rows"]
            ax.plot([r["epoch"] for r in ep],
                    [r["test_nll"] if math.isfinite(r["test_nll"]) else math.nan
                     for r in ep],
                    marker="o", markersize=2, linewidth=1.4, label=rule)
        ax.set_title(f"depth = {d}")
        ax.set_xlabel("epoch")
        if j == 0:
            ax.set_ylabel("test NLL")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("TAGI-V regression: test NLL (lower is better)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"nll.{ext}", dpi=140)
    plt.close(fig)

    # 3. Predicted V² evolution (the headline TAGI-V quantity).
    fig, axes = plt.subplots(1, len(depths), figsize=(3.6 * len(depths), 3.5),
                             sharey=True, squeeze=False)
    for j, d in enumerate(depths):
        ax = axes[0, j]
        for rule in rules:
            ep = results[f"{rule}__d{d}"]["epoch_rows"]
            ax.plot([r["epoch"] for r in ep],
                    [r["test_v2_mean"] if math.isfinite(r["test_v2_mean"]) else math.nan
                     for r in ep],
                    marker="o", markersize=2, linewidth=1.4, label=rule)
        ax.set_title(f"depth = {d}")
        ax.set_xlabel("epoch")
        if j == 0:
            ax.set_ylabel("mean predicted V² (test)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Learned aleatoric variance V² across training")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"v2_trace.{ext}", dpi=140)
    plt.close(fig)

    # 4. Per-layer chi heatmap (one panel per (rule × depth)).
    fig, axes = plt.subplots(len(rules), len(depths),
                             figsize=(3.2 * len(depths), 2.5 * len(rules)),
                             squeeze=False)
    all_chi = [v for r in results.values() for pl in r["per_layer_rows"]
               for v in [pl["chi_p95"]] if math.isfinite(v) and v > 0]
    if all_chi:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=min(all_chi), vmax=max(all_chi))
    else:
        norm = None
    for i, rule in enumerate(rules):
        for j, d in enumerate(depths):
            ax = axes[i, j]
            pl = results[f"{rule}__d{d}"]["per_layer_rows"]
            if not pl:
                ax.set_axis_off()
                continue
            n_lin = max(r["linear_idx"] for r in pl) + 1
            n_ep = max(r["epoch"] for r in pl)
            mat = np.full((n_lin, n_ep), np.nan)
            for r in pl:
                mat[r["linear_idx"], r["epoch"] - 1] = r["chi_p95"]
            im = ax.imshow(mat, aspect="auto", cmap="magma", norm=norm, origin="upper")
            if i == 0:
                ax.set_title(f"depth = {d}")
            if j == 0:
                ax.set_ylabel(f"{rule}\nLinear idx", fontsize=8)
            ax.set_xlabel("epoch" if i == len(rules) - 1 else "")
    fig.suptitle("Per-layer p95 raw_chi across (depth × rule)")
    fig.tight_layout()
    if norm is not None:
        cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
        fig.colorbar(im, cax=cax)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"chi_heatmap.{ext}", dpi=140)
    plt.close(fig)

    print(f"  Figures saved to {fig_dir}/")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


_SMOKE = dict(depths=[1], rules=list(DEFAULT_RULES), n_epochs=3)
_DEFAULT = dict(depths=[1, 3, 5, 7], rules=list(DEFAULT_RULES), n_epochs=50)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny grid (depth=1, 3 epochs) — verify the script runs.")
    parser.add_argument("--depths", type=int, nargs="+", default=None)
    parser.add_argument("--rules", type=str, nargs="+", default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gain_w", type=float, default=1.0)
    parser.add_argument("--n_train", type=int, default=800)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    defaults = _SMOKE if args.smoke else _DEFAULT
    depths = args.depths or defaults["depths"]
    rules = args.rules or defaults["rules"]
    n_epochs = args.n_epochs or defaults["n_epochs"]

    for r in rules:
        if r not in ALL_RULES:
            raise SystemExit(f"unknown rule: {r!r}")

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("runs") / f"pn_tagi_stageR_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  PN-TAGI Stage R — TAGI-V regression depth sweep")
    print("=" * 64)
    print(f"  device      : {DEVICE}")
    print(f"  depths      : {depths}")
    print(f"  rules       : {rules}")
    print(f"  hidden      : {args.hidden}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  n_epochs    : {n_epochs}")
    print(f"  gain_w      : {args.gain_w}")
    print(f"  rho         : {args.rho}")
    print(f"  n_train     : {args.n_train}")
    print(f"  out_dir     : {out_dir}\n")

    device = torch.device(DEVICE)
    x_tr_raw, y_tr_raw, x_te_raw, y_te_raw = generate_data(
        n_train=args.n_train, n_test=args.n_test, seed=args.seed,
    )
    x_tr, y_tr, x_te, y_te, _ = normalise(x_tr_raw, y_tr_raw, x_te_raw, y_te_raw)
    data = dict(
        x_train=torch.from_numpy(x_tr).to(device),
        y_train=torch.from_numpy(y_tr).to(device),
        x_test=torch.from_numpy(x_te).to(device),
        y_test=torch.from_numpy(y_te).to(device),
    )
    print(f"  Train: {x_tr.shape[0]:,}  |  Test: {x_te.shape[0]:,}")

    sweep = run_sweep(
        out_dir=out_dir, depths=depths, rules=rules, rho=args.rho,
        hidden=args.hidden, batch_size=args.batch_size, n_epochs=n_epochs,
        gain_w=args.gain_w, seed=args.seed, data=data,
    )

    make_plots(out_dir=out_dir, results=sweep["results"], depths=depths, rules=rules)

    print("\n  Final summary:")
    for r in sweep["summary_rows"]:
        rmse_s = (f"{r['final_rmse']:7.4f}" if math.isfinite(r['final_rmse']) else "   NaN ")
        nll_s = (f"{r['final_nll']:+7.3f}" if math.isfinite(r['final_nll']) else "   NaN ")
        print(f"    {r['config_id']:<46} rmse={rmse_s}  nll={nll_s}  "
              f"early_chi_p95={r['early_chi_p95']:.2g}  "
              f"{'(NaN)' if r['ever_nan'] else ''}")
    print(f"\n  Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
