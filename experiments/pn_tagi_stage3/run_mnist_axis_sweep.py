"""Stage 3 axis sweeps — locate the boundary of PN-TAGI usability.

The Stage 3 dissection showed that plain PN-TAGI overshoots μ on the
first batch when σ_v is small and S_w is also small, slamming the
network into a dead-ReLU pocket. This script probes the three knobs
that change that early-batch geometry — batch size B, observation
noise σ_v, and weight-init gain — to map where each rule works.

For each axis the others are pinned to their cuTAGI-like defaults
(depth=3, B=512, σ_v=0.05, gain_w=1.0), and each rule is trained for a
small number of epochs. Final test accuracy + an early-batch health
snapshot are reported per (axis_value × rule).

Rules tested (configurable via ``--rules``):

    capped_additive               cuTAGI baseline
    precision_normalized          PN-TAGI (variance only)
    capped_precision_normalized   hybrid (cap + PN variance) — added
                                  specifically to test the Stage 3
                                  failure-mechanism hypothesis

Usage::

    python experiments/pn_tagi_stage3/run_mnist_axis_sweep.py
    python experiments/pn_tagi_stage3/run_mnist_axis_sweep.py --axes batch sigma_v
    python experiments/pn_tagi_stage3/run_mnist_axis_sweep.py --n_epochs 5 --depth 5
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

DEFAULT_RULES = (
    "capped_additive",
    "precision_normalized",
    "capped_precision_normalized",
)

# Pinned defaults for the un-swept axes (matches Stage 3 headline run).
_FIXED = dict(depth=3, batch_size=512, sigma_v=0.05, gain_w=1.0, hidden=256)

_AXES = {
    "batch": dict(name="batch_size", values=[16, 64, 256, 512, 1024], unit="B"),
    "sigma_v": dict(name="sigma_v", values=[0.02, 0.05, 0.1, 0.3, 1.0], unit="σ_v"),
    "gain_w": dict(name="gain_w", values=[0.25, 0.5, 1.0, 2.0, 4.0], unit="gain"),
}


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
#  Net + per-config trainer
# ---------------------------------------------------------------------------


def build_mlp(depth, hidden, gain_w, device, update_rule, rho):
    layers = [Linear(784, hidden, device=device, gain_w=gain_w, gain_b=gain_w), ReLU()]
    for _ in range(depth - 1):
        layers.append(Linear(hidden, hidden, device=device, gain_w=gain_w, gain_b=gain_w))
        layers.append(ReLU())
    layers.append(Linear(hidden, 10, device=device, gain_w=gain_w, gain_b=gain_w))
    layers.append(Remax())
    return Sequential(layers, device=device, update_rule=update_rule, rho=rho, record_chi=True)


def relu_dead_fractions(net: Sequential, x_batch: torch.Tensor) -> list[float]:
    """Walk a forward pass and read fraction of below-1e-6 activations
    immediately after each ReLU. Returned in network order."""
    fracs: list[float] = []
    ma = x_batch
    Sa = torch.zeros_like(ma)
    for layer in net.layers:
        ma, Sa = layer.forward(ma, Sa)
        if isinstance(layer, ReLU):
            fracs.append(float((ma.abs() < 1e-6).float().mean().item()))
    return fracs


def evaluate(net: Sequential, x, y_labels, batch_size=1024) -> float:
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            mu, _ = net.forward(x[i : i + batch_size])
            correct += int((mu.argmax(dim=1) == y_labels[i : i + batch_size]).sum().item())
    net.train()
    return correct / len(x)


def any_nan(net: Sequential) -> bool:
    for layer in net.layers:
        if isinstance(layer, Linear):
            for t in (layer.mw, layer.Sw, layer.mb, layer.Sb):
                if t is None:
                    continue
                if not torch.isfinite(t).all():
                    return True
    return False


def train_one(
    *,
    rule: str,
    rho: float,
    depth: int,
    hidden: int,
    batch_size: int,
    sigma_v: float,
    gain_w: float,
    n_epochs: int,
    seed: int,
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device(DEVICE)
    net = build_mlp(depth=depth, hidden=hidden, gain_w=gain_w,
                    device=device, update_rule=rule, rho=rho)

    n_train = x_train.size(0)
    epoch_accs: list[float] = []
    dead_after_first_epoch: list[float] = []

    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n_train, device=device)
        xs = x_train[perm]
        ys = y_train_oh[perm]
        for i in range(0, n_train, batch_size):
            net.step(xs[i : i + batch_size], ys[i : i + batch_size], sigma_v)
        if any_nan(net):
            break
        acc = evaluate(net, x_test, y_test_labels)
        epoch_accs.append(acc)
        if epoch == 1:
            dead_after_first_epoch = relu_dead_fractions(net, xs[:batch_size])
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    return {
        "rule": rule,
        "final_acc": (epoch_accs[-1] if epoch_accs else math.nan),
        "epoch1_acc": (epoch_accs[0] if epoch_accs else math.nan),
        "all_epoch_accs": epoch_accs,
        "dead_relu_frac_max_e1": (max(dead_after_first_epoch) if dead_after_first_epoch else math.nan),
        "dead_relu_frac_mean_e1": (float(np.mean(dead_after_first_epoch))
                                   if dead_after_first_epoch else math.nan),
        "ever_nan": any_nan(net),
        "wall_s": wall,
    }


# ---------------------------------------------------------------------------
#  Sweep
# ---------------------------------------------------------------------------


def run_axis_sweep(
    *,
    axis: str,
    rules: list[str],
    rho: float,
    fixed: dict,
    n_epochs: int,
    seed: int,
    out_dir: Path,
    data: dict,
) -> list[dict]:
    spec = _AXES[axis]
    axis_arg = spec["name"]
    values = spec["values"]

    print(f"\n  ── axis: {axis} ({axis_arg}) over {values} ──")
    rows: list[dict] = []
    for v in values:
        for rule in rules:
            args = dict(fixed)
            args[axis_arg] = v
            res = train_one(
                rule=rule, rho=rho, n_epochs=n_epochs, seed=seed,
                **args,
                x_train=data["x_train"], y_train_oh=data["y_train_oh"],
                y_train_labels=data["y_train_labels"],
                x_test=data["x_test"], y_test_labels=data["y_test_labels"],
            )
            row = {
                axis_arg: v,
                "rule": rule,
                "final_acc": res["final_acc"],
                "epoch1_acc": res["epoch1_acc"],
                "dead_relu_frac_max_e1": res["dead_relu_frac_max_e1"],
                "dead_relu_frac_mean_e1": res["dead_relu_frac_mean_e1"],
                "ever_nan": res["ever_nan"],
                "wall_s": res["wall_s"],
                "all_epoch_accs": ",".join(f"{a:.4f}" for a in res["all_epoch_accs"]),
            }
            rows.append(row)
            acc_str = (f"{res['final_acc']*100:5.2f}%" if math.isfinite(res["final_acc"])
                       else "  NaN ")
            dead_str = (f"{res['dead_relu_frac_max_e1']*100:4.0f}%"
                        if math.isfinite(res["dead_relu_frac_max_e1"]) else "  - ")
            print(f"    {axis_arg}={v:<8}  rule={rule:<32}  "
                  f"final={acc_str}  dead_e1_max={dead_str}  wall={res['wall_s']:5.1f}s")

    # Per-axis CSV.
    out_csv = out_dir / f"sweep_{axis}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"    CSV: {out_csv}")
    return rows


# ---------------------------------------------------------------------------
#  Plots
# ---------------------------------------------------------------------------


def plot_axis(
    out_dir: Path,
    axis: str,
    rows: list[dict],
    rules: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figures")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    spec = _AXES[axis]
    axis_arg = spec["name"]
    values = spec["values"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 0: final test accuracy vs axis.
    ax = axes[0]
    for rule in rules:
        ys = []
        for v in values:
            match = next((r for r in rows if r[axis_arg] == v and r["rule"] == rule), None)
            ys.append(match["final_acc"] * 100 if match and math.isfinite(match["final_acc"])
                      else math.nan)
        ax.plot(values, ys, marker="o", markersize=4, label=rule, linewidth=1.5)
    ax.set_xlabel(axis_arg)
    ax.set_ylabel("final test acc (%)")
    if axis in ("batch", "sigma_v", "gain_w"):
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Test accuracy vs {axis_arg}")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)

    # Panel 1: max-ReLU-dead fraction at end of epoch 1.
    ax = axes[1]
    for rule in rules:
        ys = []
        for v in values:
            match = next((r for r in rows if r[axis_arg] == v and r["rule"] == rule), None)
            ys.append(match["dead_relu_frac_max_e1"] * 100
                      if match and math.isfinite(match["dead_relu_frac_max_e1"]) else math.nan)
        ax.plot(values, ys, marker="o", markersize=4, label=rule, linewidth=1.5)
    ax.set_xlabel(axis_arg)
    ax.set_ylabel("worst ReLU dead-fraction at end of epoch 1 (%)")
    if axis in ("batch", "sigma_v", "gain_w"):
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Dead-ReLU fraction vs {axis_arg}")
    ax.set_ylim(0, 102)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Stage 3 axis sweep — {axis_arg}  "
        f"(depth={_FIXED['depth']}, others pinned)"
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"sweep_{axis}.{ext}", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axes", type=str, nargs="+",
                        default=["batch", "sigma_v", "gain_w"],
                        choices=list(_AXES.keys()))
    parser.add_argument("--rules", type=str, nargs="+", default=list(DEFAULT_RULES))
    parser.add_argument("--depth", type=int, default=_FIXED["depth"])
    parser.add_argument("--hidden", type=int, default=_FIXED["hidden"])
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("runs") / f"pn_tagi_stage3_axis_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  PN-TAGI Stage 3 — axis sweeps")
    print("=" * 64)
    print(f"  depth      : {args.depth}")
    print(f"  hidden     : {args.hidden}")
    print(f"  n_epochs   : {args.n_epochs}")
    print(f"  rho        : {args.rho}")
    print(f"  axes       : {args.axes}")
    print(f"  rules      : {args.rules}")
    print(f"  fixed defaults: {_FIXED}")
    print(f"  out_dir    : {out_dir}\n")

    device = torch.device(DEVICE)
    print(f"  Loading MNIST from '{args.data_dir}'...")
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels = load_mnist(args.data_dir, device)
    data = dict(
        x_train=x_train, y_train_oh=y_train_oh, y_train_labels=y_train_labels,
        x_test=x_test, y_test_labels=y_test_labels,
    )

    fixed = dict(_FIXED, depth=args.depth, hidden=args.hidden)

    all_rows: dict[str, list[dict]] = {}
    for axis in args.axes:
        rows = run_axis_sweep(
            axis=axis, rules=args.rules, rho=args.rho, fixed=fixed,
            n_epochs=args.n_epochs, seed=args.seed, out_dir=out_dir, data=data,
        )
        all_rows[axis] = rows
        plot_axis(out_dir, axis, rows, args.rules)

    print(f"\n  Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
