"""Stage 3 dissect v2 — log ζ, χ, κ, Γ on the canonical (PN-fail / CPN /
small-gain-PN / capped) cases at depth=3.

This is the test of the unifying identity

    κ² ≡ (Δμ_θ)² / σ²_θ   =   χ · ζ
    χ  ≡ -Δσ²_θ / σ²_θ        (variance contraction — already exposed)
    ζ  ≡ (y - μ_Y)ᵀ Σ_Y⁻¹ (y - μ_Y) / |B|   (innovation surprise)

derived from the Kalman conditioning Δμ_θ = Σ_{θY}/Σ_Y · (y - μ_Y)
and -Δσ²_θ = Σ_{θY}² / Σ_Y. When the likelihood is calibrated, E[ζ] ≈ 1
and the mean-step and variance-contraction are rigidly coupled. The
hypothesis under test:

    PN failure        = low χ, high κ/Γ, high dead-ReLU, large ζ
    CPN success       = bounded κ/Γ (cap is doing the κ-bound job)
    Small-gain PN     = naturally bounded κ/Γ because σ²_w₀ is small
    Capped baseline   = bounded κ via δ̄ ≈ σ/c_B

Per Linear layer ``j`` we also compute the self-consistency ratio

    Γ_j = ‖ΔμZ_param^{(j)}‖₂ / ‖ΔμZ_RTS^{(j)}‖₂

with

    ΔμZ_param^{(j)} = A^{(j-1)} · Δμ_W^{(j)} + Δμ_B^{(j)}     (forward-prop of param update)
    ΔμZ_RTS^{(j)}   = S_Z^{(j)} ⊙ δμ_z^{(j)}                  (RTS-smoothed posterior shift)

Γ_j ≈ 1 means the parameter update moves z by the same amount the
hidden-state RTS posterior wants z to move — i.e. the update is
self-consistent at the layer level. Γ_j ≫ 1 is the cap-factor's
regime of action.

Output: a per-batch CSV per case plus two diagnostic plots:
    - per-batch traces of ζ, κ_p95, χ_p95, Γ_j, dead-ReLU
    - scatter of (χ, κ²) colored by case — should fall on y = ζ̄ · x line.

Usage::

    python experiments/pn_tagi_stage3/run_mnist_dissect_v2.py
    python experiments/pn_tagi_stage3/run_mnist_dissect_v2.py --n_batches 50
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
from triton_tagi.base import LearnableLayer
from triton_tagi.update.parameters import chi_stats, get_cap_factor
from triton_tagi.update.observation import compute_innovation


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
#  Data + net builder (same as run_mnist_dissect.py)
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


def build_mlp(*, depth, hidden, gain_w, device, update_rule, rho):
    layers = [Linear(784, hidden, device=device, gain_w=gain_w, gain_b=gain_w), ReLU()]
    for _ in range(depth - 1):
        layers.append(Linear(hidden, hidden, device=device, gain_w=gain_w, gain_b=gain_w))
        layers.append(ReLU())
    layers.append(Linear(hidden, 10, device=device, gain_w=gain_w, gain_b=gain_w))
    layers.append(Remax())
    return Sequential(layers, device=device, update_rule=update_rule, rho=rho, record_chi=True)


# ---------------------------------------------------------------------------
#  Manual step that captures all per-layer intermediates we need.
# ---------------------------------------------------------------------------


def dissect_step(net, x_batch, y_oh_batch, y_label_batch, sigma_v):
    """One TAGI step with full per-layer instrumentation.

    Replicates ``Sequential.step`` but does the forward / backward / update
    walk manually so we can record:

        - per-layer (ma_in, ma_out, Sa_in, Sa_out) on the forward
        - δμ_z and δS_z entering each layer's backward
        - per-Linear: κ from (Δμ_w, Sw); Γ from (ΔμZ_param vs ΔμZ_RTS)
        - global: ζ at the output

    Returns a dict with one entry per Linear layer plus the per-batch
    global scalars (ζ, train_acc, output entropy, max_prob).
    """
    # --- 1. Forward, capturing (ma_in, Sa_in, ma_out, Sa_out) per layer ---
    forward = []
    ma = x_batch
    Sa = torch.zeros_like(ma)
    for layer in net.layers:
        ma_in = ma
        Sa_in = Sa
        ma, Sa = layer.forward(ma, Sa)
        forward.append({
            "layer": layer,
            "ma_in": ma_in.detach(),
            "Sa_in": Sa_in.detach(),
            "ma_out": ma.detach(),
            "Sa_out": Sa.detach(),
        })
    mu_Y = ma
    S_Y = Sa  # this is the Remax output variance (epistemic)

    # --- 2. Output innovation (matches what Sequential.step computes) ---
    delta_mu, delta_var = compute_innovation(y_oh_batch, mu_Y, S_Y, sigma_v)

    # ζ at the output:
    #   ζ_naive = Σ_i (y_i - μ_Y_i)² / (S_Y_i + σ_v²)
    #   ζ_cal   = Σ_i (y_i - μ_Y_i)² / (S_Y_i + μ_Y_i(1 - μ_Y_i) + σ_v²)
    # then average over the batch. Both are scalars per batch.
    residual = y_oh_batch - mu_Y
    sv2 = sigma_v ** 2
    denom_naive = (S_Y + sv2).clamp_min(1e-8)
    categorical = mu_Y * (1.0 - mu_Y)
    denom_cal = (S_Y + categorical + sv2).clamp_min(1e-8)
    zeta_naive = float(((residual ** 2) / denom_naive).sum(dim=-1).mean().item())
    zeta_cal = float(((residual ** 2) / denom_cal).sum(dim=-1).mean().item())

    # --- 3. Backward, capturing δμ_z and δS_z entering each layer ---
    backward = [None] * len(net.layers)  # one entry per layer
    cur_dmu = delta_mu
    cur_dvar = delta_var
    for idx in range(len(net.layers) - 1, -1, -1):
        layer = net.layers[idx]
        # cur_dmu / cur_dvar are δ at THIS layer's *output* (i.e. the input
        # to its backward call).
        backward[idx] = {
            "delta_mu_at_output": cur_dmu.detach(),
            "delta_S_at_output": cur_dvar.detach(),
        }
        cur_dmu, cur_dvar = layer.backward(cur_dmu, cur_dvar)

    # --- 4. Per-Linear diagnostics (BEFORE update fires) ---
    per_layer = []
    linear_idx = 0
    for idx, layer in enumerate(net.layers):
        if not isinstance(layer, Linear):
            continue

        Sw = layer.Sw
        dmw = layer.delta_mw  # (in, out)
        # κ_p = |Δμ_w| / sqrt(Sw)   ⇒   κ²_p = Δμ_w² / Sw
        kappa_sq = (dmw ** 2) / Sw.clamp_min(1e-12)
        kappa = kappa_sq.sqrt()
        # χ_p = -ΔS_w / Sw
        if layer.delta_Sw is not None:
            chi = -layer.delta_Sw / Sw.clamp_min(1e-12)
        else:
            chi = torch.zeros_like(Sw)

        # ΔμZ_param: forward-propagate the layer's mean-param delta.
        #   shape (B, out)  =  (B, in) @ (in, out)  +  (out,)
        ma_in_flat = forward[idx]["ma_in"].reshape(-1, layer.in_features)
        dmu_param = ma_in_flat @ dmw
        if layer.has_bias and layer.delta_mb is not None:
            dmu_param = dmu_param + layer.delta_mb

        # ΔμZ_RTS: S_Z ⊙ δμ_z at the layer's *output*. For a Linear,
        # ma_out == mu_z and Sa_out == S_z (no activation applied yet).
        Sz_out = forward[idx]["Sa_out"].reshape(-1, layer.out_features)
        delta_mu_z = backward[idx]["delta_mu_at_output"].reshape(-1, layer.out_features)
        dmu_rts = Sz_out * delta_mu_z

        Gamma = (dmu_param.norm() / dmu_rts.norm().clamp_min(1e-12)).item()
        # Alignment cos(angle): the α_j-without-the-norm.
        align = (
            (dmu_param * dmu_rts).sum()
            / (dmu_param.norm() * dmu_rts.norm()).clamp_min(1e-12)
        ).item()
        # α_j = min(1, ⟨param, RTS⟩ / ‖param‖²) — the per-layer scaling that
        # would make the parameter update match the RTS posterior shift.
        alpha = (
            (dmu_param * dmu_rts).sum() / (dmu_param ** 2).sum().clamp_min(1e-12)
        ).item()
        alpha = min(1.0, alpha)

        per_layer.append({
            "linear_idx": linear_idx,
            "seq_idx": idx,
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "kappa_median": float(kappa.median().item()),
            "kappa_p95": float(kappa.quantile(0.95).item()),
            "kappa_max": float(kappa.max().item()),
            "chi_median": float(chi.median().item()),
            "chi_p95": float(chi.quantile(0.95).item()),
            "chi_max": float(chi.max().item()),
            "kappa_sq_over_chi_median": float(
                (kappa_sq / chi.clamp_min(1e-30)).median().item()
            ),  # should ≈ ζ if locally calibrated
            "Gamma": Gamma,
            "alignment_cos": align,
            "alpha_j": alpha,
            "Sw_mean": float(Sw.mean().item()),
            "Sw_min": float(Sw.min().item()),
            "dmu_param_norm": float(dmu_param.norm().item()),
            "dmu_rts_norm": float(dmu_rts.norm().item()),
            "mw_mean_abs": float(layer.mw.abs().mean().item()),
        })
        linear_idx += 1

    # --- 5. Apply the actual update ---
    cap_factor = get_cap_factor(x_batch.size(0))
    for layer in net.layers:
        if isinstance(layer, LearnableLayer):
            layer.update(cap_factor,
                         update_rule=net.update_rule,
                         rho=net.rho,
                         record_chi=net.record_chi)

    # --- 6. Per-batch global diagnostics ---
    pred = mu_Y.argmax(dim=1)
    train_acc = float((pred == y_label_batch).float().mean().item())
    p = mu_Y.clamp_min(1e-12)
    entropy = float((-p * p.log()).sum(dim=1).mean().item())
    max_prob = float(p.max(dim=1).values.mean().item())

    # Dead-ReLU fraction at each ReLU output (for headline cross-ref).
    relu_dead = []
    for ft in forward:
        if isinstance(ft["layer"], ReLU):
            relu_dead.append(float((ft["ma_out"].abs() < 1e-6).float().mean().item()))

    return {
        "train_acc": train_acc,
        "out_entropy": entropy,
        "out_max_prob": max_prob,
        "zeta_naive": zeta_naive,
        "zeta_cal": zeta_cal,
        "relu_dead": relu_dead,
        "per_layer": per_layer,
    }


# ---------------------------------------------------------------------------
#  Run one case
# ---------------------------------------------------------------------------


def run_one(*, name, rule, depth, hidden, gain_w, sigma_v, batch_size,
            n_batches, seed, x_train, y_train_oh, y_train_labels, device):
    print(f"\n  ▶ {name}  (rule={rule}, gain_w={gain_w})")
    torch.manual_seed(seed)
    net = build_mlp(depth=depth, hidden=hidden, gain_w=gain_w,
                    device=device, update_rule=rule, rho=1.0)

    perm = torch.randperm(x_train.size(0), device=device)
    x_s = x_train[perm]
    y_s = y_train_oh[perm]
    lab_s = y_train_labels[perm]

    rows = []
    for b in range(1, n_batches + 1):
        i = (b - 1) * batch_size
        xb = x_s[i : i + batch_size]
        yb = y_s[i : i + batch_size]
        lb = lab_s[i : i + batch_size]
        if xb.size(0) == 0:
            break
        diag = dissect_step(net, xb, yb, lb, sigma_v)
        rows.append({"batch": b, **diag})

    # Print sparse summary.
    step = max(1, n_batches // 6)
    for r in rows[::step]:
        n_lin = len(r["per_layer"])
        worst_kappa = max((p["kappa_p95"] for p in r["per_layer"]), default=math.nan)
        worst_Gamma = max((p["Gamma"] for p in r["per_layer"]), default=math.nan)
        print(
            f"      b={r['batch']:3d}  acc={r['train_acc']*100:5.2f}%  "
            f"ζ_cal={r['zeta_cal']:7.2f}  κ_p95={worst_kappa:6.3f}  "
            f"Γ={worst_Gamma:7.2f}  entropy={r['out_entropy']:.2f}"
        )
    return rows


# ---------------------------------------------------------------------------
#  CSV writers
# ---------------------------------------------------------------------------


def write_csvs(out_dir: Path, name: str, rows: list[dict]) -> None:
    # Per-batch globals.
    g_path = out_dir / f"{name}__global.csv"
    with g_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "train_acc", "zeta_cal", "zeta_naive",
                    "out_entropy", "out_max_prob", "relu_dead_max"])
        for r in rows:
            relu_dead_max = max(r["relu_dead"], default=math.nan)
            w.writerow([r["batch"], r["train_acc"], r["zeta_cal"], r["zeta_naive"],
                        r["out_entropy"], r["out_max_prob"], relu_dead_max])
    # Per-layer.
    if rows and rows[0]["per_layer"]:
        pl_path = out_dir / f"{name}__per_layer.csv"
        first = rows[0]["per_layer"][0]
        fields = ["batch"] + list(first.keys())
        with pl_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
            for r in rows:
                for pl in r["per_layer"]:
                    w.writerow([r["batch"]] + [pl[k] for k in fields[1:]])


# ---------------------------------------------------------------------------
#  Plots
# ---------------------------------------------------------------------------


def make_plots(out_dir: Path, runs: dict[str, list[dict]], case_order: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figures")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # --- 1. 6-row trace plot, column per case ---
    n_cases = len(case_order)
    fig, axes = plt.subplots(6, n_cases, figsize=(5.0 * n_cases, 16), squeeze=False)

    for c, name in enumerate(case_order):
        rows = runs[name]
        batches = [r["batch"] for r in rows]
        n_lin = len(rows[0]["per_layer"]) if rows and rows[0]["per_layer"] else 0

        # Row 0: train_acc + ζ_cal
        ax = axes[0, c]
        ax.plot(batches, [r["train_acc"] * 100 for r in rows],
                color="tab:blue", marker="o", markersize=2)
        ax.set_ylabel("train acc (%)", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.set_xlabel("batch")
        ax2 = ax.twinx()
        ax2.plot(batches, [r["zeta_cal"] for r in rows],
                 color="tab:red", linewidth=1.4, linestyle="--", label="ζ_cal")
        ax2.plot(batches, [r["zeta_naive"] for r in rows],
                 color="tab:orange", linewidth=1.0, linestyle=":", label="ζ_naive")
        ax2.axhline(1.0, color="grey", linestyle=":", linewidth=0.6)
        ax2.set_ylabel("ζ (innovation surprise)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.set_yscale("log")
        ax2.legend(fontsize=7, loc="upper right")
        ax.set_title(f"{name}: acc + ζ")

        # Row 1: per-layer κ_p95
        ax = axes[1, c]
        for li in range(n_lin):
            ys = [r["per_layer"][li]["kappa_p95"] for r in rows]
            ax.plot(batches, ys, marker=".", markersize=2, label=f"L{li}")
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.6)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_ylabel("per-Linear κ_p95")
        ax.set_xlabel("batch")
        ax.legend(fontsize=6, ncol=2)
        ax.set_title("κ_p95  (step in σ units)")

        # Row 2: per-layer χ_p95
        ax = axes[2, c]
        for li in range(n_lin):
            ys = [r["per_layer"][li]["chi_p95"] for r in rows]
            ax.plot(batches, ys, marker=".", markersize=2, label=f"L{li}")
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.6)
        ax.set_yscale("symlog", linthresh=1e-9)
        ax.set_ylabel("per-Linear χ_p95")
        ax.set_xlabel("batch")
        ax.legend(fontsize=6, ncol=2)
        ax.set_title("χ_p95  (variance contraction)")

        # Row 3: per-layer Γ_j
        ax = axes[3, c]
        for li in range(n_lin):
            ys = [r["per_layer"][li]["Gamma"] for r in rows]
            ax.plot(batches, ys, marker=".", markersize=2, label=f"L{li}")
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.6)
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_ylabel("per-Linear Γ_j")
        ax.set_xlabel("batch")
        ax.legend(fontsize=6, ncol=2)
        ax.set_title("Γ_j = ‖ΔμZ_param‖ / ‖ΔμZ_RTS‖")

        # Row 4: per-layer α_j (the proposed self-consistency scaling)
        ax = axes[4, c]
        for li in range(n_lin):
            ys = [r["per_layer"][li]["alpha_j"] for r in rows]
            ax.plot(batches, ys, marker=".", markersize=2, label=f"L{li}")
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.6)
        ax.set_ylabel("per-Linear α_j (clipped)")
        ax.set_xlabel("batch")
        ax.set_ylim(-0.2, 1.2)
        ax.legend(fontsize=6, ncol=2)
        ax.set_title("α_j  (the proposed self-consistency cap)")

        # Row 5: ReLU dead fraction
        ax = axes[5, c]
        if rows and rows[0]["relu_dead"]:
            n_relu = len(rows[0]["relu_dead"])
            for k in range(n_relu):
                ys = [r["relu_dead"][k] * 100 for r in rows]
                ax.plot(batches, ys, marker=".", markersize=2, label=f"ReLU{k}")
        ax.set_ylabel("dead-ReLU fraction (%)")
        ax.set_xlabel("batch")
        ax.set_ylim(-2, 102)
        ax.legend(fontsize=6, ncol=2)
        ax.set_title("Dead-ReLU fraction (cross-ref to v1)")

    fig.suptitle(
        "Stage 3 dissect v2 — κ² = χ·ζ identity probed across canonical regimes\n"
        "(depth=3, B=512, σ_v=0.05, first 30 batches)"
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"dissect_v2_traces.{ext}", dpi=130)
    plt.close(fig)

    # --- 2. κ² vs χ scatter, points colored by case ---
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {
        "PN_fail":         "tab:orange",
        "capped_baseline": "tab:blue",
        "CPN":             "tab:green",
        "PN_smallgain":    "tab:purple",
    }
    for name in case_order:
        rows = runs[name]
        xs, ys = [], []
        for r in rows:
            for pl in r["per_layer"]:
                # κ² (median across the layer's params), χ (median)
                # We use the median values for stability; max would emphasise tails.
                k2 = pl["kappa_median"] ** 2
                c = max(pl["chi_median"], 1e-30)
                if c <= 0:
                    continue
                xs.append(c)
                ys.append(k2)
        ax.scatter(xs, ys, s=10, alpha=0.6,
                   color=colors.get(name, "black"), label=name)

    # y = ζ̄·x diagonals at ζ ∈ {0.1, 1, 10, 100}.
    if any(any(p["chi_median"] > 0 for p in r["per_layer"]) for n in case_order
           for r in runs[n]):
        from itertools import chain
        all_x = list(chain.from_iterable(
            (max(p["chi_median"], 1e-12) for p in r["per_layer"])
            for n in case_order for r in runs[n]
        ))
        all_y = list(chain.from_iterable(
            (p["kappa_median"] ** 2 for p in r["per_layer"])
            for n in case_order for r in runs[n]
        ))
        if all_x and all_y:
            xmin = max(min(all_x) * 0.5, 1e-12)
            xmax = max(all_x) * 2
            ymin = max(min(all_y or [1e-12]) * 0.5, 1e-12)
            ymax = max(all_y or [1.0]) * 2
            xs_line = np.geomspace(xmin, xmax, 50)
            for zeta_ref, label in [(0.1, "ζ̄ = 0.1"), (1.0, "ζ̄ = 1"),
                                    (10.0, "ζ̄ = 10"), (100.0, "ζ̄ = 100")]:
                ax.plot(xs_line, zeta_ref * xs_line, linestyle="--",
                        linewidth=0.7, alpha=0.5, label=label)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("χ (per-layer median)")
    ax.set_ylabel("κ² (per-layer median)")
    ax.set_title("κ² = χ·ζ — calibrated runs lie on the ζ̄ = 1 diagonal")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"kappa_chi_scatter.{ext}", dpi=130)
    plt.close(fig)

    print(f"  Figures saved to {fig_dir}/")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


# Four canonical cases at depth=3, σ_v=0.05, B=512 — covering the regime where
# the Stage 3 dissect originally found PN's failure.
DEFAULT_CASES = [
    {"name": "PN_fail",         "rule": "precision_normalized",          "gain_w": 1.0},
    {"name": "capped_baseline", "rule": "capped_additive",               "gain_w": 1.0},
    {"name": "CPN",             "rule": "capped_precision_normalized",   "gain_w": 1.0},
    {"name": "PN_smallgain",    "rule": "precision_normalized",          "gain_w": 0.25},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--sigma_v", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_batches", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("runs") / f"pn_tagi_stage3_dissect_v2_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Stage 3 dissect v2 — κ²=χ·ζ + Γ_j diagnostics")
    print("=" * 64)
    print(f"  depth      : {args.depth}")
    print(f"  hidden     : {args.hidden}")
    print(f"  sigma_v    : {args.sigma_v}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  n_batches  : {args.n_batches}")
    print(f"  out_dir    : {out_dir}\n")

    device = torch.device(DEVICE)
    print(f"  Loading MNIST from '{args.data_dir}'...")
    x_train, y_train_oh, y_train_labels, _, _ = load_mnist(args.data_dir, device)

    runs = {}
    for case in DEFAULT_CASES:
        rows = run_one(
            name=case["name"], rule=case["rule"], gain_w=case["gain_w"],
            depth=args.depth, hidden=args.hidden, sigma_v=args.sigma_v,
            batch_size=args.batch_size, n_batches=args.n_batches, seed=args.seed,
            x_train=x_train, y_train_oh=y_train_oh, y_train_labels=y_train_labels,
            device=device,
        )
        runs[case["name"]] = rows
        write_csvs(out_dir, case["name"], rows)

    make_plots(out_dir, runs, [c["name"] for c in DEFAULT_CASES])
    print(f"\n  Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
