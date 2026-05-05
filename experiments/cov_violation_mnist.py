"""
Cauchy-Schwarz bound violation diagnostic for MM-Remax during training.

Reviewer zv4Q (UAI 2026 #416), Q3:
    "How often is the bound in the covariance violated (from Section 4.2)?"

The bound (TRTCT.tex:197) is

        |Cov(Z_i, A_i)|  <=  sigma_{Z_i} * sigma_{A_i}.                  (CS)

The Triton kernel enforces it by clipping (triton_tagi/layers/remax.py:107-108):

        cs_bound = sqrt(var_a * var_z)
        cov_a_z  = min(cs_bound, cov_a_m / max(jcb_m, EPS))

This script trains a 784-256-128-10 MNIST FNN with MM-Remax (cuTAGI parity:
MixtureReLU + log-normal path) and, on every forward, computes the *pre-clip*
analytical |cov_a_m / Phi(alpha)| and the bound sqrt(var_a * var_z), then
records the ratio rho = pre_clip / bound. rho > 1 => the kernel's min(.,.)
fired and clipped (i.e. the analytical formula violated CS).

Per epoch: violation rate, mean/median/max rho, histogram. Outputs go to
``runs/cov_violation/`` as a single npz plus two PNG figures.

Run:
    python experiments/cov_violation_mnist.py --epochs 10 --seed 0
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torchvision import datasets

from triton_tagi.layers.linear import Linear as TLinear
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.layers.remax import Remax as TRemax, triton_remax
from triton_tagi.layers.remax_chain import RemaxChain, triton_remax_chain
from triton_tagi.network import Sequential as TSequential


IN_F, H1, H2, OUT_F = 784, 256, 128, 10
BATCH = 512
SIGMA_V = 0.05
DEVICE = "cuda"
DATA_ROOT = "data"
EPS = 1e-6

# Histogram support for rho = pre_clip / cs_bound. The CS bound at rho = 1
# sits on a bin edge (log10(1) = 0). Bins are log10-spaced from 1e-2 (rho=0.01)
# to 1e4 (rho=10000), plus underflow / overflow bins. Inactive units push rho
# into the 1e2-1e4 regime; healthy units cluster around rho < 1.
LOG10_EDGES = np.linspace(-2.0, 4.0, 49)         # 48 bins, width 0.125 in log10
HIST_OVERFLOW_INDEX = len(LOG10_EDGES) - 1       # last bin holds rho >= 1e4
HIST_UNDERFLOW_INDEX = 0                          # first bin holds rho < 1e-2

# Quantiles to track per epoch
QUANTILES = (0.5, 0.75, 0.9, 0.99, 0.999, 0.9999)


# ----------------------------------------------------------------------
#  Pure-Torch reproduction of the Remax forward, exposing pre-clip Cov(Z,A)
# ----------------------------------------------------------------------


def remax_diagnostic(mu_z: Tensor, var_z: Tensor) -> dict:
    """Mirror of ``_remax_kernel`` (triton_tagi/layers/remax.py) in PyTorch.

    Returns ``cov_a_z_pre_clip`` (the analytical value before the
    Cauchy-Schwarz min(.,.) clip) plus the CS bound and rho = pre/bound.
    Strictly read-only: no effect on training; diagnostics only.
    """
    INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

    # MixtureReLU moments of M = max(0, Z)
    var_z_safe = torch.clamp(var_z, min=EPS)
    std_z = torch.sqrt(var_z_safe)
    alpha = mu_z / std_z
    pdf_alpha = INV_SQRT_2PI * torch.exp(-0.5 * alpha * alpha)
    cdf_alpha = 0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))

    mu_m = mu_z * cdf_alpha + std_z * pdf_alpha
    mu_m = torch.clamp(mu_m, min=EPS)
    var_m = (
        -mu_m * mu_m
        + 2.0 * mu_m * mu_z
        - mu_z * std_z * pdf_alpha
        + (var_z - mu_z * mu_z) * cdf_alpha
    )
    var_m = torch.clamp(var_m, min=EPS)
    jcb_m = cdf_alpha  # = Phi(alpha)

    # Sum moments M_tilde = sum_k M_k
    mu_mt = torch.clamp(mu_m.sum(dim=-1, keepdim=True), min=EPS)
    var_mt = torch.clamp(var_m.sum(dim=-1, keepdim=True), min=EPS)

    # Log-normal moments of M and M_tilde
    var_log_m = torch.log(1.0 + var_m / (mu_m * mu_m))
    mu_log_m = torch.log(mu_m) - 0.5 * var_log_m
    var_log_mt = torch.log(1.0 + var_mt / (mu_mt * mu_mt))
    mu_log_mt = torch.log(mu_mt) - 0.5 * var_log_mt

    cov_log_m_mt = torch.log(1.0 + var_m / (mu_m * mu_mt))

    mu_log_a = mu_log_m - mu_log_mt
    var_log_a = torch.clamp(var_log_m + var_log_mt - 2.0 * cov_log_m_mt, min=0.0)

    mu_a_raw = torch.clamp(torch.exp(mu_log_a + 0.5 * var_log_a), min=EPS)
    sum_mu_a = torch.clamp(mu_a_raw.sum(dim=-1, keepdim=True), min=EPS)
    mu_a = mu_a_raw / sum_mu_a
    var_a = (torch.exp(var_log_a) - 1.0) * mu_a * mu_a
    var_a = torch.clamp(var_a, min=0.0)

    # Cov(A, M) via log-normal identity (shared across both formulas)
    cov_log_a_log_m = var_log_m - cov_log_m_mt
    cov_a_m = (torch.exp(cov_log_a_log_m) - 1.0) * mu_a * mu_m

    cs_bound = torch.sqrt(torch.clamp(var_a * var_z, min=0.0))
    bound_safe = torch.clamp(cs_bound, min=EPS)

    # Formula A (cuTAGI-parity / triton-tagi production): cov_a_m / Phi(alpha)
    pre_clip_orig = cov_a_m / torch.clamp(jcb_m, min=EPS)
    rho_orig = pre_clip_orig.abs() / bound_safe

    # Formula B (corrected chain-rule): cov_a_m * Cov(Z, M) / Var(M)
    cov_z_m = var_z * jcb_m
    pre_clip_chain = cov_a_m * cov_z_m / var_m
    rho_chain = pre_clip_chain.abs() / bound_safe

    return {
        "rho_orig": rho_orig,
        "rho_chain": rho_chain,
        "violated_orig": rho_orig > 1.0,
        "violated_chain": rho_chain > 1.0,
        "cs_bound": cs_bound,
        "alpha": alpha,
        "abs_pre_clip_orig": pre_clip_orig.abs(),
        "abs_pre_clip_chain": pre_clip_chain.abs(),
    }


class RemaxDiag(TRemax):
    """``TRemax``-shaped wrapper that runs *either* the original or the
    corrected kernel for training and accumulates CS-bound diagnostics for
    *both* formulas on every forward.

    The training kernel choice is set via ``which_kernel``; the diagnostic
    always tracks both ``rho_orig`` (= cov_a_m / Phi(alpha)) and
    ``rho_chain`` (= cov_a_m * Cov(Z, M) / Var(M)).
    """

    def __init__(self, which_kernel: str = "original") -> None:
        super().__init__()
        if which_kernel not in ("original", "chain"):
            raise ValueError(f"unknown kernel: {which_kernel}")
        self.which_kernel = which_kernel
        self.reset_stats()

    def _zero_stats(self) -> dict:
        return {
            "n_total": 0,
            "n_violated": 0,
            # Active/inactive partition on alpha = mu_z / sigma_z.
            # Active = alpha >= 0 (MixtureReLU mean linear-on); inactive = alpha < 0.
            "n_active": 0,
            "n_inactive": 0,
            "n_violated_active": 0,
            "n_violated_inactive": 0,
            # |pre-clip Cov(Z, A)| and CS bound, summed over clipped (rho > 1)
            # vs unclipped (rho <= 1) entries.
            "sum_abs_pre_clip_clipped": 0.0,
            "sum_abs_pre_clip_unclipped": 0.0,
            "sum_cs_bound_clipped": 0.0,
            "sum_cs_bound_unclipped": 0.0,
            "sum_log_rho": 0.0,
            "sum_log_rho_sq": 0.0,
            "max_rho": 0.0,
            "hist": np.zeros(len(LOG10_EDGES) - 1, dtype=np.int64),
            "rho_samples": [],
        }

    def reset_stats(self) -> None:
        self._stats = {"orig": self._zero_stats(), "chain": self._zero_stats()}
        self._sample_quota_per_batch = 8192

    def forward(self, mz: Tensor, Sz: Tensor) -> tuple[Tensor, Tensor]:
        if self.which_kernel == "original":
            mu_a, Sa, J = triton_remax(mz, Sz)
        else:
            mu_a, Sa, J = triton_remax_chain(mz, Sz)
        self.J = J

        with torch.no_grad():
            d = remax_diagnostic(mz.detach(), Sz.detach())
            alpha = d["alpha"].float()
            active_mask = alpha >= 0.0
            inactive_mask = ~active_mask
            cs_bound = d["cs_bound"].float()
            for tag, rho_t, viol_t, abs_pc_t in (
                ("orig",  d["rho_orig"],  d["violated_orig"],  d["abs_pre_clip_orig"]),
                ("chain", d["rho_chain"], d["violated_chain"], d["abs_pre_clip_chain"]),
            ):
                s = self._stats[tag]
                rho = rho_t.float()
                abs_pc = abs_pc_t.float()
                s["n_total"] += rho.numel()
                s["n_violated"] += int(viol_t.sum().item())
                s["n_active"] += int(active_mask.sum().item())
                s["n_inactive"] += int(inactive_mask.sum().item())
                s["n_violated_active"] += int((viol_t & active_mask).sum().item())
                s["n_violated_inactive"] += int((viol_t & inactive_mask).sum().item())
                unclipped_mask = ~viol_t
                s["sum_abs_pre_clip_clipped"] += float(abs_pc[viol_t].sum().item())
                s["sum_abs_pre_clip_unclipped"] += float(abs_pc[unclipped_mask].sum().item())
                s["sum_cs_bound_clipped"] += float(cs_bound[viol_t].sum().item())
                s["sum_cs_bound_unclipped"] += float(cs_bound[unclipped_mask].sum().item())
                s["max_rho"] = max(s["max_rho"], float(rho.max().item()))
                log_rho = torch.log10(torch.clamp(rho, min=1e-12))
                s["sum_log_rho"] += float(log_rho.sum().item())
                s["sum_log_rho_sq"] += float((log_rho * log_rho).sum().item())
                rho_np = rho.flatten().cpu().numpy()
                log_np = np.log10(np.clip(rho_np, 1e-12, None))
                counts, _ = np.histogram(log_np, bins=LOG10_EDGES)
                s["hist"] += counts
                if rho_np.size > self._sample_quota_per_batch:
                    idx = np.random.choice(rho_np.size, self._sample_quota_per_batch, replace=False)
                    s["rho_samples"].append(rho_np[idx])
                else:
                    s["rho_samples"].append(rho_np)

        return mu_a, Sa

    def _snapshot_one(self, tag: str) -> dict:
        s = self._stats[tag]
        n = max(s["n_total"], 1)
        mean_log = s["sum_log_rho"] / n
        var_log = max(s["sum_log_rho_sq"] / n - mean_log * mean_log, 0.0)
        samples = (
            np.concatenate(s["rho_samples"])
            if s["rho_samples"] else np.zeros(0, dtype=np.float32)
        )
        quantiles = (
            np.quantile(samples, list(QUANTILES)).tolist()
            if samples.size else [0.0] * len(QUANTILES)
        )
        n_unclipped = max(s["n_total"] - s["n_violated"], 0)
        n_clipped = s["n_violated"]
        return {
            "n_total": s["n_total"],
            "n_violated": s["n_violated"],
            "violation_rate": s["n_violated"] / n,
            "geomean_rho": 10.0 ** mean_log,
            "geostd_log10_rho": math.sqrt(var_log),
            "max_rho": s["max_rho"],
            "quantiles": quantiles,
            "quantile_levels": list(QUANTILES),
            "hist": s["hist"].copy(),
            "rho_samples": samples,
            # Active/inactive partition on alpha = mu_z / sigma_z.
            "n_active": s["n_active"],
            "n_inactive": s["n_inactive"],
            "active_fraction": s["n_active"] / n,
            "inactive_fraction": s["n_inactive"] / n,
            "violation_rate_active": s["n_violated_active"] / max(s["n_active"], 1),
            "violation_rate_inactive": s["n_violated_inactive"] / max(s["n_inactive"], 1),
            "frac_violations_inactive": (
                s["n_violated_inactive"] / max(s["n_violated"], 1)
            ),
            # Mean |pre-clip Cov(Z, A)| and CS bound on each subgroup.
            "mean_abs_pre_clip_clipped": (
                s["sum_abs_pre_clip_clipped"] / max(n_clipped, 1)
            ),
            "mean_abs_pre_clip_unclipped": (
                s["sum_abs_pre_clip_unclipped"] / max(n_unclipped, 1)
            ),
            "mean_cs_bound_clipped": (
                s["sum_cs_bound_clipped"] / max(n_clipped, 1)
            ),
            "mean_cs_bound_unclipped": (
                s["sum_cs_bound_unclipped"] / max(n_unclipped, 1)
            ),
        }

    def snapshot(self) -> dict:
        return {"orig": self._snapshot_one("orig"), "chain": self._snapshot_one("chain")}


# ----------------------------------------------------------------------
#  MNIST + training boilerplate (mirrors test_mnist_remax_10epochs.py)
# ----------------------------------------------------------------------


def load_mnist():
    train_ds = datasets.MNIST(DATA_ROOT, train=True, download=True)
    test_ds = datasets.MNIST(DATA_ROOT, train=False, download=True)
    x_train = train_ds.data.float().view(-1, IN_F) / 255.0
    x_test = test_ds.data.float().view(-1, IN_F) / 255.0
    mu, sigma = x_train.mean(), x_train.std()
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    y_labels_train = train_ds.targets
    y_labels_test = test_ds.targets
    y_train_oh = torch.zeros(len(y_labels_train), OUT_F)
    y_train_oh.scatter_(1, y_labels_train.unsqueeze(1), 1.0)
    return x_train, y_train_oh, y_labels_train, x_test, y_labels_test


def he_init(fan_in: int, fan_out: int):
    scale = math.sqrt(1.0 / fan_in)
    mw = torch.randn(fan_in, fan_out) * scale
    Sw = torch.full((fan_in, fan_out), scale ** 2)
    mb = torch.randn(1, fan_out) * scale
    Sb = torch.full((1, fan_out), scale ** 2)
    return mw, Sw, mb, Sb


def build_net(seed: int, which_kernel: str) -> tuple[TSequential, RemaxDiag]:
    torch.manual_seed(seed)
    params = [he_init(IN_F, H1), he_init(H1, H2), he_init(H2, OUT_F)]
    layers = []
    for (mw, Sw, mb, Sb), (fi, fo) in zip(
        params, [(IN_F, H1), (H1, H2), (H2, OUT_F)]
    ):
        l = TLinear(fi, fo, device=DEVICE)
        l.mw = mw.to(DEVICE)
        l.Sw = Sw.to(DEVICE)
        l.mb = mb.to(DEVICE)
        l.Sb = Sb.to(DEVICE)
        layers.append(l)
    diag = RemaxDiag(which_kernel=which_kernel)
    net = TSequential(
        [layers[0], TReLU(), layers[1], TReLU(), layers[2], diag],
        device=DEVICE,
    )
    return net, diag


def evaluate(net, x_test, y_labels):
    net.eval()
    correct = 0
    x = x_test.to(DEVICE)
    for i in range(0, len(x), BATCH):
        mu, _ = net.forward(x[i : i + BATCH])
        correct += (mu.argmax(dim=1).cpu() == y_labels[i : i + BATCH]).sum().item()
    net.train()
    return correct / len(y_labels)


# ----------------------------------------------------------------------
#  Driver
# ----------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kernel", choices=["original", "chain"], default="original",
                    help="Which Remax kernel to use for training. The diagnostic"
                         " always tracks both formulas regardless.")
    p.add_argument("--output-dir", type=str, default="runs/cov_violation")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_train, y_train_oh, _, x_test, y_test_labels = load_mnist()
    net, diag = build_net(args.seed, args.kernel)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_epoch_snapshots = []
    test_accs = []

    # Initial-state diagnostic (epoch 0): one forward over the train set, no update.
    diag.reset_stats()
    net.eval()
    x_d = x_train.to(DEVICE)
    for i in range(0, len(x_d), BATCH):
        net.forward(x_d[i : i + BATCH])
    def fmt_line(epoch_label, snap):
        o, c = snap["orig"], snap["chain"]
        return (
            f"  {epoch_label}  "
            f"orig: vr={o['violation_rate']:.4f} "
            f"vr_act={o['violation_rate_active']:.4f} "
            f"vr_inact={o['violation_rate_inactive']:.4f} "
            f"|pc|_clip={o['mean_abs_pre_clip_clipped']:.3e} "
            f"|pc|_unclip={o['mean_abs_pre_clip_unclipped']:.3e} "
            f"max={o['max_rho']:.2e}  |  "
            f"chain: vr={c['violation_rate']:.4f} "
            f"vr_act={c['violation_rate_active']:.4f} "
            f"vr_inact={c['violation_rate_inactive']:.4f} "
            f"|pc|_clip={c['mean_abs_pre_clip_clipped']:.3e} "
            f"|pc|_unclip={c['mean_abs_pre_clip_unclipped']:.3e} "
            f"max={c['max_rho']:.2e}  |  "
            f"acc={snap['test_acc'] * 100:.2f}%"
        )

    snap0 = diag.snapshot()
    snap0["epoch"] = 0
    snap0["test_acc"] = evaluate(net, x_test, y_test_labels)
    per_epoch_snapshots.append(snap0)
    print(fmt_line("epoch=0  init       ", snap0), flush=True)

    for epoch in range(1, args.epochs + 1):
        net.train()
        diag.reset_stats()
        perm = torch.randperm(len(x_train))
        x_s = x_train[perm].to(DEVICE)
        y_s = y_train_oh[perm].to(DEVICE)
        for i in range(0, len(x_s), BATCH):
            net.step(x_s[i : i + BATCH], y_s[i : i + BATCH], SIGMA_V)
        snap = diag.snapshot()
        snap["epoch"] = epoch
        snap["test_acc"] = evaluate(net, x_test, y_test_labels)
        per_epoch_snapshots.append(snap)
        test_accs.append(snap["test_acc"])
        print(fmt_line(f"epoch={epoch:>2d} trained    ", snap), flush=True)

    # ---- Save raw arrays ----
    npz_path = out / f"cov_violation_seed{args.seed}.npz"
    save_arrays = {
        "epochs": np.array([s["epoch"] for s in per_epoch_snapshots]),
        "test_acc": np.array([s["test_acc"] for s in per_epoch_snapshots]),
        "log10_edges": LOG10_EDGES,
        "quantile_levels": np.array(QUANTILES),
        "trained_with_kernel": np.array(args.kernel),
    }
    for tag in ("orig", "chain"):
        save_arrays[f"{tag}_violation_rate"] = np.array(
            [s[tag]["violation_rate"] for s in per_epoch_snapshots]
        )
        save_arrays[f"{tag}_geomean_rho"] = np.array(
            [s[tag]["geomean_rho"] for s in per_epoch_snapshots]
        )
        save_arrays[f"{tag}_max_rho"] = np.array(
            [s[tag]["max_rho"] for s in per_epoch_snapshots]
        )
        save_arrays[f"{tag}_quantiles"] = np.array(
            [s[tag]["quantiles"] for s in per_epoch_snapshots]
        )
        save_arrays[f"{tag}_hist"] = np.stack(
            [s[tag]["hist"] for s in per_epoch_snapshots], axis=0
        )
        for key in (
            "active_fraction",
            "inactive_fraction",
            "violation_rate_active",
            "violation_rate_inactive",
            "frac_violations_inactive",
            "mean_abs_pre_clip_clipped",
            "mean_abs_pre_clip_unclipped",
            "mean_cs_bound_clipped",
            "mean_cs_bound_unclipped",
        ):
            save_arrays[f"{tag}_{key}"] = np.array(
                [s[tag][key] for s in per_epoch_snapshots]
            )
    np.savez(npz_path, **save_arrays)
    pick = [0, len(per_epoch_snapshots) // 2, len(per_epoch_snapshots) - 1]
    for k in pick:
        for tag in ("orig", "chain"):
            np.save(
                out / f"rho_{tag}_seed{args.seed}_epoch{per_epoch_snapshots[k]['epoch']}.npy",
                per_epoch_snapshots[k][tag]["rho_samples"].astype(np.float32),
            )
    print(f"\nSaved diagnostics to {npz_path}")

    # ---- Plot 1: violation rate vs epoch (both formulas) ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ep = np.array([s["epoch"] for s in per_epoch_snapshots])
    for tag, label, color in (
        ("orig",  r"original ($\mathrm{cov}_{AM}/\Phi$)",     "C3"),
        ("chain", r"corrected ($\mathrm{cov}_{AM}\!\cdot\!\mathrm{cov}_{ZM}/\mathrm{Var}\,M$)", "C0"),
    ):
        vr = np.array([s[tag]["violation_rate"] for s in per_epoch_snapshots]) * 100.0
        ax.plot(ep, vr, "o-", lw=2, label=label, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CS-bound violation rate (%)")
    ax.set_title(
        f"CS-bound violations, MNIST FNN ($\\sigma_v=0.05$, seed={args.seed}, "
        f"trained with: {args.kernel})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"violation_rate_seed{args.seed}.png", dpi=150)
    plt.close(fig)

    # ---- Plot 2: log-scale histograms — final epoch, both formulas ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    centers = 0.5 * (LOG10_EDGES[:-1] + LOG10_EDGES[1:])
    final = per_epoch_snapshots[-1]
    for tag, label, color in (
        ("orig",  "original  ($\\mathrm{cov}_{AM}/\\Phi$)", "C3"),
        ("chain", "corrected ($\\mathrm{cov}_{AM}\\cdot\\mathrm{cov}_{ZM}/\\mathrm{Var}\\,M$)", "C0"),
    ):
        h = final[tag]["hist"].astype(np.float64)
        h = h / max(h.sum(), 1.0)
        ax.plot(10.0 ** centers, h, label=label, lw=1.8, color=color)
    ax.axvline(1.0, color="black", lw=1.0, linestyle="--",
               label="CS bound ($\\rho=1$)")
    ax.set_xscale("log")
    ax.set_xlabel(
        r"$\rho = |\mathrm{Cov}(Z_i, A_i)|_{\mathrm{pre\text{-}clip}} \;/\; \sigma_{Z_i}\sigma_{A_i}$"
    )
    ax.set_ylabel("Density (normalised count)")
    ax.set_title(
        f"Distribution of $\\rho$ at final epoch (MNIST FNN, "
        f"trained with: {args.kernel})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / f"rho_hist_seed{args.seed}.png", dpi=150)
    plt.close(fig)

    print(f"Saved plots to {out}/")


if __name__ == "__main__":
    main()
