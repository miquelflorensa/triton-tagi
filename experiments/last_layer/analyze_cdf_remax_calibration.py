"""Reliability diagnostics and a no-aleatoric ablation for CDF-Remax.

This intentionally reuses the train/select/calibrate/test separation from
``run_cdf_remax_cifar.py``.  The selected head is trained once.  We then score
five read-outs of that same frozen head:

* the full predictive model at unit deviation scale;
* the full model with its scale fitted on the calibration split;
* the epistemic-only model at unit scale;
* the epistemic-only model with the full model's fitted scale (an intervention
  on the noise term only); and
* the epistemic-only model with its own scale refitted on the same calibration
  split (the fair calibrated ablation).

"Epistemic only" sets the learned aleatoric innovation V exactly to zero while
retaining the prediction-head variance and the shared scale.  It uses the same
Laplace-Remax kernels as the full model, without passing through a tiny-noise
approximation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from experiments.last_layer.run_cdf_remax_cifar import (
    CALIBRATION_GRID_SIZE,
    CALIBRATION_REFINEMENTS,
    CALIBRATION_ROWS,
    HERMITE_ORDER,
    LAPLACE_ORDER,
    build,
    make_splits,
)
from triton_tagi.cdf_remax import remax_scale_moments
from triton_tagi.hsm_calibration import (
    _grid_moments,
    _laplace_moments,
    _prior_grid_bounds,
    _refined_grid,
)
from triton_tagi.metrics import classification_metrics
from triton_tagi.remax_kernels import hermite_rule, laplace_rule, remax_kernels

DEFAULT_CELL = {"gain": 1.0, "epsilon": 0.001, "kappa": 0.05}
PRIOR_MEAN = 0.0
PRIOR_VARIANCE = 0.5**2
SCALE_ORDER = 20
N_BINS = 15
_LOG_FLOOR = 1e-300


@torch.no_grad()
def gaussian_remax_probabilities(
    mu_z: Tensor,
    var_z: Tensor,
    *,
    log_scale: float,
    laplace_order: int = LAPLACE_ORDER,
    batch_size: int = 2048,
) -> Tensor:
    """Return E[Remax(X)] for X = mu + exp(l) (Z - mu).

    This is the exact zero-aleatoric special case of the CDF-Remax predictive
    model under its factorised Gaussian logit approximation.
    """

    if mu_z.shape != var_z.shape or mu_z.dim() != 2:
        raise ValueError("mu_z and var_z must share shape (samples, classes)")
    outputs = []
    for start in range(0, mu_z.shape[0], batch_size):
        mean = mu_z[start : start + batch_size].double()
        variance = var_z[start : start + batch_size].double().clamp_min(0.0)
        # Exact zero variance needs a deterministic rectified-logit path.  The
        # trained head is strictly positive here; tiny is only a guard for an
        # otherwise undefined Gaussian standardisation, not injected noise.
        deviation = math.exp(log_scale) * variance.sqrt()
        if bool((deviation == 0).any()):
            raise ArithmeticError("prediction-head variance contains an exact zero")

        t, weights = laplace_rule(laplace_order, reference=mean)
        kernel_l, kernel_d = remax_kernels(
            t,
            mean.unsqueeze(-1),
            deviation.unsqueeze(-1),
            highest=1,
        )
        log_l = kernel_l.clamp_min(_LOG_FLOOR).log()
        log_product = log_l.sum(dim=1, keepdim=True)
        product_except = (log_product - log_l).exp()
        first = (weights * kernel_d * product_except).sum(dim=-1)

        zero = torch.special.ndtr(-(mean / deviation))
        atom = zero.clamp_min(_LOG_FLOOR).log().sum(dim=-1).exp()
        first = first + (atom / mean.shape[1]).unsqueeze(-1)
        outputs.append(first / first.sum(dim=-1, keepdim=True))
    return torch.cat(outputs)


@torch.no_grad()
def integrated_gaussian_remax_probabilities(
    mu_z: Tensor,
    var_z: Tensor,
    *,
    scale_mean: float,
    scale_variance: float,
    scale_order: int = SCALE_ORDER,
) -> Tensor:
    """Integrate the epistemic-only read-out over a Gaussian log-scale."""

    nodes, weights = hermite_rule(scale_order, reference=mu_z)
    deviation = math.sqrt(max(scale_variance, 0.0))
    output = torch.zeros_like(mu_z, dtype=torch.float64)
    for node, weight in zip(nodes.tolist(), weights.tolist(), strict=True):
        output += weight * gaussian_remax_probabilities(
            mu_z, var_z, log_scale=scale_mean + deviation * node
        )
    return output / output.sum(dim=-1, keepdim=True)


@torch.no_grad()
def fit_epistemic_only_scale(
    mu_z: Tensor,
    var_z: Tensor,
    labels: Tensor,
    *,
    grid_size: int,
    refinements: int,
) -> dict[str, float]:
    """Fit the scalar log-scale after setting the aleatoric noise to zero."""

    labels = labels.to(mu_z.device).long()
    bounds = _prior_grid_bounds(PRIOR_MEAN, PRIOR_VARIANCE)
    grid: Tensor | None = None
    log_posterior: Tensor | None = None
    prior_deviation = math.sqrt(PRIOR_VARIANCE)
    for sweep in range(refinements + 1):
        if grid is None:
            grid = torch.linspace(
                bounds[0], bounds[1], grid_size, dtype=torch.float64, device=mu_z.device
            )
        values = []
        for log_scale in grid.tolist():
            probabilities = gaussian_remax_probabilities(
                mu_z, var_z, log_scale=log_scale
            )
            true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
            likelihood = true_probability.clamp_min(_LOG_FLOOR).log().sum()
            penalty = -0.5 * (log_scale - PRIOR_MEAN) ** 2 / PRIOR_VARIANCE
            values.append(likelihood + penalty)
        log_posterior = torch.stack(values)
        if sweep < refinements:
            mode, curvature = _laplace_moments(grid[None, :], log_posterior[None, :])
            grid = _refined_grid(mode, curvature, prior_deviation, grid_size).reshape(-1)

    assert grid is not None and log_posterior is not None
    mean, variance = _grid_moments(grid[None, :], log_posterior[None, :])
    resolved_mean = float(mean.reshape(-1)[0])
    resolved_variance = float(variance.reshape(-1)[0].clamp_min(0.0))
    return {
        "log_scale_mean": resolved_mean,
        "log_scale_variance": resolved_variance,
        "scale_median": math.exp(resolved_mean),
        "scale_mean": math.exp(resolved_mean + 0.5 * resolved_variance),
        "samples": float(labels.numel()),
        "method": "grid_exact_zero_aleatoric",
    }


def reliability_bins(
    probabilities: Tensor, labels: Tensor, *, n_bins: int = N_BINS
) -> dict[str, Any]:
    """Return top-label and pooled one-vs-rest reliability-bin summaries."""

    probabilities = probabilities.detach().double().cpu()
    labels = labels.detach().long().cpu()
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels).double()
    one_hot = torch.zeros_like(probabilities).scatter_(1, labels[:, None], 1.0)

    def binned(values: Tensor, targets: Tensor) -> list[dict[str, float | int]]:
        values, targets = values.reshape(-1), targets.reshape(-1)
        bucket = (values * n_bins).floor().long().clamp_(0, n_bins - 1)
        rows = []
        for index in range(n_bins):
            mask = bucket == index
            count = int(mask.sum())
            mean_probability = float(values[mask].mean()) if count else math.nan
            empirical_frequency = float(targets[mask].mean()) if count else math.nan
            gap = empirical_frequency - mean_probability if count else math.nan
            rows.append(
                {
                    "lower": index / n_bins,
                    "upper": (index + 1) / n_bins,
                    "count": count,
                    "mean_probability": mean_probability,
                    "empirical_frequency": empirical_frequency,
                    "signed_gap": gap,
                    "ece_contribution": (count / values.numel()) * abs(gap) if count else 0.0,
                }
            )
        return rows

    per_class = []
    for class_index in range(probabilities.shape[1]):
        rows = binned(probabilities[:, class_index], one_hot[:, class_index])
        per_class.append(sum(float(row["ece_contribution"]) for row in rows))
    per_class_array = np.asarray(per_class)
    return {
        "top_label": binned(confidence, correct),
        "pooled_one_vs_rest": binned(probabilities, one_hot),
        "per_class_ece": per_class,
        "per_class_ece_summary": {
            "min": float(per_class_array.min()),
            "median": float(np.median(per_class_array)),
            "mean": float(per_class_array.mean()),
            "p90": float(np.quantile(per_class_array, 0.9)),
            "max": float(per_class_array.max()),
        },
    }


def plot_reliability(variants: dict[str, dict[str, Any]], destination: Path) -> None:
    """Write a compact view of top-label and multiclass calibration."""

    labels = {
        "full_uncalibrated": "full, s=1",
        "full_calibrated": "full, fitted s",
        "no_aleatoric_uncalibrated": "no aleatoric, s=1",
        "no_aleatoric_full_scale": "no aleatoric, full-model s",
        "no_aleatoric_refit": "no aleatoric, refitted s",
    }
    colors = {
        "full_uncalibrated": "#4477AA",
        "full_calibrated": "#114477",
        "no_aleatoric_uncalibrated": "#EE9944",
        "no_aleatoric_full_scale": "#999999",
        "no_aleatoric_refit": "#BB4411",
    }
    styles = {
        "full_uncalibrated": "--",
        "full_calibrated": "-",
        "no_aleatoric_uncalibrated": "--",
        "no_aleatoric_full_scale": ":",
        "no_aleatoric_refit": "-",
    }

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    top_rel, top_hist, multi_rel, class_hist = axes.flat
    top_rel.plot([0, 1], [0, 1], color="black", linewidth=1, alpha=0.6)
    multi_rel.plot([0, 1], [0, 1], color="black", linewidth=1, alpha=0.6)

    for key, block in variants.items():
        reliability = block["reliability"]
        top = reliability["top_label"]
        x = np.asarray([row["mean_probability"] for row in top], dtype=float)
        y = np.asarray([row["empirical_frequency"] for row in top], dtype=float)
        count = np.asarray([row["count"] for row in top], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        top_rel.plot(
            x[valid], y[valid], marker="o", markersize=4, linewidth=1.7,
            color=colors[key], linestyle=styles[key], label=labels[key]
        )
        centers = (np.arange(N_BINS) + 0.5) / N_BINS
        top_hist.step(
            centers, np.maximum(count, 0.5), where="mid", linewidth=1.6,
            color=colors[key], linestyle=styles[key], label=labels[key]
        )

        multi = reliability["pooled_one_vs_rest"]
        mx = np.asarray([row["mean_probability"] for row in multi], dtype=float)
        my = np.asarray([row["empirical_frequency"] for row in multi], dtype=float)
        valid = np.isfinite(mx) & np.isfinite(my)
        multi_rel.plot(
            mx[valid], my[valid], marker="o", markersize=4, linewidth=1.7,
            color=colors[key], linestyle=styles[key], label=labels[key]
        )
        class_hist.hist(
            reliability["per_class_ece"], bins=16, histtype="step", linewidth=1.8,
            color=colors[key], linestyle=styles[key], label=labels[key]
        )

    top_rel.set(title="Top-label reliability (15 equal-width bins)", xlabel="mean confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
    top_rel.legend(fontsize=8)
    top_rel.grid(alpha=0.2)
    top_hist.set(title="Top-label confidence histogram", xlabel="confidence", ylabel="test examples (log scale)", xlim=(0, 1), yscale="log")
    top_hist.grid(alpha=0.2)
    multi_rel.set(title="Multiclass one-vs-rest reliability", xlabel="mean class probability", ylabel="empirical class frequency", xlim=(0, 0.5), ylim=(0, 0.5))
    multi_rel.grid(alpha=0.2)
    class_hist.set(title="Distribution of one-vs-rest ECE over 100 classes", xlabel="per-class ECE", ylabel="classes")
    class_hist.grid(alpha=0.2)
    figure.suptitle("CDF-TAGI-V/Remax on CIFAR-100: aleatoric ablation (seed 0)", fontsize=15)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--calibration-rows", type=int, default=CALIBRATION_ROWS or 2000)
    parser.add_argument("--grid-size", type=int, default=CALIBRATION_GRID_SIZE)
    parser.add_argument("--output", type=Path, default=Path("experiments/last_layer/cdf_remax_cifar100_aleatoric_ablation.json"))
    parser.add_argument("--figure", type=Path, default=Path("experiments/last_layer/cdf_remax_cifar100_reliability.png"))
    arguments = parser.parse_args()

    started = time.time()
    splits = make_splits("cifar100")
    train_x, train_y = splits["train"]
    calibration_x, calibration_y = splits["calibrate"]
    test_x, test_y = splits["test"]
    classifier = build(train_x.shape[1], 100, **DEFAULT_CELL)
    print(f"training selected head, seed {arguments.seed}", flush=True)
    classifier.fit(train_x, train_y, epochs=arguments.epochs, batch_size=256, seed=arguments.seed)

    calibration_x = calibration_x[: arguments.calibration_rows]
    calibration_y = calibration_y[: arguments.calibration_rows]
    print("extracting frozen forward summaries", flush=True)
    cal_mu, cal_var, cal_nu, cal_r = classifier.cdf_remax_moments(calibration_x, batch_size=2048)
    test_mu, test_var, test_nu, test_r = classifier.cdf_remax_moments(test_x, batch_size=2048)

    print("fitting full and epistemic-only scales", flush=True)
    full_posterior = classifier.calibrate_remax_log_scale(
        calibration_x,
        calibration_y,
        method="grid",
        grid_size=arguments.grid_size,
        refinements=CALIBRATION_REFINEMENTS,
        batch_size=2048,
    )
    epistemic_posterior = fit_epistemic_only_scale(
        cal_mu,
        cal_var,
        calibration_y,
        grid_size=arguments.grid_size,
        refinements=CALIBRATION_REFINEMENTS,
    )

    print("evaluating five read-outs", flush=True)
    full_uncalibrated, _, _ = remax_scale_moments(
        test_mu,
        test_var,
        test_nu,
        test_r,
        scale_mean=0.0,
        scale_variance=0.0,
        epsilon=DEFAULT_CELL["epsilon"],
        kappa=DEFAULT_CELL["kappa"],
        laplace_order=LAPLACE_ORDER,
        hermite_order=HERMITE_ORDER,
        scale_order=SCALE_ORDER,
    )
    full_calibrated, _, _ = remax_scale_moments(
        test_mu,
        test_var,
        test_nu,
        test_r,
        scale_mean=full_posterior.mean,
        scale_variance=full_posterior.variance,
        epsilon=DEFAULT_CELL["epsilon"],
        kappa=DEFAULT_CELL["kappa"],
        laplace_order=LAPLACE_ORDER,
        hermite_order=HERMITE_ORDER,
        scale_order=SCALE_ORDER,
    )
    no_aleatoric_uncalibrated = gaussian_remax_probabilities(test_mu, test_var, log_scale=0.0)
    no_aleatoric_full_scale = integrated_gaussian_remax_probabilities(
        test_mu,
        test_var,
        scale_mean=full_posterior.mean,
        scale_variance=full_posterior.variance,
    )
    no_aleatoric_refit = integrated_gaussian_remax_probabilities(
        test_mu,
        test_var,
        scale_mean=epistemic_posterior["log_scale_mean"],
        scale_variance=epistemic_posterior["log_scale_variance"],
    )
    probabilities = {
        "full_uncalibrated": full_uncalibrated,
        "full_calibrated": full_calibrated,
        "no_aleatoric_uncalibrated": no_aleatoric_uncalibrated,
        "no_aleatoric_full_scale": no_aleatoric_full_scale,
        "no_aleatoric_refit": no_aleatoric_refit,
    }
    variants = {}
    for name, value in probabilities.items():
        variants[name] = {
            "metrics": classification_metrics(value, test_y, n_bins=N_BINS),
            "reliability": reliability_bins(value, test_y, n_bins=N_BINS),
        }
        metric = variants[name]["metrics"]
        print(
            f"  {name:<29} acc {metric['accuracy']:.4f}  nll {metric['nll']:.4f}  "
            f"ece {metric['ece']:.4f}  classwise {metric['classwise_ece']:.4f}",
            flush=True,
        )

    result = {
        "dataset": "cifar100",
        "seed": arguments.seed,
        "epochs": arguments.epochs,
        "cell": DEFAULT_CELL,
        "calibration_rows": arguments.calibration_rows,
        "n_bins": N_BINS,
        "full_scale": full_posterior.summary(),
        "epistemic_only_scale": epistemic_posterior,
        "variants": variants,
        "elapsed_seconds": time.time() - started,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    plot_reliability(variants, arguments.figure)
    print(f"wrote {arguments.output}", flush=True)
    print(f"wrote {arguments.figure}", flush=True)


if __name__ == "__main__":
    main()
