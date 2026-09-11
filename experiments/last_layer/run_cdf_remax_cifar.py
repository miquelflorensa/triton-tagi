"""CDF-TAGI-V/Remax head and shared-scale calibration on frozen CIFAR features.

A standalone driver, deliberately outside ``run_study.py``'s run-hash space so
it cannot disturb the finished initialization study, but reusing that study's
cached frozen features.

Protocol, kept to the separation the formulation requires:

* the head trains on the train split through the Gaussian logit channel;
* model selection -- both the prior gain and the CDF head's ``(epsilon,
  kappa)`` constants -- reads the *first half* of the validation split, since
  those constants are a model choice that must be fixed before calibration;
* the shared deviation scale is calibrated on the *second half*, disjoint from
  both training and selection, because a post-update moment has already
  assimilated its own label and is not a valid calibration input;
* everything reported comes from the untouched test split.

Since ``Remax(g x) = Remax(x)``, the calibrated quantity is not a temperature
but the positive deviation scale ``s = e^L`` of
``X_{i,L} = mu_i + e^L [(Z_i - mu_i) + V_i]``, which holds the predictive logit
mean fixed and rescales the combined epistemic and aleatoric spread.

Three stages, each selectable with ``--stage``:

``gain``    screen the prior gain at fixed head constants;
``sweep``   screen ``(epsilon, kappa)`` at a fixed gain. The variance head is
            bounded in ``(epsilon, epsilon + kappa)``, and a head pinned at
            either end has stopped adapting, so this stage reports the learned
            aleatoric variance and the saturated fraction alongside the
            metrics -- a cell can win on NLL while the head does no work;
``confirm`` repeat the chosen cell over seeds, which is what separates a real
            calibration benefit from calibration-split noise.

Run ``python -m experiments.last_layer.run_cdf_remax_cifar --stage sweep``
from the repository root.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from triton_tagi.cdf_remax import remax_forward_diagnostics
from triton_tagi.cdf_variance import cdf_variance_moments
from triton_tagi.classification import TAGILastLayerClassifier
from triton_tagi.metrics import classification_metrics, evaluate_ood_comprehensive

FEATURES_ROOT = Path("runs/last_layer/cifar_frozen_last_layer_v2_40k10k/features")
BATCH_SIZE = 256

# At ten classes the reduced quadrature costs 4e-7 in probability and runs four
# times faster than the module defaults; see the note's instruction to recheck
# the orders whenever the parameter range changes.
LAPLACE_ORDER = 80
HERMITE_ORDER = 16

MEAN_INIT = "zero"
GAIN_GRID = (0.1, 0.3, 1.0)

# The first CIFAR-10 run left the head pinned at its floor with h ~= 0.022
# against a cap of 1.52, so the grid brackets that value from both sides. The
# starting aleatoric variance is held fixed across cells, so what varies is the
# range the head is allowed to move over, not where it begins.
EPSILON_GRID = (1e-3, 1e-2)
KAPPA_GRID = (0.05, 0.2, 1.5)
ALEATORIC_INIT = 0.02

# Fraction of kappa counted as pinned against either end of the range.
SATURATION_TOLERANCE = 1e-2

CALIBRATION_GRID_SIZE = 65
CALIBRATION_REFINEMENTS = 2

# Calibration rows actually used. The scale is a single scalar, so its
# posterior is already tight on a couple of thousand labels, and the fit cost
# is linear in the rows -- at a hundred classes the full half-split would
# dominate the whole run for no gain in the estimate.
CALIBRATION_ROWS: int | None = None

# The refinement check inside the diagnostics evaluates the moments twice, at
# doubled orders, so it dominates a hundred-class arm. A few hundred rows is
# ample for a saturation and atom summary.
DIAGNOSTIC_ROWS = 512


def load_split(dataset: str, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(
        FEATURES_ROOT / dataset / f"{name}.pt", map_location="cpu", weights_only=False
    )
    return payload["features"].cuda(), payload["labels"].cuda()


def build(
    input_dim: int,
    num_classes: int,
    *,
    gain: float,
    epsilon: float,
    kappa: float,
) -> TAGILastLayerClassifier:
    return TAGILastLayerClassifier(
        input_dim,
        num_classes,
        head="cdf_remax",
        device="cuda",
        gain_w=gain,
        gain_b=gain,
        mean_init=MEAN_INIT,
        cdf_epsilon=epsilon,
        cdf_kappa=kappa,
        cdf_aleatoric_init=ALEATORIC_INIT,
        cdf_laplace_order=LAPLACE_ORDER,
        cdf_hermite_order=HERMITE_ORDER,
    )


def probabilities(
    classifier: TAGILastLayerClassifier, features: torch.Tensor, batch_size: int = 2048
) -> torch.Tensor:
    parts = [
        classifier.predict(features[start : start + batch_size]).probabilities
        for start in range(0, features.shape[0], batch_size)
    ]
    return torch.cat(parts)


def score(
    classifier: TAGILastLayerClassifier, features: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    return classification_metrics(probabilities(classifier, features), labels)


def head_diagnostics(
    classifier: TAGILastLayerClassifier, features: torch.Tensor
) -> dict[str, float]:
    """Report whether the bounded variance head has any room left to move."""

    mu_z, var_z, nu, r = classifier.cdf_remax_moments(features[:DIAGNOSTIC_ROWS], batch_size=1024)
    h_mean, _, _ = cdf_variance_moments(
        nu, r, epsilon=classifier.cdf_epsilon, kappa=classifier.cdf_kappa
    )
    diagnostics = remax_forward_diagnostics(
        mu_z,
        var_z,
        nu,
        r,
        scale_mean=classifier.cdf_scale_mean,
        scale_variance=classifier.cdf_scale_variance,
        epsilon=classifier.cdf_epsilon,
        kappa=classifier.cdf_kappa,
        laplace_order=LAPLACE_ORDER,
        hermite_order=HERMITE_ORDER,
        saturation_tolerance=SATURATION_TOLERANCE,
    )
    span = classifier.cdf_kappa
    return {
        "aleatoric_min": float(h_mean.min()),
        "aleatoric_median": float(h_mean.median()),
        "aleatoric_max": float(h_mean.max()),
        # Where the head sits inside its own range, 0 at the floor, 1 at the cap.
        "range_position_median": float((h_mean.median() - classifier.cdf_epsilon) / span),
        "range_used": float((h_mean.max() - h_mean.min()) / span),
        "epistemic_median": float(var_z.median()),
        "saturated_fraction": float(diagnostics["head_saturation_fraction"]),
        "all_zero_mean": float(diagnostics["all_zero_probability"].mean()),
        "all_zero_max": float(diagnostics["all_zero_probability"].max()),
        "quadrature_refinement_error": float(diagnostics["quadrature_refinement_error"]),
    }


def train_calibrate_evaluate(
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    gain: float,
    epsilon: float,
    kappa: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    """One complete arm: train, calibrate on held-out labels, evaluate."""

    train_x, train_y = splits["train"]
    select_x, select_y = splits["select"]
    calib_x, calib_y = splits["calibrate"]
    test_x, test_y = splits["test"]

    classifier = build(
        train_x.shape[1],
        int(train_y.max()) + 1,
        gain=gain,
        epsilon=epsilon,
        kappa=kappa,
    )
    classifier.fit(train_x, train_y, epochs=epochs, batch_size=BATCH_SIZE, seed=seed)

    select = score(classifier, select_x, select_y)
    uncalibrated = score(classifier, test_x, test_y)
    rows = CALIBRATION_ROWS or calib_x.shape[0]
    posterior = classifier.calibrate_remax_log_scale(
        calib_x[:rows],
        calib_y[:rows],
        method="grid",
        grid_size=CALIBRATION_GRID_SIZE,
        refinements=CALIBRATION_REFINEMENTS,
        batch_size=2048,
    )
    calibrated = score(classifier, test_x, test_y)
    select_calibrated = score(classifier, select_x, select_y)

    return {
        "gain": gain,
        "epsilon": epsilon,
        "kappa": kappa,
        "seed": seed,
        "scale": posterior.summary(),
        "calibration_rows": int(rows),
        "select": select,
        "select_calibrated": select_calibrated,
        "test": {"uncalibrated": uncalibrated, "calibrated": calibrated},
        "head": head_diagnostics(classifier, test_x),
        "classifier": classifier,
    }


def make_splits(dataset: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    train = load_split(dataset, "train")
    val_x, val_y = load_split(dataset, "validation")
    half = val_x.shape[0] // 2
    return {
        "train": train,
        "select": (val_x[:half], val_y[:half]),
        "calibrate": (val_x[half:], val_y[half:]),
        "test": load_split(dataset, "test"),
    }


def describe(splits: dict[str, tuple[torch.Tensor, torch.Tensor]], dataset: str) -> None:
    sizes = {name: value[0].shape[0] for name, value in splits.items()}
    print(
        f"{dataset}: {sizes['train']} train, {sizes['select']} select, "
        f"{sizes['calibrate']} calibrate, {sizes['test']} test, "
        f"{splits['train'][0].shape[1]} features, "
        f"{int(splits['train'][1].max()) + 1} classes\n"
    )


def stage_gain(splits, epochs: int, seed: int) -> list[dict[str, Any]]:
    print(
        f"gain screen ({epochs} epochs, mean_init={MEAN_INIT!r}, "
        f"epsilon={EPSILON_GRID[-1]}, kappa={KAPPA_GRID[-1]})"
    )
    rows = []
    for gain in GAIN_GRID:
        started = time.time()
        arm = train_calibrate_evaluate(
            splits,
            gain=gain,
            epsilon=EPSILON_GRID[-1],
            kappa=KAPPA_GRID[-1],
            epochs=epochs,
            seed=seed,
        )
        arm.pop("classifier")
        rows.append(arm)
        print(
            f"  gain {gain:<5} select nll {arm['select']['nll']:.4f}  "
            f"acc {arm['select']['accuracy']:.4f}   [{time.time() - started:.0f}s]"
        )
    return rows


def stage_sweep(splits, gain: float, epochs: int, seed: int) -> list[dict[str, Any]]:
    print(
        f"(epsilon, kappa) sweep at gain {gain}, {epochs} epochs, aleatoric_init {ALEATORIC_INIT}\n"
    )
    header = (
        f"  {'eps':>6}{'kappa':>7}{'sel nll':>9}{'test nll':>10}{'test ece':>9}"
        f"{'acc':>8}{'s':>7}{'h median':>10}{'pos':>7}{'used':>7}{'sat':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows = []
    for epsilon in EPSILON_GRID:
        for kappa in KAPPA_GRID:
            arm = train_calibrate_evaluate(
                splits,
                gain=gain,
                epsilon=epsilon,
                kappa=kappa,
                epochs=epochs,
                seed=seed,
            )
            arm.pop("classifier")
            rows.append(arm)
            head, test = arm["head"], arm["test"]["calibrated"]
            print(
                f"  {epsilon:>6.3f}{kappa:>7.2f}"
                f"{arm['select_calibrated']['nll']:>9.4f}"
                f"{test['nll']:>10.4f}{test['ece']:>9.4f}{test['accuracy']:>8.4f}"
                f"{arm['scale']['scale_median']:>7.3f}"
                f"{head['aleatoric_median']:>10.4f}"
                f"{head['range_position_median']:>7.3f}"
                f"{head['range_used']:>7.3f}{head['saturated_fraction']:>7.3f}"
            )
    return rows


def stage_confirm(splits, cell: dict[str, float], epochs: int, seeds: int) -> dict[str, Any]:
    print(
        f"\nconfirm: {seeds} seeds at gain {cell['gain']}, "
        f"epsilon {cell['epsilon']}, kappa {cell['kappa']}\n"
    )
    print(
        f"  {'seed':>5}{'uncal nll':>11}{'cal nll':>9}{'uncal ece':>11}"
        f"{'cal ece':>9}{'acc':>8}{'s':>7}"
    )
    rows = []
    for seed in range(seeds):
        arm = train_calibrate_evaluate(splits, epochs=epochs, seed=seed, **cell)
        arm.pop("classifier")
        rows.append(arm)
        before, after = arm["test"]["uncalibrated"], arm["test"]["calibrated"]
        print(
            f"  {seed:>5}{before['nll']:>11.4f}{after['nll']:>9.4f}"
            f"{before['ece']:>11.4f}{after['ece']:>9.4f}"
            f"{after['accuracy']:>8.4f}{arm['scale']['scale_median']:>7.3f}"
        )

    summary: dict[str, Any] = {"cell": cell, "seeds": rows}
    print()
    for key in ("accuracy", "nll", "ece", "adaptive_ece", "classwise_ece", "brier"):
        for arm_name in ("uncalibrated", "calibrated"):
            values = [row["test"][arm_name][key] for row in rows]
            mean = statistics.fmean(values)
            deviation = statistics.stdev(values) if len(values) > 1 else 0.0
            summary.setdefault(arm_name, {})[key] = {"mean": mean, "sd": deviation}
        before = summary["uncalibrated"][key]
        after = summary["calibrated"][key]
        print(
            f"  {key:<15} uncalibrated {before['mean']:.4f} +/- {before['sd']:.4f}"
            f"   calibrated {after['mean']:.4f} +/- {after['sd']:.4f}"
        )
    scales = [row["scale"]["scale_median"] for row in rows]
    summary["scale_median"] = {
        "mean": statistics.fmean(scales),
        "sd": statistics.stdev(scales) if len(scales) > 1 else 0.0,
    }
    print(
        f"  {'fitted scale':<15} {summary['scale_median']['mean']:.4f} "
        f"+/- {summary['scale_median']['sd']:.4f}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cifar10", choices=("cifar10", "cifar100"))
    parser.add_argument("--stage", default="sweep", choices=("gain", "sweep", "confirm", "all"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--gain", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--calibration-rows", type=int, default=None)
    parser.add_argument("--ood", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    global CALIBRATION_GRID_SIZE, CALIBRATION_ROWS
    if arguments.grid_size is not None:
        CALIBRATION_GRID_SIZE = arguments.grid_size
    if arguments.calibration_rows is not None:
        CALIBRATION_ROWS = arguments.calibration_rows

    splits = make_splits(arguments.dataset)
    describe(splits, arguments.dataset)
    results: dict[str, Any] = {
        "dataset": arguments.dataset,
        "epochs": arguments.epochs,
        "mean_init": MEAN_INIT,
        "aleatoric_init": ALEATORIC_INIT,
        "laplace_order": LAPLACE_ORDER,
        "hermite_order": HERMITE_ORDER,
        "calibration_grid_size": CALIBRATION_GRID_SIZE,
        "calibration_rows": CALIBRATION_ROWS,
    }
    gain = arguments.gain

    if arguments.stage in ("gain", "all"):
        rows = stage_gain(splits, arguments.epochs, arguments.seed)
        results["gain_screen"] = rows
        gain = min(rows, key=lambda row: row["select"]["nll"])["gain"]
        print(f"\nselected gain {gain} by select-split NLL\n")

    cell = {"gain": gain, "epsilon": arguments.epsilon, "kappa": arguments.kappa}
    if arguments.stage in ("sweep", "all"):
        rows = stage_sweep(splits, gain, arguments.epochs, arguments.seed)
        results["sweep"] = rows
        # Selection reads the calibrated select split, never the test split.
        best = min(rows, key=lambda row: row["select_calibrated"]["nll"])
        cell = {
            "gain": best["gain"],
            "epsilon": best["epsilon"],
            "kappa": best["kappa"],
        }
        print(
            f"\nselected epsilon {cell['epsilon']}, kappa {cell['kappa']} "
            "by calibrated select-split NLL"
        )

    if arguments.stage in ("confirm", "all"):
        if cell["epsilon"] is None or cell["kappa"] is None:
            raise SystemExit("--stage confirm needs --epsilon and --kappa, or run --stage all")
        results["confirm"] = stage_confirm(splits, cell, arguments.epochs, arguments.seeds)

    if arguments.ood and cell["epsilon"] is not None:
        print("\nout-of-distribution, SVHN as the outlier set")
        arm = train_calibrate_evaluate(splits, epochs=arguments.epochs, seed=arguments.seed, **cell)
        classifier = arm.pop("classifier")
        svhn_x, _ = load_split(arguments.dataset, "svhn")
        ood = evaluate_ood_comprehensive(
            probabilities(classifier, splits["test"][0]),
            probabilities(classifier, svhn_x),
        )
        results["ood"] = ood
        for score_name, block in sorted(ood.items()):
            print(f"  {score_name:<26} auroc {block['auroc']:.4f}  fpr95 {block['fpr95']:.4f}")

    destination = arguments.output or Path(__file__).resolve().parent / (
        f"cdf_remax_{arguments.dataset}_{arguments.stage}.json"
    )
    destination.write_text(json.dumps(results, indent=2, default=float) + "\n")
    print(f"\nwritten to {destination}")


if __name__ == "__main__":
    main()
