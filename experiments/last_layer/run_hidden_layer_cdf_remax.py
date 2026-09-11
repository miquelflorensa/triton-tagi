"""The CDF-TAGI-V/Remax head with and without a TAGI hidden layer.

A companion to ``run_hidden_layer_cifar100.py``, kept separate because
``cdf_remax`` needs a different protocol and mixing the two would invalidate
both. This head learns its own observation variance, so there is no ``sigma_v``
axis to screen, and its shared deviation scale is fitted *after* training on
labels the head has never assimilated. That forces the three-way split
``run_cdf_remax_cifar.py`` established and this script reuses:

* train on the 40k train split;
* select the prior gain on the *first half* of the validation split;
* calibrate the log-scale on the *second half*, disjoint from both;
* report the untouched test split, uncalibrated and calibrated.

The head constants are pinned at the values the finished CIFAR-100 screen
selected (epsilon = 1e-2, kappa = 1.5, aleatoric init = 0.02), so the only
thing moving is the depth of the head.

Worth stating up front: on CIFAR-100 this head already reaches test accuracy
0.766 against a 0.765 deterministic reference, so it has no accuracy gap to
close. What is open is its NLL and ECE, which post-hoc calibration improves but
does not fix. This arm asks whether depth moves those.

Usage:
    python experiments/last_layer/run_hidden_layer_cdf_remax.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi.classification import TAGILastLayerClassifier  # noqa: E402
from triton_tagi.metrics import classification_metrics  # noqa: E402

FEATURES_ROOT = Path("runs/last_layer/cifar_frozen_last_layer_v2_40k10k/features")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "hidden_layer_cdf_remax.json"
BATCH_SIZE = 256

# Pinned at the finished CIFAR-100 screen's selection; see cdf_remax_cifar100_all.json.
EPSILON = 1e-2
KAPPA = 1.5
ALEATORIC_INIT = 0.02
LAPLACE_ORDER = 80
HERMITE_ORDER = 16
CALIBRATION_GRID_SIZE = 33
CALIBRATION_REFINEMENTS = 2
CALIBRATION_ROWS = 2000

GAIN_GRID = (0.1, 0.3, 1.0)
MEAN_INITS = ("zero", "random")
HIDDEN_ARMS = ((), (512,))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_split(dataset: str, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(
        FEATURES_ROOT / dataset / f"{name}.pt", map_location="cpu", weights_only=False
    )
    return payload["features"].cuda(), payload["labels"].cuda().long()


def build_splits(dataset: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    train_x, train_y = load_split(dataset, "train")
    val_x, val_y = load_split(dataset, "validation")
    test_x, test_y = load_split(dataset, "test")
    half = val_x.shape[0] // 2
    return {
        "train": (train_x, train_y),
        "select": (val_x[:half], val_y[:half]),
        "calibrate": (val_x[half:], val_y[half:]),
        "test": (test_x, test_y),
    }


def probabilities(classifier, features, batch_size: int = 2048) -> torch.Tensor:
    # The Laplace-Remax average is quadrature over every class, so a hundred-class
    # test split is chunked rather than scored in one call.
    return torch.cat(
        [
            classifier.predict(features[start : start + batch_size]).probabilities
            for start in range(0, features.shape[0], batch_size)
        ]
    )


def score(classifier, features, labels) -> dict[str, float]:
    return classification_metrics(probabilities(classifier, features), labels)


def run_arm(
    splits,
    *,
    hidden: tuple[int, ...],
    gain: float,
    mean_init: str,
    num_classes: int,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    train_x, train_y = splits["train"]
    select_x, select_y = splits["select"]
    calibrate_x, calibrate_y = splits["calibrate"]
    test_x, test_y = splits["test"]

    torch.manual_seed(seed)
    classifier = TAGILastLayerClassifier(
        train_x.shape[1],
        num_classes,
        head="cdf_remax",
        device="cuda",
        hidden_dims=hidden,
        mean_init=mean_init,
        gain_w=gain,
        gain_b=gain,
        cdf_epsilon=EPSILON,
        cdf_kappa=KAPPA,
        cdf_aleatoric_init=ALEATORIC_INIT,
        cdf_laplace_order=LAPLACE_ORDER,
        cdf_hermite_order=HERMITE_ORDER,
    )
    started = time.perf_counter()
    classifier.fit(train_x, train_y, epochs=epochs, batch_size=BATCH_SIZE, seed=seed)

    select_uncalibrated = score(classifier, select_x, select_y)
    test_uncalibrated = score(classifier, test_x, test_y)
    posterior = classifier.calibrate_remax_log_scale(
        calibrate_x[:CALIBRATION_ROWS],
        calibrate_y[:CALIBRATION_ROWS],
        grid_size=CALIBRATION_GRID_SIZE,
        refinements=CALIBRATION_REFINEMENTS,
    )
    return {
        "hidden_dims": list(hidden),
        "gain": gain,
        "mean_init": mean_init,
        "epochs": epochs,
        "seed": seed,
        "wall_s": time.perf_counter() - started,
        "parameters": classifier.net.num_parameters(),
        "log_scale_mean": float(posterior.mean),
        "log_scale_variance": float(posterior.variance),
        "select": {
            "uncalibrated": select_uncalibrated,
            "calibrated": score(classifier, select_x, select_y),
        },
        "test": {
            "uncalibrated": test_uncalibrated,
            "calibrated": score(classifier, test_x, test_y),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cifar100", choices=("cifar10", "cifar100"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gains", nargs="+", type=float, default=list(GAIN_GRID))
    parser.add_argument("--mean-inits", nargs="+", default=list(MEAN_INITS))
    parser.add_argument("--hidden", nargs="+", default=["0", "512"])
    args = parser.parse_args()

    splits = build_splits(args.dataset)
    num_classes = int(splits["train"][1].max().item()) + 1
    hidden_arms = [() if width == "0" else (int(width),) for width in args.hidden]
    print(
        f"{args.dataset}: {splits['train'][0].shape[0]} train, "
        f"{splits['select'][0].shape[0]} select, {splits['calibrate'][0].shape[0]} calibrate, "
        f"{splits['test'][0].shape[0]} test, {num_classes} classes",
        flush=True,
    )

    existing = json.loads(args.output.read_text()) if args.output.exists() else {}
    runs: dict[str, Any] = existing.get("runs", {})
    payload = {
        "dataset": args.dataset,
        "epsilon": EPSILON,
        "kappa": KAPPA,
        "aleatoric_init": ALEATORIC_INIT,
        "calibration_rows": CALIBRATION_ROWS,
        "protocol": "train / select-half / calibrate-half / test, per run_cdf_remax_cifar.py",
        "runs": runs,
    }

    grid = list(itertools.product(hidden_arms, args.mean_inits, args.gains))
    for index, (hidden, mean_init, gain) in enumerate(grid, start=1):
        width = "flat" if not hidden else "x".join(str(w) for w in hidden)
        key = f"{width}|{mean_init}|g{gain}"
        if key in runs:
            continue
        runs[key] = run_arm(
            splits,
            hidden=hidden,
            gain=gain,
            mean_init=mean_init,
            num_classes=num_classes,
            epochs=args.epochs,
            seed=args.seed,
        )
        row = runs[key]
        print(
            f"[{index:2d}/{len(grid)}] {key:<22} "
            f"select_nll={row['select']['calibrated']['nll']:.4f}  "
            f"test_acc={row['test']['calibrated']['accuracy']:.4f} "
            f"test_nll={row['test']['calibrated']['nll']:.4f} "
            f"test_ece={row['test']['calibrated']['ece']:.4f} "
            f"({row['wall_s']:.0f}s)",
            flush=True,
        )
        atomic_json(args.output, payload)

    atomic_json(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
