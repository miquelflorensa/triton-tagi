"""Hierarchical-probit CIFAR study on frozen ResNet-18 features.

The main comparison is two models and nothing else:

  * TAGI, the hierarchical-probit head on the full K-leaf tree, with unit
    latent probit noise and prior branch offsets. No sigma_v, no fitted scale,
    no normalizer, no post-hoc calibration.
  * A conventional linear head trained with backprop cross-entropy. No
    temperature.

Both are scored with ordinary categorical cross-entropy in nats. Everything
else this script computes is an ablation against that comparison: the padded
tree and its categorical normalizer, the Gaussian +/-1 pseudo-target update, a
latent scale fitted on validation NLL, its Laplace posterior, and softmax
temperature scaling.

The train split fits the head, the validation split selects the prior gain (and
sigma_v, which only the Gaussian ablation has), and the test split is read once
per record.

Examples:
  python experiments/last_layer/run_hrc_calibration.py select --num-classes 10
  python experiments/last_layer/run_hrc_calibration.py run --num-classes 10
  python experiments/last_layer/run_hrc_calibration.py run --num-classes 8
  python experiments/last_layer/run_hrc_calibration.py report
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import (  # noqa: E402
    TAGILastLayerClassifier,
    classification_metrics,
    fit_softmax_temperature,
    probability_simplex_deviation,
)
from triton_tagi.cifar_study import load_feature_shard, seed_everything  # noqa: E402
from triton_tagi.hrc_probit import (  # noqa: E402
    alpha_from_log_tau,
    fit_hrc_log_tau,
    fit_hrc_log_tau_laplace,
    hrc_log_probs,
    log_tau_from_alpha,
)

DEFAULT_FEATURE_ROOT = (
    REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k/features/cifar10"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/last_layer/hrc_calibration/cifar10"

GAIN_GRID = (0.03, 0.1, 0.3, 1.0)
SIGMA_V_GRID = (0.05, 0.1, 0.3, 1.0)
SOFTMAX_LR_GRID = (0.01, 0.1)
LEGACY_LOG_TAU = log_tau_from_alpha(3.0)


# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingArm:
    """One trained head; several prediction variants are read off each arm."""

    name: str
    head: str
    tree: str
    role: str

    @property
    def is_probit(self) -> bool:
        return self.head == "hrc_probit"


TRAINING_ARMS = (
    TrainingArm("probit_full", "hrc_probit", "full", "main"),
    TrainingArm("probit_padded", "hrc_probit", "padded", "ablation"),
    TrainingArm("gaussian_padded", "hrc", "padded", "ablation"),
)

# The proposed head reports at its structural unit scale with no normalizer;
# every other read-out of the same trained weights is an ablation.
MAIN_VARIANT = "unit_scale"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


# ──────────────────────────────────────────────────────────────────────────────
#  Data
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Split:
    features: Tensor
    labels: Tensor


def load_splits(feature_root: Path, num_classes: int, device: str) -> dict[str, Split]:
    """Load the cached shards, restricted to the first ``num_classes`` classes."""

    splits: dict[str, Split] = {}
    for name in ("train", "validation", "test"):
        shard = load_feature_shard(feature_root / f"{name}.pt")
        features = shard["features"].to(device)
        labels = shard["labels"].to(device).long()
        keep = labels < num_classes
        splits[name] = Split(features[keep].contiguous(), labels[keep].contiguous())
    return splits


def class_permutation(num_classes: int, seed: int, device: str) -> Tensor:
    """Return the class-to-leaf assignment for one tree labelling.

    Seed zero is the identity, so the headline numbers use the natural class
    order and the remaining seeds measure the head's sensitivity to it.
    """

    if seed == 0:
        return torch.arange(num_classes, device=device)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_classes, generator=generator).to(device)


# ──────────────────────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────────────────────


def train_arm(
    arm: TrainingArm,
    splits: dict[str, Split],
    *,
    num_classes: int,
    permutation: Tensor,
    gain: float,
    sigma_v: float | None,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> TAGILastLayerClassifier:
    """Fit one head on leaf-indexed labels."""

    seed_everything(seed)
    classifier = TAGILastLayerClassifier(
        splits["train"].features.shape[1],
        num_classes,
        head=arm.head,
        hrc_tree=arm.tree,
        device=device,
        gain_w=gain,
        gain_b=gain,
        sigma_v=sigma_v,
    )
    classifier.fit(
        splits["train"].features,
        permutation[splits["train"].labels],
        epochs=epochs,
        batch_size=batch_size,
        sigma_v=sigma_v,
        seed=seed,
        record_initial=False,
    )
    return classifier


def train_softmax(
    splits: dict[str, Split],
    *,
    num_classes: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> torch.nn.Linear:
    """Fit the conventional backprop cross-entropy baseline."""

    seed_everything(seed)
    head = torch.nn.Linear(splits["train"].features.shape[1], num_classes).to(device)
    optimizer = torch.optim.SGD(
        head.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    features, labels = splits["train"].features, splits["train"].labels
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(features.shape[0], generator=generator).to(device)
        for start in range(0, order.shape[0], batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(features[index]), labels[index])
            loss.backward()
            optimizer.step()
        scheduler.step()
    return head


# ──────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────────────────────


def restore_class_order(values: Tensor, permutation: Tensor) -> Tensor:
    """Map leaf-indexed columns back to class-indexed columns."""

    return values.index_select(1, permutation)


def raw_score_diagnostics(scores: Tensor, labels: Tensor) -> dict[str, float]:
    """Score the unnormalized leaf products, which are not a distribution."""

    accuracy = float(scores.argmax(dim=1).eq(labels).float().mean())
    picked = scores.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12)
    one_hot = torch.zeros_like(scores).scatter_(1, labels[:, None], 1.0)
    return {
        "accuracy": accuracy,
        "nll": float(-picked.log().mean()),
        "brier": float(((scores - one_hot) ** 2).sum(dim=1).mean()),
        "simplex_deviation": probability_simplex_deviation(scores),
        "mean_total_mass": float(scores.sum(dim=1).mean()),
    }


def probability_metrics(probabilities: Tensor, labels: Tensor) -> dict[str, float]:
    metrics = classification_metrics(probabilities, labels)
    metrics["simplex_deviation"] = probability_simplex_deviation(probabilities)
    return {
        key: metrics[key]
        for key in (
            "accuracy",
            "nll",
            "brier",
            "ece",
            "adaptive_ece",
            "classwise_ece",
            "mean_confidence",
            "simplex_deviation",
        )
    }


@torch.no_grad()
def evaluate_hrc_variants(
    arm: TrainingArm,
    classifier: TAGILastLayerClassifier,
    splits: dict[str, Split],
    permutation: Tensor,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Score every prediction variant that this trained arm supports."""

    assert classifier.hrc is not None
    hrc = classifier.hrc
    moments = {
        name: classifier.hrc_node_moments(split.features, batch_size=batch_size)
        for name, split in splits.items()
        if name != "train"
    }
    leaf_labels = {
        name: permutation[split.labels] for name, split in splits.items() if name != "train"
    }

    fixed_log_tau = LEGACY_LOG_TAU if arm.head == "hrc" else 0.0
    fitted_log_tau = fit_hrc_log_tau(
        *moments["validation"], leaf_labels["validation"], hrc, normalize=True
    )
    posterior = fit_hrc_log_tau_laplace(
        *moments["validation"], leaf_labels["validation"], hrc, prior_variance=4.0
    )

    # unit_scale is the model as proposed: the structural scale, and a
    # normalizer only where the padded topology cannot do without one.
    variants: list[tuple[str, dict[str, Any]]] = [
        (MAIN_VARIANT, {"log_tau": fixed_log_tau, "normalize": not hrc.is_full}),
        ("unnormalized", {"log_tau": fixed_log_tau, "normalize": False}),
        ("normalized", {"log_tau": fixed_log_tau, "normalize": True}),
        ("fitted_scale", {"log_tau": fitted_log_tau, "normalize": True}),
        ("laplace_scale", {"log_tau": posterior.mean, "normalize": True}),
    ]

    records: list[dict[str, Any]] = []
    for variant, setting in variants:
        record: dict[str, Any] = {
            "variant": variant,
            "role": "main" if (arm.role == "main" and variant == MAIN_VARIANT) else "ablation",
            "log_tau": setting["log_tau"],
            "alpha_impl": alpha_from_log_tau(setting["log_tau"]),
            "normalized": setting["normalize"],
            "laplace_log_tau_mean": posterior.mean,
            "laplace_log_tau_variance": posterior.variance,
        }
        for name in ("validation", "test"):
            mean, variance = moments[name]
            scores = restore_class_order(
                hrc_log_probs(
                    mean,
                    variance,
                    hrc,
                    log_tau=setting["log_tau"],
                    normalize=setting["normalize"],
                ).exp(),
                permutation,
            )
            # Whether a row is a distribution is a property of the tree, not of
            # the normalize flag: a full tree sums to one with it switched off.
            # Only rows that are distributions get a proper score.
            if probability_simplex_deviation(scores) < 1e-6:
                scored = probability_metrics(scores, splits[name].labels)
            else:
                scored = raw_score_diagnostics(scores, splits[name].labels)
            record.update({f"{name}_{key}": value for key, value in scored.items()})
        records.append(record)
    return records


@torch.no_grad()
def evaluate_softmax_variants(
    head: torch.nn.Linear,
    splits: dict[str, Split],
) -> list[dict[str, Any]]:
    logits = {
        name: head(split.features).double() for name, split in splits.items() if name != "train"
    }
    temperature = fit_softmax_temperature(logits["validation"], splits["validation"].labels)
    records: list[dict[str, Any]] = []
    for variant, value in ((MAIN_VARIANT, 1.0), ("fitted_scale", temperature)):
        record: dict[str, Any] = {
            "variant": variant,
            "role": "main" if variant == MAIN_VARIANT else "ablation",
            "temperature": value,
        }
        for name in ("validation", "test"):
            probabilities = torch.softmax(logits[name] / value, dim=1)
            record.update(
                {
                    f"{name}_{key}": item
                    for key, item in probability_metrics(probabilities, splits[name].labels).items()
                }
            )
        records.append(record)
    return records


# ──────────────────────────────────────────────────────────────────────────────
#  Stages
# ──────────────────────────────────────────────────────────────────────────────


def selection_key(record: dict[str, float]) -> tuple[float, float]:
    return record["validation_nll"], record["validation_brier"]


def stage_select(args: argparse.Namespace) -> None:
    """Choose the prior gain, and sigma_v where the head needs one, by val NLL."""

    splits = load_splits(args.feature_root, args.num_classes, args.device)
    permutation = class_permutation(args.num_classes, 0, args.device)
    selection: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []

    for arm in TRAINING_ARMS:
        sigma_grid: tuple[float | None, ...] = SIGMA_V_GRID if arm.head == "hrc" else (None,)
        candidates: list[dict[str, Any]] = []
        for gain, sigma_v in itertools.product(GAIN_GRID, sigma_grid):
            started = time.perf_counter()
            classifier = train_arm(
                arm,
                splits,
                num_classes=args.num_classes,
                permutation=permutation,
                gain=gain,
                sigma_v=sigma_v,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                device=args.device,
            )
            assert classifier.hrc is not None
            mean, variance = classifier.hrc_node_moments(
                splits["validation"].features, batch_size=args.prediction_batch_size
            )
            probabilities = restore_class_order(
                hrc_log_probs(
                    mean,
                    variance,
                    classifier.hrc,
                    log_tau=LEGACY_LOG_TAU if arm.head == "hrc" else 0.0,
                ).exp(),
                permutation,
            )
            scored = probability_metrics(probabilities, splits["validation"].labels)
            candidate = {
                "arm": arm.name,
                "gain": gain,
                "sigma_v": sigma_v,
                "seconds": time.perf_counter() - started,
                **{f"validation_{key}": value for key, value in scored.items()},
            }
            candidates.append(candidate)
            trace.append(candidate)
            print(
                f"{arm.name:16s} gain={gain:<5g} sigma_v={sigma_v} "
                f"val_nll={candidate['validation_nll']:.4f} "
                f"val_acc={candidate['validation_accuracy']:.4f}",
                flush=True,
            )
        best = min(candidates, key=selection_key)
        selection[arm.name] = {"gain": best["gain"], "sigma_v": best["sigma_v"]}

    for learning_rate in SOFTMAX_LR_GRID:
        head = train_softmax(
            splits,
            num_classes=args.num_classes,
            learning_rate=learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        with torch.no_grad():
            probabilities = torch.softmax(head(splits["validation"].features).double(), 1)
        scored = probability_metrics(probabilities, splits["validation"].labels)
        candidate = {
            "arm": "softmax_ce",
            "learning_rate": learning_rate,
            **{f"validation_{key}": value for key, value in scored.items()},
        }
        trace.append(candidate)
        print(
            f"{'softmax_ce':16s} lr={learning_rate:<5g} "
            f"val_nll={candidate['validation_nll']:.4f} "
            f"val_acc={candidate['validation_accuracy']:.4f}",
            flush=True,
        )
    softmax_candidates = [item for item in trace if item["arm"] == "softmax_ce"]
    selection["softmax_ce"] = {
        "learning_rate": min(softmax_candidates, key=selection_key)["learning_rate"]
    }

    destination = args.output / f"k{args.num_classes}"
    atomic_json(destination / "selection.json", selection)
    atomic_json(destination / "selection_trace.json", trace)
    print(f"\nwrote {destination / 'selection.json'}")


def stage_run(args: argparse.Namespace) -> None:
    """Train the selected configurations across tree labellings and score them."""

    destination = args.output / f"k{args.num_classes}"
    selection_path = destination / "selection.json"
    if not selection_path.exists():
        raise SystemExit(f"run select first: {selection_path} is missing")
    selection = json.loads(selection_path.read_text())
    splits = load_splits(args.feature_root, args.num_classes, args.device)

    records: list[dict[str, Any]] = []
    for permutation_seed in args.permutation_seeds:
        permutation = class_permutation(args.num_classes, permutation_seed, args.device)
        for arm in TRAINING_ARMS:
            choice = selection[arm.name]
            started = time.perf_counter()
            classifier = train_arm(
                arm,
                splits,
                num_classes=args.num_classes,
                permutation=permutation,
                gain=choice["gain"],
                sigma_v=choice["sigma_v"],
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                device=args.device,
            )
            assert classifier.hrc is not None
            common = {
                "num_classes": args.num_classes,
                "permutation_seed": permutation_seed,
                "arm": arm.name,
                "training_rule": "probit_event" if arm.is_probit else "gaussian_pm1",
                "tree": arm.tree,
                "tree_nodes": classifier.hrc.len,
                "gain": choice["gain"],
                "sigma_v": choice["sigma_v"],
                "epochs": args.epochs,
                "seconds": time.perf_counter() - started,
            }
            for record in evaluate_hrc_variants(
                arm,
                classifier,
                splits,
                permutation,
                batch_size=args.prediction_batch_size,
            ):
                records.append({**common, **record})
                print(
                    f"perm={permutation_seed} {arm.name:16s} {record['variant']:16s} "
                    f"test_nll={records[-1]['test_nll']:.4f} "
                    f"test_acc={records[-1]['test_accuracy']:.4f} "
                    f"dev={records[-1]['test_simplex_deviation']:.2e}",
                    flush=True,
                )

        head = train_softmax(
            splits,
            num_classes=args.num_classes,
            learning_rate=selection["softmax_ce"]["learning_rate"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        for record in evaluate_softmax_variants(head, splits):
            records.append(
                {
                    "num_classes": args.num_classes,
                    "permutation_seed": permutation_seed,
                    "arm": "softmax_ce",
                    "training_rule": "backprop_ce",
                    "tree": "none",
                    "tree_nodes": args.num_classes,
                    "learning_rate": selection["softmax_ce"]["learning_rate"],
                    "epochs": args.epochs,
                    **record,
                }
            )
            print(
                f"perm={permutation_seed} {'softmax_ce':16s} {record['variant']:16s} "
                f"test_nll={records[-1]['test_nll']:.4f} "
                f"test_acc={records[-1]['test_accuracy']:.4f}",
                flush=True,
            )

    atomic_json(destination / "records.json", records)
    print(f"\nwrote {destination / 'records.json'}")


def stage_report(args: argparse.Namespace) -> None:
    """Aggregate every ``records.json`` into one CSV, averaged over labellings."""

    records: list[dict[str, Any]] = []
    for path in sorted(args.output.glob("k*/records.json")):
        records.extend(json.loads(path.read_text()))
    if not records:
        raise SystemExit(f"no records under {args.output}")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["num_classes"], record["arm"], record["variant"])
        grouped.setdefault(key, []).append(record)

    columns = [
        "num_classes",
        "role",
        "arm",
        "variant",
        "training_rule",
        "tree",
        "tree_nodes",
        "labellings",
        "log_tau",
        "alpha_impl",
        "temperature",
        "test_nll",
        "test_nll_std",
        "test_accuracy",
        "test_brier",
        "test_ece",
        "test_simplex_deviation",
        "validation_nll",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        first = group[0]

        def average(name: str, rows_in_group: list[dict[str, Any]] = group) -> float | str:
            values = [
                item[name] for item in rows_in_group if isinstance(item.get(name), (int, float))
            ]
            return sum(values) / len(values) if values else ""

        deviation = ""
        losses = [item["test_nll"] for item in group]
        if len(losses) > 1:
            centre = sum(losses) / len(losses)
            deviation = math.sqrt(
                sum((value - centre) ** 2 for value in losses) / (len(losses) - 1)
            )
        rows.append(
            {
                "num_classes": key[0],
                "role": first["role"],
                "arm": key[1],
                "variant": key[2],
                "training_rule": first["training_rule"],
                "tree": first["tree"],
                "tree_nodes": first["tree_nodes"],
                "labellings": len(group),
                "log_tau": average("log_tau"),
                "alpha_impl": average("alpha_impl"),
                "temperature": average("temperature"),
                "test_nll": average("test_nll"),
                "test_nll_std": deviation,
                "test_accuracy": average("test_accuracy"),
                "test_brier": average("test_brier"),
                "test_ece": average("test_ece"),
                "test_simplex_deviation": average("test_simplex_deviation"),
                "validation_nll": average("validation_nll"),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / "report.csv"
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(args.output / "report.json", rows)

    def cell(value: Any, width: int, spec: str) -> str:
        # The raw-score rows carry no ECE: an unnormalized row has no
        # calibration to measure.
        return format(value, spec) if isinstance(value, (int, float)) else "-".rjust(width)

    header = (
        f"{'K':>3} {'arm':16s} {'variant':16s} {'test NLL':>9} {'acc':>7} {'ECE':>7} {'|sum-1|':>9}"
    )

    def show(title: str, selected: list[dict[str, Any]]) -> None:
        print(f"\n{title}")
        print(header)
        print("-" * len(header))
        for row in selected:
            print(
                f"{row['num_classes']:>3} {row['arm']:16s} {row['variant']:16s} "
                f"{cell(row['test_nll'], 9, '>9.4f')} "
                f"{cell(row['test_accuracy'], 7, '>7.4f')} "
                f"{cell(row['test_ece'], 7, '>7.4f')} "
                f"{cell(row['test_simplex_deviation'], 9, '>9.2e')}"
            )

    show("MAIN COMPARISON", [row for row in rows if row["role"] == "main"])
    show("ABLATIONS", [row for row in rows if row["role"] != "main"])
    print(f"\nwrote {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "run", "report"])
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--permutation-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Tree labellings to average over; seed zero is the identity.",
    )
    args = parser.parse_args()
    if args.num_classes < 2:
        raise SystemExit("--num-classes must be at least two")

    {"select": stage_select, "run": stage_run, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
