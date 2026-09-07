"""Out-of-distribution detection for the calibrated hierarchical-probit head.

Continues ``run_hsm_calibration.py`` onto the shifted and out-of-distribution
splits. The head, the calibration protocol, and the arms are identical; only
the evaluation sets are added, so any difference is attributable to the gain.

Three scores are ranked, declared before the run and all reported:

  * ``entropy``, the Shannon entropy of ``E[Q]``.
  * ``negative_max_probability``, the standard maximum-softmax-probability
    baseline written as a score that grows with uncertainty.
  * ``native_epistemic``, ``sum_c Var(Q_c)``, which exists only because the
    gain carries a posterior. It is zero-information for a deterministic gain
    up to the state variance the frozen head already had, so a gain that helps
    detection has to show up here.

Evaluation sets are SVHN, which is a different dataset, and CIFAR-10-C, which
is the same classes under corruption. Corrupted data is scored two ways: as a
shift, where accuracy and calibration are what matter, and as a detection
target, which is the harder and less natural reading of it.

Examples:
  python experiments/last_layer/evaluate_hsm_ood.py run
  python experiments/last_layer/evaluate_hsm_ood.py run --severities 1 3 5
  python experiments/last_layer/evaluate_hsm_ood.py report
"""

from __future__ import annotations

import argparse
import csv
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
    LogGainPosterior,
    evaluate_ood_comprehensive,
    expected_calibration_error,
    fit_hsm_log_gain,
    hsm_class_moments,
)

from run_hsm_calibration import (  # noqa: E402
    CONVENTION_ALPHA,
    HEAD_GAIN,
    LATENT_SCALES,
    PREDICTION_CHUNK,
    PRIOR_LOG_GAIN_VARIANCE,
    QUADRATURE_ORDER,
    Split,
    atomic_json,
    calibration_subset,
    class_permutation,
    channel_sigma_v,
    load_splits,
    restore_class_order,
    train_probit_head,
    uncalibrated_posterior,
)

DEFAULT_FEATURE_ROOT = (
    REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k/features/cifar10"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/last_layer/hsm_calibration/cifar10_ood"

CORRUPTIONS = (
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "snow",
    "zoom_blur",
)
CALIBRATION_SIZES = (100, 1000, 10000)
SUBSAMPLE_SEEDS = (0, 1, 2)
SCORES = ("entropy", "negative_max_probability", "native_epistemic")


@dataclass(frozen=True)
class Arm:
    """One prediction-time gain, named as in ``run_hsm_calibration.py``."""

    name: str
    sharing: str | None
    gain_uncertainty: bool


ARMS = (
    Arm("uncalibrated", None, False),
    Arm("hsm_global", "global", True),
    Arm("hsm_level", "level", True),
    Arm("hsm_node", "node", True),
    Arm("hsm_global_point", "global", False),
    Arm("hsm_level_point", "level", False),
    Arm("hsm_node_point", "node", False),
)


# ──────────────────────────────────────────────────────────────────────────────
#  Prediction
# ──────────────────────────────────────────────────────────────────────────────


def load_shifted(feature_root: Path, severities: tuple[int, ...], device: str) -> dict[str, Split]:
    """Load SVHN and the requested CIFAR-10-C corruption shards."""

    shards: dict[str, Split] = {}
    payload = torch.load(feature_root / "svhn.pt", map_location="cpu", weights_only=False)
    shards["svhn"] = Split(
        payload["features"].to(device).contiguous(),
        payload["labels"].to(device).long().contiguous(),
    )
    for corruption in CORRUPTIONS:
        for severity in severities:
            path = feature_root / "corruptions" / f"{corruption}_s{severity}.pt"
            if not path.exists():
                raise SystemExit(f"missing corruption shard {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            shards[f"{corruption}_s{severity}"] = Split(
                payload["features"].to(device).contiguous(),
                payload["labels"].to(device).long().contiguous(),
            )
    return shards


@dataclass(frozen=True)
class Prediction:
    """Class probabilities and the total epistemic dispersion of one split."""

    probabilities: Tensor
    dispersion: Tensor


def predict(
    moments: tuple[Tensor, Tensor],
    hrc,
    posterior: LogGainPosterior,
    permutation: Tensor,
) -> Prediction:
    """Return calibrated class probabilities and ``sum_c Var(Q_c)``."""

    class_moments = hsm_class_moments(
        *moments, hrc, posterior, order=QUADRATURE_ORDER, chunk_size=PREDICTION_CHUNK
    )
    return Prediction(
        probabilities=restore_class_order(class_moments.mean, permutation),
        dispersion=class_moments.variance.sum(dim=1),
    )


def accuracy_metrics(prediction: Prediction, labels: Tensor) -> dict[str, float]:
    """Return the four columns this study reports, and nothing else.

    ``classification_metrics`` also computes adaptive and classwise ECE, whose
    cost grows with the class count; at K = 100 that is fifteen times the cost
    of what is reported here, once per corruption per arm.
    """

    probabilities = prediction.probabilities
    target = labels.to(probabilities.device).long()
    one_hot = torch.zeros_like(probabilities).scatter_(1, target[:, None], 1.0)
    picked = probabilities.gather(1, target[:, None]).squeeze(1)
    return {
        "accuracy": float(probabilities.argmax(dim=1).eq(target).double().mean()),
        "nll": float(-picked.clamp_min(1e-12).log().mean()),
        "brier": float(((probabilities - one_hot) ** 2).sum(dim=1).mean()),
        "ece": expected_calibration_error(probabilities, target),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Stages
# ──────────────────────────────────────────────────────────────────────────────


def stage_run(args: argparse.Namespace) -> None:
    """Score every arm on the clean, corrupted, and out-of-distribution splits."""

    splits = load_splits(args.feature_root, args.device)
    shifted = load_shifted(args.feature_root, tuple(args.severities), args.device)
    sigma_v = channel_sigma_v(args)
    num_classes = int(splits["train"].labels.max()) + 1
    validation_rows = splits["validation"].features.shape[0]
    sizes = [size for size in args.calibration_sizes if size <= validation_rows]

    records: list[dict[str, Any]] = []
    for permutation_seed in args.permutation_seeds:
        permutation = class_permutation(num_classes, permutation_seed, args.device)
        started = time.perf_counter()
        classifier = train_probit_head(
            splits,
            permutation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            head_gain=args.head_gain,
            head=args.head,
            sigma_v=args.sigma_v,
        )
        assert classifier.hrc is not None
        hrc = classifier.hrc
        moments = {
            name: classifier.hrc_node_moments(
                split.features, batch_size=args.prediction_batch_size
            )
            for name, split in list(splits.items()) + list(shifted.items())
            if name != "train"
        }
        print(
            f"perm={permutation_seed} trained and cached "
            f"{len(moments)} splits in {time.perf_counter() - started:.0f}s",
            flush=True,
        )

        for size in sizes:
            for subsample_seed in SUBSAMPLE_SEEDS if size < validation_rows else (0,):
                index = calibration_subset(size, validation_rows, subsample_seed).to(args.device)
                calibration_moments = tuple(
                    tensor.index_select(0, index) for tensor in moments["validation"]
                )
                calibration_labels = permutation[splits["validation"].labels[index]]
                for arm in ARMS:
                    if arm.sharing is None:
                        posterior = uncalibrated_posterior(hrc, sigma_v, args.device)
                    else:
                        posterior = fit_hsm_log_gain(
                            *calibration_moments,
                            calibration_labels,
                            hrc,
                            sharing=arm.sharing,
                            prior_mean=0.0,
                            prior_variance=PRIOR_LOG_GAIN_VARIANCE,
                            sigma_v=sigma_v,
                            chunk_size=PREDICTION_CHUNK,
                        )
                        if not arm.gain_uncertainty:
                            posterior = posterior.deterministic()

                    clean = predict(moments["test"], hrc, posterior, permutation)
                    record: dict[str, Any] = {
                        "permutation_seed": permutation_seed,
                        "subsample_seed": subsample_seed,
                        "calibration_size": int(index.shape[0]),
                        "arm": arm.name,
                        "sharing": arm.sharing or "none",
                        "gain_uncertainty": arm.gain_uncertainty,
                        "head": args.head,
                        "head_gain": args.head_gain,
                        "sigma_v": args.sigma_v,
                        "latent_scale": args.latent_scale,
                        "channel_sigma_v": sigma_v,
                        "convention_alpha": 1.0 / sigma_v,
                        "prior_log_gain_mean": 0.0,
                        "prior_log_gain_variance": PRIOR_LOG_GAIN_VARIANCE,
                        "mean_dispersion_clean": float(clean.dispersion.mean()),
                    }
                    clean_scored = accuracy_metrics(clean, splits["test"].labels)
                    record.update({f"clean_{key}": value for key, value in clean_scored.items()})

                    for name, split in shifted.items():
                        target = predict(moments[name], hrc, posterior, permutation)
                        detection = evaluate_ood_comprehensive(
                            clean.probabilities,
                            target.probabilities,
                            epistemic_id=clean.dispersion,
                            epistemic_ood=target.dispersion,
                        )
                        for score in SCORES:
                            for metric, value in detection[score].items():
                                record[f"{name}_{score}_{metric}"] = value
                        record[f"{name}_mean_dispersion"] = float(target.dispersion.mean())
                        # SVHN has no shared label space with CIFAR-10, so only
                        # the corrupted splits get accuracy and calibration.
                        if name != "svhn":
                            for key, value in accuracy_metrics(target, split.labels).items():
                                record[f"{name}_{key}"] = value
                    records.append(record)
                    print(
                        f"perm={permutation_seed} n_cal={index.shape[0]:>5} "
                        f"sub={subsample_seed} {arm.name:18s} "
                        f"svhn_auroc(ent)={record['svhn_entropy_auroc']:.4f} "
                        f"svhn_auroc(eps)={record['svhn_native_epistemic_auroc']:.4f}",
                        flush=True,
                    )
        atomic_json(args.output / "records.json", records)
    print(f"\nwrote {args.output / 'records.json'}")


def mean_of(records: list[dict[str, Any]], key: str) -> float | str:
    values = [item[key] for item in records if isinstance(item.get(key), (int, float))]
    return sum(values) / len(values) if values else ""


def corruption_mean(
    records: list[dict[str, Any]],
    suffix: str,
    severities: list[int],
) -> float | str:
    """Average one metric over every corruption and severity."""

    values: list[float] = []
    for corruption in CORRUPTIONS:
        for severity in severities:
            values.extend(
                item[f"{corruption}_s{severity}_{suffix}"]
                for item in records
                if isinstance(item.get(f"{corruption}_s{severity}_{suffix}"), (int, float))
            )
    return sum(values) / len(values) if values else ""


def stage_report(args: argparse.Namespace) -> None:
    """Aggregate over labellings, subsamples, corruptions, and severities."""

    path = args.output / "records.json"
    if not path.exists():
        raise SystemExit(f"run first: {path} is missing")
    records = json.loads(path.read_text())
    severities = sorted(
        {
            int(key.rsplit("_s", 1)[1].split("_")[0])
            for key in records[0]
            if "_s" in key and key.split("_s")[0] in CORRUPTIONS
        }
    )

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["calibration_size"], record["arm"]), []).append(record)

    rows: list[dict[str, Any]] = []
    for (size, arm), group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "calibration_size": size,
            "arm": arm,
            "sharing": group[0]["sharing"],
            "gain_uncertainty": group[0]["gain_uncertainty"],
            "records": len(group),
            "clean_accuracy": mean_of(group, "clean_accuracy"),
            "clean_nll": mean_of(group, "clean_nll"),
            "clean_ece": mean_of(group, "clean_ece"),
            "corrupt_accuracy": corruption_mean(group, "accuracy", severities),
            "corrupt_nll": corruption_mean(group, "nll", severities),
            "corrupt_ece": corruption_mean(group, "ece", severities),
            "mean_dispersion_clean": mean_of(group, "mean_dispersion_clean"),
        }
        for score in SCORES:
            for metric in ("auroc", "fpr95", "aupr_ood"):
                row[f"svhn_{score}_{metric}"] = mean_of(group, f"svhn_{score}_{metric}")
                row[f"corrupt_{score}_{metric}"] = corruption_mean(
                    group, f"{score}_{metric}", severities
                )
        rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / "report.csv"
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(args.output / "report.json", rows)

    def cell(value: Any, width: int, spec: str) -> str:
        return format(value, spec) if isinstance(value, (int, float)) else "-".rjust(width)

    header = (
        f"{'n_cal':>6} {'arm':18s} {'acc':>7} {'NLL':>7} {'ECE':>7} | "
        f"{'ent':>7} {'mxp':>7} {'eps':>7} | {'ent':>7} {'mxp':>7} {'eps':>7}"
    )
    dataset = args.feature_root.name
    print(
        f"\nCLEAN {dataset} (left) | SVHN AUROC (middle) | "
        f"{dataset}-C AUROC (right)"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['calibration_size']:>6} {row['arm']:18s} "
            f"{cell(row['clean_accuracy'], 7, '>7.4f')} "
            f"{cell(row['clean_nll'], 7, '>7.4f')} "
            f"{cell(row['clean_ece'], 7, '>7.4f')} | "
            f"{cell(row['svhn_entropy_auroc'], 7, '>7.4f')} "
            f"{cell(row['svhn_negative_max_probability_auroc'], 7, '>7.4f')} "
            f"{cell(row['svhn_native_epistemic_auroc'], 7, '>7.4f')} | "
            f"{cell(row['corrupt_entropy_auroc'], 7, '>7.4f')} "
            f"{cell(row['corrupt_negative_max_probability_auroc'], 7, '>7.4f')} "
            f"{cell(row['corrupt_native_epistemic_auroc'], 7, '>7.4f')}"
        )
    print(f"\nwrote {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run", "report"])
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head-gain", type=float, default=HEAD_GAIN)
    parser.add_argument("--head", default="hrc_probit", choices=sorted(CONVENTION_ALPHA))
    parser.add_argument("--sigma-v", type=float, default=None)
    parser.add_argument("--latent-scale", default="sigma_v", choices=LATENT_SCALES)
    parser.add_argument("--severities", type=int, nargs="+", default=[5])
    parser.add_argument("--calibration-sizes", type=int, nargs="+", default=list(CALIBRATION_SIZES))
    parser.add_argument("--permutation-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    {"run": stage_run, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
