"""Evaluate the five-seed exact unit-probit HRC checkpoints on OOD data.

This is a focused continuation of the frozen-feature CIFAR-10 study. It reuses
the validation-selected checkpoints listed in ``five_seed_results.json`` and
the cached clean, SVHN, and CIFAR-10-C representations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import torch

from triton_tagi.classification import TAGILastLayerClassifier
from triton_tagi.cifar_study import load_feature_shard
from triton_tagi.metrics import classification_metrics, evaluate_ood_comprehensive


ROOT = Path("runs/last_layer/cifar_frozen_last_layer_v2_40k10k")
FEATURE_ROOT = ROOT / "features" / "cifar10"
HEAD_ROOT = (
    ROOT
    / "heads"
    / "unit_probit_5seed"
    / "cifar10"
    / "hrc_probit"
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def predict_batches(
    classifier: TAGILastLayerClassifier,
    features: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities, epistemic = [], []
    for start in range(0, features.shape[0], batch_size):
        prediction = classifier.predict(features[start : start + batch_size])
        probabilities.append(prediction.probabilities.cpu())
        if prediction.epistemic_variance is None:
            raise RuntimeError("hrc_probit did not expose epistemic variance")
        epistemic.append(
            prediction.epistemic_variance.reshape(
                prediction.epistemic_variance.shape[0], -1
            )
            .mean(1)
            .cpu()
        )
    return torch.cat(probabilities), torch.cat(epistemic)


def evaluate_pair(
    probabilities_id: torch.Tensor,
    labels_id: torch.Tensor,
    epistemic_id: torch.Tensor,
    probabilities_ood: torch.Tensor,
    epistemic_ood: torch.Tensor,
    labels_ood: torch.Tensor | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ood": evaluate_ood_comprehensive(
            probabilities_id,
            probabilities_ood,
            epistemic_id=epistemic_id,
            epistemic_ood=epistemic_ood,
        )
    }
    if labels_ood is not None:
        result["shift_classification"] = classification_metrics(
            probabilities_ood, labels_ood
        )
    return result


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}_{key}" if prefix else key
            result.update(flatten_numeric(child, name))
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        result[prefix] = float(value)
    return result


def mean_records(records: list[dict[str, Any]]) -> dict[str, float]:
    flattened = [flatten_numeric(record) for record in records]
    common = set.intersection(*(set(record) for record in flattened))
    return {key: fmean(record[key] for record in flattened) for key in sorted(common)}


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    flattened = [flatten_numeric(record) for record in records]
    common = set.intersection(*(set(record) for record in flattened))
    summary: dict[str, dict[str, float]] = {}
    for key in sorted(common):
        values = [record[key] for record in flattened]
        summary[key] = {
            "mean": fmean(values),
            "sample_std": stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--skip-corruptions",
        action="store_true",
        help="Only evaluate clean CIFAR-10 against SVHN.",
    )
    args = parser.parse_args()

    selection = json.loads((HEAD_ROOT / "five_seed_results.json").read_text())
    clean = load_feature_shard(FEATURE_ROOT / "test.pt")
    svhn = load_feature_shard(FEATURE_ROOT / "svhn.pt")
    corruption_paths = (
        []
        if args.skip_corruptions
        else sorted((FEATURE_ROOT / "corruptions").glob("*.pt"))
    )

    runs: list[dict[str, Any]] = []
    for selected in selection["runs"]:
        checkpoint = Path(selected["checkpoint"])
        classifier, metadata = TAGILastLayerClassifier.load(
            checkpoint, device=args.device
        )
        probabilities_id, epistemic_id = predict_batches(
            classifier, clean["features"], args.batch_size
        )
        probabilities_svhn, epistemic_svhn = predict_batches(
            classifier, svhn["features"], args.batch_size
        )
        run: dict[str, Any] = {
            "seed": selected["seed"],
            "checkpoint": str(checkpoint),
            "metadata": metadata,
            "clean_classification": classification_metrics(
                probabilities_id, clean["labels"]
            ),
            "svhn": evaluate_pair(
                probabilities_id,
                clean["labels"],
                epistemic_id,
                probabilities_svhn,
                epistemic_svhn,
            ),
            "corruptions": {},
        }
        for shard_path in corruption_paths:
            shard = load_feature_shard(shard_path)
            probabilities_ood, epistemic_ood = predict_batches(
                classifier, shard["features"], args.batch_size
            )
            run["corruptions"][shard_path.stem] = evaluate_pair(
                probabilities_id,
                clean["labels"],
                epistemic_id,
                probabilities_ood,
                epistemic_ood,
                shard["labels"],
            )
        if run["corruptions"]:
            run["corruption_macro"] = mean_records(
                list(run["corruptions"].values())
            )
        runs.append(run)
        print(f"evaluated seed {selected['seed']}")

    payload: dict[str, Any] = {
        "head": "hrc_probit",
        "formulation": "exact half-space moments with structural unit probit scale",
        "selection": selection["selection_rule"],
        "runs": runs,
        "svhn_summary": summarize([run["svhn"] for run in runs]),
        "clean_summary": summarize(
            [run["clean_classification"] for run in runs]
        ),
    }
    if corruption_paths:
        payload["corruption_count"] = len(corruption_paths)
        payload["corruption_macro_summary"] = summarize(
            [run["corruption_macro"] for run in runs]
        )
    destination = HEAD_ROOT / "ood_results.json"
    atomic_json(destination, payload)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
