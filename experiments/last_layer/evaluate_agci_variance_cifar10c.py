"""Evaluate decision-aware uses of AGCI variance on CIFAR-10-C.

The checkpoint and all score hyperparameters are selected without corruption
data.  Clean validation output variance defines the robust reference used by
the two-sided feature-leverage surprise score.  CIFAR-10-C is evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import TAGILastLayerClassifier, classification_metrics
from triton_tagi.cifar_study import load_feature_shard
from triton_tagi.metrics import (
    negative_max_probability,
    ood_detection_metrics_full,
    predictive_entropy,
)


DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k"
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


@torch.no_grad()
def predict_moments(
    classifier: TAGILastLayerClassifier,
    features: Tensor,
    batch_size: int,
) -> dict[str, Tensor]:
    parts: dict[str, list[Tensor]] = defaultdict(list)
    for start in range(0, features.shape[0], batch_size):
        prediction = classifier.predict(features[start : start + batch_size])
        parts["probabilities"].append(prediction.probabilities.cpu())
        parts["mean"].append(prediction.output_mean.cpu())
        parts["variance"].append(prediction.output_variance.cpu())
    return {name: torch.cat(values) for name, values in parts.items()}


def score_outputs(
    output: dict[str, Tensor],
    *,
    leverage_center: Tensor,
    leverage_scale: Tensor,
) -> dict[str, Tensor]:
    probabilities = output["probabilities"]
    mean = output["mean"]
    variance = output["variance"]
    top = mean.topk(2, dim=1)
    row = torch.arange(mean.shape[0])
    winner = top.indices[:, 0]
    runner_up = top.indices[:, 1]
    margin = top.values[:, 0] - top.values[:, 1]
    contrast_variance = variance[row, winner] + variance[row, runner_up]
    raw_variance = variance.mean(dim=1)
    log_variance = raw_variance.clamp_min(torch.finfo(raw_variance.dtype).tiny).log()
    return {
        "entropy": predictive_entropy(probabilities),
        "negative_max_probability": negative_max_probability(probabilities),
        "native_variance": raw_variance,
        "negative_native_variance": -raw_variance,
        "leverage_surprise": (log_variance - leverage_center).abs()
        / leverage_scale,
        "negative_margin": -margin,
        "variance_normalized_margin": -margin
        / contrast_variance.clamp_min(1e-30).sqrt(),
    }


def error_detection(scores: dict[str, Tensor], probabilities: Tensor, labels: Tensor):
    errors = probabilities.argmax(dim=1) != labels
    if not bool(errors.any()) or bool(errors.all()):
        return {}
    return {
        name: ood_detection_metrics_full(score[~errors], score[errors])
        for name, score in scores.items()
    }


def mean_nested(values: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    common_keys = set.intersection(*(set(value) for value in values))
    for key in sorted(common_keys):
        children = [value[key] for value in values]
        if all(isinstance(child, dict) for child in children):
            result[key] = mean_nested(children)
        elif all(isinstance(child, (int, float)) for child in children):
            result[key] = sum(float(child) for child in children) / len(children)
    return result


def evaluate_condition(
    clean_scores: dict[str, Tensor],
    output: dict[str, Tensor],
    labels: Tensor,
    *,
    leverage_center: Tensor,
    leverage_scale: Tensor,
) -> dict[str, Any]:
    scores = score_outputs(
        output,
        leverage_center=leverage_center,
        leverage_scale=leverage_scale,
    )
    return {
        "classification": classification_metrics(output["probabilities"], labels),
        "score_means": {name: value.mean().item() for name, value in scores.items()},
        "ood": {
            name: ood_detection_metrics_full(clean_scores[name], value)
            for name, value in scores.items()
        },
        "error_detection": error_detection(scores, output["probabilities"], labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint or (
        args.study_root
        / "heads/agci/cifar10/checkpoints/epoch_0010.pt"
    )
    destination = args.output or (
        args.study_root / "heads/agci/cifar10/cifar10c_variance_scores.json"
    )
    if destination.exists() and not args.force:
        print(destination.read_text())
        return

    feature_root = args.study_root / "features/cifar10"
    validation = load_feature_shard(feature_root / "validation.pt")
    clean = load_feature_shard(feature_root / "test.pt")
    classifier, metadata = TAGILastLayerClassifier.load(
        checkpoint, device=args.device
    )

    validation_output = predict_moments(
        classifier, validation["features"], args.batch_size
    )
    validation_log_variance = validation_output["variance"].mean(1).clamp_min(
        torch.finfo(validation_output["variance"].dtype).tiny
    ).log()
    leverage_center = validation_log_variance.median()
    leverage_scale = (validation_log_variance - leverage_center).abs().median()
    leverage_scale = leverage_scale.clamp_min(1e-12)

    clean_output = predict_moments(classifier, clean["features"], args.batch_size)
    clean_scores = score_outputs(
        clean_output,
        leverage_center=leverage_center,
        leverage_scale=leverage_scale,
    )
    clean_result = {
        "classification": classification_metrics(
            clean_output["probabilities"], clean["labels"]
        ),
        "score_means": {
            name: value.mean().item() for name, value in clean_scores.items()
        },
        "error_detection": error_detection(
            clean_scores, clean_output["probabilities"], clean["labels"]
        ),
    }

    conditions = []
    for index, path in enumerate(sorted((feature_root / "corruptions").glob("*.pt")), 1):
        shard = load_feature_shard(path)
        output = predict_moments(classifier, shard["features"], args.batch_size)
        result = evaluate_condition(
            clean_scores,
            output,
            shard["labels"],
            leverage_center=leverage_center,
            leverage_scale=leverage_scale,
        )
        metadata_corruption = shard.get("metadata", {})
        condition = {
            "name": path.stem,
            "corruption": metadata_corruption.get(
                "corruption", path.stem.rsplit("_s", 1)[0]
            ),
            "severity": int(
                metadata_corruption.get("severity", path.stem.rsplit("_s", 1)[1])
            ),
            **result,
        }
        conditions.append(condition)
        print(
            f"[{index:02d}/75] {path.stem}: "
            f"acc={result['classification']['accuracy']:.4f} "
            f"entropy_AUROC={result['ood']['entropy']['auroc']:.4f} "
            f"decision_AUROC={result['ood']['variance_normalized_margin']['auroc']:.4f} "
            f"leverage_AUROC={result['ood']['leverage_surprise']['auroc']:.4f}",
            flush=True,
        )

    metric_payloads = [
        {key: value for key, value in condition.items() if isinstance(value, dict)}
        for condition in conditions
    ]
    by_severity = {
        str(severity): mean_nested(
            [
                {key: value for key, value in condition.items() if isinstance(value, dict)}
                for condition in conditions
                if condition["severity"] == severity
            ]
        )
        for severity in range(1, 6)
    }
    corruption_names = sorted({condition["corruption"] for condition in conditions})
    by_corruption = {
        name: mean_nested(
            [
                {key: value for key, value in condition.items() if isinstance(value, dict)}
                for condition in conditions
                if condition["corruption"] == name
            ]
        )
        for name in corruption_names
    }
    result = {
        "config": {
            "model": "agci_last_layer",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_metadata": metadata,
            "batch_size": args.batch_size,
            "conditions": len(conditions),
            "selection_uses_corruptions": False,
            "leverage_reference": "clean_validation_log_variance_median_and_MAD",
            "leverage_center": leverage_center.item(),
            "leverage_scale": leverage_scale.item(),
            "decision_score": "-(top1_mean-top2_mean)/sqrt(top1_variance+top2_variance)",
        },
        "clean": clean_result,
        "cifar10_c": {
            "conditions": conditions,
            "macro": mean_nested(metric_payloads),
            "by_severity": by_severity,
            "by_corruption": by_corruption,
        },
    }
    if len(conditions) != 75:
        raise RuntimeError(f"expected 75 CIFAR-10-C conditions, found {len(conditions)}")
    if not math.isfinite(result["cifar10_c"]["macro"]["classification"]["nll"]):
        raise RuntimeError("non-finite macro result")
    atomic_json(destination, result)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
