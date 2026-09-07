"""Evaluate a Bayesian feature-support gate in front of multinomial-probit ADF.

The gate is fitted exclusively on the predeclared CIFAR-10 training features.
Neither SVHN nor CIFAR-10-C is used for fitting, selection, calibration, or
thresholding. The experiment has no statistical tuning parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import (  # noqa: E402
    BayesianFeatureSupportGate,
    TAGILastLayerClassifier,
    classification_metrics,
    negative_max_probability,
    ood_detection_metrics_full,
    predictive_entropy,
)
from triton_tagi.cifar_study import load_feature_shard  # noqa: E402


DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k"
)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def mean_nested(records: list[dict]) -> dict:
    if not records:
        return {}
    result = {}
    for key, value in records[0].items():
        if isinstance(value, dict):
            result[key] = mean_nested([record[key] for record in records])
        elif isinstance(value, (int, float)):
            result[key] = sum(record[key] for record in records) / len(records)
    return result


@torch.no_grad()
def predict_batches(
    classifier: TAGILastLayerClassifier,
    gate: BayesianFeatureSupportGate,
    features: torch.Tensor,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    parts: dict[str, list[torch.Tensor]] = {
        "conditional_probabilities": [],
        "open_set_probabilities": [],
        "negative_log_id_evidence": [],
        "negative_log_bayes_factor": [],
        "ood_probability": [],
    }
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size]
        conditional = classifier.predict(batch).probabilities
        support = gate.predict(batch, conditional)
        assert support.probabilities is not None
        parts["conditional_probabilities"].append(conditional.cpu())
        parts["open_set_probabilities"].append(support.probabilities.cpu())
        parts["negative_log_id_evidence"].append(-support.log_id_evidence.cpu())
        parts["negative_log_bayes_factor"].append(-support.log_bayes_factor.cpu())
        parts["ood_probability"].append(support.ood_probability.cpu())
    return {key: torch.cat(value) for key, value in parts.items()}


def ood_metrics(
    clean: dict[str, torch.Tensor],
    shifted: dict[str, torch.Tensor],
) -> dict:
    conditional_clean = clean["conditional_probabilities"]
    conditional_shifted = shifted["conditional_probabilities"]
    result = {
        "adf_entropy": ood_detection_metrics_full(
            predictive_entropy(conditional_clean),
            predictive_entropy(conditional_shifted),
        ),
        "adf_negative_max_probability": ood_detection_metrics_full(
            negative_max_probability(conditional_clean),
            negative_max_probability(conditional_shifted),
        ),
    }
    for score in (
        "negative_log_id_evidence",
        "negative_log_bayes_factor",
        "ood_probability",
    ):
        result[score] = ood_detection_metrics_full(clean[score], shifted[score])
    return result


def resolve_checkpoint(study_root: Path) -> tuple[Path, dict]:
    head_root = study_root / "heads/multinomial_probit_adf/cifar10"
    summary = json.loads((head_root / "summary.json").read_text())
    winner = summary["winner"]
    config = winner["config"]
    run_name = (
        f"tau2_{config['probit_tau2']:g}_gain_{config['gain_w']:g}_"
        f"seed_{config['seed']}_epochs_{config['epochs']}"
    )
    checkpoint = (
        head_root
        / run_name
        / "checkpoints"
        / f"epoch_{winner['selected_epoch']:04d}.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"selected ADF checkpoint does not exist: {checkpoint}")
    return checkpoint, winner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--svhn-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.study_root / "feature_support_gate/multinomial_probit_adf"
    result_path = output / "results.json"
    state_path = output / "gate.pt"
    if result_path.exists() and not args.force:
        raise FileExistsError(f"result already exists: {result_path}; use --force")

    feature_root = args.study_root / "features/cifar10"
    train = load_feature_shard(feature_root / "train.pt")
    clean_shard = load_feature_shard(feature_root / "test.pt")
    svhn_shard = load_feature_shard(feature_root / "svhn.pt")
    checkpoint, winner = resolve_checkpoint(args.study_root)
    classifier, _ = TAGILastLayerClassifier.load(checkpoint, device=args.device)

    started = time.perf_counter()
    gate = BayesianFeatureSupportGate.fit(
        train["features"], train["labels"], device=args.device
    )
    fit_wall_s = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {key: value.cpu() for key, value in gate.state_dict().items()},
        state_path,
    )

    clean = predict_batches(classifier, gate, clean_shard["features"], args.batch_size)
    svhn = predict_batches(classifier, gate, svhn_shard["features"], args.batch_size)
    result = {
        "config": {
            "model": "objective_bayes_student_t_feature_support",
            "known_hypothesis": "class_conditional_full_covariance_mixture",
            "background_hypothesis": "class_agnostic_isotropic",
            "domain_prior": "equal_odds",
            "fit_split": "cifar10_train_only",
            "selection_uses_ood": False,
            "statistical_hyperparameters": 0,
            "adf_checkpoint": str(checkpoint.resolve()),
            "adf_winner": winner,
        },
        "fit_wall_s": fit_wall_s,
        "clean_conditional": classification_metrics(
            clean["conditional_probabilities"], clean_shard["labels"]
        ),
        "clean_open_set": classification_metrics(
            clean["open_set_probabilities"], clean_shard["labels"]
        ),
        "svhn": ood_metrics(clean, svhn),
        "score_summaries": {
            score: {
                "clean_mean": clean[score].mean().item(),
                "svhn_mean": svhn[score].mean().item(),
            }
            for score in (
                "negative_log_id_evidence",
                "negative_log_bayes_factor",
                "ood_probability",
            )
        },
    }
    atomic_json(result_path, result)
    print(
        f"SVHN raw-evidence AUROC="
        f"{result['svhn']['negative_log_id_evidence']['auroc']:.4f} "
        f"Bayes-factor AUROC="
        f"{result['svhn']['negative_log_bayes_factor']['auroc']:.4f}",
        flush=True,
    )

    if not args.svhn_only:
        corruption_results = []
        corruption_root = feature_root / "corruptions"
        corruption_paths = sorted(corruption_root.glob("*.pt"))
        for index, path in enumerate(corruption_paths, 1):
            shard = load_feature_shard(path)
            prediction = predict_batches(
                classifier, gate, shard["features"], args.batch_size
            )
            row = {
                "name": path.stem,
                "conditional_classification": classification_metrics(
                    prediction["conditional_probabilities"], shard["labels"]
                ),
                "open_set_classification": classification_metrics(
                    prediction["open_set_probabilities"], shard["labels"]
                ),
                "ood": ood_metrics(clean, prediction),
            }
            corruption_results.append(row)
            result["cifar10_c"] = {
                "macro": mean_nested(corruption_results),
                "conditions": corruption_results,
            }
            atomic_json(result_path, result)
            print(
                f"CIFAR-10-C {index:02d}/{len(corruption_paths)} {path.stem} "
                f"support_AUROC="
                f"{row['ood']['negative_log_id_evidence']['auroc']:.4f}",
                flush=True,
            )

    result["wall_s"] = time.perf_counter() - started
    atomic_json(result_path, result)
    print(f"saved {result_path}")
    print(f"saved {state_path}")


if __name__ == "__main__":
    main()
