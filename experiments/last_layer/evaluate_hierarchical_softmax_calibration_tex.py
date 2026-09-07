"""Evaluate the literal TeX HSM formulation on CIFAR-10 OOD and corruptions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi.classification import TAGILastLayerClassifier  # noqa: E402
from triton_tagi.hierarchical_softmax_calibration import (  # noqa: E402
    GaussianGainPosterior,
    calibrate_tex_hsm_gain_adf,
    tex_gain_groups,
    tex_hsm_class_moments,
    tex_hsm_class_probabilities,
)
from triton_tagi.metrics import (  # noqa: E402
    classification_metrics,
    evaluate_ood_comprehensive,
)

from run_hierarchical_softmax_calibration_tex import load_split, seed_everything  # noqa: E402


FEATURE_ROOT = (
    REPOSITORY_ROOT
    / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k/features/cifar10"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "runs/last_layer/hsm_calibration_tex/cifar10/node_sv0.3_20ep/ood_result.json"
)
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


def predict(
    classifier: TAGILastLayerClassifier,
    posterior: GaussianGainPosterior,
    features: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert classifier.hrc is not None
    node_moments = classifier.hrc_node_moments(features, batch_size=batch_size)
    class_moments = tex_hsm_class_moments(*node_moments, classifier.hrc, posterior)
    probabilities = tex_hsm_class_probabilities(*node_moments, classifier.hrc, posterior)
    # This is the literal raw Eq. (class) dispersion. The TeX does not derive
    # the variance after its prescribed normalization for a padded tree.
    return probabilities, class_moments.variance.sum(dim=1)


def compact_classification(probabilities: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    metrics = classification_metrics(probabilities, labels)
    return {
        key: float(metrics[key])
        for key in ("accuracy", "nll", "brier", "ece", "adaptive_ece")
    }


def macro(records: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values: list[float] = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        values.append(float(value))
    return sum(values) / len(values)


def run(args: argparse.Namespace) -> None:
    train = load_split(args.feature_root, "train", args.device)
    validation = load_split(args.feature_root, "validation", args.device)
    test = load_split(args.feature_root, "test", args.device)
    seed_everything(args.seed)
    classifier = TAGILastLayerClassifier(
        train[0].shape[1],
        10,
        head="hrc",
        hrc_tree="padded",
        gain_w=args.head_gain,
        gain_b=args.head_gain,
        sigma_v=args.sigma_v,
        device=args.device,
    )
    assert classifier.hrc is not None
    posterior = GaussianGainPosterior.prior(
        tex_gain_groups(classifier.hrc, args.sharing, device=args.device),
        mean=args.prior_gain_mean,
        variance=args.prior_gain_variance,
    )
    for epoch in range(1, args.epochs + 1):
        classifier.fit(
            *train,
            epochs=1,
            batch_size=args.batch_size,
            sigma_v=args.sigma_v,
            seed=args.seed + epoch - 1,
            record_initial=False,
        )
        validation_moments = classifier.hrc_node_moments(
            validation[0], batch_size=args.prediction_batch_size
        )
        posterior = calibrate_tex_hsm_gain_adf(
            *validation_moments,
            validation[1],
            classifier.hrc,
            sharing=args.sharing,
            prior_mean=args.prior_gain_mean,
            prior_variance=args.prior_gain_variance,
            initial_mean=posterior.mean,
        )
        print(f"trained/calibrated epoch {epoch:03d}", flush=True)

    clean_probability, clean_dispersion = predict(
        classifier, posterior, test[0], args.prediction_batch_size
    )
    svhn = load_split(args.feature_root, "svhn", args.device)
    svhn_probability, svhn_dispersion = predict(
        classifier, posterior, svhn[0], args.prediction_batch_size
    )
    semantic_ood = evaluate_ood_comprehensive(
        clean_probability,
        svhn_probability,
        epistemic_id=clean_dispersion,
        epistemic_ood=svhn_dispersion,
    )

    conditions: list[dict[str, Any]] = []
    for corruption in CORRUPTIONS:
        for severity in range(1, 6):
            name = f"{corruption}_s{severity}"
            shifted = load_split(args.feature_root / "corruptions", name, args.device)
            probability, dispersion = predict(
                classifier, posterior, shifted[0], args.prediction_batch_size
            )
            conditions.append(
                {
                    "name": name,
                    "classification": compact_classification(probability, shifted[1]),
                    "detection": evaluate_ood_comprehensive(
                        clean_probability,
                        probability,
                        epistemic_id=clean_dispersion,
                        epistemic_ood=dispersion,
                    ),
                }
            )
        print(f"scored {corruption}", flush=True)

    macro_result: dict[str, Any] = {
        "classification": {
            key: macro(conditions, ("classification", key))
            for key in ("accuracy", "nll", "brier", "ece", "adaptive_ece")
        },
        "detection": {
            score: {
                metric: macro(conditions, ("detection", score, metric))
                for metric in ("auroc", "aupr_ood", "aupr_id", "fpr95")
            }
            for score in ("entropy", "negative_max_probability", "native_epistemic")
        },
    }
    result = {
        "formulation": "literal_hierarchical_softmax_calibration_tex",
        "configuration": {
            "epochs": args.epochs,
            "head": "hrc",
            "tree": "padded_cutagi",
            "head_gain": args.head_gain,
            "sigma_v_training_only": args.sigma_v,
            "sharing": args.sharing,
            "prior_gain_mean": args.prior_gain_mean,
            "prior_gain_variance_reset_each_epoch": args.prior_gain_variance,
        },
        "gain_mean": posterior.mean.tolist(),
        "gain_variance": posterior.variance.tolist(),
        "clean": compact_classification(clean_probability, test[1]),
        "svhn": semantic_ood,
        "cifar10_c": {"macro": macro_result, "conditions": conditions},
        "native_epistemic_note": (
            "sum_c raw Var(Q_c) from Eq. (class); the TeX does not derive variance "
            "after normalizing its padded-tree class probabilities"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    temporary.replace(args.output)
    print(f"wrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head-gain", type=float, default=0.1)
    parser.add_argument("--sigma-v", type=float, default=0.3)
    parser.add_argument("--sharing", choices=("global", "level", "node"), default="node")
    parser.add_argument("--prior-gain-mean", type=float, default=0.3)
    parser.add_argument("--prior-gain-variance", type=float, default=1.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
