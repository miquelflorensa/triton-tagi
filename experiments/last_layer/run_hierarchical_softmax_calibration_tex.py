"""Run the literal hierarchical_softmax_calibration.tex algorithm on CIFAR.

This runner is intentionally separate from ``run_hsm_calibration.py``.  That
runner evaluates the later positive-log-gain/grid-posterior reformulation;
this one uses a Gaussian gain, the padded cuTAGI tree, and Algorithm 1's
sequential validation pass after every TAGI training epoch.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
from triton_tagi.metrics import classification_metrics  # noqa: E402


DEFAULT_STUDY_ROOT = REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(root: Path, name: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(root / f"{name}.pt", map_location="cpu", weights_only=False)
    return (
        payload["features"].to(device).contiguous(),
        payload["labels"].to(device).long().contiguous(),
    )


def score(
    classifier: TAGILastLayerClassifier,
    posterior: GaussianGainPosterior,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    assert classifier.hrc is not None
    mean, variance = classifier.hrc_node_moments(features, batch_size=batch_size)
    raw = tex_hsm_class_moments(mean, variance, classifier.hrc, posterior)
    probabilities = tex_hsm_class_probabilities(mean, variance, classifier.hrc, posterior)
    metrics = classification_metrics(probabilities, labels)
    return {
        key: float(metrics[key])
        for key in ("accuracy", "nll", "brier", "ece", "adaptive_ece", "classwise_ece")
    } | {
        "raw_probability_sum_max_deviation": float((raw.mean.sum(-1) - 1.0).abs().max()),
        "raw_class_probability_variance_mean": float(raw.variance.mean()),
        "note": (
            "The TeX does not derive variance after normalization for a padded tree; "
            "the reported class variance is the raw Eq. (class) quantity."
        ),
    }


def run(args: argparse.Namespace) -> None:
    feature_root = args.study_root / "features" / args.dataset
    splits = {
        name: load_split(feature_root, name, args.device)
        for name in ("train", "validation", "test")
    }
    num_classes = int(splits["train"][1].max()) + 1
    seed_everything(args.seed)
    classifier = TAGILastLayerClassifier(
        splits["train"][0].shape[1],
        num_classes,
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

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        classifier.fit(
            *splits["train"],
            epochs=1,
            batch_size=args.batch_size,
            sigma_v=args.sigma_v,
            seed=args.seed + epoch - 1,
            record_initial=False,
        )
        validation_moments = classifier.hrc_node_moments(
            splits["validation"][0], batch_size=args.prediction_batch_size
        )
        posterior = calibrate_tex_hsm_gain_adf(
            *validation_moments,
            splits["validation"][1],
            classifier.hrc,
            sharing=args.sharing,
            prior_mean=args.prior_gain_mean,
            prior_variance=args.prior_gain_variance,
            initial_mean=posterior.mean,
        )
        history.append(
            {
                "epoch": epoch,
                "gain_mean": posterior.mean.tolist(),
                "gain_variance": posterior.variance.tolist(),
                "visits": posterior.visits.tolist() if posterior.visits is not None else None,
                "validation": score(
                    classifier,
                    posterior,
                    *splits["validation"],
                    args.prediction_batch_size,
                ),
            }
        )
        print(
            f"epoch={epoch:03d} val_nll={history[-1]['validation']['nll']:.4f} "
            f"val_acc={history[-1]['validation']['accuracy']:.4f}",
            flush=True,
        )

    result = {
        "formulation": "literal_hierarchical_softmax_calibration_tex",
        "dataset": args.dataset,
        "head": "hrc",
        "tree": "padded_cutagi",
        "sigma_v_training_only": args.sigma_v,
        "gain_distribution": "Gaussian",
        "calibration": "sequential_ADF_after_each_epoch",
        "sharing": args.sharing,
        "prior_gain_mean": args.prior_gain_mean,
        "prior_gain_variance_reset_each_epoch": args.prior_gain_variance,
        "history": history,
        "test": score(
            classifier, posterior, *splits["test"], args.prediction_batch_size
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    temporary.replace(args.output)
    print(f"wrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "runs/last_layer/hsm_calibration_tex/cifar10/result.json",
    )
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
    args = parser.parse_args()
    if args.dataset != "cifar10" and str(args.output).endswith("cifar10/result.json"):
        args.output = args.output.parent.parent / args.dataset / "result.json"
    run(args)


if __name__ == "__main__":
    main()
