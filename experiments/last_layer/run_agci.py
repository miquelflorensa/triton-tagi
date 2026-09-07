"""Run fixed-tau AGCI on a cached ResNet-18 CIFAR last layer.

The predictive link can be either the canonical noisy-argmax probability or a
ReMax moment map over the AGCI-trained Gaussian utilities.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import (
    A_STAR,
    TAGILastLayerClassifier,
    agci_weight_gain_from_kappa,
    classification_metrics,
    evaluate_ood_comprehensive,
    fit_softmax_temperature,
    predictive_entropy,
)
from triton_tagi.cifar_study import load_feature_shard, seed_everything


DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k"
)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


@torch.no_grad()
def predict_batches(
    classifier: TAGILastLayerClassifier,
    features: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    probability_parts = []
    epistemic_parts = []
    for start in range(0, features.shape[0], batch_size):
        prediction = classifier.predict(features[start : start + batch_size])
        probability_parts.append(prediction.probabilities.cpu())
        epistemic_parts.append(prediction.epistemic_variance.mean(dim=-1).cpu())
    return torch.cat(probability_parts), torch.cat(epistemic_parts)


def selection_key(record: dict[str, float]) -> tuple[float, float, float]:
    return record["val_nll"], record["val_brier"], record["val_ece"]


def select_record(records: list[dict[str, float]]) -> dict[str, float]:
    best_accuracy = max(record["val_accuracy"] for record in records)
    eligible = [
        record
        for record in records
        if record["val_accuracy"] >= best_accuracy - 0.01
    ]
    return min(eligible, key=selection_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=1024)
    prior_scale = parser.add_mutually_exclusive_group()
    prior_scale.add_argument(
        "--kappa",
        type=float,
        help="target prior utility standard deviation divided by tau; defaults to 1",
    )
    prior_scale.add_argument(
        "--gain",
        type=float,
        help="explicit weight gain override instead of deriving it from kappa",
    )
    parser.add_argument(
        "--bias-gain",
        type=float,
        default=1.0,
        help="bias prior gain; its variance is bias_gain^2 / feature dimension",
    )
    parser.add_argument(
        "--no-center-features",
        dest="center_features",
        action="store_false",
        help="disable training-mean feature centering",
    )
    parser.set_defaults(center_features=True)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--num-quad", type=int, default=48)
    parser.add_argument(
        "--class-chunk-size",
        type=int,
        help="candidate classes per AGCI moment chunk; defaults to automatic",
    )
    parser.add_argument(
        "--predictive-link",
        choices=("agci", "remax"),
        default="agci",
        help="Probability map used for validation selection and evaluation.",
    )
    parser.add_argument(
        "--decision-noise",
        choices=("gaussian", "gumbel", "core_tail"),
        default="gaussian",
        help=(
            "Categorical link. Gaussian gives the multinomial-probit link; "
            "gumbel gives the exact logit link, whose class probability decays "
            "linearly rather than quadratically in the utility margin; "
            "core_tail gives the Core-Tail link, which has the logit tail and "
            "the variance-matched probit slope at a tie."
        ),
    )
    parser.add_argument(
        "--core-tail-a-star",
        type=float,
        default=A_STAR,
        help=(
            "Core-Tail interior coefficient; the default is derived from "
            "variance matching and 0 reduces the link to softmax"
        ),
    )
    parser.add_argument(
        "--gumbel-num-samples",
        type=int,
        default=32,
        help="antithetic prior draws per Gumbel-link update",
    )
    parser.add_argument(
        "--remax-approximation",
        choices=("lognormal", "laplace"),
        default="lognormal",
    )
    parser.add_argument(
        "--remax-jacobian", choices=("diag", "full"), default="diag"
    )
    parser.add_argument("--remax-num-quad", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="select and save a checkpoint without reading test or OOD features",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.tau <= 0.0 or not math.isfinite(args.tau):
        raise ValueError("tau must be finite and positive")
    if args.bias_gain < 0.0 or not math.isfinite(args.bias_gain):
        raise ValueError("bias_gain must be finite and nonnegative")

    feature_root = (
        args.feature_root
        if args.feature_root is not None
        else DEFAULT_STUDY_ROOT / "features" / args.dataset
    )
    if args.decision_noise in ("gumbel", "core_tail"):
        if args.predictive_link != "agci":
            raise ValueError(
                f"the {args.decision_noise} link supplies its own probability "
                "map and cannot be combined with --predictive-link remax"
            )
        head = "gumbel_agci" if args.decision_noise == "gumbel" else "ct_agci"
    else:
        head = "agci" if args.predictive_link == "agci" else "agci_remax"
    output = (
        args.output
        if args.output is not None
        else DEFAULT_STUDY_ROOT / "heads" / head / args.dataset
    )

    train = load_feature_shard(feature_root / "train.pt")
    validation = load_feature_shard(feature_root / "validation.pt")
    test = None if args.validation_only else load_feature_shard(feature_root / "test.pt")
    svhn = None if args.validation_only else load_feature_shard(feature_root / "svhn.pt")
    input_dim = train["features"].shape[1]
    feature_mean = (
        train["features"].float().mean(dim=0) if args.center_features else None
    )
    transformed_train = train["features"].float()
    if feature_mean is not None:
        transformed_train = transformed_train - feature_mean
    feature_energy = (
        transformed_train.square().sum(dim=1).mean() / input_dim
    ).item()
    target_kappa = 1.0 if args.kappa is None and args.gain is None else args.kappa
    # Both non-Gaussian links have a gauge-fixed unit utility scale, so kappa
    # is expressed against that rather than against tau, which they do not use.
    noise_unit = args.tau if args.decision_noise == "gaussian" else 1.0
    if args.gain is None:
        assert target_kappa is not None
        gain_w = agci_weight_gain_from_kappa(
            feature_energy,
            input_dim,
            target_kappa,
            tau=noise_unit,
            bias_gain=args.bias_gain,
        )
        prior_parameterization = "kappa"
    else:
        if args.gain < 0.0:
            raise ValueError("gain must be nonnegative")
        gain_w = args.gain
        prior_parameterization = "explicit_gain"
    expected_prior_utility_variance = (
        gain_w**2 * feature_energy + args.bias_gain**2 / input_dim
    )
    actual_kappa = math.sqrt(expected_prior_utility_variance) / noise_unit

    run_config = {
        "dataset": args.dataset,
        "head": head,
        "agci_update": "observed_event",
        "decision_noise": args.decision_noise,
        "noise_unit": noise_unit,
        "gumbel_num_samples": (
            args.gumbel_num_samples if args.decision_noise == "gumbel" else None
        ),
        "core_tail_a_star": (
            args.core_tail_a_star if args.decision_noise == "core_tail" else None
        ),
        "agci_tau": args.tau if args.decision_noise == "gaussian" else None,
        "agci_num_quad": (
            args.num_quad if args.decision_noise == "gaussian" else None
        ),
        "agci_class_chunk_size": (
            args.class_chunk_size if args.decision_noise == "gaussian" else None
        ),
        "remax_approximation": args.remax_approximation,
        "remax_jacobian": args.remax_jacobian,
        "remax_num_quad": args.remax_num_quad,
        "prior_parameterization": prior_parameterization,
        "target_kappa": target_kappa,
        "actual_kappa": actual_kappa,
        "feature_centering": args.center_features,
        "feature_scale": 1.0,
        "feature_energy": feature_energy,
        "expected_prior_utility_variance": expected_prior_utility_variance,
        "gain_w": gain_w,
        "gain_b": args.bias_gain,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "validation_only": args.validation_only,
    }
    result_path = output / "result.json"
    if result_path.exists() and not args.force:
        existing = json.loads(result_path.read_text())
        if existing.get("config") == run_config:
            print(result_path.read_text())
            return
        raise ValueError(
            f"{result_path} contains a different configuration; pass --force "
            "to replace it or choose another --output"
        )

    checkpoint_epochs = {
        epoch
        for epoch in (0, 1, 2, 5, 10, 20, 30, 50)
        if epoch <= args.epochs
    }
    checkpoint_epochs.add(args.epochs)

    seed_everything(args.seed)
    classifier = TAGILastLayerClassifier(
        input_dim,
        train["logits"].shape[1],
        head=head,
        device=args.device,
        gain_w=gain_w,
        gain_b=args.bias_gain,
        feature_mean=feature_mean,
        agci_tau=args.tau,
        agci_num_quad=args.num_quad,
        agci_class_chunk_size=args.class_chunk_size,
        # beta is the Gumbel utility unit exactly as tau is the Gaussian one,
        # so kappa keeps its meaning and beta stays at its gauge-fixed value.
        gumbel_beta=1.0,
        gumbel_num_samples=args.gumbel_num_samples,
        gumbel_seed=args.seed,
        core_tail_beta=1.0,
        core_tail_a_star=args.core_tail_a_star,
        remax_approximation=args.remax_approximation,
        remax_jacobian=args.remax_jacobian,
        remax_num_quad=args.remax_num_quad,
    )
    started = time.perf_counter()
    history = classifier.fit(
        train["features"],
        train["labels"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        validation=(validation["features"], validation["labels"]),
        validation_batch_size=args.prediction_batch_size,
        checkpoint_dir=output / "checkpoints",
        checkpoint_epochs=checkpoint_epochs,
        record_initial=not args.validation_only,
        callback=lambda epoch, _, row: print(
            f"{head} epoch={epoch:02d} acc={row['val_accuracy']:.4f} "
            f"nll={row['val_nll']:.4f} ece={row['val_ece']:.4f}",
            flush=True,
        ),
    )
    records = list(history.records)
    selected = select_record(
        [record for record in records if int(record["epoch"]) in checkpoint_epochs]
    )
    selected_epoch = int(selected["epoch"])
    selected_classifier, _ = TAGILastLayerClassifier.load(
        output / "checkpoints" / f"epoch_{selected_epoch:04d}.pt",
        device=args.device,
    )
    if args.validation_only:
        result = {
            "config": run_config,
            "selection_rule": "minimum validation NLL within 1pp of best accuracy",
            "selected_epoch": selected_epoch,
            "validation": {
                key.removeprefix("val_"): value
                for key, value in selected.items()
                if key.startswith("val_")
            },
            "history": records,
            "wall_s": time.perf_counter() - started,
        }
        atomic_json(result_path, result)
        print(json.dumps(result, indent=2))
        return

    assert test is not None and svhn is not None
    probabilities, epistemic = predict_batches(
        selected_classifier, test["features"], args.prediction_batch_size
    )
    svhn_probabilities, svhn_epistemic = predict_batches(
        selected_classifier, svhn["features"], args.prediction_batch_size
    )
    baseline_probabilities = torch.softmax(test["logits"], dim=-1)
    baseline_svhn_probabilities = torch.softmax(svhn["logits"], dim=-1)
    softmax_temperature = fit_softmax_temperature(
        validation["logits"], validation["labels"]
    )
    calibrated_validation_probabilities = torch.softmax(
        validation["logits"] / softmax_temperature, dim=-1
    )
    calibrated_baseline_probabilities = torch.softmax(
        test["logits"] / softmax_temperature, dim=-1
    )
    calibrated_baseline_svhn_probabilities = torch.softmax(
        svhn["logits"] / softmax_temperature, dim=-1
    )
    result = {
        "config": run_config,
        "selection_rule": "minimum validation NLL within 1pp of best accuracy",
        "selected_epoch": selected_epoch,
        "validation": {
            key.removeprefix("val_"): value
            for key, value in selected.items()
            if key.startswith("val_")
        },
        "test": classification_metrics(probabilities, test["labels"]),
        "test_epistemic_mean": epistemic.mean().item(),
        "baseline_test": classification_metrics(
            baseline_probabilities, test["labels"]
        ),
        "temperature_scaled_softmax": {
            "temperature": softmax_temperature,
            "validation": classification_metrics(
                calibrated_validation_probabilities, validation["labels"]
            ),
            "test": classification_metrics(
                calibrated_baseline_probabilities, test["labels"]
            ),
        },
        "ood": {
            "protocol": {
                "id_dataset": f"{args.dataset}_test",
                "ood_dataset": "svhn_test",
                "selection_uses_ood": False,
                "interpretation": "closed-set uncertainty ranking, not an OOD class",
            },
            head: evaluate_ood_comprehensive(
                probabilities,
                svhn_probabilities,
                epistemic_id=epistemic,
                epistemic_ood=svhn_epistemic,
            ),
            "softmax": evaluate_ood_comprehensive(
                baseline_probabilities, baseline_svhn_probabilities
            ),
            "temperature_scaled_softmax": evaluate_ood_comprehensive(
                calibrated_baseline_probabilities,
                calibrated_baseline_svhn_probabilities,
            ),
            "means": {
                f"{head}_clean_entropy": predictive_entropy(probabilities).mean().item(),
                f"{head}_svhn_entropy": predictive_entropy(
                    svhn_probabilities
                ).mean().item(),
                f"{head}_clean_epistemic": epistemic.mean().item(),
                f"{head}_svhn_epistemic": svhn_epistemic.mean().item(),
            },
        },
        "history": records,
        "wall_s": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
