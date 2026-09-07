"""Logit-space TAGI-V distillation on frozen CIFAR features.

The head regresses the frozen backbone's own logits instead of the one-hot
label, and learns the observation variance of that regression with TAGI-V. The
question the runner answers is whether the learned, input-dependent logit noise
is worth its parameters once a post-hoc temperature is available, which is the
strong one-parameter calibration baseline.

Two target regimes are run, because they differ in whether the observation
noise exists at all:

``clean``
    Targets are the backbone's logits on the same evaluation-transform image
    whose features the head reads. The student's hypothesis class contains the
    teacher's own last layer, so the residual is near zero by construction and
    the variance head should collapse toward its floor. This is the control.

``augmented``
    Targets are the backbone's logits under the training augmentation while the
    features stay clean. The residual is then the genuine spread a random crop
    and flip induce in the teacher's logits, an input-dependent quantity that
    can be measured directly on held-out data. Comparing the learned variance
    against that empirical spread is the falsifiable part: if TAGI-V's variance
    head cannot recover a noise level that is measurable by resampling, the
    formulation is not doing what it claims.

The comparison holds everything else fixed and reports the same test metrics
for the frozen teacher, the teacher after temperature scaling, and a label
trained ``categorical_tagiv`` head of identical width.

Examples:
  python experiments/last_layer/run_logit_tagiv.py cache --repeats 8
  python experiments/last_layer/run_logit_tagiv.py run --seeds 0 1 2
  python experiments/last_layer/run_logit_tagiv.py report
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset
from torchvision import transforms

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import (  # noqa: E402
    LOGIT_VARIANCE_FLOOR,
    TAGILastLayerClassifier,
    classification_metrics,
    fit_logit_calibration,
    fit_softmax_temperature,
    logit_feature_energy,
    logit_replicate_variance,
    logit_tagiv_predictive_probs,
    logit_tagiv_uncertainty,
    logit_variance_head_prior,
    logit_variance_prior_split,
    negative_max_probability,
    ood_detection_metrics_full,
    predictive_entropy,
    prepare_logit_targets,
    replicate_log_variance_noise,
    replicate_log_variance_offset,
)
from triton_tagi.cifar_study import (  # noqa: E402
    CANONICAL_CORRUPTIONS,
    CifarCorruption,
    CifarResNet18,
    clean_datasets,
    extract_frozen_features,
    get_spec,
    load_feature_shard,
    seed_everything,
    sha256_file,
    training_transform,
)

DEFAULT_STUDY_ROOT = REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/last_layer/logit_tagiv"
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"
REGIMES = ("clean", "augmented")
SPLIT_SEED = 2026
CALIBRATION_SIZE = 5000
REPORTED_METRICS = ("accuracy", "nll", "ece", "brier")
VARIANCE_CONTROLS = ("learned", "constant", "shuffled")
SHUFFLE_SEED = 917


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def feature_root(study_root: Path, dataset: str) -> Path:
    return study_root / "features" / dataset


def augmented_path(output: Path, dataset: str) -> Path:
    return output / dataset / "augmented_logits.pt"


def head_path(output: Path, dataset: str, regime: str, seed: int) -> Path:
    return output / dataset / "heads" / f"{regime}_seed{seed:02d}.pt"


# ======================================================================
#  Augmented teacher logits
# ======================================================================


def cache_augmented_logits(args: argparse.Namespace) -> None:
    """Store repeated teacher logits under the training augmentation.

    The features stay on the evaluation transform, so each repeat is another
    draw of the observation noise the head is asked to model, and their
    empirical variance is the ground truth the learned variance is scored
    against.
    """

    root = feature_root(args.study_root, args.dataset)
    shards = {name: load_feature_shard(root / f"{name}.pt") for name in ("train", "validation")}
    checkpoint = args.study_root / "backbones" / f"{args.dataset}_resnet18.pt"
    expected = shards["train"]["metadata"].get("checkpoint_sha256")
    observed = sha256_file(checkpoint)
    if expected is not None and observed != expected:
        raise ValueError(
            "the cached features and the backbone checkpoint disagree; re-cache "
            "the study features before extracting augmented logits"
        )

    model = CifarResNet18(shards["train"]["logits"].shape[1])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    augmented, _ = clean_datasets(args.dataset, args.data_root, augment_train=True, download=False)

    payload: dict[str, torch.Tensor] = {}
    for name, shard in shards.items():
        indices = shard["metadata"]["indices"]
        subset = Subset(augmented, indices)
        repeats = []
        for repeat in range(args.repeats):
            seed_everything(args.seed + repeat)
            extracted = extract_frozen_features(
                model, subset, device=args.device, batch_size=args.batch_size, workers=args.workers
            )
            if not torch.equal(extracted["labels"], shard["labels"]):
                raise ValueError(f"augmented {name} subset does not follow the cached order")
            repeats.append(extracted["logits"])
            print(f"{name}: repeat {repeat + 1}/{args.repeats}")
        payload[f"{name}_logits"] = torch.stack(repeats, dim=1)

    destination = augmented_path(args.output, args.dataset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **payload,
            "metadata": {
                "repeats": args.repeats,
                "seed": args.seed,
                "checkpoint_sha256": observed,
                "transform": "dataset_training_augmentation",
            },
        },
        destination,
    )
    print(f"wrote {destination}")


# ======================================================================
#  Training and evaluation
# ======================================================================


def validation_split(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the calibration and selection halves of the validation split."""

    generator = torch.Generator().manual_seed(SPLIT_SEED)
    permutation = torch.randperm(size, generator=generator)
    return permutation[:CALIBRATION_SIZE], permutation[CALIBRATION_SIZE:]


def regime_targets(
    regime: str,
    train_features: torch.Tensor,
    train_logits: torch.Tensor,
    augmented: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return the training rows, their logit targets, and the target scale."""

    if regime == "clean":
        targets, scale = prepare_logit_targets(train_logits)
        return train_features, targets, scale
    if augmented is None:
        raise ValueError("the augmented regime needs cached augmented logits; run 'cache' first")
    repeated = augmented["train_logits"]
    repeats = repeated.shape[1]
    # Every repeat is an independent observation of the same features, so the
    # features are tiled rather than averaged; averaging would destroy exactly
    # the dispersion the variance head has to learn.
    flat = repeated.reshape(-1, repeated.shape[-1]).float()
    targets, scale = prepare_logit_targets(flat)
    tiled = train_features.repeat_interleave(repeats, dim=0)
    return tiled, targets, scale


def empirical_logit_variance(repeated: torch.Tensor, scale: float) -> torch.Tensor:
    """Return the per-sample, per-class variance of centered teacher logits.

    Args:
        repeated: Teacher logits, shape (N, repeats, K).
        scale: The normalization the targets were divided by.

    Returns:
        The variance over repeats, shape (N, K).
    """

    centered = repeated.float() - repeated.float().mean(dim=-1, keepdim=True)
    return (centered / scale).var(dim=1, unbiased=True)


def rank_correlation(left: torch.Tensor, right: torch.Tensor, *, cap: int = 400_000) -> float:
    """Return Spearman's rho, subsampling deterministically for large inputs."""

    left, right = left.flatten().double(), right.flatten().double()
    if left.numel() > cap:
        generator = torch.Generator().manual_seed(SHUFFLE_SEED)
        index = torch.randperm(left.numel(), generator=generator)[:cap]
        left, right = left[index], right[index]
    return float(
        torch.corrcoef(
            torch.stack(
                [
                    left.argsort().argsort().double(),
                    right.argsort().argsort().double(),
                ]
            )
        )[0, 1]
    )


def train_logit_head(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    scale: float,
    args: argparse.Namespace,
    seed: int,
) -> tuple[TAGILastLayerClassifier, float]:
    seed_everything(seed)
    classifier = TAGILastLayerClassifier(
        features.shape[1],
        num_classes,
        head="logit_tagiv",
        device=args.device,
        gain_w=args.gain,
        gain_b=args.gain,
        logit_aleatoric_init=args.aleatoric_init,
        logit_variance_cv=args.variance_cv,
        logit_variance_weight_share=args.variance_weight_share,
        logit_variance_feature_energy=logit_feature_energy(features),
        logit_scale=scale,
        logit_num_samples=args.num_samples,
        logit_seed=seed,
    )
    started = time.perf_counter()
    classifier.fit(features, targets=targets, epochs=args.epochs, batch_size=args.batch_size)
    return classifier, time.perf_counter() - started


def train_label_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    args: argparse.Namespace,
    seed: int,
) -> tuple[TAGILastLayerClassifier, float]:
    seed_everything(seed)
    classifier = TAGILastLayerClassifier(
        features.shape[1],
        num_classes,
        head="categorical_tagiv",
        device=args.device,
        gain_w=args.gain,
        gain_b=args.gain,
    )
    started = time.perf_counter()
    classifier.fit(features, labels, epochs=args.epochs, batch_size=args.batch_size)
    return classifier, time.perf_counter() - started


def substitute_variance(
    aleatoric: torch.Tensor,
    control: str,
    *,
    class_mean: torch.Tensor | None = None,
    seed: int = SHUFFLE_SEED,
) -> torch.Tensor:
    """Return the aleatoric channel a control replaces the learned one with.

    ``constant`` keeps the mean noise level and removes every trace of input
    dependence, so it isolates what the level alone buys. ``shuffled`` keeps the
    marginal distribution of the learned variance and destroys only its
    association with the input, so it isolates what the ranking buys. A control
    that beats neither has learned nothing beyond a global scale.
    """

    if control == "learned":
        return aleatoric
    if control == "constant":
        if class_mean is None:
            raise ValueError("the constant control needs a reference class mean")
        return class_mean.to(aleatoric.device).expand_as(aleatoric)
    if control == "shuffled":
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(aleatoric.shape[0], generator=generator)
        return aleatoric[permutation.to(aleatoric.device)]
    raise ValueError(f"unknown variance control {control!r}")


def teacher_ood_scores(
    identifier_logits: torch.Tensor, outlier_logits: torch.Tensor
) -> dict[str, dict[str, float]]:
    """Score SVHN with the three standard post-hoc scores read off the teacher.

    These are the proxies already present in the frozen logits. The learned
    variance has to beat them to be worth its parameters.
    """

    identifier_probabilities = torch.softmax(identifier_logits, dim=-1)
    outlier_probabilities = torch.softmax(outlier_logits, dim=-1)
    return {
        "teacher_msp": ood_detection_metrics_full(
            negative_max_probability(identifier_probabilities),
            negative_max_probability(outlier_probabilities),
        ),
        "teacher_entropy": ood_detection_metrics_full(
            predictive_entropy(identifier_probabilities),
            predictive_entropy(outlier_probabilities),
        ),
        "teacher_energy": ood_detection_metrics_full(
            -torch.logsumexp(identifier_logits, dim=-1),
            -torch.logsumexp(outlier_logits, dim=-1),
        ),
    }


def entropy_auroc(
    identifier_probabilities: torch.Tensor, outlier_probabilities: torch.Tensor
) -> dict[str, float]:
    return ood_detection_metrics_full(
        predictive_entropy(identifier_probabilities),
        predictive_entropy(outlier_probabilities),
    )


def evaluate_logit_head(
    classifier: TAGILastLayerClassifier,
    data: dict[str, torch.Tensor],
    *,
    args: argparse.Namespace,
    reference_variance: torch.Tensor | None,
) -> dict[str, Any]:
    """Calibrate the head, then re-run the calibration against its controls."""

    calibration, selection = validation_split(data["validation_features"].shape[0])
    validation_labels = data["validation_labels"].to(classifier.device)
    moments = {
        name: classifier.logit_moments(
            data[f"{name}_features"], batch_size=args.prediction_batch_size
        )
        for name in ("validation", "test", "ood")
    }
    validation_mean, validation_epistemic, validation_aleatoric = moments["validation"]
    test_mean, test_epistemic, test_aleatoric = moments["test"]
    ood_mean, ood_epistemic, ood_aleatoric = moments["ood"]
    base_samples = classifier._resolve_logit_base_samples(test_mean.dtype)
    test_labels = data["test_labels"]
    record: dict[str, Any] = {"variants": {}, "variance_controls": {}}

    def probabilities(mean, epistemic, aleatoric, temperature, alpha):
        return logit_tagiv_predictive_probs(
            mean,
            epistemic,
            aleatoric,
            temperature=temperature,
            alpha=alpha,
            base_samples=base_samples,
        )

    for name, mode, alpha in (
        ("temperature_only", "temperature", 0.0),
        ("fixed_multiplier", "temperature", 1.0),
        ("joint", "joint", None),
    ):
        classifier.logit_temperature, classifier.logit_alpha = 1.0, 1.0
        fitted = classifier.calibrate(
            data["validation_features"][calibration],
            data["validation_labels"][calibration],
            mode=mode,
            alpha=alpha,
            batch_size=args.prediction_batch_size,
        )
        metrics = classification_metrics(
            classifier.predict(data["test_features"]).probabilities, test_labels
        )
        held_out = classifier.predict(data["validation_features"][selection]).probabilities
        record["variants"][name] = {
            "temperature": fitted.temperature,
            "alpha": fitted.alpha,
            "calibration_nll": fitted.nll,
            "selection_nll": classification_metrics(held_out, data["validation_labels"][selection])[
                "nll"
            ],
            **{key: metrics[key] for key in REPORTED_METRICS},
        }

    # The controls each get their own (T, alpha) so that none of them is
    # handicapped by a calibration fitted for a different variance channel.
    class_mean = validation_aleatoric[calibration].mean(dim=0, keepdim=True)
    pooled = torch.cat([test_aleatoric, ood_aleatoric])
    for control in VARIANCE_CONTROLS:
        fitted = fit_logit_calibration(
            validation_mean[calibration],
            validation_epistemic[calibration],
            substitute_variance(validation_aleatoric[calibration], control, class_mean=class_mean),
            validation_labels[calibration],
            mode="joint",
            base_samples=base_samples,
        )
        identifier = probabilities(
            test_mean,
            test_epistemic,
            substitute_variance(test_aleatoric, control, class_mean=class_mean),
            fitted.temperature,
            fitted.alpha,
        )
        # The out-of-distribution permutation runs over the pooled test and SVHN
        # rows, which is what actually severs the link between the variance and
        # the input; permuting inside each split would leave the two marginals,
        # and therefore the AUROC, untouched.
        shuffled_pool = substitute_variance(pooled, control, class_mean=class_mean)
        split = test_aleatoric.shape[0]
        outlier = probabilities(
            ood_mean,
            ood_epistemic,
            shuffled_pool[split:],
            fitted.temperature,
            fitted.alpha,
        )
        pooled_identifier = probabilities(
            test_mean,
            test_epistemic,
            shuffled_pool[:split],
            fitted.temperature,
            fitted.alpha,
        )
        metrics = classification_metrics(identifier, test_labels)
        record["variance_controls"][control] = {
            "temperature": fitted.temperature,
            "alpha": fitted.alpha,
            **{key: metrics[key] for key in REPORTED_METRICS},
            "entropy_auroc": entropy_auroc(pooled_identifier, outlier)["auroc"],
        }

    joint = record["variants"]["joint"]
    classifier.logit_temperature, classifier.logit_alpha = joint["temperature"], joint["alpha"]
    identifier = probabilities(
        test_mean, test_epistemic, test_aleatoric, joint["temperature"], joint["alpha"]
    )
    outlier = probabilities(
        ood_mean, ood_epistemic, ood_aleatoric, joint["temperature"], joint["alpha"]
    )
    record["ood"] = {
        "predictive_entropy": entropy_auroc(identifier, outlier),
        "aleatoric_variance": ood_detection_metrics_full(
            test_aleatoric.mean(dim=-1), ood_aleatoric.mean(dim=-1)
        ),
        "epistemic_variance": ood_detection_metrics_full(
            test_epistemic.mean(dim=-1), ood_epistemic.mean(dim=-1)
        ),
        **teacher_ood_scores(
            data["test_logits"].to(test_mean.device), data["ood_logits"].to(test_mean.device)
        ),
    }

    decomposition = logit_tagiv_uncertainty(
        test_mean,
        test_epistemic,
        test_aleatoric,
        temperature=joint["temperature"],
        alpha=joint["alpha"],
        epistemic_samples=args.decomposition_samples,
        aleatoric_samples=args.decomposition_samples,
    )
    record["moments"] = {
        "test_aleatoric": float(test_aleatoric.mean()),
        "test_epistemic": float(test_epistemic.mean()),
        "ood_aleatoric": float(ood_aleatoric.mean()),
        "ood_epistemic": float(ood_epistemic.mean()),
        "test_aleatoric_entropy": float(decomposition.aleatoric_entropy.mean()),
        "test_epistemic_entropy": float(decomposition.epistemic_entropy.mean()),
        "logit_rmse": float(
            (test_mean - data["test_targets"].to(test_mean.device)).pow(2).mean().sqrt()
        ),
    }

    if reference_variance is not None:
        learned = validation_aleatoric.cpu().double()
        measured = reference_variance.double()
        labels = data["validation_labels"].long()
        rows = torch.arange(labels.numel())
        # The per (example, class) statistic is the primary one. With many
        # classes the class average is dominated by the observed class channel,
        # whose learned variance is the least reliable, so that summary can
        # invert the sign of a correlation that is positive channel by channel.
        record["noise_recovery"] = {
            "spearman_per_class": rank_correlation(learned, measured),
            "spearman_observed_class": rank_correlation(
                learned[rows, labels], measured[rows, labels]
            ),
            "spearman_class_mean": rank_correlation(learned.mean(dim=-1), measured.mean(dim=-1)),
            "pearson_log": float(
                torch.corrcoef(
                    torch.stack(
                        [
                            learned.mean(dim=-1).log(),
                            measured.mean(dim=-1).clamp_min(1e-12).log(),
                        ]
                    )
                )[0, 1]
            ),
            "learned_mean": float(learned.mean()),
            "measured_mean": float(measured.mean()),
            "learned_observed_class_mean": float(learned[rows, labels].mean()),
            "measured_observed_class_mean": float(measured[rows, labels].mean()),
            # The learned variance is log-normal by construction, so its mean is
            # set by the upper tail of a linear score. Robust statistics say
            # whether the bulk is right when the mean is not.
            "learned_observed_class_median": float(learned[rows, labels].median()),
            "measured_observed_class_median": float(measured[rows, labels].median()),
            "learned_observed_class_p90": float(learned[rows, labels].quantile(0.9)),
            "measured_observed_class_p90": float(measured[rows, labels].quantile(0.9)),
            "learned_log_dispersion": float(learned[rows, labels].clamp_min(1e-12).log().std()),
            "measured_log_dispersion": float(measured[rows, labels].clamp_min(1e-12).log().std()),
        }
    return record


def run(args: argparse.Namespace) -> None:
    root = feature_root(args.study_root, args.dataset)
    shards = {
        name: load_feature_shard(root / f"{name}.pt")
        for name in ("train", "validation", "test", "svhn")
    }
    augmented_file = augmented_path(args.output, args.dataset)
    augmented = (
        torch.load(augmented_file, map_location="cpu", weights_only=False)
        if augmented_file.exists()
        else None
    )

    num_classes = shards["train"]["logits"].shape[1]
    train_features = shards["train"]["features"].float()
    train_labels = shards["train"]["labels"].long()
    test_targets, _ = prepare_logit_targets(shards["test"]["logits"].float())
    data = {
        "validation_features": shards["validation"]["features"].float(),
        "validation_labels": shards["validation"]["labels"].long(),
        "test_features": shards["test"]["features"].float(),
        "test_labels": shards["test"]["labels"].long(),
        "test_targets": test_targets,
        "test_logits": shards["test"]["logits"].float(),
        "ood_features": shards["svhn"]["features"].float(),
        "ood_logits": shards["svhn"]["logits"].float(),
    }

    calibration, selection = validation_split(data["validation_features"].shape[0])
    teacher_logits = shards["test"]["logits"].float()
    validation_logits = shards["validation"]["logits"].float()
    temperature = fit_softmax_temperature(
        validation_logits[calibration], data["validation_labels"][calibration]
    )
    baselines = {
        "teacher": classification_metrics(
            torch.softmax(teacher_logits, dim=-1), data["test_labels"]
        ),
        "teacher_temperature": classification_metrics(
            torch.softmax(teacher_logits / temperature, dim=-1), data["test_labels"]
        ),
    }
    baselines["teacher_temperature"]["temperature"] = temperature

    results: dict[str, Any] = {
        "config": {
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gain": args.gain,
            "aleatoric_init": args.aleatoric_init,
            "variance_cv": args.variance_cv,
            "variance_weight_share": args.variance_weight_share,
            "num_samples": args.num_samples,
            "repeats": None if augmented is None else augmented["metadata"]["repeats"],
        },
        "baselines": {
            name: {key: value[key] for key in (*REPORTED_METRICS, "temperature") if key in value}
            for name, value in baselines.items()
        },
        "seeds": {},
    }

    for seed in args.seeds:
        seed_record: dict[str, Any] = {}
        label_head, label_seconds = train_label_head(
            train_features, train_labels, num_classes=num_classes, args=args, seed=seed
        )
        label_probabilities = label_head.predict(data["test_features"]).probabilities
        seed_record["categorical_tagiv"] = {
            "wall_s": label_seconds,
            **{
                key: classification_metrics(label_probabilities, data["test_labels"])[key]
                for key in REPORTED_METRICS
            },
            "ood": {
                "predictive_entropy": entropy_auroc(
                    label_probabilities,
                    label_head.predict(data["ood_features"]).probabilities,
                )
            },
        }

        for regime in args.regimes:
            if regime == "augmented" and augmented is None:
                continue
            features, targets, scale = regime_targets(
                regime, train_features, shards["train"]["logits"].float(), augmented
            )
            classifier, seconds = train_logit_head(
                features,
                targets,
                num_classes=num_classes,
                scale=scale,
                args=args,
                seed=seed,
            )
            reference = (
                empirical_logit_variance(augmented["validation_logits"], scale)
                if regime == "augmented" and augmented is not None
                else None
            )
            record = evaluate_logit_head(classifier, data, args=args, reference_variance=reference)
            record["wall_s"] = seconds
            record["target_scale"] = scale
            destination = head_path(args.output, args.dataset, regime, seed)
            classifier.save(destination, metadata={"regime": regime, "seed": seed})
            record["checkpoint"] = str(destination.relative_to(REPOSITORY_ROOT))
            seed_record[regime] = record
            print(
                f"seed {seed} {regime}: "
                + " ".join(
                    f"{name}=nll {values['nll']:.4f}/ece {values['ece']:.4f}"
                    for name, values in record["variants"].items()
                )
            )
        results["seeds"][str(seed)] = seed_record

    destination = args.output / args.dataset / "results.json"
    atomic_json(destination, results)
    print(f"wrote {destination}")


# ======================================================================
#  Corruption severity
# ======================================================================


def corruption_root(dataset: str) -> Path:
    """Return the canonical CIFAR-C archive directory for a dataset."""

    suffix = {"cifar10": "CIFAR-10-C", "cifar100": "CIFAR-100-C"}[dataset]
    return DEFAULT_DATA_ROOT / suffix


def corruption_shard_path(study_root: Path, dataset: str, corruption: str, severity: int) -> Path:
    return feature_root(study_root, dataset) / "corruptions" / f"{corruption}_s{severity}.pt"


def augmentation_variance_on_corruption(
    args: argparse.Namespace,
    corruption: str,
    severity: int,
    scale: float,
) -> torch.Tensor:
    """Measure the teacher's augmentation logit spread on corrupted images.

    The corrupted images are re-read through the training transform, so the
    measured quantity is the same one the head was trained to predict, only on
    shifted inputs.
    """

    model = CifarResNet18(args.num_classes)
    checkpoint = args.study_root / "backbones" / f"{args.dataset}_resnet18.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    dataset = CifarCorruption(
        args.corruption_root or corruption_root(args.dataset),
        args.dataset,
        corruption,
        severity,
    )
    # CifarCorruption yields raw ndarray frames, which the evaluation transform
    # accepts and the training transform's RandomCrop does not.
    dataset.transform = transforms.Compose(
        [transforms.ToPILImage(), training_transform(get_spec(args.dataset))]
    )
    repeats = []
    for repeat in range(args.augmentation_repeats):
        seed_everything(repeat)
        extracted = extract_frozen_features(
            model,
            dataset,
            device=args.device,
            batch_size=args.prediction_batch_size,
            workers=args.workers,
        )
        repeats.append(extracted["logits"])
    return empirical_logit_variance(torch.stack(repeats, dim=1), scale)


def corruptions(args: argparse.Namespace) -> None:
    """Sweep every canonical corruption and severity with one calibrated head."""

    classifier, metadata = TAGILastLayerClassifier.load(
        head_path(args.output, args.dataset, args.regime, args.seed), device=args.device
    )
    args.num_classes = classifier.num_classes
    clean = load_feature_shard(feature_root(args.study_root, args.dataset) / "test.pt")
    clean_features = clean["features"].float()
    clean_probabilities = classifier.predict(clean_features).probabilities
    _, clean_epistemic, clean_aleatoric = classifier.logit_moments(
        clean_features, batch_size=args.prediction_batch_size
    )
    clean_metrics = classification_metrics(clean_probabilities, clean["labels"].long())

    records: list[dict[str, Any]] = []
    for corruption in CANONICAL_CORRUPTIONS:
        for severity in range(1, 6):
            path = corruption_shard_path(args.study_root, args.dataset, corruption, severity)
            if not path.exists():
                continue
            shard = load_feature_shard(path)
            features = shard["features"].float()
            labels = shard["labels"].long()
            probabilities = classifier.predict(features).probabilities
            _, epistemic, aleatoric = classifier.logit_moments(
                features, batch_size=args.prediction_batch_size
            )
            metrics = classification_metrics(probabilities, labels)
            record = {
                "corruption": corruption,
                "severity": severity,
                "aleatoric": float(aleatoric.mean()),
                "epistemic": float(epistemic.mean()),
                **{key: metrics[key] for key in REPORTED_METRICS},
                "aurc": metrics["aurc"],
                "risk_at_90_coverage": metrics["risk_at_90_coverage"],
                "detection_aleatoric": ood_detection_metrics_full(
                    clean_aleatoric.mean(dim=-1), aleatoric.mean(dim=-1)
                )["auroc"],
                "detection_entropy": ood_detection_metrics_full(
                    predictive_entropy(clean_probabilities), predictive_entropy(probabilities)
                )["auroc"],
            }
            if args.augmentation_repeats > 0 and corruption in args.measure_corruptions:
                measured = augmentation_variance_on_corruption(
                    args, corruption, severity, classifier.logit_scale
                )
                learned = aleatoric.mean(dim=-1).cpu().double()
                reference = measured.mean(dim=-1).double()
                record["spearman"] = float(
                    torch.corrcoef(
                        torch.stack(
                            [
                                learned.argsort().argsort().double(),
                                reference.argsort().argsort().double(),
                            ]
                        )
                    )[0, 1]
                )
                record["measured_aleatoric"] = float(reference.mean())
            records.append(record)
            print(
                f"{corruption} s{severity}: acc {record['accuracy']:.4f} "
                f"nll {record['nll']:.4f} a {record['aleatoric']:.4f}"
            )

    payload = {
        "config": {
            "dataset": args.dataset,
            "regime": args.regime,
            "seed": args.seed,
            "checkpoint_metadata": metadata,
            "temperature": classifier.logit_temperature,
            "alpha": classifier.logit_alpha,
            "augmentation_repeats": args.augmentation_repeats,
        },
        "clean": {
            "aleatoric": float(clean_aleatoric.mean()),
            "epistemic": float(clean_epistemic.mean()),
            **{key: clean_metrics[key] for key in REPORTED_METRICS},
            "aurc": clean_metrics["aurc"],
            "risk_at_90_coverage": clean_metrics["risk_at_90_coverage"],
        },
        "records": records,
    }
    destination = args.output / args.dataset / "corruptions.json"
    atomic_json(destination, payload)
    print(f"wrote {destination}")


def corruption_report(args: argparse.Namespace) -> None:
    source = args.output / args.dataset / "corruptions.json"
    payload = json.loads(source.read_text())
    records = payload["records"]
    clean = payload["clean"]
    lines = [
        f"# Corruption severity, {args.dataset}",
        "",
        f"One calibrated `{payload['config']['regime']}` head "
        f"(T = {payload['config']['temperature']:.4f}, "
        f"alpha = {payload['config']['alpha']:.4f}), every canonical corruption at "
        "every severity. Detection AUROC separates the clean test split from the "
        "corrupted one at that severity.",
        "",
        "| severity | accuracy | NLL | ECE | mean a(x) | AURC | risk@90 | AUROC a(x) "
        "| AUROC entropy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| 0 (clean) | {clean['accuracy']:.4f} | {clean['nll']:.4f} | {clean['ece']:.4f} | "
        f"{clean['aleatoric']:.4f} | {clean['aurc']:.4f} | "
        f"{clean['risk_at_90_coverage']:.4f} | — | — |",
    ]
    for severity in range(1, 6):
        rows = [record for record in records if record["severity"] == severity]
        if not rows:
            continue

        def mean(key: str, rows=rows) -> float:
            return statistics.mean(record[key] for record in rows)

        lines.append(
            f"| {severity} | {mean('accuracy'):.4f} | {mean('nll'):.4f} | {mean('ece'):.4f} | "
            f"{mean('aleatoric'):.4f} | {mean('aurc'):.4f} | "
            f"{mean('risk_at_90_coverage'):.4f} | {mean('detection_aleatoric'):.4f} | "
            f"{mean('detection_entropy'):.4f} |"
        )

    measured = [record for record in records if "spearman" in record]
    if measured:
        lines += [
            "",
            "## Learned variance against the measured augmentation spread",
            "",
            "| corruption | severity | rank correlation | learned a(x) | measured |",
            "| --- | --- | --- | --- | --- |",
        ]
        for record in measured:
            lines.append(
                f"| {record['corruption']} | {record['severity']} | "
                f"{record['spearman']:.4f} | {record['aleatoric']:.4f} | "
                f"{record['measured_aleatoric']:.4f} |"
            )

    lines += [
        "",
        "## Per-corruption mean aleatoric variance",
        "",
        "| corruption | " + " | ".join(f"s{severity}" for severity in range(1, 6)) + " |",
        "| --- | " + " | ".join("---" for _ in range(5)) + " |",
    ]
    for corruption in sorted({record["corruption"] for record in records}):
        cells = []
        for severity in range(1, 6):
            match = [
                record
                for record in records
                if record["corruption"] == corruption and record["severity"] == severity
            ]
            cells.append(f"{match[0]['aleatoric']:.4f}" if match else "—")
        lines.append(f"| {corruption} | " + " | ".join(cells) + " |")

    destination = args.output / args.dataset / "CORRUPTIONS.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


# ======================================================================
#  Variance-fitting study
# ======================================================================

VARIANCE_METHODS = ("joint", "warmup_agvi", "warmup_replicate")


def build_logit_head(
    features: torch.Tensor,
    num_classes: int,
    scale: float,
    args: argparse.Namespace,
    seed: int,
) -> TAGILastLayerClassifier:
    seed_everything(seed)
    return TAGILastLayerClassifier(
        features.shape[1],
        num_classes,
        head="logit_tagiv",
        device=args.device,
        gain_w=args.gain,
        gain_b=args.gain,
        logit_aleatoric_init=args.aleatoric_init,
        logit_variance_cv=args.variance_cv,
        logit_variance_weight_share=args.variance_weight_share,
        logit_variance_feature_energy=logit_feature_energy(features),
        logit_scale=scale,
        logit_num_samples=args.num_samples,
        logit_seed=seed,
    )


def fit_variance_method(
    method: str,
    *,
    clean_features: torch.Tensor,
    tiled_features: torch.Tensor,
    tiled_targets: torch.Tensor,
    replicate_mean: torch.Tensor,
    replicate_variance: torch.Tensor,
    repeats: int,
    num_classes: int,
    scale: float,
    args: argparse.Namespace,
    seed: int,
) -> tuple[TAGILastLayerClassifier, float]:
    """Fit one point of the matrix and return it with its wall time.

    ``joint`` is the cold start: mean and variance learn together from the first
    step, so the early residual on a high-magnitude target is available to be
    stored as noise. The two warm-up methods fit the mean first under a fixed
    observation variance, reset the variance head to its prior, and only then
    let it see residuals. They differ in what it sees: ``warmup_agvi`` still
    infers the residual from ``Y - Z``, while ``warmup_replicate`` observes the
    sample variance across augmentation repeats, which never passes through the
    mean head at all.
    """

    started = time.perf_counter()
    if method == "joint":
        classifier = build_logit_head(tiled_features, num_classes, scale, args, seed)
        classifier.fit(
            tiled_features,
            targets=tiled_targets,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        return classifier, time.perf_counter() - started

    classifier = build_logit_head(clean_features, num_classes, scale, args, seed)
    classifier.fit_mean(
        clean_features,
        replicate_mean,
        epochs=args.mean_epochs,
        batch_size=args.batch_size,
        observation_variance=args.mean_observation_variance,
        seed=seed,
    )
    classifier.reset_variance_head()
    if method == "warmup_agvi":
        classifier.fit_variance(
            tiled_features,
            targets=tiled_targets,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed,
        )
    elif method == "warmup_replicate":
        classifier.fit_variance(
            clean_features,
            residual_variance=replicate_variance,
            repeats=repeats,
            epochs=args.variance_epochs,
            batch_size=args.batch_size,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown variance method {method!r}")
    return classifier, time.perf_counter() - started


def variance_study(args: argparse.Namespace) -> None:
    """Compare cold-start AGVI against two mean-first variance fits."""

    root = feature_root(args.study_root, args.dataset)
    shards = {
        name: load_feature_shard(root / f"{name}.pt")
        for name in ("train", "validation", "test", "svhn")
    }
    augmented = torch.load(
        augmented_path(args.output, args.dataset), map_location="cpu", weights_only=False
    )
    repeats = int(augmented["metadata"]["repeats"])
    num_classes = shards["train"]["logits"].shape[1]
    clean_features = shards["train"]["features"].float()
    tiled_features, tiled_targets, scale = regime_targets(
        "augmented", clean_features, shards["train"]["logits"].float(), augmented
    )
    replicate_mean, replicate_variance = logit_replicate_variance(
        augmented["train_logits"].float(), scale=scale
    )
    measured = empirical_logit_variance(augmented["validation_logits"], scale)

    test_targets, _ = prepare_logit_targets(shards["test"]["logits"].float())
    data = {
        "validation_features": shards["validation"]["features"].float(),
        "validation_labels": shards["validation"]["labels"].long(),
        "test_features": shards["test"]["features"].float(),
        "test_labels": shards["test"]["labels"].long(),
        "test_targets": test_targets,
        "test_logits": shards["test"]["logits"].float(),
        "ood_features": shards["svhn"]["features"].float(),
        "ood_logits": shards["svhn"]["logits"].float(),
    }
    labels = data["validation_labels"]
    rows = torch.arange(labels.numel())

    calibration, _ = validation_split(data["validation_features"].shape[0])
    temperature = fit_softmax_temperature(
        shards["validation"]["logits"].float()[calibration],
        data["validation_labels"][calibration],
    )
    teacher_logits = data["test_logits"]
    baselines = {
        "teacher": classification_metrics(
            torch.softmax(teacher_logits, dim=-1), data["test_labels"]
        ),
        "teacher_temperature": classification_metrics(
            torch.softmax(teacher_logits / temperature, dim=-1), data["test_labels"]
        ),
    }
    baselines["teacher_temperature"]["temperature"] = temperature

    results: dict[str, Any] = {
        "config": {
            "dataset": args.dataset,
            "repeats": repeats,
            "epochs": args.epochs,
            "mean_epochs": args.mean_epochs,
            "variance_epochs": args.variance_epochs,
            "mean_observation_variance": args.mean_observation_variance,
            "batch_size": args.batch_size,
            "gain": args.gain,
            "aleatoric_init": args.aleatoric_init,
            "variance_weight_share": args.variance_weight_share,
        },
        "measured": {
            "all_channels": float(measured.mean()),
            "observed_class": float(measured[rows, labels].mean()),
        },
        "methods": {},
    }

    for method in args.methods:
        for seed in args.seeds:
            classifier, seconds = fit_variance_method(
                method,
                clean_features=clean_features,
                tiled_features=tiled_features,
                tiled_targets=tiled_targets,
                replicate_mean=replicate_mean,
                replicate_variance=replicate_variance,
                repeats=repeats,
                num_classes=num_classes,
                scale=scale,
                args=args,
                seed=seed,
            )
            record = evaluate_logit_head(classifier, data, args=args, reference_variance=measured)
            record["wall_s"] = seconds
            logit_mean, _, aleatoric = classifier.logit_moments(
                data["ood_features"], batch_size=args.prediction_batch_size
            )
            outlier = channel_aggregations(logit_mean, aleatoric)
            clean_mean, _, clean_aleatoric = classifier.logit_moments(
                data["test_features"], batch_size=args.prediction_batch_size
            )
            identifier = channel_aggregations(clean_mean, clean_aleatoric)
            record["channel_auroc"] = {
                name: ood_detection_metrics_full(identifier[name], outlier[name])["auroc"]
                for name in identifier
            }
            results["methods"].setdefault(method, {})[str(seed)] = record
            recovery = record["noise_recovery"]
            print(
                f"{method} seed {seed}: observed-class a="
                f"{recovery['learned_observed_class_mean']:.4f} "
                f"(measured {recovery['measured_observed_class_mean']:.4f}) "
                f"rank/pair={recovery['spearman_per_class']:+.3f} "
                f"nll={record['variants']['joint']['nll']:.4f} "
                f"alpha={record['variants']['joint']['alpha']:.3f}"
            )
            destination = head_path(args.output, args.dataset, f"study_{method}", seed)
            classifier.save(destination, metadata={"method": method, "seed": seed})

    atomic_json(args.output / args.dataset / "variance_study.json", results)
    print(f"wrote {args.output / args.dataset / 'variance_study.json'}")


def variance_study_report(args: argparse.Namespace) -> None:
    source = args.output / args.dataset / "variance_study.json"
    results = json.loads(source.read_text())

    def gather(method: str, path: tuple[str, ...]) -> list[float]:
        collected = []
        for record in results["methods"].get(method, {}).values():
            node: Any = record
            for key in path:
                if not isinstance(node, dict) or key not in node:
                    node = None
                    break
                node = node[key]
            if isinstance(node, (int, float)):
                collected.append(float(node))
        return collected

    measured = results["measured"]
    lines = [
        f"# Separating mean fitting from variance fitting, {args.dataset}",
        "",
        "`joint` is the cold start, in which the mean and the variance learn "
        "together and the early residual on a high-magnitude target can be "
        "stored as noise. `warmup_agvi` fits the mean first under a fixed "
        "observation variance, resets the variance head, and then runs the same "
        "single-observation AGVI. `warmup_replicate` replaces that last step "
        "with an update driven by the sample variance across augmentation "
        "repeats, which never passes through the mean head.",
        "",
        f"Measured augmentation spread: {measured['observed_class']:.4f} on the "
        f"observed class, {measured['all_channels']:.4f} over all channels. The "
        "learned variance is log-normal by construction, so read its median "
        "alongside its mean: a linear score with too much spread inflates the "
        "mean through its upper tail while the bulk stays close.",
        "",
        "| method | observed-class a (mean) | observed-class a (median) | "
        "sd(log a) | rank corr per (x,k) | alpha | NLL | ECE | accuracy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method in VARIANCE_METHODS:
        if not gather(method, ("wall_s",)):
            continue
        lines.append(
            f"| {method} | "
            f"{summarize(gather(method, ('noise_recovery', 'learned_observed_class_mean')))} | "
            f"{summarize(gather(method, ('noise_recovery', 'learned_observed_class_median')))} | "
            f"{summarize(gather(method, ('noise_recovery', 'learned_log_dispersion')))} | "
            f"{summarize(gather(method, ('noise_recovery', 'spearman_per_class')))} | "
            f"{summarize(gather(method, ('variants', 'joint', 'alpha')))} | "
            f"{summarize(gather(method, ('variants', 'joint', 'nll')))} | "
            f"{summarize(gather(method, ('variants', 'joint', 'ece')))} | "
            f"{summarize(gather(method, ('variants', 'joint', 'accuracy')))} |"
        )

    lines += [
        "",
        "## Variance controls",
        "",
        "| method | control | alpha | NLL | ECE | SVHN AUROC (entropy) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for method in VARIANCE_METHODS:
        for control in VARIANCE_CONTROLS:
            values = gather(method, ("variance_controls", control, "nll"))
            if not values:
                continue
            lines.append(
                f"| {method} | {control} | "
                f"{summarize(gather(method, ('variance_controls', control, 'alpha')))} | "
                f"{summarize(values)} | "
                f"{summarize(gather(method, ('variance_controls', control, 'ece')))} | "
                f"{summarize(gather(method, ('variance_controls', control, 'entropy_auroc')))} |"
            )

    lines += [
        "",
        "## SVHN detection (AUROC)",
        "",
        "| method | predictive entropy | a mean | a top | a rest | a median |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for method in VARIANCE_METHODS:
        values = gather(method, ("ood", "predictive_entropy", "auroc"))
        if not values:
            continue
        lines.append(
            f"| {method} | {summarize(values)} | "
            + " | ".join(
                summarize(gather(method, ("channel_auroc", name)))
                for name in ("mean", "top", "rest", "median")
            )
            + " |"
        )

    destination = args.output / args.dataset / "VARIANCE_STUDY.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


# ======================================================================
#  Spread control
# ======================================================================

POWERS = (1.0, 0.75, 0.6, 0.5, 0.4, 0.25)
# The per-batch shrink factor is 1 / (1 + lambda * Sw), and the variance head's
# weight prior is Sw = share * v_G / feature_energy, of order 1e-3. A rate has
# to reach O(1 / Sw) before it competes with the update pushing the weights back
# out each step, so the useful grid is in the hundreds, not in the units.
SHRINKAGE_RATES = (0.0, 30.0, 100.0, 150.0, 200.0, 300.0, 1000.0)
# A rate is only meaningful next to the weight variance it acts on, so the grid
# that transfers between datasets is the dimensionless kappa_0 = lambda * v_Wg,0.
SHRINKAGE_KAPPAS = (0.0, 0.05, 0.10, 0.164, 0.25, 0.40)


def exact_quantile(values: torch.Tensor, quantile: float) -> float:
    """Return a linear-interpolation quantile without torch.quantile's size cap.

    ``torch.quantile`` refuses inputs beyond about 16M elements, which a
    thousand-way head reaches from twenty thousand rows alone. Sorting has no
    such limit and reproduces the same definition exactly.
    """

    ordered = values.flatten().sort().values
    position = quantile * (ordered.numel() - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def variance_quantiles(learned: torch.Tensor, measured: torch.Tensor) -> dict[str, float]:
    """Return the level statistics that a log-normal variance must be judged on."""

    quantiles = (0.5, 0.9, 0.99)
    learned, measured = learned.double(), measured.double()
    result = {
        "learned_mean": float(learned.mean()),
        "measured_mean": float(measured.mean()),
        "learned_log_dispersion": float(learned.clamp_min(1e-12).log().std()),
        "measured_log_dispersion": float(measured.clamp_min(1e-12).log().std()),
    }
    for name, value in zip(("p50", "p90", "p99"), quantiles, strict=True):
        result[f"learned_{name}"] = exact_quantile(learned, value)
        result[f"measured_{name}"] = exact_quantile(measured, value)
    return result


def quantile_disagreement(levels: dict[str, float]) -> float:
    """Return the mean absolute log ratio across p50, p90 and p99.

    A squared error in log variance over every channel is dominated by the
    bulk, which a cold-started head already fits, so it cannot see a tail that
    is an order of magnitude too large. Scoring the quantiles directly is what
    the level criterion actually asks for.
    """

    ratios = [
        abs(math.log(max(levels[f"learned_{name}"], 1e-12) / levels[f"measured_{name}"]))
        for name in ("p50", "p90", "p99")
    ]
    return sum(ratios) / len(ratios)


def log_variance_error(learned: torch.Tensor, measured: torch.Tensor, *, repeats: int) -> float:
    """Return the mean squared error in log space against an unbiased target.

    ``log R`` underestimates ``log S`` by a known digamma offset, and carries an
    irreducible trigamma variance, so the offset is added back before scoring
    and the floor is reported alongside.
    """

    target = measured.double().clamp_min(1e-12).log() + replicate_log_variance_offset(repeats)
    return float((learned.double().clamp_min(1e-12).log() - target).square().mean())


def power_diagnostic(args: argparse.Namespace) -> None:
    """Sweep the variance compression exponent on a saved head, without retraining.

    The transform is strictly increasing, so it cannot change any ranking the
    variance induces. Whatever it does change is therefore attributable to the
    level and to the predictive integration alone, which makes it a clean causal
    check of the over-dispersion hypothesis.
    """

    root = feature_root(args.study_root, args.dataset)
    shards = {
        name: load_feature_shard(root / f"{name}.pt") for name in ("validation", "test", "svhn")
    }
    augmented = torch.load(
        augmented_path(args.output, args.dataset), map_location="cpu", weights_only=False
    )
    repeats = int(augmented["metadata"]["repeats"])
    classifier, _ = TAGILastLayerClassifier.load(
        head_path(args.output, args.dataset, args.regime, args.seed), device=args.device
    )
    measured = empirical_logit_variance(augmented["validation_logits"], classifier.logit_scale)

    calibration, _ = validation_split(shards["validation"]["features"].shape[0])
    validation_features = shards["validation"]["features"].float()
    validation_labels = shards["validation"]["labels"].long()
    labels = validation_labels
    rows = torch.arange(labels.numel())

    classifier.logit_variance_power = 1.0
    base_mean, base_epistemic, base_aleatoric = classifier.logit_moments(
        validation_features, batch_size=args.prediction_batch_size
    )
    reference = float(base_aleatoric[calibration].median())
    classifier.logit_variance_reference = reference
    outlier_mean, outlier_epistemic, outlier_aleatoric = classifier.logit_moments(
        shards["svhn"]["features"].float(), batch_size=args.prediction_batch_size
    )
    test_mean, test_epistemic, test_aleatoric = classifier.logit_moments(
        shards["test"]["features"].float(), batch_size=args.prediction_batch_size
    )
    test_labels = shards["test"]["labels"].long()
    base_samples = classifier._resolve_logit_base_samples(base_mean.dtype)

    records = []
    for gamma in args.powers:
        classifier.logit_variance_power = gamma
        learned = classifier._compress_variance(base_aleatoric)
        fitted = fit_logit_calibration(
            base_mean[calibration],
            base_epistemic[calibration],
            learned[calibration],
            validation_labels[calibration].to(classifier.device),
            mode="joint",
            base_samples=base_samples,
        )
        probabilities = logit_tagiv_predictive_probs(
            test_mean,
            test_epistemic,
            classifier._compress_variance(test_aleatoric),
            temperature=fitted.temperature,
            alpha=fitted.alpha,
            base_samples=base_samples,
        )
        metrics = classification_metrics(probabilities, test_labels)
        identifier = channel_aggregations(test_mean, classifier._compress_variance(test_aleatoric))
        outlier = channel_aggregations(
            outlier_mean, classifier._compress_variance(outlier_aleatoric)
        )
        observed = learned[rows, labels].cpu()
        record = {
            "gamma": gamma,
            "temperature": fitted.temperature,
            "alpha": fitted.alpha,
            **{key: metrics[key] for key in REPORTED_METRICS},
            **variance_quantiles(observed, measured[rows, labels]),
            "log_error": log_variance_error(learned.cpu(), measured, repeats=repeats),
            "spearman_per_class": rank_correlation(learned.cpu(), measured),
            "auroc_aleatoric": ood_detection_metrics_full(identifier["mean"], outlier["mean"])[
                "auroc"
            ],
        }
        records.append(record)
        print(
            f"gamma={gamma:<5} T={record['temperature']:.3f} alpha={record['alpha']:.3f} "
            f"nll={record['nll']:.4f} p50={record['learned_p50']:.4f} "
            f"p99={record['learned_p99']:.4f} rank={record['spearman_per_class']:+.3f} "
            f"auroc_a={record['auroc_aleatoric']:.3f}"
        )

    payload = {
        "config": {
            "dataset": args.dataset,
            "regime": args.regime,
            "seed": args.seed,
            "reference": reference,
            "repeats": repeats,
            "log_error_floor": replicate_log_variance_noise(repeats),
        },
        "measured": {
            "observed_class": variance_quantiles(measured[rows, labels], measured[rows, labels])
        },
        "records": records,
    }
    atomic_json(args.output / args.dataset / "power_diagnostic.json", payload)

    floor = payload["config"]["log_error_floor"]
    reference_row = records[0]
    lines = [
        f"# Variance compression on a saved head, {args.dataset}",
        "",
        "`a_gamma = s_min2 + a_ref (a / a_ref)^gamma` with `a_ref` the "
        f"calibration-set median ({reference:.4f}). The map is strictly "
        "increasing, so the rank correlation and the a-only AUROC are invariant "
        "by construction; only the level and the predictive integration move.",
        "",
        f"Measured observed-class spread: p50 {reference_row['measured_p50']:.4f}, "
        f"p90 {reference_row['measured_p90']:.4f}, p99 {reference_row['measured_p99']:.4f}, "
        f"sd(log R) {reference_row['measured_log_dispersion']:.3f}. Log-space error "
        f"cannot fall below {floor:.3f}, the sampling noise of R at M={repeats}.",
        "",
        "| gamma | T | alpha | NLL | ECE | p50 | p90 | p99 | sd(log a) | "
        "log error | rank corr | AUROC a |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record['gamma']} | {record['temperature']:.3f} | {record['alpha']:.3f} | "
            f"{record['nll']:.4f} | {record['ece']:.4f} | {record['learned_p50']:.4f} | "
            f"{record['learned_p90']:.4f} | {record['learned_p99']:.4f} | "
            f"{record['learned_log_dispersion']:.3f} | {record['log_error']:.3f} | "
            f"{record['spearman_per_class']:+.3f} | {record['auroc_aleatoric']:.3f} |"
        )
    destination = args.output / args.dataset / "POWER_DIAGNOSTIC.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


def prior_weight_variance(features: torch.Tensor, args: argparse.Namespace) -> float:
    """Return ``v_Wg,0``, the variance-head weight prior the shrinkage acts on."""

    _, prior_variance = logit_variance_head_prior(
        aleatoric_init=args.aleatoric_init,
        variance_floor=LOGIT_VARIANCE_FLOOR,
        coefficient_of_variation=args.variance_cv,
    )
    weight_variance, _ = logit_variance_prior_split(
        prior_variance,
        feature_energy=logit_feature_energy(features),
        weight_share=args.variance_weight_share,
    )
    return weight_variance


def shrinkage_study(args: argparse.Namespace) -> None:
    """Sweep persistent shrinkage on the variance head's weights.

    The mean is fitted once and shared, so every rate sees the same converged
    latent stream and the only thing that changes is how much across-input
    spread the log-variance is allowed to develop. The rate is selected on
    held-out replicated logits, never on labels, so that the variance function,
    the aleatoric multiplier and the temperature are each fitted against the
    quantity they are responsible for.
    """

    root = feature_root(args.study_root, args.dataset)
    shards = {
        name: load_feature_shard(root / f"{name}.pt")
        for name in ("train", "validation", "test", "svhn")
    }
    augmented = torch.load(
        augmented_path(args.output, args.dataset), map_location="cpu", weights_only=False
    )
    repeats = int(augmented["metadata"]["repeats"])
    num_classes = shards["train"]["logits"].shape[1]
    clean_features = shards["train"]["features"].float()
    _, _, scale = regime_targets(
        "augmented", clean_features, shards["train"]["logits"].float(), augmented
    )
    replicate_mean, replicate_variance = logit_replicate_variance(
        augmented["train_logits"].float(), scale=scale
    )
    measured = empirical_logit_variance(augmented["validation_logits"], scale)

    test_targets, _ = prepare_logit_targets(shards["test"]["logits"].float())
    data = {
        "validation_features": shards["validation"]["features"].float(),
        "validation_labels": shards["validation"]["labels"].long(),
        "test_features": shards["test"]["features"].float(),
        "test_labels": shards["test"]["labels"].long(),
        "test_targets": test_targets,
        "test_logits": shards["test"]["logits"].float(),
        "ood_features": shards["svhn"]["features"].float(),
        "ood_logits": shards["svhn"]["logits"].float(),
    }
    labels = data["validation_labels"]
    rows = torch.arange(labels.numel())

    weight_prior = prior_weight_variance(clean_features, args)
    rates = (
        [kappa / weight_prior for kappa in args.kappas]
        if args.kappas is not None
        else list(args.rates)
    )
    calibration, _ = validation_split(data["validation_features"].shape[0])
    temperature = fit_softmax_temperature(
        shards["validation"]["logits"].float()[calibration],
        data["validation_labels"][calibration],
    )
    baselines = {
        "teacher": classification_metrics(
            torch.softmax(data["test_logits"], dim=-1), data["test_labels"]
        ),
        "teacher_temperature": classification_metrics(
            torch.softmax(data["test_logits"] / temperature, dim=-1), data["test_labels"]
        ),
    }
    baselines["teacher_temperature"]["temperature"] = temperature

    results: dict[str, Any] = {
        "config": {
            "dataset": args.dataset,
            "repeats": repeats,
            "mean_epochs": args.mean_epochs,
            "variance_epochs": args.variance_epochs,
            "batch_size": args.batch_size,
            "training_rows": int(clean_features.shape[0]),
            "batches_per_epoch": math.ceil(clean_features.shape[0] / args.batch_size),
            "shrinkage_applications": args.variance_epochs
            * math.ceil(clean_features.shape[0] / args.batch_size),
            "log_error_floor": replicate_log_variance_noise(repeats),
            "prior_weight_variance": weight_prior,
            "feature_energy": logit_feature_energy(clean_features),
        },
        "baselines": {
            name: {key: value[key] for key in (*REPORTED_METRICS, "temperature") if key in value}
            for name, value in baselines.items()
        },
        "measured": variance_quantiles(measured[rows, labels], measured[rows, labels]),
        "rates": {},
    }

    for seed in args.seeds:
        warm = build_logit_head(clean_features, num_classes, scale, args, seed)
        warm.fit_mean(
            clean_features,
            replicate_mean,
            epochs=args.mean_epochs,
            batch_size=args.batch_size,
            observation_variance=args.mean_observation_variance,
            seed=seed,
        )
        warm_path = head_path(args.output, args.dataset, "warm_mean", seed)
        warm.save(warm_path, metadata={"phase": "mean", "seed": seed})

        for rate in rates:
            classifier, _ = TAGILastLayerClassifier.load(warm_path, device=args.device)
            classifier.reset_variance_head()
            classifier.fit_variance(
                clean_features,
                residual_variance=replicate_variance,
                repeats=repeats,
                epochs=args.variance_epochs,
                batch_size=args.batch_size,
                seed=seed,
                weight_shrinkage=rate,
            )
            record = evaluate_logit_head(classifier, data, args=args, reference_variance=measured)
            _, _, learned = classifier.logit_moments(
                data["validation_features"], batch_size=args.prediction_batch_size
            )
            learned = learned.cpu()
            record["levels"] = variance_quantiles(learned[rows, labels], measured[rows, labels])
            record["log_error"] = log_variance_error(learned, measured, repeats=repeats)
            record["quantile_disagreement"] = quantile_disagreement(record["levels"])
            # A rate is only meaningful next to the weight variance it acts on:
            # the per-application factor is 1 / (1 + kappa) with kappa = lambda * v.
            head_weight_prior, _ = logit_variance_prior_split(
                logit_variance_head_prior(
                    aleatoric_init=classifier.logit_aleatoric_init,
                    variance_floor=classifier.logit_variance_floor,
                    coefficient_of_variation=classifier.logit_variance_cv,
                )[1],
                feature_energy=classifier.logit_variance_feature_energy or 1.0,
                weight_share=classifier.logit_variance_weight_share,
            )
            final_weight_variance = float(classifier.linear.Sw[:, 1::2].mean())
            record["portability"] = {
                "rate": rate,
                "prior_weight_variance": head_weight_prior,
                "final_weight_variance": final_weight_variance,
                "kappa_initial": rate * head_weight_prior,
                "kappa_final": rate * final_weight_variance,
            }
            record["observed_log_error"] = log_variance_error(
                learned[rows, labels], measured[rows, labels], repeats=repeats
            )
            results["rates"].setdefault(f"{rate:g}", {})[str(seed)] = record
            print(
                f"lambda={rate:<6g} seed {seed}: quantile={record['quantile_disagreement']:.3f} "
                f"log_error={record['log_error']:.3f} "
                f"p50={record['levels']['learned_p50']:.4f} "
                f"p90={record['levels']['learned_p90']:.4f} "
                f"p99={record['levels']['learned_p99']:.4f} "
                f"sd={record['levels']['learned_log_dispersion']:.3f} "
                f"rank={record['noise_recovery']['spearman_per_class']:+.3f} "
                f"nll={record['variants']['joint']['nll']:.4f}"
            )
            classifier.save(
                head_path(args.output, args.dataset, f"shrink_{rate:g}", seed),
                metadata={"shrinkage": rate, "seed": seed},
            )

    atomic_json(args.output / args.dataset / "shrinkage_study.json", results)
    print(f"wrote {args.output / args.dataset / 'shrinkage_study.json'}")


def shrinkage_study_report(args: argparse.Namespace) -> None:
    source = args.output / args.dataset / "shrinkage_study.json"
    results = json.loads(source.read_text())
    measured = results["measured"]
    floor = results["config"]["log_error_floor"]

    def gather(rate: str, path: tuple[str, ...]) -> list[float]:
        collected = []
        for record in results["rates"][rate].values():
            node: Any = record
            for key in path:
                if not isinstance(node, dict) or key not in node:
                    node = None
                    break
                node = node[key]
            if isinstance(node, (int, float)):
                collected.append(float(node))
        return collected

    ordered = sorted(results["rates"], key=float)
    selected = min(
        ordered, key=lambda rate: statistics.mean(gather(rate, ("quantile_disagreement",)))
    )
    lines = [
        f"# Persistent shrinkage of the variance-head weights, {args.dataset}",
        "",
        "A zero-mean Gaussian prior of precision `lambda_g` is reapplied to the "
        "variance head's weights after every batch, leaving the bias free so the "
        "level stays unconstrained while the across-input spread is bounded. The "
        "rate is selected on held-out replicated logits, never on labels, by the "
        "mean absolute log ratio across p50, p90 and p99 of the observed class. "
        "A squared error in log variance over every channel is not usable here: "
        "it is dominated by the 99 off-diagonal channels that a cold start "
        "already fits, so it is blind to a tail an order of magnitude too large "
        "and selects no shrinkage at all.",
        "",
        f"Measured observed-class spread: p50 {measured['measured_p50']:.4f}, "
        f"p90 {measured['measured_p90']:.4f}, p99 {measured['measured_p99']:.4f}, "
        f"sd(log R) {measured['measured_log_dispersion']:.3f}. Log-space error "
        f"cannot fall below {floor:.3f}. Selected rate: `{selected}`.",
        "",
        f"Applied {results['config']['shrinkage_applications']} times: "
        f"{results['config']['variance_epochs']} epochs of "
        f"{results['config']['batches_per_epoch']} batches over "
        f"{results['config']['training_rows']} rows at batch size "
        f"{results['config']['batch_size']}. A rate is not portable on its own; "
        "its strength is `kappa = lambda_g * v_Wg`, the per-application factor "
        "being `1 / (1 + kappa)`. Select it afresh on another dataset, or "
        "parameterize it through a dimensionless initial `kappa`.",
        "",
        "| lambda | kappa initial | kappa final | v_Wg prior | v_Wg final |",
        "| --- | --- | --- | --- | --- |",
    ]
    for rate in ordered:
        if not gather(rate, ("portability", "kappa_initial")):
            continue
        lines.append(
            f"| {rate} | {summarize(gather(rate, ('portability', 'kappa_initial')))} | "
            f"{summarize(gather(rate, ('portability', 'kappa_final')))} | "
            f"{summarize(gather(rate, ('portability', 'prior_weight_variance')))} | "
            f"{summarize(gather(rate, ('portability', 'final_weight_variance')))} |"
        )
    lines += [
        "",
        "| lambda | kappa | quantile gap | p50 | p90 | p99 | sd(log a) | "
        "rank corr | T | alpha | NLL | ECE | accuracy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rate in ordered:
        marker = " **" if rate == selected else " "
        lines.append(
            f"|{marker}{rate}{marker.strip()} | "
            f"{summarize(gather(rate, ('portability', 'kappa_initial')))} | "
            f"{summarize(gather(rate, ('quantile_disagreement',)))} | "
            f"{summarize(gather(rate, ('levels', 'learned_p50')))} | "
            f"{summarize(gather(rate, ('levels', 'learned_p90')))} | "
            f"{summarize(gather(rate, ('levels', 'learned_p99')))} | "
            f"{summarize(gather(rate, ('levels', 'learned_log_dispersion')))} | "
            f"{summarize(gather(rate, ('noise_recovery', 'spearman_per_class')))} | "
            f"{summarize(gather(rate, ('variants', 'joint', 'temperature')))} | "
            f"{summarize(gather(rate, ('variants', 'joint', 'alpha')))} | "
            f"{summarize(gather(rate, ('variants', 'joint', 'nll')))} | "
            f"{summarize(gather(rate, ('variants', 'joint', 'ece')))} | "
            f"{summarize(gather(rate, ('variants', 'joint', 'accuracy')))} |"
        )

    baselines = results.get("baselines", {})
    if baselines:
        lines += [
            "",
            "## Test metrics against the frozen teacher",
            "",
            "| model | accuracy | NLL | ECE | Brier |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, values in baselines.items():
            lines.append(
                f"| {name} | " + " | ".join(f"{values[k]:.4f}" for k in REPORTED_METRICS) + " |"
            )
        for control in VARIANCE_CONTROLS:
            row = [
                summarize(gather(selected, ("variance_controls", control, key)))
                for key in REPORTED_METRICS
            ]
            lines.append(
                f"| logit_tagiv, {control} (lambda {selected}) | " + " | ".join(row) + " |"
            )
    destination = args.output / args.dataset / "SHRINKAGE_STUDY.md"
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


# ======================================================================
#  Channel aggregation diagnostic
# ======================================================================


def channel_aggregations(
    logit_mean: torch.Tensor, aleatoric: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return per-example summaries of the learned variance across classes.

    The predicted class channel carries the largest target and therefore the
    largest early residual, so it is the channel most exposed to storing mean
    error as noise. Splitting it off from the rest says whether an aggregate
    reversal is caused by that one channel alone.
    """

    classes = aleatoric.shape[-1]
    rows = torch.arange(aleatoric.shape[0], device=aleatoric.device)
    top_index = logit_mean.argmax(dim=-1)
    top = aleatoric[rows, top_index]
    rest = (aleatoric.sum(dim=-1) - top) / (classes - 1)
    ordered = aleatoric.sort(dim=-1).values
    trim = max(1, classes // 10)
    return {
        "mean": aleatoric.mean(dim=-1),
        "top": top,
        "rest": rest,
        "median": aleatoric.median(dim=-1).values,
        "trimmed_mean": ordered[:, trim : classes - trim].mean(dim=-1),
    }


def channels(args: argparse.Namespace) -> None:
    """Score every channel aggregation against SVHN and against CIFAR-C severity."""

    classifier, _ = TAGILastLayerClassifier.load(
        head_path(args.output, args.dataset, args.regime, args.seed), device=args.device
    )
    root = feature_root(args.study_root, args.dataset)

    def summaries(path: Path) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        shard = load_feature_shard(path)
        features = shard["features"].float()
        logit_mean, _, aleatoric = classifier.logit_moments(
            features, batch_size=args.prediction_batch_size
        )
        aggregations = channel_aggregations(logit_mean, aleatoric)
        aggregations["predictive_entropy"] = predictive_entropy(
            classifier.predict(features).probabilities
        )
        return aggregations, shard["logits"].float().to(logit_mean.device)

    clean, clean_logits = summaries(root / "test.pt")
    outlier, outlier_logits = summaries(root / "svhn.pt")
    payload: dict[str, Any] = {
        "config": {
            "dataset": args.dataset,
            "regime": args.regime,
            "seed": args.seed,
            "classes": classifier.num_classes,
        },
        "clean": {name: float(value.mean()) for name, value in clean.items()},
        "svhn_auroc": {
            name: ood_detection_metrics_full(clean[name], outlier[name])["auroc"] for name in clean
        }
        | {
            name: scores["auroc"]
            for name, scores in teacher_ood_scores(clean_logits, outlier_logits).items()
        },
        "severity": {},
    }
    for severity in range(1, 6):
        levels: dict[str, list[float]] = {name: [] for name in clean}
        aurocs: dict[str, list[float]] = {name: [] for name in clean}
        for corruption in CANONICAL_CORRUPTIONS:
            path = corruption_shard_path(args.study_root, args.dataset, corruption, severity)
            if not path.exists():
                continue
            corrupted, corrupted_logits = summaries(path)
            for name in clean:
                levels[name].append(float(corrupted[name].mean()))
                aurocs[name].append(
                    ood_detection_metrics_full(clean[name], corrupted[name])["auroc"]
                )
            for name, scores in teacher_ood_scores(clean_logits, corrupted_logits).items():
                levels.setdefault(name, []).append(float("nan"))
                aurocs.setdefault(name, []).append(scores["auroc"])
        if not levels["mean"]:
            continue
        payload["severity"][str(severity)] = {
            "level": {name: statistics.mean(values) for name, values in levels.items()},
            "auroc": {name: statistics.mean(values) for name, values in aurocs.items()},
        }
        print(
            f"severity {severity}: "
            + " ".join(
                f"{name}={statistics.mean(levels[name]):.4f}/{statistics.mean(aurocs[name]):.3f}"
                for name in ("top", "rest", "median")
            )
        )

    destination = args.output / args.dataset / "channels.json"
    atomic_json(destination, payload)
    lines = [
        f"# Channel aggregation of the learned variance, {args.dataset}",
        "",
        f"One calibrated `{args.regime}` head. Each row is a different way of "
        "reducing the K learned variances to one number per example. `top` is the "
        "predicted class channel, `rest` the mean of the other K-1.",
        "",
        "| aggregation | clean level | SVHN AUROC | "
        + " | ".join(f"s{severity} level" for severity in range(1, 6))
        + " | "
        + " | ".join(f"s{severity} AUROC" for severity in range(1, 6))
        + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in range(10)) + " |",
    ]
    for name in (
        "mean",
        "top",
        "rest",
        "median",
        "trimmed_mean",
        "predictive_entropy",
        "teacher_msp",
        "teacher_entropy",
        "teacher_energy",
    ):
        if name not in payload["svhn_auroc"]:
            continue

        def cell(severity: int, field: str, name: str = name) -> str:
            if str(severity) not in payload["severity"]:
                return "-"
            value = payload["severity"][str(severity)][field].get(name, float("nan"))
            return "-" if value != value else f"{value:.4f}"

        levels = " | ".join(cell(severity, "level") for severity in range(1, 6))
        aurocs = " | ".join(cell(severity, "auroc") for severity in range(1, 6))
        clean_level = payload["clean"].get(name, float("nan"))
        rendered = "-" if clean_level != clean_level else f"{clean_level:.4f}"
        lines.append(
            f"| {name} | {rendered} | {payload['svhn_auroc'][name]:.3f} | {levels} | {aurocs} |"
        )
    report_path = args.output / args.dataset / "CHANNELS.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination} and {report_path}")


# ======================================================================
#  Reporting
# ======================================================================


def summarize(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def collect(results: dict[str, Any], path: tuple[str, ...]) -> list[float]:
    collected = []
    for record in results["seeds"].values():
        node: Any = record
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, (int, float)):
            collected.append(float(node))
    return collected


def report(args: argparse.Namespace) -> None:
    source = args.output / args.dataset / "results.json"
    results = json.loads(source.read_text())
    lines = [
        "# Logit-space TAGI-V distillation on frozen CIFAR features",
        "",
        f"Dataset `{args.dataset}`, {len(results['seeds'])} seed(s), "
        f"{results['config']['epochs']} epochs.",
        "",
        "## Test metrics",
        "",
        "| model | accuracy | NLL | ECE | Brier |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, values in results["baselines"].items():
        lines.append(
            f"| {name} | " + " | ".join(f"{values[key]:.4f}" for key in REPORTED_METRICS) + " |"
        )
    rows = [("categorical_tagiv (labels)", ("categorical_tagiv",))]
    for regime in REGIMES:
        for variant in ("temperature_only", "fixed_multiplier", "joint"):
            rows.append((f"logit_tagiv {regime} / {variant}", (regime, "variants", variant)))
    for label, path in rows:
        values = [collect(results, (*path, key)) for key in REPORTED_METRICS]
        if not values[0]:
            continue
        lines.append(f"| {label} | " + " | ".join(summarize(value) for value in values) + " |")

    lines += [
        "",
        "## Calibration and learned noise",
        "",
        "| regime | T | alpha | mean aleatoric | mean epistemic | logit RMSE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for regime in REGIMES:
        temperature = collect(results, (regime, "variants", "joint", "temperature"))
        if not temperature:
            continue
        lines.append(
            f"| {regime} | {summarize(temperature)} | "
            f"{summarize(collect(results, (regime, 'variants', 'joint', 'alpha')))} | "
            f"{summarize(collect(results, (regime, 'moments', 'test_aleatoric')))} | "
            f"{summarize(collect(results, (regime, 'moments', 'test_epistemic')))} | "
            f"{summarize(collect(results, (regime, 'moments', 'logit_rmse')))} |"
        )

    recovery = collect(results, ("augmented", "noise_recovery", "spearman_per_class"))
    if recovery:

        def noise(key: str) -> str:
            return summarize(collect(results, ("augmented", "noise_recovery", key)))

        lines += [
            "",
            "## Noise recovery, augmented regime",
            "",
            "Learned aleatoric variance against the measured spread of the "
            "teacher's logits over augmentation repeats, on the held-out "
            "validation split. The per (example, class) rank correlation is the "
            "primary statistic: with many classes the class average is dominated "
            "by the observed class channel, whose learned variance is the least "
            "reliable, and that summary can invert a correlation that is positive "
            "channel by channel.",
            "",
            "| statistic | value |",
            "| --- | --- |",
            f"| rank correlation, per (example, class) | {noise('spearman_per_class')} |",
            f"| rank correlation, observed class channel | {noise('spearman_observed_class')} |",
            f"| rank correlation, class averaged | {noise('spearman_class_mean')} |",
            f"| observed class, learned median | {noise('learned_observed_class_median')} |",
            f"| observed class, measured median | {noise('measured_observed_class_median')} |",
            f"| observed class, learned p90 | {noise('learned_observed_class_p90')} |",
            f"| observed class, measured p90 | {noise('measured_observed_class_p90')} |",
            f"| observed class, sd(log a) learned | {noise('learned_log_dispersion')} |",
            f"| observed class, sd(log R) measured | {noise('measured_log_dispersion')} |",
            f"| learned mean, all channels | {noise('learned_mean')} |",
            f"| measured mean, all channels | {noise('measured_mean')} |",
            f"| learned mean, observed class | {noise('learned_observed_class_mean')} |",
            f"| measured mean, observed class | {noise('measured_observed_class_mean')} |",
        ]

    lines += [
        "",
        "## Variance controls, augmented regime",
        "",
        "Each control refits its own (T, alpha) on the calibration half. "
        "`constant` replaces the learned variance with its calibration-split "
        "class mean, which keeps the noise level and discards the input "
        "dependence. `shuffled` permutes the learned variance across examples, "
        "which keeps its marginal and discards the association with the input; "
        "the SVHN permutation runs over the pooled test and SVHN rows, since "
        "permuting inside each split would leave both marginals, and therefore "
        "the AUROC, untouched.",
        "",
        "| control | alpha | NLL | ECE | SVHN AUROC (entropy) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for regime in REGIMES:
        for control in VARIANCE_CONTROLS:
            path = (regime, "variance_controls", control)
            alpha = collect(results, (*path, "alpha"))
            if not alpha:
                continue
            lines.append(
                f"| {regime} / {control} | {summarize(alpha)} | "
                f"{summarize(collect(results, (*path, 'nll')))} | "
                f"{summarize(collect(results, (*path, 'ece')))} | "
                f"{summarize(collect(results, (*path, 'entropy_auroc')))} |"
            )

    lines += [
        "",
        "## SVHN detection (AUROC)",
        "",
        "| score | AUROC |",
        "| --- | --- |",
    ]
    scored: list[tuple[str, tuple[str, ...]]] = [
        (
            "categorical_tagiv (labels), predictive entropy",
            ("categorical_tagiv", "ood", "predictive_entropy"),
        ),
        ("teacher MSP", ("augmented", "ood", "teacher_msp")),
        ("teacher entropy", ("augmented", "ood", "teacher_entropy")),
        ("teacher energy", ("augmented", "ood", "teacher_energy")),
    ]
    for regime in REGIMES:
        for label, key in (
            ("predictive entropy after (T, alpha)", "predictive_entropy"),
            ("learned aleatoric variance", "aleatoric_variance"),
            ("epistemic variance", "epistemic_variance"),
        ):
            scored.append((f"logit_tagiv {regime}: {label}", (regime, "ood", key)))
    for label, path in scored:
        values = collect(results, (*path, "auroc"))
        if values:
            lines.append(f"| {label} | {summarize(values)} |")

    destination = args.output / args.dataset / "REPORT.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    common.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    common.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    cacher = subparsers.add_parser(
        "cache", parents=[common], help="extract repeated augmented teacher logits"
    )
    cacher.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    cacher.add_argument("--repeats", type=int, default=8)
    cacher.add_argument("--batch-size", type=int, default=512)
    cacher.add_argument("--workers", type=int, default=4)
    cacher.add_argument("--seed", type=int, default=0)
    cacher.add_argument("--device", default="cuda")

    runner = subparsers.add_parser(parents=[common], name="run", help="train and evaluate")
    runner.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    runner.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    runner.add_argument("--epochs", type=int, default=5)
    runner.add_argument("--batch-size", type=int, default=256)
    runner.add_argument("--prediction-batch-size", type=int, default=4096)
    runner.add_argument("--gain", type=float, default=0.1)
    runner.add_argument("--aleatoric-init", type=float, default=0.1)
    runner.add_argument("--variance-cv", type=float, default=0.5)
    runner.add_argument("--variance-weight-share", type=float, default=0.5)
    runner.add_argument("--num-samples", type=int, default=256)
    runner.add_argument("--decomposition-samples", type=int, default=64)
    runner.add_argument("--device", default="cuda")

    subparsers.add_parser("report", parents=[common], help="write the markdown summary")

    corrupter = subparsers.add_parser(
        "corruptions", parents=[common], help="sweep CIFAR-C severity with one head"
    )
    corrupter.add_argument("--regime", default="augmented")
    corrupter.add_argument("--seed", type=int, default=0)
    corrupter.add_argument("--corruption-root", type=Path)
    corrupter.add_argument("--prediction-batch-size", type=int, default=1024)
    corrupter.add_argument("--augmentation-repeats", type=int, default=0)
    corrupter.add_argument(
        "--measure-corruptions",
        nargs="+",
        default=["gaussian_noise", "fog", "jpeg_compression"],
        help="corruptions to re-measure the teacher's augmentation spread on",
    )
    corrupter.add_argument("--workers", type=int, default=4)
    corrupter.add_argument("--device", default="cuda")

    subparsers.add_parser("corruption-report", parents=[common], help="write the severity summary")

    channeller = subparsers.add_parser(
        "channels", parents=[common], help="score channel aggregations of a(x)"
    )
    channeller.add_argument("--regime", default="augmented")
    channeller.add_argument("--seed", type=int, default=0)
    channeller.add_argument("--prediction-batch-size", type=int, default=2048)
    channeller.add_argument("--device", default="cuda")

    study = subparsers.add_parser(
        "variance-study", parents=[common], help="compare cold start against mean-first fits"
    )
    study.add_argument(
        "--methods", nargs="+", choices=VARIANCE_METHODS, default=list(VARIANCE_METHODS)
    )
    study.add_argument("--seeds", nargs="+", type=int, default=[0])
    study.add_argument("--epochs", type=int, default=5)
    study.add_argument("--mean-epochs", type=int, default=20)
    study.add_argument("--variance-epochs", type=int, default=20)
    study.add_argument("--mean-observation-variance", type=float, default=0.01)
    study.add_argument("--batch-size", type=int, default=256)
    study.add_argument("--prediction-batch-size", type=int, default=2048)
    study.add_argument("--gain", type=float, default=0.1)
    study.add_argument("--aleatoric-init", type=float, default=0.1)
    study.add_argument("--variance-cv", type=float, default=0.5)
    study.add_argument("--variance-weight-share", type=float, default=0.5)
    study.add_argument("--num-samples", type=int, default=256)
    study.add_argument("--decomposition-samples", type=int, default=32)
    study.add_argument("--device", default="cuda")

    subparsers.add_parser(
        "variance-study-report", parents=[common], help="write the variance-study summary"
    )

    power = subparsers.add_parser(
        "power-diagnostic", parents=[common], help="sweep variance compression on a saved head"
    )
    power.add_argument("--regime", default="study_warmup_replicate")
    power.add_argument("--seed", type=int, default=0)
    power.add_argument("--powers", nargs="+", type=float, default=list(POWERS))
    power.add_argument("--prediction-batch-size", type=int, default=2048)
    power.add_argument("--device", default="cuda")

    shrink = subparsers.add_parser(
        "shrinkage-study", parents=[common], help="sweep persistent variance-weight shrinkage"
    )
    shrink.add_argument("--rates", nargs="+", type=float, default=list(SHRINKAGE_RATES))
    shrink.add_argument(
        "--kappas",
        nargs="+",
        type=float,
        help="dimensionless kappa_0 = lambda * v_Wg,0; converted per dataset and "
        "overriding --rates. This is the grid that transfers between datasets.",
    )
    shrink.add_argument("--seeds", nargs="+", type=int, default=[0])
    shrink.add_argument("--mean-epochs", type=int, default=20)
    shrink.add_argument("--variance-epochs", type=int, default=20)
    shrink.add_argument("--mean-observation-variance", type=float, default=0.01)
    shrink.add_argument("--batch-size", type=int, default=256)
    shrink.add_argument("--prediction-batch-size", type=int, default=2048)
    shrink.add_argument("--gain", type=float, default=0.1)
    shrink.add_argument("--aleatoric-init", type=float, default=0.1)
    shrink.add_argument("--variance-cv", type=float, default=0.5)
    shrink.add_argument("--variance-weight-share", type=float, default=0.5)
    shrink.add_argument("--num-samples", type=int, default=256)
    shrink.add_argument("--decomposition-samples", type=int, default=32)
    shrink.add_argument("--epochs", type=int, default=5)
    shrink.add_argument("--device", default="cuda")

    subparsers.add_parser(
        "shrinkage-study-report", parents=[common], help="write the shrinkage summary"
    )

    args = parser.parse_args()
    if args.command == "cache":
        cache_augmented_logits(args)
    elif args.command == "run":
        run(args)
    elif args.command == "corruptions":
        corruptions(args)
    elif args.command == "corruption-report":
        corruption_report(args)
    elif args.command == "channels":
        channels(args)
    elif args.command == "variance-study":
        variance_study(args)
    elif args.command == "variance-study-report":
        variance_study_report(args)
    elif args.command == "power-diagnostic":
        power_diagnostic(args)
    elif args.command == "shrinkage-study":
        shrinkage_study(args)
    elif args.command == "shrinkage-study-report":
        shrinkage_study_report(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
