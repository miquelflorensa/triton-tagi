"""Frozen torchvision ResNet-18 plus a 1,000-class TAGI last layer.

The ``cache`` command expects a prepared ImageFolder tree with ``train`` and
``val`` directories. It downloads/loads torchvision's official ImageNet-1K
ResNet-18 weights and writes sharded 512-dimensional features. The ``run``
command trains only the Bayesian linear layer with either an AGCI or HRC head.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import (
    A_STAR,
    TAGILastLayerClassifier,
    agci_weight_gain_from_kappa,
    classification_metrics,
    fit_softmax_temperature,
    ood_detection_metrics_full,
)
from triton_tagi.cifar_study import load_feature_shard, save_feature_shard


DEFAULT_ROOT = REPOSITORY_ROOT / "runs/imagenet/resnet18_agci"

# Heads that condition Gaussian utilities on the observed argmax event and so
# share the kappa prior parameterization and the zero-mean construction.
_AGCI_RUNNER_HEADS = ("agci", "gumbel_agci", "logit_site", "ct_agci")
# Links whose utility unit is a gauge fixed at one rather than a tau.
_UNIT_SCALE_HEADS = ("gumbel_agci", "logit_site", "ct_agci")

# The TAGI cap threshold is sqrt(S)/cap_factor; this makes it unreachable
# (~4e10 for the gain-1.0 prior) without dividing by zero in the kernel.
_CAP_DISABLED = 1e-12


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def resolve_weights(name: str) -> ResNet18_Weights:
    try:
        return ResNet18_Weights[name]
    except KeyError as error:
        raise ValueError(
            f"unknown ResNet-18 weights {name!r}; expected one of "
            f"{[item.name for item in ResNet18_Weights]}"
        ) from error


class FeatureExtractor(nn.Module):
    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.features = nn.Sequential(*list(network.children())[:-1])
        self.classifier = network.fc

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        features = torch.flatten(self.features(inputs), 1)
        return features, self.classifier(features)


@torch.inference_mode()
def extract_subset(
    model: FeatureExtractor,
    dataset: ImageFolder,
    indices: range,
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, Tensor]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    feature_parts, logit_parts, label_parts = [], [], []
    for inputs, labels in loader:
        features, logits = model(inputs.to(device, non_blocking=True))
        feature_parts.append(features.cpu())
        logit_parts.append(logits.cpu())
        label_parts.append(torch.as_tensor(labels).long().cpu())
    return {
        "features": torch.cat(feature_parts),
        "logits": torch.cat(logit_parts),
        "labels": torch.cat(label_parts),
    }


def cache_split(
    split: str,
    dataset: ImageFolder,
    model: FeatureExtractor,
    output: Path,
    *,
    weights: ResNet18_Weights,
    device: torch.device,
    batch_size: int,
    workers: int,
    shard_size: int,
    max_images: int | None,
    force: bool,
) -> list[dict[str, Any]]:
    count = len(dataset) if max_images is None else min(len(dataset), max_images)
    records = []
    for shard_index, start in enumerate(range(0, count, shard_size)):
        stop = min(start + shard_size, count)
        path = output / f"{split}-{shard_index:04d}.pt"
        if path.exists() and not force:
            payload = load_feature_shard(path)
        else:
            payload = extract_subset(
                model,
                dataset,
                range(start, stop),
                device=device,
                batch_size=batch_size,
                workers=workers,
            )
            metadata = {
                "dataset": "imagenet1k",
                "split": split,
                "start": start,
                "stop": stop,
                "weights": weights.name,
                "weights_url": weights.url,
                "transform": repr(weights.transforms()),
            }
            save_feature_shard(path, payload, metadata)
            payload["metadata"] = metadata
        records.append(
            {
                "path": str(path.resolve()),
                "start": start,
                "stop": stop,
                "count": payload["labels"].numel(),
            }
        )
        print(f"cached {split} [{start}:{stop}] -> {path}", flush=True)
    return records


def cache_command(args: argparse.Namespace) -> None:
    weights = resolve_weights(args.weights)
    transform = weights.transforms()
    train_root, val_root = args.data_root / "train", args.data_root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError(
            f"expected ImageFolder directories {train_root} and {val_root}"
        )
    train = ImageFolder(train_root, transform=transform)
    validation = ImageFolder(val_root, transform=transform)
    if len(train.classes) != 1000 or train.class_to_idx != validation.class_to_idx:
        raise ValueError("train/val must contain the same 1,000 ImageNet classes")

    device = torch.device(args.device)
    network = resnet18(weights=weights).to(device).eval()
    model = FeatureExtractor(network).to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    fc_path = args.output / "pretrained_fc.pt"
    torch.save(
        {
            "weight": network.fc.weight.detach().cpu(),
            "bias": network.fc.bias.detach().cpu(),
            "weights": weights.name,
        },
        fc_path,
    )
    started = time.perf_counter()
    train_records = cache_split(
        "train", train, model, args.output,
        weights=weights, device=device, batch_size=args.batch_size,
        workers=args.workers, shard_size=args.shard_size,
        max_images=args.max_train_images, force=args.force,
    )
    val_records = cache_split(
        "val", validation, model, args.output,
        weights=weights, device=device, batch_size=args.batch_size,
        workers=args.workers, shard_size=args.shard_size,
        max_images=args.max_val_images, force=args.force,
    )
    atomic_json(
        args.output / "manifest.json",
        {
            "weights": weights.name,
            "weights_url": weights.url,
            "data_root": str(args.data_root.resolve()),
            "classes": len(train.classes),
            "class_to_idx": train.class_to_idx,
            "pretrained_fc": str(fc_path.resolve()),
            "train": train_records,
            "val": val_records,
            "wall_s": time.perf_counter() - started,
        },
    )


def list_shards(root: Path, split: str) -> list[Path]:
    paths = sorted(root.glob(f"{split}-*.pt"))
    if not paths:
        raise FileNotFoundError(f"no {split} feature shards under {root}")
    return paths


def feature_statistics(paths: list[Path]) -> tuple[Tensor, float, int]:
    """Compute the training mean and mean centered energy per dimension."""

    total = None
    squared_norm_sum = 0.0
    count = 0
    for index, path in enumerate(paths, 1):
        features = load_feature_shard(path)["features"].double()
        if total is None:
            total = torch.zeros(features.shape[1], dtype=torch.float64)
        total.add_(features.sum(dim=0))
        squared_norm_sum += features.square().sum().item()
        count += features.shape[0]
        print(f"statistics {index}/{len(paths)} ({path.name})", flush=True)
    assert total is not None and count > 0
    mean = total / count
    centered_squared_norm = squared_norm_sum - count * mean.square().sum().item()
    energy = centered_squared_norm / (count * mean.numel())
    return mean.float(), energy, count


def load_or_compute_feature_statistics(
    paths: list[Path], cache_path: Path
) -> tuple[Tensor, float, int]:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        expected_names = [path.name for path in paths]
        if payload.get("shards") == expected_names:
            return (
                payload["mean"].float(),
                float(payload["energy"]),
                int(payload["count"]),
            )
    mean, energy, count = feature_statistics(paths)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean,
            "energy": energy,
            "count": count,
            "shards": [path.name for path in paths],
        },
        cache_path,
    )
    return mean, energy, count


def initialize_pretrained_mean(
    classifier: TAGILastLayerClassifier,
    feature_root: Path,
    variance_scale: float = 1.0,
) -> None:
    payload = torch.load(
        feature_root / "pretrained_fc.pt", map_location="cpu", weights_only=False
    )
    weight, bias = payload["weight"], payload["bias"]
    if weight.shape != classifier.linear.mw.T.shape:
        raise ValueError("pretrained classifier shape does not match AGCI head")
    classifier.linear.mw.copy_(weight.T.to(classifier.device))
    adjusted_bias = bias.to(classifier.device)
    if classifier.feature_mean is not None:
        adjusted_bias = adjusted_bias + weight.to(classifier.device).matmul(
            classifier.feature_mean.squeeze(0)
        )
    classifier.linear.mb.copy_(adjusted_bias.unsqueeze(0))
    if variance_scale != 1.0:
        # The default prior claims sd comparable to the weights themselves,
        # which lets the first updates move means we already trust.
        classifier.linear.Sw.mul_(variance_scale)
        classifier.linear.Sb.mul_(variance_scale)


@torch.no_grad()
def evaluate_shards(
    classifier: TAGILastLayerClassifier,
    paths: list[Path],
    prediction_batch_size: int,
    *,
    calibration_per_class: int = 0,
    softmax_temperature: float = 1.0,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    probability_parts, label_parts, variance_parts, baseline_parts = [], [], [], []
    seen = torch.zeros(1000, dtype=torch.long)
    for path in paths:
        shard = load_feature_shard(path)
        mask = validation_partition_mask(
            shard["labels"], seen, calibration_per_class, calibration=False
        )
        features = shard["features"][mask]
        for batch_start in range(0, features.shape[0], prediction_batch_size):
            prediction = classifier.predict(
                features[batch_start : batch_start + prediction_batch_size]
            )
            probability_parts.append(prediction.probabilities.cpu())
            variance_parts.append(prediction.output_variance.mean(1).cpu())
        label_parts.append(shard["labels"][mask])
        baseline_parts.append(shard["logits"][mask])
        print(f"predicted {path.name}", flush=True)
    probabilities = torch.cat(probability_parts)
    labels = torch.cat(label_parts)
    variance = torch.cat(variance_parts)
    baseline_logits = torch.cat(baseline_parts)
    baseline = torch.softmax(baseline_logits, dim=1)
    calibrated_baseline = torch.softmax(
        baseline_logits / softmax_temperature, dim=1
    )
    errors = probabilities.argmax(1) != labels
    variance_error = (
        ood_detection_metrics_full(variance[~errors], variance[errors])
        if bool(errors.any()) and bool((~errors).any())
        else {}
    )
    return (
        classification_metrics(probabilities, labels),
        classification_metrics(baseline, labels),
        classification_metrics(calibrated_baseline, labels),
        {
            "mean": variance.mean().item(),
            **{f"error_{key}": value for key, value in variance_error.items()},
        },
    )


def validation_partition_mask(
    labels: Tensor,
    seen: Tensor,
    calibration_per_class: int,
    *,
    calibration: bool,
) -> Tensor:
    """Select a deterministic per-class calibration set or its complement."""

    mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    for class_index in labels.unique().tolist():
        indices = torch.nonzero(labels == class_index, as_tuple=False).flatten()
        already_seen = int(seen[class_index])
        needed = max(0, calibration_per_class - already_seen)
        take = min(needed, indices.numel())
        if calibration:
            mask[indices[:take]] = True
        else:
            mask[indices[take:]] = True
        seen[class_index] += indices.numel()
    return mask


def load_calibration_cohort(
    paths: list[Path], per_class: int
) -> dict[str, Tensor]:
    """Return the same number of validation examples from every class."""

    feature_parts, logit_parts, label_parts = [], [], []
    seen = torch.zeros(1000, dtype=torch.long)
    for path in paths:
        shard = load_feature_shard(path)
        mask = validation_partition_mask(
            shard["labels"], seen, per_class, calibration=True
        )
        feature_parts.append(shard["features"][mask])
        logit_parts.append(shard["logits"][mask])
        label_parts.append(shard["labels"][mask])
    labels = torch.cat(label_parts)
    expected = per_class * 1000
    if labels.numel() != expected:
        raise ValueError(
            f"expected {expected} calibration examples, found {labels.numel()}"
        )
    return {
        "features": torch.cat(feature_parts),
        "logits": torch.cat(logit_parts),
        "labels": labels,
    }


@torch.no_grad()
def evaluate_diagnostic(
    classifier: TAGILastLayerClassifier,
    cohort: dict[str, Tensor],
    prediction_batch_size: int,
) -> tuple[dict[str, float], float]:
    """Score the balanced calibration cohort after each training epoch."""

    features, labels = cohort["features"], cohort["labels"]
    probability_parts, variance_parts = [], []
    for batch_start in range(0, features.shape[0], prediction_batch_size):
        prediction = classifier.predict(
            features[batch_start : batch_start + prediction_batch_size]
        )
        probability_parts.append(prediction.probabilities.cpu())
        variance_parts.append(prediction.output_variance.mean(1).cpu())
    probabilities = torch.cat(probability_parts)
    variance = torch.cat(variance_parts)
    errors = probabilities.argmax(1) != labels
    variance_error = (
        ood_detection_metrics_full(variance[~errors], variance[errors])
        if bool(errors.any()) and bool((~errors).any())
        else {}
    )
    temperature = fit_softmax_temperature(cohort["logits"], labels)
    baseline = torch.softmax(cohort["logits"], dim=1)
    calibrated_baseline = torch.softmax(cohort["logits"] / temperature, dim=1)
    return {
        **{
            f"val_{key}": value
            for key, value in classification_metrics(probabilities, labels).items()
        },
        "val_variance_mean": variance.mean().item(),
        **{
            f"val_variance_error_{key}": value
            for key, value in variance_error.items()
        },
        **{
            f"softmax_{key}": value
            for key, value in classification_metrics(baseline, labels).items()
        },
        **{
            f"temperature_softmax_{key}": value
            for key, value in classification_metrics(
                calibrated_baseline, labels
            ).items()
        },
        "softmax_temperature": temperature,
    }, temperature


def run_command(args: argparse.Namespace) -> None:
    train_paths = list_shards(args.feature_root, "train")
    validation_root = args.validation_root or args.feature_root
    val_paths = list_shards(validation_root, "val")
    first = load_feature_shard(train_paths[0])
    input_dim, num_classes = first["features"].shape[1], first["logits"].shape[1]
    if num_classes != 1000:
        raise ValueError(f"expected 1,000 ImageNet classes, found {num_classes}")
    if not 0 <= args.calibration_per_class < 50:
        raise ValueError("calibration-per-class must be in [0, 49]")
    if args.bias_gain < 0.0 or not math.isfinite(args.bias_gain):
        raise ValueError("bias-gain must be finite and nonnegative")
    if args.tau <= 0.0 or not math.isfinite(args.tau):
        raise ValueError("tau must be finite and positive")
    if args.head not in _AGCI_RUNNER_HEADS and args.kappa is not None:
        raise ValueError("kappa parameterization is only available for AGCI")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    feature_mean, feature_energy, training_count = load_or_compute_feature_statistics(
        train_paths, args.feature_statistics
    )
    if not args.center_features:
        feature_energy += feature_mean.square().mean().item()
        feature_mean = None
    target_kappa = 1.0 if args.kappa is None and args.gain is None else args.kappa
    # beta is the Gumbel utility unit exactly as tau is the Gaussian one, so
    # kappa keeps its dimensionless meaning across both decision-noise laws.
    noise_unit = args.gumbel_beta if args.head in _UNIT_SCALE_HEADS else args.tau
    if args.head in _AGCI_RUNNER_HEADS and args.gain is None:
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
        gain_w = 1.0 if args.gain is None else args.gain
        if gain_w < 0.0 or not math.isfinite(gain_w):
            raise ValueError("gain must be finite and nonnegative")
        prior_parameterization = "explicit_gain"
    gain_b = args.bias_gain
    expected_prior_utility_variance = (
        gain_w**2 * feature_energy + gain_b**2 / input_dim
    )
    actual_kappa = math.sqrt(expected_prior_utility_variance) / noise_unit

    classifier_kwargs: dict[str, Any] = {
        "head": args.head,
        "device": args.device,
        "gain_w": gain_w,
        "gain_b": gain_b,
        "feature_mean": feature_mean,
    }
    if args.head == "agci":
        classifier_kwargs.update(
            agci_tau=args.tau,
            agci_num_quad=args.num_quad,
            agci_class_chunk_size=args.class_chunk_size,
        )
    elif args.head == "ct_agci":
        # The Core-Tail link shares the unit utility gauge with the logit
        # links, so gumbel_beta names it here too.
        classifier_kwargs.update(
            core_tail_beta=args.gumbel_beta,
            core_tail_a_star=args.core_tail_a_star,
        )
    elif args.head in _UNIT_SCALE_HEADS:
        classifier_kwargs.update(
            gumbel_beta=args.gumbel_beta,
            gumbel_num_samples=args.gumbel_num_samples,
            gumbel_seed=args.seed,
        )
    else:
        classifier_kwargs["sigma_v"] = args.sigma_v
    classifier = TAGILastLayerClassifier(
        input_dim,
        num_classes,
        **classifier_kwargs,
    )
    if args.no_cap:
        classifier.net.cap_factor_override = _CAP_DISABLED

    if args.mean_init == "pretrained":
        if args.head not in _AGCI_RUNNER_HEADS:
            raise ValueError(
                "pretrained mean initialization is only defined for the "
                "1,000-way AGCI heads"
            )
        initialize_pretrained_mean(
            classifier, validation_root, args.pretrained_var_scale
        )
    elif args.mean_init == "zero":
        classifier.linear.mw.zero_()
        classifier.linear.mb.zero_()

    args.output.mkdir(parents=True, exist_ok=True)
    cohort = (
        load_calibration_cohort(val_paths, args.calibration_per_class)
        if args.calibration_per_class > 0
        else None
    )
    history: list[dict[str, float]] = []
    softmax_temperature = 1.0

    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_paths = list(train_paths)
        random.Random(args.seed + epoch).shuffle(epoch_paths)
        for shard_index, path in enumerate(epoch_paths, 1):
            shard = load_feature_shard(path)
            generator = torch.Generator().manual_seed(
                args.seed * 1_000_003 + epoch * 10_007 + shard_index
            )
            order = torch.randperm(shard["labels"].numel(), generator=generator)
            for batch_start in range(0, order.numel(), args.batch_size):
                indices = order[batch_start : batch_start + args.batch_size]
                classifier.train_step(
                    shard["features"][indices], shard["labels"][indices]
                )
            print(
                f"epoch={epoch} trained {shard_index}/{len(epoch_paths)} "
                f"({path.name})",
                flush=True,
            )

        checkpoint = classifier.save(
            args.output / f"epoch_{epoch:04d}.pt",
            metadata={
                "epoch": epoch,
                "head": args.head,
                "mean_init": args.mean_init,
                "agci_update": (
                    "observed_event" if args.head in _AGCI_RUNNER_HEADS else None
                ),
                "decision_noise": (
                    "gumbel"
                    if args.head in _UNIT_SCALE_HEADS
                    else "gaussian"
                    if args.head == "agci"
                    else None
                ),
                "noise_unit": noise_unit if args.head in _AGCI_RUNNER_HEADS else None,
                "gumbel_num_samples": (
                    args.gumbel_num_samples if args.head == "gumbel_agci" else None
                ),
                "core_tail_a_star": (
                    args.core_tail_a_star if args.head == "ct_agci" else None
                ),
                "target_kappa": target_kappa,
                "actual_kappa": actual_kappa,
                "feature_energy": feature_energy,
            },
        )
        if cohort is not None:
            diagnostic, softmax_temperature = evaluate_diagnostic(
                classifier, cohort, args.prediction_batch_size
            )
            record = {
                "epoch": epoch,
                "wall_s": time.perf_counter() - started,
                **diagnostic,
            }
            history.append(record)
            atomic_json(args.output / "history.json", history)
            print(
                f"epoch={epoch} diagnostic " + json.dumps(record, sort_keys=True),
                flush=True,
            )

    if args.skip_final_eval:
        metrics, baseline_metrics, calibrated_baseline_metrics, variance = (
            {},
            {},
            {},
            {},
        )
    else:
        (
            metrics,
            baseline_metrics,
            calibrated_baseline_metrics,
            variance,
        ) = evaluate_shards(
            classifier,
            val_paths,
            args.prediction_batch_size,
            calibration_per_class=args.calibration_per_class,
            softmax_temperature=softmax_temperature,
        )
    validation_count = sum(
        load_feature_shard(path)["labels"].numel() for path in val_paths
    )
    result = {
        "config": {
            "head": args.head,
            "agci_update": (
                "observed_event" if args.head in _AGCI_RUNNER_HEADS else None
            ),
            "decision_noise": (
                "gumbel"
                if args.head in _UNIT_SCALE_HEADS
                else "gaussian"
                if args.head == "agci"
                else None
            ),
            "link_integration": (
                "posterior_predictive"
                if args.head == "gumbel_agci"
                else "prior_mean"
                if args.head == "logit_site"
                else None
            ),
            "noise_unit": noise_unit if args.head in _AGCI_RUNNER_HEADS else None,
            "gumbel_beta": (
                args.gumbel_beta if args.head == "gumbel_agci" else None
            ),
            "gumbel_num_samples": (
                args.gumbel_num_samples if args.head == "gumbel_agci" else None
            ),
            "core_tail_a_star": (
                args.core_tail_a_star if args.head == "ct_agci" else None
            ),
            "classes": num_classes,
            "input_dim": input_dim,
            "prior_parameterization": prior_parameterization,
            "target_kappa": target_kappa,
            "actual_kappa": actual_kappa,
            "gain_w": gain_w,
            "gain_b": gain_b,
            "feature_centering": args.center_features,
            "feature_energy": feature_energy,
            "expected_prior_utility_variance": expected_prior_utility_variance,
            "training_count": training_count,
            "sigma_v": args.sigma_v if args.head == "hrc" else None,
            "tau": args.tau if args.head == "agci" else None,
            "num_quad": args.num_quad if args.head == "agci" else None,
            "class_chunk_size": (
                args.class_chunk_size if args.head == "agci" else None
            ),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "prediction_batch_size": args.prediction_batch_size,
            "mean_init": args.mean_init,
            "pretrained_var_scale": args.pretrained_var_scale,
            "cap_disabled": bool(args.no_cap),
            "seed": args.seed,
        },
        "protocol": {
            "calibration_per_class": args.calibration_per_class,
            "calibration_count": args.calibration_per_class * num_classes,
            "heldout_validation_count": (
                validation_count - args.calibration_per_class * num_classes
            ),
            "selection_uses_heldout_validation": False,
        },
        "checkpoint": str(checkpoint.resolve()),
        "history": history,
        "validation": metrics,
        "pretrained_softmax_validation": baseline_metrics,
        "temperature_scaled_softmax": {
            "temperature": softmax_temperature,
            "validation": calibrated_baseline_metrics,
        },
        "output_variance": variance,
        "wall_s": time.perf_counter() - started,
    }
    atomic_json(args.output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def evaluate_command(args: argparse.Namespace) -> None:
    val_paths = list_shards(args.validation_root, "val")
    classifier, checkpoint_metadata = TAGILastLayerClassifier.load(
        args.checkpoint, device=args.device
    )
    classifier.agci_class_chunk_size = args.class_chunk_size
    cohort = load_calibration_cohort(val_paths, args.calibration_per_class)
    softmax_temperature = fit_softmax_temperature(
        cohort["logits"], cohort["labels"]
    )
    calibration_softmax = torch.softmax(cohort["logits"], dim=1)
    calibration_temperature_softmax = torch.softmax(
        cohort["logits"] / softmax_temperature, dim=1
    )
    started = time.perf_counter()
    (
        metrics,
        baseline_metrics,
        calibrated_baseline_metrics,
        variance,
    ) = evaluate_shards(
        classifier,
        val_paths,
        args.prediction_batch_size,
        calibration_per_class=args.calibration_per_class,
        softmax_temperature=softmax_temperature,
    )
    validation_count = sum(
        load_feature_shard(path)["labels"].numel() for path in val_paths
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata": checkpoint_metadata,
        "protocol": {
            "calibration_per_class": args.calibration_per_class,
            "calibration_count": args.calibration_per_class * 1000,
            "heldout_validation_count": (
                validation_count - args.calibration_per_class * 1000
            ),
            "selection_uses_heldout_validation": False,
        },
        "calibration": {
            "pretrained_softmax": classification_metrics(
                calibration_softmax, cohort["labels"]
            ),
            "temperature_scaled_softmax": classification_metrics(
                calibration_temperature_softmax, cohort["labels"]
            ),
        },
        "heldout_validation": {
            # Name the row after the checkpoint's own head so a Gaussian and a
            # Gumbel evaluation are never confused in the record.
            checkpoint_metadata.get("head") or "event_agci": metrics,
            "pretrained_softmax": baseline_metrics,
            "temperature_scaled_softmax": calibrated_baseline_metrics,
        },
        "softmax_temperature": softmax_temperature,
        "output_variance": variance,
        "wall_s": time.perf_counter() - started,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache")
    cache.add_argument("--data-root", type=Path, required=True)
    cache.add_argument("--output", type=Path, default=DEFAULT_ROOT / "features")
    cache.add_argument("--weights", default="IMAGENET1K_V1")
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--batch-size", type=int, default=256)
    cache.add_argument("--workers", type=int, default=8)
    cache.add_argument("--shard-size", type=int, default=10_000)
    cache.add_argument("--max-train-images", type=int)
    cache.add_argument("--max-val-images", type=int)
    cache.add_argument("--force", action="store_true")
    cache.set_defaults(function=cache_command)

    run = subparsers.add_parser("run")
    run.add_argument("--feature-root", type=Path, default=DEFAULT_ROOT / "features")
    run.add_argument(
        "--validation-root",
        type=Path,
        help="root containing val shards; defaults to --feature-root",
    )
    run.add_argument(
        "--feature-statistics",
        type=Path,
        default=DEFAULT_ROOT / "features_shuffled" / "feature_statistics.pt",
    )
    run.add_argument("--output", type=Path, default=DEFAULT_ROOT / "kappa_1")
    run.add_argument("--device", default="cuda")
    run.add_argument("--epochs", type=int, default=1)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--prediction-batch-size", type=int, default=16)
    run.add_argument(
        "--calibration-per-class",
        type=int,
        default=2,
        help="balanced validation examples/class used for kappa and temperature selection",
    )
    run.add_argument("--skip-final-eval", action="store_true")
    run.add_argument("--pretrained-var-scale", type=float, default=1.0)
    run.add_argument("--no-cap", action="store_true")
    prior_scale = run.add_mutually_exclusive_group()
    prior_scale.add_argument(
        "--kappa",
        type=float,
        help="target prior utility standard deviation divided by tau; defaults to 1",
    )
    prior_scale.add_argument(
        "--gain", type=float, help="explicit weight gain override"
    )
    run.add_argument(
        "--head",
        choices=("agci", "gumbel_agci", "logit_site", "ct_agci", "hrc"),
        default="agci",
    )
    run.add_argument(
        "--core-tail-a-star",
        type=float,
        default=A_STAR,
        help=(
            "Core-Tail interior coefficient; the default is derived from "
            "variance matching and 0 reduces the link to logit_site"
        ),
    )
    run.add_argument(
        "--gumbel-beta",
        type=float,
        default=1.0,
        help="Gumbel decision-noise scale; the utility unit for the logit link",
    )
    run.add_argument("--gumbel-num-samples", type=int, default=32)
    run.add_argument("--tau", type=float, default=1.0)
    run.add_argument(
        "--bias-gain",
        type=float,
        default=1.0,
        help="bias prior gain",
    )
    run.add_argument(
        "--no-center-features", dest="center_features", action="store_false"
    )
    run.set_defaults(center_features=True)
    run.add_argument(
        "--sigma-v",
        type=float,
        default=0.1,
        help="HRC training observation-noise standard deviation",
    )
    run.add_argument("--num-quad", type=int, default=48)
    run.add_argument(
        "--class-chunk-size",
        type=int,
        help="candidate classes per AGCI moment chunk; defaults to automatic",
    )
    run.add_argument(
        "--mean-init", choices=("random", "pretrained", "zero"), default="zero"
    )
    run.add_argument("--seed", type=int, default=0)
    run.set_defaults(function=run_command)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--validation-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--prediction-batch-size", type=int, default=16)
    evaluate.add_argument("--class-chunk-size", type=int, default=16)
    evaluate.add_argument("--calibration-per-class", type=int, default=2)
    evaluate.set_defaults(function=evaluate_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
