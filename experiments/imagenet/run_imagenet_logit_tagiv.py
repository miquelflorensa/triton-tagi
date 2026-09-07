"""Logit-space TAGI-V on frozen ImageNet features, under the frozen protocol.

The method, its hyperparameters, the dimensionless shrinkage grid, the selection
criterion and the success criteria are pre-registered in
``experiments/last_layer/PROTOCOL.md`` and are not adjusted here. The scoring
helpers are imported from the CIFAR runner rather than reimplemented, so that
every metric is defined identically across datasets.

The backbone is torchvision's ImageNet-1k ResNet-18. Features are read on the
evaluation transform; the replicate targets are the same backbone's logits under
the standard ImageNet training augmentation, reduced online to their mean and
unbiased sample variance so that only two tensors per split are stored.

Examples:
  python experiments/imagenet/run_imagenet_logit_tagiv.py cache --classes 100 --repeats 8
  python experiments/imagenet/run_imagenet_logit_tagiv.py run --classes 100 --seeds 0 1 2
  python experiments/imagenet/run_imagenet_logit_tagiv.py report --classes 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "experiments" / "last_layer"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Identical metric definitions to the CIFAR study, by construction. The module
# lives in experiments/last_layer, which is on sys.path above; this file is
# deliberately named differently so the import can never resolve to itself.
from run_logit_tagiv import (  # noqa: E402
    SHRINKAGE_KAPPAS,
    VARIANCE_CONTROLS,
    channel_aggregations,
    log_variance_error,
    quantile_disagreement,
    rank_correlation,
    substitute_variance,
    summarize,
    variance_quantiles,
)

from triton_tagi import (  # noqa: E402
    LOGIT_VARIANCE_FLOOR,
    TAGILastLayerClassifier,
    classification_metrics,
    fit_logit_calibration,
    fit_softmax_temperature,
    logit_feature_energy,
    logit_tagiv_predictive_probs,
    logit_target_scale,
    logit_variance_head_prior,
    logit_variance_prior_split,
)
from triton_tagi.cifar_study import seed_everything  # noqa: E402

DEFAULT_DATA = Path("/usr/local/share/imagenet/ILSVRC/Data/CLS-LOC")
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/imagenet/logit_tagiv"
REPORTED_METRICS = ("accuracy", "nll", "ece", "brier")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def study_root(output: Path, classes: int) -> Path:
    return output / f"imagenet{classes}"


def selected_classes(total: int, classes: int) -> list[int]:
    """Return a deterministic, evenly spaced subset of the 1000 wnid indices."""

    if classes == total:
        return list(range(total))
    if not 1 < classes < total:
        raise ValueError("classes must lie in (1, 1000]")
    step = total / classes
    return [min(total - 1, int(round(index * step))) for index in range(classes)]


def build_backbone(device: str):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).to(device).eval().requires_grad_(False)
    return model, weights


def forward_features(model, inputs: torch.Tensor) -> torch.Tensor:
    hidden = model.maxpool(model.relu(model.bn1(model.conv1(inputs))))
    hidden = model.layer4(model.layer3(model.layer2(model.layer1(hidden))))
    return torch.flatten(model.avgpool(hidden), 1)


def subset_indices(dataset: ImageFolder, keep: set[int]) -> tuple[list[int], dict[int, int]]:
    remap = {original: index for index, original in enumerate(sorted(keep))}
    indices = [i for i, (_, target) in enumerate(dataset.samples) if target in keep]
    return indices, remap


def cache(args: argparse.Namespace) -> None:
    """Extract clean features and the online-reduced replicate targets."""

    device = args.device
    model, weights = build_backbone(device)
    evaluation = weights.transforms()
    augmentation = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    destination = study_root(args.output, args.classes)
    destination.mkdir(parents=True, exist_ok=True)

    probe = ImageFolder(args.data_root / "train")
    keep = set(selected_classes(len(probe.classes), args.classes))
    columns = torch.tensor(sorted(keep))

    def loader_for(split: str, transform, indices: list[int]) -> DataLoader:
        dataset = ImageFolder(args.data_root / split, transform=transform)
        return DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=False,
        )

    for split in ("train", "val"):
        base = ImageFolder(args.data_root / split)
        indices, remap = subset_indices(base, keep)
        labels = torch.tensor([remap[base.samples[i][1]] for i in indices], dtype=torch.long)
        count = len(indices)
        print(f"{split}: {count} images over {args.classes} classes")

        started = time.perf_counter()
        features, logits = [], []
        with torch.no_grad():
            for inputs, _ in loader_for(split, evaluation, indices):
                hidden = forward_features(model, inputs.to(device, non_blocking=True))
                features.append(hidden.cpu())
                logits.append(model.fc(hidden)[:, columns.to(device)].cpu())
        payload = {
            "features": torch.cat(features),
            "logits": torch.cat(logits),
            "labels": labels,
        }
        clean_seconds = time.perf_counter() - started
        print(f"  clean pass {clean_seconds:.0f}s")

        # Welford over the augmentation repeats: only the mean and the sum of
        # squared deviations are kept, so storage is two tensors rather than one
        # per repeat. Both splits are reduced, so that the augmentation-mean
        # teacher can be reported as its own reference on the test rows.
        if True:
            mean = torch.zeros(count, args.classes, dtype=torch.float64)
            squares = torch.zeros_like(mean)
            for repeat in range(args.repeats):
                seed_everything(args.seed + repeat, deterministic=False)
                position = 0
                with torch.no_grad():
                    for inputs, _ in loader_for(split, augmentation, indices):
                        hidden = forward_features(model, inputs.to(device, non_blocking=True))
                        batch = model.fc(hidden)[:, columns.to(device)].double().cpu()
                        # Center each replicate before reducing: the protocol's
                        # target is the centered logit, and the variance of a
                        # centered vector is not the variance of the raw one.
                        batch = batch - batch.mean(dim=-1, keepdim=True)
                        stop = position + batch.shape[0]
                        delta = batch - mean[position:stop]
                        mean[position:stop] += delta / (repeat + 1)
                        squares[position:stop] += delta * (batch - mean[position:stop])
                        position = stop
                print(f"  augmented pass {repeat + 1}/{args.repeats}")
            payload["replicate_mean"] = mean.float()
            payload["replicate_squares"] = squares.float()
            payload["repeats"] = torch.tensor(args.repeats)

        payload["teacher_weight"] = model.fc.weight.detach()[columns].cpu()
        payload["teacher_bias"] = model.fc.bias.detach()[columns].cpu()
        torch.save(
            {**payload, "metadata": {"classes": args.classes, "wnids": columns.tolist()}},
            destination / f"{split}.pt",
        )
        print(f"  wrote {destination / f'{split}.pt'} in {time.perf_counter() - started:.0f}s")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Load the shards and reduce them to the tensors the protocol consumes."""

    root = study_root(args.output, args.classes)
    train = torch.load(root / "train.pt", map_location="cpu", weights_only=False)
    val = torch.load(root / "val.pt", map_location="cpu", weights_only=False)
    repeats = int(train["repeats"])

    def centered(logits: torch.Tensor) -> torch.Tensor:
        return logits - logits.mean(dim=-1, keepdim=True)

    # One scale, taken from the clean training logits, so that the mean head's
    # target and the variance head's target live in the same units.
    clean_train = centered(train["logits"])
    scale = logit_target_scale(clean_train)
    replicate_mean = train["replicate_mean"] / scale
    replicate_variance = train["replicate_squares"] / (repeats - 1) / (scale**2)
    val_replicate_mean = val["replicate_mean"] / scale
    val_replicate_variance = val["replicate_squares"] / (repeats - 1) / (scale**2)

    generator = torch.Generator().manual_seed(2026)
    permutation = torch.randperm(val["features"].shape[0], generator=generator)
    half = permutation.numel() // 2
    return {
        "train_features": train["features"],
        "clean_mean": clean_train / scale,
        "replicate_mean": replicate_mean,
        "replicate_variance": replicate_variance,
        "repeats": repeats,
        "scale": scale,
        "teacher_weight": train["teacher_weight"],
        "teacher_bias": train["teacher_bias"],
        "val_features": val["features"],
        "val_labels": val["labels"],
        "val_logits": val["logits"],
        "val_replicate_mean": val_replicate_mean,
        "val_replicate_variance": val_replicate_variance,
        "calibration": permutation[:half],
        "test": permutation[half:],
    }


def monte_carlo_chunk(classes: int, num_samples: int, budget: int = 100_000_000) -> int:
    """Rows per Monte Carlo block, so the draw tensor stays within a budget.

    Purely a batching choice: every row is integrated independently, so the
    result does not depend on it. At K = 1000 the library default would allocate
    several gigabytes per block.
    """

    return max(64, budget // max(1, num_samples * classes))


def build_head(features: torch.Tensor, classes: int, scale: float, args) -> TAGILastLayerClassifier:
    return TAGILastLayerClassifier(
        features.shape[1],
        classes,
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
    )


def run(args: argparse.Namespace) -> None:
    data = prepare(args)
    classes = args.classes
    features = data["train_features"]
    val_features, val_labels = data["val_features"], data["val_labels"]
    calibration, test = data["calibration"], data["test"]
    test_labels = val_labels[test]

    _, prior_variance = logit_variance_head_prior(
        aleatoric_init=args.aleatoric_init,
        variance_floor=LOGIT_VARIANCE_FLOOR,
        coefficient_of_variation=args.variance_cv,
    )
    weight_prior, _ = logit_variance_prior_split(
        prior_variance,
        feature_energy=logit_feature_energy(features),
        weight_share=args.variance_weight_share,
    )
    rates = [kappa / weight_prior for kappa in args.kappas]

    teacher_logits = data["val_logits"]
    temperature = fit_softmax_temperature(teacher_logits[calibration], val_labels[calibration])
    baselines = {
        "teacher": classification_metrics(torch.softmax(teacher_logits[test], dim=-1), test_labels),
        "teacher_temperature": classification_metrics(
            torch.softmax(teacher_logits[test] / temperature, dim=-1), test_labels
        ),
    }
    baselines["teacher_temperature"]["temperature"] = temperature
    augmentation_mean = data["val_replicate_mean"]
    baselines["teacher_augmentation_mean"] = classification_metrics(
        torch.softmax(augmentation_mean[test], dim=-1), test_labels
    )

    results: dict[str, Any] = {
        "config": {
            "classes": classes,
            "repeats": data["repeats"],
            "scale": data["scale"],
            "train_rows": int(features.shape[0]),
            "mean_epochs": args.mean_epochs,
            "variance_epochs": args.variance_epochs,
            "batch_size": args.batch_size,
            "prior_weight_variance": weight_prior,
            "feature_energy": logit_feature_energy(features),
            "mean_target": args.mean_target,
            "teacher_init": bool(args.teacher_init),
        },
        "baselines": {
            name: {k: v[k] for k in (*REPORTED_METRICS, "temperature") if k in v}
            for name, v in baselines.items()
        },
        "kappas": {},
    }

    mean_targets = data["clean_mean"] if args.mean_target == "clean" else data["replicate_mean"]
    chunk = monte_carlo_chunk(classes, args.num_samples)
    print(f"Monte Carlo chunk {chunk} rows for K={classes}, {args.num_samples} draws")
    for seed in args.seeds:
        seed_everything(seed)
        warm = build_head(features, classes, data["scale"], args)
        if args.mean_target == "clean" and args.teacher_init:
            # The clean target is exactly this affine map of these features, so
            # the warm-up assimilates rather than re-derives.
            warm.initialize_mean_from_teacher(
                data["teacher_weight"], data["teacher_bias"], scale=data["scale"]
            )
        started = time.perf_counter()
        warm.fit_mean(
            features,
            mean_targets,
            epochs=args.mean_epochs,
            batch_size=args.batch_size,
            observation_variance=args.mean_observation_variance,
            seed=seed,
        )
        mean_seconds = time.perf_counter() - started
        warm_path = (
            study_root(args.output, classes) / f"heads/{args.mean_target}_warm_seed{seed:02d}.pt"
        )
        warm.save(warm_path, metadata={"phase": "mean", "seed": seed})
        print(f"seed {seed}: mean warm-up {mean_seconds:.0f}s")

        for kappa, rate in zip(args.kappas, rates, strict=True):
            classifier, _ = TAGILastLayerClassifier.load(warm_path, device=args.device)
            classifier.reset_variance_head()
            started = time.perf_counter()
            classifier.fit_variance(
                features,
                residual_variance=data["replicate_variance"],
                repeats=data["repeats"],
                epochs=args.variance_epochs,
                batch_size=args.batch_size,
                seed=seed,
                weight_shrinkage=rate,
            )
            variance_seconds = time.perf_counter() - started

            logit_mean, epistemic, aleatoric = classifier.logit_moments(
                val_features, batch_size=args.prediction_batch_size
            )
            base_samples = classifier._resolve_logit_base_samples(logit_mean.dtype)
            record: dict[str, Any] = {
                "kappa": kappa,
                "rate": rate,
                "mean_wall_s": mean_seconds,
                "variance_wall_s": variance_seconds,
                "controls": {},
            }
            class_mean = aleatoric[calibration].mean(dim=0, keepdim=True)
            for control in VARIANCE_CONTROLS:
                fitted = fit_logit_calibration(
                    logit_mean[calibration],
                    epistemic[calibration],
                    substitute_variance(aleatoric[calibration], control, class_mean=class_mean),
                    val_labels[calibration].to(classifier.device),
                    mode="joint",
                    base_samples=base_samples,
                    chunk_size=chunk,
                )
                probabilities = logit_tagiv_predictive_probs(
                    logit_mean[test],
                    epistemic[test],
                    substitute_variance(aleatoric[test], control, class_mean=class_mean),
                    temperature=fitted.temperature,
                    alpha=fitted.alpha,
                    base_samples=base_samples,
                    chunk_size=chunk,
                )
                metrics = classification_metrics(probabilities, test_labels)
                record["controls"][control] = {
                    "temperature": fitted.temperature,
                    "alpha": fitted.alpha,
                    **{k: metrics[k] for k in REPORTED_METRICS},
                    "aurc": metrics["aurc"],
                    "risk_at_90_coverage": metrics["risk_at_90_coverage"],
                }

            # The replicate variance is a training-set quantity; scoring it on the
            # validation split needs its own measured target, which ImageNet does
            # not have, so the level diagnostics are reported on the training rows.
            learned_train = classifier.logit_moments(
                features[: args.diagnostic_rows], batch_size=args.prediction_batch_size
            )[2].cpu()
            measured_train = data["replicate_variance"][: args.diagnostic_rows]
            record["levels"] = variance_quantiles(learned_train.flatten(), measured_train.flatten())
            record["quantile_disagreement"] = quantile_disagreement(record["levels"])
            record["log_error"] = log_variance_error(
                learned_train, measured_train, repeats=data["repeats"]
            )
            record["spearman_per_class"] = rank_correlation(learned_train, measured_train)
            aggregations = channel_aggregations(logit_mean[test], aleatoric[test])
            record["aggregation_levels"] = {
                name: float(value.mean()) for name, value in aggregations.items()
            }
            results["kappas"].setdefault(f"{kappa:g}", {})[str(seed)] = record
            print(
                f"seed {seed} kappa={kappa:g}: quantile={record['quantile_disagreement']:.3f} "
                f"rank={record['spearman_per_class']:+.3f} "
                f"nll={record['controls']['learned']['nll']:.4f} "
                f"alpha={record['controls']['learned']['alpha']:.2f} "
                f"[{variance_seconds:.0f}s]"
            )
            classifier.save(
                study_root(args.output, classes)
                / f"heads/{args.mean_target}_kappa{kappa:g}_seed{seed:02d}.pt",
                metadata={"kappa": kappa, "seed": seed},
            )

    name = "results.json" if args.tag is None else f"results_{args.tag}.json"
    atomic_json(study_root(args.output, classes) / name, results)
    print(f"wrote {study_root(args.output, classes) / name}")


def paired_bootstrap(
    differences: torch.Tensor, *, draws: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Return the paired mean difference with a bootstrap confidence interval.

    Seed stability does not cover this: every seed is scored on the same finite
    test examples, so the sampling uncertainty of the test set itself has to be
    quantified separately.
    """

    values = differences.double()
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(values.numel(), (draws, values.numel()), generator=generator)
    means = values[index].mean(dim=1)
    return {
        "mean": float(values.mean()),
        "low": float(means.quantile(0.025)),
        "high": float(means.quantile(0.975)),
        "excludes_zero": bool(means.quantile(0.025) > 0.0 or means.quantile(0.975) < 0.0),
    }


def negative_log_probability(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    index = labels.to(probabilities.device).long().unsqueeze(1)
    return -probabilities.gather(1, index).squeeze(1).clamp_min(1e-12).log()


def stratum_record(
    mask: torch.Tensor,
    losses: dict[str, torch.Tensor],
    correct: torch.Tensor,
    draws: int,
) -> dict[str, Any]:
    """Score one stratum, and its additive share of the aggregate gain.

    The aggregate is the mass-weighted sum of the per-stratum gains, so
    ``share = p_s * delta_s`` says how much of it each stratum supplies and
    makes a cancellation visible rather than implicit.
    """

    share = float(mask.float().mean())
    record: dict[str, Any] = {
        "rows": int(mask.sum()),
        "mass": share,
        "teacher_accuracy": float(correct[mask].float().mean()),
    }
    for candidate in ("learned", "oracle_measured_variance", "tta_calibrated"):
        interval = paired_bootstrap(
            (losses["teacher_temperature"][mask] - losses[candidate][mask]).cpu(), draws=draws
        )
        interval["contribution"] = share * interval["mean"]
        record[candidate] = interval
    return record


def ceiling(args: argparse.Namespace) -> None:
    """Bound what augmentation variance can buy, before spending ImageNet-1k.

    Three questions: is the learned gain distinguishable from test-set sampling
    noise; how large is the gain when the variance is measured directly rather
    than learned; and how much of any benefit is a mean shift that
    test-time augmentation would capture instead.
    """

    data = prepare(args)
    classes = args.classes
    val_features = data["val_features"]
    labels = data["val_labels"]
    calibration, test = data["calibration"], data["test"]
    test_labels = labels[test]
    scale = data["scale"]

    clean = data["val_logits"]
    clean = (clean - clean.mean(dim=-1, keepdim=True)) / scale
    oracle_variance = data["val_replicate_variance"]
    device = args.device

    head_path = (
        study_root(args.output, classes)
        / f"heads/{args.mean_target}_kappa{args.kappa:g}_seed{args.seed:02d}.pt"
    )
    classifier, _ = TAGILastLayerClassifier.load(head_path, device=device)
    logit_mean, epistemic, aleatoric = classifier.logit_moments(
        val_features, batch_size=args.prediction_batch_size
    )
    base_samples = classifier._resolve_logit_base_samples(logit_mean.dtype)
    chunk = monte_carlo_chunk(classes, classifier.logit_num_samples)

    def calibrated(mean, epistemic_var, aleatoric_var, name):
        fitted = fit_logit_calibration(
            mean[calibration],
            epistemic_var[calibration],
            aleatoric_var[calibration],
            labels[calibration].to(device),
            mode="joint",
            base_samples=base_samples,
            chunk_size=chunk,
        )
        probabilities = logit_tagiv_predictive_probs(
            mean[test],
            epistemic_var[test],
            aleatoric_var[test],
            temperature=fitted.temperature,
            alpha=fitted.alpha,
            base_samples=base_samples,
        )
        metrics = classification_metrics(probabilities, test_labels)
        return (
            probabilities,
            fitted,
            {
                "temperature": fitted.temperature,
                "alpha": fitted.alpha,
                **{k: metrics[k] for k in REPORTED_METRICS},
            },
        )

    class_mean = aleatoric[calibration].mean(dim=0, keepdim=True)
    entries: dict[str, Any] = {}
    losses: dict[str, torch.Tensor] = {}

    for control in VARIANCE_CONTROLS:
        probabilities, _, summary = calibrated(
            logit_mean,
            epistemic,
            substitute_variance(aleatoric, control, class_mean=class_mean),
            control,
        )
        entries[control] = summary
        losses[control] = negative_log_probability(probabilities, test_labels).cpu()

    # Teacher plus temperature, on the same test rows.
    teacher_temperature = fit_softmax_temperature(
        data["val_logits"][calibration], labels[calibration]
    )
    teacher_probabilities = torch.softmax(data["val_logits"][test] / teacher_temperature, dim=-1)
    entries["teacher_temperature"] = {
        "temperature": teacher_temperature,
        **{
            k: classification_metrics(teacher_probabilities, test_labels)[k]
            for k in REPORTED_METRICS
        },
    }
    losses["teacher_temperature"] = negative_log_probability(teacher_probabilities, test_labels)

    # Oracle: the teacher's own clean logits with the measured replicate variance.
    zeros = torch.zeros_like(clean).to(device)
    probabilities, _, summary = calibrated(
        clean.to(device), zeros, oracle_variance.to(device), "oracle"
    )
    entries["oracle_measured_variance"] = summary
    losses["oracle_measured_variance"] = negative_log_probability(probabilities, test_labels).cpu()

    # Oracle control: the same clean logits with no variance channel at all.
    probabilities, _, summary = calibrated(clean.to(device), zeros, zeros, "oracle_none")
    entries["oracle_no_variance"] = summary
    losses["oracle_no_variance"] = negative_log_probability(probabilities, test_labels).cpu()

    # Test-time augmentation, temperature-calibrated. Averaging logits changes
    # their scale, so the uncalibrated number is not a usable reference.
    augmentation_mean = data["val_replicate_mean"]
    tta_temperature = fit_softmax_temperature(augmentation_mean[calibration], labels[calibration])
    tta_probabilities = torch.softmax(augmentation_mean[test] / tta_temperature, dim=-1)
    entries["tta_calibrated"] = {
        "temperature": tta_temperature,
        "forward_passes": data["repeats"],
        **{k: classification_metrics(tta_probabilities, test_labels)[k] for k in REPORTED_METRICS},
    }
    losses["tta_calibrated"] = negative_log_probability(tta_probabilities, test_labels)

    comparisons = {}
    for reference in ("teacher_temperature", "constant", "shuffled"):
        comparisons[f"learned_vs_{reference}"] = paired_bootstrap(
            losses[reference].cpu() - losses["learned"].cpu(), draws=args.bootstrap_draws
        )
    comparisons["oracle_vs_no_variance"] = paired_bootstrap(
        losses["oracle_no_variance"].cpu() - losses["oracle_measured_variance"].cpu(),
        draws=args.bootstrap_draws,
    )
    comparisons["tta_vs_teacher_temperature"] = paired_bootstrap(
        losses["teacher_temperature"].cpu() - losses["tta_calibrated"].cpu(),
        draws=args.bootstrap_draws,
    )

    # Stratify by how hard the teacher finds each example. If no stratum shows a
    # stable oracle benefit, the "variance helps on hard examples" hypothesis is
    # dead; if the hardest one does, that is a testable prediction for the full
    # task.
    teacher_test = data["val_logits"][test]
    confidence = torch.softmax(teacher_test / teacher_temperature, dim=-1).max(dim=1).values
    ordered = teacher_test.sort(dim=-1, descending=True).values
    margin = ordered[:, 0] - ordered[:, 1]
    correct = teacher_test.argmax(dim=-1) == test_labels
    strata: dict[str, Any] = {}
    for name, score in (("confidence", confidence), ("margin", margin)):
        quantiles = score.quantile(torch.linspace(0.0, 1.0, 6))
        for index in range(5):
            low, high = quantiles[index], quantiles[index + 1]
            mask = (score >= low) & (score <= high if index == 4 else score < high)
            if int(mask.sum()) < 50:
                continue
            strata[f"{name}_q{index + 1}"] = stratum_record(
                mask, losses, correct, args.bootstrap_draws
            )
    for name, mask in (("correct", correct), ("incorrect", ~correct)):
        if int(mask.sum()) < 50:
            continue
        strata[name] = stratum_record(mask, losses, correct, args.bootstrap_draws)

    payload = {
        "config": {
            "classes": classes,
            "mean_target": args.mean_target,
            "kappa": args.kappa,
            "seed": args.seed,
            "test_rows": int(test.numel()),
            "bootstrap_draws": args.bootstrap_draws,
            "repeats": data["repeats"],
        },
        "entries": entries,
        "comparisons": comparisons,
        "strata": strata,
    }
    atomic_json(study_root(args.output, classes) / "ceiling.json", payload)

    lines = [
        f"# Ceiling analysis, ImageNet-{classes}",
        "",
        f"Head `{args.mean_target}` at kappa {args.kappa:g}, seed {args.seed}; "
        f"{int(test.numel())} test rows, {args.bootstrap_draws} bootstrap draws. "
        "A positive difference favours the first named model.",
        "",
        "| model | accuracy | NLL | ECE | Brier | T | alpha |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, values in entries.items():
        lines.append(
            f"| {name} | "
            + " | ".join(f"{values[k]:.4f}" for k in REPORTED_METRICS)
            + f" | {values.get('temperature', float('nan')):.4f}"
            + f" | {values.get('alpha', float('nan')):.4f} |"
        )
    lines += [
        "",
        "## Paired bootstrap on per-example NLL",
        "",
        "| comparison | mean gain | 95% CI | excludes zero |",
        "| --- | --- | --- | --- |",
    ]
    for name, values in comparisons.items():
        lines.append(
            f"| {name} | {values['mean']:+.5f} | "
            f"[{values['low']:+.5f}, {values['high']:+.5f}] | "
            f"{'yes' if values['excludes_zero'] else 'no'} |"
        )
    lines += [
        "",
        "## Paired gain over teacher + T, by teacher difficulty",
        "",
        "Quintiles of the teacher's calibrated confidence and of its top-two "
        "logit margin, plus the correct and incorrect splits. An interval that "
        "excludes zero is marked.",
        "",
        "Each cell is the paired mean gain, with its contribution `p_s * delta_s` "
        "to the aggregate in brackets. A star marks an interval excluding zero.",
        "",
        "| stratum | rows | mass | teacher acc | learned | oracle | calibrated TTA |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    def cell(values: dict[str, float]) -> str:
        mark = "*" if values["excludes_zero"] else ""
        return f"{values['mean']:+.5f}{mark} [{values['contribution']:+.5f}]"

    for name, values in strata.items():
        lines.append(
            f"| {name} | {values['rows']} | {values['mass']:.3f} | "
            f"{values['teacher_accuracy']:.3f} | "
            f"{cell(values['learned'])} | {cell(values['oracle_measured_variance'])} | "
            f"{cell(values['tta_calibrated'])} |"
        )
    for candidate, label in (
        ("learned", "learned"),
        ("oracle_measured_variance", "oracle"),
        ("tta_calibrated", "calibrated TTA"),
    ):
        total = sum(
            values[candidate]["contribution"]
            for name, values in strata.items()
            if name in ("correct", "incorrect")
        )
        lines.append(
            f"| correct + incorrect total, {label} | | | | {total:+.5f} | | |"
            if candidate == "learned"
            else f"| correct + incorrect total, {label} | | | | | {total:+.5f} | |"
            if candidate == "oracle_measured_variance"
            else f"| correct + incorrect total, {label} | | | | | | {total:+.5f} |"
        )

    destination = study_root(args.output, classes) / "CEILING.md"
    destination.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {destination}")


def report(args: argparse.Namespace) -> None:
    name = "results.json" if args.tag is None else f"results_{args.tag}.json"
    source = study_root(args.output, args.classes) / name
    results = json.loads(source.read_text())

    def gather(kappa: str, path: tuple[str, ...]) -> list[float]:
        collected = []
        for record in results["kappas"][kappa].values():
            node: Any = record
            for key in path:
                if not isinstance(node, dict) or key not in node:
                    node = None
                    break
                node = node[key]
            if isinstance(node, (int, float)):
                collected.append(float(node))
        return collected

    ordered = sorted(results["kappas"], key=float)
    selected = min(ordered, key=lambda k: statistics.mean(gather(k, ("quantile_disagreement",))))
    config = results["config"]
    lines = [
        f"# Logit-space TAGI-V on ImageNet-{config['classes']}",
        "",
        "Frozen protocol, no tuning beyond the pre-registered grid. The shrinkage "
        "rate is selected on held-out replicate quantiles; `(T, alpha)` is fitted "
        "on the label-calibration half of the official validation split and the "
        "other half is the test set.",
        "",
        f"{config['train_rows']:,} training rows, {config['repeats']} replicates, "
        f"feature energy {config['feature_energy']:.1f}, "
        f"v_Wg,0 {config['prior_weight_variance']:.3e}. Selected kappa: `{selected}`.",
        "",
        "| kappa | quantile gap | rank corr | alpha | NLL | ECE | accuracy | variance fit (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for kappa in ordered:
        marker = "**" if kappa == selected else ""
        lines.append(
            f"| {marker}{kappa}{marker} | "
            f"{summarize(gather(kappa, ('quantile_disagreement',)))} | "
            f"{summarize(gather(kappa, ('spearman_per_class',)))} | "
            f"{summarize(gather(kappa, ('controls', 'learned', 'alpha')))} | "
            f"{summarize(gather(kappa, ('controls', 'learned', 'nll')))} | "
            f"{summarize(gather(kappa, ('controls', 'learned', 'ece')))} | "
            f"{summarize(gather(kappa, ('controls', 'learned', 'accuracy')))} | "
            f"{summarize(gather(kappa, ('variance_wall_s',)))} |"
        )

    lines += [
        "",
        "## Test metrics at the selected rate",
        "",
        "| model | accuracy | NLL | ECE | Brier |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, values in results["baselines"].items():
        lines.append(
            f"| {name} | " + " | ".join(f"{values[k]:.4f}" for k in REPORTED_METRICS) + " |"
        )
    for control in VARIANCE_CONTROLS:
        row = [summarize(gather(selected, ("controls", control, key))) for key in REPORTED_METRICS]
        lines.append(f"| logit_tagiv, {control} | " + " | ".join(row) + " |")

    lines += [
        "",
        "## Variance level, training rows",
        "",
        "| statistic | learned | measured |",
        "| --- | --- | --- |",
    ]
    for name in ("p50", "p90", "p99"):
        lines.append(
            f"| {name} | {summarize(gather(selected, ('levels', f'learned_{name}')))} | "
            f"{summarize(gather(selected, ('levels', f'measured_{name}')))} |"
        )
    lines.append(
        f"| sd(log) | {summarize(gather(selected, ('levels', 'learned_log_dispersion')))} | "
        f"{summarize(gather(selected, ('levels', 'measured_log_dispersion')))} |"
    )

    destination = study_root(args.output, args.classes) / (
        "REPORT.md" if args.tag is None else f"REPORT_{args.tag}.md"
    )
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--classes", type=int, default=100)
    common.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    cacher = subparsers.add_parser("cache", parents=[common], help="extract features and targets")
    cacher.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    cacher.add_argument("--repeats", type=int, default=8)
    cacher.add_argument("--batch-size", type=int, default=256)
    cacher.add_argument("--workers", type=int, default=16)
    cacher.add_argument("--seed", type=int, default=0)
    cacher.add_argument("--device", default="cuda")

    runner = subparsers.add_parser("run", parents=[common], help="fit and evaluate")
    runner.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    runner.add_argument("--kappas", nargs="+", type=float, default=list(SHRINKAGE_KAPPAS))
    runner.add_argument("--mean-epochs", type=int, default=20)
    runner.add_argument("--variance-epochs", type=int, default=20)
    runner.add_argument("--mean-observation-variance", type=float, default=0.01)
    runner.add_argument("--batch-size", type=int, default=256)
    runner.add_argument("--prediction-batch-size", type=int, default=2048)
    runner.add_argument("--diagnostic-rows", type=int, default=20000)
    runner.add_argument("--gain", type=float, default=0.1)
    runner.add_argument("--aleatoric-init", type=float, default=0.1)
    runner.add_argument("--variance-cv", type=float, default=0.5)
    runner.add_argument("--variance-weight-share", type=float, default=0.5)
    runner.add_argument("--num-samples", type=int, default=256)
    runner.add_argument("--device", default="cuda")
    runner.add_argument(
        "--mean-target",
        choices=("clean", "augmented"),
        default="clean",
        help="clean distils the teacher's clean logits, which the head can "
        "represent exactly; augmented is the original, unrepresentable target "
        "retained as a negative ablation.",
    )
    runner.add_argument("--teacher-init", action="store_true", default=True)
    runner.add_argument("--no-teacher-init", dest="teacher_init", action="store_false")
    runner.add_argument("--tag", default=None, help="suffix for the results file")

    ceiler = subparsers.add_parser(
        "ceiling", parents=[common], help="bound what augmentation variance can buy"
    )
    ceiler.add_argument("--mean-target", choices=("clean", "augmented"), default="clean")
    ceiler.add_argument("--kappa", type=float, default=0.0)
    ceiler.add_argument("--seed", type=int, default=0)
    ceiler.add_argument("--bootstrap-draws", type=int, default=10000)
    ceiler.add_argument("--prediction-batch-size", type=int, default=2048)
    ceiler.add_argument("--num-samples", type=int, default=256)
    ceiler.add_argument("--gain", type=float, default=0.1)
    ceiler.add_argument("--aleatoric-init", type=float, default=0.1)
    ceiler.add_argument("--variance-cv", type=float, default=0.5)
    ceiler.add_argument("--variance-weight-share", type=float, default=0.5)
    ceiler.add_argument("--device", default="cuda")

    reporter = subparsers.add_parser("report", parents=[common], help="write the markdown summary")
    reporter.add_argument("--tag", default=None)

    args = parser.parse_args()
    if args.command == "cache":
        cache(args)
    elif args.command == "run":
        run(args)
    elif args.command == "ceiling":
        ceiling(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
