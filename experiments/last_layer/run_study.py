"""Reproducible frozen-feature CIFAR last-layer study runner.

Examples:
  python experiments/last_layer/run_study.py backbone --dataset cifar10
  python experiments/last_layer/run_study.py cache --dataset cifar10
  python experiments/last_layer/run_study.py run --dataset cifar10 --stage screen
  python experiments/last_layer/run_study.py select --dataset cifar10 --stage screen
  python experiments/last_layer/run_study.py run --dataset cifar10 --stage refine
  python experiments/last_layer/run_study.py select --dataset cifar10 --stage refine
  python experiments/last_layer/run_study.py run --dataset cifar10 --stage confirm
  python experiments/last_layer/run_study.py evaluate --dataset cifar10
  python experiments/last_layer/run_study.py report
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import (  # noqa: E402
    TAGILastLayerClassifier,
    classification_metrics,
    epistemic_convergence,
    evaluate_ood_comprehensive,
    training_required_epoch,
)
from triton_tagi.cifar_study import (  # noqa: E402
    CANONICAL_CORRUPTIONS,
    CifarCorruption,
    CifarResNet18,
    clean_datasets,
    extract_frozen_features,
    feature_metadata,
    get_spec,
    load_feature_shard,
    save_feature_shard,
    seed_everything,
    stable_hash,
    stratified_split,
    svhn_dataset,
)


DEFAULT_MANIFEST = Path(__file__).with_name("study.json")


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError("study manifest schema_version must be 1")
    return manifest


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(destination)


def study_split(
    manifest: dict[str, Any], dataset_object
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return the predeclared train/validation partition and its provenance."""

    targets = getattr(dataset_object, "targets", None)
    if targets is None:
        raise ValueError("the clean training dataset must expose ordered targets")
    labels = torch.as_tensor(targets).long()
    if labels.ndim != 1 or labels.numel() != len(dataset_object):
        raise ValueError("training dataset targets do not match the dataset length")
    split_config = manifest["split"]
    train_indices, validation_indices = stratified_split(
        labels,
        split_config["validation_size"],
        split_config["seed"],
    )
    provenance = {
        "seed": split_config["seed"],
        "train_size": train_indices.numel(),
        "validation_size": validation_indices.numel(),
        "train_indices_fingerprint": stable_hash(train_indices.tolist(), length=32),
        "validation_indices_fingerprint": stable_hash(
            validation_indices.tolist(), length=32
        ),
    }
    return train_indices, validation_indices, provenance


def artifact_root(manifest: dict[str, Any]) -> Path:
    return REPOSITORY_ROOT / manifest["paths"]["artifacts"] / manifest["study_id"]


def backbone_path(manifest: dict[str, Any], dataset: str) -> Path:
    return artifact_root(manifest) / "backbones" / f"{dataset}_resnet18.pt"


def feature_root(manifest: dict[str, Any], dataset: str) -> Path:
    return artifact_root(manifest) / "features" / dataset


def stage_root(manifest: dict[str, Any], stage: str, dataset: str) -> Path:
    return artifact_root(manifest) / "heads" / stage / dataset


def load_backbone(manifest: dict[str, Any], dataset: str, device: torch.device):
    spec = get_spec(dataset)
    model = CifarResNet18(spec.num_classes)
    checkpoint = torch.load(
        backbone_path(manifest, dataset), map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint


@torch.no_grad()
def evaluate_backbone(
    model: CifarResNet18,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        prediction = model(inputs.to(device, non_blocking=True)).argmax(1).cpu()
        labels = torch.as_tensor(labels).long()
        correct += prediction.eq(labels).sum().item()
        total += labels.numel()
    return correct / total


def train_backbone(args, manifest: dict[str, Any]) -> None:
    spec = get_spec(args.dataset)
    config = manifest["backbone"]
    seed_everything(config["seed"])
    device = torch.device(args.device)
    augmented_training_set, _ = clean_datasets(
        args.dataset,
        REPOSITORY_ROOT / manifest["paths"]["data"],
        augment_train=True,
        download=not args.no_download,
    )
    evaluation_training_set, _ = clean_datasets(
        args.dataset,
        REPOSITORY_ROOT / manifest["paths"]["data"],
        augment_train=False,
        download=not args.no_download,
    )
    train_indices, validation_indices, split_provenance = study_split(
        manifest, augmented_training_set
    )
    evaluation_train_indices, evaluation_validation_indices, evaluation_split = (
        study_split(manifest, evaluation_training_set)
    )
    if (
        split_provenance != evaluation_split
        or not torch.equal(train_indices, evaluation_train_indices)
        or not torch.equal(validation_indices, evaluation_validation_indices)
    ):
        raise RuntimeError("augmented and evaluation datasets produced different splits")
    train_set = Subset(augmented_training_set, train_indices.tolist())
    validation_set = Subset(
        evaluation_training_set, validation_indices.tolist()
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=config["evaluation_batch_size"],
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = CifarResNet18(spec.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )
    criterion = nn.CrossEntropyLoss()
    history = []
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = samples = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.numel()
            samples += labels.numel()
        scheduler.step()
        validation_accuracy = evaluate_backbone(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / samples,
            "validation_accuracy": validation_accuracy,
            "wall_s": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"{args.dataset} epoch {epoch:3d}/{config['epochs']} "
            f"loss={row['train_loss']:.4f} val_acc={validation_accuracy:.3%}",
            flush=True,
        )

    if history[-1]["validation_accuracy"] < spec.backbone_accuracy_gate:
        raise RuntimeError(
            f"final validation accuracy {history[-1]['validation_accuracy']:.3%} "
            "is below the "
            f"predeclared {spec.backbone_accuracy_gate:.1%} health gate"
        )
    destination = backbone_path(manifest, args.dataset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "dataset": args.dataset,
        "architecture": "cifar_resnet18",
        "config": config,
        "split": split_provenance,
        "history": history,
    }
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    atomic_json(
        destination.with_suffix(".json"),
        {key: value for key, value in payload.items() if key != "model_state_dict"},
    )
    print(f"saved {destination}")


def cache_features(args, manifest: dict[str, Any]) -> None:
    seed_everything(manifest["backbone"]["seed"])
    device = torch.device(args.device)
    model, checkpoint = load_backbone(manifest, args.dataset, device)
    root = feature_root(manifest, args.dataset)
    train_set, test_set = clean_datasets(
        args.dataset,
        REPOSITORY_ROOT / manifest["paths"]["data"],
        augment_train=False,
        download=not args.no_download,
    )
    train_all = extract_frozen_features(
        model,
        train_set,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    train_idx, validation_idx, split_provenance = study_split(manifest, train_set)
    if checkpoint.get("split") != split_provenance:
        raise ValueError(
            "backbone split metadata does not match the requested study split; "
            "retrain the backbone before caching features"
        )
    for name, indices in (("train", train_idx), ("validation", validation_idx)):
        tensors = {
            key: value[indices]
            for key, value in train_all.items()
        }
        metadata = feature_metadata(
            dataset=args.dataset,
            split=name,
            checkpoint=backbone_path(manifest, args.dataset),
            transform="dataset_evaluation_normalization",
            extra={
                "split": split_provenance,
                "indices": indices.tolist(),
            },
        )
        save_feature_shard(root / f"{name}.pt", tensors, metadata)

    for name, dataset_object in (
        ("test", test_set),
        (
            "svhn",
            svhn_dataset(
                args.dataset,
                REPOSITORY_ROOT / manifest["paths"]["data"],
                download=not args.no_download,
            ),
        ),
    ):
        tensors = extract_frozen_features(
            model,
            dataset_object,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        metadata = feature_metadata(
            dataset=args.dataset,
            split=name,
            checkpoint=backbone_path(manifest, args.dataset),
            transform="dataset_evaluation_normalization",
        )
        save_feature_shard(root / f"{name}.pt", tensors, metadata)
        print(f"cached {name}: {tensors['features'].shape}")

    if args.skip_corruptions:
        return
    corruption_root = REPOSITORY_ROOT / manifest["paths"]["corruptions"][args.dataset]
    for corruption in CANONICAL_CORRUPTIONS:
        for severity in range(1, 6):
            dataset_object = CifarCorruption(
                corruption_root, args.dataset, corruption, severity
            )
            tensors = extract_frozen_features(
                model,
                dataset_object,
                device=device,
                batch_size=args.batch_size,
                workers=args.workers,
            )
            name = f"{corruption}_s{severity}"
            metadata = feature_metadata(
                dataset=args.dataset,
                split=name,
                checkpoint=backbone_path(manifest, args.dataset),
                transform="dataset_evaluation_normalization",
                extra={"corruption": corruption, "severity": severity},
            )
            save_feature_shard(root / "corruptions" / f"{name}.pt", tensors, metadata)
            print(f"cached {name}")


def fixed_screen_configs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    fixed = manifest["fixed_noise"]
    configs = [
        {
            "head": head,
            "sigma_v": sigma_v,
            "gain_w": gain,
            "gain_b": gain,
        }
        for head, sigma_v, gain in itertools.product(
            fixed["heads"], fixed["sigma_v"], fixed["tied_gains"]
        )
    ]
    unit_probit = manifest["unit_probit"]
    configs.extend(
        {
            "head": head,
            "gain_w": gain,
            "gain_b": gain,
        }
        for head, gain in itertools.product(
            unit_probit["heads"], unit_probit["tied_gains"]
        )
    )
    return configs


def tagiv_configs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    tagiv = manifest["tagiv"]
    return [
        {
            "head": head,
            "sigma_v": None,
            "gain_w": tagiv["gain_w"],
            "gain_b": tagiv["gain_b"],
            "v2bar_init": initial,
            "v2bar_weight_var": tagiv["v2bar_weight_var"],
            "v2bar_bias_var": bias_variance,
        }
        for head, initial, bias_variance in itertools.product(
            tagiv["heads"], tagiv["v2bar_init"], tagiv["v2bar_bias_var"]
        )
    ]


def init_screen_configs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """The mean-initialization grid: init x gain x sigma_v, per head.

    ``logit_tagiv`` zeroes its own latent means, so "random" and "zero" are the
    same head for it and only the distinct arms are emitted. Deduplicating here
    rather than dropping the rows later keeps the cell count honest.
    """

    study = manifest["init_study"]
    configs = [
        {
            "head": head,
            "sigma_v": sigma_v,
            "gain_w": gain,
            "gain_b": gain,
            "mean_init": mean_init,
        }
        for head, mean_init, gain, sigma_v in itertools.product(
            study["fixed_noise_heads"],
            study["mean_init"],
            study["tied_gains"],
            study["sigma_v"],
        )
    ]
    configs.extend(
        {
            "head": head,
            "sigma_v": None,
            "gain_w": gain,
            "gain_b": gain,
            "mean_init": mean_init,
        }
        for head, mean_init, gain in itertools.product(
            study["logit_heads"],
            [arm for arm in study["mean_init"] if arm != "random"],
            study["tied_gains"],
        )
    )
    return configs


def init_confirm_configs(manifest: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    """The selected arm per head, plus its ``random`` counterpart for the delta.

    Table 2 of the deck is the initialization delta at the full protocol, not
    only at 20 epochs, so every head is confirmed twice: once at whatever the
    screen chose and once at the status quo it is being compared against.
    """

    selected = load_selection(manifest, dataset, "init_screen")["selected"]
    configs: list[dict[str, Any]] = []
    for choice in selected.values():
        base = dict(choice["config"])
        configs.append(base)
        baseline = {**base, "mean_init": "random"}
        if baseline not in configs:
            configs.append(baseline)
    return configs


def load_selection(manifest: dict[str, Any], dataset: str, stage: str) -> dict[str, Any]:
    path = stage_root(manifest, stage, dataset) / "selection.json"
    with path.open() as handle:
        return json.load(handle)


def neighboring_gains(best: float, master: list[float]) -> list[float]:
    index = master.index(best)
    start, stop = max(0, index - 1), min(len(master), index + 2)
    return master[start:stop]


def stage_configs(
    manifest: dict[str, Any],
    dataset: str,
    stage: str,
) -> tuple[list[dict[str, Any]], list[int], int, list[int]]:
    if stage == "screen":
        return (
            fixed_screen_configs(manifest),
            manifest["screen"]["seeds"],
            manifest["screen"]["epochs"],
            manifest["screen"]["checkpoints"],
        )
    if stage == "init_screen":
        phase = manifest["init_study"]["screen"]
        return (
            init_screen_configs(manifest),
            phase["seeds"],
            phase["epochs"],
            phase["checkpoints"],
        )
    if stage == "init_confirm":
        phase = manifest["init_study"]["confirm"]
        return (
            init_confirm_configs(manifest, dataset),
            phase["seeds"],
            phase["epochs"],
            phase["checkpoints"],
        )
    if stage == "tagiv":
        return (
            tagiv_configs(manifest),
            manifest["screen"]["seeds"],
            manifest["screen"]["epochs"],
            manifest["screen"]["checkpoints"],
        )
    if stage == "refine":
        selected = load_selection(manifest, dataset, "screen")["selected"]
        fixed = manifest["fixed_noise"]
        unit_probit = manifest["unit_probit"]
        configs = []
        for choice in selected.values():
            base = choice["config"]
            if base["head"] in unit_probit["heads"]:
                neighbors = neighboring_gains(
                    base["gain_w"], unit_probit["gain_master"]
                )
                for gain_w, gain_b in itertools.product(neighbors, neighbors):
                    configs.append(
                        {
                            **base,
                            "gain_w": gain_w,
                            "gain_b": gain_b,
                        }
                    )
                continue
            neighbors = neighboring_gains(base["gain_w"], fixed["gain_master"])
            sigma_values = [base["sigma_v"]]
            if base["sigma_v"] == min(fixed["sigma_v"]):
                sigma_values.append(0.003)
            elif base["sigma_v"] == max(fixed["sigma_v"]):
                sigma_values.append(3.0)
            for sigma_v, gain_w, gain_b in itertools.product(
                sigma_values, neighbors, neighbors
            ):
                configs.append(
                    {
                        **base,
                        "sigma_v": sigma_v,
                        "gain_w": gain_w,
                        "gain_b": gain_b,
                    }
                )
        return (
            configs,
            manifest["screen"]["seeds"],
            manifest["screen"]["epochs"],
            manifest["screen"]["checkpoints"],
        )
    if stage == "confirm":
        fixed = load_selection(manifest, dataset, "refine")["selected"]
        tagiv_path = stage_root(manifest, "tagiv", dataset) / "selection.json"
        selected = {**fixed}
        if tagiv_path.exists():
            selected.update(load_selection(manifest, dataset, "tagiv")["selected"])
        configs = [choice["config"] for choice in selected.values()]
        return (
            configs,
            manifest["confirmation"]["seeds"],
            manifest["confirmation"]["epochs"],
            manifest["confirmation"]["checkpoints"],
        )
    raise ValueError(
        "stage must be screen, refine, tagiv, confirm, init_screen, or init_confirm"
    )


def backbone_fc_weights(
    manifest: dict[str, Any], dataset: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the trained backbone's ``fc`` weight and bias for the warm start."""

    state = torch.load(
        backbone_path(manifest, dataset), map_location="cpu", weights_only=False
    )["model_state_dict"]
    return state["network.fc.weight"], state["network.fc.bias"]


def native_diagnostics(classifier, features: torch.Tensor) -> dict[str, float]:
    prediction = classifier.predict(features, sigma_v=classifier.sigma_v)
    epistemic = prediction.epistemic_variance
    result = {
        "weight_variance_mean": classifier.linear.Sw.mean().item(),
        "weight_variance_median": classifier.linear.Sw.median().item(),
        "bias_variance_mean": classifier.linear.Sb.mean().item(),
    }
    if epistemic is not None:
        flattened = epistemic.flatten()
        result.update(
            {
                "epistemic_mean": flattened.mean().item(),
                "epistemic_median": flattened.median().item(),
                "epistemic_p90": torch.quantile(flattened, 0.9).item(),
            }
        )
    if prediction.aleatoric_variance is not None:
        aleatoric = prediction.aleatoric_variance
        result["aleatoric_mean"] = aleatoric.mean().item()
        if epistemic is not None and epistemic.shape == aleatoric.shape:
            result["epistemic_fraction"] = (
                epistemic / (epistemic + aleatoric).clamp_min(1e-12)
            ).mean().item()
    return result


def run_configuration(
    manifest: dict[str, Any],
    dataset: str,
    stage: str,
    config: dict[str, Any],
    seed: int,
    epochs: int,
    checkpoints: list[int],
    args,
) -> None:
    root = feature_root(manifest, dataset)
    train = load_feature_shard(root / "train.pt")
    validation = load_feature_shard(root / "validation.pt")
    run_config = {
        "dataset": dataset,
        "stage": stage,
        "seed": seed,
        "epochs": epochs,
        "batch_size": manifest["last_layer"]["batch_size"],
        **config,
    }
    run_id = stable_hash(run_config)
    run_dir = stage_root(manifest, stage, dataset) / config["head"] / f"{run_id}_seed{seed}"
    complete = run_dir / "complete.json"
    if complete.exists() and not args.force:
        print(f"skip complete {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "config.json", run_config)

    seed_everything(seed)
    head_kwargs = dict(config)
    if head_kwargs.get("mean_init") == "backbone":
        # Tensors cannot live in the JSON config, so they are looked up here
        # and never enter the run hash; mean_init alone identifies the arm.
        head_kwargs["backbone_fc"] = backbone_fc_weights(manifest, dataset)
    targets = None
    if config["head"] == "logit_tagiv":
        if "logits" not in train:
            raise ValueError(
                f"{dataset} features carry no teacher logits, which logit_tagiv needs"
            )
        # The head regresses class-centered teacher logits.
        targets = train["logits"] - train["logits"].mean(dim=1, keepdim=True)
        head_kwargs.setdefault(
            "logit_variance_feature_energy",
            float((train["features"] ** 2).sum(dim=1).mean()),
        )
        head_kwargs.setdefault("logit_variance_weight_share", 0.0)
    classifier = TAGILastLayerClassifier(
        train["features"].shape[1],
        get_spec(dataset).num_classes,
        device=args.device,
        **head_kwargs,
    )
    cohort_size = min(
        manifest["last_layer"]["diagnostic_cohort_size"],
        validation["features"].shape[0],
    )
    cohort = validation["features"][:cohort_size]

    def callback(epoch, current, row):
        row.update(native_diagnostics(current, cohort))

    history = classifier.fit(
        train["features"],
        train["labels"],
        targets=targets,
        epochs=epochs,
        batch_size=manifest["last_layer"]["batch_size"],
        seed=seed,
        validation=(validation["features"], validation["labels"]),
        checkpoint_dir=run_dir / "checkpoints",
        checkpoint_epochs=set(checkpoints),
        callback=callback,
    )
    classifier.save(run_dir / "checkpoints" / f"epoch_{epochs:04d}.pt", metadata={"epoch": epochs})
    records = list(history.records)
    atomic_json(run_dir / "history.json", records)
    summary = {
        "config": run_config,
        "training_required_epoch": training_required_epoch(records),
    }
    if epochs >= 200 and all("epistemic_mean" in row for row in records):
        summary["epistemic_convergence"] = epistemic_convergence(
            torch.tensor([row["epoch"] for row in records]),
            torch.tensor([row["epistemic_mean"] for row in records]),
        )
    atomic_json(complete, summary)
    print(f"complete {run_dir}")


def run_stage(args, manifest: dict[str, Any]) -> None:
    configs, seeds, epochs, checkpoints = stage_configs(
        manifest, args.dataset, args.stage
    )
    for config in configs:
        for seed in seeds:
            run_configuration(
                manifest,
                args.dataset,
                args.stage,
                config,
                seed,
                epochs,
                checkpoints,
                args,
            )


def select_stage(args, manifest: dict[str, Any]) -> None:
    root = stage_root(manifest, args.stage, args.dataset)
    candidates = []
    if args.stage.startswith("init_"):
        phase = "confirm" if args.stage == "init_confirm" else "screen"
        checkpoint_epochs = set(manifest["init_study"][phase]["checkpoints"])
    else:
        checkpoint_epochs = set(
            manifest["confirmation"]["checkpoints"]
            if args.stage == "confirm"
            else manifest["screen"]["checkpoints"]
        )
    for history_path in root.glob("*/*/history.json"):
        config = json.loads((history_path.parent / "config.json").read_text())
        records = json.loads(history_path.read_text())
        for record in records:
            if int(record["epoch"]) in checkpoint_epochs:
                candidates.append(
                    {
                        "head": config["head"],
                        "config": {
                            key: config[key]
                            for key in (
                                "head",
                                "sigma_v",
                                "gain_w",
                                "gain_b",
                                "mean_init",
                                "v2bar_init",
                                "v2bar_weight_var",
                                "v2bar_bias_var",
                            )
                            if key in config
                        },
                        "seed": config["seed"],
                        "run_dir": str(history_path.parent),
                        "record": record,
                    }
                )
    if not candidates:
        raise FileNotFoundError(f"no completed candidates under {root}")
    selected = {}
    for head in sorted({candidate["head"] for candidate in candidates}):
        group = [candidate for candidate in candidates if candidate["head"] == head]
        best_accuracy = max(item["record"]["val_accuracy"] for item in group)
        eligible = [
            item
            for item in group
            if item["record"]["val_accuracy"] >= best_accuracy - 0.01
        ]
        winner = min(
            eligible,
            key=lambda item: (
                item["record"]["val_nll"],
                item["record"]["val_brier"],
                item["record"]["val_ece"],
            ),
        )
        selected[head] = winner
    output = {
        "dataset": args.dataset,
        "stage": args.stage,
        "selection_rule": "min val NLL within 1pp of best accuracy; ties Brier then ECE",
        "selected": selected,
    }
    atomic_json(root / "selection.json", output)
    print(json.dumps(output, indent=2))


def predict_batches(classifier, features: torch.Tensor, batch_size: int):
    probability_parts, epistemic_parts = [], []
    for start in range(0, features.shape[0], batch_size):
        prediction = classifier.predict(
            features[start : start + batch_size], sigma_v=classifier.sigma_v
        )
        probability_parts.append(prediction.probabilities.cpu())
        if prediction.epistemic_variance is not None:
            epistemic_parts.append(
                prediction.epistemic_variance.reshape(
                    prediction.epistemic_variance.shape[0], -1
                ).mean(1).cpu()
            )
    probabilities = torch.cat(probability_parts)
    epistemic = torch.cat(epistemic_parts) if epistemic_parts else None
    return probabilities, epistemic


def evaluate_probabilities(
    probabilities_id,
    labels_id,
    probabilities_ood,
    epistemic_id=None,
    epistemic_ood=None,
    labels_ood=None,
):
    result = {
        "classification": classification_metrics(probabilities_id, labels_id),
        "ood": evaluate_ood_comprehensive(
            probabilities_id,
            probabilities_ood,
            epistemic_id=epistemic_id,
            epistemic_ood=epistemic_ood,
        ),
    }
    if labels_ood is not None:
        result["shift_classification"] = classification_metrics(
            probabilities_ood, labels_ood
        )
    return result


CONFIRM_STAGES = {
    "confirm": ("confirmation", None),
    "init_confirm": ("init_study", "init_confirm"),
}


def confirm_checkpoints(manifest: dict[str, Any], stage: str) -> set[int]:
    """Return the checkpoint epochs the given confirmation stage saved."""

    if stage == "init_confirm":
        return set(manifest["init_study"]["confirm"]["checkpoints"])
    return set(manifest["confirmation"]["checkpoints"])


def evaluate_study(args, manifest: dict[str, Any]) -> None:
    dataset = args.dataset
    stage = getattr(args, "stage", None) or "confirm"
    if stage not in CONFIRM_STAGES:
        raise ValueError(f"evaluate expects a confirmation stage, got {stage}")
    features = feature_root(manifest, dataset)
    clean = load_feature_shard(features / "test.pt")
    svhn = load_feature_shard(features / "svhn.pt")
    output_root = artifact_root(manifest) / "evaluations" / dataset
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_id = torch.softmax(clean["logits"], dim=1)
    baseline_ood = torch.softmax(svhn["logits"], dim=1)
    baseline = {
        "clean_and_svhn": evaluate_probabilities(
            baseline_id, clean["labels"], baseline_ood
        ),
        "corruptions": {},
    }
    for shard_path in sorted((features / "corruptions").glob("*.pt")):
        shard = load_feature_shard(shard_path)
        probabilities = torch.softmax(shard["logits"], dim=1)
        baseline["corruptions"][shard_path.stem] = evaluate_probabilities(
            baseline_id, clean["labels"], probabilities,
            labels_ood=shard["labels"],
        )
    atomic_json(output_root / "pytorch_softmax.json", baseline)

    confirm = stage_root(manifest, stage, dataset)
    # ``confirm`` keeps the original flat layout so the evaluations already on
    # disk stay where the v2 report expects them; later stages get a subtree.
    run_output_root = output_root if stage == "confirm" else output_root / stage
    for run_config_path in confirm.glob("*/*/config.json"):
        run_dir = run_config_path.parent
        config = json.loads(run_config_path.read_text())
        history = json.loads((run_dir / "history.json").read_text())
        checkpoint_epochs = confirm_checkpoints(manifest, stage)
        selectable = [
            row for row in history if int(row["epoch"]) in checkpoint_epochs
        ]
        best_accuracy = max(row["val_accuracy"] for row in selectable)
        eligible = [
            row for row in selectable if row["val_accuracy"] >= best_accuracy - 0.01
        ]
        best = min(eligible, key=lambda row: (row["val_nll"], row["val_brier"], row["val_ece"]))
        requested_epochs = sorted({int(best["epoch"]), config["epochs"]})
        for epoch in requested_epochs:
            checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
            if not checkpoint.exists():
                continue
            classifier, metadata = TAGILastLayerClassifier.load(
                checkpoint, device=args.device
            )
            probability_id, epistemic_id = predict_batches(
                classifier, clean["features"], args.batch_size
            )
            probability_svhn, epistemic_svhn = predict_batches(
                classifier, svhn["features"], args.batch_size
            )
            result = {
                "config": config,
                "checkpoint": str(checkpoint),
                "metadata": metadata,
                "selection_epoch": int(best["epoch"]),
                "clean_and_svhn": evaluate_probabilities(
                    probability_id,
                    clean["labels"],
                    probability_svhn,
                    epistemic_id,
                    epistemic_svhn,
                ),
                "corruptions": {},
            }
            for shard_path in sorted((features / "corruptions").glob("*.pt")):
                shard = load_feature_shard(shard_path)
                probabilities, epistemic = predict_batches(
                    classifier, shard["features"], args.batch_size
                )
                result["corruptions"][shard_path.stem] = evaluate_probabilities(
                    probability_id,
                    clean["labels"],
                    probabilities,
                    epistemic_id,
                    epistemic,
                    labels_ood=shard["labels"],
                )
            relative = run_dir.relative_to(confirm)
            destination = run_output_root / relative / f"epoch_{epoch:04d}.json"
            atomic_json(destination, result)
            print(f"evaluated {destination}")


def flatten_metrics(prefix: str, value: Any, row: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_metrics(f"{prefix}_{key}" if prefix else key, child, row)
    elif isinstance(value, (int, float)):
        row[prefix] = value


def _t_critical_95(sample_count: int) -> float:
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    return table.get(sample_count, 1.96 if sample_count > 30 else 2.262)


CONFIG_IDENTITY_FIELDS = ("stage", "mean_init", "gain_w", "gain_b", "sigma_v")


def config_identity(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return the config fields that identify a study cell, blank when absent.

    The init study evaluates several cells per (dataset, head) -- the selected
    initialization arm and its ``random`` counterpart -- so these fields have to
    travel into the report and into the summary group key, or the arms average
    together. Runs predating an axis leave its field blank rather than claiming
    a value that was never recorded.
    """

    config = config or {}
    identity = {}
    for field in CONFIG_IDENTITY_FIELDS:
        value = config.get(field)
        identity[field] = "" if value is None else value
    return identity


def evaluation_kinds(epoch: int, selection_epoch: int, final_epoch: int) -> list[str]:
    """Return every report group to which an evaluated checkpoint belongs."""

    kinds = []
    if epoch == selection_epoch:
        kinds.append("validation_selected")
    if epoch == final_epoch:
        kinds.append("epoch_200")
    return kinds


def report_study(args, manifest: dict[str, Any]) -> None:
    evaluation_root = artifact_root(manifest) / "evaluations"
    rows = []
    for path in sorted(evaluation_root.glob("**/*.json")):
        payload = json.loads(path.read_text())
        if path.name == "pytorch_softmax.json":
            candidate_rows = [
                {
                    "dataset": path.parent.name,
                    "head": "pytorch_softmax",
                    "seed": "",
                    "evaluation_kind": "fixed_baseline",
                    **config_identity(None),
                }
            ]
        elif "config" in payload:
            epoch = int(payload["metadata"]["epoch"])
            kinds = evaluation_kinds(
                epoch,
                int(payload.get("selection_epoch", epoch)),
                int(payload["config"]["epochs"]),
            )
            candidate_rows = [
                {
                    "dataset": payload["config"]["dataset"],
                    "head": payload["config"]["head"],
                    "seed": payload["config"]["seed"],
                    "epoch": epoch,
                    "evaluation_kind": kind,
                    "checkpoint": payload["checkpoint"],
                    **config_identity(payload["config"]),
                }
                for kind in kinds
            ]
        else:
            continue
        for row in candidate_rows:
            flatten_metrics("clean_svhn", payload["clean_and_svhn"], row)
            corruption_metrics = payload.get("corruptions", {})
            if corruption_metrics:
                metric_values: dict[str, list[float]] = defaultdict(list)
                for result in corruption_metrics.values():
                    flat = {}
                    flatten_metrics("", result, flat)
                    for key, value in flat.items():
                        metric_values[key].append(value)
                for key, values in metric_values.items():
                    row[f"corruption_macro_{key}"] = sum(values) / len(values)
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no evaluations under {evaluation_root}")
    fieldnames = sorted({key for row in rows for key in row})
    destination = artifact_root(manifest) / "report.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(destination.with_suffix(".json"), rows)

    identifiers = {
        "dataset",
        "head",
        "seed",
        "epoch",
        "evaluation_kind",
        "checkpoint",
        *CONFIG_IDENTITY_FIELDS,
    }
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["dataset"],
                row["head"],
                row["evaluation_kind"],
                *(str(row[field]) for field in CONFIG_IDENTITY_FIELDS),
            )
        ].append(row)
    summaries = []
    for group_key, group in sorted(groups.items()):
        dataset, head, evaluation_kind = group_key[:3]
        cell = dict(zip(CONFIG_IDENTITY_FIELDS, group_key[3:]))
        for field in CONFIG_IDENTITY_FIELDS:
            cell[field] = group[0][field]
        metric_names = sorted(
            {
                key
                for row in group
                for key, value in row.items()
                if key not in identifiers and isinstance(value, (int, float))
            }
        )
        for metric in metric_names:
            values = torch.tensor(
                [row[metric] for row in group if metric in row], dtype=torch.float64
            )
            count = values.numel()
            mean = values.mean().item()
            standard_deviation = values.std(unbiased=True).item() if count > 1 else 0.0
            ci95 = (
                _t_critical_95(count) * standard_deviation / math.sqrt(count)
                if count > 1
                else 0.0
            )
            summaries.append(
                {
                    "dataset": dataset,
                    "head": head,
                    "evaluation_kind": evaluation_kind,
                    **cell,
                    "metric": metric,
                    "n": count,
                    "mean": mean,
                    "std": standard_deviation,
                    "ci95": ci95,
                }
            )
    summary_path = destination.with_name("report_summary.csv")
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    atomic_json(summary_path.with_suffix(".json"), summaries)
    print(f"wrote {destination} and {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    subparsers = parser.add_subparsers(dest="command", required=True)

    backbone = subparsers.add_parser("backbone")
    backbone.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    backbone.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    backbone.add_argument("--workers", type=int, default=4)
    backbone.add_argument("--no-download", action="store_true")

    cache = subparsers.add_parser("cache")
    cache.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    cache.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cache.add_argument("--batch-size", type=int, default=256)
    cache.add_argument("--workers", type=int, default=4)
    cache.add_argument("--no-download", action="store_true")
    cache.add_argument("--skip-corruptions", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    run.add_argument(
        "--stage",
        choices=("screen", "refine", "tagiv", "confirm", "init_screen", "init_confirm"),
        required=True,
    )
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--force", action="store_true")

    select = subparsers.add_parser("select")
    select.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    select.add_argument(
        "--stage",
        choices=("screen", "refine", "tagiv", "confirm", "init_screen", "init_confirm"),
        required=True,
    )

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    evaluate.add_argument(
        "--stage", choices=tuple(CONFIRM_STAGES), default="confirm"
    )
    evaluate.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    evaluate.add_argument("--batch-size", type=int, default=512)

    subparsers.add_parser("report")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.command == "backbone":
        train_backbone(args, manifest)
    elif args.command == "cache":
        cache_features(args, manifest)
    elif args.command == "run":
        run_stage(args, manifest)
    elif args.command == "select":
        select_stage(args, manifest)
    elif args.command == "evaluate":
        evaluate_study(args, manifest)
    elif args.command == "report":
        report_study(args, manifest)


if __name__ == "__main__":
    main()
