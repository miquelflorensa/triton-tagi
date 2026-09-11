"""ImageNet-1k arm of the frozen last-layer mean-initialization study.

The CIFAR arm runs through ``run_study.py``, which holds a whole split in
memory. ImageNet cannot: 1.28 M x 512 features plus 1.28 M x 1000 teacher
logits is far past this box's 16 GB of VRAM, so this driver streams the
``features_shuffled`` shards instead and is otherwise the same experiment.

Accuracy and calibration only. No ImageNet OOD source is cached; that was
decided (MEETING_PLAN.md, gap 1), not overlooked, so the OOD columns are
absent rather than empty.

Examples:
  python experiments/last_layer/run_imagenet_init_study.py run --stage screen
  python experiments/last_layer/run_imagenet_init_study.py run --stage screen \
      --gpu-shard 0 --gpu-shards 2
  python experiments/last_layer/run_imagenet_init_study.py select --stage screen
  python experiments/last_layer/run_imagenet_init_study.py run --stage confirm
  python experiments/last_layer/run_imagenet_init_study.py report
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import (  # noqa: E402
    TAGILastLayerClassifier,
    classification_metrics,
)
from triton_tagi.cifar_study import seed_everything, stable_hash  # noqa: E402

DEFAULT_MANIFEST = Path(__file__).with_name("imagenet_init_study.json")
NUM_CLASSES = 1000
INPUT_DIM = 512


# ──────────────────────────────────────────────────────────────────────────────
#  Manifest and paths
# ──────────────────────────────────────────────────────────────────────────────


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


def feature_root(manifest: dict[str, Any]) -> Path:
    return REPOSITORY_ROOT / manifest["paths"]["features"]


def artifact_root(manifest: dict[str, Any]) -> Path:
    return REPOSITORY_ROOT / manifest["paths"]["artifacts"]


def stage_root(manifest: dict[str, Any], stage: str) -> Path:
    return artifact_root(manifest) / stage


def train_shards(manifest: dict[str, Any]) -> list[Path]:
    shards = sorted(feature_root(manifest).glob("train-*.pt"))
    if not shards:
        raise FileNotFoundError(f"no train shards under {feature_root(manifest)}")
    return shards


def validation_shards(manifest: dict[str, Any]) -> list[Path]:
    shards = sorted(feature_root(manifest).glob("val-*.pt"))
    if not shards:
        raise FileNotFoundError(f"no val shards under {feature_root(manifest)}")
    return shards


def backbone_fc_weights(manifest: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the pretrained ResNet-18 ``fc`` weight and bias."""

    payload = torch.load(
        feature_root(manifest) / "pretrained_fc.pt", map_location="cpu", weights_only=False
    )
    return payload["weight"], payload["bias"]


def feature_energy(manifest: dict[str, Any]) -> float:
    """Mean squared feature norm, which the logit variance head's prior needs."""

    statistics = torch.load(
        feature_root(manifest) / "feature_statistics.pt", map_location="cpu", weights_only=False
    )
    return float(statistics["energy"]) * INPUT_DIM


# ──────────────────────────────────────────────────────────────────────────────
#  Grid
# ──────────────────────────────────────────────────────────────────────────────


def screen_configs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """The init x gain x sigma_v grid, with logit_tagiv's duplicate arm dropped.

    ``logit_tagiv`` zeroes its own latent means, so "random" and "zero" are the
    same head for it; only the distinct arms are emitted.
    """

    configs = [
        {
            "head": head,
            "sigma_v": sigma_v,
            "gain_w": gain,
            "gain_b": gain,
            "mean_init": mean_init,
        }
        for head, mean_init, gain, sigma_v in itertools.product(
            manifest["fixed_noise_heads"],
            manifest["mean_init"],
            manifest["tied_gains"],
            manifest["sigma_v"],
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
            manifest["logit_heads"],
            [arm for arm in manifest["mean_init"] if arm != "random"],
            manifest["tied_gains"],
        )
    )
    return configs


def confirm_configs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """The selected arm per head, plus its ``random`` counterpart for the delta.

    Heads that zero their own latent means -- ``logit_tagiv`` -- have no
    distinct ``random`` arm, for the same reason ``screen_configs`` emits only
    two arms for them. Pairing one anyway would spend a full confirm cell
    recomputing the selected arm under a different run hash and would put a
    delta of exactly zero in the table as though it had been measured.
    """

    selection = json.loads((stage_root(manifest, "screen") / "selection.json").read_text())
    zero_mean_heads = set(manifest["logit_heads"])
    configs: list[dict[str, Any]] = []
    for choice in selection["selected"].values():
        base = dict(choice["config"])
        if base not in configs:
            configs.append(base)
        if base["head"] in zero_mean_heads:
            continue
        baseline = {**base, "mean_init": "random"}
        if baseline not in configs:
            configs.append(baseline)
    return configs


def stage_configs(
    manifest: dict[str, Any], stage: str
) -> tuple[list[dict[str, Any]], list[int], int]:
    if stage == "screen":
        phase = manifest["screen"]
        return screen_configs(manifest), phase["seeds"], phase["epochs"]
    if stage == "confirm":
        phase = manifest["confirm"]
        return confirm_configs(manifest), phase["seeds"], phase["epochs"]
    raise ValueError("stage must be screen or confirm")


# ──────────────────────────────────────────────────────────────────────────────
#  Streaming train and evaluate
# ──────────────────────────────────────────────────────────────────────────────


# remax_laplace_diag's Laplace Jacobian materializes a (batch, K, K) tensor in
# laplace_remax's EAZ einsum, so its memory grows with the square of the class
# count. At K = 1000 the manifest's 1024-row validation batch asks for
# 1024 * 1000^2 * 8 = 8.19 GB in a single allocation, which is exactly the
# request that OOMed the first ImageNet screen before this head had trained one
# step. Every other head is unaffected and keeps the manifest's batch.
#
# The budget below sizes the EAZ tensor alone, which is a proxy: measured peak
# allocation is about 5x it, because the einsum's intermediates are live at the
# same time. Measured on a 15.57 GiB card with 11.17 GiB free --
#   batch 134 -> 5.51 GiB peak, batch 256 -> 10.50 GiB peak, batch 1024 -> OOM
# -- so a 1 GiB budget yields batch 134 and a ~5.5 GiB peak. Batch 256 does fit
# in isolation, but at 94% of free memory it is far too tight for an unattended
# run sharing the card, so this head trains at a smaller batch than the others
# on ImageNet. That is a real confound between heads and is stated in the
# deliverable table's caption rather than hidden.
QUADRATIC_HEADS = frozenset({"remax_laplace_diag"})
QUADRATIC_BYTE_BUDGET = 1 << 30
BYTES_PER_ELEMENT = 8


def safe_batch_size(requested: int, head: str, num_classes: int) -> int:
    """Shrink a batch for heads whose activation grows with the class count."""

    if head not in QUADRATIC_HEADS:
        return requested
    per_row = num_classes * num_classes * BYTES_PER_ELEMENT
    affordable = max(1, QUADRATIC_BYTE_BUDGET // per_row)
    return min(requested, affordable)


def centered_logits(logits: torch.Tensor) -> torch.Tensor:
    """The distillation target: teacher logits with the class mean removed."""

    return logits - logits.mean(dim=1, keepdim=True)


def stream_epoch(
    classifier: TAGILastLayerClassifier,
    shards: list[Path],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> int:
    """Run one pass over every shard, shuffling shard order and rows within.

    The shards are already globally shuffled on disk, so this only needs to
    decorrelate repeated epochs from each other.
    """

    generator = torch.Generator().manual_seed(seed * 10_000 + epoch)
    order = torch.randperm(len(shards), generator=generator).tolist()
    is_logit_head = classifier.head == "logit_tagiv"
    seen = 0
    for index in order:
        payload = torch.load(shards[index], map_location="cpu", weights_only=False)
        features, labels = payload["features"], payload["labels"]
        targets = centered_logits(payload["logits"]) if is_logit_head else None
        rows = torch.randperm(features.shape[0], generator=generator)
        for start in range(0, rows.numel(), batch_size):
            selected = rows[start : start + batch_size]
            if is_logit_head:
                classifier.train_step(
                    features[selected], labels[selected], targets=targets[selected]
                )
            else:
                classifier.train_step(
                    features[selected], labels[selected], sigma_v=classifier.sigma_v
                )
            seen += selected.numel()
        del payload, features, labels, targets
    return seen


@torch.no_grad()
def evaluate_validation(
    classifier: TAGILastLayerClassifier,
    shards: list[Path],
    *,
    batch_size: int,
) -> dict[str, float]:
    """Return accuracy and calibration metrics over the held-out shards."""

    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for shard in shards:
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        features = payload["features"]
        for start in range(0, features.shape[0], batch_size):
            prediction = classifier.predict(
                features[start : start + batch_size], sigma_v=classifier.sigma_v
            )
            probabilities.append(prediction.probabilities.float().cpu())
        labels.append(payload["labels"].cpu())
        del payload, features
    metrics = classification_metrics(torch.cat(probabilities), torch.cat(labels))
    return {f"val_{name}": value for name, value in metrics.items()}


def build_classifier(
    manifest: dict[str, Any], config: dict[str, Any], device: str
) -> TAGILastLayerClassifier:
    head_kwargs = dict(config)
    if head_kwargs.get("mean_init") == "backbone":
        head_kwargs["backbone_fc"] = backbone_fc_weights(manifest)
    if config["head"] == "logit_tagiv":
        head_kwargs.setdefault("logit_variance_feature_energy", feature_energy(manifest))
        head_kwargs.setdefault("logit_variance_weight_share", 0.0)
    return TAGILastLayerClassifier(INPUT_DIM, NUM_CLASSES, device=device, **head_kwargs)


def run_configuration(
    manifest: dict[str, Any],
    stage: str,
    config: dict[str, Any],
    seed: int,
    epochs: int,
    args: argparse.Namespace,
) -> None:
    batch_size = safe_batch_size(
        manifest["last_layer"]["batch_size"], config["head"], NUM_CLASSES
    )
    run_config = {
        "dataset": "imagenet1k",
        "stage": stage,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        **config,
    }
    run_id = stable_hash(run_config)
    run_dir = stage_root(manifest, stage) / config["head"] / f"{run_id}_seed{seed}"
    complete = run_dir / "complete.json"
    if complete.exists() and not args.force:
        print(f"skip complete {run_dir}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "config.json", run_config)

    train = train_shards(manifest)
    validation = validation_shards(manifest)
    validation_batch = safe_batch_size(
        manifest["last_layer"]["validation_batch_size"], config["head"], NUM_CLASSES
    )

    seed_everything(seed)
    classifier = build_classifier(manifest, config, args.device)
    started = perf_counter()
    records: list[dict[str, float]] = []
    row = {"epoch": 0.0, "wall_s": 0.0}
    row.update(evaluate_validation(classifier, validation, batch_size=validation_batch))
    records.append(row)
    for epoch in range(1, epochs + 1):
        stream_epoch(
            classifier, train, batch_size=batch_size, seed=seed, epoch=epoch
        )
        row = {"epoch": float(epoch), "wall_s": perf_counter() - started}
        row.update(evaluate_validation(classifier, validation, batch_size=validation_batch))
        records.append(row)
        atomic_json(run_dir / "history.json", records)
        print(
            f"  epoch {epoch}/{epochs} acc={row['val_accuracy']:.4f} "
            f"nll={row['val_nll']:.4f} ece={row['val_ece']:.4f} "
            f"({row['wall_s'] / 60:.1f} min)",
            flush=True,
        )
    classifier.save(run_dir / "checkpoints" / f"epoch_{epochs:04d}.pt", metadata={"epoch": epochs})
    atomic_json(run_dir / "history.json", records)
    atomic_json(complete, {"config": run_config, "wall_s": records[-1]["wall_s"]})
    print(f"complete {run_dir}", flush=True)


def run_stage(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    configs, seeds, epochs = stage_configs(manifest, args.stage)
    cells = [(config, seed) for config in configs for seed in seeds]
    if args.gpu_shards > 1:
        cells = cells[args.gpu_shard :: args.gpu_shards]
    print(
        f"{args.stage}: {len(cells)} cells "
        f"(worker {args.gpu_shard + 1}/{args.gpu_shards}, {epochs} epochs each)",
        flush=True,
    )
    for index, (config, seed) in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] {config} seed={seed}", flush=True)
        run_configuration(manifest, args.stage, config, seed, epochs, args)


# ──────────────────────────────────────────────────────────────────────────────
#  Select and report
# ──────────────────────────────────────────────────────────────────────────────


def select_stage(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    """Pick, per head, the lowest validation NLL within 1pp of the best accuracy.

    The same rule ``run_study.py`` uses on CIFAR, so the two arms of the study
    are selected the same way.
    """

    root = stage_root(manifest, args.stage)
    candidates = []
    for history_path in root.glob("*/*/history.json"):
        config = json.loads((history_path.parent / "config.json").read_text())
        for record in json.loads(history_path.read_text()):
            if record["epoch"] == 0.0:
                continue  # the prior, before any data
            candidates.append(
                {
                    "head": config["head"],
                    "config": {
                        key: config[key]
                        for key in ("head", "sigma_v", "gain_w", "gain_b", "mean_init")
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
            item for item in group if item["record"]["val_accuracy"] >= best_accuracy - 0.01
        ]
        selected[head] = min(
            eligible,
            key=lambda item: (
                item["record"]["val_nll"],
                item["record"]["val_brier"],
                item["record"]["val_ece"],
            ),
        )
    output = {
        "dataset": "imagenet1k",
        "stage": args.stage,
        "selection_rule": "min val NLL within 1pp of best accuracy; ties Brier then ECE",
        "selected": selected,
    }
    atomic_json(root / "selection.json", output)
    print(json.dumps(output, indent=2))


def report_study(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    """Write one row per (stage, config, seed, epoch) to report.csv."""

    rows = []
    for stage in ("screen", "confirm"):
        root = stage_root(manifest, stage)
        for history_path in sorted(root.glob("*/*/history.json")):
            config = json.loads((history_path.parent / "config.json").read_text())
            for record in json.loads(history_path.read_text()):
                rows.append(
                    {
                        "stage": stage,
                        "head": config["head"],
                        "mean_init": config.get("mean_init", "random"),
                        "gain_w": config["gain_w"],
                        "gain_b": config["gain_b"],
                        "sigma_v": config.get("sigma_v"),
                        "seed": config["seed"],
                        "run_dir": str(history_path.parent.relative_to(REPOSITORY_ROOT)),
                        **record,
                    }
                )
    if not rows:
        raise FileNotFoundError("no histories to report")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    destination = artifact_root(manifest) / "report.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {destination} ({len(rows)} rows)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--stage", choices=("screen", "confirm"), required=True)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--force", action="store_true")
    run.add_argument(
        "--gpu-shard", type=int, default=0, help="index of this worker, for a split queue"
    )
    run.add_argument(
        "--gpu-shards", type=int, default=1, help="number of workers splitting the queue"
    )

    select = subparsers.add_parser("select")
    select.add_argument("--stage", choices=("screen", "confirm"), required=True)

    subparsers.add_parser("report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_manifest(args.manifest)
    if args.command == "run":
        if not 0 <= args.gpu_shard < args.gpu_shards:
            raise ValueError("--gpu-shard must lie in [0, --gpu-shards)")
        run_stage(args, manifest)
    elif args.command == "select":
        select_stage(args, manifest)
    else:
        report_study(args, manifest)


if __name__ == "__main__":
    main()
