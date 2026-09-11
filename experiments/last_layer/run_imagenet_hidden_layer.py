"""Does a TAGI hidden layer close the ImageNet frozen-head gap?

The CIFAR-100 answer (``run_hidden_layer_cifar100.py``) was head-dependent:
``hrc`` gained 2.4-2.8 points and 0.21 nats from a 512-unit hidden layer, while
the Remax family gained nothing. The reading was that ``hrc`` asks a *linear*
map to score tree nodes over arbitrary class subsets -- which the backbone
never made linearly separable -- whereas Remax's per-class logits are exactly
the function the deterministic softmax already computes.

ImageNet is where that reading gets tested against a real gap. The frozen
``pretrained_fc`` reaches 0.6976 through a single linear layer, and against it
``hrc`` manages 0.390 random / 0.433 zero. If the CIFAR reading holds, depth
should move ``hrc`` substantially here and leave ``remax_lognormal`` where it
is; ``remax_lognormal`` is carried precisely so the prediction can fail.

Deliberately outside ``run_imagenet_init_study.py``'s run-hash space, under its
own artifact root, so the finished 225-row study cannot be disturbed. The
protocol is otherwise that study's: stream the shuffled shards, screen at one
epoch on seed 0, confirm the selected cell at four epochs over three seeds.
Flat arms are re-run here rather than read from the finished study, so every
number in the comparison comes from one grid under one code version.

Examples:
  python experiments/last_layer/run_imagenet_hidden_layer.py run --stage screen
  python experiments/last_layer/run_imagenet_hidden_layer.py report --stage screen
  python experiments/last_layer/run_imagenet_hidden_layer.py run --stage confirm
"""

from __future__ import annotations

import argparse
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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_imagenet_init_study as base  # noqa: E402

from triton_tagi import TAGILastLayerClassifier  # noqa: E402
from triton_tagi.cifar_study import seed_everything, stable_hash  # noqa: E402

NUM_CLASSES = base.NUM_CLASSES
INPUT_DIM = base.INPUT_DIM
ARTIFACT_ROOT = REPOSITORY_ROOT / "runs/imagenet/hidden_layer"

# ``hrc`` is the head the CIFAR result says should gain, and the one with the
# largest ImageNet gap to close. ``remax_lognormal`` is the control the
# prediction is allowed to fail against.
HEADS = ("hrc", "remax_lognormal")
# 512 keeps the backbone's own width; 1024 asks whether a 1001-node tree wants
# more room than the feature dimension gives it.
HIDDEN_ARMS: dict[str, tuple[tuple[int, ...], ...]] = {
    "hrc": ((), (512,), (1024,), (2048,)),
    "remax_lognormal": ((), (512,)),
}
MEAN_INITS = ("random", "zero")
TIED_GAINS = (0.1, 0.3, 1.0)
SIGMA_V = (0.1, 0.3)


def depth_name(hidden: tuple[int, ...]) -> str:
    return "flat" if not hidden else "x".join(str(width) for width in hidden)


def screen_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for head in HEADS:
        for hidden, mean_init, gain, sigma_v in itertools.product(
            HIDDEN_ARMS[head], MEAN_INITS, TIED_GAINS, SIGMA_V
        ):
            config: dict[str, Any] = {
                "head": head,
                "sigma_v": sigma_v,
                "gain_w": gain,
                "gain_b": gain,
                "mean_init": mean_init,
            }
            # Emitted only when non-empty, so a flat cell's config is
            # byte-identical to the finished study's and stays comparable.
            if hidden:
                config["hidden_dims"] = list(hidden)
            configs.append(config)
    return configs


def confirm_configs(stage_root: Path) -> list[dict[str, Any]]:
    selection = json.loads((stage_root / "selection.json").read_text())
    return [choice["config"] for choice in selection["selected"].values()]


def build_classifier(config: dict[str, Any], device: str) -> TAGILastLayerClassifier:
    head_kwargs = dict(config)
    hidden = tuple(head_kwargs.pop("hidden_dims", ()) or ())
    return TAGILastLayerClassifier(
        INPUT_DIM, NUM_CLASSES, device=device, hidden_dims=hidden, **head_kwargs
    )


def run_configuration(
    manifest: dict[str, Any],
    stage: str,
    config: dict[str, Any],
    seed: int,
    epochs: int,
    args: argparse.Namespace,
) -> None:
    batch_size = base.safe_batch_size(
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
    depth = depth_name(tuple(config.get("hidden_dims", ()) or ()))
    run_dir = ARTIFACT_ROOT / stage / config["head"] / depth / f"{run_id}_seed{seed}"
    complete = run_dir / "complete.json"
    if complete.exists() and not args.force:
        print(f"skip complete {run_dir}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    base.atomic_json(run_dir / "config.json", run_config)

    train = base.train_shards(manifest)
    validation = base.validation_shards(manifest)
    validation_batch = base.safe_batch_size(
        manifest["last_layer"]["validation_batch_size"], config["head"], NUM_CLASSES
    )

    seed_everything(seed)
    classifier = build_classifier(config, args.device)
    started = perf_counter()
    records: list[dict[str, float]] = []
    row = {"epoch": 0.0, "wall_s": 0.0}
    row.update(base.evaluate_validation(classifier, validation, batch_size=validation_batch))
    records.append(row)
    for epoch in range(1, epochs + 1):
        base.stream_epoch(classifier, train, batch_size=batch_size, seed=seed, epoch=epoch)
        row = {"epoch": float(epoch), "wall_s": perf_counter() - started}
        row.update(base.evaluate_validation(classifier, validation, batch_size=validation_batch))
        records.append(row)
        base.atomic_json(run_dir / "history.json", records)
        print(
            f"  epoch {epoch}/{epochs} acc={row['val_accuracy']:.4f} "
            f"nll={row['val_nll']:.4f} ece={row['val_ece']:.4f} "
            f"({row['wall_s'] / 60:.1f} min)",
            flush=True,
        )
    classifier.save(run_dir / "checkpoints" / f"epoch_{epochs:04d}.pt", metadata={"epoch": epochs})
    base.atomic_json(run_dir / "history.json", records)
    base.atomic_json(complete, {"config": run_config, "wall_s": records[-1]["wall_s"]})
    print(f"complete {run_dir}", flush=True)


def run_stage(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    if args.stage == "screen":
        configs, seeds, epochs = screen_configs(), [0], args.epochs
    else:
        configs = confirm_configs(ARTIFACT_ROOT / "screen")
        seeds, epochs = [0, 1, 2], args.confirm_epochs
    if args.heads:
        configs = [config for config in configs if config["head"] in args.heads]
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


def collect(stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for complete in sorted((ARTIFACT_ROOT / stage).rglob("complete.json")):
        config = json.loads(complete.read_text())["config"]
        history = json.loads((complete.parent / "history.json").read_text())
        rows.append({**config, "depth": depth_name(tuple(config.get("hidden_dims", ()) or ())),
                     **history[-1]})
    return rows


def select_stage(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    """Lowest validation NLL within 1pp of the best accuracy, per (head, depth, init).

    The rule ``run_imagenet_init_study.select_stage`` uses, refined by depth so
    each arm selects its own cell rather than one depth's gain winning for both.
    """

    rows = collect(args.stage)
    if not rows:
        raise SystemExit(f"no completed runs under {ARTIFACT_ROOT / args.stage}")
    arms: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        arms.setdefault(f"{row['head']}|{row['depth']}|{row['mean_init']}", []).append(row)
    selected = {}
    for name, candidates in arms.items():
        best_accuracy = max(row["val_accuracy"] for row in candidates)
        eligible = [row for row in candidates if row["val_accuracy"] >= best_accuracy - 0.01]
        choice = min(eligible, key=lambda row: row["val_nll"])
        selected[name] = {
            "config": {
                key: choice[key]
                for key in ("head", "sigma_v", "gain_w", "gain_b", "mean_init")
            }
            | ({"hidden_dims": choice["hidden_dims"]} if choice.get("hidden_dims") else {}),
            "val_accuracy": choice["val_accuracy"],
            "val_nll": choice["val_nll"],
            "val_ece": choice["val_ece"],
        }
    base.atomic_json(ARTIFACT_ROOT / args.stage / "selection.json", {"selected": selected})
    print(f"wrote {ARTIFACT_ROOT / args.stage / 'selection.json'}")
    report_stage(args, manifest)


def report_stage(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    rows = collect(args.stage)
    if not rows:
        raise SystemExit(f"no completed runs under {ARTIFACT_ROOT / args.stage}")
    arms: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        arms.setdefault((row["head"], row["mean_init"], row["depth"]), []).append(row)

    print(f"\n# imagenet1k — TAGI hidden layer, {args.stage} stage\n")
    print("Reference: frozen `pretrained_fc` reaches 0.6976 through one linear layer.\n")
    print("| head | init | depth | gain | sigma_v | val acc | val NLL | val ECE | n |")
    print("|---|---|---|---|---|---|---|---|---|")
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key in sorted(arms):
        candidates = arms[key]
        top = max(candidates, key=lambda row: row["val_accuracy"])
        best[key] = top
        head, init, depth = key
        print(
            f"| `{head}` | {init} | {depth} | {top['gain_w']} | {top['sigma_v']} | "
            f"{top['val_accuracy']:.4f} | {top['val_nll']:.4f} | {top['val_ece']:.4f} | "
            f"{len(candidates)} |"
        )

    print("\n## What the hidden layer changed\n")
    print("| head | init | depth | accuracy delta | NLL delta |")
    print("|---|---|---|---|---|")
    for (head, init, depth), top in sorted(best.items()):
        if depth == "flat":
            continue
        flat = best.get((head, init, "flat"))
        if flat is None:
            continue
        print(
            f"| `{head}` | {init} | {depth} | "
            f"{top['val_accuracy'] - flat['val_accuracy']:+.4f} | "
            f"{top['val_nll'] - flat['val_nll']:+.4f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "select", "report"))
    parser.add_argument("--stage", default="screen", choices=("screen", "confirm"))
    parser.add_argument("--manifest", default=base.DEFAULT_MANIFEST)
    parser.add_argument("--heads", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--confirm-epochs", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-shard", type=int, default=0)
    parser.add_argument("--gpu-shards", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = base.load_manifest(args.manifest)
    if args.command == "run":
        run_stage(args, manifest)
    elif args.command == "select":
        select_stage(args, manifest)
    else:
        report_stage(args, manifest)


if __name__ == "__main__":
    main()
