"""Focused CIFAR-10 frozen-last-layer study for multinomial-probit ADF."""

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

from triton_tagi import TAGILastLayerClassifier, classification_metrics
from triton_tagi.cifar_study import load_feature_shard, seed_everything


DEFAULT_FEATURE_ROOT = (
    REPOSITORY_ROOT
    / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k/features/cifar10"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k"
    / "heads/multinomial_probit_adf/cifar10"
)


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


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
        epistemic_parts.append(
            prediction.epistemic_variance.mean(dim=-1).cpu()
        )
    return torch.cat(probability_parts), torch.cat(epistemic_parts)


def selection_key(record: dict[str, float]) -> tuple[float, float, float]:
    return record["val_nll"], record["val_brier"], record["val_ece"]


def select_record(
    records: list[dict[str, float]],
    checkpoint_epochs: set[int],
) -> dict[str, float]:
    candidates = [
        record
        for record in records
        if int(record["epoch"]) in checkpoint_epochs
    ]
    results = [
        json.loads(path.read_text())
        for path in sorted(args.output.glob("*/result.json"))
    ]
    best_accuracy = max(record["val_accuracy"] for record in candidates)
    eligible = [
        record
        for record in candidates
        if record["val_accuracy"] >= best_accuracy - 0.01
    ]
    return min(eligible, key=selection_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=1024)
    parser.add_argument("--gains", default="0.03,0.1,0.3,1.0")
    parser.add_argument("--tau2", default="0,1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train = load_feature_shard(args.feature_root / "train.pt")
    validation = load_feature_shard(args.feature_root / "validation.pt")
    test = load_feature_shard(args.feature_root / "test.pt")
    checkpoint_epochs = {
        epoch
        for epoch in (0, 1, 2, 5, 10, 20, 30, 50)
        if epoch <= args.epochs
    }
    checkpoint_epochs.add(args.epochs)
    results = []
    started = time.perf_counter()

    for tau2 in parse_floats(args.tau2):
        for gain in parse_floats(args.gains):
            name = f"tau2_{tau2:g}_gain_{gain:g}_seed_{args.seed}_epochs_{args.epochs}"
            run_dir = args.output / name
            result_path = run_dir / "result.json"
            if result_path.exists() and not args.force:
                results.append(json.loads(result_path.read_text()))
                print(f"skip complete {name}", flush=True)
                continue

            seed_everything(args.seed)
            classifier = TAGILastLayerClassifier(
                train["features"].shape[1],
                10,
                head="multinomial_probit",
                device=args.device,
                gain_w=gain,
                gain_b=gain,
                probit_tau2=tau2,
            )
            history = classifier.fit(
                train["features"],
                train["labels"],
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                validation=(validation["features"], validation["labels"]),
                checkpoint_dir=run_dir / "checkpoints",
                checkpoint_epochs=checkpoint_epochs,
                callback=lambda epoch, _, row: print(
                    f"{name} epoch={epoch:02d} "
                    f"acc={row['val_accuracy']:.4f} "
                    f"nll={row['val_nll']:.4f} "
                    f"ece={row['val_ece']:.4f}",
                    flush=True,
                ),
            )
            records = list(history.records)
            selected = select_record(records, checkpoint_epochs)
            selected_epoch = int(selected["epoch"])
            selected_classifier, _ = TAGILastLayerClassifier.load(
                run_dir / "checkpoints" / f"epoch_{selected_epoch:04d}.pt",
                device=args.device,
            )
            probabilities, epistemic = predict_batches(
                selected_classifier,
                test["features"],
                args.prediction_batch_size,
            )
            result = {
                "config": {
                    "head": "multinomial_probit",
                    "probit_tau2": tau2,
                    "gain_w": gain,
                    "gain_b": gain,
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                },
                "selected_epoch": selected_epoch,
                "validation": {
                    key.removeprefix("val_"): value
                    for key, value in selected.items()
                    if key.startswith("val_")
                },
                "test": classification_metrics(
                    probabilities, test["labels"]
                ),
                "test_epistemic_mean": epistemic.mean().item(),
                "history": records,
                "wall_s": time.perf_counter() - started,
            }
            atomic_json(result_path, result)
            results.append(result)
            print(
                f"complete {name} selected={selected_epoch} "
                f"test_acc={result['test']['accuracy']:.4f} "
                f"test_nll={result['test']['nll']:.4f}",
                flush=True,
            )

    results = [
        json.loads(path.read_text())
        for path in sorted(args.output.glob("*/result.json"))
    ]
    best_accuracy = max(result["validation"]["accuracy"] for result in results)
    eligible = [
        result
        for result in results
        if result["validation"]["accuracy"] >= best_accuracy - 0.01
    ]
    winner = min(
        eligible,
        key=lambda result: (
            result["validation"]["nll"],
            result["validation"]["brier"],
            result["validation"]["ece"],
        ),
    )
    baseline = classification_metrics(
        torch.softmax(test["logits"], dim=-1),
        test["labels"],
    )
    summary = {
        "selection_rule": "minimum validation NLL within 1pp of best accuracy",
        "winner": winner,
        "baseline_test": baseline,
        "all_results": results,
        "wall_s": time.perf_counter() - started,
    }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps({
        "winner_config": winner["config"],
        "selected_epoch": winner["selected_epoch"],
        "validation": winner["validation"],
        "test": winner["test"],
        "baseline_test": baseline,
        "wall_s": summary["wall_s"],
    }, indent=2))


if __name__ == "__main__":
    main()
