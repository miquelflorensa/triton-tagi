"""Evaluate the directional full-covariance TAGI head by epoch."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triton_tagi import (  # noqa: E402
    FullCovarianceADFClassifier,
    classification_metrics,
    multinomial_probit_epistemic_mutual_information,
    negative_max_probability,
    ood_detection_metrics_full,
    predictive_entropy,
)
from triton_tagi.cifar_study import load_feature_shard, seed_everything  # noqa: E402
from triton_tagi.multinomial_probit import (  # noqa: E402
    multinomial_probit_adf_predictive_probs,
)

DEFAULT_ROOT = ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


@torch.no_grad()
def predict_batches(model, features, batch_size, bald_samples):
    parts = {name: [] for name in ("p", "p0", "variance", "logdet", "bald")}
    for start in range(0, features.shape[0], batch_size):
        prediction = model.predict(features[start : start + batch_size])
        parts["p"].append(prediction.probabilities.cpu())
        parts["p0"].append(
            multinomial_probit_adf_predictive_probs(
                prediction.output_mean,
                prediction.output_variance,
                probit_tau2=0.0,
            ).cpu()
        )
        parts["variance"].append(prediction.output_variance.mean(1).cpu())
        parts["logdet"].append(prediction.contrast_logdet.cpu())
        parts["bald"].append(
            multinomial_probit_epistemic_mutual_information(
                prediction.output_mean,
                prediction.output_variance,
                probit_tau2=model.probit_tau2,
                num_samples=bald_samples,
            ).cpu()
        )
    return {name: torch.cat(values) for name, values in parts.items()}


def score_ood(clean, shifted):
    pairs = {
        "entropy": (
            predictive_entropy(clean["p"]),
            predictive_entropy(shifted["p"]),
        ),
        "negative_max_probability": (
            negative_max_probability(clean["p"]),
            negative_max_probability(shifted["p"]),
        ),
        "native_epistemic": (clean["variance"], shifted["variance"]),
        "contrast_logdet": (clean["logdet"], shifted["logdet"]),
        "bald": (clean["bald"], shifted["bald"]),
        "epistemic_only_entropy": (
            predictive_entropy(clean["p0"]),
            predictive_entropy(shifted["p0"]),
        ),
    }
    return {
        name: ood_detection_metrics_full(id_score, ood_score)
        for name, (id_score, ood_score) in pairs.items()
    }


def flatten(record):
    row = {"epoch": record["epoch"], "wall_s": record["wall_s"]}
    row.update({f"clean_{k}": v for k, v in record["clean"].items()})
    row.update(record["score_means"])
    for score, metrics in record["svhn"].items():
        row.update({f"svhn_{score}_{k}": v for k, v in metrics.items()})
    return row


def write_csv(path, records):
    rows = [flatten(record) for record in records]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=1024)
    parser.add_argument("--bald-samples", type=int, default=16)
    parser.add_argument("--gain", type=float, default=0.1)
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.study_root / "full_covariance_adf"
    result_path = output / "results.json"
    if result_path.exists() and not args.force:
        raise FileExistsError(f"result exists: {result_path}; use --force")

    feature_root = args.study_root / "features/cifar10"
    train = load_feature_shard(feature_root / "train.pt")
    clean_shard = load_feature_shard(feature_root / "test.pt")
    svhn_shard = load_feature_shard(feature_root / "svhn.pt")
    seed_everything(args.seed)
    model = FullCovarianceADFClassifier(
        train["features"].shape[1],
        10,
        device=args.device,
        gain_w=args.gain,
        gain_b=args.gain,
        probit_tau2=args.tau2,
    )
    trajectory = []
    started = time.perf_counter()

    def evaluate(epoch, current, fit_record):
        clean = predict_batches(
            current,
            clean_shard["features"],
            args.prediction_batch_size,
            args.bald_samples,
        )
        svhn = predict_batches(
            current,
            svhn_shard["features"],
            args.prediction_batch_size,
            args.bald_samples,
        )
        record = {
            "epoch": epoch,
            "wall_s": time.perf_counter() - started,
            "fit": fit_record,
            "clean": classification_metrics(
                clean["p"], clean_shard["labels"]
            ),
            "svhn": score_ood(clean, svhn),
            "score_means": {
                "clean_entropy": predictive_entropy(clean["p"]).mean().item(),
                "svhn_entropy": predictive_entropy(svhn["p"]).mean().item(),
                "clean_variance": clean["variance"].mean().item(),
                "svhn_variance": svhn["variance"].mean().item(),
                "clean_bald": clean["bald"].mean().item(),
                "svhn_bald": svhn["bald"].mean().item(),
            },
        }
        trajectory.append(record)
        atomic_json(
            result_path,
            {
                "config": {
                    **vars(args),
                    "study_root": str(args.study_root),
                    "model": "full_covariance_multinomial_probit_block_ep",
                    "fit_split": "cifar10_train_only",
                    "uses_ood_training": False,
                    "selection_uses_ood": False,
                },
                "trajectory": trajectory,
            },
        )
        print(
            f"epoch={epoch} acc={record['clean']['accuracy']:.4f} "
            f"entropy={record['svhn']['entropy']['auroc']:.4f} "
            f"BALD={record['svhn']['bald']['auroc']:.4f} "
            f"variance={record['svhn']['native_epistemic']['auroc']:.4f}",
            flush=True,
        )

    model.fit(
        train["features"],
        train["labels"],
        batch_size=args.batch_size,
        seed=args.seed,
        epochs=args.epochs,
        epoch_callback=evaluate,
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            key: value.cpu() if isinstance(value, torch.Tensor) else value
            for key, value in model.state_dict().items()
        },
        output / "model.pt",
    )
    write_csv(output / "trajectory.csv", trajectory)
    result = json.loads(result_path.read_text())
    result["wall_s"] = time.perf_counter() - started
    result["best_diagnostic_entropy_epoch"] = max(
        trajectory, key=lambda row: row["svhn"]["entropy"]["auroc"]
    )["epoch"]
    atomic_json(result_path, result)
    print(f"saved {result_path}", flush=True)


if __name__ == "__main__":
    main()
