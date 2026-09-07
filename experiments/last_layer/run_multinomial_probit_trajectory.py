"""Track CIFAR-10/SVHN OOD metrics while training multinomial-probit ADF."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import (  # noqa: E402
    TAGILastLayerClassifier,
    classification_metrics,
    negative_max_probability,
    ood_detection_metrics_full,
    predictive_entropy,
)
from triton_tagi.cifar_study import (  # noqa: E402
    load_feature_shard,
    seed_everything,
)


DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT
    / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k"
)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def atomic_csv(path: Path, records: list[dict[str, float]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


@torch.no_grad()
def predict_batches(
    classifier: TAGILastLayerClassifier,
    features: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = []
    epistemic = []
    for start in range(0, features.shape[0], batch_size):
        prediction = classifier.predict(features[start : start + batch_size])
        probabilities.append(prediction.probabilities.cpu())
        assert prediction.epistemic_variance is not None
        epistemic.append(prediction.epistemic_variance.mean(-1).cpu())
    return torch.cat(probabilities), torch.cat(epistemic)


def score_pair(
    probabilities_id: torch.Tensor,
    probabilities_ood: torch.Tensor,
    epistemic_id: torch.Tensor,
    epistemic_ood: torch.Tensor,
) -> dict[str, dict[str, float]]:
    return {
        "entropy": ood_detection_metrics_full(
            predictive_entropy(probabilities_id),
            predictive_entropy(probabilities_ood),
        ),
        "negative_max_probability": ood_detection_metrics_full(
            negative_max_probability(probabilities_id),
            negative_max_probability(probabilities_ood),
        ),
        "native_epistemic": ood_detection_metrics_full(
            epistemic_id,
            epistemic_ood,
        ),
    }


def flatten_epoch_metrics(
    epoch: int,
    wall_s: float,
    classifier: TAGILastLayerClassifier,
    clean_probabilities: torch.Tensor,
    clean_epistemic: torch.Tensor,
    clean_labels: torch.Tensor,
    svhn_probabilities: torch.Tensor,
    svhn_epistemic: torch.Tensor,
) -> dict[str, float]:
    clean = classification_metrics(clean_probabilities, clean_labels)
    ood = score_pair(
        clean_probabilities,
        svhn_probabilities,
        clean_epistemic,
        svhn_epistemic,
    )
    clean_entropy = predictive_entropy(clean_probabilities)
    svhn_entropy = predictive_entropy(svhn_probabilities)
    row: dict[str, float] = {
        "epoch": float(epoch),
        "wall_s": wall_s,
        "clean_accuracy": clean["accuracy"],
        "clean_nll": clean["nll"],
        "clean_brier": clean["brier"],
        "clean_ece": clean["ece"],
        "clean_entropy_mean": clean_entropy.mean().item(),
        "svhn_entropy_mean": svhn_entropy.mean().item(),
        "clean_epistemic_mean": clean_epistemic.mean().item(),
        "svhn_epistemic_mean": svhn_epistemic.mean().item(),
        "epistemic_ood_to_id_ratio": (
            svhn_epistemic.mean() / clean_epistemic.mean().clamp_min(1e-12)
        ).item(),
        "weight_variance_mean": classifier.linear.Sw.mean().item(),
        "bias_variance_mean": classifier.linear.Sb.mean().item(),
    }
    for score_name, metrics in ood.items():
        for metric_name, value in metrics.items():
            row[f"svhn_{score_name}_{metric_name}"] = value
    return row


def save_plot(
    path: Path,
    records: list[dict[str, float]],
    baseline_entropy_auroc: float,
    baseline_entropy_fpr95: float,
    selected_epoch: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in records]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(
        epochs,
        [100 * row["svhn_entropy_auroc"] for row in records],
        label="Predictive entropy",
        color="#F5A623",
        linewidth=2,
    )
    ax.plot(
        epochs,
        [100 * row["svhn_negative_max_probability_auroc"] for row in records],
        label="1 - max probability",
        color="#4C78A8",
    )
    ax.axhline(
        100 * baseline_entropy_auroc,
        color="#666666",
        linestyle="--",
        label="Frozen softmax entropy",
    )
    ax.set_ylabel("SVHN AUROC (%)")
    ax.set_title("Predictive OOD discrimination")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(
        epochs,
        [100 * row["svhn_entropy_fpr95"] for row in records],
        label="Entropy FPR95",
        color="#F5A623",
        linewidth=2,
    )
    ax.plot(
        epochs,
        [100 * row["svhn_native_epistemic_auroc"] for row in records],
        label="Native epistemic AUROC",
        color="#E45756",
    )
    ax.axhline(
        100 * baseline_entropy_fpr95,
        color="#666666",
        linestyle="--",
        label="Softmax entropy FPR95",
    )
    ax.set_ylabel("Metric (%)")
    ax.set_title("Tail error and native uncertainty")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(
        epochs,
        [row["clean_entropy_mean"] for row in records],
        label="Clean CIFAR-10",
        color="#4C78A8",
    )
    ax.plot(
        epochs,
        [row["svhn_entropy_mean"] for row in records],
        label="SVHN",
        color="#E45756",
    )
    ax.set_ylabel("Mean predictive entropy")
    ax.set_title("Entropy separation")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.semilogy(
        epochs,
        [row["clean_epistemic_mean"] for row in records],
        label="Clean CIFAR-10",
        color="#4C78A8",
    )
    ax.semilogy(
        epochs,
        [row["svhn_epistemic_mean"] for row in records],
        label="SVHN",
        color="#E45756",
    )
    ax.set_ylabel("Mean TAGI output variance")
    ax.set_title("Epistemic contraction")
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.axvline(selected_epoch, color="#999999", linestyle=":", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    figure.suptitle(
        "Multinomial-probit ADF: CIFAR-10 to SVHN OOD trajectory",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=1024)
    parser.add_argument("--gain", type=float, default=0.1)
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selected-epoch", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = (
        args.study_root
        / "trajectories/multinomial_probit"
        / f"tau2_{args.tau2:g}_gain_{args.gain:g}_seed_{args.seed}_epochs_{args.epochs}"
    )
    json_path = output / "trajectory.json"
    csv_path = output / "trajectory.csv"
    plot_path = output / "trajectory.png"
    if json_path.exists() and not args.force:
        raise FileExistsError(f"trajectory already exists: {json_path}; use --force")

    features = args.study_root / "features/cifar10"
    train = load_feature_shard(features / "train.pt")
    clean = load_feature_shard(features / "test.pt")
    svhn = load_feature_shard(features / "svhn.pt")
    baseline_clean = torch.softmax(clean["logits"].float(), dim=-1)
    baseline_svhn = torch.softmax(svhn["logits"].float(), dim=-1)
    baseline = score_pair(
        baseline_clean,
        baseline_svhn,
        torch.zeros(baseline_clean.shape[0]),
        torch.zeros(baseline_svhn.shape[0]),
    )["entropy"]

    seed_everything(args.seed)
    classifier = TAGILastLayerClassifier(
        train["features"].shape[1],
        10,
        head="multinomial_probit",
        device=args.device,
        gain_w=args.gain,
        gain_b=args.gain,
        probit_tau2=args.tau2,
    )
    generator = torch.Generator().manual_seed(args.seed)
    config = {
        "head": "multinomial_probit",
        "dataset": "cifar10",
        "ood_dataset": "svhn",
        "gain_w": args.gain,
        "gain_b": args.gain,
        "probit_tau2": args.tau2,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "selection_uses_ood": False,
    }
    records: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(args.epochs + 1):
        clean_probabilities, clean_epistemic = predict_batches(
            classifier, clean["features"], args.prediction_batch_size
        )
        svhn_probabilities, svhn_epistemic = predict_batches(
            classifier, svhn["features"], args.prediction_batch_size
        )
        row = flatten_epoch_metrics(
            epoch,
            time.perf_counter() - started,
            classifier,
            clean_probabilities,
            clean_epistemic,
            clean["labels"],
            svhn_probabilities,
            svhn_epistemic,
        )
        records.append(row)
        atomic_json(
            json_path,
            {"config": config, "baseline_entropy": baseline, "records": records},
        )
        atomic_csv(csv_path, records)
        print(
            f"epoch={epoch:02d} acc={row['clean_accuracy']:.4f} "
            f"nll={row['clean_nll']:.4f} "
            f"entropy_auroc={row['svhn_entropy_auroc']:.4f} "
            f"native_auroc={row['svhn_native_epistemic_auroc']:.4f} "
            f"epi_ratio={row['epistemic_ood_to_id_ratio']:.3f}",
            flush=True,
        )
        if epoch == args.epochs:
            break

        permutation = torch.randperm(train["features"].shape[0], generator=generator)
        for start in range(0, permutation.numel(), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            classifier.train_step(
                train["features"][indices],
                train["labels"][indices],
            )

    save_plot(
        plot_path,
        records,
        baseline["auroc"],
        baseline["fpr95"],
        args.selected_epoch,
    )
    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
