"""Compare deterministic softmax with TAGI heads on frozen ResNet-18 outputs.

The script performs no temperature scaling or other post-hoc calibration.

Usage:
    python examples/compare_cifar10_tagi_heads.py --checkpoint runs/model.pt
    python examples/compare_cifar10_tagi_heads.py --checkpoint runs/model.pt --smoke
    python examples/compare_cifar10_tagi_heads.py --checkpoint runs/model.pt \
        --remax-approximation laplace --remax-jacobian diag
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from cifar10_torch_common import (
    CifarResNet18,
    cifar10_datasets,
    feature_and_logits,
    svhn_dataset,
)
from triton_tagi import (
    FrozenTorchBackbone,
    RunDir,
    TAGILastLayerClassifier,
    classification_metrics,
    copy_torch_linear_,
    evaluate_ood,
    initialize_identity_linear_,
    load_torch_checkpoint,
    predictive_entropy,
)


@dataclass(frozen=True)
class Arm:
    name: str
    boundary: str
    head: str
    initialization: str


DEFAULT_ARMS = (
    Arm("feature_remax_fresh", "features", "remax", "fresh"),
    Arm("feature_remax_warm", "features", "remax", "warm"),
    Arm("logit_remax_fresh", "logits", "remax", "fresh"),
    Arm("logit_remax_identity", "logits", "remax", "identity"),
    Arm("feature_hrc_fresh", "features", "hrc", "fresh"),
    Arm("logit_hrc_fresh", "logits", "hrc", "fresh"),
)


@torch.no_grad()
def extract_dataset(adapter, dataset, batch_size, workers, max_samples=None):
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    feature_chunks, logit_chunks, label_chunks = [], [], []
    for inputs, labels in loader:
        features, logits = adapter.representations(inputs)
        feature_chunks.append(features.cpu())
        logit_chunks.append(logits.cpu())
        label_chunks.append(torch.as_tensor(labels).long().cpu())
    return {
        "features": torch.cat(feature_chunks),
        "logits": torch.cat(logit_chunks),
        "labels": torch.cat(label_chunks),
    }


def predict_in_batches(classifier, values, batch_size):
    probability_chunks = []
    for start in range(0, len(values), batch_size):
        prediction = classifier.predict(values[start : start + batch_size])
        probability_chunks.append(prediction.probabilities.cpu())
    return torch.cat(probability_chunks)


def train_head(classifier, values, labels, epochs, batch_size, sigma_v, seed):
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(values), generator=generator)
        for start in range(0, len(values), batch_size):
            indices = permutation[start : start + batch_size]
            classifier.train_step(values[indices], labels[indices], sigma_v)
        print(f"    epoch {epoch:3d}/{epochs}", flush=True)


def make_classifier(arm, train, model, args):
    input_dim = train[arm.boundary].shape[1]
    classifier = TAGILastLayerClassifier(
        input_dim,
        10,
        head=arm.head,
        device=args.device,
        gain_w=args.gain_w,
        gain_b=args.gain_b,
        remax_approximation=args.remax_approximation,
        remax_jacobian=args.remax_jacobian,
        remax_num_quad=args.remax_num_quad,
    )
    if arm.initialization == "warm":
        copy_torch_linear_(classifier.linear, model.classifier)
    elif arm.initialization == "identity":
        initialize_identity_linear_(classifier.linear)
    return classifier


def reliability_bins(probabilities, labels, n_bins):
    confidence, prediction = probabilities.max(1)
    correct = prediction.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    accuracy, mean_confidence = [], []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (confidence >= lower) & (
            confidence <= upper if index == n_bins - 1 else confidence < upper
        )
        accuracy.append(correct[mask].mean().item() if mask.any() else float("nan"))
        mean_confidence.append(confidence[mask].mean().item() if mask.any() else float("nan"))
    return edges, accuracy, mean_confidence


def save_plots(run, name, probabilities_id, labels, probabilities_ood, n_bins):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    _, accuracy, confidence = reliability_bins(probabilities_id, labels, n_bins)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.plot(confidence, accuracy, "o-")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy", title=name)
    fig.tight_layout()
    fig.savefig(run.figures / f"{name}_reliability.png", dpi=150)
    plt.close(fig)

    entropy_id = predictive_entropy(probabilities_id).numpy()
    entropy_ood = predictive_entropy(probabilities_ood).numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(entropy_id, bins=40, density=True, alpha=0.55, label="CIFAR-10 (ID)")
    ax.hist(entropy_ood, bins=40, density=True, alpha=0.55, label="SVHN (OOD)")
    ax.set(xlabel="Predictive entropy", ylabel="Density", title=name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run.figures / f"{name}_ood_entropy.png", dpi=150)
    plt.close(fig)


def flatten_result(name, result):
    row = {"arm": name, **result["classification"]}
    for score, metrics in result["ood"].items():
        for metric, value in metrics.items():
            row[f"ood_{score}_{metric}"] = value
    return row


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run = RunDir("cifar10", "resnet18_last_layer", "comparison", base=args.output_dir)
    selected_names = set(args.arms.split(",")) if args.arms else {arm.name for arm in DEFAULT_ARMS}
    unknown = selected_names - {arm.name for arm in DEFAULT_ARMS}
    if unknown:
        raise ValueError(f"Unknown arms: {', '.join(sorted(unknown))}")
    arms = [arm for arm in DEFAULT_ARMS if arm.name in selected_names]
    epochs = 1 if args.smoke else args.epochs
    max_train = 1024 if args.smoke else args.max_train_samples
    max_eval = 512 if args.smoke else args.max_eval_samples
    config = {
        **vars(args),
        "epochs_effective": epochs,
        "arms_effective": [asdict(arm) for arm in arms],
        "max_train_samples_effective": max_train,
        "max_eval_samples_effective": max_eval,
    }
    run.save_config(config)
    print(f"results: {run.path}")

    model = CifarResNet18()
    checkpoint_info = load_torch_checkpoint(
        model,
        args.checkpoint,
        state_key=args.state_key,
        device=device,
        strict=True,
    )
    print(f"loaded deterministic checkpoint: {checkpoint_info.path}")
    adapter = FrozenTorchBackbone(model, feature_and_logits)
    train_set, test_set = cifar10_datasets(args.data_dir, augment_train=False)
    ood_set = svhn_dataset(args.data_dir)

    print("extracting frozen train representations")
    train = extract_dataset(adapter, train_set, args.extract_batch_size, args.workers, max_train)
    print("extracting CIFAR-10 test representations")
    test = extract_dataset(adapter, test_set, args.extract_batch_size, args.workers, max_eval)
    print("extracting SVHN OOD representations")
    ood = extract_dataset(adapter, ood_set, args.extract_batch_size, args.workers, max_eval)
    torch.save({"train": train, "test": test, "ood": ood}, run.path / "representations.pt")

    baseline_id = torch.softmax(test["logits"], dim=1)
    baseline_ood = torch.softmax(ood["logits"], dim=1)
    results = {
        "deterministic_softmax": {
            "classification": classification_metrics(
                baseline_id, test["labels"], n_bins=args.n_bins
            ),
            "ood": evaluate_ood(baseline_id, baseline_ood),
        }
    }
    save_plots(run, "deterministic_softmax", baseline_id, test["labels"], baseline_ood, args.n_bins)

    for arm_index, arm in enumerate(arms):
        print(f"training {arm.name}")
        classifier = make_classifier(arm, train, model, args)
        train_head(
            classifier,
            train[arm.boundary],
            train["labels"],
            epochs,
            args.batch_size,
            args.sigma_v,
            args.seed + arm_index,
        )
        probabilities_id = predict_in_batches(classifier, test[arm.boundary], args.batch_size)
        probabilities_ood = predict_in_batches(classifier, ood[arm.boundary], args.batch_size)
        results[arm.name] = {
            "arm": asdict(arm),
            "classification": classification_metrics(
                probabilities_id, test["labels"], n_bins=args.n_bins
            ),
            "ood": evaluate_ood(probabilities_id, probabilities_ood),
        }
        arm_run = RunDir("cifar10", arm.name, "tagi", base=str(run.path / "heads"))
        arm_config = {
            **config,
            "arm": asdict(arm),
            "input_dim": classifier.input_dim,
            "num_classes": classifier.num_classes,
        }
        arm_run.save_config(arm_config)
        arm_run.save_checkpoint(classifier.net, epochs, arm_config)
        save_plots(
            run, arm.name, probabilities_id, test["labels"], probabilities_ood, args.n_bins
        )

    report = {
        "checkpoint": str(checkpoint_info.path),
        "checkpoint_state_key": checkpoint_info.state_key,
        "config": config,
        "results": results,
    }
    (run.path / "comparison.json").write_text(json.dumps(report, indent=2))
    rows = [flatten_result(name, result) for name, result in results.items()]
    with (run.path / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\narm                         acc     ECE     NLL   Brier  OOD AUROC(entropy)")
    for name, result in results.items():
        metrics = result["classification"]
        auroc = result["ood"]["entropy"]["auroc"]
        print(
            f"{name:28s} {metrics['accuracy']:7.3f} {metrics['ece']:7.3f} "
            f"{metrics['nll']:7.3f} {metrics['brier']:7.3f} {auroc:10.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--extract-batch-size", type=int, default=256)
    parser.add_argument("--sigma-v", type=float, default=0.05)
    parser.add_argument("--gain-w", type=float, default=0.1)
    parser.add_argument("--gain-b", type=float, default=0.1)
    parser.add_argument(
        "--remax-approximation",
        choices=["lognormal", "laplace"],
        default="lognormal",
    )
    parser.add_argument("--remax-jacobian", choices=["diag", "full"], default="diag")
    parser.add_argument("--remax-num-quad", type=int, default=48)
    parser.add_argument(
        "--arms", default=None, help="Comma-separated subset of documented arm names"
    )
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
