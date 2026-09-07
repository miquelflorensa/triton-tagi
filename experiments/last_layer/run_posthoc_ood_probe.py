"""Diagnose OOD information in the fixed CIFAR-10 ResNet-18 representation.

This is deliberately an oracle-style isolation experiment, not a proposed OOD
methodology. A Gaussian shared-covariance domain discriminant is fitted on the
predeclared CIFAR-10 validation features and one half of SVHN. Evaluation uses
the untouched CIFAR-10 test set and the disjoint second half of SVHN.

The selected multinomial-probit ADF head is never retrained. Its conditional
class probabilities are composed with the post-hoc domain posterior to form a
proper K+1 open-set distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import (  # noqa: E402
    TAGILastLayerClassifier,
    classification_metrics,
    ood_detection_metrics_full,
    predictive_entropy,
)
from triton_tagi.cifar_study import load_feature_shard  # noqa: E402


DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k"
)
RIDGES = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SAMPLE_COUNTS = (300, 500, 1000, 2500, 5000, 10000)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def stable_cholesky(matrix: Tensor) -> Tensor:
    factor, info = torch.linalg.cholesky_ex(matrix)
    if not bool(info.any()):
        return factor
    dimension = matrix.shape[0]
    scale = matrix.diagonal().mean().clamp_min(torch.finfo(matrix.dtype).tiny)
    identity = torch.eye(dimension, device=matrix.device, dtype=matrix.dtype)
    jitter = torch.finfo(matrix.dtype).eps * dimension * scale
    for _ in range(8):
        factor, info = torch.linalg.cholesky_ex(matrix + jitter * identity)
        if not bool(info.any()):
            return factor
        jitter *= 10.0
    raise ValueError("pooled feature covariance is singular beyond roundoff")


@torch.no_grad()
def fit_domain_discriminant(
    id_features: Tensor,
    ood_features: Tensor,
    *,
    ridge: float,
    device: str | torch.device,
) -> dict[str, Tensor | float]:
    """Fit the equal-covariance Gaussian likelihood-ratio classifier."""

    if ridge < 0.0 or not math.isfinite(ridge):
        raise ValueError("ridge must be finite and nonnegative")
    device = torch.device(device)
    id_features = id_features.to(device=device, dtype=torch.float64)
    ood_features = ood_features.to(device=device, dtype=torch.float64)
    if id_features.dim() != 2 or ood_features.shape[1:] != id_features.shape[1:]:
        raise ValueError("ID and OOD features must have matching N x D shapes")
    id_mean = id_features.mean(0)
    ood_mean = ood_features.mean(0)
    residual = torch.cat((id_features - id_mean, ood_features - ood_mean))
    covariance = residual.mT @ residual / (
        id_features.shape[0] + ood_features.shape[0] - 2
    )
    if ridge:
        covariance = covariance + ridge * torch.eye(
            covariance.shape[0], device=device, dtype=covariance.dtype
        )
    factor = stable_cholesky(covariance)
    weight = torch.cholesky_solve((ood_mean - id_mean)[:, None], factor).squeeze(1)
    log_prior_odds = math.log(ood_features.shape[0] / id_features.shape[0])
    bias = -0.5 * torch.dot(ood_mean + id_mean, weight) + log_prior_odds
    return {
        "weight": weight,
        "bias": bias,
        "id_mean": id_mean,
        "ood_mean": ood_mean,
        "covariance_cholesky": factor,
        "ridge": ridge,
    }


@torch.no_grad()
def domain_score(model: dict[str, Tensor | float], features: Tensor) -> Tensor:
    weight = model["weight"]
    bias = model["bias"]
    assert isinstance(weight, Tensor)
    assert isinstance(bias, Tensor)
    work = features.to(device=weight.device, dtype=weight.dtype)
    return work @ weight + bias


@torch.no_grad()
def subspace_overlap(direction: Tensor, row_vectors: Tensor) -> dict[str, float]:
    """Return how much Euclidean OOD direction lies in a classifier row space."""

    direction = direction.to(device=row_vectors.device, dtype=torch.float64)
    rows = row_vectors.to(dtype=torch.float64)
    projected = rows.mT @ torch.linalg.solve(rows @ rows.mT, rows @ direction)
    norm_fraction = projected.norm() / direction.norm()
    cosine = torch.nn.functional.cosine_similarity(
        direction[None, :], rows, dim=1
    ).abs()
    return {
        "norm_fraction": norm_fraction.item(),
        "energy_fraction": norm_fraction.square().item(),
        "maximum_absolute_row_cosine": cosine.max().item(),
    }


def resolve_checkpoint(study_root: Path) -> Path:
    root = study_root / "heads/multinomial_probit_adf/cifar10"
    winner = json.loads((root / "summary.json").read_text())["winner"]
    config = winner["config"]
    run = (
        f"tau2_{config['probit_tau2']:g}_gain_{config['gain_w']:g}_"
        f"seed_{config['seed']}_epochs_{config['epochs']}"
    )
    return (
        root
        / run
        / "checkpoints"
        / f"epoch_{winner['selected_epoch']:04d}.pt"
    )


@torch.no_grad()
def predict_adf(
    classifier: TAGILastLayerClassifier,
    features: Tensor,
    batch_size: int,
) -> Tensor:
    parts = []
    for start in range(0, features.shape[0], batch_size):
        parts.append(
            classifier.predict(features[start : start + batch_size]).probabilities.cpu()
        )
    return torch.cat(parts)


def save_plot(path: Path, ridge_records: list[dict], sample_records: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].semilogx(
        [max(record["ridge"], 1e-10) for record in ridge_records],
        [100.0 * record["auroc"] for record in ridge_records],
        marker="o",
    )
    axes[0].axhspan(97.0, 98.0, color="#54A24B", alpha=0.15)
    axes[0].set_xlabel("LDA covariance ridge (0 shown at 1e-10)")
    axes[0].set_ylabel("SVHN AUROC (%)")
    axes[0].set_title("Regularization ceiling")

    axes[1].semilogx(
        [record["samples_per_domain"] for record in sample_records],
        [100.0 * record["auroc"] for record in sample_records],
        marker="o",
        color="#E45756",
    )
    axes[1].axhspan(97.0, 98.0, color="#54A24B", alpha=0.15)
    axes[1].set_xlabel("Post-hoc examples per domain")
    axes[1].set_ylabel("SVHN AUROC (%)")
    axes[1].set_title("OOD-supervision learning curve")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Fixed ResNet-18 feature-space OOD probe")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.study_root / "posthoc_ood_probe/multinomial_probit_adf"
    result_path = output / "results.json"
    model_path = output / "domain_discriminant.pt"
    plot_path = output / "diagnostic.png"
    if result_path.exists() and not args.force:
        raise FileExistsError(f"result already exists: {result_path}; use --force")

    feature_root = args.study_root / "features/cifar10"
    validation = load_feature_shard(feature_root / "validation.pt")
    clean = load_feature_shard(feature_root / "test.pt")
    svhn = load_feature_shard(feature_root / "svhn.pt")
    generator = torch.Generator().manual_seed(args.split_seed)
    svhn_order = torch.randperm(svhn["features"].shape[0], generator=generator)
    midpoint = svhn_order.numel() // 2
    svhn_fit_indices = svhn_order[:midpoint]
    svhn_eval_indices = svhn_order[midpoint:]
    id_order = torch.randperm(validation["features"].shape[0], generator=generator)
    id_fit = validation["features"][id_order]
    ood_fit = svhn["features"][svhn_fit_indices]
    ood_eval = svhn["features"][svhn_eval_indices]
    fit_count = min(id_fit.shape[0], ood_fit.shape[0])
    id_fit = id_fit[:fit_count]
    ood_fit = ood_fit[:fit_count]

    started = time.perf_counter()
    ridge_records = []
    models = {}
    for ridge in RIDGES:
        model = fit_domain_discriminant(
            id_fit, ood_fit, ridge=ridge, device=args.device
        )
        models[ridge] = model
        metrics = ood_detection_metrics_full(
            domain_score(model, clean["features"]).cpu(),
            domain_score(model, ood_eval).cpu(),
        )
        ridge_records.append({"ridge": ridge, **metrics})
        print(
            f"ridge={ridge:g} AUROC={metrics['auroc']:.6f} "
            f"FPR95={metrics['fpr95']:.6f}",
            flush=True,
        )

    sample_records = []
    for count in SAMPLE_COUNTS:
        model = fit_domain_discriminant(
            id_fit[:count],
            ood_fit[:count],
            ridge=1e-6,
            device=args.device,
        )
        metrics = ood_detection_metrics_full(
            domain_score(model, clean["features"]).cpu(),
            domain_score(model, ood_eval).cpu(),
        )
        sample_records.append({"samples_per_domain": count, **metrics})
        print(
            f"count={count:05d} AUROC={metrics['auroc']:.6f} "
            f"FPR95={metrics['fpr95']:.6f}",
            flush=True,
        )

    # Unregularized maximum-likelihood LDA is fixed before inspecting the sweep.
    final_model = models[0.0]
    clean_score = domain_score(final_model, clean["features"]).cpu()
    svhn_score = domain_score(final_model, ood_eval).cpu()
    clean_ood_probability = torch.sigmoid(clean_score)
    svhn_ood_probability = torch.sigmoid(svhn_score)

    # Repeat the identical discriminant after the backbone's 512 -> 10 class
    # projection. This reveals whether OOD information survives in the logits.
    logit_model = fit_domain_discriminant(
        validation["logits"][id_order[:fit_count]],
        svhn["logits"][svhn_fit_indices[:fit_count]],
        ridge=0.0,
        device=args.device,
    )
    logit_domain_metrics = ood_detection_metrics_full(
        domain_score(logit_model, clean["logits"]).cpu(),
        domain_score(logit_model, svhn["logits"][svhn_eval_indices]).cpu(),
    )

    checkpoint = resolve_checkpoint(args.study_root)
    classifier, _ = TAGILastLayerClassifier.load(checkpoint, device=args.device)
    direction = final_model["weight"]
    assert isinstance(direction, Tensor)
    backbone_state = torch.load(
        args.study_root / "backbones/cifar10_resnet18.pt",
        map_location=args.device,
        weights_only=False,
    )["model_state_dict"]
    subspace = {
        "backbone_classifier": subspace_overlap(
            direction, backbone_state["network.fc.weight"]
        ),
        "adf_classifier": subspace_overlap(
            direction, classifier.linear.mw.mT
        ),
    }
    clean_adf = predict_adf(classifier, clean["features"], args.batch_size)
    svhn_adf = predict_adf(classifier, ood_eval, args.batch_size)
    open_set_clean = torch.cat(
        (
            clean_adf.double() * (1.0 - clean_ood_probability[:, None]),
            clean_ood_probability[:, None],
        ),
        dim=1,
    )
    domain_metrics = ood_detection_metrics_full(clean_score, svhn_score)
    adf_metrics = ood_detection_metrics_full(
        predictive_entropy(clean_adf), predictive_entropy(svhn_adf)
    )
    state = {
        key: value.cpu() if isinstance(value, Tensor) else value
        for key, value in final_model.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(state, model_path)

    result = {
        "protocol": {
            "purpose": "oracle isolation experiment, not final methodology",
            "representation": "fixed ResNet-18 cached features",
            "conditional_classifier": "selected multinomial-probit ADF unchanged",
            "domain_model": "shared-full-covariance Gaussian LDA",
            "domain_fit_id": "CIFAR-10 validation",
            "domain_fit_ood": "first deterministic half of SVHN test",
            "domain_evaluation_id": "CIFAR-10 test",
            "domain_evaluation_ood": "disjoint second half of SVHN test",
            "split_seed": args.split_seed,
            "fit_samples_per_domain": fit_count,
            "evaluation_id_samples": clean["features"].shape[0],
            "evaluation_ood_samples": ood_eval.shape[0],
            "final_ridge": 0.0,
            "adf_checkpoint": str(checkpoint.resolve()),
        },
        "adf_entropy": adf_metrics,
        "posthoc_domain": domain_metrics,
        "posthoc_backbone_logits": logit_domain_metrics,
        "ood_direction_subspace": subspace,
        "ridge_sweep": ridge_records,
        "sample_learning_curve": sample_records,
        "clean_conditional": classification_metrics(clean_adf, clean["labels"]),
        "clean_open_set": classification_metrics(open_set_clean, clean["labels"]),
        "domain_probability": {
            "clean_mean": clean_ood_probability.mean().item(),
            "svhn_mean": svhn_ood_probability.mean().item(),
            "clean_rejected_at_half": (clean_ood_probability >= 0.5).double().mean().item(),
            "svhn_detected_at_half": (svhn_ood_probability >= 0.5).double().mean().item(),
        },
        "wall_s": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    save_plot(plot_path, ridge_records, sample_records)
    print(json.dumps({
        "adf_entropy": adf_metrics,
        "posthoc_domain": domain_metrics,
        "posthoc_backbone_logits": logit_domain_metrics,
        "ood_direction_subspace": subspace,
        "domain_probability": result["domain_probability"],
        "clean_open_set": result["clean_open_set"],
        "wall_s": result["wall_s"],
    }, indent=2))
    print(f"saved {result_path}")
    print(f"saved {model_path}")
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
