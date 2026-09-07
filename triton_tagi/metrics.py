"""Calibration, proper scoring, and out-of-distribution metrics."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def fit_softmax_temperature(
    logits: Tensor,
    labels: Tensor,
    *,
    log_temperature_bounds: tuple[float, float] = (-5.0, 5.0),
    iterations: int = 80,
) -> float:
    """Fit one positive softmax temperature by validation NLL.

    A deterministic golden-section search in log-temperature space avoids an
    optimizer-state dependency and keeps this calibration baseline strictly
    one-dimensional.
    """

    if logits.dim() != 2 or logits.shape[0] == 0:
        raise ValueError("logits must have non-empty shape (samples, classes)")
    if labels.dim() != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape (samples,)")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    lower, upper = log_temperature_bounds
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("log_temperature_bounds must be finite and increasing")
    work_logits = logits.detach().double().cpu()
    work_labels = labels.detach().long().cpu()
    if bool(((work_labels < 0) | (work_labels >= logits.shape[1])).any()):
        raise ValueError("labels contain an invalid class index")

    def nll(log_temperature: float) -> float:
        scaled = work_logits / math.exp(log_temperature)
        return torch.nn.functional.cross_entropy(scaled, work_labels).item()

    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    loss_left, loss_right = nll(left), nll(right)
    for _ in range(iterations):
        if loss_left <= loss_right:
            upper, right, loss_right = right, left, loss_left
            left = upper - ratio * (upper - lower)
            loss_left = nll(left)
        else:
            lower, left, loss_left = left, right, loss_right
            right = lower + ratio * (upper - lower)
            loss_right = nll(right)
    return math.exp(0.5 * (lower + upper))


def _validate_probabilities(probabilities: Tensor, labels: Tensor | None = None) -> None:
    if probabilities.dim() != 2 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must have non-empty shape (samples, classes)")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("probabilities must be finite")
    if bool((probabilities < 0).any()):
        raise ValueError("probabilities must be nonnegative")
    if not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(
            probabilities.shape[0],
            device=probabilities.device,
            dtype=probabilities.dtype,
        ),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("each probability row must sum to one")
    if labels is not None:
        if labels.dim() != 1 or labels.shape[0] != probabilities.shape[0]:
            raise ValueError("labels must have shape (samples,)")
        if bool(((labels < 0) | (labels >= probabilities.shape[1])).any()):
            raise ValueError("labels contain an invalid class index")


def expected_calibration_error(
    probabilities: Tensor,
    labels: Tensor,
    *,
    n_bins: int = 15,
    adaptive: bool = False,
) -> float:
    """Return equal-width or equal-mass expected calibration error."""

    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    labels = labels.to(probabilities.device).long()
    _validate_probabilities(probabilities, labels)
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels).float()
    if adaptive:
        order = confidence.argsort()
        bins = torch.tensor_split(order, min(n_bins, probabilities.shape[0]))
        error = probabilities.new_zeros(())
        for indices in bins:
            if indices.numel():
                error += (indices.numel() / probabilities.shape[0]) * (
                    correct[indices].mean() - confidence[indices].mean()
                ).abs()
        return error.item()

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probabilities.device)
    error = probabilities.new_zeros(())
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (confidence >= lower) & (
            confidence <= upper if index == n_bins - 1 else confidence < upper
        )
        if bool(mask.any()):
            error += mask.float().mean() * (correct[mask].mean() - confidence[mask].mean()).abs()
    return error.item()


def classwise_calibration_error(
    probabilities: Tensor,
    labels: Tensor,
    *,
    n_bins: int = 15,
) -> float:
    """Return mean one-vs-rest ECE across classes."""

    labels = labels.to(probabilities.device).long()
    _validate_probabilities(probabilities, labels)
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probabilities.device)
    class_errors = []
    for class_index in range(probabilities.shape[1]):
        confidence = probabilities[:, class_index]
        target = labels.eq(class_index).float()
        error = probabilities.new_zeros(())
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            mask = (confidence >= lower) & (
                confidence <= upper if index == n_bins - 1 else confidence < upper
            )
            if bool(mask.any()):
                error += mask.float().mean() * (target[mask].mean() - confidence[mask].mean()).abs()
        class_errors.append(error)
    return torch.stack(class_errors).mean().item()


def selective_classification_metrics(
    probabilities: Tensor,
    labels: Tensor,
    *,
    coverages: tuple[float, ...] = (0.8, 0.9, 0.95),
) -> dict[str, float]:
    """Return AURC and risks at requested confidence coverages."""

    labels = labels.to(probabilities.device).long()
    _validate_probabilities(probabilities, labels)
    confidence, prediction = probabilities.max(dim=1)
    errors = (~prediction.eq(labels)).float()[confidence.argsort(descending=True)]
    cumulative_risk = errors.cumsum(0) / torch.arange(
        1, errors.numel() + 1, device=errors.device, dtype=errors.dtype
    )
    coverage_axis = (
        torch.arange(1, errors.numel() + 1, device=errors.device, dtype=errors.dtype)
        / errors.numel()
    )
    result = {"aurc": torch.trapezoid(cumulative_risk, coverage_axis).item()}
    for coverage in coverages:
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage values must lie in (0, 1]")
        count = max(1, math.ceil(coverage * errors.numel()))
        result[f"risk_at_{int(round(100 * coverage))}_coverage"] = cumulative_risk[count - 1].item()
    return result


def classification_metrics(
    probabilities: Tensor,
    labels: Tensor,
    *,
    n_bins: int = 15,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Return accuracy, confidence, ECE, multiclass NLL, and Brier score."""

    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    labels = labels.to(probabilities.device).long()
    _validate_probabilities(probabilities, labels)
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels)
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probabilities.device)
    ece = probabilities.new_zeros(())
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (confidence >= lower) & (
            confidence <= upper if index == n_bins - 1 else confidence < upper
        )
        if bool(mask.any()):
            ece += (
                mask.float().mean() * (correct[mask].float().mean() - confidence[mask].mean()).abs()
            )
    true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
    one_hot = torch.zeros_like(probabilities).scatter_(1, labels[:, None], 1.0)
    top_k = min(5, probabilities.shape[1])
    top5 = probabilities.topk(top_k, dim=1).indices.eq(labels[:, None]).any(dim=1)
    result = {
        "accuracy": correct.float().mean().item(),
        "top5_accuracy": top5.float().mean().item(),
        "mean_confidence": confidence.mean().item(),
        "ece": ece.item(),
        "adaptive_ece": expected_calibration_error(
            probabilities, labels, n_bins=n_bins, adaptive=True
        ),
        "classwise_ece": classwise_calibration_error(probabilities, labels, n_bins=n_bins),
        "nll": -true_probability.clamp_min(eps).log().mean().item(),
        "brier": ((probabilities - one_hot) ** 2).sum(dim=1).mean().item(),
    }
    result.update(selective_classification_metrics(probabilities, labels))
    return result


def probability_simplex_deviation(probabilities: Tensor) -> float:
    """Return ``max_n |sum_c p_nc - 1|``.

    A distribution invariant worth reporting alongside any proper score: a
    hierarchical head on a padded tree returns leaf products that do not sum
    to one, so NLL and Brier are meaningless until this is zero.
    """

    if probabilities.dim() != 2 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must have non-empty shape (samples, classes)")
    return float((probabilities.double().sum(dim=1) - 1.0).abs().max())


def predictive_entropy(probabilities: Tensor, eps: float = 1e-12) -> Tensor:
    """Shannon entropy for each probability row."""

    _validate_probabilities(probabilities)
    return -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=1)


def negative_max_probability(probabilities: Tensor) -> Tensor:
    """OOD score where larger values indicate lower maximum confidence."""

    _validate_probabilities(probabilities)
    return -probabilities.max(dim=1).values


def _binary_curve(id_scores: Tensor, ood_scores: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if (
        id_scores.dim() != 1
        or ood_scores.dim() != 1
        or not id_scores.numel()
        or not ood_scores.numel()
    ):
        raise ValueError("ID and OOD scores must be non-empty one-dimensional tensors")
    scores = torch.cat((id_scores.detach().double().cpu(), ood_scores.detach().double().cpu()))
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("OOD scores must be finite")
    labels = torch.cat((torch.zeros(id_scores.numel()), torch.ones(ood_scores.numel())))
    order = torch.argsort(scores, descending=True, stable=True)
    scores, labels = scores[order], labels[order]
    distinct = torch.ones(scores.numel(), dtype=torch.bool)
    distinct[:-1] = scores[:-1] != scores[1:]
    indices = torch.nonzero(distinct, as_tuple=False).squeeze(1)
    tp = labels.cumsum(0)[indices]
    fp = (1.0 - labels).cumsum(0)[indices]
    tp = torch.cat((torch.zeros(1), tp))
    fp = torch.cat((torch.zeros(1), fp))
    tpr = tp / ood_scores.numel()
    fpr = fp / id_scores.numel()
    precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.ones_like(tp))
    return fpr, tpr, precision


def ood_detection_metrics(id_scores: Tensor, ood_scores: Tensor) -> dict[str, float]:
    """Return AUROC, average precision (AUPR-OOD), and FPR at 95% TPR."""

    fpr, tpr, precision = _binary_curve(id_scores, ood_scores)
    auroc = torch.trapezoid(tpr, fpr)
    aupr = ((tpr[1:] - tpr[:-1]) * precision[1:]).sum()
    candidates = fpr[tpr >= 0.95]
    fpr95 = candidates.min() if candidates.numel() else torch.tensor(1.0)
    return {"auroc": auroc.item(), "aupr": aupr.item(), "fpr95": fpr95.item()}


def evaluate_ood(
    probabilities_id: Tensor, probabilities_ood: Tensor
) -> dict[str, dict[str, float]]:
    """Evaluate entropy and negative-max-probability OOD scores."""

    return {
        "entropy": ood_detection_metrics(
            predictive_entropy(probabilities_id), predictive_entropy(probabilities_ood)
        ),
        "negative_max_probability": ood_detection_metrics(
            negative_max_probability(probabilities_id),
            negative_max_probability(probabilities_ood),
        ),
    }


def ood_detection_metrics_full(
    id_scores: Tensor,
    ood_scores: Tensor,
) -> dict[str, float]:
    """Return AUROC, AUPR-OOD, AUPR-ID, and FPR95."""

    base = ood_detection_metrics(id_scores, ood_scores)
    id_positive = ood_detection_metrics(-ood_scores, -id_scores)
    return {
        "auroc": base["auroc"],
        "aupr_ood": base["aupr"],
        "aupr_id": id_positive["aupr"],
        "fpr95": base["fpr95"],
    }


def evaluate_ood_comprehensive(
    probabilities_id: Tensor,
    probabilities_ood: Tensor,
    *,
    epistemic_id: Tensor | None = None,
    epistemic_ood: Tensor | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate common OOD scores and an optional native epistemic score."""

    result = {
        "entropy": ood_detection_metrics_full(
            predictive_entropy(probabilities_id), predictive_entropy(probabilities_ood)
        ),
        "negative_max_probability": ood_detection_metrics_full(
            negative_max_probability(probabilities_id),
            negative_max_probability(probabilities_ood),
        ),
    }
    if (epistemic_id is None) != (epistemic_ood is None):
        raise ValueError("both epistemic score tensors must be supplied together")
    if epistemic_id is not None and epistemic_ood is not None:
        result["native_epistemic"] = ood_detection_metrics_full(epistemic_id, epistemic_ood)
    return result


def training_required_epoch(
    records: list[dict[str, float]] | tuple[dict[str, float], ...],
    *,
    nll_key: str = "val_nll",
    accuracy_key: str = "val_accuracy",
    nll_tolerance: float = 0.01,
    accuracy_tolerance: float = 0.005,
    sustain: int = 2,
) -> int | None:
    """Return the earliest checkpoint satisfying sustained performance criteria."""

    usable = [
        record
        for record in records
        if nll_key in record and accuracy_key in record and "epoch" in record
    ]
    if not usable:
        return None
    min_nll = min(record[nll_key] for record in usable)
    max_accuracy = max(record[accuracy_key] for record in usable)
    for index, record in enumerate(usable):
        window = usable[index : index + sustain + 1]
        if len(window) < sustain + 1:
            break
        if all(
            candidate[nll_key] <= min_nll * (1.0 + nll_tolerance)
            and candidate[accuracy_key] >= max_accuracy - accuracy_tolerance
            for candidate in window
        ):
            return int(record["epoch"])
    return None


def epistemic_convergence(
    epochs: Tensor,
    values: Tensor,
) -> dict[str, float | str | None]:
    """Summarize long-horizon epistemic shrinkage using the study contract."""

    epochs = epochs.detach().double().cpu().flatten()
    values = values.detach().double().cpu().flatten()
    if epochs.numel() != values.numel() or epochs.numel() < 2:
        raise ValueError("epochs and values must be matching vectors with at least two entries")
    if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError("epistemic values must be finite and positive")

    def value_at(epoch: int) -> float | None:
        match = torch.nonzero(epochs == epoch, as_tuple=False).flatten()
        return None if not match.numel() else float(values[match[0]].item())

    changes = []
    for start, end in ((170, 180), (180, 190), (190, 200)):
        first, last = value_at(start), value_at(end)
        if first is None or last is None:
            changes = []
            break
        changes.append((last - first) / first)

    if not changes:
        status = "insufficient_horizon"
    elif all(abs(change) <= 0.01 for change in changes):
        status = "plateau"
    elif all(change < -0.01 for change in changes):
        status = "still_shrinking"
    else:
        status = "rebound_or_unstable"

    tail = epochs >= 101
    slope = None
    if int(tail.sum()) >= 2:
        x = epochs[tail]
        y = values[tail].log()
        centered = x - x.mean()
        slope = float(((centered * (y - y.mean())).sum() / centered.square().sum()).item())
    return {
        "status": status,
        "log_slope_101_200": slope,
        "u_final_over_u_initial": float((values[-1] / values[0]).item()),
        "change_170_180": None if not changes else changes[0],
        "change_180_190": None if not changes else changes[1],
        "change_190_200": None if not changes else changes[2],
    }
