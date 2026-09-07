"""Tests for calibration, proper-scoring, and OOD metrics."""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi.metrics import (
    classification_metrics,
    classwise_calibration_error,
    evaluate_ood,
    fit_softmax_temperature,
    negative_max_probability,
    ood_detection_metrics,
    predictive_entropy,
)


def classwise_calibration_error_by_binning(
    probabilities: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """Mean one-vs-rest ECE, written as an explicit loop over classes and bins.

    The shipped implementation accumulates the same quantity with one scatter,
    which is ~19x faster at 100 classes and ~700x at 1000 and so is what makes
    the ImageNet arm of the initialization study affordable. This is the
    definition it has to agree with.
    """

    labels = labels.to(probabilities.device).long()
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
                error += (
                    mask.float().mean()
                    * (target[mask].mean() - confidence[mask].mean()).abs()
                )
        class_errors.append(error)
    return torch.stack(class_errors).mean().item()


@pytest.mark.parametrize(
    ("samples", "classes"),
    ((64, 2), (500, 3), (2000, 10), (2000, 100)),
)
def test_classwise_calibration_error_matches_explicit_binning(samples, classes):
    generator = torch.Generator().manual_seed(samples + classes)
    probabilities = (3.0 * torch.randn(samples, classes, generator=generator)).softmax(dim=1)
    labels = torch.randint(0, classes, (samples,), generator=generator)
    assert classwise_calibration_error(probabilities, labels) == pytest.approx(
        classwise_calibration_error_by_binning(probabilities, labels), abs=1e-6
    )


def test_classwise_calibration_error_handles_probabilities_at_zero_and_one():
    """A one-hot head puts mass exactly on a bin edge and exactly at one."""

    probabilities = torch.eye(5).repeat(20, 1)
    labels = torch.arange(5).repeat(20)
    assert classwise_calibration_error(probabilities, labels) == pytest.approx(0.0, abs=1e-7)
    assert classwise_calibration_error(probabilities, labels) == pytest.approx(
        classwise_calibration_error_by_binning(probabilities, labels), abs=1e-7
    )


def test_classwise_calibration_error_rejects_a_nonpositive_bin_count():
    with pytest.raises(ValueError, match="n_bins must be positive"):
        classwise_calibration_error(torch.tensor([[0.5, 0.5]]), torch.tensor([0]), n_bins=0)


def test_temperature_scaling_reduces_nll_without_changing_predictions():
    logits = torch.tensor(
        [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 1, 1, 0])
    temperature = fit_softmax_temperature(logits, labels)
    raw = torch.softmax(logits, dim=-1)
    calibrated = torch.softmax(logits / temperature, dim=-1)

    assert temperature > 1.0
    assert torch.equal(raw.argmax(dim=-1), calibrated.argmax(dim=-1))
    assert classification_metrics(calibrated, labels)["nll"] < classification_metrics(
        raw, labels
    )["nll"]


def test_classification_metrics_hand_computed():
    probabilities = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    labels = torch.tensor([0, 0])
    metrics = classification_metrics(probabilities, labels, n_bins=2)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["mean_confidence"] == pytest.approx(0.7)
    assert metrics["ece"] == pytest.approx(0.2)
    assert metrics["nll"] == pytest.approx(-(math.log(0.8) + math.log(0.4)) / 2)
    assert metrics["brier"] == pytest.approx((0.08 + 0.72) / 2)


def test_entropy_and_max_probability_scores():
    probabilities = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    entropy = predictive_entropy(probabilities)
    torch.testing.assert_close(entropy, torch.tensor([0.0, math.log(2.0)]))
    torch.testing.assert_close(negative_max_probability(probabilities), torch.tensor([-1.0, -0.5]))


def test_ood_metrics_perfect_and_reversed():
    perfect = ood_detection_metrics(torch.tensor([0.0, 0.1]), torch.tensor([0.9, 1.0]))
    assert perfect == pytest.approx({"auroc": 1.0, "aupr": 1.0, "fpr95": 0.0})
    reversed_metrics = ood_detection_metrics(
        torch.tensor([0.9, 1.0]), torch.tensor([0.0, 0.1])
    )
    assert reversed_metrics["auroc"] == pytest.approx(0.0)
    assert reversed_metrics["fpr95"] == pytest.approx(1.0)


def test_ood_metrics_ties_are_neutral():
    tied = ood_detection_metrics(torch.zeros(4), torch.zeros(4))
    assert tied["auroc"] == pytest.approx(0.5)
    assert tied["aupr"] == pytest.approx(0.5)
    assert tied["fpr95"] == pytest.approx(1.0)


def test_evaluate_ood_has_both_scores():
    probability_id = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    probability_ood = torch.tensor([[0.5, 0.5], [0.6, 0.4]])
    result = evaluate_ood(probability_id, probability_ood)
    assert set(result) == {"entropy", "negative_max_probability"}
    assert all(set(metrics) == {"auroc", "aupr", "fpr95"} for metrics in result.values())


def test_invalid_probabilities_raise():
    with pytest.raises(ValueError, match="sum"):
        classification_metrics(torch.tensor([[0.2, 0.2]]), torch.tensor([0]))
