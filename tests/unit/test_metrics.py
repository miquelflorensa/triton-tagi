"""Tests for calibration, proper-scoring, and OOD metrics."""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi.metrics import (
    classification_metrics,
    evaluate_ood,
    fit_softmax_temperature,
    negative_max_probability,
    ood_detection_metrics,
    predictive_entropy,
)


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
