"""Tests for the reusable frozen-feature classification study APIs."""

from __future__ import annotations

import pytest
import torch

from triton_tagi import (
    TAGILastLayerClassifier,
    epistemic_convergence,
    training_required_epoch,
)
from triton_tagi.metrics import (
    classwise_calibration_error,
    evaluate_ood_comprehensive,
    expected_calibration_error,
    selective_classification_metrics,
)


def test_extended_probability_metrics_and_training_epoch():
    probabilities = torch.tensor(
        [[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.4, 0.6]]
    )
    labels = torch.tensor([0, 1, 1, 0])
    assert expected_calibration_error(probabilities, labels, n_bins=2) >= 0.0
    assert expected_calibration_error(
        probabilities, labels, n_bins=2, adaptive=True
    ) >= 0.0
    assert classwise_calibration_error(probabilities, labels, n_bins=2) >= 0.0
    selective = selective_classification_metrics(probabilities, labels)
    assert set(selective) == {
        "aurc",
        "risk_at_80_coverage",
        "risk_at_90_coverage",
        "risk_at_95_coverage",
    }
    records = [
        {"epoch": 0.0, "val_nll": 1.0, "val_accuracy": 0.5},
        {"epoch": 1.0, "val_nll": 0.51, "val_accuracy": 0.79},
        {"epoch": 2.0, "val_nll": 0.50, "val_accuracy": 0.80},
        {"epoch": 3.0, "val_nll": 0.50, "val_accuracy": 0.80},
    ]
    assert training_required_epoch(records, sustain=1) == 2


def test_comprehensive_ood_and_epistemic_convergence():
    probability_id = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    probability_ood = torch.tensor([[0.5, 0.5], [0.55, 0.45]])
    result = evaluate_ood_comprehensive(
        probability_id,
        probability_ood,
        epistemic_id=torch.tensor([0.1, 0.2]),
        epistemic_ood=torch.tensor([0.8, 0.9]),
    )
    assert set(result) == {
        "entropy",
        "negative_max_probability",
        "native_epistemic",
    }
    assert set(result["entropy"]) == {"auroc", "aupr_ood", "aupr_id", "fpr95"}

    epochs = torch.arange(0, 201, dtype=torch.float64)
    values = torch.exp(-0.01 * epochs)
    summary = epistemic_convergence(epochs, values)
    assert summary["status"] == "still_shrinking"
    assert summary["log_slope_101_200"] == pytest.approx(-0.01)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("head", "sigma_v"),
    [
        ("probit_ovr", 0.05),
        ("hrc_probit", None),
        ("categorical_tagiv", None),
        ("hrc_tagiv", None),
    ],
)
def test_new_last_layer_heads_train_predict_and_roundtrip(tmp_path, head, sigma_v):
    torch.manual_seed(12)
    classifier = TAGILastLayerClassifier(
        5,
        4,
        head=head,
        sigma_v=sigma_v,
        device="cuda",
        v2bar_bias_var=0.1,
    )
    features = torch.randn(12, 5)
    labels = torch.arange(12) % 4
    before = classifier.linear.mw.clone()
    classifier.train_step(features, labels)
    assert not torch.equal(before, classifier.linear.mw)

    prediction = classifier.predict(features)
    assert prediction.probabilities.shape == (12, 4)
    assert torch.isfinite(prediction.probabilities).all()
    torch.testing.assert_close(
        prediction.probabilities.sum(1),
        torch.ones_like(prediction.probabilities[:, 0]),
    )
    if head.endswith("tagiv"):
        assert prediction.aleatoric_variance is not None
        assert bool((prediction.aleatoric_variance > 0).all())
        with pytest.raises(ValueError, match="do not accept sigma_v"):
            classifier.train_step(features, labels, sigma_v=0.1)

    path = classifier.save(tmp_path / f"{head}.pt", metadata={"epoch": 1})
    restored, metadata = TAGILastLayerClassifier.load(path, device="cuda")
    actual = restored.predict(features).probabilities
    torch.testing.assert_close(actual, prediction.probabilities)
    assert metadata == {"epoch": 1}
