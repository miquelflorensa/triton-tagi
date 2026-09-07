"""Tests for dense multinomial-probit ADF class-event updates."""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi import (
    TAGILastLayerClassifier,
    compute_multinomial_probit_adf_innovation,
    multinomial_probit_adf_event,
    multinomial_probit_adf_predictive_probs,
)


def test_binary_event_matches_exact_half_space_moments():
    mean = torch.tensor([[0.7, -0.2]], dtype=torch.float64)
    variance = torch.tensor([[0.4, 0.9]], dtype=torch.float64)
    tau2 = 1.0

    log_mass, posterior_mean, posterior_covariance = (
        multinomial_probit_adf_event(
            mean, variance, torch.tensor([0]), probit_tau2=tau2
        )
    )

    scale2 = variance[0].sum() + 2.0 * tau2
    scale = torch.sqrt(scale2)
    t = (mean[0, 0] - mean[0, 1]) / scale
    log_expected = torch.special.log_ndtr(t)
    mills = torch.exp(
        -0.5 * t.square()
        - 0.5 * math.log(2.0 * math.pi)
        - log_expected
    )
    contraction = mills * (mills + t)
    gain = torch.tensor(
        [variance[0, 0] / scale, -variance[0, 1] / scale],
        dtype=torch.float64,
    )
    expected_mean = mean[0] + gain * mills
    expected_covariance = torch.diag(variance[0]) - contraction * torch.outer(
        gain, gain
    )

    torch.testing.assert_close(log_mass[0], log_expected)
    torch.testing.assert_close(posterior_mean[0], expected_mean)
    torch.testing.assert_close(posterior_covariance[0], expected_covariance)


def test_binary_predictive_probability_is_exact():
    mean = torch.tensor([[1.3, -0.4]], dtype=torch.float64)
    variance = torch.tensor([[0.2, 0.7]], dtype=torch.float64)
    probabilities = multinomial_probit_adf_predictive_probs(
        mean, variance, probit_tau2=1.0
    )
    t = (mean[0, 0] - mean[0, 1]) / torch.sqrt(variance.sum() + 2.0)
    expected = torch.special.ndtr(t)
    torch.testing.assert_close(
        probabilities,
        torch.stack((expected, 1.0 - expected)).unsqueeze(0),
    )


def test_equal_multiclass_utilities_are_uniform_and_normalized():
    mean = torch.zeros(3, 5, dtype=torch.float64)
    variance = torch.full_like(mean, 0.6)
    probabilities = multinomial_probit_adf_predictive_probs(mean, variance)

    torch.testing.assert_close(
        probabilities,
        torch.full_like(probabilities, 0.2),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        probabilities.sum(-1), torch.ones(3, dtype=torch.float64)
    )


def test_common_epistemic_shift_cancels_from_probabilities():
    mean = torch.tensor([[0.8, -0.1, 0.2]], dtype=torch.float64)
    diagonal = torch.tensor([0.3, 0.7, 0.5], dtype=torch.float64)
    base = torch.diag(diagonal).unsqueeze(0)
    shared_shift = base + 4.0 * torch.ones(
        1, 3, 3, dtype=torch.float64
    )

    base_probs = multinomial_probit_adf_predictive_probs(mean, base)
    shifted_probs = multinomial_probit_adf_predictive_probs(mean, shared_shift)
    torch.testing.assert_close(base_probs, shifted_probs, rtol=1e-12, atol=1e-12)


def test_more_epistemic_difference_variance_reduces_binary_confidence():
    mean = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    low = multinomial_probit_adf_predictive_probs(
        mean, torch.tensor([[0.05, 0.05]], dtype=torch.float64), probit_tau2=0.0
    )
    high = multinomial_probit_adf_predictive_probs(
        mean, torch.tensor([[8.0, 8.0]], dtype=torch.float64), probit_tau2=0.0
    )

    assert 0.5 < high[0, 0] < low[0, 0] < 1.0


def test_extreme_negative_margins_remain_finite():
    mean = torch.tensor([[-100.0, 0.0, 100.0]], dtype=torch.float32)
    variance = torch.full_like(mean, 1e-4)
    log_mass, posterior_mean, posterior_covariance = multinomial_probit_adf_event(
        mean, variance, torch.tensor([0]), probit_tau2=0.0
    )
    probabilities = multinomial_probit_adf_predictive_probs(
        mean, variance, probit_tau2=0.0
    )

    assert torch.isfinite(log_mass).all()
    assert torch.isfinite(posterior_mean).all()
    assert torch.isfinite(posterior_covariance).all()
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))


def test_innovation_reconstructs_projected_posterior_diagonal():
    mean = torch.tensor([[0.2, -0.4, 0.1]], dtype=torch.float64)
    variance = torch.tensor([[0.4, 0.7, 0.5]], dtype=torch.float64)
    labels = torch.tensor([2])
    _, posterior_mean, posterior_covariance = multinomial_probit_adf_event(
        mean, variance, labels
    )
    delta_mean, delta_variance = compute_multinomial_probit_adf_innovation(
        labels, mean, variance
    )

    torch.testing.assert_close(
        mean + variance * delta_mean,
        posterior_mean,
    )
    torch.testing.assert_close(
        variance + variance.square() * delta_variance,
        posterior_covariance.diagonal(dim1=-2, dim2=-1),
    )
    assert bool((delta_variance <= 0.0).all())


@pytest.mark.parametrize("tau2", [-1.0, float("inf"), float("nan")])
def test_invalid_probit_variance_is_rejected(tau2):
    with pytest.raises(ValueError, match="probit_tau2"):
        multinomial_probit_adf_event(
            torch.zeros(1, 2),
            torch.ones(1, 2),
            torch.tensor([0]),
            probit_tau2=tau2,
        )


@pytest.mark.cuda
@pytest.mark.parametrize("tau2", [0.0, 1.0])
def test_last_layer_head_trains_predicts_and_roundtrips(tmp_path, tau2):
    torch.manual_seed(7)
    classifier = TAGILastLayerClassifier(
        4,
        3,
        head="multinomial_probit",
        device="cuda",
        gain_w=0.1,
        gain_b=0.1,
        probit_tau2=tau2,
    )
    features = torch.randn(12, 4)
    labels = torch.arange(12) % 3
    before = classifier.linear.mw.clone()
    classifier.train_step(features, labels)
    assert not torch.equal(before, classifier.linear.mw)

    prediction = classifier.predict(features)
    assert prediction.probabilities.shape == (12, 3)
    assert torch.isfinite(prediction.probabilities).all()
    torch.testing.assert_close(
        prediction.probabilities.sum(-1),
        torch.ones(12, device="cuda"),
    )

    path = tmp_path / "multinomial_probit.pt"
    classifier.save(path)
    restored, _ = TAGILastLayerClassifier.load(path, device="cuda")
    assert restored.probit_tau2 == tau2
    torch.testing.assert_close(
        restored.predict(features).probabilities,
        prediction.probabilities,
    )
    with pytest.raises(ValueError, match="does not accept sigma_v"):
        classifier.train_step(features, labels, sigma_v=1.0)
