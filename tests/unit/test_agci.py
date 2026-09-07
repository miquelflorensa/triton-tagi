"""Tests for fixed-tau categorical AGCI and event-conditioned diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi import (
    TAGILastLayerClassifier,
    agci_categorical_moments,
    agci_categorical_posterior,
    agci_event,
    agci_event_diagonal,
    agci_predictive_probs,
    agci_weight_gain_from_kappa,
    binary_probit_agci_event,
    binary_probit_agci_posterior,
    binary_probit_agci_predictive_probs,
    compute_agci_innovation,
)


def test_scalar_binary_event_matches_closed_form():
    mean = torch.tensor([0.7, -0.4], dtype=torch.float64)
    variance = torch.tensor([0.4, 0.9], dtype=torch.float64)
    labels = torch.tensor([1, 0])
    tau = 1.0

    log_mass, posterior_mean, posterior_variance = binary_probit_agci_event(
        mean, variance, labels, tau=tau
    )
    sign = labels.double().mul(2).sub(1)
    scale = torch.sqrt(variance + tau**2)
    t = sign * mean / scale
    expected_log_mass = torch.special.log_ndtr(t)
    mills = torch.exp(
        -0.5 * t.square()
        - 0.5 * math.log(2.0 * math.pi)
        - expected_log_mass
    )
    expected_mean = mean + sign * variance / scale * mills
    expected_variance = variance - (
        variance.square() / scale.square() * mills * (mills + t)
    )

    torch.testing.assert_close(log_mass, expected_log_mass)
    torch.testing.assert_close(posterior_mean, expected_mean)
    torch.testing.assert_close(posterior_variance, expected_variance)
    probabilities = binary_probit_agci_predictive_probs(mean, variance, tau=tau)
    torch.testing.assert_close(probabilities[:, 1], torch.special.ndtr(mean / scale))
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, dtype=torch.float64))


def test_scalar_binary_original_agci_covariance_matches_closed_form():
    mean = torch.tensor([0.7], dtype=torch.float64)
    variance = torch.tensor([0.4], dtype=torch.float64)
    tau = 1.0
    _, mean_zero, covariance_zero = binary_probit_agci_posterior(
        mean, variance, torch.tensor([0]), tau=tau
    )
    _, mean_one, covariance_one = binary_probit_agci_posterior(
        mean, variance, torch.tensor([1]), tau=tau
    )

    scale = torch.sqrt(variance + tau**2)
    standardized = mean / scale
    probability_one = torch.special.ndtr(standardized)
    density = torch.exp(
        -0.5 * standardized.square() - 0.5 * math.log(2.0 * math.pi)
    )
    cross_covariance = variance / scale * density
    expected_covariance = variance - cross_covariance.square() / (
        probability_one * (1.0 - probability_one)
    )

    assert not torch.equal(mean_zero, mean_one)
    torch.testing.assert_close(covariance_zero, expected_covariance)
    torch.testing.assert_close(covariance_one, expected_covariance)


def test_two_utility_agci_matches_exact_half_space_moments():
    mean = torch.tensor([[0.7, -0.2]], dtype=torch.float64)
    variance = torch.tensor([[0.4, 0.9]], dtype=torch.float64)
    tau = 1.0
    log_mass, posterior_mean, posterior_covariance = agci_event(
        mean, variance, torch.tensor([0]), tau=tau, num_quad=64
    )

    scale2 = variance.sum() + 2.0 * tau**2
    scale = torch.sqrt(scale2)
    t = (mean[0, 0] - mean[0, 1]) / scale
    expected_log_mass = torch.special.log_ndtr(t)
    mills = torch.exp(
        -0.5 * t.square()
        - 0.5 * math.log(2.0 * math.pi)
        - expected_log_mass
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

    torch.testing.assert_close(log_mass[0], expected_log_mass, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(posterior_mean[0], expected_mean, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(
        posterior_covariance[0], expected_covariance, rtol=1e-8, atol=1e-10
    )


def test_multiclass_event_covariance_is_symmetric_positive_semidefinite():
    mean = torch.tensor([[0.7, -0.2, 0.1, -0.8]], dtype=torch.float64)
    variance = torch.tensor([[0.4, 0.9, 0.3, 0.6]], dtype=torch.float64)
    _, _, posterior_covariance = agci_event(
        mean, variance, torch.tensor([2]), tau=0.8, num_quad=96
    )

    torch.testing.assert_close(
        posterior_covariance, posterior_covariance.transpose(-1, -2)
    )
    assert torch.linalg.eigvalsh(posterior_covariance).min() >= -1e-12


def test_multiclass_probabilities_are_categorical_and_translation_invariant():
    mean = torch.tensor(
        [[0.8, -0.1, 0.2], [-1.0, 0.3, 1.2]], dtype=torch.float64
    )
    variance = torch.tensor(
        [[0.3, 0.7, 0.5], [0.4, 0.2, 0.8]], dtype=torch.float64
    )
    probabilities = agci_predictive_probs(mean, variance, num_quad=64)
    shifted = agci_predictive_probs(mean + 123.0, variance, num_quad=64)

    assert bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
    torch.testing.assert_close(
        probabilities.sum(-1), torch.ones(2, dtype=torch.float64)
    )
    torch.testing.assert_close(probabilities, shifted, rtol=1e-11, atol=1e-12)


def test_equal_multiclass_utilities_are_uniform():
    mean = torch.zeros(3, 5, dtype=torch.float64)
    variance = torch.full_like(mean, 0.6)
    probabilities = agci_predictive_probs(mean, variance, num_quad=64)
    torch.testing.assert_close(
        probabilities, torch.full_like(probabilities, 0.2), rtol=1e-12, atol=1e-12
    )


def test_original_agci_dense_covariance_is_class_averaged_and_gain_is_exact():
    mean = torch.tensor([[0.2, -0.4, 0.1]], dtype=torch.float64)
    variance = torch.tensor([[0.4, 0.7, 0.5]], dtype=torch.float64)
    probabilities, conditional_means = agci_categorical_moments(
        mean, variance, num_quad=96, class_chunk_size=1
    )

    event_covariances = torch.stack(
        [
            agci_event(mean, variance, torch.tensor([label]), num_quad=96)[2][0]
            for label in range(3)
        ]
    )
    expected_covariance = torch.einsum(
        "bc,cij->bij", probabilities, event_covariances
    )
    label_covariance = torch.diag_embed(probabilities) - (
        probabilities.unsqueeze(-1) * probabilities.unsqueeze(-2)
    )
    deviations = conditional_means - mean.unsqueeze(-2)
    output_label_covariance = deviations.transpose(-1, -2) * probabilities.unsqueeze(-2)
    gain = output_label_covariance @ torch.linalg.pinv(label_covariance)

    for label in range(3):
        _, posterior_mean, posterior_covariance = agci_categorical_posterior(
            mean,
            variance,
            torch.tensor([label]),
            num_quad=96,
            class_chunk_size=2,
        )
        innovation = torch.nn.functional.one_hot(
            torch.tensor([label]), num_classes=3
        ).to(mean) - probabilities
        gain_mean = mean + torch.einsum("bij,bj->bi", gain, innovation)
        torch.testing.assert_close(posterior_mean, conditional_means[:, label])
        torch.testing.assert_close(gain_mean, posterior_mean, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(
            posterior_covariance, expected_covariance, rtol=1e-10, atol=1e-12
        )


def test_innovation_reconstructs_observed_event_projection():
    mean = torch.tensor([[0.2, -0.4, 0.1]], dtype=torch.float64)
    variance = torch.tensor([[0.4, 0.7, 0.5]], dtype=torch.float64)
    labels = torch.tensor([2])
    _, posterior_mean, posterior_variance = agci_event_diagonal(
        mean, variance, labels, num_quad=64
    )
    delta_mean, delta_variance = compute_agci_innovation(
        labels, mean, variance, num_quad=64
    )

    torch.testing.assert_close(mean + variance * delta_mean, posterior_mean)
    torch.testing.assert_close(
        variance + variance.square() * delta_variance,
        posterior_variance,
    )
    assert bool((delta_variance <= 0.0).all())


def test_diagonal_event_path_matches_dense_event_diagonal():
    mean = torch.tensor([[0.2, -0.4, 0.1]], dtype=torch.float64)
    variance = torch.tensor([[0.4, 0.7, 0.5]], dtype=torch.float64)
    labels = torch.tensor([1])

    dense = agci_event(mean, variance, labels, num_quad=96)
    diagonal = agci_event_diagonal(mean, variance, labels, num_quad=96)

    torch.testing.assert_close(diagonal[0], dense[0])
    torch.testing.assert_close(diagonal[1], dense[1])
    torch.testing.assert_close(
        diagonal[2], dense[2].diagonal(dim1=-2, dim2=-1)
    )


def test_joint_utility_scale_invariance():
    mean = torch.tensor([[0.8, -0.1, 0.2]], dtype=torch.float64)
    variance = torch.tensor([[0.3, 0.7, 0.5]], dtype=torch.float64)
    scale = 3.7

    expected = agci_predictive_probs(mean, variance, tau=0.9, num_quad=64)
    actual = agci_predictive_probs(
        scale * mean,
        scale**2 * variance,
        tau=scale * 0.9,
        num_quad=64,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


def test_kappa_initialization_matches_target_utility_variance():
    feature_energy = 0.25
    input_dim = 5
    kappa = 0.8
    tau = 1.2
    bias_gain = 0.3
    gain_w = agci_weight_gain_from_kappa(
        feature_energy,
        input_dim,
        kappa,
        tau=tau,
        bias_gain=bias_gain,
    )

    utility_variance = gain_w**2 * feature_energy + bias_gain**2 / input_dim
    assert utility_variance == pytest.approx(kappa**2 * tau**2)


def test_fresh_agci_prior_and_feature_transform_roundtrip(tmp_path):
    feature_mean = torch.arange(5, dtype=torch.float32)
    classifier = TAGILastLayerClassifier(
        5,
        3,
        head="agci",
        device="cpu",
        gain_w=0.4,
        gain_b=0.2,
        feature_mean=feature_mean,
        feature_scale=2.0,
    )

    torch.testing.assert_close(
        classifier.linear.mw, torch.zeros_like(classifier.linear.mw)
    )
    assert classifier.linear.mb is not None
    torch.testing.assert_close(
        classifier.linear.mb, torch.zeros_like(classifier.linear.mb)
    )
    torch.testing.assert_close(
        classifier.linear.Sw, torch.full_like(classifier.linear.Sw, 0.4**2 / 5)
    )
    assert classifier.linear.Sb is not None
    torch.testing.assert_close(
        classifier.linear.Sb, torch.full_like(classifier.linear.Sb, 0.2**2 / 5)
    )
    raw = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    expected = (raw - feature_mean) / 2.0
    torch.testing.assert_close(classifier._transform_representations(raw), expected)

    classifier.save(tmp_path / "centered_agci.pt")
    restored, _ = TAGILastLayerClassifier.load(
        tmp_path / "centered_agci.pt", device="cpu"
    )
    torch.testing.assert_close(restored._transform_representations(raw), expected)


def test_dense_correlated_covariance_is_rejected_explicitly():
    mean = torch.zeros(1, 3)
    covariance = torch.eye(3).unsqueeze(0)
    covariance[:, 0, 1] = covariance[:, 1, 0] = 0.2
    with pytest.raises(ValueError, match="diagonal TAGI"):
        agci_event(mean, covariance, torch.tensor([0]))


@pytest.mark.parametrize("tau", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_fixed_tau_is_rejected(tau):
    with pytest.raises(ValueError, match="tau"):
        agci_event(torch.zeros(1, 2), torch.ones(1, 2), torch.tensor([0]), tau=tau)


@pytest.mark.cuda
def test_fit_can_batch_validation_predictions():
    classifier = TAGILastLayerClassifier(
        4, 3, head="agci", device="cuda", agci_num_quad=16
    )
    features = torch.randn(7, 4)
    labels = torch.arange(7) % 3
    history = classifier.fit(
        features,
        labels,
        epochs=0,
        validation=(features, labels),
        validation_batch_size=2,
    )

    assert len(history.records) == 1
    assert math.isfinite(history.records[0]["val_nll"])
    with pytest.raises(ValueError, match="validation_batch_size"):
        classifier.fit(features, labels, epochs=0, validation_batch_size=0)


@pytest.mark.cuda
def test_last_layer_agci_trains_predicts_and_roundtrips(tmp_path):
    torch.manual_seed(7)
    classifier = TAGILastLayerClassifier(
        4,
        3,
        head="agci",
        device="cuda",
        gain_w=0.1,
        gain_b=0.1,
        agci_tau=1.0,
        agci_num_quad=32,
        agci_class_chunk_size=2,
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
        prediction.probabilities.sum(-1), torch.ones(12, device="cuda")
    )

    path = tmp_path / "agci.pt"
    classifier.save(path)
    restored, _ = TAGILastLayerClassifier.load(path, device="cuda")
    assert restored.agci_tau == 1.0
    assert restored.agci_num_quad == 32
    assert restored.agci_class_chunk_size == 2
    assert restored.agci_update == "observed_event"
    torch.testing.assert_close(
        restored.predict(features).probabilities, prediction.probabilities
    )
    with pytest.raises(ValueError, match="does not accept sigma_v"):
        classifier.train_step(features, labels, sigma_v=1.0)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("approximation", "jacobian"),
    [("lognormal", "diag"), ("laplace", "full")],
)
def test_agci_remax_trains_on_utilities_and_predicts_remax_moments(
    tmp_path, approximation, jacobian
):
    torch.manual_seed(11)
    classifier = TAGILastLayerClassifier(
        4,
        3,
        head="agci_remax",
        device="cuda",
        agci_num_quad=16,
        remax_approximation=approximation,
        remax_jacobian=jacobian,
        remax_num_quad=8,
    )
    assert len(classifier.net.layers) == 1

    features = torch.randn(9, 4)
    labels = torch.arange(9) % 3
    before = classifier.linear.mw.clone()
    classifier.train_step(features, labels)
    assert not torch.equal(before, classifier.linear.mw)

    prediction = classifier.predict(features)
    assert prediction.diagnostics is not None
    assert prediction.diagnostics["utility_mean"].shape == (9, 3)
    expected_probabilities = prediction.output_mean / prediction.output_mean.sum(
        dim=-1, keepdim=True
    )
    torch.testing.assert_close(prediction.probabilities, expected_probabilities)
    torch.testing.assert_close(prediction.output_variance, prediction.epistemic_variance)
    torch.testing.assert_close(
        prediction.probabilities.sum(-1), torch.ones(9, device="cuda")
    )

    # AGCI only identifies utility contrasts. The predictive ReMax map must
    # therefore be invariant to a shared utility offset.
    assert classifier.linear.mb is not None
    classifier.linear.mb.add_(5.0)
    shifted_prediction = classifier.predict(features)
    torch.testing.assert_close(
        shifted_prediction.probabilities, prediction.probabilities
    )

    path = tmp_path / "agci_remax.pt"
    classifier.save(path)
    restored, _ = TAGILastLayerClassifier.load(path, device="cuda")
    assert restored.head == "agci_remax"
    torch.testing.assert_close(
        restored.predict(features).probabilities, shifted_prediction.probabilities
    )
