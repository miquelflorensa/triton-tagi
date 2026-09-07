"""Tests for AGCI with Gumbel decision noise, the argmax-event logit link."""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi import (
    TAGILastLayerClassifier,
    compute_gumbel_agci_innovation,
    compute_logit_site_innovation,
    gumbel_agci_log_evidence_derivatives,
    gumbel_agci_predictive_probs,
    logit_predictive_probs,
)


def _moments(dtype=torch.float64):
    generator = torch.Generator().manual_seed(11)
    mean = torch.randn(3, 4, generator=generator, dtype=dtype)
    variance = torch.rand(3, 4, generator=generator, dtype=dtype) * 0.5 + 0.05
    labels = torch.tensor([0, 2, 3])
    return mean, variance, labels


def _reference_log_evidence(mean, variance, labels, beta, samples=200_000, seed=99):
    """Independent brute-force Monte Carlo estimate of the log evidence."""

    generator = torch.Generator().manual_seed(seed)
    rows = torch.arange(mean.shape[0])
    scale = variance.sqrt()
    parts = []
    for start in range(0, samples, 20_000):
        count = min(20_000, samples - start)
        noise = torch.randn(
            count, *mean.shape, generator=generator, dtype=mean.dtype
        )
        utility = (mean + scale * noise) / beta
        parts.append(utility[:, rows, labels] - torch.logsumexp(utility, dim=-1))
    stacked = torch.cat(parts, dim=0)
    return torch.logsumexp(stacked, dim=0) - math.log(samples)


def test_log_evidence_matches_brute_force_monte_carlo():
    mean, variance, labels = _moments()
    beta = 0.7
    log_evidence, _, _ = gumbel_agci_log_evidence_derivatives(
        labels, mean, variance, beta=beta, num_samples=8192, seed=7
    )
    reference = _reference_log_evidence(mean, variance, labels, beta)
    assert torch.allclose(log_evidence, reference, atol=5e-3)


def test_derivatives_match_finite_differences_of_the_log_evidence():
    mean, variance, labels = _moments()
    beta = 0.7
    _, first, second = gumbel_agci_log_evidence_derivatives(
        labels, mean, variance, beta=beta, num_samples=8192, seed=7
    )
    step = 1e-3
    # Common random numbers keep the finite differences of the reference
    # estimator far more accurate than its absolute Monte Carlo error.
    center = _reference_log_evidence(mean, variance, labels, beta)
    for index in range(mean.shape[1]):
        plus, minus = mean.clone(), mean.clone()
        plus[:, index] += step
        minus[:, index] -= step
        upper = _reference_log_evidence(plus, variance, labels, beta)
        lower = _reference_log_evidence(minus, variance, labels, beta)
        assert torch.allclose(
            first[:, index], (upper - lower) / (2.0 * step), atol=1e-2
        )
        assert torch.allclose(
            second[:, index],
            (upper - 2.0 * center + lower) / step**2,
            atol=2e-2,
        )


def test_deterministic_limit_reproduces_the_multinomial_logit_adf():
    mean, variance, labels = _moments()
    beta = 0.8
    tiny = torch.full_like(variance, 1e-12)
    _, first, second = gumbel_agci_log_evidence_derivatives(
        labels, mean, tiny, beta=beta, num_samples=2, seed=0
    )
    probabilities = torch.softmax(mean / beta, dim=-1)
    residual = -probabilities.clone()
    residual[torch.arange(mean.shape[0]), labels] += 1.0
    assert torch.allclose(first, residual / beta, atol=1e-9)
    assert torch.allclose(
        second, -probabilities * (1.0 - probabilities) / beta**2, atol=1e-9
    )


def test_predictive_probabilities_are_a_normalized_partition():
    mean, variance, labels = _moments()
    beta = 0.7
    probabilities = gumbel_agci_predictive_probs(
        mean, variance, beta=beta, num_samples=8192, seed=7
    )
    assert torch.allclose(
        probabilities.sum(dim=-1), torch.ones(mean.shape[0], dtype=mean.dtype)
    )
    assert bool((probabilities > 0.0).all())
    # The observed-class predictive probability is the evidence itself.
    log_evidence, _, _ = gumbel_agci_log_evidence_derivatives(
        labels, mean, variance, beta=beta, num_samples=8192, seed=7
    )
    observed = probabilities[torch.arange(mean.shape[0]), labels]
    assert torch.allclose(observed, log_evidence.exp(), atol=1e-12)


def test_innovation_keeps_the_implied_posterior_variance_positive():
    mean, variance, labels = _moments()
    delta_mean, delta_variance = compute_gumbel_agci_innovation(
        labels, mean, variance, beta=0.5, num_samples=64, seed=3
    )
    posterior_variance = variance + variance.square() * delta_variance
    assert bool((posterior_variance > 0.0).all())
    assert bool((posterior_variance <= variance + 1e-12).all())
    assert torch.isfinite(delta_mean).all()


def test_zero_output_variance_gives_a_zero_innovation():
    mean, _, labels = _moments()
    zero = torch.zeros_like(mean)
    delta_mean, delta_variance = compute_gumbel_agci_innovation(
        labels, mean, zero, beta=1.0, num_samples=4, seed=0
    )
    assert torch.equal(delta_mean, zero)
    assert torch.equal(delta_variance, zero)


def test_results_are_reproducible_for_a_fixed_seed():
    mean, variance, labels = _moments()
    first = compute_gumbel_agci_innovation(
        labels, mean, variance, beta=0.9, num_samples=16, seed=5
    )
    second = compute_gumbel_agci_innovation(
        labels, mean, variance, beta=0.9, num_samples=16, seed=5
    )
    other = compute_gumbel_agci_innovation(
        labels, mean, variance, beta=0.9, num_samples=16, seed=6
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert not torch.equal(first[0], other[0])


def test_beta_is_a_gauge_of_the_declared_prior_scale():
    """Scaling the utilities and beta together leaves the model unchanged."""

    mean, variance, labels = _moments()
    scale = 3.0
    base = gumbel_agci_predictive_probs(mean, variance, beta=1.0, num_samples=256, seed=1)
    scaled = gumbel_agci_predictive_probs(
        scale * mean, scale**2 * variance, beta=scale, num_samples=256, seed=1
    )
    assert torch.allclose(base, scaled, atol=1e-10)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beta": 0.0},
        {"beta": float("nan")},
        {"num_samples": 3},
        {"num_samples": 0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    mean, variance, labels = _moments()
    with pytest.raises(ValueError):
        compute_gumbel_agci_innovation(labels, mean, variance, **kwargs)


def test_labels_must_index_a_declared_class():
    mean, variance, _ = _moments()
    with pytest.raises(ValueError):
        compute_gumbel_agci_innovation(
            torch.tensor([0, 1, 4]), mean, variance, beta=1.0, num_samples=4
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_classifier_head_trains_and_serializes():
    torch.manual_seed(0)
    features = torch.randn(512, 24, device="cuda")
    weights = torch.randn(24, 6, device="cuda")
    labels = (features @ weights).argmax(dim=1)

    classifier = TAGILastLayerClassifier(
        24,
        6,
        head="gumbel_agci",
        gain_w=0.5,
        gain_b=0.5,
        gumbel_beta=1.0,
        gumbel_num_samples=16,
    )
    before = classifier.predict(features).probabilities
    for start in range(0, 512, 128):
        classifier.train_step(features[start : start + 128], labels[start : start + 128])
    after = classifier.predict(features).probabilities
    assert not torch.allclose(before, after)
    assert torch.allclose(after.sum(dim=-1), torch.ones(512, device="cuda"), atol=1e-5)

    accuracy_before = (before.argmax(1) == labels).float().mean()
    accuracy_after = (after.argmax(1) == labels).float().mean()
    assert accuracy_after > accuracy_before

    path = classifier.save("/tmp/gumbel_agci_head.pt")
    restored, _ = TAGILastLayerClassifier.load(path)
    assert restored.gumbel_beta == classifier.gumbel_beta
    assert restored.gumbel_num_samples == classifier.gumbel_num_samples
    assert torch.allclose(restored.predict(features).probabilities, after)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_head_rejects_sigma_v():
    with pytest.raises(ValueError):
        TAGILastLayerClassifier(8, 3, head="gumbel_agci", sigma_v=0.1)


def test_logit_site_reproduces_the_specified_gaussian_site_algebra():
    """The minimal head must match score, curvature and precision addition."""

    mean, variance, labels = _moments()
    beta = 0.9
    delta_mean, delta_variance = compute_logit_site_innovation(
        labels, mean, variance, beta=beta
    )
    probabilities = torch.softmax(mean / beta, dim=-1)
    one_hot = torch.zeros_like(mean)
    one_hot[torch.arange(mean.shape[0]), labels] = 1.0
    score = (one_hot - probabilities) / beta
    curvature = probabilities * (1.0 - probabilities) / beta**2
    expected_variance = 1.0 / (1.0 / variance + curvature)
    expected_mean = mean + expected_variance * score

    assert torch.allclose(mean + variance * delta_mean, expected_mean, atol=1e-12)
    assert torch.allclose(
        variance + variance.square() * delta_variance, expected_variance, atol=1e-12
    )


def test_logit_site_never_increases_output_variance():
    mean, variance, labels = _moments()
    _, delta_variance = compute_logit_site_innovation(
        labels, mean, variance, beta=1.0
    )
    assert bool((delta_variance <= 0.0).all())
    posterior_variance = variance + variance.square() * delta_variance
    assert bool((posterior_variance > 0.0).all())


def test_logit_site_is_the_zero_variance_limit_of_the_integrated_head():
    mean, _, labels = _moments()
    tiny = torch.full_like(mean, 1e-10)
    site = compute_logit_site_innovation(labels, mean, tiny, beta=0.8)
    integrated = compute_gumbel_agci_innovation(
        labels, mean, tiny, beta=0.8, num_samples=2, seed=0
    )
    assert torch.allclose(site[0], integrated[0], atol=1e-6)
    assert torch.allclose(site[1], integrated[1], atol=1e-6)


def test_logit_site_and_integrated_head_differ_at_realistic_variance():
    """The two ablations must not silently coincide where it matters."""

    mean, _, labels = _moments()
    variance = torch.full_like(mean, 0.5)
    site = compute_logit_site_innovation(labels, mean, variance, beta=1.0)
    integrated = compute_gumbel_agci_innovation(
        labels, mean, variance, beta=1.0, num_samples=4096, seed=1
    )
    assert (site[0] - integrated[0]).abs().max() > 1e-3
    assert (site[1] - integrated[1]).abs().max() > 1e-3


def test_logit_predictive_probs_ignores_output_variance():
    mean, _, _ = _moments()
    probabilities = logit_predictive_probs(mean, beta=0.7)
    assert torch.allclose(probabilities, torch.softmax(mean / 0.7, dim=-1))
    assert torch.allclose(
        probabilities.sum(dim=-1), torch.ones(mean.shape[0], dtype=mean.dtype)
    )


def test_logit_site_rejects_invalid_beta_and_labels():
    mean, variance, _ = _moments()
    with pytest.raises(ValueError):
        compute_logit_site_innovation(
            torch.tensor([0, 1, 2]), mean, variance, beta=0.0
        )
    with pytest.raises(ValueError):
        compute_logit_site_innovation(
            torch.tensor([0, 1, 9]), mean, variance, beta=1.0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_logit_site_head_trains_and_serializes():
    torch.manual_seed(0)
    features = torch.randn(512, 24, device="cuda")
    labels = (features @ torch.randn(24, 6, device="cuda")).argmax(dim=1)
    classifier = TAGILastLayerClassifier(
        24, 6, head="logit_site", gain_w=0.5, gain_b=0.5, gumbel_beta=1.0
    )
    before = (classifier.predict(features).probabilities.argmax(1) == labels).float().mean()
    for start in range(0, 512, 128):
        classifier.train_step(features[start : start + 128], labels[start : start + 128])
    prediction = classifier.predict(features).probabilities
    after = (prediction.argmax(1) == labels).float().mean()
    assert after > before
    assert torch.allclose(prediction.sum(-1), torch.ones(512, device="cuda"), atol=1e-5)

    path = classifier.save("/tmp/logit_site_head.pt")
    restored, _ = TAGILastLayerClassifier.load(path)
    assert restored.head == "logit_site"
    assert torch.allclose(restored.predict(features).probabilities, prediction)
