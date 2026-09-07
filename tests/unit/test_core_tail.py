"""Tests for the Core-Tail categorical link and the CT-AGCI site update.

Every claim the construction makes is checked against an independent
reference: the coefficient against its closed-form derivation, the link
against a projected-gradient solve of the convex program it is defined by, and
the score and curvature against autograd and a dense Fisher contraction.
"""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi import (
    A_STAR,
    TAGILastLayerClassifier,
    compute_core_tail_site_innovation,
    compute_logit_site_innovation,
    core_tail_jacobian,
    core_tail_log_probabilities,
    core_tail_predictive_probs,
    core_tail_probabilities,
    core_tail_score_and_curvature,
    core_tail_stationarity_residual,
)

EXACT = 64


def _logits(rows=6, classes=7, scale=2.0, seed=3, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, classes, generator=generator, dtype=dtype) * scale


def _regularizer(probabilities, a_star):
    """Return ``Omega*(p)``, the objective's convex regularizer."""

    return (
        probabilities * probabilities.clamp_min(1e-300).log()
        - a_star * probabilities.square() * (1.0 - probabilities).square()
    ).sum(dim=-1)


def _objective(logits, probabilities, a_star):
    return (logits * probabilities).sum(dim=-1) - _regularizer(probabilities, a_star)


def test_coefficient_matches_its_closed_form_derivation():
    # Gumbel utility differences are standard logistic with variance pi^2 / 3,
    # so the variance-matched Gaussian link is Phi(sqrt(3) d / pi) and its
    # slope at a tie is sqrt(3) / (pi sqrt(2 pi)).
    probit_slope = math.sqrt(3.0) / (math.pi * math.sqrt(2.0 * math.pi))
    assert 1.0 / (4.0 + 2.0 * A_STAR) == pytest.approx(probit_slope, rel=1e-15)
    assert A_STAR == pytest.approx(0.2732603854486113, rel=1e-15)


def test_binary_slope_at_a_tie_equals_the_variance_matched_probit_slope():
    difference = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    logits = torch.stack([difference, torch.zeros_like(difference)], dim=-1)
    probabilities = core_tail_probabilities(logits, num_iterations=EXACT)
    slope = torch.autograd.grad(probabilities[0, 0], difference)[0].item()
    assert slope == pytest.approx(
        math.sqrt(3.0) / (math.pi * math.sqrt(2.0 * math.pi)), rel=1e-12
    )


def test_probabilities_are_a_valid_normalized_simplex_point():
    probabilities = core_tail_probabilities(_logits(scale=6.0), num_iterations=EXACT)
    assert torch.all(probabilities > 0.0)
    assert torch.allclose(
        probabilities.sum(dim=-1), torch.ones(probabilities.shape[0], dtype=torch.float64)
    )


@pytest.mark.parametrize("classes", [2, 10, 100])
@pytest.mark.parametrize("scale", [0.1, 1.0, 5.0, 20.0])
def test_picard_iteration_solves_the_stationarity_fixed_point(classes, scale):
    logits = _logits(classes=classes, scale=scale)
    probabilities = core_tail_probabilities(logits, num_iterations=EXACT)
    residual = core_tail_stationarity_residual(logits, probabilities)
    assert float(residual.max()) < 1e-12


def test_six_iterations_suffice_at_single_precision():
    logits = _logits(rows=256, classes=100, scale=3.0).float()
    reference = core_tail_probabilities(logits.double(), num_iterations=EXACT).float()
    default = core_tail_probabilities(logits)
    assert float((default - reference).abs().max()) < 1e-6


def test_link_maximizes_the_convex_program_it_is_defined_by():
    """Projected-gradient ascent from a different start reaches the same point."""

    logits = _logits(rows=4, classes=5, scale=1.5)
    solution = core_tail_probabilities(logits, num_iterations=EXACT)

    # Exponentiated-gradient ascent stays strictly inside the simplex.
    iterate = torch.full_like(logits, 1.0 / logits.shape[-1])
    for _ in range(4000):
        variable = iterate.clone().requires_grad_(True)
        gradient = torch.autograd.grad(
            _objective(logits, variable, A_STAR).sum(), variable
        )[0]
        iterate = torch.softmax(iterate.clamp_min(1e-300).log() + 0.05 * gradient, dim=-1)

    assert float((iterate - solution).abs().max()) < 1e-6
    assert torch.all(
        _objective(logits, solution, A_STAR) >= _objective(logits, iterate, A_STAR) - 1e-12
    )


def test_regularizer_is_strictly_convex_with_the_stated_minimum_curvature():
    grid = torch.linspace(1e-6, 1.0, 200_001, dtype=torch.float64)
    second = 1.0 / grid - A_STAR * (2.0 - 12.0 * grid + 12.0 * grid.square())
    assert float(second.min()) == pytest.approx(1.0 - 2.0 * A_STAR, abs=1e-4)
    assert float(second.min()) > 0.0


def test_tail_recovers_the_softmax_log_probability_including_its_constant():
    for margin in (10.0, 20.0, 40.0):
        logits = torch.tensor([[margin, 0.0, 0.0, 0.0]], dtype=torch.float64)
        log_probabilities = core_tail_log_probabilities(logits, num_iterations=EXACT)
        assert -float(log_probabilities[0, 1]) == pytest.approx(margin, abs=1e-3)


def test_maximum_logit_correction_is_small():
    grid = torch.linspace(0.0, 1.0, 100_001, dtype=torch.float64)
    correction = 2.0 * A_STAR * grid * (1.0 - grid) * (1.0 - 2.0 * grid)
    assert float(correction.abs().max()) == pytest.approx(0.05262, abs=1e-4)
    assert float(correction.max() - correction.min()) == pytest.approx(0.10524, abs=1e-4)


def test_jacobian_matches_autograd_and_is_symmetric():
    logits = _logits(rows=3, classes=6, scale=1.5).requires_grad_(True)
    probabilities = core_tail_probabilities(logits, num_iterations=EXACT)
    columns = [
        torch.autograd.grad(probabilities[:, k].sum(), logits, retain_graph=True)[0]
        for k in range(logits.shape[-1])
    ]
    reference = torch.stack(columns, dim=1)
    jacobian = core_tail_jacobian(probabilities.detach())
    assert torch.allclose(jacobian, reference, atol=1e-12)
    assert torch.allclose(jacobian, jacobian.transpose(-1, -2), atol=1e-14)
    assert torch.allclose(
        jacobian.sum(dim=-1), torch.zeros_like(jacobian[..., 0]), atol=1e-12
    )


def test_score_matches_the_gradient_of_the_observed_log_probability():
    logits = _logits(rows=4, classes=7, scale=1.5).requires_grad_(True)
    labels = torch.tensor([0, 3, 6, 2])
    probabilities = core_tail_probabilities(logits, num_iterations=EXACT)
    rows = torch.arange(labels.shape[0])
    reference = torch.autograd.grad(
        probabilities[rows, labels].clamp_min(1e-300).log().sum(), logits
    )[0]
    score, _ = core_tail_score_and_curvature(labels, probabilities.detach())
    assert torch.allclose(score, reference, atol=1e-12)


def test_curvature_matches_the_dense_fisher_diagonal_and_is_nonnegative():
    logits = _logits(rows=8, classes=9, scale=2.5)
    labels = torch.arange(8) % 9
    probabilities = core_tail_probabilities(logits, num_iterations=EXACT)
    jacobian = core_tail_jacobian(probabilities)
    dense = torch.einsum("bjk,bj,bjl->bkl", jacobian, 1.0 / probabilities, jacobian)
    _, curvature = core_tail_score_and_curvature(labels, probabilities)
    assert torch.allclose(
        curvature, dense.diagonal(dim1=-2, dim2=-1), atol=1e-12
    )
    assert torch.all(curvature >= 0.0)


def test_score_expectation_vanishes_under_the_model():
    """The Fisher identity E_p[g] = 0 holds for the exact link."""

    logits = _logits(rows=1, classes=6, scale=1.5)
    probabilities = core_tail_probabilities(logits, num_iterations=EXACT)
    expectation = torch.zeros_like(logits)
    for label in range(logits.shape[-1]):
        score, _ = core_tail_score_and_curvature(
            torch.tensor([label]), probabilities
        )
        expectation += probabilities[:, label : label + 1] * score
    assert float(expectation.abs().max()) < 1e-12


def test_zero_coefficient_reduces_to_softmax_and_the_gumbel_site():
    logits = _logits(rows=5, classes=8, scale=2.0)
    labels = torch.tensor([0, 1, 7, 3, 4])
    probabilities = core_tail_probabilities(logits, a_star=0.0, num_iterations=1)
    assert torch.allclose(probabilities, torch.softmax(logits, dim=-1), atol=1e-15)

    score, curvature = core_tail_score_and_curvature(
        labels, probabilities, a_star=0.0
    )
    one_hot = torch.zeros_like(probabilities)
    one_hot[torch.arange(labels.shape[0]), labels] = 1.0
    assert torch.allclose(score, one_hot - probabilities, atol=1e-14)
    assert torch.allclose(
        curvature, probabilities * (1.0 - probabilities), atol=1e-14
    )


def test_zero_coefficient_site_reproduces_the_logit_site_innovation():
    generator = torch.Generator().manual_seed(17)
    mean = torch.randn(6, 9, generator=generator, dtype=torch.float64) * 1.5
    variance = torch.rand(6, 9, generator=generator, dtype=torch.float64) + 0.05
    labels = torch.tensor([0, 8, 3, 3, 5, 1])

    expected = compute_logit_site_innovation(labels, mean, variance)
    actual = compute_core_tail_site_innovation(labels, mean, variance, a_star=0.0)
    assert torch.allclose(actual[0], expected[0], atol=1e-14)
    assert torch.allclose(actual[1], expected[1], atol=1e-14)


def test_site_innovation_matches_the_explicit_gaussian_site():
    generator = torch.Generator().manual_seed(23)
    mean = torch.randn(5, 6, generator=generator, dtype=torch.float64) * 2.0
    variance = torch.rand(5, 6, generator=generator, dtype=torch.float64) + 0.1
    labels = torch.tensor([2, 4, 0, 5, 1])

    probabilities = core_tail_probabilities(mean, num_iterations=EXACT)
    score, curvature = core_tail_score_and_curvature(labels, probabilities)
    posterior_variance = 1.0 / (1.0 / variance + curvature)
    posterior_mean = mean + posterior_variance * score

    delta_mean, delta_variance = compute_core_tail_site_innovation(
        labels, mean, variance
    )
    assert torch.allclose(delta_mean, (posterior_mean - mean) / variance, atol=1e-13)
    assert torch.allclose(
        delta_variance, (posterior_variance - variance) / variance.square(), atol=1e-13
    )
    assert torch.all(posterior_variance > 0.0)
    assert torch.all(posterior_variance <= variance)


def test_site_moves_the_observed_class_mean_upward():
    generator = torch.Generator().manual_seed(29)
    mean = torch.randn(64, 10, generator=generator, dtype=torch.float64)
    variance = torch.full_like(mean, 0.3)
    labels = torch.randint(0, 10, (64,), generator=generator)
    delta_mean, _ = compute_core_tail_site_innovation(labels, mean, variance)
    observed = delta_mean[torch.arange(64), labels]
    assert torch.all(observed > 0.0)


def test_site_is_translation_invariant_and_permutation_equivariant():
    generator = torch.Generator().manual_seed(31)
    mean = torch.randn(4, 6, generator=generator, dtype=torch.float64) * 2.0
    variance = torch.full_like(mean, 0.25)
    labels = torch.tensor([1, 5, 0, 3])

    shifted = mean + torch.tensor([[3.0], [-2.0], [0.5], [7.0]], dtype=torch.float64)
    base = compute_core_tail_site_innovation(labels, mean, variance)
    assert torch.allclose(
        compute_core_tail_site_innovation(labels, shifted, variance)[0],
        base[0],
        atol=1e-12,
    )

    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(6)
    permuted = compute_core_tail_site_innovation(
        inverse[labels], mean[:, permutation], variance[:, permutation]
    )
    assert torch.allclose(permuted[0], base[0][:, permutation], atol=1e-14)
    assert torch.allclose(permuted[1], base[1][:, permutation], atol=1e-14)


def test_beta_rescales_the_link_and_the_site():
    generator = torch.Generator().manual_seed(37)
    mean = torch.randn(4, 5, generator=generator, dtype=torch.float64) * 2.0
    variance = torch.full_like(mean, 0.4)
    labels = torch.tensor([0, 4, 2, 1])

    assert torch.allclose(
        core_tail_predictive_probs(mean, beta=2.0),
        core_tail_probabilities(mean / 2.0),
        atol=1e-14,
    )
    probabilities = core_tail_probabilities(mean / 2.0, num_iterations=EXACT)
    score, curvature = core_tail_score_and_curvature(labels, probabilities)
    posterior_variance = 1.0 / (1.0 / variance + curvature / 4.0)
    delta_mean, _ = compute_core_tail_site_innovation(
        labels, mean, variance, beta=2.0
    )
    assert torch.allclose(
        delta_mean, posterior_variance * (score / 2.0) / variance, atol=1e-13
    )


def test_extreme_logits_stay_finite_at_single_precision():
    logits = torch.tensor(
        [[120.0, 0.0, -60.0, -200.0], [-30.0, -30.0, -30.0, -30.0]],
        dtype=torch.float32,
    )
    variance = torch.full_like(logits, 0.5)
    labels = torch.tensor([3, 0])
    probabilities = core_tail_predictive_probs(logits)
    assert torch.isfinite(probabilities).all()
    assert float(probabilities.min()) >= 0.0
    delta_mean, delta_variance = compute_core_tail_site_innovation(
        labels, logits, variance
    )
    assert torch.isfinite(delta_mean).all()
    assert torch.isfinite(delta_variance).all()
    assert torch.all(delta_variance <= 0.0)


def test_zero_variance_outputs_receive_no_innovation():
    mean = torch.zeros(2, 4, dtype=torch.float64)
    variance = torch.zeros_like(mean)
    labels = torch.tensor([0, 1])
    delta_mean, delta_variance = compute_core_tail_site_innovation(
        labels, mean, variance
    )
    assert torch.all(delta_mean == 0.0)
    assert torch.all(delta_variance == 0.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"a_star": 0.5}, "a_star"),
        ({"a_star": -0.1}, "a_star"),
        ({"num_iterations": 0}, "num_iterations"),
    ],
)
def test_invalid_link_arguments_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        core_tail_probabilities(torch.zeros(2, 3), **kwargs)


def test_invalid_site_arguments_are_rejected():
    mean = torch.zeros(2, 3)
    variance = torch.ones(2, 3)
    labels = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="beta"):
        compute_core_tail_site_innovation(labels, mean, variance, beta=0.0)
    with pytest.raises(ValueError, match="matching diagonal"):
        compute_core_tail_site_innovation(labels, mean, torch.ones(2, 4))
    with pytest.raises(ValueError, match="two-dimensional"):
        compute_core_tail_site_innovation(labels, torch.zeros(3), torch.ones(3))
    with pytest.raises(ValueError, match="at least two classes"):
        compute_core_tail_site_innovation(labels, torch.zeros(2, 1), torch.ones(2, 1))
    with pytest.raises(ValueError, match="invalid class index"):
        compute_core_tail_site_innovation(torch.tensor([0, 9]), mean, variance)
    with pytest.raises(ValueError, match="leading shape"):
        compute_core_tail_site_innovation(torch.tensor([0]), mean, variance)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_classifier_head_trains_and_round_trips(tmp_path):
    torch.manual_seed(5)
    features = torch.randn(512, 12, device="cuda")
    weights = torch.randn(12, 4, device="cuda")
    labels = (features @ weights).argmax(dim=-1)

    classifier = TAGILastLayerClassifier(
        12, 4, head="ct_agci", device="cuda", gain_w=0.3, gain_b=0.3
    )
    history = classifier.fit(
        features, labels, epochs=6, batch_size=64, validation=(features, labels)
    )
    accuracies = history.values("val_accuracy")
    assert accuracies[-1] > accuracies[0]
    assert accuracies[-1] > 0.7

    prediction = classifier.predict(features)
    assert torch.allclose(
        prediction.probabilities.sum(dim=-1),
        torch.ones(512, device="cuda"),
        atol=1e-5,
    )
    assert prediction.diagnostics is not None
    assert "utility_mean" in prediction.diagnostics

    path = classifier.save(tmp_path / "ct_agci.pt")
    restored, _ = TAGILastLayerClassifier.load(path, device="cuda")
    assert restored.head == "ct_agci"
    assert restored.core_tail_a_star == pytest.approx(A_STAR)
    assert torch.allclose(
        restored.predict(features).probabilities, prediction.probabilities, atol=1e-6
    )


def test_classifier_rejects_sigma_v_and_invalid_coefficients():
    with pytest.raises(ValueError, match="sigma_v"):
        TAGILastLayerClassifier(4, 3, head="ct_agci", device="cpu", sigma_v=1.0)
    with pytest.raises(ValueError, match="core_tail_a_star"):
        TAGILastLayerClassifier(4, 3, head="ct_agci", device="cpu", core_tail_a_star=0.9)
    with pytest.raises(ValueError, match="core_tail_beta"):
        TAGILastLayerClassifier(4, 3, head="ct_agci", device="cpu", core_tail_beta=-1.0)
    with pytest.raises(ValueError, match="core_tail_iterations"):
        TAGILastLayerClassifier(
            4, 3, head="ct_agci", device="cpu", core_tail_iterations=0
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_zero_coefficient_head_tracks_the_logit_site_head():
    torch.manual_seed(9)
    features = torch.randn(256, 8, device="cuda")
    labels = torch.randint(0, 5, (256,), device="cuda")

    common = dict(device="cuda", gain_w=0.4, gain_b=0.4)
    reference = TAGILastLayerClassifier(8, 5, head="logit_site", **common)
    candidate = TAGILastLayerClassifier(
        8, 5, head="ct_agci", core_tail_a_star=0.0, **common
    )
    for name in ("mw", "Sw", "mb", "Sb"):
        getattr(candidate.linear, name).copy_(getattr(reference.linear, name))

    for start in range(0, 256, 64):
        batch = slice(start, start + 64)
        reference.train_step(features[batch], labels[batch])
        candidate.train_step(features[batch], labels[batch])

    assert torch.allclose(
        candidate.predict(features).probabilities,
        reference.predict(features).probabilities,
        atol=1e-6,
    )
