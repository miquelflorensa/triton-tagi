"""End-to-end tests for the CDF-TAGI-V/Remax last-layer head."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from triton_tagi.cdf_variance import cdf_variance_activation
from triton_tagi.classification import TAGILastLayerClassifier
from triton_tagi.layers import EvenProbit, Linear, Remax
from triton_tagi.remax_scale import (
    DEFAULT_PRIOR_VARIANCE,
    remax_scale_negative_log_likelihood,
)

# The head is buildable on the CPU -- construction only allocates the prior --
# but every forward pass runs Triton kernels, so anything that calls predict,
# train_step's update channel, or fit is marked cuda.


def _head(*, device: str = "cpu", **overrides) -> TAGILastLayerClassifier:
    settings = {"head": "cdf_remax", "device": device}
    settings.update(overrides)
    return TAGILastLayerClassifier(6, 4, **settings)


def _separable_problem(samples: int = 512, features: int = 8, classes: int = 4, seed: int = 0):
    """Return well-separated Gaussian clusters and their labels."""

    generator = torch.Generator().manual_seed(seed)
    centers = 2.0 * torch.randn(classes, features, generator=generator)
    labels = torch.randint(0, classes, (samples,), generator=generator)
    features_out = centers[labels] + 0.3 * torch.randn(samples, features, generator=generator)
    return features_out, labels


def test_architecture_is_an_interleaved_linear_and_probit_pair():
    classifier = _head()

    assert [type(layer) for layer in classifier.net.layers] == [Linear, EvenProbit]
    assert classifier.linear.mw.shape == (6, 8)
    # No Remax layer: the probabilities are analytic, not a forward activation.
    assert not any(isinstance(layer, Remax) for layer in classifier.net.layers)
    assert classifier.cdf_head is classifier.net.layers[-1]
    assert classifier.cdf_head.epsilon == classifier.cdf_epsilon
    assert classifier.cdf_head.kappa == classifier.cdf_kappa
    assert classifier.cdf_head.half_width == 4
    # An uncalibrated head is the degenerate scale s = e^0 = 1.
    assert classifier.cdf_scale_mean == 0.0
    assert classifier.cdf_scale_variance == 0.0


def test_variance_head_starts_at_the_requested_aleatoric_variance():
    # h(0) = epsilon + kappa/2 exactly, so the midpoint pins the inverse to
    # machine precision through the layer's float32 bias.
    midpoint = _head(cdf_epsilon=0.05, cdf_kappa=2.0, cdf_aleatoric_init=0.05 + 1.0)
    nu = midpoint.linear.mb[:, 1::2].double()
    aleatoric = cdf_variance_activation(nu, epsilon=midpoint.cdf_epsilon, kappa=midpoint.cdf_kappa)
    assert float((aleatoric - midpoint.cdf_aleatoric_init).abs().max()) < 1e-12

    # Away from the midpoint the float32 bias bounds the recovery; the inverse
    # itself is exact, so what is left is only the storage round-off.
    classifier = _head()
    nu = classifier.linear.mb[:, 1::2].double()
    aleatoric = cdf_variance_activation(
        nu, epsilon=classifier.cdf_epsilon, kappa=classifier.cdf_kappa
    )
    assert float((aleatoric - classifier.cdf_aleatoric_init).abs().max()) < 1e-6
    expected = torch.special.ndtri(
        torch.tensor(
            (classifier.cdf_aleatoric_init - classifier.cdf_epsilon) / classifier.cdf_kappa,
            dtype=torch.float64,
        )
    )
    assert torch.allclose(nu, expected.expand_as(nu).to(nu.dtype), atol=1e-7)

    # The weight means are zero so the initial noise holds at every input, and
    # the odd slots carry the shared TAGI-V prior variances.
    assert float(classifier.linear.mw[:, 1::2].abs().max()) == 0.0
    assert float(classifier.linear.mw[:, 0::2].abs().max()) > 0.0
    assert torch.all(classifier.linear.Sw[:, 1::2] == classifier.v2bar_weight_var)
    assert torch.all(classifier.linear.Sb[:, 1::2] == classifier.v2bar_bias_var)


def test_zero_mean_init_leaves_the_variance_head_alone():
    classifier = _head(mean_init="zero")

    assert float(classifier.linear.mw.abs().max()) == 0.0
    nu = classifier.linear.mb[:, 1::2].double()
    aleatoric = cdf_variance_activation(
        nu, epsilon=classifier.cdf_epsilon, kappa=classifier.cdf_kappa
    )
    assert float((aleatoric - classifier.cdf_aleatoric_init).abs().max()) < 1e-6
    assert float(classifier.linear.mb[:, 0::2].abs().max()) == 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cdf_epsilon": 0.0}, "cdf_epsilon must be finite and positive"),
        ({"cdf_epsilon": float("nan")}, "cdf_epsilon must be finite and positive"),
        ({"cdf_kappa": -1.0}, "cdf_kappa must be finite and positive"),
        ({"cdf_laplace_order": 3}, "quadrature order must be at least four"),
        ({"cdf_hermite_order": 0}, "quadrature order must be at least four"),
        ({"cdf_scale_order": 3}, "quadrature order must be at least four"),
        ({"cdf_train_hermite_order": 1}, "quadrature order must be at least four"),
        ({"cdf_decomposition_samples": 1}, "cdf_decomposition_samples must be None"),
        ({"cdf_aleatoric_init": 0.01}, "must lie strictly inside the attainable range"),
        ({"cdf_aleatoric_init": 0.02}, "must lie strictly inside the attainable range"),
        ({"cdf_aleatoric_init": 1.52}, "must lie strictly inside the attainable range"),
        ({"cdf_aleatoric_init": 9.0}, "must lie strictly inside the attainable range"),
        ({"sigma_v": 0.5}, "does not accept sigma_v"),
    ],
)
def test_constructor_rejects_invalid_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        _head(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"cdf_epsilon": 0.05},
        {"cdf_kappa": 3.0},
        {"cdf_aleatoric_init": 0.5},
        {"cdf_laplace_order": 40},
        {"cdf_hermite_order": 8},
        {"cdf_scale_order": 8},
        {"cdf_train_hermite_order": 16},
        {"cdf_decomposition_samples": 128},
    ],
)
def test_cdf_keywords_are_rejected_for_other_heads(overrides):
    name = next(iter(overrides))
    with pytest.raises(ValueError, match=f"{name} is only valid for head='cdf_remax'"):
        TAGILastLayerClassifier(6, 4, head="remax_lognormal", device="cpu", **overrides)

    # An untouched keyword is not a supplied one, so every other head still
    # constructs, which is what load() relies on for old checkpoints.
    TAGILastLayerClassifier(6, 4, head="remax_lognormal", device="cpu")


def test_train_step_rejects_logit_targets_and_fixed_observation_noise():
    classifier = _head()
    features = torch.randn(3, 6)
    labels = torch.zeros(3, dtype=torch.long)

    with pytest.raises(ValueError, match="trains on class labels, not on logit targets"):
        classifier.train_step(features, labels, targets=torch.randn(3, 4))
    with pytest.raises(ValueError, match="requires class labels"):
        classifier.train_step(features, None)
    with pytest.raises(ValueError, match="does not accept sigma_v"):
        classifier.train_step(features, labels, 0.5)


def test_checkpoint_round_trip_preserves_the_cdf_fields(tmp_path):
    classifier = _head(
        cdf_epsilon=0.05,
        cdf_kappa=2.0,
        cdf_aleatoric_init=0.3,
        cdf_laplace_order=80,
        cdf_hermite_order=16,
        cdf_scale_order=8,
        cdf_train_hermite_order=32,
        cdf_decomposition_samples=64,
    )
    classifier.cdf_scale_mean = -0.4
    classifier.cdf_scale_variance = 0.09
    config = classifier.config()
    assert config["cdf_scale_mean"] == -0.4
    assert config["cdf_scale_variance"] == 0.09

    destination = classifier.save(tmp_path / "cdf_remax.pt")
    restored, metadata = TAGILastLayerClassifier.load(destination, device="cpu")

    assert metadata == {}
    assert restored.config() == config
    for name in ("mw", "Sw", "mb", "Sb"):
        assert torch.equal(getattr(restored.linear, name), getattr(classifier.linear, name))
    # The activation constants must travel with the head, not just the config.
    assert restored.cdf_head.epsilon == 0.05
    assert restored.cdf_head.kappa == 2.0


@pytest.mark.cuda
def test_predict_returns_normalized_class_probabilities():
    classifier = _head(device="cuda")
    features = torch.randn(7, 6)

    prediction = classifier.predict(features)

    assert prediction.probabilities.shape == (7, 4)
    assert float(prediction.probabilities.min()) >= 0.0
    assert torch.allclose(
        prediction.probabilities.sum(dim=-1),
        torch.ones(7, dtype=prediction.probabilities.dtype, device=prediction.probabilities.device),
        atol=1e-9,
    )
    # The raw interleaved network output is reported as-is, as on every head.
    assert prediction.output_mean.shape == (7, 8)
    assert prediction.output_variance.shape == (7, 8)
    assert prediction.diagnostics is not None
    assert prediction.diagnostics["cov_scale"].shape == (7, 4)

    with pytest.raises(ValueError, match="does not accept sigma_v"):
        classifier.predict(features, sigma_v=0.5)


@pytest.mark.cuda
def test_uncertainty_split_is_opt_in_and_adds_up_to_the_indicator_variance():
    features = torch.randn(6, 6)

    # Without a sample budget there is deliberately no cheap surrogate.
    cheap = _head(device="cuda").predict(features)
    assert cheap.epistemic_variance is None
    assert cheap.aleatoric_variance is None

    classifier = _head(device="cuda", cdf_decomposition_samples=256)
    prediction = classifier.predict(features)
    epistemic = prediction.epistemic_variance
    aleatoric = prediction.aleatoric_variance

    assert epistemic is not None and aleatoric is not None
    assert epistemic.shape == (6, 4) and aleatoric.shape == (6, 4)
    assert float(epistemic.min()) >= 0.0
    assert float(aleatoric.min()) >= 0.0

    # Sigma_epi + Sigma_ale = diag(p) - p p^T, whose diagonal is p (1 - p).
    probabilities = prediction.probabilities
    total = epistemic + aleatoric
    assert float((total - probabilities * (1.0 - probabilities)).abs().max()) < 1.5e-3


@pytest.mark.cuda
def test_calibration_state_is_consulted_by_predict():
    classifier = _head(device="cuda")
    features = torch.randn(5, 6)

    uncalibrated = classifier.predict(features).probabilities
    classifier.cdf_scale_mean = 0.7
    shifted = classifier.predict(features).probabilities
    assert float((shifted - uncalibrated).abs().max()) > 1e-3

    classifier.cdf_scale_variance = 0.25
    spread = classifier.predict(features).probabilities
    assert float((spread - shifted).abs().max()) > 1e-6

    # Back to lambda = q = 0, the uncalibrated s = 1, reproduces the head.
    classifier.cdf_scale_mean = 0.0
    classifier.cdf_scale_variance = 0.0
    assert torch.allclose(classifier.predict(features).probabilities, uncalibrated, atol=1e-12)


@pytest.mark.cuda
def test_training_channel_separates_gaussian_clusters():
    features, labels = _separable_problem()
    classifier = TAGILastLayerClassifier(
        features.shape[1], 4, head="cdf_remax", device="cuda", mean_init="zero"
    )

    history = classifier.fit(
        features[:384],
        labels[:384],
        epochs=3,
        batch_size=64,
        validation=(features[384:], labels[384:]),
    )
    accuracy = history.values("val_accuracy")
    nll = history.values("val_nll")

    assert len(accuracy) == 4
    # The zero-mean prior starts at chance and the update must move it.
    assert accuracy[0] < 0.5
    assert accuracy[-1] > 0.9
    assert nll[-1] < nll[0]


# ──────────────────────────────────────────────────────────────────────────────
#  Auxiliary calibration of the shared deviation scale
# ──────────────────────────────────────────────────────────────────────────────


def test_forward_summaries_are_refused_for_other_heads() -> None:
    classifier = TAGILastLayerClassifier(6, 4, head="remax_lognormal", device="cpu")
    with pytest.raises(ValueError, match="CDF-Remax forward summaries"):
        classifier.cdf_remax_moments(torch.zeros(3, 6))


def test_calibration_rejects_an_unknown_method() -> None:
    classifier = _head()
    with pytest.raises(ValueError, match="Unknown method"):
        classifier.calibrate_remax_log_scale(
            torch.zeros(4, 6), torch.zeros(4, dtype=torch.long), method="nonsense"
        )


def test_tracking_options_are_refused_for_the_batch_methods() -> None:
    classifier = _head()
    with pytest.raises(ValueError, match="only to the sequential"):
        classifier.calibrate_remax_log_scale(
            torch.zeros(4, 6),
            torch.zeros(4, dtype=torch.long),
            method="grid",
            initial_mean=0.3,
        )


@pytest.mark.cuda
def test_forward_summaries_have_the_expected_shapes_and_dtype() -> None:
    torch.manual_seed(31)
    classifier = _head(device="cuda")
    features = torch.randn(9, 6, device="cuda")
    mu_z, var_z, nu, r = classifier.cdf_remax_moments(features, batch_size=4)
    for tensor in (mu_z, var_z, nu, r):
        assert tensor.shape == (9, classifier.num_classes)
        assert tensor.dtype == torch.float64
        assert bool(torch.isfinite(tensor).all())
    assert bool((var_z >= 0.0).all()) and bool((r >= 0.0).all())


@pytest.mark.cuda
def test_calibration_is_stored_and_changes_the_predictions() -> None:
    torch.manual_seed(32)
    classifier = _head(device="cuda")
    features = torch.randn(48, 6, device="cuda")
    labels = torch.randint(0, classifier.num_classes, (48,), device="cuda")

    before = classifier.predict(features).probabilities.clone()
    assert classifier.cdf_scale_posterior is None

    posterior = classifier.calibrate_remax_log_scale(features, labels)
    assert classifier.cdf_scale_posterior is posterior
    assert classifier.cdf_scale_mean == pytest.approx(posterior.mean)
    assert classifier.cdf_scale_variance == pytest.approx(posterior.variance)
    assert posterior.variance >= 0.0

    after = classifier.predict(features).probabilities
    assert float((after - before).abs().max()) > 0.0
    torch.testing.assert_close(
        after.sum(dim=1),
        torch.ones(48, dtype=after.dtype, device=after.device),
        atol=1e-6,
        rtol=0.0,
    )


@pytest.mark.cuda
def test_calibration_can_be_fitted_without_being_applied() -> None:
    torch.manual_seed(33)
    classifier = _head(device="cuda")
    features = torch.randn(32, 6, device="cuda")
    labels = torch.randint(0, classifier.num_classes, (32,), device="cuda")

    posterior = classifier.calibrate_remax_log_scale(features, labels, apply=False)
    assert classifier.cdf_scale_posterior is None
    assert classifier.cdf_scale_mean == 0.0
    assert classifier.cdf_scale_variance == 0.0
    assert posterior.variance >= 0.0


@pytest.mark.cuda
def test_the_batch_fit_does_not_increase_the_calibration_objective() -> None:
    """A near-flat prior makes the MAP the negative-log-likelihood minimiser.

    The objective is ``-sum_n log E[A_{c_n}]``, so this must hold up to the
    grid resolution; it is the one guarantee the batch reference offers that
    the ordering-sensitive sequential recursion does not.
    """

    torch.manual_seed(34)
    classifier = _head(device="cuda")
    features = torch.randn(64, 6, device="cuda")
    labels = torch.randint(0, classifier.num_classes, (64,), device="cuda")
    summaries = classifier.cdf_remax_moments(features)

    posterior = classifier.calibrate_remax_log_scale(
        features, labels, prior_variance=100.0, apply=False
    )
    uncalibrated = dataclasses.replace(posterior, mean=0.0, variance=0.0)
    fitted = remax_scale_negative_log_likelihood(
        *summaries, labels.long(), posterior.deterministic()
    )
    baseline = remax_scale_negative_log_likelihood(*summaries, labels.long(), uncalibrated)
    assert float(fitted) <= float(baseline) + 1e-9


@pytest.mark.cuda
@pytest.mark.parametrize("method", ["event", "full", "tilt"])
def test_the_sequential_recursion_runs_for_every_variant(method: str) -> None:
    torch.manual_seed(35)
    classifier = _head(device="cuda")
    features = torch.randn(24, 6, device="cuda")
    labels = torch.randint(0, classifier.num_classes, (24,), device="cuda")

    posterior = classifier.calibrate_remax_log_scale(features, labels, method=method)
    assert math.isfinite(posterior.mean)
    assert 0.0 <= posterior.variance <= DEFAULT_PRIOR_VARIANCE + 1e-12
    # The sequential fits are tagged apart from the batch ones.
    assert posterior.method == f"adf-{method}"


@pytest.mark.cuda
def test_retaining_the_mean_across_epochs_differs_from_a_fresh_prior() -> None:
    """The note's epoch-wise heuristic: keep ``lambda``, reset ``q``."""

    torch.manual_seed(36)
    classifier = _head(device="cuda")
    features = torch.randn(24, 6, device="cuda")
    labels = torch.randint(0, classifier.num_classes, (24,), device="cuda")

    fresh = classifier.calibrate_remax_log_scale(features, labels, method="event", apply=False)
    retained = classifier.calibrate_remax_log_scale(
        features, labels, method="event", initial_mean=fresh.mean, apply=False
    )
    assert retained.mean != pytest.approx(fresh.mean, abs=1e-12)
    assert retained.variance == pytest.approx(fresh.variance, rel=0.5)
