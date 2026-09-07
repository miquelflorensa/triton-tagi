"""End-to-end tests for the logit TAGI-V last-layer head."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from triton_tagi.classification import TAGILastLayerClassifier
from triton_tagi.logit_tagiv import (
    logit_feature_energy,
    logit_tagiv_predictive_nll,
    logit_tagiv_predictive_probs,
    logit_tagiv_uncertainty,
    prepare_logit_targets,
)

pytestmark = pytest.mark.cuda


def _heteroscedastic_problem(
    samples: int = 4096,
    features: int = 12,
    classes: int = 4,
    noise_scale: float = 0.35,
    seed: int = 0,
):
    """Return features, planted logits, planted noise variance, and observations."""

    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(samples, features, generator=generator)
    weights = torch.randn(features, classes, generator=generator) / features**0.5
    logits = inputs @ weights
    logits = logits - logits.mean(dim=-1, keepdim=True)
    log_variance = inputs @ (noise_scale * torch.randn(features, classes, generator=generator))
    variance = 1e-4 + (log_variance - 2.0).exp()
    observations = logits + variance.sqrt() * torch.randn(samples, classes, generator=generator)
    return inputs, logits, variance, observations


def _fit_head(inputs, observations, *, epochs: int = 80, **overrides):
    settings = {
        "head": "logit_tagiv",
        "device": "cuda",
        "gain_w": 0.1,
        "gain_b": 0.1,
        "logit_aleatoric_init": 0.2,
        "logit_variance_feature_energy": logit_feature_energy(inputs),
    }
    settings.update(overrides)
    classifier = TAGILastLayerClassifier(inputs.shape[1], observations.shape[1], **settings)
    classifier.fit(inputs, targets=observations, epochs=epochs, batch_size=256)
    return classifier


@pytest.fixture(scope="module")
def trained_head():
    """A converged head plus the problem it was fitted on, shared by tests."""

    inputs, logits, variance, observations = _heteroscedastic_problem()
    return _fit_head(inputs, observations), inputs, logits, variance, observations


def test_head_recovers_planted_logits_and_input_dependent_noise(trained_head):
    classifier, inputs, logits, variance, observations = trained_head

    logit_mean, epistemic, aleatoric = classifier.logit_moments(inputs)
    logit_mean = logit_mean.cpu()
    aleatoric = aleatoric.cpu()

    # The latent stream must track the planted logits far better than the
    # observations do, since it has to average the injected noise away.
    residual = (logit_mean - logits).pow(2).mean().sqrt()
    assert float(residual) < 0.2 * float(logits.std())
    assert float((observations - logits).pow(2).mean().sqrt()) > float(residual)

    # The variance stream must order the samples by their true noise level.
    correlation = torch.corrcoef(
        torch.stack([aleatoric.flatten().log(), variance.flatten().log()])
    )[0, 1]
    assert float(correlation) > 0.75
    ratio = float(aleatoric.mean() / variance.mean())
    assert 0.5 < ratio < 2.0
    assert bool((epistemic > 0.0).all())


def test_variance_head_identifies_a_known_homoscedastic_noise_level():
    """A constant feature isolates the variance head from the mean stream.

    With one active input the latent logit converges immediately, so every
    residual the AGVI stage sees is observation noise. Starting the head
    between the two planted variances also shows that it moves in both
    directions rather than only inflating.
    """

    initial = 0.25
    learned = {}
    for variance in (0.09, 0.64):
        generator = torch.Generator().manual_seed(5)
        inputs = torch.zeros(4096, 2)
        inputs[:, 0] = 1.0
        observations = variance**0.5 * torch.randn(4096, 3, generator=generator)
        classifier = TAGILastLayerClassifier(
            2,
            3,
            head="logit_tagiv",
            device="cuda",
            gain_w=0.1,
            gain_b=0.1,
            logit_aleatoric_init=initial,
            logit_variance_feature_energy=logit_feature_energy(inputs),
        )
        classifier.fit(inputs, targets=observations, epochs=120, batch_size=256)
        logit_mean, _, aleatoric = classifier.logit_moments(inputs[:8])
        learned[variance] = float(aleatoric.mean())
        assert float(logit_mean.abs().max()) < 0.1
        assert 0.7 < learned[variance] / variance < 1.4

    assert learned[0.09] < initial < learned[0.64]


def test_predict_returns_normalized_probabilities_and_split_moments(trained_head):
    classifier, inputs = trained_head[0], trained_head[1]

    prediction = classifier.predict(inputs[:64])
    assert prediction.probabilities.shape == (64, 4)
    assert prediction.output_mean.shape == (64, 8)
    torch.testing.assert_close(
        prediction.probabilities.sum(-1), torch.ones(64, device=prediction.probabilities.device)
    )
    assert prediction.epistemic_variance is not None
    assert prediction.aleatoric_variance is not None
    assert bool((prediction.aleatoric_variance > 0.0).all())
    assert prediction.diagnostics is not None
    assert "logit_mean" in prediction.diagnostics
    assert "aleatoric_moment_variance" in prediction.diagnostics

    logit_mean, epistemic, aleatoric = classifier.logit_moments(inputs[:64])
    torch.testing.assert_close(
        prediction.probabilities,
        logit_tagiv_predictive_probs(
            logit_mean,
            epistemic,
            aleatoric,
            temperature=classifier.logit_temperature,
            alpha=classifier.logit_alpha,
            base_samples=classifier._resolve_logit_base_samples(logit_mean.dtype),
        ),
    )


def test_calibration_lowers_validation_nll_and_recovers_the_target_scale():
    generator = torch.Generator().manual_seed(3)
    inputs = torch.randn(6000, 12, generator=generator)
    weights = 2.0 * torch.randn(12, 4, generator=generator) / 12**0.5
    logits = inputs @ weights
    labels = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).squeeze(1)
    targets, scale = prepare_logit_targets(logits)

    classifier = TAGILastLayerClassifier(
        12,
        4,
        head="logit_tagiv",
        device="cuda",
        gain_w=0.1,
        gain_b=0.1,
        logit_aleatoric_init=0.05,
        logit_scale=scale,
        logit_variance_feature_energy=logit_feature_energy(inputs[:4000]),
    )
    classifier.fit(inputs[:4000], targets=targets[:4000], epochs=40, batch_size=200)

    moments = classifier.logit_moments(inputs[4000:])
    calibration_labels = labels[4000:].cuda()
    uncalibrated = logit_tagiv_predictive_nll(
        *moments, calibration_labels, temperature=1.0, alpha=1.0
    )
    temperature_only = classifier.calibrate(
        inputs[4000:], labels[4000:], mode="temperature", alpha=1.0
    )
    assert temperature_only.nll < uncalibrated
    assert classifier.logit_temperature == temperature_only.temperature
    assert classifier.logit_alpha == 1.0
    # The normalization divided the teacher logits by ``scale``, so the
    # temperature that undoes it sits near ``1 / scale``.
    assert 0.5 / scale < temperature_only.temperature < 2.0 / scale

    joint = classifier.calibrate(inputs[4000:], labels[4000:], mode="joint")
    assert joint.nll <= temperature_only.nll + 1e-9
    assert classifier.logit_alpha == joint.alpha
    assert classifier.logit_temperature == joint.temperature

    accuracy = (
        (classifier.predict(inputs[4000:]).probabilities.argmax(-1).cpu() == labels[4000:])
        .float()
        .mean()
    )
    teacher_accuracy = (logits[4000:].argmax(-1) == labels[4000:]).float().mean()
    assert float(accuracy) > float(teacher_accuracy) - 0.05


def test_uncertainty_decomposition_runs_on_head_moments(trained_head):
    classifier, inputs = trained_head[0], trained_head[1]
    decomposition = logit_tagiv_uncertainty(
        *classifier.logit_moments(inputs[:128]),
        epistemic_samples=32,
        aleatoric_samples=32,
    )
    assert decomposition.total_entropy.shape == (128,)
    assert bool((decomposition.aleatoric_entropy > 0.0).all())
    assert bool((decomposition.epistemic_entropy >= -1e-6).all())


def test_checkpoint_round_trip_preserves_predictions():
    inputs, _, _, observations = _heteroscedastic_problem(samples=1024)
    classifier = _fit_head(inputs, observations, epochs=10, logit_scale=2.5)
    classifier.logit_temperature, classifier.logit_alpha = 0.8, 0.3
    expected = classifier.predict(inputs[:32]).probabilities

    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "logit_tagiv.pt"
        classifier.save(destination, metadata={"epoch": 40})
        restored, metadata = TAGILastLayerClassifier.load(destination, device="cuda")
    assert metadata == {"epoch": 40}
    assert restored.head == "logit_tagiv"
    assert restored.logit_scale == 2.5
    assert restored.logit_temperature == 0.8
    assert restored.logit_alpha == 0.3
    torch.testing.assert_close(restored.predict(inputs[:32]).probabilities, expected)


def test_head_rejects_label_training_and_fixed_observation_noise():
    inputs, _, _, observations = _heteroscedastic_problem(samples=256)
    classifier = TAGILastLayerClassifier(
        12, 4, head="logit_tagiv", device="cuda", logit_variance_weight_share=0.0
    )
    labels = torch.zeros(256, dtype=torch.long)

    with pytest.raises(ValueError, match="teacher logits"):
        classifier.train_step(inputs, labels)
    with pytest.raises(ValueError, match="does not accept sigma_v"):
        classifier.train_step(inputs, targets=observations, sigma_v=0.1)
    with pytest.raises(ValueError, match="teacher logits"):
        classifier.fit(inputs, labels, epochs=1)
    with pytest.raises(ValueError, match="shape"):
        classifier.train_step(inputs, targets=observations[:, :2])
    with pytest.raises(ValueError, match="do not accept sigma_v"):
        TAGILastLayerClassifier(12, 4, head="logit_tagiv", device="cuda", sigma_v=0.5)
    with pytest.raises(ValueError, match="logit_variance_feature_energy"):
        TAGILastLayerClassifier(12, 4, head="logit_tagiv", device="cuda")


def test_label_heads_reject_logit_targets():
    inputs = torch.randn(32, 12)
    targets = torch.randn(32, 4)
    classifier = TAGILastLayerClassifier(12, 4, head="categorical_tagiv", device="cuda")
    with pytest.raises(ValueError, match="not on logit targets"):
        classifier.train_step(inputs, torch.zeros(32, dtype=torch.long), targets=targets)
    with pytest.raises(ValueError, match="has no logit TAGI-V moments"):
        classifier.logit_moments(inputs)


def test_teacher_initialization_reproduces_the_centered_target_exactly():
    """The distillation target is an affine map of the same features, so the
    teacher-initialized head must predict it with zero error before any fitting."""

    generator = torch.Generator().manual_seed(11)
    features = torch.randn(256, 12, generator=generator)
    weight = torch.randn(4, 12, generator=generator)
    bias = torch.randn(4, generator=generator)
    logits = features @ weight.T + bias
    targets, scale = prepare_logit_targets(logits)

    classifier = TAGILastLayerClassifier(
        12,
        4,
        head="logit_tagiv",
        device="cuda",
        logit_variance_feature_energy=logit_feature_energy(features),
        logit_scale=scale,
    )
    classifier.initialize_mean_from_teacher(weight, bias, scale=scale)
    logit_mean, _, _ = classifier.logit_moments(features)
    torch.testing.assert_close(logit_mean.cpu(), targets, atol=1e-4, rtol=0)

    # An assimilation pass must leave the means where they are while the
    # parameter covariance contracts.
    before_weight = classifier.linear.mw[:, 0::2].clone()
    before_variance = classifier.linear.Sw[:, 0::2].mean().item()
    classifier.fit_mean(features, targets, epochs=2, observation_variance=0.01)
    torch.testing.assert_close(classifier.linear.mw[:, 0::2], before_weight, atol=2e-3, rtol=0)
    assert classifier.linear.Sw[:, 0::2].mean().item() < before_variance


def test_teacher_initialization_rejects_a_mismatched_classifier():
    classifier = TAGILastLayerClassifier(
        12, 4, head="logit_tagiv", device="cuda", logit_variance_weight_share=0.0
    )
    with pytest.raises(ValueError, match="num_classes, input_dim"):
        classifier.initialize_mean_from_teacher(torch.randn(12, 4))
    with pytest.raises(ValueError, match="bias must have shape"):
        classifier.initialize_mean_from_teacher(torch.randn(4, 12), torch.randn(3))
