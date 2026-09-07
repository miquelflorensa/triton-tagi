"""Tests for reusable TAGI last-layer classifiers."""

from __future__ import annotations

import pytest
import torch

from triton_tagi.classification import (
    TAGILastLayerClassifier,
    normalize_class_probabilities,
)
from triton_tagi.hrc_probit import (
    hrc_log_partition_deviation,
    hrc_negative_log_likelihood,
)
from triton_tagi.hrc_softmax import obs_to_class_probs
from triton_tagi.hsm_calibration import LogGainPosterior

pytestmark = pytest.mark.cuda


def test_probability_normalization():
    values = torch.tensor([[2.0, 1.0, -1e-8]])
    normalized = normalize_class_probabilities(values)
    torch.testing.assert_close(normalized, torch.tensor([[2 / 3, 1 / 3, 0.0]]))


def test_hrc_head_trains_and_returns_probabilities_on_cuda():
    torch.manual_seed(0)
    classifier = TAGILastLayerClassifier(4, 3, head="hrc", device="cuda")
    x = torch.randn(8, 4)
    labels = torch.arange(8) % 3
    before = classifier.linear.mw.clone()
    classifier.train_step(x, labels, sigma_v=0.05)
    assert not torch.equal(before, classifier.linear.mw)
    prediction = classifier.predict(x)
    assert prediction.output_mean.shape == (8, classifier.hrc.len)
    assert prediction.probabilities.shape == (8, 3)
    assert torch.isfinite(prediction.probabilities).all()
    torch.testing.assert_close(
        prediction.probabilities.sum(1), torch.ones_like(prediction.probabilities[:, 0])
    )


def test_exact_hrc_probit_head_trains_and_returns_probabilities_on_cuda():
    torch.manual_seed(0)
    classifier = TAGILastLayerClassifier(4, 10, head="hrc_probit", device="cuda")
    x = torch.randn(16, 4)
    labels = torch.arange(16) % 10
    before = classifier.linear.mw.clone()
    classifier.train_step(x, labels)

    assert not torch.equal(before, classifier.linear.mw)
    prediction = classifier.predict(x)
    assert prediction.output_mean.shape == (16, classifier.hrc.len)
    assert prediction.probabilities.shape == (16, 10)
    assert torch.isfinite(prediction.probabilities).all()
    torch.testing.assert_close(
        prediction.probabilities.sum(1),
        torch.ones_like(prediction.probabilities[:, 0]),
    )
    with pytest.raises(ValueError, match="scale at one"):
        classifier.train_step(x, labels, sigma_v=1.0)
    with pytest.raises(ValueError, match="scale at one"):
        classifier.predict(x, sigma_v=1.0)
    with pytest.raises(ValueError, match="scale at one"):
        TAGILastLayerClassifier(4, 10, head="hrc_probit", device="cuda", sigma_v=1.0)


def test_full_tree_hrc_probit_head_needs_only_k_minus_one_nodes_on_cuda():
    torch.manual_seed(0)
    classifier = TAGILastLayerClassifier(4, 10, head="hrc_probit", hrc_tree="full", device="cuda")
    assert classifier.hrc is not None
    assert classifier.hrc.len == 9
    assert classifier.linear.mw.shape[1] == 9

    x = torch.randn(64, 4)
    labels = torch.arange(64) % 10
    before = classifier.linear.mw.clone()
    classifier.train_step(x, labels)
    assert not torch.equal(before, classifier.linear.mw)

    prediction = classifier.predict(x)
    assert prediction.probabilities.shape == (64, 10)
    torch.testing.assert_close(
        prediction.probabilities.sum(1),
        torch.ones_like(prediction.probabilities[:, 0]),
    )
    # A full tree needs no categorical normalizer, so the raw path products
    # already sum to one.
    mean, variance = classifier.hrc_node_moments(x)
    assert hrc_log_partition_deviation(mean, variance, classifier.hrc, log_tau=0.0) < 1e-8


def test_hrc_tagiv_rejects_the_full_tree():
    with pytest.raises(ValueError, match="padded tree"):
        TAGILastLayerClassifier(4, 10, head="hrc_tagiv", hrc_tree="full", device="cuda")


@pytest.mark.parametrize("head", ["hrc", "hrc_probit"])
def test_calibrating_the_latent_scale_lowers_validation_nll_on_cuda(head):
    torch.manual_seed(0)
    classifier = TAGILastLayerClassifier(
        8,
        10,
        head=head,
        hrc_tree="full",
        device="cuda",
        sigma_v=0.1 if head == "hrc" else None,
    )
    features = torch.randn(512, 8)
    weights = torch.randn(8, 10)
    labels = (features @ weights).argmax(dim=1)
    classifier.fit(
        features,
        labels,
        epochs=4,
        batch_size=64,
        sigma_v=0.1 if head == "hrc" else None,
    )

    mean, variance = classifier.hrc_node_moments(features)
    default_scale = classifier.hrc_log_tau
    before = float(
        hrc_negative_log_likelihood(
            mean, variance, labels.cuda(), classifier.hrc, log_tau=default_scale
        )
    )
    fitted = classifier.calibrate_hrc_log_tau(features, labels)
    after = float(
        hrc_negative_log_likelihood(mean, variance, labels.cuda(), classifier.hrc, log_tau=fitted)
    )
    assert classifier.hrc_log_tau == fitted
    assert after <= before + 1e-9


def test_hrc_configuration_survives_a_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    classifier = TAGILastLayerClassifier(
        4,
        10,
        head="hrc_probit",
        hrc_tree="full",
        hrc_prior_offsets=False,
        device="cuda",
    )
    x = torch.randn(16, 4)
    classifier.train_step(x, torch.arange(16) % 10)
    classifier.hrc_log_tau = 0.75
    path = classifier.save(tmp_path / "hrc.pt")

    restored, _ = TAGILastLayerClassifier.load(path, device="cuda")
    assert restored.hrc_tree == "full"
    assert restored.hrc_prior_offsets is False
    assert restored.hrc_log_tau == 0.75
    assert restored.hrc is not None and restored.hrc.len == 9
    torch.testing.assert_close(
        restored.predict(x).probabilities, classifier.predict(x).probabilities
    )


def test_hsm_calibration_reads_the_base_hrc_head_at_its_own_channel_on_cuda():
    torch.manual_seed(0)
    classifier = TAGILastLayerClassifier(
        4, 8, head="hrc", hrc_tree="full", device="cuda", sigma_v=0.3
    )
    x = torch.randn(64, 4)
    labels = torch.arange(64) % 8
    classifier.train_step(x, labels)

    assert classifier.hrc is not None
    mean, variance = classifier.hrc_node_moments(x)
    # With no fitted belief the calibrated readout is the frozen head itself,
    # at the observation noise it trained with rather than cuTAGI's alpha = 3.
    # The head carries float32 moments, so the two routes agree at that
    # precision; on float64 inputs they agree to 1e-15, which
    # tests/unit/test_hsm_base_hrc.py pins.
    torch.testing.assert_close(
        classifier.hsm_probabilities(x).double(),
        obs_to_class_probs(mean, variance, classifier.hrc, alpha=1.0 / 0.3).double(),
        atol=1e-6,
        rtol=0.0,
    )

    posterior = classifier.calibrate_hsm_log_gain(x, labels, prior_variance=4.0)
    assert posterior.sigma_v == 0.3
    probabilities = classifier.hsm_probabilities(x)
    torch.testing.assert_close(
        probabilities.sum(1), torch.ones_like(probabilities[:, 0]), atol=1e-6, rtol=0.0
    )


def test_hsm_probabilities_rejects_a_belief_from_another_channel_on_cuda():
    torch.manual_seed(0)
    classifier = TAGILastLayerClassifier(
        4, 8, head="hrc", hrc_tree="full", device="cuda", sigma_v=0.3
    )
    classifier.train_step(torch.randn(16, 4), torch.arange(16) % 8)
    assert classifier.hrc is not None
    foreign = LogGainPosterior.unit(classifier.hrc, sigma_v=1.0)
    with pytest.raises(ValueError, match="sigma_v"):
        classifier.hsm_probabilities(torch.randn(8, 4), posterior=foreign)
