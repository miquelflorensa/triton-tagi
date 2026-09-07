import pytest
import torch

from triton_tagi.full_covariance_adf import (
    FullCovarianceADFClassifier,
    contrast_covariance_logdet,
    multinomial_probit_epistemic_mutual_information,
)
from triton_tagi.multinomial_probit import multinomial_probit_adf_event


def test_one_sample_parameter_update_reproduces_adf_output_marginals():
    torch.manual_seed(1)
    classifier = FullCovarianceADFClassifier(
        2, 2, device="cpu", gain_w=0.7, gain_b=0.7
    )
    features = torch.tensor([[0.4, -1.2]])
    labels = torch.tensor([1])
    prior_mean, prior_variance = classifier.output_moments(features)
    _, expected_mean, expected_covariance = multinomial_probit_adf_event(
        prior_mean, prior_variance, labels
    )

    classifier.train_step(features, labels)
    actual_mean, actual_variance = classifier.output_moments(features)

    torch.testing.assert_close(actual_mean, expected_mean, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(
        actual_variance,
        expected_covariance.diagonal(dim1=-2, dim2=-1),
        rtol=1e-8,
        atol=1e-8,
    )


def test_full_covariance_learns_directional_contraction():
    torch.manual_seed(2)
    classifier = FullCovarianceADFClassifier(
        2, 2, device="cpu", gain_w=1.0, gain_b=1.0
    )
    observed = torch.tensor([[1.0, 0.0], [-1.0, 0.0]]).repeat(8, 1)
    labels = torch.tensor([0, 1]).repeat(8)
    classifier.fit(observed, labels, batch_size=4)

    _, seen_variance = classifier.output_moments(torch.tensor([[1.0, 0.0]]))
    _, unseen_variance = classifier.output_moments(torch.tensor([[0.0, 1.0]]))

    assert bool((seen_variance < unseen_variance).all())


def test_contrast_logdet_is_finite_and_increases_with_diagonal_variance():
    base = torch.tensor([[0.2, 0.4, 0.8]], dtype=torch.float64)
    shifted = base + 0.5

    assert contrast_covariance_logdet(shifted) > contrast_covariance_logdet(base)
    assert torch.isfinite(contrast_covariance_logdet(base)).all()


def test_epistemic_mutual_information_is_zero_without_epistemic_variance():
    mean = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float64)
    variance = torch.zeros_like(mean)

    score = multinomial_probit_epistemic_mutual_information(
        mean, variance, num_samples=8
    )

    torch.testing.assert_close(score, torch.zeros_like(score), atol=1e-10, rtol=0)


def test_fit_rejects_a_second_pass():
    classifier = FullCovarianceADFClassifier(1, 2, device="cpu")
    features = torch.tensor([[-1.0], [1.0]])
    labels = torch.tensor([0, 1])
    classifier.fit(features, labels)

    with pytest.raises(RuntimeError, match="only once"):
        classifier.fit(features, labels)


def test_site_replacement_does_not_count_samples_twice():
    torch.manual_seed(3)
    classifier = FullCovarianceADFClassifier(1, 2, device="cpu")
    features = torch.tensor([[-1.0], [-0.5], [0.5], [1.0]])
    labels = torch.tensor([0, 0, 1, 1])

    records = classifier.fit(features, labels, batch_size=2, epochs=2)

    assert len(records) == 2
    assert classifier.samples_seen == features.shape[0]
    assert classifier.site_precision.shape == (4, 2)
    assert torch.isfinite(classifier.precision).all()


def test_fit_calls_epoch_callback_with_current_posterior() -> None:
    torch.manual_seed(17)
    features = torch.randn(9, 2, dtype=torch.float64)
    labels = torch.arange(9) % 3
    classifier = FullCovarianceADFClassifier(
        2, 3, device="cpu", gain_w=0.2, gain_b=0.2
    )
    seen: list[tuple[int, float]] = []

    def callback(epoch, current, record) -> None:
        prediction = current.predict(features)
        seen.append((epoch, prediction.probabilities.mean().item()))
        assert record["epoch"] == float(epoch)

    classifier.fit(
        features,
        labels,
        batch_size=3,
        epochs=2,
        epoch_callback=callback,
    )

    assert [epoch for epoch, _ in seen] == [1, 2]
