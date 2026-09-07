"""CUDA integration tests for Remax last-layer classifiers."""

from __future__ import annotations

import pytest
import torch

from triton_tagi.classification import TAGILastLayerClassifier

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize(
    ("approximation", "jacobian"),
    [("lognormal", "diag"), ("laplace", "full")],
)
def test_remax_head_trains_and_predicts(approximation, jacobian):
    torch.manual_seed(4)
    classifier = TAGILastLayerClassifier(
        4,
        3,
        head="remax",
        device="cuda",
        remax_approximation=approximation,
        remax_jacobian=jacobian,
        remax_num_quad=8,
    )
    values = torch.randn(8, 4)
    labels = torch.arange(8) % 3
    before = classifier.linear.mw.clone()
    classifier.train_step(values, labels, sigma_v=0.05)
    assert not torch.equal(before, classifier.linear.mw)
    prediction = classifier.predict(values)
    assert prediction.probabilities.shape == (8, 3)
    assert torch.isfinite(prediction.output_mean).all()
    assert torch.isfinite(prediction.output_variance).all()
    torch.testing.assert_close(
        prediction.probabilities.sum(1), torch.ones_like(prediction.probabilities[:, 0])
    )
