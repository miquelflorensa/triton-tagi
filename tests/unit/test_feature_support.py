import pytest
import torch

from triton_tagi.feature_support import BayesianFeatureSupportGate
from triton_tagi.metrics import classification_metrics


def make_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    first = 0.35 * torch.randn(40, 2, generator=generator) + torch.tensor([-2.0, 0.0])
    second = 0.35 * torch.randn(45, 2, generator=generator) + torch.tensor([2.0, 0.0])
    return torch.cat((first, second)), torch.cat(
        (torch.zeros(40, dtype=torch.long), torch.ones(45, dtype=torch.long))
    )


def test_support_evidence_is_finite_and_rejects_far_features():
    features, labels = make_training_data()
    gate = BayesianFeatureSupportGate.fit(features, labels)
    query = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [20.0, 20.0]])

    log_id, log_background = gate.log_evidence(query)

    assert log_id.shape == (3,)
    assert torch.isfinite(log_id).all()
    assert torch.isfinite(log_background).all()
    assert log_id[:2].min() > log_id[2]


def test_gate_builds_normalized_k_plus_one_distribution():
    features, labels = make_training_data()
    gate = BayesianFeatureSupportGate.fit(features, labels)
    query = torch.tensor([[-2.0, 0.0], [20.0, 20.0]])
    conditional = torch.tensor([[0.8, 0.2], [0.4, 0.6]])

    prediction = gate.predict(query, conditional)

    assert prediction.probabilities is not None
    assert prediction.probabilities.shape == (2, 3)
    torch.testing.assert_close(
        prediction.probabilities.sum(dim=1), torch.ones(2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        prediction.log_bayes_factor,
        prediction.log_id_evidence - prediction.log_background_evidence,
    )
    torch.testing.assert_close(
        prediction.id_probability + prediction.ood_probability,
        torch.ones(2, dtype=torch.float64),
    )
    metrics = classification_metrics(
        prediction.probabilities, torch.tensor([0, 1])
    )
    assert metrics["accuracy"] == 0.5


def test_support_gate_state_round_trip():
    features, labels = make_training_data()
    gate = BayesianFeatureSupportGate.fit(features, labels)
    restored = BayesianFeatureSupportGate.from_state_dict(gate.state_dict())
    query = torch.tensor([[-1.5, 0.1], [1.5, -0.1]])

    expected = gate.predict(query)
    actual = restored.predict(query)

    torch.testing.assert_close(actual.log_id_evidence, expected.log_id_evidence)
    torch.testing.assert_close(actual.log_bayes_factor, expected.log_bayes_factor)


def test_support_gate_requires_more_samples_than_dimensions_per_class():
    features = torch.randn(6, 3)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    with pytest.raises(ValueError, match="more samples than dimensions"):
        BayesianFeatureSupportGate.fit(features, labels)


def test_support_gate_rejects_malformed_conditional_probabilities():
    features, labels = make_training_data()
    gate = BayesianFeatureSupportGate.fit(features, labels)

    with pytest.raises(ValueError, match="normalized"):
        gate.predict(torch.zeros(1, 2), torch.tensor([[0.2, 0.2]]))
