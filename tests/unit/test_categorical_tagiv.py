"""Analytic checks for dense and hierarchical categorical TAGI-V."""

from __future__ import annotations

import torch

from triton_tagi.hrc_softmax import (
    class_to_obs,
    labels_to_hrc,
    obs_to_class_probs_tagiv,
)
from triton_tagi.update.observation import (
    compute_categorical_innovation,
    compute_hrc_tagiv_innovation,
    tempered_categorical_probs,
)


def test_tempered_probabilities_are_shift_and_permutation_invariant():
    torch.manual_seed(3)
    mean = torch.randn(7, 5, dtype=torch.float64)
    variance = torch.rand(7, 5, dtype=torch.float64)
    aleatoric = torch.rand(7, 5, dtype=torch.float64)
    expected = tempered_categorical_probs(mean, variance, aleatoric)
    shifted = tempered_categorical_probs(mean + 17.0, variance, aleatoric)
    torch.testing.assert_close(shifted, expected)

    permutation = torch.tensor([2, 4, 0, 1, 3])
    permuted = tempered_categorical_probs(
        mean[:, permutation], variance[:, permutation], aleatoric[:, permutation]
    )
    torch.testing.assert_close(permuted, expected[:, permutation])


def test_dense_categorical_tagiv_innovation_is_finite_and_centered():
    torch.manual_seed(4)
    batch, classes = 6, 4
    mean = torch.empty(batch, 2 * classes, dtype=torch.float64)
    variance = torch.empty_like(mean)
    mean[:, 0::2] = torch.randn(batch, classes, dtype=torch.float64)
    mean[:, 1::2] = 0.05 + torch.rand(batch, classes, dtype=torch.float64)
    variance[:, 0::2] = 0.01 + torch.rand(batch, classes, dtype=torch.float64)
    variance[:, 1::2] = 0.01 + torch.rand(batch, classes, dtype=torch.float64)
    labels = torch.arange(batch) % classes

    delta_mean, delta_variance = compute_categorical_innovation(
        labels, mean, variance, classes
    )
    assert delta_mean.shape == delta_variance.shape == mean.shape
    assert torch.isfinite(delta_mean).all()
    assert torch.isfinite(delta_variance).all()
    total_variance = variance[:, 0::2] + mean[:, 1::2]
    torch.testing.assert_close(
        (delta_mean[:, 0::2] * total_variance).sum(1),
        torch.zeros(batch, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_hrc_tagiv_updates_only_selected_path_nodes():
    hrc = class_to_obs(4)
    labels = torch.tensor([0, 3])
    observations, indices = labels_to_hrc(labels, hrc)
    mean = torch.zeros(2, 2 * hrc.len)
    variance = torch.full_like(mean, 0.1)
    mean[:, 1::2] = 0.2

    delta_mean, delta_variance = compute_hrc_tagiv_innovation(
        mean, variance, observations, indices
    )
    selected = torch.zeros_like(mean, dtype=torch.bool)
    node_indices = indices.long() - 1
    selected.scatter_(1, 2 * node_indices, True)
    selected.scatter_(1, 2 * node_indices + 1, True)
    assert torch.equal(delta_mean[~selected], torch.zeros_like(delta_mean[~selected]))
    assert torch.equal(
        delta_variance[~selected], torch.zeros_like(delta_variance[~selected])
    )

    probabilities = obs_to_class_probs_tagiv(mean, variance, hrc)
    assert probabilities.shape == (2, 4)
    assert torch.isfinite(probabilities).all()
