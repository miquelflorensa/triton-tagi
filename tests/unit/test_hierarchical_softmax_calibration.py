"""Equation-level tests for the literal hierarchical softmax calibration TeX."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from triton_tagi.hierarchical_softmax_calibration import (
    GaussianGainPosterior,
    calibrate_tex_hsm_gain_adf,
    tex_gain_groups,
    tex_hsm_adf_projection,
    tex_hsm_class_moments,
    tex_hsm_class_probabilities,
    tex_hsm_node_moments,
)
from triton_tagi.hrc_softmax import class_to_obs, class_to_obs_full


def test_node_moments_reproduce_the_tex_worked_example():
    hrc = class_to_obs_full(2, use_prior_offsets=False)
    posterior = GaussianGainPosterior.prior(
        tex_gain_groups(hrc), mean=1.5, variance=0.3**2
    )
    moments = tex_hsm_node_moments(
        torch.tensor([[0.5]], dtype=torch.float64),
        torch.tensor([[0.5**2]], dtype=torch.float64),
        hrc,
        posterior,
    )
    assert float(moments.mean) == pytest.approx(0.7229, abs=5e-4)
    assert math.sqrt(float(moments.variance)) == pytest.approx(0.2142, abs=5e-4)
    assert float(moments.cov_state) == pytest.approx(0.0977, abs=5e-4)
    assert float(moments.cov_gain) == pytest.approx(0.00773, abs=5e-5)


def test_node_moments_are_the_literal_equations_s_and_node():
    hrc = class_to_obs_full(2, use_prior_offsets=False)
    posterior = GaussianGainPosterior.prior(
        tex_gain_groups(hrc), mean=1.2, variance=0.4
    )
    mean_z = torch.tensor([[0.7]], dtype=torch.float64)
    variance_z = torch.tensor([[0.3]], dtype=torch.float64)
    moments = tex_hsm_node_moments(mean_z, variance_z, hrc, posterior)
    variance_s = 0.4 * 0.3 + 0.4 * 0.7**2 + 1.2**2 * 0.3
    argument = 1.2 * 0.7 / math.sqrt(1.0 + variance_s)
    density = math.exp(-0.5 * argument**2) / math.sqrt(2.0 * math.pi)
    expected_cov_gain = (
        0.4 * 0.7 * (1.0 + variance_s - 1.2**2 * 0.3)
        / (1.0 + variance_s) ** 1.5 * density
    )
    assert float(moments.mean) == pytest.approx(
        0.5 * math.erfc(-argument / math.sqrt(2.0)), rel=1e-14
    )
    assert float(moments.cov_gain) == pytest.approx(expected_cov_gain, rel=1e-14)


def test_adf_projection_is_equation_update_without_a_decrement_cap():
    mean, variance = tex_hsm_adf_projection(1.5, 0.09, 0.7229, 0.00773)
    assert mean == pytest.approx(1.5 + 0.00773 / 0.7229)
    assert variance == pytest.approx(0.09 - 0.00773**2 / (0.7229 * (1.0 - 0.7229)))
    assert math.sqrt(variance) == pytest.approx(0.2995, abs=2e-4)


def test_sequential_pass_uses_one_oriented_update_per_path_node():
    hrc = class_to_obs(10)
    rows = 7
    mean = torch.zeros(rows, hrc.len, dtype=torch.float64)
    variance = torch.ones_like(mean) * 0.25
    labels = torch.arange(rows) % 10
    posterior = calibrate_tex_hsm_gain_adf(mean, variance, labels, hrc, sharing="global")
    assert int(posterior.visits.sum()) == rows * hrc.n_obs
    assert float(posterior.mean[0]) == pytest.approx(0.3)
    assert float(posterior.variance[0]) == pytest.approx(1.0)


def test_a_new_epoch_keeps_the_mean_and_resets_the_variance():
    hrc = class_to_obs_full(2, use_prior_offsets=False)
    mean = torch.tensor([[0.8]], dtype=torch.float64)
    variance = torch.tensor([[0.64]], dtype=torch.float64)
    labels = torch.tensor([0])
    first = calibrate_tex_hsm_gain_adf(mean, variance, labels, hrc)
    second = calibrate_tex_hsm_gain_adf(
        mean, variance, labels, hrc, initial_mean=first.mean, prior_variance=1.0
    )
    assert float(second.mean[0]) > float(first.mean[0])
    assert float(second.variance[0]) > float(first.variance[0])


def test_class_moments_are_the_product_in_equation_class():
    hrc = class_to_obs_full(8, use_prior_offsets=False)
    generator = torch.Generator().manual_seed(7)
    mean = torch.randn(5, hrc.len, generator=generator, dtype=torch.float64)
    variance = torch.rand(5, hrc.len, generator=generator, dtype=torch.float64)
    posterior = GaussianGainPosterior.prior(tex_gain_groups(hrc), mean=1.0, variance=0.2)
    nodes = tex_hsm_node_moments(mean, variance, hrc, posterior)
    classes = tex_hsm_class_moments(mean, variance, hrc, posterior)
    codes = hrc.obs.double()
    indices = hrc.idx.long() - 1
    oriented_mean = torch.where(
        codes[None] > 0, nodes.mean[:, indices], 1 - nodes.mean[:, indices]
    )
    oriented_second = torch.where(
        codes[None] > 0,
        nodes.variance[:, indices] + nodes.mean[:, indices].square(),
        nodes.variance[:, indices] + (1 - nodes.mean[:, indices]).square(),
    )
    expected_mean = oriented_mean.prod(-1)
    assert torch.allclose(classes.mean, expected_mean, atol=1e-14)
    assert torch.allclose(
        classes.variance, oriented_second.prod(-1) - expected_mean.square(), atol=1e-14
    )


def test_padded_tree_probabilities_receive_the_prescribed_normalization():
    hrc = class_to_obs(10)
    mean = torch.zeros(3, hrc.len, dtype=torch.float64)
    variance = torch.zeros_like(mean)
    posterior = GaussianGainPosterior.prior(tex_gain_groups(hrc), mean=1.0, variance=0.0)
    raw = tex_hsm_class_moments(mean, variance, hrc, posterior).mean
    probabilities = tex_hsm_class_probabilities(mean, variance, hrc, posterior)
    assert torch.allclose(raw.sum(-1), torch.full((3,), 10.0 / 16.0, dtype=torch.float64))
    assert torch.allclose(probabilities.sum(-1), torch.ones(3, dtype=torch.float64))


def test_the_tex_api_has_no_sigma_v_parameter():
    for function in (
        tex_hsm_node_moments,
        tex_hsm_class_moments,
        tex_hsm_class_probabilities,
        calibrate_tex_hsm_gain_adf,
    ):
        assert "sigma_v" not in inspect.signature(function).parameters
