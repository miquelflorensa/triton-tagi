"""Analytic and Monte Carlo checks for logit-space TAGI-V."""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi.logit_tagiv import (
    LOGIT_VARIANCE_FLOOR,
    center_logits,
    compute_logit_mean_innovation,
    compute_logit_replicate_innovation,
    compute_logit_tagiv_innovation,
    fit_logit_calibration,
    gaussian_shrinkage_,
    logit_exp_variance_moments,
    logit_feature_energy,
    logit_replicate_variance,
    logit_tagiv_predictive_nll,
    logit_tagiv_predictive_probs,
    logit_tagiv_uncertainty,
    logit_target_scale,
    logit_variance_head_prior,
    logit_variance_prior_split,
    power_compress_variance,
    prepare_logit_targets,
    replicate_log_variance_noise,
    replicate_log_variance_offset,
    split_logit_tagiv_outputs,
    standard_normal_base_samples,
)

DOUBLE = torch.float64


def _interleave(mu_z, v_z, mu_x, v_x):
    """Pack logit and post-EvenExp noise moments into the 2K layout."""

    ma = torch.empty(*mu_z.shape[:-1], 2 * mu_z.shape[-1], dtype=mu_z.dtype)
    Sa = torch.empty_like(ma)
    ma[..., 0::2], Sa[..., 0::2] = mu_z, v_z
    ma[..., 1::2], Sa[..., 1::2] = mu_x, v_x
    return ma, Sa


def _reference_innovation(targets, ma, Sa, variance_floor):
    """Recompute the innovation literally from the formulation.

    This keeps the divisions by ``v_S`` and ``v_G`` that
    :func:`compute_logit_tagiv_innovation` cancels analytically, so agreement
    confirms the cancellation rather than restating it.
    """

    mu_z, v_z = ma[..., 0::2], Sa[..., 0::2]
    mu_x, v_x = ma[..., 1::2], Sa[..., 1::2]
    mu_s, v_s = variance_floor + mu_x, v_x

    # Invert the log-normal moments to recover the pre-activation prior.
    v_g = torch.log1p(v_x / mu_x.square())

    residual = targets - mu_z
    q = v_z + mu_s
    mu_z_post = mu_z + v_z / q * residual
    v_z_post = v_z - v_z.square() / q
    mu_v_post = mu_s / q * residual
    v_v_post = mu_s - mu_s.square() / q

    mu_r_post = mu_v_post.square() + v_v_post
    v_r_post = 2.0 * v_v_post.square() + 4.0 * v_v_post * mu_v_post.square()
    v_r = 3.0 * v_s + 2.0 * mu_s.square()
    gain_s = v_s / v_r
    mu_s_post = mu_s + gain_s * (mu_r_post - mu_s)
    v_s_post = v_s + gain_s.square() * (v_r_post - v_r)

    gain_g = v_g * mu_x / v_s
    mu_g_post_change = gain_g * (mu_s_post - mu_s)
    v_g_post_change = gain_g.square() * (v_s_post - v_s)

    delta_ma = torch.empty_like(ma)
    delta_Sa = torch.empty_like(Sa)
    delta_ma[..., 0::2] = (mu_z_post - mu_z) / v_z
    delta_Sa[..., 0::2] = (v_z_post - v_z) / v_z.square()
    delta_ma[..., 1::2] = mu_g_post_change / v_g
    delta_Sa[..., 1::2] = v_g_post_change / v_g.square()
    return delta_ma, delta_Sa


# ======================================================================
#  Targets
# ======================================================================


def test_center_logits_removes_the_unidentified_offset():
    torch.manual_seed(0)
    logits = torch.randn(6, 5, dtype=DOUBLE)
    centered = center_logits(logits)
    torch.testing.assert_close(
        centered.mean(dim=-1), torch.zeros(6, dtype=DOUBLE), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(torch.softmax(centered, dim=-1), torch.softmax(logits, dim=-1))
    torch.testing.assert_close(center_logits(logits + 9.0), centered)


def test_prepare_logit_targets_normalizes_the_global_scale():
    torch.manual_seed(1)
    logits = 4.0 * torch.randn(512, 7, dtype=DOUBLE)
    targets, scale = prepare_logit_targets(logits)
    assert scale == pytest.approx(logit_target_scale(center_logits(logits)))
    assert float(targets.square().mean().sqrt()) == pytest.approx(1.0, abs=1e-12)

    held_out, reused = prepare_logit_targets(logits[:16], scale=scale)
    torch.testing.assert_close(held_out, targets[:16])
    assert reused == scale


def test_target_helpers_reject_degenerate_input():
    with pytest.raises(ValueError, match="class dimension"):
        center_logits(torch.zeros(4, 1, dtype=DOUBLE))
    with pytest.raises(ValueError, match="finite"):
        center_logits(torch.full((2, 3), float("nan"), dtype=DOUBLE))
    with pytest.raises(ValueError, match="positive scale"):
        logit_target_scale(torch.zeros(4, 3, dtype=DOUBLE))


# ======================================================================
#  Variance head moments
# ======================================================================


def test_variance_head_prior_hits_the_requested_noise_moments():
    mu_g, v_g = logit_variance_head_prior(
        aleatoric_init=0.25, variance_floor=1e-6, coefficient_of_variation=0.4
    )
    mu_s, v_s, _ = logit_exp_variance_moments(
        torch.tensor([mu_g], dtype=DOUBLE),
        torch.tensor([v_g], dtype=DOUBLE),
        variance_floor=1e-6,
    )
    assert float(mu_s) == pytest.approx(0.25, rel=1e-12)
    mu_x = float(mu_s) - 1e-6
    assert math.sqrt(float(v_s)) / mu_x == pytest.approx(0.4, rel=1e-12)


def test_variance_head_prior_rejects_a_collapsed_prior():
    with pytest.raises(ValueError, match="must exceed variance_floor"):
        logit_variance_head_prior(aleatoric_init=1e-6, variance_floor=1e-6)
    with pytest.raises(ValueError, match="coefficient_of_variation"):
        logit_variance_head_prior(aleatoric_init=0.1, coefficient_of_variation=0.0)


def test_exp_variance_moments_match_monte_carlo():
    torch.manual_seed(2)
    mz = torch.tensor([-2.0, -0.5, 0.75], dtype=DOUBLE)
    Sz = torch.tensor([0.05, 0.3, 0.6], dtype=DOUBLE)
    mu_s, v_s, cov = logit_exp_variance_moments(mz, Sz, variance_floor=0.01)

    draws = mz + Sz.sqrt() * torch.randn(2_000_000, 3, dtype=DOUBLE)
    sampled = 0.01 + draws.exp()
    torch.testing.assert_close(sampled.mean(0), mu_s, rtol=5e-3, atol=0)
    torch.testing.assert_close(sampled.var(0), v_s, rtol=3e-2, atol=0)
    centered = (draws - mz) * (sampled - sampled.mean(0))
    torch.testing.assert_close(centered.mean(0), cov, rtol=2e-2, atol=1e-6)


def test_exp_variance_moments_stay_accurate_for_a_tiny_prior_variance():
    mz = torch.tensor([-1.0], dtype=DOUBLE)
    Sz = torch.tensor([1e-13], dtype=DOUBLE)
    mu_s, v_s, _ = logit_exp_variance_moments(mz, Sz, variance_floor=0.0)
    # The delta-method limit v_X -> mu_X**2 * v_G is exact to first order; a
    # naive exp(2m + s) - exp(2m + 2s) difference loses it to cancellation.
    torch.testing.assert_close(v_s, mu_s.square() * Sz, rtol=1e-6, atol=0)


def test_split_rejects_an_odd_output_width():
    with pytest.raises(ValueError, match="even width"):
        split_logit_tagiv_outputs(torch.zeros(2, 5), torch.zeros(2, 5))


# ======================================================================
#  Observation update
# ======================================================================


def test_innovation_matches_the_uncancelled_reference():
    torch.manual_seed(3)
    batch, classes = 5, 4
    mu_z = torch.randn(batch, classes, dtype=DOUBLE)
    v_z = 0.05 + torch.rand(batch, classes, dtype=DOUBLE)
    mu_x = 0.02 + torch.rand(batch, classes, dtype=DOUBLE)
    v_x = 0.01 + torch.rand(batch, classes, dtype=DOUBLE)
    ma, Sa = _interleave(mu_z, v_z, mu_x, v_x)
    targets = torch.randn(batch, classes, dtype=DOUBLE)

    delta_ma, delta_Sa = compute_logit_tagiv_innovation(targets, ma, Sa, variance_floor=1e-3)
    reference_ma, reference_Sa = _reference_innovation(targets, ma, Sa, 1e-3)
    torch.testing.assert_close(delta_ma, reference_ma, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(delta_Sa, reference_Sa, rtol=1e-10, atol=1e-12)


def test_latent_logit_posterior_matches_kalman_conditioning():
    torch.manual_seed(4)
    mu_z = torch.randn(3, 2, dtype=DOUBLE)
    v_z = 0.1 + torch.rand(3, 2, dtype=DOUBLE)
    mu_x = 0.3 + torch.rand(3, 2, dtype=DOUBLE)
    v_x = 1e-9 * torch.ones(3, 2, dtype=DOUBLE)
    ma, Sa = _interleave(mu_z, v_z, mu_x, v_x)
    targets = torch.randn(3, 2, dtype=DOUBLE)

    delta_ma, delta_Sa = compute_logit_tagiv_innovation(targets, ma, Sa, variance_floor=0.0)
    posterior_mean = mu_z + v_z * delta_ma[:, 0::2]
    posterior_variance = v_z + v_z.square() * delta_Sa[:, 0::2]

    total = v_z + mu_x
    torch.testing.assert_close(posterior_mean, mu_z + v_z / total * (targets - mu_z))
    torch.testing.assert_close(posterior_variance, v_z - v_z.square() / total)
    # A frozen variance head leaves the observation model Gaussian, so the
    # posterior variance must equal the exact conditional one.
    torch.testing.assert_close(posterior_variance, 1.0 / (1.0 / v_z + 1.0 / mu_x))


def test_agvi_prior_moments_match_monte_carlo():
    torch.manual_seed(5)
    mz = torch.tensor([-1.2], dtype=DOUBLE)
    Sz = torch.tensor([0.4], dtype=DOUBLE)
    floor = 0.02
    mu_s, v_s, _ = logit_exp_variance_moments(mz, Sz, variance_floor=floor)

    noise = mz + Sz.sqrt() * torch.randn(4_000_000, 1, dtype=DOUBLE)
    variance = floor + noise.exp()
    residual = variance.sqrt() * torch.randn_like(variance)
    square = residual.square()
    assert float(square.mean()) == pytest.approx(float(mu_s), rel=5e-3)
    assert float(square.var()) == pytest.approx(float(3.0 * v_s + 2.0 * mu_s.square()), rel=5e-2)


def test_residual_square_posterior_matches_monte_carlo():
    torch.manual_seed(6)
    mu_z = torch.tensor([[0.4]], dtype=DOUBLE)
    v_z = torch.tensor([[0.3]], dtype=DOUBLE)
    noise_variance = torch.tensor([[0.2]], dtype=DOUBLE)
    target = torch.tensor([[1.1]], dtype=DOUBLE)

    total = v_z + noise_variance
    mu_v = noise_variance / total * (target - mu_z)
    v_v = noise_variance - noise_variance.square() / total

    # Sample the joint prior and reweight by the Gaussian observation density.
    latent = mu_z + v_z.sqrt() * torch.randn(2_000_000, 1, dtype=DOUBLE)
    residual_draw = noise_variance.sqrt() * torch.randn(2_000_000, 1, dtype=DOUBLE)
    observation = latent + residual_draw
    weight = torch.exp(-0.5 * (observation - target).square() / 1e-4)
    weight = weight / weight.sum()
    weighted_mean = float((weight * residual_draw).sum())
    weighted_var = float((weight * (residual_draw - weighted_mean).square()).sum())
    assert weighted_mean == pytest.approx(float(mu_v), abs=5e-3)
    assert weighted_var == pytest.approx(float(v_v), rel=5e-2)

    square_mean = mu_v.square() + v_v
    square_var = 2.0 * v_v.square() + 4.0 * v_v * mu_v.square()
    gaussian = mu_v + v_v.sqrt() * torch.randn(2_000_000, 1, dtype=DOUBLE)
    assert float(gaussian.square().mean()) == pytest.approx(float(square_mean), rel=5e-3)
    assert float(gaussian.square().var()) == pytest.approx(float(square_var), rel=2e-2)


def test_variance_head_moves_toward_the_observed_residual_size():
    mu_z = torch.zeros(2, 1, dtype=DOUBLE)
    v_z = torch.full((2, 1), 0.01, dtype=DOUBLE)
    mu_x = torch.full((2, 1), 0.2, dtype=DOUBLE)
    v_x = torch.full((2, 1), 0.05, dtype=DOUBLE)
    ma, Sa = _interleave(mu_z, v_z, mu_x, v_x)
    targets = torch.tensor([[0.0], [3.0]], dtype=DOUBLE)

    delta_ma, delta_Sa = compute_logit_tagiv_innovation(targets, ma, Sa)
    # A residual far smaller than the current noise scale shrinks the head and a
    # large one inflates it.
    assert float(delta_ma[0, 1]) < 0.0
    assert float(delta_ma[1, 1]) > 0.0
    # AGVI matches moments rather than conditioning exactly, so a residual
    # square well above its prior expectation widens the noise posterior; only
    # the consistent row contracts it.
    assert float(delta_Sa[0, 1]) < 0.0
    assert float(delta_Sa[1, 1]) > 0.0


def test_innovation_is_finite_for_a_frozen_variance_head():
    mu_z = torch.zeros(1, 2, dtype=DOUBLE)
    v_z = torch.full((1, 2), 0.1, dtype=DOUBLE)
    mu_x = torch.full((1, 2), 0.2, dtype=DOUBLE)
    v_x = torch.zeros(1, 2, dtype=DOUBLE)
    ma, Sa = _interleave(mu_z, v_z, mu_x, v_x)

    delta_ma, delta_Sa = compute_logit_tagiv_innovation(torch.ones(1, 2, dtype=DOUBLE), ma, Sa)
    assert torch.isfinite(delta_ma).all() and torch.isfinite(delta_Sa).all()
    # The deltas are standardized by v_G, so they stay finite at v_G = 0 while
    # the posterior change they encode, v_G * delta, vanishes with it. The
    # parameter update reintroduces that factor through the prior variances.
    target = torch.ones(1, 2, dtype=DOUBLE)
    changes = []
    for prior_variance in (1e-3, 1e-6, 1e-9):
        perturbed_x = mu_x.square() * torch.expm1(torch.tensor(prior_variance, dtype=DOUBLE))
        ma_p, Sa_p = _interleave(mu_z, v_z, mu_x, perturbed_x)
        delta_p, _ = compute_logit_tagiv_innovation(target, ma_p, Sa_p)
        changes.append(float(delta_p[0, 1]) * prior_variance)
    assert changes[0] / changes[1] == pytest.approx(1000.0, rel=1e-2)
    assert changes[1] / changes[2] == pytest.approx(1000.0, rel=1e-2)


def test_innovation_rejects_mismatched_targets():
    ma = torch.zeros(2, 6, dtype=DOUBLE)
    Sa = torch.full((2, 6), 0.1, dtype=DOUBLE)
    with pytest.raises(ValueError, match="targets must have shape"):
        compute_logit_tagiv_innovation(torch.zeros(2, 6, dtype=DOUBLE), ma, Sa)
    with pytest.raises(ValueError, match="finite"):
        compute_logit_tagiv_innovation(torch.full((2, 3), float("inf"), dtype=DOUBLE), ma, Sa)


# ======================================================================
#  Prediction, decomposition, calibration
# ======================================================================


def test_predictive_probabilities_reduce_to_a_tempered_softmax():
    torch.manual_seed(7)
    mu = torch.randn(4, 5, dtype=DOUBLE)
    zeros = torch.zeros_like(mu)
    probabilities = logit_tagiv_predictive_probs(mu, zeros, zeros, temperature=1.7, num_samples=16)
    torch.testing.assert_close(probabilities, torch.softmax(mu / 1.7, dim=-1))


def test_predictive_probabilities_ignore_a_common_logit_offset():
    torch.manual_seed(8)
    mu = torch.randn(4, 5, dtype=DOUBLE)
    epistemic = 0.1 + torch.rand(4, 5, dtype=DOUBLE)
    aleatoric = 0.1 + torch.rand(4, 5, dtype=DOUBLE)
    samples = standard_normal_base_samples(128, 5, dtype=DOUBLE)
    baseline = logit_tagiv_predictive_probs(mu, epistemic, aleatoric, base_samples=samples)
    shifted = logit_tagiv_predictive_probs(mu + 3.0, epistemic, aleatoric, base_samples=samples)
    torch.testing.assert_close(shifted, baseline)
    torch.testing.assert_close(baseline.sum(-1), torch.ones(4, dtype=DOUBLE), atol=1e-12, rtol=0)


def test_sobol_predictions_agree_with_plain_monte_carlo():
    torch.manual_seed(9)
    mu = torch.randn(3, 4, dtype=DOUBLE)
    epistemic = 0.2 + torch.rand(3, 4, dtype=DOUBLE)
    aleatoric = 0.3 + torch.rand(3, 4, dtype=DOUBLE)
    sobol = logit_tagiv_predictive_probs(mu, epistemic, aleatoric, num_samples=4096)
    plain = logit_tagiv_predictive_probs(
        mu,
        epistemic,
        aleatoric,
        base_samples=standard_normal_base_samples(200_000, 4, method="gaussian", dtype=DOUBLE),
    )
    torch.testing.assert_close(sobol, plain, atol=2e-3, rtol=0)


def test_aleatoric_variance_and_temperature_both_flatten_predictions():
    torch.manual_seed(10)
    mu = 3.0 * torch.randn(8, 6, dtype=DOUBLE)
    epistemic = 0.05 * torch.ones_like(mu)
    aleatoric = torch.ones_like(mu)
    samples = standard_normal_base_samples(512, 6, dtype=DOUBLE)

    def entropy(temperature, alpha):
        probabilities = logit_tagiv_predictive_probs(
            mu,
            epistemic,
            aleatoric,
            temperature=temperature,
            alpha=alpha,
            base_samples=samples,
        )
        return float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).mean())

    assert entropy(1.0, 0.0) < entropy(1.0, 1.0) < entropy(1.0, 4.0)
    assert entropy(0.5, 1.0) < entropy(1.0, 1.0) < entropy(2.0, 1.0)


def test_uncertainty_decomposition_is_additive_and_nonnegative():
    torch.manual_seed(11)
    mu = torch.randn(16, 5, dtype=DOUBLE)
    epistemic = 0.4 * torch.rand(16, 5, dtype=DOUBLE)
    aleatoric = 0.5 * torch.rand(16, 5, dtype=DOUBLE)
    decomposition = logit_tagiv_uncertainty(
        mu, epistemic, aleatoric, epistemic_samples=64, aleatoric_samples=64
    )
    torch.testing.assert_close(
        decomposition.total_entropy,
        decomposition.aleatoric_entropy + decomposition.epistemic_entropy,
    )
    assert bool((decomposition.epistemic_entropy >= -1e-12).all())
    assert bool((decomposition.aleatoric_entropy >= 0.0).all())
    torch.testing.assert_close(
        decomposition.probabilities.sum(-1),
        torch.ones(16, dtype=DOUBLE),
        atol=1e-12,
        rtol=0,
    )


def test_decomposition_assigns_no_epistemic_entropy_without_epistemic_variance():
    torch.manual_seed(12)
    mu = torch.randn(6, 4, dtype=DOUBLE)
    aleatoric = 0.6 * torch.ones_like(mu)
    decomposition = logit_tagiv_uncertainty(
        mu, torch.zeros_like(mu), aleatoric, epistemic_samples=32, aleatoric_samples=256
    )
    torch.testing.assert_close(
        decomposition.epistemic_entropy,
        torch.zeros(6, dtype=DOUBLE),
        atol=1e-12,
        rtol=0,
    )


def test_calibration_recovers_a_planted_temperature():
    generator = torch.Generator().manual_seed(13)
    mu = 4.0 * torch.randn(8000, 4, generator=generator, dtype=DOUBLE)
    true_temperature = 2.5
    labels = torch.multinomial(
        torch.softmax(mu / true_temperature, dim=-1), 1, generator=generator
    ).squeeze(1)
    zeros = torch.zeros_like(mu)
    calibration = fit_logit_calibration(
        mu, zeros, zeros, labels, mode="temperature", alpha=0.0, num_samples=8
    )
    assert calibration.temperature == pytest.approx(true_temperature, rel=0.1)
    assert calibration.nll < logit_tagiv_predictive_nll(
        mu, zeros, zeros, labels, temperature=1.0, alpha=0.0, num_samples=8
    )


def test_joint_calibration_never_loses_to_the_fixed_multiplier():
    generator = torch.Generator().manual_seed(14)
    mu = 2.0 * torch.randn(3000, 5, generator=generator, dtype=DOUBLE)
    labels = torch.multinomial(torch.softmax(mu, dim=-1), 1, generator=generator).squeeze(1)
    epistemic = 0.05 * torch.ones_like(mu)
    # An aleatoric channel that carries no information about the labels; the
    # joint search must be free to shrink it away.
    aleatoric = 4.0 * torch.rand(3000, 5, generator=generator, dtype=DOUBLE)
    samples = standard_normal_base_samples(128, 5, dtype=DOUBLE)

    fixed = fit_logit_calibration(
        mu,
        epistemic,
        aleatoric,
        labels,
        mode="temperature",
        alpha=1.0,
        base_samples=samples,
    )
    joint = fit_logit_calibration(
        mu, epistemic, aleatoric, labels, mode="joint", base_samples=samples
    )
    assert joint.nll <= fixed.nll + 1e-9
    assert joint.alpha < 1.0
    assert fixed.alpha == 1.0


def test_calibration_rejects_an_unknown_mode():
    mu = torch.zeros(4, 3, dtype=DOUBLE)
    with pytest.raises(ValueError, match="mode must be"):
        fit_logit_calibration(mu, mu, mu, torch.zeros(4, dtype=torch.long), mode="grid")


def test_default_variance_floor_is_positive():
    assert LOGIT_VARIANCE_FLOOR > 0.0


def test_variance_prior_split_reconstructs_the_target_at_the_mean_energy():
    weight_variance, bias_variance = logit_variance_prior_split(
        0.25, feature_energy=40.0, weight_share=0.5
    )
    assert weight_variance * 40.0 + bias_variance == pytest.approx(0.25)
    assert weight_variance * 40.0 == pytest.approx(bias_variance)

    frozen_weight, frozen_bias = logit_variance_prior_split(
        0.25, feature_energy=40.0, weight_share=0.0
    )
    assert frozen_weight == 0.0
    assert frozen_bias == pytest.approx(0.25)


def test_variance_prior_split_rejects_a_full_weight_share():
    with pytest.raises(ValueError, match="weight_share"):
        logit_variance_prior_split(0.25, feature_energy=40.0, weight_share=1.0)
    with pytest.raises(ValueError, match="feature_energy"):
        logit_variance_prior_split(0.25, feature_energy=0.0, weight_share=0.5)


def test_feature_energy_is_the_mean_squared_norm():
    features = torch.tensor([[3.0, 4.0], [0.0, 1.0]], dtype=DOUBLE)
    assert logit_feature_energy(features) == pytest.approx(13.0)
    with pytest.raises(ValueError, match="samples, dimensions"):
        logit_feature_energy(torch.zeros(0, 3, dtype=DOUBLE))


# ======================================================================
#  Two-stage and replicate-aware variance updates
# ======================================================================


def test_frozen_mean_innovation_zeroes_only_the_latent_stream():
    torch.manual_seed(20)
    mu_z = torch.randn(4, 3, dtype=DOUBLE)
    v_z = 0.1 + torch.rand(4, 3, dtype=DOUBLE)
    mu_x = 0.2 + torch.rand(4, 3, dtype=DOUBLE)
    v_x = 0.05 + torch.rand(4, 3, dtype=DOUBLE)
    ma, Sa = _interleave(mu_z, v_z, mu_x, v_x)
    targets = torch.randn(4, 3, dtype=DOUBLE)

    free_ma, free_Sa = compute_logit_tagiv_innovation(targets, ma, Sa)
    frozen_ma, frozen_Sa = compute_logit_tagiv_innovation(targets, ma, Sa, update_mean=False)
    zeros = torch.zeros(4, 3, dtype=DOUBLE)
    torch.testing.assert_close(frozen_ma[:, 0::2], zeros, atol=0, rtol=0)
    torch.testing.assert_close(frozen_Sa[:, 0::2], zeros, atol=0, rtol=0)
    # Freezing the mean must not disturb the variance stream's own update.
    torch.testing.assert_close(frozen_ma[:, 1::2], free_ma[:, 1::2])
    torch.testing.assert_close(frozen_Sa[:, 1::2], free_Sa[:, 1::2])


def test_mean_innovation_is_the_fixed_noise_kalman_update():
    torch.manual_seed(21)
    mu_z = torch.randn(5, 2, dtype=DOUBLE)
    v_z = 0.2 + torch.rand(5, 2, dtype=DOUBLE)
    ma, Sa = _interleave(mu_z, v_z, torch.rand(5, 2, dtype=DOUBLE), torch.rand(5, 2, dtype=DOUBLE))
    targets = torch.randn(5, 2, dtype=DOUBLE)

    delta_ma, delta_Sa = compute_logit_mean_innovation(targets, ma, Sa, observation_variance=0.25)
    total = v_z + 0.25
    torch.testing.assert_close(delta_ma[:, 0::2], (targets - mu_z) / total)
    torch.testing.assert_close(delta_Sa[:, 0::2], -1.0 / total)
    zeros = torch.zeros(5, 2, dtype=DOUBLE)
    torch.testing.assert_close(delta_ma[:, 1::2], zeros, atol=0, rtol=0)
    torch.testing.assert_close(delta_Sa[:, 1::2], zeros, atol=0, rtol=0)
    # The learned noise must not enter a fixed-noise warm-up gain at all.
    other_ma, _ = compute_logit_mean_innovation(
        targets,
        *_interleave(
            mu_z, v_z, 9.0 * torch.ones(5, 2, dtype=DOUBLE), torch.rand(5, 2, dtype=DOUBLE)
        ),
        observation_variance=0.25,
    )
    torch.testing.assert_close(other_ma, delta_ma)


def test_replicate_sample_variance_moments_match_monte_carlo():
    torch.manual_seed(22)
    repeats, variance = 6, 0.35
    draws = variance**0.5 * torch.randn(400_000, repeats, dtype=DOUBLE)
    sample = draws.var(dim=1, unbiased=True)
    dispersion = 2.0 / (repeats - 1)
    assert float(sample.mean()) == pytest.approx(variance, rel=5e-3)
    assert float(sample.var()) == pytest.approx(dispersion * variance**2, rel=5e-2)


def test_replicate_innovation_matches_the_uncancelled_kalman_form():
    torch.manual_seed(23)
    mu_x = 0.1 + torch.rand(4, 3, dtype=DOUBLE)
    v_x = 0.02 + torch.rand(4, 3, dtype=DOUBLE)
    ma, Sa = _interleave(torch.randn(4, 3, dtype=DOUBLE), torch.rand(4, 3, dtype=DOUBLE), mu_x, v_x)
    observed = 0.05 + torch.rand(4, 3, dtype=DOUBLE)
    floor, repeats = 1e-3, 8

    delta_ma, delta_Sa = compute_logit_replicate_innovation(
        observed, ma, Sa, repeats=repeats, variance_floor=floor
    )

    mu_s, v_s = floor + mu_x, v_x
    v_g = torch.log1p(v_x / mu_x.square())
    dispersion = 2.0 / (repeats - 1)
    var_r = (1.0 + dispersion) * v_s + dispersion * mu_s.square()
    gain_s = v_s / var_r
    mu_s_post = mu_s + gain_s * (observed - mu_s)
    v_s_post = v_s - v_s.square() / var_r
    gain_g = v_g * mu_x / v_s
    torch.testing.assert_close(
        delta_ma[:, 1::2], gain_g * (mu_s_post - mu_s) / v_g, rtol=1e-10, atol=1e-12
    )
    torch.testing.assert_close(
        delta_Sa[:, 1::2], gain_g.square() * (v_s_post - v_s) / v_g.square(), rtol=1e-10, atol=1e-12
    )
    zeros = torch.zeros(4, 3, dtype=DOUBLE)
    torch.testing.assert_close(delta_ma[:, 0::2], zeros, atol=0, rtol=0)


def test_replicate_update_always_contracts_the_variance_posterior():
    torch.manual_seed(24)
    ma, Sa = _interleave(
        torch.zeros(64, 4, dtype=DOUBLE),
        torch.full((64, 4), 0.1, dtype=DOUBLE),
        0.05 + torch.rand(64, 4, dtype=DOUBLE),
        0.01 + torch.rand(64, 4, dtype=DOUBLE),
    )
    # Even an observation far above the prior mean must not inflate v_S, which
    # the single-observation AGVI form is free to do.
    observed = 20.0 * torch.rand(64, 4, dtype=DOUBLE)
    _, delta_Sa = compute_logit_replicate_innovation(observed, ma, Sa, repeats=4)
    assert bool((delta_Sa[:, 1::2] < 0.0).all())


def test_replicate_update_is_blind_to_the_latent_mean():
    """The whole point: a mean-head error cannot be recorded as noise."""

    torch.manual_seed(25)
    mu_x = 0.2 + torch.rand(6, 5, dtype=DOUBLE)
    v_x = 0.03 + torch.rand(6, 5, dtype=DOUBLE)
    observed = 0.1 + torch.rand(6, 5, dtype=DOUBLE)
    accurate, Sa = _interleave(
        torch.zeros(6, 5, dtype=DOUBLE), torch.rand(6, 5, dtype=DOUBLE), mu_x, v_x
    )
    wrong, _ = _interleave(
        50.0 * torch.ones(6, 5, dtype=DOUBLE), torch.rand(6, 5, dtype=DOUBLE), mu_x, v_x
    )
    reference = compute_logit_replicate_innovation(observed, accurate, Sa, repeats=8)
    perturbed = compute_logit_replicate_innovation(observed, wrong, Sa, repeats=8)
    torch.testing.assert_close(perturbed[0], reference[0], atol=0, rtol=0)
    torch.testing.assert_close(perturbed[1], reference[1], atol=0, rtol=0)

    # The single-observation form is not blind to it, which is the contamination
    # the replicate-aware update removes.
    single = compute_logit_tagiv_innovation(observed, accurate, Sa, update_mean=False)
    single_wrong = compute_logit_tagiv_innovation(observed, wrong, Sa, update_mean=False)
    assert not torch.allclose(single_wrong[0][:, 1::2], single[0][:, 1::2])


def test_replicate_update_recovers_a_planted_variance():
    """Iterating the update on replicated draws must converge to their variance."""

    torch.manual_seed(26)
    truth = 0.6
    prior_mu, prior_v = logit_variance_head_prior(aleatoric_init=0.05, coefficient_of_variation=1.0)
    mu_g = torch.full((1, 1), prior_mu, dtype=DOUBLE)
    v_g = torch.full((1, 1), prior_v, dtype=DOUBLE)
    repeats = 8
    for _ in range(300):
        draws = truth**0.5 * torch.randn(1, repeats, dtype=DOUBLE)
        observed = draws.var(dim=1, unbiased=True).reshape(1, 1)
        mu_s, v_s, _ = logit_exp_variance_moments(mu_g, v_g)
        ma, Sa = _interleave(
            torch.zeros(1, 1, dtype=DOUBLE),
            torch.zeros(1, 1, dtype=DOUBLE),
            mu_s - LOGIT_VARIANCE_FLOOR,
            v_s,
        )
        delta_ma, delta_Sa = compute_logit_replicate_innovation(observed, ma, Sa, repeats=repeats)
        mu_g = mu_g + v_g * delta_ma[:, 1::2]
        v_g = (v_g + v_g.square() * delta_Sa[:, 1::2]).clamp_min(1e-8)
    final = float(logit_exp_variance_moments(mu_g, v_g)[0])
    assert final == pytest.approx(truth, rel=0.2)


def test_replicate_innovation_rejects_bad_input():
    ma = torch.zeros(2, 4, dtype=DOUBLE)
    Sa = torch.full((2, 4), 0.1, dtype=DOUBLE)
    with pytest.raises(ValueError, match="at least two repeats"):
        compute_logit_replicate_innovation(torch.ones(2, 2, dtype=DOUBLE), ma, Sa, repeats=1)
    with pytest.raises(ValueError, match="nonnegative"):
        compute_logit_replicate_innovation(-torch.ones(2, 2, dtype=DOUBLE), ma, Sa, repeats=4)


def test_replicate_variance_helper_matches_manual_moments():
    torch.manual_seed(27)
    repeated = torch.randn(5, 6, 3, dtype=DOUBLE)
    mean, variance = logit_replicate_variance(repeated, scale=2.0)
    centered = (repeated - repeated.mean(dim=-1, keepdim=True)) / 2.0
    torch.testing.assert_close(mean, centered.mean(dim=1))
    torch.testing.assert_close(variance, centered.var(dim=1, unbiased=True))
    with pytest.raises(ValueError, match="repeats >= 2"):
        logit_replicate_variance(torch.randn(4, 1, 3, dtype=DOUBLE))


# ======================================================================
#  Spread control
# ======================================================================


def test_gaussian_shrinkage_matches_the_precision_product():
    mean = torch.tensor([2.0, -3.0], dtype=DOUBLE)
    variance = torch.tensor([0.5, 0.25], dtype=DOUBLE)
    expected_variance = 1.0 / (1.0 / variance + 4.0)
    expected_mean = mean / (1.0 + 4.0 * variance)
    gaussian_shrinkage_(mean, variance, 4.0)
    torch.testing.assert_close(variance, expected_variance)
    torch.testing.assert_close(mean, expected_mean)


def test_gaussian_shrinkage_is_a_noop_at_zero_rate():
    mean = torch.tensor([1.5], dtype=DOUBLE)
    variance = torch.tensor([0.3], dtype=DOUBLE)
    gaussian_shrinkage_(mean, variance, 0.0)
    torch.testing.assert_close(mean, torch.tensor([1.5], dtype=DOUBLE))
    torch.testing.assert_close(variance, torch.tensor([0.3], dtype=DOUBLE))
    with pytest.raises(ValueError, match="nonnegative"):
        gaussian_shrinkage_(mean, variance, -1.0)


def test_repeated_shrinkage_preserves_the_mean_to_variance_ratio():
    """Both moments scale by the same factor, so the regularizer self-attenuates.

    The mean can only shrink in proportion to the variance, and the variance
    decays as ``v_0 / (1 + rate * v_0 * steps)``. Once the variance head has
    become confident the shrinkage barely moves it, which bounds how much
    spread control any single rate can deliver late in training.
    """

    mean = torch.tensor([5.0], dtype=DOUBLE)
    variance = torch.tensor([0.01], dtype=DOUBLE)
    ratio = float(mean / variance)
    steps, rate = 2000, 1.0
    for _ in range(steps):
        gaussian_shrinkage_(mean, variance, rate)
    assert float(mean / variance) == pytest.approx(ratio, rel=1e-9)
    assert float(variance) == pytest.approx(0.01 / (1.0 + rate * 0.01 * steps), rel=1e-9)
    assert float(mean) == pytest.approx(5.0 * float(variance) / 0.01, rel=1e-9)


def test_power_compression_fixes_the_anchor_and_scales_log_spread():
    torch.manual_seed(30)
    reference, floor = 0.4, 1e-6
    aleatoric = (
        floor + torch.distributions.LogNormal(math.log(reference), 1.5).sample((20000,)).double()
    )

    identity = power_compress_variance(
        aleatoric, gamma=1.0, reference=reference, variance_floor=floor
    )
    torch.testing.assert_close(identity, aleatoric)

    for gamma in (0.25, 0.5, 0.75):
        compressed = power_compress_variance(
            aleatoric, gamma=gamma, reference=reference, variance_floor=floor
        )
        anchor = power_compress_variance(
            torch.tensor([floor + reference], dtype=DOUBLE),
            gamma=gamma,
            reference=reference,
            variance_floor=floor,
        )
        assert float(anchor) == pytest.approx(floor + reference, rel=1e-12)
        spread = (compressed - floor).log().std() / (aleatoric - floor).log().std()
        assert float(spread) == pytest.approx(gamma, rel=1e-6)


def test_power_compression_preserves_every_ranking():
    torch.manual_seed(31)
    aleatoric = 1e-6 + torch.rand(500, 7, dtype=DOUBLE)
    compressed = power_compress_variance(aleatoric, gamma=0.4, reference=0.5)
    torch.testing.assert_close(compressed.flatten().argsort(), aleatoric.flatten().argsort())
    with pytest.raises(ValueError, match="gamma"):
        power_compress_variance(aleatoric, gamma=0.0, reference=0.5)
    with pytest.raises(ValueError, match="reference"):
        power_compress_variance(aleatoric, gamma=0.5, reference=0.0)


def test_replicate_log_variance_corrections_match_their_definitions():
    for repeats in (2, 4, 8, 16):
        half = 0.5 * (repeats - 1)
        assert replicate_log_variance_offset(repeats) == pytest.approx(
            math.log(half) - float(torch.special.digamma(torch.tensor(half)))
        )
        assert replicate_log_variance_noise(repeats) == pytest.approx(
            float(torch.special.polygamma(1, torch.tensor(half)))
        )
    with pytest.raises(ValueError, match="at least two"):
        replicate_log_variance_offset(1)


def test_offset_makes_the_log_sample_variance_unbiased():
    torch.manual_seed(32)
    repeats, truth = 8, 0.35
    draws = truth**0.5 * torch.randn(400_000, repeats, dtype=DOUBLE)
    sample = draws.var(dim=1, unbiased=True)
    corrected = sample.log() + replicate_log_variance_offset(repeats)
    assert float(corrected.mean()) == pytest.approx(math.log(truth), abs=5e-3)
    assert float(sample.log().var()) == pytest.approx(
        replicate_log_variance_noise(repeats), rel=2e-2
    )
