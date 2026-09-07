"""Tests for calibrated hierarchical-probit classification."""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi.hrc_probit import (
    LogTauPosterior,
    alpha_from_log_tau,
    fit_hrc_log_tau,
    fit_hrc_log_tau_laplace,
    hrc_branch_log_probabilities,
    hrc_class_probabilities,
    hrc_log_partition_deviation,
    hrc_log_probs,
    hrc_log_tau_score_curvature,
    hrc_negative_log_likelihood,
    hrc_path_log_scores,
    log_tau_from_alpha,
    probit_log_tau_factor_score_curvature,
    update_log_tau_gaussian,
)
from triton_tagi.hrc_softmax import (
    class_to_obs,
    class_to_obs_full,
    labels_to_hrc,
    labels_to_hrc_mask,
    obs_to_class_probs,
)
from triton_tagi.update.observation import (
    compute_innovation_with_indices,
    compute_probit_innovation_with_indices,
)


def random_moments(batch: int, hrc, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    mean = torch.randn(batch, hrc.len, generator=generator)
    variance = torch.rand(batch, hrc.len, generator=generator) * 2.0
    return mean, variance


# ──────────────────────────────────────────────────────────────────────────────
#  Full binary tree
# ──────────────────────────────────────────────────────────────────────────────


class TestFullTree:
    @pytest.mark.parametrize("num_classes", [2, 3, 5, 8, 10, 16, 17, 100])
    def test_has_exactly_num_classes_minus_one_nodes(self, num_classes):
        hrc = class_to_obs_full(num_classes)
        assert hrc.len == num_classes - 1
        assert hrc.is_full
        assert hrc.n_classes == num_classes

    def test_ten_classes_matches_expected_depth_profile(self):
        hrc = class_to_obs_full(10)
        depths = hrc.mask.sum(dim=1)
        assert hrc.len == 9
        assert hrc.n_obs == 4
        assert int((depths == 3).sum()) == 6
        assert int((depths == 4).sum()) == 4

    @pytest.mark.parametrize("num_classes", [3, 5, 10, 17])
    def test_depths_differ_by_at_most_one(self, num_classes):
        depths = class_to_obs_full(num_classes).mask.sum(dim=1)
        assert float(depths.max() - depths.min()) <= 1.0

    @pytest.mark.parametrize("num_classes", [2, 3, 5, 10, 17])
    def test_every_node_is_used_by_both_branches(self, num_classes):
        hrc = class_to_obs_full(num_classes)
        for node in range(1, hrc.len + 1):
            on_node = (hrc.idx == node) & (hrc.mask > 0)
            assert bool((hrc.obs[on_node] > 0).any())
            assert bool((hrc.obs[on_node] < 0).any())

    @pytest.mark.parametrize("num_classes", [2, 3, 5, 10, 17])
    def test_leaf_probabilities_sum_to_one_without_a_normalizer(self, num_classes):
        hrc = class_to_obs_full(num_classes)
        mean, variance = random_moments(16, hrc, seed=num_classes)
        assert hrc_log_partition_deviation(mean, variance, hrc) < 1e-10

    @pytest.mark.parametrize("num_classes", [3, 5, 10, 17])
    def test_prior_offsets_make_a_zero_output_uniform(self, num_classes):
        hrc = class_to_obs_full(num_classes)
        zeros = torch.zeros(1, hrc.len)
        probabilities = hrc_class_probabilities(zeros, zeros, hrc)
        torch.testing.assert_close(
            probabilities,
            torch.full_like(probabilities, 1.0 / num_classes),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_prior_offsets_reproduce_a_nonuniform_class_prior(self):
        priors = torch.tensor([0.5, 0.25, 0.15, 0.1])
        hrc = class_to_obs_full(4, class_priors=priors)
        zeros = torch.zeros(1, hrc.len)
        probabilities = hrc_class_probabilities(zeros, zeros, hrc)
        torch.testing.assert_close(probabilities.squeeze(0), priors, rtol=1e-5, atol=1e-6)

    def test_disabling_offsets_keeps_normalization_but_loses_uniformity(self):
        hrc = class_to_obs_full(10, use_prior_offsets=False)
        assert hrc.offset is not None
        assert bool((hrc.offset == 0).all())
        zeros = torch.zeros(1, hrc.len)
        assert hrc_log_partition_deviation(zeros, zeros, hrc) < 1e-10
        probabilities = hrc_class_probabilities(zeros, zeros, hrc).squeeze(0)
        # Depth three leaves get 1/8, depth four leaves get 1/16.
        assert float(probabilities.max()) == pytest.approx(0.125, rel=1e-4)
        assert float(probabilities.min()) == pytest.approx(0.0625, rel=1e-4)

    def test_larger_latent_scale_falls_back_to_the_class_prior(self):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(4, hrc, seed=3)
        probabilities = hrc_class_probabilities(mean, variance, hrc, log_tau=9.0)
        torch.testing.assert_close(
            probabilities,
            torch.full_like(probabilities, 0.1),
            rtol=1e-3,
            atol=1e-4,
        )

    def test_rejects_invalid_class_counts_and_priors(self):
        with pytest.raises(ValueError):
            class_to_obs_full(1)
        with pytest.raises(ValueError):
            class_to_obs_full(4, class_priors=[1.0, 1.0])
        with pytest.raises(ValueError):
            class_to_obs_full(3, class_priors=[0.0, 0.0, 0.0])


# ──────────────────────────────────────────────────────────────────────────────
#  Normalization of the padded tree
# ──────────────────────────────────────────────────────────────────────────────


class TestPaddedTreeNormalization:
    def test_uninformative_padded_tree_loses_the_discarded_leaf_mass(self):
        hrc = class_to_obs(10)
        zeros = torch.zeros(1, hrc.len)
        raw = hrc_path_log_scores(zeros, zeros, hrc).exp()
        assert float(raw.sum()) == pytest.approx(10.0 / 16.0, rel=1e-6)

    def test_normalized_padded_tree_is_a_categorical_distribution(self):
        hrc = class_to_obs(10)
        mean, variance = random_moments(32, hrc, seed=1)
        probabilities = hrc_class_probabilities(mean, variance, hrc)
        torch.testing.assert_close(
            probabilities.sum(dim=1),
            torch.ones(32),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_padded_scores_match_the_legacy_alpha_parameterization(self):
        hrc = class_to_obs(10)
        mean, variance = random_moments(8, hrc, seed=2)
        legacy = obs_to_class_probs(mean, variance, hrc, alpha=3.0)
        scores = hrc_path_log_scores(mean, variance, hrc, log_tau=log_tau_from_alpha(3.0)).exp()
        torch.testing.assert_close(scores.float(), legacy, rtol=1e-5, atol=1e-7)

    def test_alpha_and_log_tau_are_inverse_parameterizations(self):
        assert alpha_from_log_tau(log_tau_from_alpha(3.0)) == pytest.approx(3.0)
        assert log_tau_from_alpha(1.0) == 0.0
        with pytest.raises(ValueError):
            log_tau_from_alpha(0.0)

    def test_power_of_two_padded_tree_needs_no_normalizer(self):
        hrc = class_to_obs(8)
        mean, variance = random_moments(8, hrc, seed=4)
        assert hrc.len == 7
        assert hrc_log_partition_deviation(mean, variance, hrc) < 1e-10


# ──────────────────────────────────────────────────────────────────────────────
#  Likelihood
# ──────────────────────────────────────────────────────────────────────────────


class TestLikelihood:
    @pytest.mark.parametrize("builder", [class_to_obs, class_to_obs_full])
    def test_negative_log_likelihood_is_categorical_cross_entropy(self, builder):
        hrc = builder(10)
        mean, variance = random_moments(64, hrc, seed=5)
        labels = torch.arange(64) % 10
        log_probabilities = hrc_log_probs(mean, variance, hrc)
        expected = torch.nn.functional.nll_loss(log_probabilities, labels)
        torch.testing.assert_close(
            hrc_negative_log_likelihood(mean, variance, labels, hrc), expected
        )

    def test_branch_log_probabilities_are_complementary(self):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(8, hrc, seed=6)
        log_left, log_right = hrc_branch_log_probabilities(mean, variance, hrc)
        total = torch.logsumexp(torch.stack((log_left, log_right)), dim=0)
        torch.testing.assert_close(total, torch.zeros_like(total), atol=1e-12, rtol=0)

    def test_rejects_malformed_moments_and_labels(self):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(4, hrc, seed=7)
        with pytest.raises(ValueError):
            hrc_log_probs(mean[:, :-1], variance[:, :-1], hrc)
        with pytest.raises(ValueError):
            hrc_log_probs(mean, -variance, hrc)
        with pytest.raises(ValueError):
            hrc_negative_log_likelihood(mean, variance, torch.zeros(3).long(), hrc)
        with pytest.raises(ValueError):
            hrc_negative_log_likelihood(mean, variance, torch.full((4,), 10), hrc)


# ──────────────────────────────────────────────────────────────────────────────
#  Post-hoc scalar calibration
# ──────────────────────────────────────────────────────────────────────────────


def planted_calibration_problem(
    log_tau: float,
    num_samples: int = 4096,
    seed: int = 0,
    num_classes: int = 10,
):
    """Sample labels from the hierarchical model at a known latent scale."""

    hrc = class_to_obs_full(num_classes)
    generator = torch.Generator().manual_seed(seed)
    mean = torch.randn(num_samples, hrc.len, generator=generator) * 1.5
    variance = torch.rand(num_samples, hrc.len, generator=generator) * 0.5
    probabilities = hrc_class_probabilities(mean, variance, hrc, log_tau=log_tau)
    labels = torch.multinomial(probabilities.double(), 1, generator=generator).squeeze(1)
    return hrc, mean, variance, labels


class TestCalibration:
    @pytest.mark.parametrize("planted", [-0.7, 0.0, 1.0])
    def test_recovers_a_planted_latent_scale(self, planted):
        hrc, mean, variance, labels = planted_calibration_problem(planted, seed=11)
        fitted = fit_hrc_log_tau(mean, variance, labels, hrc)
        assert fitted == pytest.approx(planted, abs=0.15)

    def test_calibration_does_not_increase_validation_nll(self):
        hrc, mean, variance, labels = planted_calibration_problem(1.2, seed=12)
        before = float(hrc_negative_log_likelihood(mean, variance, labels, hrc))
        fitted = fit_hrc_log_tau(mean, variance, labels, hrc)
        after = float(hrc_negative_log_likelihood(mean, variance, labels, hrc, log_tau=fitted))
        assert after <= before + 1e-9

    def test_calibrating_the_padded_tree_improves_the_default_scale(self):
        hrc = class_to_obs(10)
        generator = torch.Generator().manual_seed(13)
        mean = torch.randn(2048, hrc.len, generator=generator) * 2.0
        variance = torch.rand(2048, hrc.len, generator=generator) * 0.2
        probabilities = hrc_class_probabilities(mean, variance, hrc, log_tau=0.5)
        labels = torch.multinomial(probabilities.double(), 1, generator=generator).squeeze(1)
        default = float(
            hrc_negative_log_likelihood(
                mean, variance, labels, hrc, log_tau=log_tau_from_alpha(3.0)
            )
        )
        fitted = fit_hrc_log_tau(mean, variance, labels, hrc)
        calibrated = float(hrc_negative_log_likelihood(mean, variance, labels, hrc, log_tau=fitted))
        assert calibrated < default

    def test_rejects_invalid_search_configuration(self):
        hrc, mean, variance, labels = planted_calibration_problem(0.0, 32, seed=14)
        with pytest.raises(ValueError):
            fit_hrc_log_tau(mean, variance, labels, hrc, log_tau_bounds=(1.0, -1.0))
        with pytest.raises(ValueError):
            fit_hrc_log_tau(mean, variance, labels, hrc, grid_size=2)


# ──────────────────────────────────────────────────────────────────────────────
#  Score, curvature, and the Bayesian scale
# ──────────────────────────────────────────────────────────────────────────────


class TestScoreAndCurvature:
    @pytest.mark.parametrize("builder", [class_to_obs, class_to_obs_full])
    def test_matches_central_finite_differences(self, builder):
        hrc = builder(10)
        mean, variance = random_moments(128, hrc, seed=15)
        labels = torch.arange(128) % 10
        center = 0.3
        step = 1e-4

        def total(value: float) -> float:
            return (
                -float(hrc_negative_log_likelihood(mean, variance, labels, hrc, log_tau=value))
                * mean.shape[0]
            )

        score, curvature = hrc_log_tau_score_curvature(mean, variance, labels, hrc, log_tau=center)
        numeric_score = (total(center + step) - total(center - step)) / (2.0 * step)
        numeric_curvature = (
            total(center + step) - 2.0 * total(center) + total(center - step)
        ) / step**2
        assert score == pytest.approx(numeric_score, rel=1e-4, abs=1e-4)
        assert curvature == pytest.approx(numeric_curvature, rel=1e-2, abs=1e-2)

    def test_closed_form_factor_derivatives_match_autodiff(self):
        # A single node with a zero offset isolates the closed form from both
        # the branch prior and the categorical normalizer.
        hrc = class_to_obs_full(2, use_prior_offsets=False)
        generator = torch.Generator().manual_seed(16)
        mean = torch.randn(256, 1, generator=generator)
        variance = torch.rand(256, 1, generator=generator) * 3.0
        labels = torch.randint(0, 2, (256,), generator=generator)
        log_tau = -0.4
        tau2 = math.exp(2.0 * log_tau)

        signs = torch.where(labels == 0, 1.0, -1.0).unsqueeze(1).double()
        gamma = signs * mean.double() / torch.sqrt(variance.double() + tau2)
        rho = torch.full_like(gamma, tau2) / (variance.double() + tau2)
        score, curvature = probit_log_tau_factor_score_curvature(gamma, rho)

        expected_score, expected_curvature = hrc_log_tau_score_curvature(
            mean, variance, labels, hrc, log_tau=log_tau, normalize=False
        )
        assert float(score.sum()) == pytest.approx(expected_score, rel=1e-8, abs=1e-8)
        assert float(curvature.sum()) == pytest.approx(expected_curvature, rel=1e-8, abs=1e-8)

    def test_score_vanishes_at_the_fitted_scale(self):
        hrc, mean, variance, labels = planted_calibration_problem(0.4, seed=17)
        fitted = fit_hrc_log_tau(mean, variance, labels, hrc)
        score, curvature = hrc_log_tau_score_curvature(mean, variance, labels, hrc, log_tau=fitted)
        assert abs(score) < 1e-2 * mean.shape[0]
        assert curvature < 0.0


class TestBayesianScale:
    def test_laplace_posterior_concentrates_near_the_map(self):
        hrc, mean, variance, labels = planted_calibration_problem(0.6, seed=18)
        fitted = fit_hrc_log_tau(mean, variance, labels, hrc)
        posterior = fit_hrc_log_tau_laplace(mean, variance, labels, hrc, prior_variance=4.0)
        assert posterior.mean == pytest.approx(fitted, abs=0.05)
        assert 0.0 < posterior.variance <= 4.0
        assert posterior.tau == pytest.approx(math.exp(posterior.mean))
        assert posterior.alpha_impl == pytest.approx(1.0 / posterior.tau)

    def test_posterior_variance_shrinks_with_more_observations(self):
        small = planted_calibration_problem(0.3, num_samples=256, seed=19)
        large = planted_calibration_problem(0.3, num_samples=4096, seed=19)
        narrow = fit_hrc_log_tau_laplace(*small[1:3], small[3], small[0])
        wide = fit_hrc_log_tau_laplace(*large[1:3], large[3], large[0])
        assert wide.variance < narrow.variance

    def test_tight_prior_dominates_a_small_sample(self):
        hrc, mean, variance, labels = planted_calibration_problem(2.0, num_samples=64, seed=20)
        posterior = fit_hrc_log_tau_laplace(
            mean, variance, labels, hrc, prior_mean=0.0, prior_variance=1e-4
        )
        assert abs(posterior.mean) < 0.1

    def test_online_update_moves_the_mean_along_the_score(self):
        prior = LogTauPosterior(mean=0.0, variance=0.25)
        posterior = update_log_tau_gaussian(prior, score=2.0, curvature=-4.0)
        assert posterior.mean > prior.mean
        assert 0.0 < posterior.variance < prior.variance
        assert update_log_tau_gaussian(prior, score=-2.0, curvature=-4.0).mean < 0.0

    def test_online_update_caps_the_step_and_ignores_positive_curvature(self):
        prior = LogTauPosterior(mean=0.0, variance=1.0)
        capped = update_log_tau_gaussian(prior, score=1e6, curvature=-1.0, max_mean_step=0.5)
        assert capped.mean == pytest.approx(0.5)
        convex = update_log_tau_gaussian(prior, score=0.0, curvature=5.0)
        assert convex.variance == pytest.approx(prior.variance)

    def test_online_update_validates_its_arguments(self):
        prior = LogTauPosterior(mean=0.0, variance=1.0)
        with pytest.raises(ValueError):
            update_log_tau_gaussian(prior, 1.0, -1.0, damping=0.0)
        with pytest.raises(ValueError):
            update_log_tau_gaussian(prior, float("nan"), -1.0)
        with pytest.raises(ValueError):
            update_log_tau_gaussian(prior, 1.0, -1.0, variance_floor=0.0)


# ──────────────────────────────────────────────────────────────────────────────
#  Sparse observation updates on a variable-depth tree
# ──────────────────────────────────────────────────────────────────────────────


class TestMaskedProbitInnovation:
    def test_all_ones_mask_matches_the_unmasked_call(self):
        hrc = class_to_obs(10)
        mean, variance = random_moments(16, hrc, seed=21)
        labels = torch.arange(16) % 10
        y_obs, y_idx = labels_to_hrc(labels, hrc)
        var_obs = torch.ones_like(y_obs)
        plain = compute_probit_innovation_with_indices(mean, variance, y_obs, var_obs, y_idx)
        masked = compute_probit_innovation_with_indices(
            mean, variance, y_obs, var_obs, y_idx, mask=torch.ones_like(y_obs)
        )
        torch.testing.assert_close(plain[0], masked[0])
        torch.testing.assert_close(plain[1], masked[1])

    def test_padded_path_entries_receive_no_innovation(self):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(10, hrc, seed=22)
        labels = torch.arange(10)
        y_obs, y_idx = labels_to_hrc(labels, hrc)
        mask = labels_to_hrc_mask(labels, hrc)
        delta_mu, delta_var = compute_probit_innovation_with_indices(
            mean, variance, y_obs, torch.ones_like(y_obs), y_idx, mask=mask
        )
        touched = torch.zeros_like(mean, dtype=torch.bool)
        for row in range(mean.shape[0]):
            for position in range(hrc.n_obs):
                if float(mask[row, position]) > 0.0:
                    touched[row, int(y_idx[row, position]) - 1] = True
        assert bool((delta_mu[~touched] == 0).all())
        assert bool((delta_var[~touched] == 0).all())
        # A depth-three class must leave exactly three nodes updated.
        depth_three = int(mask[0].sum())
        assert int((delta_var[0] != 0).sum()) == depth_three

    def test_latent_shift_matches_a_manual_probit_moment(self):
        hrc = class_to_obs_full(4)
        mean, variance = random_moments(6, hrc, seed=23)
        labels = torch.arange(6) % 4
        y_obs, y_idx = labels_to_hrc(labels, hrc)
        mask = labels_to_hrc_mask(labels, hrc)
        offset = hrc.node_offset()[y_idx.long() - 1]
        tau = 0.8
        delta_mu, _ = compute_probit_innovation_with_indices(
            mean,
            variance,
            y_obs,
            torch.full_like(y_obs, tau**2),
            y_idx,
            mask=mask,
            latent_shift=tau * offset,
        )
        node = int(y_idx[0, 0]) - 1
        scale = math.sqrt(float(variance[0, node]) + tau**2)
        gamma = float(y_obs[0, 0]) * (float(mean[0, node]) + tau * float(offset[0, 0])) / scale
        expected = (
            float(y_obs[0, 0])
            * math.exp(-0.5 * gamma**2)
            / math.sqrt(2.0 * math.pi)
            / float(torch.special.ndtr(torch.tensor(gamma)))
            / scale
        )
        assert float(delta_mu[0, node]) == pytest.approx(expected, rel=1e-5)

    def test_gaussian_innovation_honours_the_mask(self):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(10, hrc, seed=24)
        labels = torch.arange(10)
        y_obs, y_idx = labels_to_hrc(labels, hrc)
        mask = labels_to_hrc_mask(labels, hrc)
        delta_mu, delta_var = compute_innovation_with_indices(
            mean, variance, y_obs, torch.full_like(y_obs, 0.01), y_idx, mask=mask
        )
        assert int((delta_var[0] != 0).sum()) == int(mask[0].sum())

    def test_rejects_a_mismatched_mask_or_shift(self):
        hrc = class_to_obs(10)
        mean, variance = random_moments(4, hrc, seed=25)
        labels = torch.arange(4)
        y_obs, y_idx = labels_to_hrc(labels, hrc)
        with pytest.raises(ValueError):
            compute_probit_innovation_with_indices(
                mean, variance, y_obs, torch.ones_like(y_obs), y_idx, mask=torch.ones(4, 1)
            )
        with pytest.raises(ValueError):
            compute_probit_innovation_with_indices(
                mean,
                variance,
                y_obs,
                torch.ones_like(y_obs),
                y_idx,
                latent_shift=torch.zeros(4, 1),
            )

    def test_padded_tree_has_no_mask(self):
        assert labels_to_hrc_mask(torch.arange(4), class_to_obs(10)) is None


def test_score_and_curvature_survive_an_ambient_no_grad():
    hrc, mean, variance, labels = planted_calibration_problem(0.2, 256, seed=26)
    with torch.no_grad():
        score, curvature = hrc_log_tau_score_curvature(mean, variance, labels, hrc, log_tau=0.2)
        posterior = fit_hrc_log_tau_laplace(mean, variance, labels, hrc)
    assert math.isfinite(score) and math.isfinite(curvature)
    assert math.isfinite(posterior.mean) and posterior.variance > 0.0


class TestFixedDepthGuards:
    def test_legacy_helpers_reject_a_variable_depth_tree(self):
        from triton_tagi.hrc_softmax import obs_to_class_probs_probit

        hrc = class_to_obs_full(10)
        mean, variance = random_moments(4, hrc, seed=27)
        with pytest.raises(ValueError, match="variable-depth"):
            obs_to_class_probs(mean, variance, hrc)
        with pytest.raises(ValueError, match="variable-depth"):
            obs_to_class_probs_probit(mean, variance, hrc)

    def test_legacy_helpers_reject_branch_priors(self):
        # Four classes give a fixed-depth tree, so only the nonzero offsets
        # from the skewed class prior can trip the guard.
        hrc = class_to_obs_full(4, class_priors=[0.5, 0.2, 0.2, 0.1])
        mean, variance = random_moments(4, hrc, seed=28)
        assert hrc.mask is not None and bool((hrc.mask == 1).all())
        assert bool((hrc.offset != 0).any())
        with pytest.raises(ValueError, match="branch priors"):
            obs_to_class_probs(mean, variance, hrc)

    def test_legacy_helpers_accept_an_offset_free_fixed_depth_full_tree(self):
        hrc = class_to_obs_full(8, use_prior_offsets=False)
        mean, variance = random_moments(4, hrc, seed=29)
        legacy = obs_to_class_probs(mean, variance, hrc, alpha=3.0)
        scores = hrc_path_log_scores(mean, variance, hrc, log_tau=log_tau_from_alpha(3.0)).exp()
        torch.testing.assert_close(scores.float(), legacy, rtol=1e-5, atol=1e-7)


# ──────────────────────────────────────────────────────────────────────────────
#  The proposed model: unit probit noise, full tree, no normalizer
# ──────────────────────────────────────────────────────────────────────────────


class TestUnitProbitModel:
    @pytest.mark.parametrize("num_classes", [2, 3, 5, 8, 10, 17, 100])
    def test_class_probabilities_sum_to_one_without_a_normalizer(self, num_classes):
        hrc = class_to_obs_full(num_classes)
        mean, variance = random_moments(32, hrc, seed=num_classes + 40)
        unnormalized = hrc_log_probs(mean, variance, hrc, normalize=False).exp().sum(dim=1)
        torch.testing.assert_close(unnormalized, torch.ones_like(unnormalized), rtol=0, atol=1e-12)

    def test_default_normalization_follows_the_tree(self):
        full = class_to_obs_full(10)
        padded = class_to_obs(10)
        mean, variance = random_moments(8, full, seed=41)
        # A full tree is left alone; the resolved default equals normalize=False.
        torch.testing.assert_close(
            hrc_log_probs(mean, variance, full),
            hrc_log_probs(mean, variance, full, normalize=False),
        )
        padded_mean, padded_variance = random_moments(8, padded, seed=41)
        # The padded tree is normalized, and that changes the values.
        torch.testing.assert_close(
            hrc_log_probs(padded_mean, padded_variance, padded),
            hrc_log_probs(padded_mean, padded_variance, padded, normalize=True),
        )
        assert not torch.allclose(
            hrc_log_probs(padded_mean, padded_variance, padded),
            hrc_log_probs(padded_mean, padded_variance, padded, normalize=False),
        )

    @pytest.mark.parametrize("num_classes", [2, 4, 8, 16])
    def test_power_of_two_trees_have_no_branch_offsets(self, num_classes):
        hrc = class_to_obs_full(num_classes)
        assert bool((hrc.node_offset() == 0).all())

    def test_offsets_are_the_left_subtree_prior_probit(self):
        hrc = class_to_obs_full(10)
        # Ten classes split 5/5, then 2/3, then 1/1 and 1/2.
        expected = {
            0.0,
            float(torch.special.ndtri(torch.tensor(0.4))),
            float(torch.special.ndtri(torch.tensor(1.0 / 3.0))),
        }
        for value in hrc.node_offset().tolist():
            assert min(abs(value - item) for item in expected) < 1e-6

    def test_the_model_has_no_free_scale_in_its_training_interface(self):
        import inspect

        from triton_tagi.network import Sequential

        parameters = inspect.signature(Sequential.step_hrc_probit).parameters
        assert set(parameters) == {"self", "x_batch", "labels", "hrc"}

    def test_scale_invariance_motivating_the_unit_convention(self):
        # (mu, S, tau) -> (a mu, a^2 S, a tau) leaves every probability
        # unchanged, which is why tau must be fixed rather than fitted. The
        # scaling is done in float64 so the check measures the model, not the
        # rounding of a float32 product.
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(8, hrc, seed=42)
        mean, variance = mean.double(), variance.double()
        scale = 2.5
        torch.testing.assert_close(
            hrc_log_probs(mean, variance, hrc, log_tau=0.0),
            hrc_log_probs(scale * mean, scale**2 * variance, hrc, log_tau=math.log(scale)),
            rtol=1e-12,
            atol=1e-12,
        )
