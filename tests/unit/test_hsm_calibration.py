"""Tests for hierarchical probit calibration with a positive uncertain gain."""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi.hrc_probit import hrc_log_probs
from triton_tagi.hrc_softmax import class_to_obs, class_to_obs_full
from triton_tagi.hsm_calibration import (
    DEFAULT_GAIN_ORDER,
    GainGroups,
    HsmClassMoments,
    LogGainPosterior,
    _hermite_rule,
    branch_codes,
    calibrate_hsm_log_gain_adf,
    custom_gain_groups,
    expected_bernoulli_variance,
    fit_hsm_log_gain,
    gain_groups,
    hsm_adf_projection,
    hsm_calibration_visits,
    hsm_class_moments,
    hsm_class_probabilities,
    hsm_cross_class_covariance,
    hsm_log_posterior_on_grid,
    hsm_negative_log_likelihood,
    hsm_node_moments,
    hsm_partition_deviation,
    node_depths,
    owens_t,
    probit_gaussian_moments,
)

# Section "Reproducible numerical check": Z ~ N(0.5, 0.5^2) with a lognormal
# gain of mean 1.5 and standard deviation 0.3.
CHECK_LOG_GAIN_VARIANCE = math.log(1.0 + 0.3**2 / 1.5**2)
CHECK_LOG_GAIN_MEAN = math.log(1.5) - 0.5 * CHECK_LOG_GAIN_VARIANCE


def random_moments(batch: int, hrc, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    mean = torch.randn(batch, hrc.len, generator=generator, dtype=torch.float64)
    variance = torch.rand(batch, hrc.len, generator=generator, dtype=torch.float64) * 2.0
    return mean, variance


def single_node_check() -> tuple[object, LogGainPosterior, torch.Tensor, torch.Tensor]:
    """Return the paper's one-node numerical check as a tree and a belief."""

    hrc = class_to_obs_full(2, use_prior_offsets=False)
    posterior = LogGainPosterior(
        mean=torch.tensor([CHECK_LOG_GAIN_MEAN], dtype=torch.float64),
        variance=torch.tensor([CHECK_LOG_GAIN_VARIANCE], dtype=torch.float64),
        groups=gain_groups(hrc, "global"),
    )
    mean = torch.tensor([[0.5]], dtype=torch.float64)
    variance = torch.tensor([[0.25]], dtype=torch.float64)
    return hrc, posterior, mean, variance


def sample_leaf_labels(hrc, mean, variance, log_gain: float, seed: int) -> torch.Tensor:
    """Draw leaf labels from the generative probit tree at a known gain."""

    generator = torch.Generator().manual_seed(seed)
    codes = branch_codes(hrc)
    state = mean + variance.sqrt() * torch.randn(
        mean.shape, generator=generator, dtype=torch.float64
    )
    noise = torch.randn(mean.shape, generator=generator, dtype=torch.float64)
    latent = math.exp(log_gain) * state + hrc.node_offset().double() + noise
    taken = torch.where(latent >= 0.0, 1.0, -1.0)
    on_path = codes.abs()
    agrees = ((codes[None] * taken[:, None, :]) > 0.0).double() * on_path[None]
    return (agrees + (1.0 - on_path[None])).prod(-1).argmax(-1)


# ──────────────────────────────────────────────────────────────────────────────
#  Owen's T and the exact moments of Phi(X)
# ──────────────────────────────────────────────────────────────────────────────


class TestOwensT:
    def test_vanishes_at_zero_second_argument(self):
        assert float(owens_t(1.3, 0.0)) == 0.0

    def test_is_even_in_the_first_argument(self):
        assert float(owens_t(2.1, 0.6)) == pytest.approx(float(owens_t(-2.1, 0.6)), abs=1e-18)

    def test_is_odd_in_the_second_argument(self):
        assert float(owens_t(0.7, -0.4)) == pytest.approx(-float(owens_t(0.7, 0.4)), abs=1e-18)

    def test_matches_the_closed_form_at_a_zero_first_argument(self):
        # T(0, c) = arctan(c) / (2 pi).
        for value in (0.25, 0.5, 1.0):
            assert float(owens_t(0.0, value)) == pytest.approx(
                math.atan(value) / (2.0 * math.pi), rel=1e-14
            )

    def test_reproduces_the_equal_argument_variance_identity(self):
        mean = torch.tensor([-2.0, -0.3, 0.0, 0.8, 3.0], dtype=torch.float64)
        variance = torch.tensor([4.0, 0.1, 1.0, 0.5, 2.0], dtype=torch.float64)
        first, second = probit_gaussian_moments(mean, variance)
        argument = mean / (1.0 + variance).sqrt()
        upper = 1.0 / (1.0 + 2.0 * variance).sqrt()
        identity = first * (1.0 - first) - 2.0 * owens_t(argument, upper)
        assert torch.allclose(second, identity, atol=1e-15)

    @pytest.mark.parametrize("order", [24, 48, 192])
    def test_is_stable_in_the_quadrature_order(self, order):
        reference = float(owens_t(0.59, 0.8, order=384))
        assert float(owens_t(0.59, 0.8, order=order)) == pytest.approx(reference, abs=1e-15)


class TestProbitMoments:
    def test_mean_is_the_scaled_normal_cdf(self):
        mean = torch.tensor([0.5, -1.0], dtype=torch.float64)
        variance = torch.tensor([0.25, 3.0], dtype=torch.float64)
        first, _ = probit_gaussian_moments(mean, variance)
        expected = torch.special.ndtr(mean / (1.0 + variance).sqrt())
        assert torch.allclose(first, expected, atol=1e-15)

    def test_variance_vanishes_for_a_deterministic_argument(self):
        _, second = probit_gaussian_moments(
            torch.tensor([1.5, -0.2], dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
        )
        assert torch.allclose(second, torch.zeros(2, dtype=torch.float64), atol=1e-15)

    def test_variance_approaches_one_quarter_as_the_argument_spreads(self):
        _, second = probit_gaussian_moments(
            torch.zeros(1, dtype=torch.float64), torch.full((1,), 1e8, dtype=torch.float64)
        )
        assert float(second) == pytest.approx(0.25, abs=1e-4)

    def test_variance_never_exceeds_the_bernoulli_bound(self):
        mean, variance = random_moments(64, class_to_obs_full(4), seed=3)
        first, second = probit_gaussian_moments(mean, variance)
        assert bool((second >= 0.0).all())
        assert bool((second <= first * (1.0 - first) + 1e-18).all())

    def test_total_variance_decomposition_is_exact(self):
        mean, variance = random_moments(32, class_to_obs_full(4), seed=4)
        first, second = probit_gaussian_moments(mean, variance)
        residual = expected_bernoulli_variance(first, second)
        assert torch.allclose(residual + second, first * (1.0 - first), atol=1e-15)


# ──────────────────────────────────────────────────────────────────────────────
#  The paper's reproducible numerical check
# ──────────────────────────────────────────────────────────────────────────────


class TestReproducibleCheck:
    def test_log_gain_parameters_match_the_stated_lognormal(self):
        assert CHECK_LOG_GAIN_VARIANCE == pytest.approx(0.0392207, abs=5e-8)
        assert CHECK_LOG_GAIN_MEAN == pytest.approx(0.385855, abs=5e-7)

    def test_node_moments_match_the_published_values(self):
        hrc, posterior, mean, variance = single_node_check()
        moments = hsm_node_moments(mean, variance, hrc, posterior, order=20)
        assert float(moments.mean) == pytest.approx(0.722540, abs=5e-7)
        assert math.sqrt(float(moments.variance)) == pytest.approx(0.207525, abs=5e-7)
        assert float(moments.cov_state) == pytest.approx(0.098390, abs=5e-7)
        assert float(moments.cov_log_gain) == pytest.approx(0.00491532, abs=5e-9)

    def test_quadrature_order_twelve_to_twenty_agrees_to_double_precision(self):
        hrc, posterior, mean, variance = single_node_check()
        coarse = hsm_node_moments(mean, variance, hrc, posterior, order=12)
        fine = hsm_node_moments(mean, variance, hrc, posterior, order=20)
        for first, second in zip(
            (coarse.mean, coarse.variance, coarse.cov_state, coarse.cov_log_gain),
            (fine.mean, fine.variance, fine.cov_state, fine.cov_log_gain),
        ):
            assert float((first - second).abs().max()) < 2e-15

    def test_one_sign_oriented_observation_matches_the_published_update(self):
        hrc, posterior, mean, variance = single_node_check()
        moments = hsm_node_moments(mean, variance, hrc, posterior, order=20)
        updated_mean, updated_variance = hsm_adf_projection(
            CHECK_LOG_GAIN_MEAN,
            CHECK_LOG_GAIN_VARIANCE,
            float(moments.mean),
            float(moments.cov_log_gain),
        )
        assert updated_mean == pytest.approx(0.392658, abs=5e-7)
        assert math.sqrt(updated_variance) == pytest.approx(0.197738, abs=5e-7)

    def test_covariance_matches_the_analytic_log_gain_score(self):
        # Cov(L, P) = q E[m'(L)] with m'(l) = s (mu - o exp(l) v) exp(l) phi(a) / d^3.
        hrc, posterior, mean, variance = single_node_check()
        moments = hsm_node_moments(mean, variance, hrc, posterior, order=20)
        nodes, weights = _hermite_rule(20)
        log_gain = CHECK_LOG_GAIN_MEAN + math.sqrt(CHECK_LOG_GAIN_VARIANCE) * nodes
        gain = log_gain.exp()
        scale = (1.0 + gain.square() * 0.25).sqrt()
        argument = gain * 0.5 / scale
        density = torch.exp(-0.5 * argument.square()) / math.sqrt(2.0 * math.pi)
        derivative = 0.5 * gain / scale.pow(3) * density
        analytic = CHECK_LOG_GAIN_VARIANCE * float((weights * derivative).sum())
        assert float(moments.cov_log_gain) == pytest.approx(analytic, rel=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
#  Limiting cases listed as implementation tests
# ──────────────────────────────────────────────────────────────────────────────


class TestLimitingCases:
    @pytest.mark.parametrize("log_gain", [0.0, -1.0986122886681098, 0.7])
    def test_zero_gain_variance_recovers_the_deterministic_probit_head(self, log_gain):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(48, hrc, seed=5)
        posterior = LogGainPosterior.prior(
            gain_groups(hrc, "global"), mean=log_gain, variance=0.0
        )
        calibrated = hsm_class_moments(mean, variance, hrc, posterior).mean
        # The gain is the reciprocal of the latent probit scale: G = 1 / tau.
        reference = hrc_log_probs(mean, variance, hrc, log_tau=-log_gain).exp()
        assert torch.allclose(calibrated, reference, atol=1e-14)

    def test_a_vanishing_gain_gives_an_uninformative_node(self):
        hrc = class_to_obs_full(2, use_prior_offsets=False)
        posterior = LogGainPosterior.prior(gain_groups(hrc, "global"), mean=-25.0, variance=0.5)
        mean = torch.tensor([[1.7]], dtype=torch.float64)
        variance = torch.tensor([[0.9]], dtype=torch.float64)
        moments = hsm_node_moments(mean, variance, hrc, posterior)
        assert float(moments.mean) == pytest.approx(0.5, abs=1e-9)
        assert float(moments.variance) == pytest.approx(0.0, abs=1e-15)
        # Both covariances carry a factor exp(L), so they vanish at its rate.
        assert abs(float(moments.cov_state)) < math.exp(-25.0)
        assert abs(float(moments.cov_log_gain)) < math.exp(-25.0)

    def test_a_certain_state_still_leaves_gain_dispersion(self):
        hrc = class_to_obs_full(2, use_prior_offsets=False)
        posterior = LogGainPosterior.prior(gain_groups(hrc, "global"), mean=0.0, variance=0.4)
        mean = torch.tensor([[1.2]], dtype=torch.float64)
        variance = torch.zeros(1, 1, dtype=torch.float64)
        moments = hsm_node_moments(mean, variance, hrc, posterior)
        assert float(moments.cov_state) == 0.0
        assert float(moments.variance) > 1e-4

    def test_reflecting_the_state_complements_the_branch_probability(self):
        # Reflecting Z is the same operation as reversing the branch sign at a
        # zero offset, so the mean is complemented and the variance is
        # unchanged. Cov(Z, U) keeps its sign because both factors flipped;
        # Cov(L, U) changes sign because only one did.
        hrc = class_to_obs_full(2, use_prior_offsets=False)
        posterior = LogGainPosterior.prior(gain_groups(hrc, "global"), mean=0.2, variance=0.3)
        mean = torch.tensor([[0.6]], dtype=torch.float64)
        variance = torch.tensor([[0.7]], dtype=torch.float64)
        left = hsm_node_moments(mean, variance, hrc, posterior)
        right = hsm_node_moments(-mean, variance, hrc, posterior)
        assert float(right.mean) == pytest.approx(1.0 - float(left.mean), abs=1e-15)
        assert float(right.variance) == pytest.approx(float(left.variance), abs=1e-15)
        assert float(right.cov_state) == pytest.approx(float(left.cov_state), abs=1e-15)
        assert float(right.cov_log_gain) == pytest.approx(-float(left.cov_log_gain), abs=1e-16)

    @pytest.mark.parametrize("num_classes", [2, 3, 5, 8, 10, 17])
    @pytest.mark.parametrize("sharing", ["global", "level", "node"])
    def test_class_probabilities_sum_to_one_pointwise(self, num_classes, sharing):
        hrc = class_to_obs_full(num_classes)
        mean, variance = random_moments(32, hrc, seed=6)
        posterior = LogGainPosterior.prior(
            gain_groups(hrc, sharing), mean=0.3, variance=0.15
        )
        deviation = hsm_partition_deviation(mean, variance, hrc, posterior)
        assert deviation < 1e-12


# ──────────────────────────────────────────────────────────────────────────────
#  Gain-sharing groups
# ──────────────────────────────────────────────────────────────────────────────


class TestGainGroups:
    def test_node_depths_follow_the_recursive_split(self):
        depths = node_depths(class_to_obs_full(10))
        assert depths.shape == (9,)
        assert int(depths[0]) == 0
        assert int(depths.max()) == 3

    def test_global_sharing_repeats_within_a_path_beyond_a_root_only_tree(self):
        groups = gain_groups(class_to_obs_full(10), "global")
        assert groups.n_groups == 1
        assert groups.is_global
        assert groups.path_repeats
        assert groups.path_group_count == 1

    @pytest.mark.parametrize("sharing", ["level", "node"])
    def test_level_and_node_sharing_are_independent_along_a_path(self, sharing):
        groups = gain_groups(class_to_obs_full(10), sharing)
        assert not groups.path_repeats

    def test_level_sharing_has_one_group_per_depth(self):
        hrc = class_to_obs_full(10)
        groups = gain_groups(hrc, "level")
        assert groups.n_groups == int(node_depths(hrc).max()) + 1

    def test_expand_broadcasts_one_value_per_group_to_every_node(self):
        groups = gain_groups(class_to_obs_full(10), "level")
        values = torch.arange(groups.n_groups, dtype=torch.float64)
        assert torch.equal(groups.expand(values), values[groups.node_group])

    def test_branch_codes_mark_the_root_on_every_path(self):
        codes = branch_codes(class_to_obs_full(10))
        assert bool((codes[:, 0].abs() == 1.0).all())
        assert int(codes.abs().sum()) == 6 * 3 + 4 * 4

    def test_an_unknown_sharing_name_is_rejected(self):
        with pytest.raises(ValueError, match="sharing must be one of"):
            gain_groups(class_to_obs_full(4), "per_batch")

    def test_a_custom_grouping_reports_its_path_dependence(self):
        hrc = class_to_obs_full(10)
        parity = node_depths(hrc) % 2
        groups = custom_gain_groups(hrc, parity, name="parity")
        assert groups.n_groups == 2
        assert groups.path_repeats
        assert groups.path_group_count == 2


class TestLogGainPosterior:
    def test_gain_summaries_use_the_lognormal_identities(self):
        posterior = LogGainPosterior.prior(
            gain_groups(class_to_obs_full(4), "global"), mean=0.4, variance=0.2
        )
        assert float(posterior.gain_median) == pytest.approx(math.exp(0.4))
        assert float(posterior.gain_mean) == pytest.approx(math.exp(0.4 + 0.1))
        assert float(posterior.alpha_median) == pytest.approx(math.exp(-0.4))

    def test_the_unit_belief_is_the_uncalibrated_head(self):
        hrc = class_to_obs_full(8)
        mean, variance = random_moments(16, hrc, seed=7)
        calibrated = hsm_class_probabilities(mean, variance, hrc, LogGainPosterior.unit(hrc))
        reference = hrc_log_probs(mean, variance, hrc, log_tau=0.0).exp()
        assert torch.allclose(calibrated, reference, atol=1e-14)

    def test_dropping_the_gain_uncertainty_keeps_the_mean(self):
        posterior = LogGainPosterior.prior(
            gain_groups(class_to_obs_full(8), "level"), mean=-0.3, variance=0.5
        )
        point = posterior.deterministic()
        assert torch.equal(point.mean, posterior.mean)
        assert float(point.variance.abs().max()) == 0.0

    def test_a_negative_variance_is_rejected(self):
        groups = gain_groups(class_to_obs_full(4), "global")
        with pytest.raises(ValueError, match="nonnegative"):
            LogGainPosterior(
                mean=torch.zeros(1, dtype=torch.float64),
                variance=torch.full((1,), -1.0, dtype=torch.float64),
                groups=groups,
            )

    def test_a_shape_mismatch_is_rejected(self):
        groups = gain_groups(class_to_obs_full(4), "node")
        with pytest.raises(ValueError, match="one entry per gain group"):
            LogGainPosterior(
                mean=torch.zeros(1, dtype=torch.float64),
                variance=torch.zeros(1, dtype=torch.float64),
                groups=groups,
            )


# ──────────────────────────────────────────────────────────────────────────────
#  Class moments and cross-class dependence
# ──────────────────────────────────────────────────────────────────────────────


class TestClassMoments:
    def test_independent_gains_use_the_exact_product_formula(self):
        hrc = class_to_obs_full(8)
        mean, variance = random_moments(24, hrc, seed=8)
        posterior = LogGainPosterior.prior(gain_groups(hrc, "node"), mean=0.1, variance=0.25)
        nodes = hsm_node_moments(mean, variance, hrc, posterior)
        moments = hsm_class_moments(mean, variance, hrc, posterior)
        codes = branch_codes(hrc)
        oriented_mean = torch.where(
            codes[None] > 0.0, nodes.mean[:, None, :], 1.0 - nodes.mean[:, None, :]
        )
        on_path = codes.abs()[None]
        factor = oriented_mean * on_path + (1.0 - on_path)
        second = (nodes.variance[:, None, :] + oriented_mean.square()) * on_path + (
            1.0 - on_path
        )
        assert torch.allclose(moments.mean, factor.prod(-1), atol=1e-14)
        assert torch.allclose(
            moments.variance, second.prod(-1) - factor.prod(-1).square(), atol=1e-14
        )

    def test_a_global_gain_is_not_the_product_of_marginal_node_moments(self):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(24, hrc, seed=9)
        shared = LogGainPosterior.prior(gain_groups(hrc, "global"), mean=0.0, variance=0.6)
        per_node = LogGainPosterior(
            mean=torch.zeros(hrc.len, dtype=torch.float64),
            variance=torch.full((hrc.len,), 0.6, dtype=torch.float64),
            groups=gain_groups(hrc, "node"),
        )
        conditional = hsm_class_moments(mean, variance, hrc, shared)
        product = hsm_class_moments(mean, variance, hrc, per_node)
        # Both are valid distributions, but the shared gain correlates the path
        # factors, so its dispersion is strictly larger somewhere.
        assert float((conditional.variance - product.variance).max()) > 1e-6

    def test_gain_uncertainty_only_disperses_the_class_probability(self):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(24, hrc, seed=10)
        groups = gain_groups(hrc, "global")
        certain = hsm_class_moments(
            mean, variance, hrc, LogGainPosterior.prior(groups, mean=0.2, variance=0.0)
        )
        uncertain = hsm_class_moments(
            mean, variance, hrc, LogGainPosterior.prior(groups, mean=0.2, variance=0.4)
        )
        assert float(certain.variance.max()) < float(uncertain.variance.max())

    def test_a_repeating_non_global_grouping_is_rejected(self):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(8, hrc, seed=11)
        groups = custom_gain_groups(hrc, node_depths(hrc) % 2, name="parity")
        with pytest.raises(ValueError, match="several shared log-gains"):
            hsm_class_moments(mean, variance, hrc, LogGainPosterior.prior(groups))

    def test_the_padded_tree_is_rejected(self):
        hrc = class_to_obs(10)
        with pytest.raises(ValueError, match="proper K-leaf tree"):
            gain_groups(hrc, "global")

    def test_negative_state_variances_are_rejected(self):
        hrc = class_to_obs_full(4)
        mean, variance = random_moments(4, hrc, seed=12)
        with pytest.raises(ValueError, match="finite and nonnegative"):
            hsm_class_moments(mean, -variance, hrc, LogGainPosterior.unit(hrc))

    def test_the_negative_log_likelihood_is_the_gathered_log_probability(self):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(40, hrc, seed=13)
        labels = torch.arange(40) % 10
        posterior = LogGainPosterior.prior(gain_groups(hrc, "level"), mean=0.1, variance=0.05)
        probabilities = hsm_class_probabilities(mean, variance, hrc, posterior)
        expected = -probabilities.gather(1, labels[:, None]).log().mean()
        loss = hsm_negative_log_likelihood(mean, variance, labels, hrc, posterior)
        assert float(loss) == pytest.approx(float(expected), rel=1e-12)

    def test_chunking_does_not_change_the_result(self):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(70, hrc, seed=14)
        posterior = LogGainPosterior.prior(gain_groups(hrc, "global"), mean=0.2, variance=0.1)
        whole = hsm_class_moments(mean, variance, hrc, posterior, chunk_size=1024)
        split = hsm_class_moments(mean, variance, hrc, posterior, chunk_size=7)
        assert torch.allclose(whole.mean, split.mean, atol=1e-15)
        assert torch.allclose(whole.variance, split.variance, atol=1e-15)


def test_the_branch_tail_survives_a_saturated_node():
    """A confident node's non-taken branch must not fall back on a complement.

    ``1 - E[U]`` cannot resolve a branch probability below machine epsilon,
    while the log-sum-exp over the same quadrature can, and the two agree
    wherever the complement is representable at all.
    """

    hrc = class_to_obs_full(10)
    posterior = LogGainPosterior.prior(gain_groups(hrc, "node"), mean=0.0, variance=0.1)
    variance = torch.full((1, hrc.len), 1e-9, dtype=torch.float64)
    moderate = hsm_node_moments(
        torch.full((1, hrc.len), 8.0, dtype=torch.float64), variance, hrc, posterior
    )
    assert float(moderate.log_complement.min()) == pytest.approx(
        math.log(float((1.0 - moderate.mean).min())), rel=1e-9
    )
    saturated = hsm_node_moments(
        torch.full((1, hrc.len), 40.0, dtype=torch.float64), variance, hrc, posterior
    )
    # The complement has already lost the magnitude here; the log form has not.
    assert float(saturated.log_complement.min()) < math.log(
        float((1.0 - saturated.mean).min())
    ) - 0.5
    probabilities = hsm_class_moments(
        torch.full((1, hrc.len), 40.0, dtype=torch.float64), variance, hrc, posterior
    ).mean
    assert float(probabilities.min()) > 0.0
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-14)


class TestCrossClassCovariance:
    @pytest.mark.parametrize("sharing", ["node", "global"])
    @pytest.mark.parametrize("num_classes", [3, 5, 10])
    def test_rows_sum_to_zero_because_the_probabilities_sum_to_one(
        self, sharing, num_classes
    ):
        hrc = class_to_obs_full(num_classes)
        mean, variance = random_moments(16, hrc, seed=15)
        posterior = LogGainPosterior.prior(gain_groups(hrc, sharing), mean=0.2, variance=0.2)
        covariance = hsm_cross_class_covariance(mean, variance, hrc, posterior)
        assert float(covariance.sum(-1).abs().max()) < 1e-12

    @pytest.mark.parametrize("sharing", ["node", "global"])
    def test_the_diagonal_is_the_class_variance(self, sharing):
        hrc = class_to_obs_full(8)
        mean, variance = random_moments(16, hrc, seed=16)
        posterior = LogGainPosterior.prior(gain_groups(hrc, sharing), mean=-0.1, variance=0.3)
        covariance = hsm_cross_class_covariance(mean, variance, hrc, posterior)
        moments = hsm_class_moments(mean, variance, hrc, posterior)
        diagonal = covariance.diagonal(dim1=-2, dim2=-1)
        assert torch.allclose(diagonal, moments.variance, atol=1e-14)

    def test_the_result_is_symmetric(self):
        hrc = class_to_obs_full(5)
        mean, variance = random_moments(8, hrc, seed=17)
        posterior = LogGainPosterior.prior(gain_groups(hrc, "node"), mean=0.0, variance=0.2)
        covariance = hsm_cross_class_covariance(mean, variance, hrc, posterior)
        assert torch.allclose(covariance, covariance.transpose(-1, -2), atol=1e-16)

    def test_level_sharing_is_rejected(self):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(8, hrc, seed=18)
        posterior = LogGainPosterior.prior(gain_groups(hrc, "level"))
        with pytest.raises(ValueError, match="per-node groups"):
            hsm_cross_class_covariance(mean, variance, hrc, posterior)


# ──────────────────────────────────────────────────────────────────────────────
#  Held-out calibration
# ──────────────────────────────────────────────────────────────────────────────


class TestBatchCalibration:
    @pytest.mark.parametrize("true_log_gain", [-0.8, 0.0, 0.9])
    def test_a_global_fit_recovers_a_known_gain(self, true_log_gain):
        hrc = class_to_obs_full(8)
        mean, variance = random_moments(20000, hrc, seed=19)
        labels = sample_leaf_labels(hrc, mean, variance, true_log_gain, seed=20)
        posterior = fit_hsm_log_gain(
            mean, variance, labels, hrc, sharing="global", prior_variance=4.0
        )
        deviation = math.sqrt(float(posterior.variance[0]))
        assert abs(float(posterior.mean[0]) - true_log_gain) < 4.0 * deviation
        assert deviation < 0.1

    def test_deeper_per_node_gains_are_less_identified(self):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(4000, hrc, seed=21)
        labels = sample_leaf_labels(hrc, mean, variance, 0.4, seed=22)
        posterior = fit_hsm_log_gain(
            mean, variance, labels, hrc, sharing="node", prior_variance=4.0
        )
        depths = node_depths(hrc)
        root = int(posterior.variance.argmin())
        assert int(depths[root]) == 0
        assert int(posterior.visits.sum()) == int(branch_codes(hrc).abs().sum(-1)[labels].sum())
        assert float(posterior.variance.max()) > float(posterior.variance.min())

    def test_visits_partition_the_calibration_node_visits(self):
        hrc = class_to_obs_full(10)
        labels = torch.arange(200) % 10
        for sharing in ("global", "level", "node"):
            groups = gain_groups(hrc, sharing)
            visits = hsm_calibration_visits(labels, hrc, groups)
            assert visits.shape == (groups.n_groups,)
            assert int(visits.sum()) == int(branch_codes(hrc).abs().sum(-1)[labels].sum())

    def test_the_grid_posterior_reduces_to_the_prior_without_data(self):
        hrc = class_to_obs_full(4)
        mean, variance = random_moments(0, hrc, seed=23)
        grid, log_posterior = hsm_log_posterior_on_grid(
            mean,
            variance,
            torch.zeros(0, dtype=torch.long),
            hrc,
            gain_groups(hrc, "global"),
            prior_mean=0.5,
            prior_variance=0.25,
        )
        weights = torch.softmax(log_posterior, dim=-1)
        assert float((weights * grid).sum()) == pytest.approx(0.5, abs=1e-6)
        centred = (weights * (grid - 0.5).square()).sum()
        assert float(centred) == pytest.approx(0.25, rel=1e-3)

    def test_the_laplace_fit_agrees_with_the_grid_moments(self):
        hrc = class_to_obs_full(8)
        mean, variance = random_moments(8000, hrc, seed=24)
        labels = sample_leaf_labels(hrc, mean, variance, 0.5, seed=25)
        grid = fit_hsm_log_gain(mean, variance, labels, hrc, prior_variance=4.0)
        laplace = fit_hsm_log_gain(
            mean, variance, labels, hrc, prior_variance=4.0, method="laplace"
        )
        assert float(laplace.mean[0]) == pytest.approx(float(grid.mean[0]), abs=0.01)
        assert math.sqrt(float(laplace.variance[0])) == pytest.approx(
            math.sqrt(float(grid.variance[0])), rel=0.1
        )

    def test_an_unknown_method_is_rejected(self):
        hrc = class_to_obs_full(4)
        mean, variance = random_moments(8, hrc, seed=26)
        with pytest.raises(ValueError, match="method must be"):
            fit_hsm_log_gain(mean, variance, torch.zeros(8, dtype=torch.long), hrc, method="mcmc")

    def test_an_invalid_label_is_rejected(self):
        hrc = class_to_obs_full(4)
        mean, variance = random_moments(8, hrc, seed=27)
        with pytest.raises(ValueError, match="invalid class index"):
            fit_hsm_log_gain(mean, variance, torch.full((8,), 9, dtype=torch.long), hrc)

    def test_calibration_lowers_the_held_out_loss_of_a_mis_scaled_head(self):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(6000, hrc, seed=28)
        labels = sample_leaf_labels(hrc, mean, variance, 0.9, seed=29)
        fit = mean[:3000], variance[:3000], labels[:3000]
        held = mean[3000:], variance[3000:], labels[3000:]
        posterior = fit_hsm_log_gain(*fit, hrc, sharing="global", prior_variance=4.0)
        before = hsm_negative_log_likelihood(*held, hrc, LogGainPosterior.unit(hrc))
        after = hsm_negative_log_likelihood(*held, hrc, posterior)
        assert float(after) < float(before)


class TestGridResolution:
    @pytest.mark.parametrize("prior_variance", [0.25, 1.0, 4.0])
    def test_a_data_free_group_keeps_its_stated_prior(self, prior_variance):
        hrc = class_to_obs_full(4)
        mean, variance = random_moments(0, hrc, seed=38)
        posterior = fit_hsm_log_gain(
            mean,
            variance,
            torch.zeros(0, dtype=torch.long),
            hrc,
            prior_mean=0.3,
            prior_variance=prior_variance,
        )
        assert float(posterior.mean[0]) == pytest.approx(0.3, abs=1e-6)
        assert float(posterior.variance[0]) == pytest.approx(prior_variance, rel=1e-6)

    def test_refinement_is_what_resolves_a_sharp_posterior(self):
        hrc = class_to_obs_full(8)
        mean, variance = random_moments(8000, hrc, seed=39)
        labels = sample_leaf_labels(hrc, mean, variance, 0.5, seed=40)
        kwargs = {"prior_variance": 4.0, "grid_size": 129}
        coarse = fit_hsm_log_gain(mean, variance, labels, hrc, refinements=0, **kwargs)
        refined = fit_hsm_log_gain(mean, variance, labels, hrc, refinements=1, **kwargs)
        laplace = fit_hsm_log_gain(
            mean, variance, labels, hrc, method="laplace", **kwargs
        )
        # The coarse spacing exceeds the posterior width, so it reads a
        # spuriously certain gain. One refinement recovers a scale that agrees
        # with the Laplace one to within its own Gaussian approximation error.
        assert float(coarse.variance[0]) < 0.1 * float(refined.variance[0])
        assert math.sqrt(float(refined.variance[0])) == pytest.approx(
            math.sqrt(float(laplace.variance[0])), rel=1e-2
        )

    @pytest.mark.parametrize("grid_size", [129, 513, 2049])
    def test_the_refined_fit_is_independent_of_the_grid_size(self, grid_size):
        hrc = class_to_obs_full(8)
        mean, variance = random_moments(4000, hrc, seed=41)
        labels = sample_leaf_labels(hrc, mean, variance, -0.3, seed=42)
        reference = fit_hsm_log_gain(
            mean, variance, labels, hrc, prior_variance=4.0, grid_size=1025
        )
        posterior = fit_hsm_log_gain(
            mean, variance, labels, hrc, prior_variance=4.0, grid_size=grid_size
        )
        assert float(posterior.mean[0]) == pytest.approx(float(reference.mean[0]), abs=1e-8)
        assert float(posterior.variance[0]) == pytest.approx(
            float(reference.variance[0]), rel=1e-8
        )

    def test_an_explicit_per_group_grid_is_accepted(self):
        hrc = class_to_obs_full(8)
        mean, variance = random_moments(500, hrc, seed=43)
        labels = sample_leaf_labels(hrc, mean, variance, 0.2, seed=44)
        groups = gain_groups(hrc, "level")
        window = torch.linspace(-1.0, 1.0, 65, dtype=torch.float64)
        grid = window[None].repeat(groups.n_groups, 1)
        returned, log_posterior = hsm_log_posterior_on_grid(
            mean, variance, labels, hrc, groups, grid=grid
        )
        assert returned.shape == grid.shape
        assert log_posterior.shape == (groups.n_groups, 65)

    def test_a_mismatched_grid_row_count_is_rejected(self):
        hrc = class_to_obs_full(8)
        mean, variance = random_moments(50, hrc, seed=45)
        groups = gain_groups(hrc, "level")
        with pytest.raises(ValueError, match="one row per gain group"):
            hsm_log_posterior_on_grid(
                mean,
                variance,
                torch.zeros(50, dtype=torch.long),
                hrc,
                groups,
                grid=torch.zeros(2, 33, dtype=torch.float64),
            )

    def test_a_negative_refinement_count_is_rejected(self):
        hrc = class_to_obs_full(4)
        mean, variance = random_moments(8, hrc, seed=46)
        with pytest.raises(ValueError, match="refinements must be nonnegative"):
            fit_hsm_log_gain(
                mean, variance, torch.zeros(8, dtype=torch.long), hrc, refinements=-1
            )


class TestSequentialCalibration:
    def test_a_neutral_observation_leaves_the_belief_alone(self):
        mean, variance = hsm_adf_projection(0.3, 0.2, 0.5, 0.0, 1.0)
        assert mean == 0.3
        assert variance == 0.2

    def test_a_degenerate_branch_mean_is_skipped(self):
        assert hsm_adf_projection(0.3, 0.2, 1.0, 0.1) == (0.3, 0.2)
        assert hsm_adf_projection(0.3, 0.2, 0.0, 0.1) == (0.3, 0.2)

    def test_the_variance_decrement_is_capped_rather_than_floored(self):
        _, variance = hsm_adf_projection(0.0, 0.1, 0.5, 10.0, 1.0)
        assert variance == pytest.approx(0.1 * (1.0 - 0.999))
        assert variance > 0.0

    def test_a_zero_observation_moves_the_mean_the_other_way(self):
        up, _ = hsm_adf_projection(0.0, 0.2, 0.7, 0.01, 1.0)
        down, _ = hsm_adf_projection(0.0, 0.2, 0.7, 0.01, 0.0)
        assert up > 0.0 > down

    def test_filtering_approaches_the_order_independent_batch_fit(self):
        hrc = class_to_obs_full(8)
        mean, variance = random_moments(4000, hrc, seed=30)
        labels = sample_leaf_labels(hrc, mean, variance, 0.6, seed=31)
        batch = fit_hsm_log_gain(
            mean, variance, labels, hrc, sharing="global", prior_variance=4.0
        )
        filtered = calibrate_hsm_log_gain_adf(
            mean, variance, labels, hrc, sharing="global", prior_variance=4.0
        )
        assert abs(float(filtered.mean[0]) - float(batch.mean[0])) < 0.02
        assert int(filtered.visits.sum()) == int(batch.visits.sum())

    def test_reordering_the_stream_perturbs_the_filtered_belief(self):
        hrc = class_to_obs_full(8)
        mean, variance = random_moments(1500, hrc, seed=32)
        labels = sample_leaf_labels(hrc, mean, variance, 0.6, seed=33)
        forward = calibrate_hsm_log_gain_adf(
            mean, variance, labels, hrc, prior_variance=4.0
        )
        reversed_order = torch.arange(1500).flip(0)
        backward = calibrate_hsm_log_gain_adf(
            mean, variance, labels, hrc, prior_variance=4.0, visit_order=reversed_order
        )
        assert float(forward.mean[0]) != float(backward.mean[0])
        assert abs(float(forward.mean[0]) - float(backward.mean[0])) < 0.05

    def test_process_noise_keeps_the_belief_from_collapsing(self):
        hrc = class_to_obs_full(8)
        mean, variance = random_moments(1500, hrc, seed=34)
        labels = sample_leaf_labels(hrc, mean, variance, 0.3, seed=35)
        still = calibrate_hsm_log_gain_adf(mean, variance, labels, hrc, prior_variance=1.0)
        drifting = calibrate_hsm_log_gain_adf(
            mean, variance, labels, hrc, prior_variance=1.0, process_variance=1e-3
        )
        assert float(drifting.variance[0]) > float(still.variance[0])

    def test_a_negative_process_variance_is_rejected(self):
        hrc = class_to_obs_full(4)
        mean, variance = random_moments(8, hrc, seed=36)
        with pytest.raises(ValueError, match="finite and nonnegative"):
            calibrate_hsm_log_gain_adf(
                mean,
                variance,
                torch.zeros(8, dtype=torch.long),
                hrc,
                process_variance=-1.0,
            )


def test_class_moments_expose_the_total_variance_decomposition():
    hrc = class_to_obs_full(10)
    mean, variance = random_moments(16, hrc, seed=37)
    posterior = LogGainPosterior.prior(gain_groups(hrc, "level"), mean=0.1, variance=0.2)
    moments = hsm_class_moments(mean, variance, hrc, posterior)
    assert isinstance(moments, HsmClassMoments)
    total = moments.expected_bernoulli_variance + moments.variance
    assert torch.allclose(total, moments.mean * (1.0 - moments.mean), atol=1e-14)


def test_groups_move_between_devices_without_changing_the_assignment():
    groups = gain_groups(class_to_obs_full(10), "level")
    moved = groups.to("cpu")
    assert isinstance(moved, GainGroups)
    assert torch.equal(moved.node_group, groups.node_group)
    assert moved.path_repeats == groups.path_repeats


def test_default_quadrature_order_is_within_the_recommended_range():
    assert 12 <= DEFAULT_GAIN_ORDER <= 20
