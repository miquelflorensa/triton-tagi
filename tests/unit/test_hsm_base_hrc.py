"""Tests for the base HRC instantiation of the hierarchical probit calibration.

The base HRC head observes ``+/-1`` at every tree node under a fixed
observation noise ``sigma_v``, so the calibrated channel is

    R_n = G_{r(n)} Z_n + sigma_v o_n + eps_n,   eps_n ~ N(0, sigma_v^2)

and ``lambda = 0`` with ``q = 0`` is that head unchanged. These tests pin the
three things that makes true: the uncalibrated belief reproduces
``obs_to_class_probs`` at ``alpha = 1 / sigma_v``, the channel noise is an exact
reparameterization of the log-gain rather than a new degree of freedom, and
every downstream moment, fit, and filter reads the same scale as the fit.
"""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi.hrc_softmax import class_to_obs_full, obs_to_class_probs
from triton_tagi.hsm_calibration import (
    GainGroups,
    LogGainPosterior,
    branch_codes,
    calibrate_hsm_log_gain_adf,
    fit_hsm_log_gain,
    gain_groups,
    hsm_class_moments,
    hsm_class_probabilities,
    hsm_cross_class_covariance,
    hsm_log_posterior_on_grid,
    hsm_node_moments,
    hsm_partition_deviation,
)

# The observation noise the CIFAR-10 base HRC head was trained with. cuTAGI
# reads that head out at a hard-coded alpha = 3, which is a channel noise of
# 1/3 and not the 0.3 it trained against; the gap is the reason the scale is
# carried explicitly rather than assumed.
TRAINED_SIGMA_V = 0.3
CUTAGI_READOUT_SIGMA_V = 1.0 / 3.0


def random_moments(batch: int, hrc, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    mean = torch.randn(batch, hrc.len, generator=generator, dtype=torch.float64)
    variance = torch.rand(batch, hrc.len, generator=generator, dtype=torch.float64) * 2.0
    return mean, variance


def sample_leaf_labels(
    hrc,
    mean: torch.Tensor,
    variance: torch.Tensor,
    log_gain: float,
    sigma_v: float,
    seed: int,
) -> torch.Tensor:
    """Draw leaf labels from the base HRC channel at a known gain and noise."""

    generator = torch.Generator().manual_seed(seed)
    codes = branch_codes(hrc)
    state = mean + variance.sqrt() * torch.randn(
        mean.shape, generator=generator, dtype=torch.float64
    )
    noise = sigma_v * torch.randn(mean.shape, generator=generator, dtype=torch.float64)
    latent = math.exp(log_gain) * state + sigma_v * hrc.node_offset().double() + noise
    taken = torch.where(latent >= 0.0, 1.0, -1.0)
    on_path = codes.abs()
    agrees = ((codes[None] * taken[:, None, :]) > 0.0).double() * on_path[None]
    return (agrees + (1.0 - on_path[None])).prod(-1).argmax(-1)


# ──────────────────────────────────────────────────────────────────────────────
#  The uncalibrated belief is the frozen base HRC head
# ──────────────────────────────────────────────────────────────────────────────


class TestUncalibratedBaseHrcHead:
    @pytest.mark.parametrize("sigma_v", [TRAINED_SIGMA_V, CUTAGI_READOUT_SIGMA_V, 1.0, 2.5])
    def test_unit_belief_reproduces_the_base_hrc_readout(self, sigma_v):
        hrc = class_to_obs_full(8, use_prior_offsets=False)
        mean, variance = random_moments(32, hrc, seed=1)
        calibrated = hsm_class_probabilities(
            mean, variance, hrc, LogGainPosterior.unit(hrc, sigma_v=sigma_v)
        )
        reference = obs_to_class_probs(mean, variance, hrc, alpha=1.0 / sigma_v)
        assert torch.allclose(calibrated, reference, atol=1e-15)

    def test_the_cutagi_readout_is_not_the_trained_channel(self):
        # Reading a head trained at sigma_v = 0.3 through cuTAGI's alpha = 3 is
        # a different model, not a different parameterization of the same one.
        hrc = class_to_obs_full(8, use_prior_offsets=False)
        mean, variance = random_moments(32, hrc, seed=2)
        trained = hsm_class_probabilities(
            mean, variance, hrc, LogGainPosterior.unit(hrc, sigma_v=TRAINED_SIGMA_V)
        )
        readout = hsm_class_probabilities(
            mean, variance, hrc, LogGainPosterior.unit(hrc, sigma_v=CUTAGI_READOUT_SIGMA_V)
        )
        assert not torch.allclose(trained, readout, atol=1e-6)

    @pytest.mark.parametrize("sigma_v", [TRAINED_SIGMA_V, 1.0, 3.0])
    def test_a_zero_output_telescopes_to_the_branch_prior(self, sigma_v):
        # The offset is a probit and enters R scaled by sigma_v, so a silent
        # network reproduces the class prior at any channel noise and any gain.
        hrc = class_to_obs_full(6)
        mean = torch.zeros(4, hrc.len, dtype=torch.float64)
        variance = torch.zeros_like(mean)
        posterior = LogGainPosterior.prior(
            gain_groups(hrc, "global"), mean=0.7, variance=0.3, sigma_v=sigma_v
        )
        probabilities = hsm_class_probabilities(mean, variance, hrc, posterior)
        expected = torch.full((4, 6), 1.0 / 6.0, dtype=torch.float64)
        assert torch.allclose(probabilities, expected, atol=1e-14)


# ──────────────────────────────────────────────────────────────────────────────
#  The channel noise is an exact reparameterization, not a new freedom
# ──────────────────────────────────────────────────────────────────────────────


class TestChannelScaleReparameterization:
    @pytest.mark.parametrize("sharing", ["global", "level", "node"])
    def test_node_moments_shift_the_log_gain_by_log_sigma_v(self, sharing):
        hrc = class_to_obs_full(8)
        mean, variance = random_moments(24, hrc, seed=3)
        groups = gain_groups(hrc, sharing)
        scaled = LogGainPosterior.prior(
            groups, mean=0.4, variance=0.35, sigma_v=TRAINED_SIGMA_V
        )
        shifted = LogGainPosterior.prior(
            groups, mean=0.4 - math.log(TRAINED_SIGMA_V), variance=0.35
        )
        left = hsm_node_moments(mean, variance, hrc, scaled)
        right = hsm_node_moments(mean, variance, hrc, shifted)
        for name in ("mean", "variance", "cov_state", "cov_log_gain"):
            assert torch.allclose(getattr(left, name), getattr(right, name), atol=1e-15)

    def test_class_moments_shift_the_log_gain_by_log_sigma_v(self):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(16, hrc, seed=4)
        groups = gain_groups(hrc, "global")
        scaled = hsm_class_moments(
            mean,
            variance,
            hrc,
            LogGainPosterior.prior(groups, mean=-0.2, variance=0.5, sigma_v=2.0),
        )
        shifted = hsm_class_moments(
            mean,
            variance,
            hrc,
            LogGainPosterior.prior(groups, mean=-0.2 - math.log(2.0), variance=0.5),
        )
        assert torch.allclose(scaled.mean, shifted.mean, atol=1e-15)
        assert torch.allclose(scaled.variance, shifted.variance, atol=1e-15)

    def test_the_grid_likelihood_shifts_with_the_channel(self):
        hrc = class_to_obs_full(8, use_prior_offsets=False)
        mean, variance = random_moments(64, hrc, seed=5)
        labels = sample_leaf_labels(hrc, mean, variance, 0.0, TRAINED_SIGMA_V, seed=6)
        groups = gain_groups(hrc, "global")
        grid = torch.linspace(-2.0, 2.0, 65, dtype=torch.float64)
        scaled = hsm_log_posterior_on_grid(
            mean, variance, labels, hrc, groups, sigma_v=TRAINED_SIGMA_V, grid=grid
        )[1]
        shifted = hsm_log_posterior_on_grid(
            mean, variance, labels, hrc, groups, grid=grid - math.log(TRAINED_SIGMA_V)
        )[1]
        # Both carry the same prior penalty only up to the shift, so compare the
        # likelihood differences rather than the levels.
        penalty = 0.5 * (grid.square() - (grid - math.log(TRAINED_SIGMA_V)).square())
        assert torch.allclose(scaled + penalty, shifted, atol=1e-12)

    def test_the_batch_fit_shifts_with_the_channel(self):
        hrc = class_to_obs_full(8, use_prior_offsets=False)
        mean, variance = random_moments(256, hrc, seed=7)
        labels = sample_leaf_labels(hrc, mean, variance, 0.3, TRAINED_SIGMA_V, seed=8)
        scaled = fit_hsm_log_gain(
            mean, variance, labels, hrc, prior_variance=4.0, sigma_v=TRAINED_SIGMA_V
        )
        shifted = fit_hsm_log_gain(
            mean,
            variance,
            labels,
            hrc,
            prior_mean=-math.log(TRAINED_SIGMA_V),
            prior_variance=4.0,
        )
        assert float(scaled.mean) == pytest.approx(
            float(shifted.mean) + math.log(TRAINED_SIGMA_V), abs=1e-9
        )
        assert float(scaled.variance) == pytest.approx(float(shifted.variance), rel=1e-9)
        assert scaled.sigma_v == TRAINED_SIGMA_V

    def test_the_sequential_filter_shifts_with_the_channel(self):
        hrc = class_to_obs_full(8, use_prior_offsets=False)
        mean, variance = random_moments(64, hrc, seed=9)
        labels = sample_leaf_labels(hrc, mean, variance, 0.0, TRAINED_SIGMA_V, seed=10)
        scaled = calibrate_hsm_log_gain_adf(
            mean, variance, labels, hrc, prior_variance=4.0, sigma_v=TRAINED_SIGMA_V
        )
        shifted = calibrate_hsm_log_gain_adf(
            mean,
            variance,
            labels,
            hrc,
            prior_mean=-math.log(TRAINED_SIGMA_V),
            prior_variance=4.0,
        )
        assert float(scaled.mean) == pytest.approx(
            float(shifted.mean) + math.log(TRAINED_SIGMA_V), abs=1e-12
        )
        assert float(scaled.variance) == pytest.approx(float(shifted.variance), rel=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
#  Structural identities survive the channel noise
# ──────────────────────────────────────────────────────────────────────────────


class TestStructuralIdentitiesAtAnyChannel:
    @pytest.mark.parametrize("sigma_v", [0.1, TRAINED_SIGMA_V, 1.0, 4.0])
    def test_class_probabilities_sum_to_one(self, sigma_v):
        hrc = class_to_obs_full(10)
        mean, variance = random_moments(32, hrc, seed=11)
        posterior = LogGainPosterior.prior(
            gain_groups(hrc, "global"), mean=0.2, variance=0.6, sigma_v=sigma_v
        )
        deviation = hsm_partition_deviation(mean, variance, hrc, posterior)
        assert deviation < 1e-14

    @pytest.mark.parametrize("sharing", ["global", "node"])
    def test_cross_class_covariance_rows_sum_to_zero(self, sharing):
        hrc = class_to_obs_full(6)
        mean, variance = random_moments(8, hrc, seed=12)
        posterior = LogGainPosterior.prior(
            gain_groups(hrc, sharing), mean=-0.1, variance=0.4, sigma_v=TRAINED_SIGMA_V
        )
        covariance = hsm_cross_class_covariance(mean, variance, hrc, posterior)
        assert float(covariance.sum(dim=-1).abs().max()) < 1e-15

    def test_the_total_variance_split_holds_per_node(self):
        hrc = class_to_obs_full(8)
        mean, variance = random_moments(16, hrc, seed=13)
        posterior = LogGainPosterior.prior(
            gain_groups(hrc, "level"), mean=0.5, variance=0.25, sigma_v=TRAINED_SIGMA_V
        )
        moments = hsm_node_moments(mean, variance, hrc, posterior)
        total = moments.mean * (1.0 - moments.mean)
        assert torch.allclose(
            total, moments.expected_bernoulli_variance + moments.variance, atol=1e-15
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Calibration against a planted gain on the base HRC channel
# ──────────────────────────────────────────────────────────────────────────────


class TestCalibrationOnTheBaseHrcChannel:
    @pytest.mark.parametrize("planted", [-0.5, 0.0, 0.6])
    def test_the_fit_recovers_a_planted_gain(self, planted):
        hrc = class_to_obs_full(8, use_prior_offsets=False)
        mean, variance = random_moments(4096, hrc, seed=14)
        labels = sample_leaf_labels(hrc, mean, variance, planted, TRAINED_SIGMA_V, seed=15)
        posterior = fit_hsm_log_gain(
            mean, variance, labels, hrc, prior_variance=4.0, sigma_v=TRAINED_SIGMA_V
        )
        deviation = float(posterior.variance.sqrt())
        # A larger planted gain saturates more branches, so the same visit count
        # identifies it less sharply: the deviation runs 0.05 to 0.17 across
        # these three. The posterior has to cover the truth, not merely be tight.
        assert abs(float(posterior.mean) - planted) < 2.0 * deviation
        assert deviation < 0.25

    def test_the_recovered_probit_scale_ignores_the_nominal_channel(self):
        # Reading the same frozen head at cuTAGI's alpha = 3 rather than at the
        # noise it trained with moves lambda by log(sigma_readout / sigma_trained)
        # and leaves the calibrated readout where it was. The gain absorbs the
        # convention; only the anchor moves.
        hrc = class_to_obs_full(8, use_prior_offsets=False)
        mean, variance = random_moments(4096, hrc, seed=16)
        labels = sample_leaf_labels(hrc, mean, variance, 0.0, TRAINED_SIGMA_V, seed=17)
        trained = fit_hsm_log_gain(
            mean, variance, labels, hrc, prior_variance=4.0, sigma_v=TRAINED_SIGMA_V
        )
        readout = fit_hsm_log_gain(
            mean, variance, labels, hrc, prior_variance=4.0, sigma_v=CUTAGI_READOUT_SIGMA_V
        )
        shift = math.log(CUTAGI_READOUT_SIGMA_V / TRAINED_SIGMA_V)
        # The two priors are both centred at zero rather than being shifts of
        # each other, which is the whole of the remaining discrepancy.
        assert float(readout.mean - trained.mean) == pytest.approx(shift, abs=1e-3)
        assert float(readout.alpha_median) == pytest.approx(float(trained.alpha_median), rel=1e-3)

    def test_the_calibration_finds_the_channel_the_labels_came_from(self):
        hrc = class_to_obs_full(8, use_prior_offsets=False)
        mean, variance = random_moments(4096, hrc, seed=14)
        labels = sample_leaf_labels(hrc, mean, variance, 0.0, TRAINED_SIGMA_V, seed=15)
        posterior = fit_hsm_log_gain(
            mean, variance, labels, hrc, prior_variance=4.0, sigma_v=CUTAGI_READOUT_SIGMA_V
        )
        assert float(posterior.alpha_median) == pytest.approx(TRAINED_SIGMA_V, rel=0.05)

    def test_the_effective_probit_scale_is_the_readout_scale(self):
        hrc = class_to_obs_full(8, use_prior_offsets=False)
        mean, variance = random_moments(32, hrc, seed=18)
        posterior = LogGainPosterior.prior(
            gain_groups(hrc, "global"), mean=0.45, variance=0.0, sigma_v=TRAINED_SIGMA_V
        )
        alpha = float(posterior.alpha_median)
        assert alpha == pytest.approx(TRAINED_SIGMA_V * math.exp(-0.45))
        calibrated = hsm_class_probabilities(mean, variance, hrc, posterior)
        reference = obs_to_class_probs(mean, variance, hrc, alpha=1.0 / alpha)
        assert torch.allclose(calibrated, reference, atol=1e-15)

    def test_visits_and_summary_report_the_channel(self):
        hrc = class_to_obs_full(8, use_prior_offsets=False)
        mean, variance = random_moments(128, hrc, seed=19)
        labels = sample_leaf_labels(hrc, mean, variance, 0.0, TRAINED_SIGMA_V, seed=20)
        posterior = fit_hsm_log_gain(
            mean, variance, labels, hrc, sharing="node", sigma_v=TRAINED_SIGMA_V
        )
        record = posterior.summary()
        assert record["sigma_v"] == TRAINED_SIGMA_V
        assert sum(record["visits"]) == 128 * hrc.n_obs
        assert posterior.deterministic().sigma_v == TRAINED_SIGMA_V


# ──────────────────────────────────────────────────────────────────────────────
#  Rejected inputs
# ──────────────────────────────────────────────────────────────────────────────


class TestRejectedChannels:
    @pytest.mark.parametrize("sigma_v", [0.0, -1.0, float("inf"), float("nan")])
    def test_the_belief_rejects_a_nonpositive_channel(self, sigma_v):
        groups = gain_groups(class_to_obs_full(4), "global")
        with pytest.raises(ValueError, match="sigma_v"):
            LogGainPosterior.prior(groups, sigma_v=sigma_v)

    def test_the_fit_rejects_a_nonpositive_channel(self):
        hrc = class_to_obs_full(4, use_prior_offsets=False)
        mean, variance = random_moments(8, hrc, seed=21)
        labels = torch.arange(8) % 4
        with pytest.raises(ValueError, match="sigma_v"):
            fit_hsm_log_gain(mean, variance, labels, hrc, sigma_v=0.0)

    def test_the_filter_rejects_a_nonpositive_channel(self):
        hrc = class_to_obs_full(4, use_prior_offsets=False)
        mean, variance = random_moments(8, hrc, seed=22)
        labels = torch.arange(8) % 4
        with pytest.raises(ValueError, match="sigma_v"):
            calibrate_hsm_log_gain_adf(mean, variance, labels, hrc, sigma_v=-2.0)

    def test_the_grid_rejects_a_nonpositive_channel(self):
        hrc = class_to_obs_full(4, use_prior_offsets=False)
        groups = gain_groups(hrc, "node")
        assert isinstance(groups, GainGroups)
        mean, variance = random_moments(8, hrc, seed=23)
        labels = torch.arange(8) % 4
        with pytest.raises(ValueError, match="sigma_v"):
            hsm_log_posterior_on_grid(mean, variance, labels, hrc, groups, sigma_v=0.0)
