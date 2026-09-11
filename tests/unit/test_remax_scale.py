"""Tests for the auxiliary Gaussian calibration channel of the Remax scale."""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi.cdf_remax import remax_conditional_moments, remax_scale_moments
from triton_tagi.remax_scale import (
    DEFAULT_PRIOR_VARIANCE,
    REMAX_SCALE_METHODS,
    LogScalePosterior,
    calibrate_remax_log_scale_adf,
    fit_remax_log_scale,
    remax_event_surrogate_variance,
    remax_scale_consistency_slack,
    remax_scale_event_update,
    remax_scale_full_update,
    remax_scale_log_posterior_on_grid,
    remax_scale_negative_log_likelihood,
    remax_scale_tilt_update,
)

# The note's pinned synthetic three-class state and CDF head.
PINNED_MU = torch.tensor([0.8, -0.3, 0.15], dtype=torch.float64)
PINNED_E = torch.tensor([0.18, 0.09, 0.25], dtype=torch.float64)
PINNED_NU = torch.tensor([-0.7, 0.1, -0.3], dtype=torch.float64)
PINNED_R = torch.tensor([0.4, 0.2, 0.3], dtype=torch.float64)
EPSILON = 0.02
KAPPA = 1.5

# The note generates its pinned batch fit at s = 1.8 from a prior N(0, 0.5^2).
PINNED_SCALE = 1.8
PINNED_LOG_SCALE = math.log(PINNED_SCALE)
PINNED_LABELS = 3000

# Reduced quadrature for the sequential driver, whose cost is one full moment
# evaluation per calibration label. The recursion's structure, not its last
# digit, is what these tests are about.
FAST_ORDERS = {"laplace_order": 64, "hermite_order": 16, "scale_order": 12}


def pinned_state(batch: int = 1) -> tuple[torch.Tensor, ...]:
    """Return the note's fixed state repeated over ``batch`` calibration rows."""

    return tuple(
        value[None].repeat(batch, 1) for value in (PINNED_MU, PINNED_E, PINNED_NU, PINNED_R)
    )


def random_state(batch: int, num_classes: int, seed: int) -> tuple[torch.Tensor, ...]:
    """Return a random but well-conditioned frozen forward summary."""

    generator = torch.Generator().manual_seed(seed)
    shape = (batch, num_classes)
    return (
        torch.randn(shape, generator=generator, dtype=torch.float64),
        torch.rand(shape, generator=generator, dtype=torch.float64) * 0.5 + 0.05,
        torch.randn(shape, generator=generator, dtype=torch.float64) * 0.5,
        torch.rand(shape, generator=generator, dtype=torch.float64) * 0.4,
    )


def pinned_generating_probabilities(log_scale: float = PINNED_LOG_SCALE) -> torch.Tensor:
    """Return ``m_i(l)`` of the pinned state, the categorical law labels come from."""

    first, _ = remax_conditional_moments(
        PINNED_MU[None],
        PINNED_E[None],
        PINNED_NU[None],
        PINNED_R[None],
        epsilon=EPSILON,
        kappa=KAPPA,
        log_scale=log_scale,
    )
    return first[0]


def sample_pinned_labels(count: int, seed: int, log_scale: float = PINNED_LOG_SCALE):
    """Draw ``count`` labels from the pinned state at a known scale."""

    generator = torch.Generator().manual_seed(seed)
    probabilities = pinned_generating_probabilities(log_scale)
    return torch.multinomial(probabilities, count, replacement=True, generator=generator)


def moments_at(state, mean: float, variance: float, **orders):
    """Return ``(p, v_A, c)`` of every row at one working prior."""

    return remax_scale_moments(
        *state,
        scale_mean=mean,
        scale_variance=variance,
        epsilon=EPSILON,
        kappa=KAPPA,
        **orders,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  The belief and its provenance
# ──────────────────────────────────────────────────────────────────────────────


class TestLogScalePosterior:
    def test_the_stated_prior_is_the_notes_lambda_zero_and_half_squared(self):
        belief = LogScalePosterior.prior(epsilon=EPSILON, kappa=KAPPA)
        assert belief.mean == 0.0
        assert belief.variance == DEFAULT_PRIOR_VARIANCE == 0.5**2

    def test_the_unit_belief_is_the_uncalibrated_head(self):
        belief = LogScalePosterior.unit(epsilon=EPSILON, kappa=KAPPA)
        assert belief.variance == 0.0
        assert belief.scale_median == 1.0
        assert belief.scale_mean == 1.0

    def test_summaries_use_the_lognormal_identities(self):
        belief = LogScalePosterior(mean=0.4, variance=0.09, epsilon=EPSILON, kappa=KAPPA)
        assert belief.scale_median == pytest.approx(math.exp(0.4), rel=1e-15)
        assert belief.scale_mean == pytest.approx(math.exp(0.4 + 0.045), rel=1e-15)
        assert belief.scale_mean > belief.scale_median

    def test_dropping_the_uncertainty_keeps_the_mean(self):
        belief = LogScalePosterior(mean=-0.2, variance=0.3, epsilon=EPSILON, kappa=KAPPA)
        point = belief.deterministic()
        assert point.mean == belief.mean
        assert point.variance == 0.0
        assert point.scale_mean == point.scale_median

    def test_the_summary_records_the_full_provenance(self):
        belief = LogScalePosterior.prior(epsilon=EPSILON, kappa=KAPPA)
        record = belief.summary()
        for key in (
            "log_scale_mean",
            "log_scale_variance",
            "scale_median",
            "scale_mean",
            "epsilon",
            "kappa",
            "laplace_order",
            "hermite_order",
            "scale_order",
            "map_scale",
        ):
            assert key in record

    def test_a_belief_cannot_be_read_under_another_cdf_head(self):
        belief = LogScalePosterior.prior(epsilon=EPSILON, kappa=KAPPA)
        belief.require_consistent(epsilon=EPSILON, kappa=KAPPA)
        with pytest.raises(ValueError, match="was fitted at kappa=1.5"):
            belief.require_consistent(epsilon=EPSILON, kappa=2.0)
        with pytest.raises(ValueError, match="was fitted at epsilon=0.02"):
            belief.require_consistent(epsilon=0.05)

    def test_a_belief_cannot_be_read_under_another_quadrature_order(self):
        belief = LogScalePosterior.prior(epsilon=EPSILON, kappa=KAPPA)
        with pytest.raises(ValueError, match="was fitted at scale_order=20"):
            belief.require_consistent(scale_order=32)

    def test_a_negative_variance_is_rejected(self):
        with pytest.raises(ValueError, match="nonnegative"):
            LogScalePosterior(mean=0.0, variance=-1e-9, epsilon=EPSILON, kappa=KAPPA)

    def test_a_nonpositive_head_constant_is_rejected(self):
        with pytest.raises(ValueError, match="epsilon must be finite and positive"):
            LogScalePosterior(mean=0.0, variance=0.1, epsilon=0.0, kappa=KAPPA)
        with pytest.raises(ValueError, match="kappa must be finite and positive"):
            LogScalePosterior(mean=0.0, variance=0.1, epsilon=EPSILON, kappa=-1.0)

    def test_a_nonfinite_mean_is_rejected(self):
        with pytest.raises(ValueError, match="mean must be finite"):
            LogScalePosterior(mean=float("inf"), variance=0.1, epsilon=EPSILON, kappa=KAPPA)

    def test_a_degenerate_quadrature_order_is_rejected(self):
        with pytest.raises(ValueError, match="laplace_order must be at least four"):
            LogScalePosterior(mean=0.0, variance=0.1, epsilon=EPSILON, kappa=KAPPA, laplace_order=2)


# ──────────────────────────────────────────────────────────────────────────────
#  Surrogate moments and their consistency
# ──────────────────────────────────────────────────────────────────────────────


class TestSurrogateMoments:
    def test_the_bernoulli_split_reproduces_the_indicator_variance(self):
        state = random_state(6, 4, seed=101)
        probabilities, class_variance, _ = moments_at(state, 0.0, 0.25)
        noise, indicator = remax_event_surrogate_variance(probabilities, class_variance)
        assert torch.allclose(indicator, probabilities * (1.0 - probabilities), atol=1e-14)
        assert bool((noise >= 0.0).all())

    def test_the_noise_is_not_the_class_variance_itself(self):
        state = random_state(4, 4, seed=102)
        probabilities, class_variance, _ = moments_at(state, 0.0, 0.25)
        noise, _ = remax_event_surrogate_variance(probabilities, class_variance)
        # R_c carries the label randomness left once the latent probability is
        # fixed, so it is strictly the complement of v_A, never a copy of it.
        assert float((noise - class_variance).abs().max()) > 1e-3

    @pytest.mark.parametrize("num_classes", [2, 3, 5])
    def test_cauchy_schwarz_holds_on_random_states(self, num_classes):
        state = random_state(8, num_classes, seed=103 + num_classes)
        variance = 0.25
        probabilities, _, cov_scale = moments_at(state, 0.0, variance)
        slack = remax_scale_consistency_slack(variance, probabilities, cov_scale)
        assert bool((slack >= 0.0).all()), float(slack.min())

    def test_the_cross_covariances_sum_to_zero_because_probabilities_sum_to_one(self):
        state = random_state(5, 4, seed=110)
        _, _, cov_scale = moments_at(state, 0.1, 0.16)
        assert torch.allclose(cov_scale.sum(-1), torch.zeros(5, dtype=torch.float64), atol=1e-12)

    def test_a_deterministic_scale_carries_no_cross_covariance(self):
        state = random_state(3, 3, seed=111)
        _, _, cov_scale = moments_at(state, 0.3, 0.0)
        assert torch.allclose(cov_scale, torch.zeros_like(cov_scale), atol=1e-14)


# ──────────────────────────────────────────────────────────────────────────────
#  The three stated variance steps
# ──────────────────────────────────────────────────────────────────────────────


class TestUpdateVariants:
    def test_all_three_variants_share_the_exact_one_step_mean(self):
        state = pinned_state()
        mean, variance = 0.1, DEFAULT_PRIOR_VARIANCE
        probabilities, _, cov_scale = moments_at(state, mean, variance)
        for observed in range(3):
            event, _ = remax_scale_event_update(
                mean, variance, probabilities[0], cov_scale[0], observed
            )
            full, _ = remax_scale_full_update(
                mean, variance, probabilities[0], cov_scale[0], observed
            )
            tilt, _ = remax_scale_tilt_update(
                mean,
                variance,
                *state,
                observed,
                epsilon=EPSILON,
                kappa=KAPPA,
            )
            closed = mean + float(cov_scale[0, observed] / probabilities[0, observed])
            assert event == full
            assert event == pytest.approx(closed, abs=1e-14)
            # E_L[L m_c(L)] / p_c = lambda + c_c / p_c is an algebraic identity
            # under the shared Gauss--Hermite rule, so the tilt mean agrees to
            # roundoff and not merely to quadrature error.
            assert tilt == pytest.approx(event, abs=1e-14)

    @pytest.mark.parametrize("num_classes", [3, 5, 8])
    def test_the_full_projection_removes_at_least_as_much_variance(self, num_classes):
        state = random_state(6, num_classes, seed=120 + num_classes)
        mean, variance = 0.0, DEFAULT_PRIOR_VARIANCE
        probabilities, _, cov_scale = moments_at(state, mean, variance)
        strict = 0
        for row in range(6):
            for observed in range(num_classes):
                _, event = remax_scale_event_update(
                    mean, variance, probabilities[row], cov_scale[row], observed
                )
                _, full = remax_scale_full_update(
                    mean, variance, probabilities[row], cov_scale[row], observed
                )
                assert full <= event + 1e-14
                strict += int(full < event - 1e-10)
        assert strict == 6 * num_classes

    def test_the_two_projections_coincide_at_two_classes(self):
        state = random_state(6, 2, seed=130)
        mean, variance = 0.05, 0.16
        probabilities, _, cov_scale = moments_at(state, mean, variance)
        for row in range(6):
            for observed in range(2):
                _, event = remax_scale_event_update(
                    mean, variance, probabilities[row], cov_scale[row], observed
                )
                _, full = remax_scale_full_update(
                    mean, variance, probabilities[row], cov_scale[row], observed
                )
                # c_2 = -c_1 and p_2 = 1 - p_1 make the two decrements the same
                # expression; only the quadrature's own 1e-13 residual in
                # sum_i p_i and sum_i c_i separates them.
                assert full == pytest.approx(event, abs=1e-13)

    @pytest.mark.parametrize("method", REMAX_SCALE_METHODS)
    def test_a_vanishing_variance_is_a_fixed_point(self, method):
        state = pinned_state()
        mean = 0.3
        probabilities, _, cov_scale = moments_at(state, mean, 0.0)
        if method == "tilt":
            updated = remax_scale_tilt_update(mean, 0.0, *state, 1, epsilon=EPSILON, kappa=KAPPA)
        else:
            projection = remax_scale_event_update if method == "event" else remax_scale_full_update
            updated = projection(mean, 0.0, probabilities[0], cov_scale[0], 1)
        assert updated == (mean, 0.0)

    def test_the_tilt_variance_is_conditional_and_may_exceed_the_prior_step(self):
        state = pinned_state()
        mean, variance = 0.1, DEFAULT_PRIOR_VARIANCE
        probabilities, _, cov_scale = moments_at(state, mean, variance)
        _, event = remax_scale_event_update(mean, variance, probabilities[0], cov_scale[0], 0)
        _, tilt = remax_scale_tilt_update(mean, variance, *state, 0, epsilon=EPSILON, kappa=KAPPA)
        # Var(L | C = c) for the realised class is not the partition average,
        # so it is free to sit above the projected decrement.
        assert tilt > event

    def test_the_tilt_update_is_stable_in_the_scale_order(self):
        state = pinned_state()
        # At q = 0.2^2 the twenty-node rule is already converged: 20 against 32
        # agrees to 1.5e-16 in the mean and 2.7e-16 in the variance.
        reference = remax_scale_tilt_update(
            0.1, 0.04, *state, 0, epsilon=EPSILON, kappa=KAPPA, scale_order=20
        )
        refined = remax_scale_tilt_update(
            0.1, 0.04, *state, 0, epsilon=EPSILON, kappa=KAPPA, scale_order=32
        )
        assert refined[0] == pytest.approx(reference[0], abs=1e-14)
        assert refined[1] == pytest.approx(reference[1], abs=1e-14)

    def test_the_tilt_update_converges_at_the_widest_stated_prior(self):
        state = pinned_state()
        # At the note's full q_0 = 0.5^2 the scale rule is not yet at the
        # double-precision floor: 20 against 32 leaves 2.2e-11 in the mean and
        # 1.0e-9 in the variance, and the residual is the Laplace-Remax moment
        # accuracy rather than the Hermite rule over L. Forty-eight against
        # sixty-four is where it bottoms out.
        coarse = remax_scale_tilt_update(
            0.1, 0.25, *state, 0, epsilon=EPSILON, kappa=KAPPA, scale_order=20
        )
        medium = remax_scale_tilt_update(
            0.1, 0.25, *state, 0, epsilon=EPSILON, kappa=KAPPA, scale_order=32
        )
        fine = remax_scale_tilt_update(
            0.1, 0.25, *state, 0, epsilon=EPSILON, kappa=KAPPA, scale_order=48
        )
        finer = remax_scale_tilt_update(
            0.1, 0.25, *state, 0, epsilon=EPSILON, kappa=KAPPA, scale_order=64
        )
        assert abs(medium[0] - coarse[0]) < 1e-9
        assert abs(medium[1] - coarse[1]) < 1e-7
        assert finer[0] == pytest.approx(fine[0], abs=1e-14)
        assert finer[1] == pytest.approx(fine[1], abs=1e-13)

    def test_the_variance_decrement_is_capped_rather_than_floored(self):
        # A covariance at the Cauchy--Schwarz bound would remove the whole
        # prior variance; the cap leaves a thousandth of it so the belief can
        # still move on the next label.
        probabilities = torch.tensor([0.5, 0.5], dtype=torch.float64)
        bound = math.sqrt(0.2 * 0.25)
        cov_scale = torch.tensor([bound, -bound], dtype=torch.float64)
        _, updated = remax_scale_event_update(0.0, 0.2, probabilities, cov_scale, 0)
        assert updated == pytest.approx(0.2 * (1.0 - 0.999), abs=1e-15)
        assert updated > 0.0

    def test_an_inconsistent_covariance_is_clamped_to_the_bound(self):
        probabilities = torch.tensor([0.5, 0.5], dtype=torch.float64)
        bound = math.sqrt(0.2 * 0.25)
        cov_scale = torch.tensor([10.0 * bound, -10.0 * bound], dtype=torch.float64)
        assert float(remax_scale_consistency_slack(0.2, probabilities, cov_scale).min()) < 0.0
        mean, _ = remax_scale_event_update(0.0, 0.2, probabilities, cov_scale, 0)
        assert mean == pytest.approx(bound / 0.5, abs=1e-15)

    def test_a_nonfinite_probability_leaves_the_belief_alone(self):
        probabilities = torch.tensor([float("nan"), 0.5], dtype=torch.float64)
        cov_scale = torch.tensor([0.1, -0.1], dtype=torch.float64)
        assert remax_scale_event_update(0.2, 0.3, probabilities, cov_scale, 0) == (0.2, 0.3)
        assert remax_scale_full_update(0.2, 0.3, probabilities, cov_scale, 0) == (0.2, 0.3)

    def test_an_out_of_range_class_is_rejected(self):
        probabilities = torch.tensor([0.4, 0.6], dtype=torch.float64)
        cov_scale = torch.tensor([0.1, -0.1], dtype=torch.float64)
        with pytest.raises(ValueError, match="out of range"):
            remax_scale_event_update(0.0, 0.2, probabilities, cov_scale, 2)
        with pytest.raises(ValueError, match="out of range"):
            remax_scale_full_update(0.0, 0.2, probabilities, cov_scale, -1)

    def test_a_mismatched_covariance_length_is_rejected(self):
        probabilities = torch.tensor([0.4, 0.6], dtype=torch.float64)
        with pytest.raises(ValueError, match="cov_scale must have 2 classes"):
            remax_scale_event_update(
                0.0, 0.2, probabilities, torch.zeros(3, dtype=torch.float64), 0
            )

    def test_a_negative_state_variance_is_rejected(self):
        probabilities = torch.tensor([0.4, 0.6], dtype=torch.float64)
        cov_scale = torch.tensor([0.1, -0.1], dtype=torch.float64)
        with pytest.raises(ValueError, match="variance must be finite and nonnegative"):
            remax_scale_event_update(0.0, -0.1, probabilities, cov_scale, 0)

    def test_an_invalid_decrement_cap_is_rejected(self):
        probabilities = torch.tensor([0.4, 0.6], dtype=torch.float64)
        cov_scale = torch.tensor([0.1, -0.1], dtype=torch.float64)
        with pytest.raises(ValueError, match=r"variance_decrement_cap must lie in \(0, 1\)"):
            remax_scale_event_update(
                0.0, 0.2, probabilities, cov_scale, 0, variance_decrement_cap=1.0
            )

    def test_the_tilt_update_takes_one_row_at_a_time(self):
        state = pinned_state(3)
        with pytest.raises(ValueError, match="one calibration row at a time"):
            remax_scale_tilt_update(0.0, 0.25, *state, 0, epsilon=EPSILON, kappa=KAPPA)


# ──────────────────────────────────────────────────────────────────────────────
#  Sequential assumed-density calibration
# ──────────────────────────────────────────────────────────────────────────────


class TestSequentialCalibration:
    @pytest.mark.parametrize("method", REMAX_SCALE_METHODS)
    def test_every_method_moves_the_prior_towards_the_generating_scale(self, method):
        labels = sample_pinned_labels(24, seed=140)
        state = pinned_state(24)
        belief = calibrate_remax_log_scale_adf(
            *state, labels, epsilon=EPSILON, kappa=KAPPA, method=method, **FAST_ORDERS
        )
        assert math.isfinite(belief.mean)
        assert 0.0 < belief.variance <= DEFAULT_PRIOR_VARIANCE
        assert abs(belief.mean - PINNED_LOG_SCALE) < abs(0.0 - PINNED_LOG_SCALE)
        assert belief.method == f"adf-{method}"
        assert belief.samples == 24.0

    def test_reordering_the_stream_perturbs_the_filtered_belief(self):
        labels = sample_pinned_labels(24, seed=141)
        state = pinned_state(24)
        forward = calibrate_remax_log_scale_adf(
            *state, labels, epsilon=EPSILON, kappa=KAPPA, **FAST_ORDERS
        )
        reversed_order = torch.arange(23, -1, -1)
        backward = calibrate_remax_log_scale_adf(
            *state,
            labels,
            epsilon=EPSILON,
            kappa=KAPPA,
            visit_order=reversed_order,
            **FAST_ORDERS,
        )
        # Assumed-density filtering is order-dependent; that is exactly why the
        # order-independent batch fit is the reference. Both must still be
        # finite and land on the same order of magnitude.
        assert forward.mean != backward.mean
        assert math.isfinite(forward.mean) and math.isfinite(backward.mean)
        assert abs(forward.mean - backward.mean) > 1e-6
        assert 0.2 < backward.mean / forward.mean < 5.0
        assert 0.2 < backward.variance / forward.variance < 5.0

    def test_process_variance_keeps_the_belief_wider(self):
        labels = sample_pinned_labels(16, seed=142)
        state = pinned_state(16)
        tight = calibrate_remax_log_scale_adf(
            *state, labels, epsilon=EPSILON, kappa=KAPPA, **FAST_ORDERS
        )
        drifting = calibrate_remax_log_scale_adf(
            *state,
            labels,
            epsilon=EPSILON,
            kappa=KAPPA,
            process_variance=0.01,
            **FAST_ORDERS,
        )
        assert drifting.variance > tight.variance

    def test_the_epoch_wise_reset_retains_the_mean_and_restores_the_prior_width(self):
        labels = sample_pinned_labels(12, seed=143)
        state = pinned_state(12)
        first = calibrate_remax_log_scale_adf(
            *state, labels, epsilon=EPSILON, kappa=KAPPA, **FAST_ORDERS
        )
        second = calibrate_remax_log_scale_adf(
            *state,
            labels,
            epsilon=EPSILON,
            kappa=KAPPA,
            initial_mean=first.mean,
            **FAST_ORDERS,
        )
        # The reset restarts from q_0, so the second pass can move the mean as
        # far as the first did. That is an adaptive-filtering heuristic and not
        # exact Bayes: the same labels are being reused.
        assert second.variance > first.variance * 0.5
        assert abs(second.mean - first.mean) > 0.0

    def test_an_empty_calibration_split_returns_the_stated_prior(self):
        state = pinned_state(0)
        belief = calibrate_remax_log_scale_adf(
            *state,
            torch.zeros(0, dtype=torch.long),
            epsilon=EPSILON,
            kappa=KAPPA,
            prior_mean=0.3,
            **FAST_ORDERS,
        )
        assert belief.mean == 0.3
        assert belief.variance == DEFAULT_PRIOR_VARIANCE

    def test_an_unknown_method_is_rejected(self):
        state = pinned_state(4)
        with pytest.raises(ValueError, match="method must be one of"):
            calibrate_remax_log_scale_adf(
                *state,
                torch.zeros(4, dtype=torch.long),
                epsilon=EPSILON,
                kappa=KAPPA,
                method="bernoulli-product",
            )

    def test_an_invalid_label_is_rejected(self):
        state = pinned_state(3)
        with pytest.raises(ValueError, match="invalid class index"):
            calibrate_remax_log_scale_adf(
                *state,
                torch.tensor([0, 1, 7]),
                epsilon=EPSILON,
                kappa=KAPPA,
                **FAST_ORDERS,
            )

    def test_a_negative_process_variance_is_rejected(self):
        state = pinned_state(2)
        with pytest.raises(ValueError, match="process_variance must be finite"):
            calibrate_remax_log_scale_adf(
                *state,
                torch.zeros(2, dtype=torch.long),
                epsilon=EPSILON,
                kappa=KAPPA,
                process_variance=-0.1,
            )

    def test_a_short_visit_order_is_rejected(self):
        state = pinned_state(4)
        with pytest.raises(ValueError, match="visit_order must be a permutation"):
            calibrate_remax_log_scale_adf(
                *state,
                torch.zeros(4, dtype=torch.long),
                epsilon=EPSILON,
                kappa=KAPPA,
                visit_order=torch.arange(3),
                **FAST_ORDERS,
            )

    def test_a_mismatched_summary_shape_is_rejected(self):
        with pytest.raises(ValueError, match="var_z must have the same shape"):
            calibrate_remax_log_scale_adf(
                PINNED_MU[None],
                PINNED_E[None, :2],
                PINNED_NU[None],
                PINNED_R[None],
                torch.zeros(1, dtype=torch.long),
                epsilon=EPSILON,
                kappa=KAPPA,
            )

    def test_a_negative_head_variance_is_rejected(self):
        with pytest.raises(ValueError, match="var_z must be finite and nonnegative"):
            calibrate_remax_log_scale_adf(
                PINNED_MU[None],
                -PINNED_E[None],
                PINNED_NU[None],
                PINNED_R[None],
                torch.zeros(1, dtype=torch.long),
                epsilon=EPSILON,
                kappa=KAPPA,
            )


# ──────────────────────────────────────────────────────────────────────────────
#  Order-independent batch reference
# ──────────────────────────────────────────────────────────────────────────────


class TestBatchReference:
    def test_the_grid_posterior_reduces_to_the_prior_without_data(self):
        state = pinned_state(0)
        grid, log_posterior = remax_scale_log_posterior_on_grid(
            *state,
            torch.zeros(0, dtype=torch.long),
            epsilon=EPSILON,
            kappa=KAPPA,
            prior_mean=0.2,
            prior_variance=0.09,
            grid_size=65,
        )
        penalty = -0.5 * (grid - 0.2).square() / 0.09
        assert torch.allclose(log_posterior, penalty, atol=1e-14)

    def test_a_data_free_fit_keeps_the_stated_prior(self):
        state = pinned_state(0)
        belief = fit_remax_log_scale(
            *state,
            torch.zeros(0, dtype=torch.long),
            epsilon=EPSILON,
            kappa=KAPPA,
            prior_mean=0.0,
            prior_variance=0.09,
            grid_size=257,
        )
        assert belief.mean == pytest.approx(0.0, abs=1e-9)
        assert belief.variance == pytest.approx(0.09, rel=1e-6)

    def test_row_multiplicities_reproduce_the_expanded_calibration_split(self):
        labels = sample_pinned_labels(120, seed=150)
        counts = torch.bincount(labels, minlength=3).double()
        grid = torch.linspace(-1.0, 2.0, 17, dtype=torch.float64)
        _, expanded = remax_scale_log_posterior_on_grid(
            *pinned_state(120), labels, epsilon=EPSILON, kappa=KAPPA, grid=grid
        )
        _, weighted = remax_scale_log_posterior_on_grid(
            *pinned_state(3),
            torch.arange(3),
            epsilon=EPSILON,
            kappa=KAPPA,
            grid=grid,
            sample_weight=counts,
        )
        assert torch.allclose(expanded, weighted, rtol=0.0, atol=1e-12)

    def test_chunking_does_not_change_the_grid_posterior(self):
        labels = sample_pinned_labels(40, seed=151)
        grid = torch.linspace(-0.5, 1.5, 9, dtype=torch.float64)
        whole = remax_scale_log_posterior_on_grid(
            *pinned_state(40), labels, epsilon=EPSILON, kappa=KAPPA, grid=grid
        )[1]
        blocked = remax_scale_log_posterior_on_grid(
            *pinned_state(40),
            labels,
            epsilon=EPSILON,
            kappa=KAPPA,
            grid=grid,
            chunk_size=7,
        )[1]
        assert torch.allclose(whole, blocked, atol=1e-12)

    def test_the_batch_fit_recovers_the_pinned_generating_scale(self):
        # The note's reproduction: the fixed three-class state, 3000 labels
        # drawn at s = 1.8, prior N(0, 0.5^2). All 3000 labels share the state,
        # so their likelihood is exactly the class counts against three rows.
        labels = sample_pinned_labels(PINNED_LABELS, seed=152)
        counts = torch.bincount(labels, minlength=3).double()
        assert float(counts.sum()) == PINNED_LABELS
        belief = fit_remax_log_scale(
            *pinned_state(3),
            torch.arange(3),
            epsilon=EPSILON,
            kappa=KAPPA,
            prior_mean=0.0,
            prior_variance=DEFAULT_PRIOR_VARIANCE,
            sample_weight=counts,
            grid_size=129,
        )
        deviation = math.sqrt(belief.variance)
        # The note reports 2.0447 for its own 3000 draws; the fit is identified
        # but only to a posterior deviation of about 0.06 nats, so a couple of
        # deviations either side of s = 1.8 is the honest tolerance.
        assert belief.mean == pytest.approx(PINNED_LOG_SCALE, abs=3.0 * deviation)
        assert 1.4 < belief.scale_median < 2.4
        assert belief.samples == float(PINNED_LABELS)

    def test_the_recovered_scale_beats_the_uncalibrated_head_out_of_sample(self):
        train = sample_pinned_labels(600, seed=153)
        held_out = sample_pinned_labels(600, seed=154)
        belief = fit_remax_log_scale(
            *pinned_state(3),
            torch.arange(3),
            epsilon=EPSILON,
            kappa=KAPPA,
            sample_weight=torch.bincount(train, minlength=3).double(),
            grid_size=129,
        )
        state = pinned_state(3)
        weights = torch.bincount(held_out, minlength=3).double()
        fitted = float(
            remax_scale_negative_log_likelihood(
                *state,
                torch.arange(3),
                belief,
                sample_weight=weights,
            )
        )
        uncalibrated = float(
            remax_scale_negative_log_likelihood(
                *state,
                torch.arange(3),
                LogScalePosterior.unit(epsilon=EPSILON, kappa=KAPPA),
                sample_weight=weights,
            )
        )
        assert fitted < uncalibrated

    def test_the_laplace_read_out_agrees_with_the_grid_moments(self):
        labels = sample_pinned_labels(400, seed=155)
        counts = torch.bincount(labels, minlength=3).double()
        common = {
            "epsilon": EPSILON,
            "kappa": KAPPA,
            "sample_weight": counts,
            "grid_size": 129,
        }
        gridded = fit_remax_log_scale(*pinned_state(3), torch.arange(3), method="grid", **common)
        laplace = fit_remax_log_scale(*pinned_state(3), torch.arange(3), method="laplace", **common)
        # A near-Gaussian scalar posterior: the mode and the mean agree well
        # inside one posterior deviation, and so do the two spreads.
        assert laplace.mean == pytest.approx(gridded.mean, abs=0.2 * math.sqrt(gridded.variance))
        assert laplace.variance == pytest.approx(gridded.variance, rel=0.1)

    def test_the_refined_fit_is_independent_of_the_grid_size(self):
        labels = sample_pinned_labels(800, seed=156)
        counts = torch.bincount(labels, minlength=3).double()
        fits = [
            fit_remax_log_scale(
                *pinned_state(3),
                torch.arange(3),
                epsilon=EPSILON,
                kappa=KAPPA,
                sample_weight=counts,
                grid_size=size,
            )
            for size in (65, 129, 257)
        ]
        for belief in fits[1:]:
            assert belief.mean == pytest.approx(fits[0].mean, abs=1e-6)
            assert belief.variance == pytest.approx(fits[0].variance, rel=1e-4)

    def test_refinement_is_what_resolves_a_sharp_posterior(self):
        labels = sample_pinned_labels(2000, seed=157)
        counts = torch.bincount(labels, minlength=3).double()
        common = {
            "epsilon": EPSILON,
            "kappa": KAPPA,
            "sample_weight": counts,
        }
        coarse = fit_remax_log_scale(
            *pinned_state(3), torch.arange(3), grid_size=33, refinements=0, **common
        )
        refined = fit_remax_log_scale(
            *pinned_state(3), torch.arange(3), grid_size=33, refinements=2, **common
        )
        resolved = fit_remax_log_scale(
            *pinned_state(3), torch.arange(3), grid_size=513, refinements=2, **common
        )
        # A 33-point sweep over [-6, 6] has a spacing of 0.375 nats, five times
        # the posterior deviation, so the unrefined moments cannot resolve the
        # peak at all: the mean lands two deviations away. Re-centring the same
        # 33 points on the mode recovers the well-resolved answer.
        deviation = math.sqrt(resolved.variance)
        assert abs(coarse.mean - resolved.mean) > 1.5 * deviation
        assert refined.mean == pytest.approx(resolved.mean, abs=1e-6)
        assert refined.variance == pytest.approx(resolved.variance, rel=1e-4)

    def test_the_batch_fit_is_the_order_independent_reference_the_filter_approaches(self):
        labels = sample_pinned_labels(24, seed=158)
        state = pinned_state(24)
        batch = fit_remax_log_scale(*state, labels, epsilon=EPSILON, kappa=KAPPA, grid_size=129)
        filtered = calibrate_remax_log_scale_adf(
            *state, labels, epsilon=EPSILON, kappa=KAPPA, **FAST_ORDERS
        )
        assert abs(filtered.mean - batch.mean) < 0.5

    def test_an_unknown_fit_method_is_rejected(self):
        state = pinned_state(3)
        with pytest.raises(ValueError, match="method must be one of"):
            fit_remax_log_scale(
                *state,
                torch.arange(3),
                epsilon=EPSILON,
                kappa=KAPPA,
                method="mle",
            )

    def test_a_negative_refinement_count_is_rejected(self):
        state = pinned_state(3)
        with pytest.raises(ValueError, match="refinements must be nonnegative"):
            fit_remax_log_scale(
                *state,
                torch.arange(3),
                epsilon=EPSILON,
                kappa=KAPPA,
                refinements=-1,
            )

    def test_a_nonpositive_prior_variance_is_rejected(self):
        state = pinned_state(3)
        with pytest.raises(ValueError, match="prior_variance must be finite and positive"):
            remax_scale_log_posterior_on_grid(
                *state,
                torch.arange(3),
                epsilon=EPSILON,
                kappa=KAPPA,
                prior_variance=0.0,
            )

    def test_a_tiny_grid_is_rejected(self):
        state = pinned_state(3)
        with pytest.raises(ValueError, match="grid_size must be at least five"):
            remax_scale_log_posterior_on_grid(
                *state, torch.arange(3), epsilon=EPSILON, kappa=KAPPA, grid_size=3
            )
        with pytest.raises(ValueError, match="grid must hold at least five points"):
            remax_scale_log_posterior_on_grid(
                *state,
                torch.arange(3),
                epsilon=EPSILON,
                kappa=KAPPA,
                grid=torch.zeros(4, dtype=torch.float64),
            )

    def test_a_negative_row_weight_is_rejected(self):
        state = pinned_state(3)
        with pytest.raises(ValueError, match="sample_weight must be finite and nonnegative"):
            remax_scale_log_posterior_on_grid(
                *state,
                torch.arange(3),
                epsilon=EPSILON,
                kappa=KAPPA,
                sample_weight=torch.tensor([1.0, -1.0, 2.0]),
                grid_size=9,
            )


# ──────────────────────────────────────────────────────────────────────────────
#  Scoring
# ──────────────────────────────────────────────────────────────────────────────


class TestScoring:
    def test_the_loss_is_the_gathered_predictive_log_probability(self):
        state = random_state(6, 4, seed=160)
        labels = torch.tensor([0, 3, 1, 2, 2, 0])
        belief = LogScalePosterior(mean=0.25, variance=0.09, epsilon=EPSILON, kappa=KAPPA)
        probabilities, _, _ = moments_at(state, belief.mean, belief.variance)
        expected = -probabilities.gather(1, labels[:, None]).squeeze(1).log().mean()
        result = remax_scale_negative_log_likelihood(*state, labels, belief)
        assert result == pytest.approx(float(expected), abs=1e-14)

    def test_chunking_does_not_change_the_loss(self):
        state = random_state(9, 3, seed=161)
        labels = torch.tensor([0, 1, 2, 2, 1, 0, 1, 2, 0])
        belief = LogScalePosterior(mean=0.1, variance=0.04, epsilon=EPSILON, kappa=KAPPA)
        whole = remax_scale_negative_log_likelihood(*state, labels, belief)
        blocked = remax_scale_negative_log_likelihood(*state, labels, belief, chunk_size=2)
        assert torch.allclose(whole, blocked, atol=1e-14)

    def test_scoring_under_another_cdf_head_is_rejected(self):
        state = random_state(3, 3, seed=162)
        labels = torch.tensor([0, 1, 2])
        belief = LogScalePosterior(mean=0.0, variance=0.04, epsilon=EPSILON, kappa=KAPPA)
        with pytest.raises(ValueError, match="cannot be read at kappa"):
            remax_scale_negative_log_likelihood(*state, labels, belief, epsilon=EPSILON, kappa=3.0)

    def test_scoring_under_another_scale_order_is_rejected(self):
        state = random_state(3, 3, seed=163)
        labels = torch.tensor([0, 1, 2])
        belief = LogScalePosterior(mean=0.0, variance=0.04, epsilon=EPSILON, kappa=KAPPA)
        with pytest.raises(ValueError, match="cannot be read at scale_order"):
            remax_scale_negative_log_likelihood(*state, labels, belief, scale_order=48)

    def test_the_deterministic_read_out_differs_from_the_integrated_one(self):
        state = random_state(8, 5, seed=164)
        labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
        belief = LogScalePosterior(mean=0.3, variance=0.25, epsilon=EPSILON, kappa=KAPPA)
        integrated = float(remax_scale_negative_log_likelihood(*state, labels, belief))
        point = float(remax_scale_negative_log_likelihood(*state, labels, belief.deterministic()))
        # Remax is nonlinear in the scale, so integrating the calibration
        # posterior is not the same as predicting at its median.
        assert integrated != point
        assert abs(integrated - point) > 1e-6
