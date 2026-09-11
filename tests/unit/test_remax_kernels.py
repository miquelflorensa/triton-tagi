"""Stable Gaussian kernels for the Laplace-Remax identities."""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi.remax_kernels import (
    TAIL_CUTOFF,
    hermite_rule,
    laplace_rule,
    rectified_exponential_integrals,
    rectified_zero_probability,
    remax_kernels,
    scaled_rectified_integrals,
)

DOUBLE = torch.float64
_INV_SQRT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)


def _dense_kernel_reference(
    mean: float,
    deviation: float,
    t: float,
    *,
    reach: float = 40.0,
    nodes: int = 2_000_001,
) -> tuple[float, float, float]:
    """Brute-force ``E[M^n exp(-tM)]`` on a fine grid, independent of quadrature."""

    x = torch.linspace(mean - reach * deviation, mean + reach * deviation, nodes, dtype=DOUBLE)
    density = torch.exp(-0.5 * ((x - mean) / deviation) ** 2) / (
        deviation * math.sqrt(2.0 * math.pi)
    )
    rectified = x.clamp_min(0.0)
    decay = torch.exp(-t * rectified)
    return (
        float(torch.trapz(density * decay, x)),
        float(torch.trapz(density * rectified * decay, x)),
        float(torch.trapz(density * rectified * rectified * decay, x)),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Quadrature rules
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("map_scale", [0.5, 1.0, 2.5])
def test_laplace_rule_integrates_the_exponential_to_one(map_scale: float) -> None:
    t, weights = laplace_rule(160, map_scale=map_scale)
    total = float((weights * torch.exp(-t)).sum())
    assert total == pytest.approx(1.0, abs=1e-13)


def test_laplace_rule_rejects_a_non_positive_map_scale() -> None:
    with pytest.raises(ValueError, match="map_scale"):
        laplace_rule(64, map_scale=0.0)


def test_hermite_rule_weights_sum_to_one_and_reproduce_gaussian_moments() -> None:
    nodes, weights = hermite_rule(32)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-15)
    assert float((weights * nodes).sum()) == pytest.approx(0.0, abs=1e-14)
    assert float((weights * nodes * nodes).sum()) == pytest.approx(1.0, abs=1e-13)
    assert float((weights * nodes**4).sum()) == pytest.approx(3.0, abs=1e-12)


@pytest.mark.parametrize("order", [3, 0])
def test_quadrature_rules_reject_a_tiny_order(order: int) -> None:
    with pytest.raises(ValueError, match="at least four"):
        hermite_rule(order)


# ──────────────────────────────────────────────────────────────────────────────
#  I_n(beta)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("beta", [2.0, 0.5, 0.0, -1.0, -3.0])
def test_zeroth_integral_matches_the_closed_form(beta: float) -> None:
    """``I_0 = sqrt(2 pi) exp(beta^2 / 2) Phi(beta)``, an independent route.

    Only usable for moderate ``beta``: the reference multiplies an overflowing
    exponential by an underflowing CDF, and ``ndtr`` reaches exactly zero near
    ``-18``. The tail is covered by dense integration instead.
    """

    value = rectified_exponential_integrals(torch.tensor(beta, dtype=DOUBLE), highest=0)[0]
    argument = torch.tensor(beta, dtype=DOUBLE)
    reference = (
        math.sqrt(2.0 * math.pi) * torch.exp(0.5 * argument**2) * torch.special.ndtr(argument)
    )
    assert float(value) == pytest.approx(float(reference), rel=1e-13)


@pytest.mark.parametrize("beta", [1.5, 0.0, -2.0, -10.0, -25.0])
def test_integrals_satisfy_the_by_parts_recursion(beta: float) -> None:
    integrals = rectified_exponential_integrals(torch.tensor(beta, dtype=DOUBLE), highest=4)
    for order in range(2, 5):
        expected = beta * float(integrals[order - 1]) + (order - 1) * float(integrals[order - 2])
        assert float(integrals[order]) == pytest.approx(expected, rel=1e-11)


def test_integrals_match_dense_integration() -> None:
    """Dense integration only resolves a peak of width ``1 / |beta|``.

    Past about ``-12`` the grid would need tens of millions of nodes, so the
    deep tail is pinned against the asymptotic series instead.
    """

    y = torch.linspace(0.0, 60.0, 3_000_001, dtype=DOUBLE)
    for beta in (1.0, 0.0, -3.0, -12.0):
        integrals = rectified_exponential_integrals(torch.tensor(beta, dtype=DOUBLE), highest=2)
        weight = torch.exp(-0.5 * y * y + beta * y)
        for order in range(3):
            reference = float(torch.trapz(y**order * weight, y))
            assert float(integrals[order]) == pytest.approx(reference, rel=1e-8)


@pytest.mark.parametrize("beta", [-25.0, -40.0, -60.0, -100.0, -1000.0])
def test_integrals_match_an_independently_written_series_in_the_tail(beta: float) -> None:
    """Reference: ``I_n(-B) = sum_k (-1)^k (n+2k)! / [k! 2^k B^(n+2k+1)]``.

    Written out here from the defining substitution rather than reusing the
    module's own coefficients, so it is a genuinely separate implementation.
    """

    reach = -beta
    integrals = rectified_exponential_integrals(torch.tensor(beta, dtype=DOUBLE), highest=2)
    for order in range(3):
        reference = math.fsum(
            (-1.0) ** k
            * math.factorial(order + 2 * k)
            / (math.factorial(k) * 2.0**k * reach ** (order + 2 * k + 1))
            for k in range(7)
        )
        assert float(integrals[order]) == pytest.approx(reference, rel=1e-9)


def test_series_and_recursion_agree_across_the_tail_cutoff() -> None:
    """The two branches must overlap, or the switch would introduce a step."""

    below = torch.tensor(TAIL_CUTOFF - 1e-9, dtype=DOUBLE)
    above = torch.tensor(TAIL_CUTOFF + 1e-9, dtype=DOUBLE)
    left = rectified_exponential_integrals(below, highest=2)
    right = rectified_exponential_integrals(above, highest=2)
    for order in range(3):
        assert float(left[order]) == pytest.approx(float(right[order]), rel=1e-9)


def test_integrals_reject_a_negative_order() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rectified_exponential_integrals(torch.zeros(1, dtype=DOUBLE), highest=-1)


# ──────────────────────────────────────────────────────────────────────────────
#  phi(alpha) I_n(beta)
# ──────────────────────────────────────────────────────────────────────────────


def test_scaled_integrals_equal_the_factored_form_where_both_are_finite() -> None:
    alpha = torch.tensor([0.5, 2.0, 5.0, -1.0], dtype=DOUBLE)
    beta = torch.tensor([-1.0, -5.0, 0.3, -8.0], dtype=DOUBLE)
    scaled = scaled_rectified_integrals(alpha, beta, highest=2)
    pure = rectified_exponential_integrals(beta, highest=2)
    density = _INV_SQRT_TWO_PI * torch.exp(-0.5 * alpha * alpha)
    for order in range(3):
        torch.testing.assert_close(scaled[order], density * pure[order], atol=0.0, rtol=1e-9)


def test_scaled_zeroth_integral_is_a_probability() -> None:
    """``phi(alpha) I_0(beta) = E[exp(-tM) 1{X > 0}]`` lies in ``[0, 1]``."""

    alpha = torch.linspace(-40.0, 40.0, 401, dtype=DOUBLE)
    beta = alpha - 3.0
    value = scaled_rectified_integrals(alpha, beta, highest=0)[0]
    assert bool((value >= 0.0).all())
    assert bool((value <= 1.0 + 1e-15).all())


def test_scaled_integrals_stay_finite_where_the_factored_form_overflows() -> None:
    """The regime that motivates carrying the product: 0 * inf."""

    alpha = torch.tensor([30.0, 60.0, 200.0], dtype=DOUBLE)
    density = _INV_SQRT_TWO_PI * torch.exp(-0.5 * alpha * alpha)
    naive = rectified_exponential_integrals(alpha, highest=2)[0]
    assert bool((density == 0.0).any()) and bool(torch.isinf(naive).any())

    scaled = scaled_rectified_integrals(alpha, alpha, highest=2)
    for tensor in scaled:
        assert bool(torch.isfinite(tensor).all())


# ──────────────────────────────────────────────────────────────────────────────
#  Kernels
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mean", "deviation"),
    [(0.8, 0.6), (-0.3, 0.3), (2.0, 0.1), (0.0, 1.0), (-2.0, 0.5)],
)
def test_kernels_at_zero_are_the_rectified_gaussian_moments(mean: float, deviation: float) -> None:
    """At ``t = 0`` the kernels collapse to ``1``, ``E[M]`` and ``E[M^2]``."""

    kernel_l, kernel_d, kernel_f = remax_kernels(
        torch.tensor(0.0, dtype=DOUBLE),
        torch.tensor(mean, dtype=DOUBLE),
        torch.tensor(deviation, dtype=DOUBLE),
    )
    argument = torch.tensor(mean / deviation, dtype=DOUBLE)
    density = _INV_SQRT_TWO_PI * torch.exp(-0.5 * argument**2)
    upper = torch.special.ndtr(argument)
    expected_first = deviation * float(density) + mean * float(upper)
    expected_second = (mean**2 + deviation**2) * float(upper) + mean * deviation * float(density)

    assert float(kernel_l) == pytest.approx(1.0, abs=1e-14)
    assert float(kernel_d) == pytest.approx(expected_first, rel=1e-12)
    assert float(kernel_f) == pytest.approx(expected_second, rel=1e-12)


@pytest.mark.parametrize("t", [0.25, 1.0, 5.0])
def test_kernels_match_dense_integration(t: float) -> None:
    mean, deviation = 0.8, 0.6
    kernels = remax_kernels(
        torch.tensor(t, dtype=DOUBLE),
        torch.tensor(mean, dtype=DOUBLE),
        torch.tensor(deviation, dtype=DOUBLE),
    )
    reference = _dense_kernel_reference(mean, deviation, t)
    for value, expected in zip(kernels, reference, strict=True):
        assert float(value) == pytest.approx(expected, rel=1e-7)


def test_kernels_reach_the_correct_large_argument_asymptotes() -> None:
    """As ``t -> inf`` mass concentrates at ``M = 0``: ``D ~ phi(alpha) / (d t^2)``."""

    mean, deviation, t = 0.8, 0.6, 5.0e3
    kernel_l, kernel_d, kernel_f = remax_kernels(
        torch.tensor(t, dtype=DOUBLE),
        torch.tensor(mean, dtype=DOUBLE),
        torch.tensor(deviation, dtype=DOUBLE),
    )
    argument = mean / deviation
    density = _INV_SQRT_TWO_PI * math.exp(-0.5 * argument**2)
    assert float(kernel_l) == pytest.approx(
        float(torch.special.ndtr(torch.tensor(-argument, dtype=DOUBLE))), rel=1e-3
    )
    assert float(kernel_d) == pytest.approx(density / deviation / t**2, rel=1e-3)
    assert float(kernel_f) == pytest.approx(2.0 * density / deviation / t**3, rel=1e-3)


def test_kernels_are_finite_and_ordered_over_extreme_regimes() -> None:
    deviation = torch.tensor([1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0], dtype=DOUBLE).view(-1, 1)
    t = torch.tensor([0.0, 1e-3, 1.0, 1e2, 1e4], dtype=DOUBLE).view(1, -1)
    for mean in (-5.0, 0.0, 0.8, 50.0):
        kernel_l, kernel_d, kernel_f = remax_kernels(t, torch.tensor(mean, dtype=DOUBLE), deviation)
        for tensor in (kernel_l, kernel_d, kernel_f):
            assert bool(torch.isfinite(tensor).all())
            assert bool((tensor >= 0.0).all())
        # exp(-tM) is non-increasing in t, so every kernel is too.
        assert bool((kernel_l.diff(dim=-1) <= 1e-15).all())
        assert bool((kernel_d.diff(dim=-1) <= 1e-15).all())


def test_zero_probability_is_the_complementary_gaussian_cdf() -> None:
    mean = torch.tensor([0.8, -0.3, 0.0], dtype=DOUBLE)
    deviation = torch.tensor([0.6, 0.3, 1.0], dtype=DOUBLE)
    value = rectified_zero_probability(mean, deviation)
    torch.testing.assert_close(value, torch.special.ndtr(-(mean / deviation)), atol=1e-15, rtol=0.0)
