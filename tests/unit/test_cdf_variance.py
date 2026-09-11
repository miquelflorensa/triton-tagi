"""Positive, bounded aleatoric variance from a Gaussian CDF activation."""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi.cdf_variance import (
    cdf_squared_error_moments,
    cdf_variance_activation,
    cdf_variance_bridge,
    cdf_variance_moments,
    upstream_cdf_covariance,
)
from triton_tagi.hsm_calibration import owens_t

DOUBLE = torch.float64
EPSILON, KAPPA = 0.02, 1.5

# The state the accompanying note uses for its numerical verification.
NOTE_NU = torch.tensor([-0.7, 0.1, -0.3], dtype=DOUBLE)
NOTE_R = torch.tensor([0.4, 0.2, 0.3], dtype=DOUBLE)


def _draw(nu: torch.Tensor, r: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(count, nu.numel(), generator=generator, dtype=DOUBLE)
    return nu + r.sqrt() * noise


# ──────────────────────────────────────────────────────────────────────────────
#  The activation
# ──────────────────────────────────────────────────────────────────────────────


def test_activation_is_bounded_on_both_sides() -> None:
    """``epsilon <= h(u) <= epsilon + kappa`` over an extreme range.

    Two-sided boundedness is the whole point of preferring the CDF over the
    exponential head. The inequality is strict mathematically, but ``Phi``
    underflows to exactly zero and rounds to exactly one in float64, so the
    realised variance saturates at both ends of the range.
    """

    u = torch.tensor([-1e3, -40.0, -1.0, 0.0, 1.0, 40.0, 1e3], dtype=DOUBLE)
    value = cdf_variance_activation(u, epsilon=EPSILON, kappa=KAPPA)
    assert bool((value >= EPSILON).all())
    assert bool((value <= EPSILON + KAPPA).all())
    assert float(value[3]) == pytest.approx(EPSILON + 0.5 * KAPPA, abs=1e-15)

    # Strictness survives the float64 sum only while kappa Phi(u) stays above
    # half an ulp of epsilon, which holds for |u| below roughly eight.
    interior = cdf_variance_activation(
        torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=DOUBLE),
        epsilon=EPSILON,
        kappa=KAPPA,
    )
    assert bool((interior > EPSILON).all())
    assert bool((interior < EPSILON + KAPPA).all())

    # Saturation is reached, which is the regime the head must be watched for.
    assert float(value[0]) == EPSILON
    assert float(value[-1]) == EPSILON + KAPPA


@pytest.mark.parametrize(("epsilon", "kappa"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)])
def test_constants_must_be_strictly_positive(epsilon: float, kappa: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        cdf_variance_activation(torch.zeros(1, dtype=DOUBLE), epsilon=epsilon, kappa=kappa)


# ──────────────────────────────────────────────────────────────────────────────
#  Exact forward moments
# ──────────────────────────────────────────────────────────────────────────────


def test_moments_match_the_owens_t_form() -> None:
    """``b = kappa^2 [Phi_2(a, a; rho) - Phi(a)^2]`` via Owen's T.

    An independent route to the same quantity: ``Phi_2(a, a; rho) = Phi(a) -
    2 T(a, sqrt[(1 - rho) / (1 + rho)])``, and with ``rho = r / (1 + r)`` that
    second argument collapses to ``1 / sqrt(1 + 2 r)``.
    """

    h_mean, h_variance, _ = cdf_variance_moments(NOTE_NU, NOTE_R, epsilon=EPSILON, kappa=KAPPA)
    argument = NOTE_NU / (1.0 + NOTE_R).sqrt()
    upper = torch.special.ndtr(argument)
    bivariate = upper - 2.0 * owens_t(argument, 1.0 / (1.0 + 2.0 * NOTE_R).sqrt())

    torch.testing.assert_close(h_mean, EPSILON + KAPPA * upper, atol=1e-15, rtol=0.0)
    torch.testing.assert_close(
        h_variance, KAPPA**2 * (bivariate - upper * upper), atol=1e-12, rtol=0.0
    )


def test_moments_match_monte_carlo() -> None:
    sample = _draw(NOTE_NU, NOTE_R, 4_000_000, seed=3)
    realised = cdf_variance_activation(sample, epsilon=EPSILON, kappa=KAPPA)
    h_mean, h_variance, cov_u_h = cdf_variance_moments(
        NOTE_NU, NOTE_R, epsilon=EPSILON, kappa=KAPPA
    )

    torch.testing.assert_close(h_mean, realised.mean(dim=0), atol=0.0, rtol=5e-3)
    torch.testing.assert_close(h_variance, realised.var(dim=0), atol=0.0, rtol=2e-2)
    empirical = ((sample - NOTE_NU) * (realised - realised.mean(dim=0))).mean(dim=0)
    torch.testing.assert_close(cov_u_h, empirical, atol=0.0, rtol=5e-3)


def test_covariance_is_the_stein_derivative_of_the_mean() -> None:
    """``Cov(U, h(U)) = r dE[h]/dnu`` exactly, by Stein's identity."""

    step = 1e-6
    _, _, cov_u_h = cdf_variance_moments(NOTE_NU, NOTE_R, epsilon=EPSILON, kappa=KAPPA)
    upper, _, _ = cdf_variance_moments(NOTE_NU + step, NOTE_R, epsilon=EPSILON, kappa=KAPPA)
    lower, _, _ = cdf_variance_moments(NOTE_NU - step, NOTE_R, epsilon=EPSILON, kappa=KAPPA)
    torch.testing.assert_close(
        cov_u_h, NOTE_R * (upper - lower) / (2.0 * step), atol=0.0, rtol=1e-8
    )


def test_a_degenerate_head_has_no_variance_and_no_covariance() -> None:
    zero = torch.zeros_like(NOTE_NU)
    h_mean, h_variance, cov_u_h = cdf_variance_moments(NOTE_NU, zero, epsilon=EPSILON, kappa=KAPPA)
    torch.testing.assert_close(
        h_mean,
        cdf_variance_activation(NOTE_NU, epsilon=EPSILON, kappa=KAPPA),
        atol=1e-15,
        rtol=0.0,
    )
    assert float(h_variance.abs().max()) == 0.0
    assert float(cov_u_h.abs().max()) == 0.0


def test_variance_respects_its_bernoulli_style_bound() -> None:
    """``Var[Phi(U)] <= Phi(a) (1 - Phi(a))`` for every prior."""

    generator = torch.Generator().manual_seed(11)
    nu = 4.0 * torch.randn(256, generator=generator, dtype=DOUBLE)
    r = 10.0 * torch.rand(256, generator=generator, dtype=DOUBLE)
    h_mean, h_variance, _ = cdf_variance_moments(nu, r, epsilon=EPSILON, kappa=KAPPA)
    probability = (h_mean - EPSILON) / KAPPA
    assert bool((h_variance >= 0.0).all())
    assert bool((h_variance <= KAPPA**2 * probability * (1.0 - probability) + 1e-15).all())


@pytest.mark.parametrize("order", [24, 48, 96])
def test_moments_are_stable_in_the_quadrature_order(order: int) -> None:
    reference = cdf_variance_moments(NOTE_NU, NOTE_R, epsilon=EPSILON, kappa=KAPPA, order=384)
    value = cdf_variance_moments(NOTE_NU, NOTE_R, epsilon=EPSILON, kappa=KAPPA, order=order)
    for computed, expected in zip(value, reference, strict=True):
        assert torch.allclose(computed, expected, atol=1e-15)


# ──────────────────────────────────────────────────────────────────────────────
#  Upstream covariance and the Gaussian bridge
# ──────────────────────────────────────────────────────────────────────────────


def test_upstream_covariance_reduces_to_the_head_covariance() -> None:
    """Setting ``Cov(T, U) = r`` must reproduce ``Cov(U, V2bar)``."""

    _, _, cov_u_h = cdf_variance_moments(NOTE_NU, NOTE_R, epsilon=EPSILON, kappa=KAPPA)
    value = upstream_cdf_covariance(NOTE_R, NOTE_NU, NOTE_R, kappa=KAPPA)
    torch.testing.assert_close(value, cov_u_h, atol=1e-15, rtol=0.0)


def test_upstream_covariance_is_linear_in_the_input_covariance() -> None:
    base = upstream_cdf_covariance(torch.ones_like(NOTE_NU), NOTE_NU, NOTE_R, kappa=KAPPA)
    scaled = upstream_cdf_covariance(3.0 * torch.ones_like(NOTE_NU), NOTE_NU, NOTE_R, kappa=KAPPA)
    torch.testing.assert_close(scaled, 3.0 * base, atol=1e-15, rtol=0.0)


def test_bridge_is_inert_when_the_posterior_equals_the_prior() -> None:
    h_mean, h_variance, _ = cdf_variance_moments(NOTE_NU, NOTE_R, epsilon=EPSILON, kappa=KAPPA)
    nu_post, r_post = cdf_variance_bridge(
        NOTE_NU, NOTE_R, h_mean, h_variance, epsilon=EPSILON, kappa=KAPPA
    )
    torch.testing.assert_close(nu_post, NOTE_NU, atol=1e-14, rtol=0.0)
    torch.testing.assert_close(r_post, NOTE_R, atol=1e-14, rtol=0.0)


def test_bridge_moves_the_head_towards_a_larger_observed_variance() -> None:
    h_mean, h_variance, _ = cdf_variance_moments(NOTE_NU, NOTE_R, epsilon=EPSILON, kappa=KAPPA)
    nu_post, r_post = cdf_variance_bridge(
        NOTE_NU,
        NOTE_R,
        h_mean + 0.05,
        0.5 * h_variance,
        epsilon=EPSILON,
        kappa=KAPPA,
    )
    assert bool((nu_post > NOTE_NU).all())
    assert bool((r_post < NOTE_R).all())
    assert bool((r_post >= 0.0).all())


def test_bridge_uses_a_zero_gain_for_a_deterministic_head() -> None:
    """``b = 0`` must take the degenerate limit, not divide by zero."""

    zero = torch.zeros_like(NOTE_NU)
    nu_post, r_post = cdf_variance_bridge(
        NOTE_NU,
        zero,
        torch.full_like(NOTE_NU, 0.9),
        torch.full_like(NOTE_NU, 0.1),
        epsilon=EPSILON,
        kappa=KAPPA,
    )
    torch.testing.assert_close(nu_post, NOTE_NU, atol=1e-15, rtol=0.0)
    torch.testing.assert_close(r_post, zero, atol=1e-15, rtol=0.0)


# ──────────────────────────────────────────────────────────────────────────────
#  Fourth moments of the hierarchical noise
# ──────────────────────────────────────────────────────────────────────────────


def test_squared_error_moments_match_monte_carlo() -> None:
    """``Var(V^2) = 2 h^2 + 3 b``, distinctly not ``2 h^2``."""

    torch.manual_seed(7)
    count = 4_000_000
    sample = _draw(NOTE_NU, NOTE_R, count, seed=5)
    realised = cdf_variance_activation(sample, epsilon=EPSILON, kappa=KAPPA)
    noise = realised.sqrt() * torch.randn(count, NOTE_NU.numel(), dtype=DOUBLE)

    h_mean, h_variance, _ = cdf_variance_moments(NOTE_NU, NOTE_R, epsilon=EPSILON, kappa=KAPPA)
    squared_variance, cross = cdf_squared_error_moments(h_mean, h_variance)

    squared = noise * noise
    torch.testing.assert_close(squared.mean(dim=0), h_mean, atol=0.0, rtol=1e-2)
    torch.testing.assert_close(squared.var(dim=0), squared_variance, atol=0.0, rtol=5e-2)
    empirical = ((realised - h_mean) * (squared - squared.mean(dim=0))).mean(dim=0)
    torch.testing.assert_close(cross, empirical, atol=0.0, rtol=8e-2)

    # Discarding b would understate the spread of the squared residual.
    assert bool((squared_variance > 2.0 * h_mean * h_mean).all())


def test_noise_is_uncorrelated_with_the_head_that_generates_it() -> None:
    """``Cov(U, V) = 0`` even though ``U`` sets the distribution of ``V``.

    This is why a linear Gaussian update on the raw residual cannot identify
    the variance head on its own.
    """

    count = 2_000_000
    sample = _draw(NOTE_NU, NOTE_R, count, seed=9)
    realised = cdf_variance_activation(sample, epsilon=EPSILON, kappa=KAPPA)
    generator = torch.Generator().manual_seed(13)
    noise = realised.sqrt() * torch.randn(count, NOTE_NU.numel(), generator=generator, dtype=DOUBLE)
    empirical = ((sample - NOTE_NU) * noise).mean(dim=0)
    bound = 4.0 * (NOTE_R.sqrt() * realised.sqrt().mean(dim=0)) / math.sqrt(count)
    assert bool((empirical.abs() < bound).all())
