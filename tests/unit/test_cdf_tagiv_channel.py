"""Checks for the Gaussian training channel with the CDF variance head.

The forward layer is verified against :mod:`triton_tagi.cdf_variance`, and the
observation update is verified twice over: once against dense Simpson
integration of the exact posterior integrals on a fine ``u`` grid, which is
independent of the Gauss--Hermite rule the implementation uses, and once against
importance-sampling Monte Carlo, which is independent of both.
"""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi import EvenProbit
from triton_tagi.cdf_variance import cdf_variance_activation, cdf_variance_moments
from triton_tagi.update.observation import compute_cdf_tagiv_innovation

DOUBLE = torch.float64

EPSILON = 0.05
KAPPA = 2.0


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _interleave(even: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
    """Pack an even and an odd stream into the interleaved 2K layout."""

    packed = torch.empty(*even.shape[:-1], 2 * even.shape[-1], dtype=even.dtype)
    packed[..., 0::2] = even
    packed[..., 1::2] = odd
    return packed


def _posteriors_from_deltas(delta_mu, delta_var, mu_z, var_z, nu, r):
    """Undo the normalized delta convention to recover posterior moments."""

    mu_z_post = mu_z + var_z * delta_mu[..., 0::2]
    var_z_post = var_z + var_z.square() * delta_var[..., 0::2]
    nu_post = nu + r * delta_mu[..., 1::2]
    r_post = r + r.square() * delta_var[..., 1::2]
    return mu_z_post, var_z_post, nu_post, r_post


def _simpson(values, step):
    """Composite Simpson quadrature over an odd number of equispaced samples."""

    coefficients = torch.full_like(values, 2.0)
    coefficients[1::2] = 4.0
    coefficients[0] = 1.0
    coefficients[-1] = 1.0
    return (step / 3.0) * (coefficients * values).sum()


def _dense_reference(targets, mu_z, var_z, nu, r, *, points=200_001, span=16.0):
    """Exact posterior moments by dense Simpson integration on a fine ``u`` grid.

    Evaluates the defining integrals of the note literally -- including the
    prior density inside ``omega`` -- so agreement confirms the Gauss--Hermite
    placement rather than restating it. Simpson rather than the trapezoid rule
    because the trapezoid error at this spacing, order ``1e-8`` relative, is
    larger than the quadrature difference under test.
    """

    mu_z_post = torch.empty_like(mu_z)
    var_z_post = torch.empty_like(var_z)
    nu_post = torch.empty_like(nu)
    r_post = torch.empty_like(r)
    flat = [t.reshape(-1) for t in (targets, mu_z, var_z, nu, r)]
    out = [t.reshape(-1) for t in (mu_z_post, var_z_post, nu_post, r_post)]

    for index in range(flat[0].numel()):
        y, mu, e, n, v = (float(t[index]) for t in flat)
        width = span * math.sqrt(v)
        grid = torch.linspace(n - width, n + width, points, dtype=DOUBLE)
        step = float(grid[1] - grid[0])
        noise = cdf_variance_activation(grid, epsilon=EPSILON, kappa=KAPPA)
        d = e + noise
        prior = torch.exp(-0.5 * (grid - n) ** 2 / v) / math.sqrt(2.0 * math.pi * v)
        likelihood = torch.exp(-0.5 * (y - mu) ** 2 / d) / torch.sqrt(2.0 * math.pi * d)
        omega = prior * likelihood

        def integrate(values, omega=omega, step=step):
            return _simpson(values * omega, step)

        mass = integrate(torch.ones_like(grid))
        head_mean = integrate(grid) / mass
        head_second = integrate(grid.square()) / mass

        conditional_mean = mu + e * (y - mu) / d
        conditional_var = e * noise / d
        z_mean = integrate(conditional_mean) / mass
        z_second = integrate(conditional_var + conditional_mean.square()) / mass

        out[0][index] = z_mean
        out[1][index] = z_second - z_mean.square()
        out[2][index] = head_mean
        out[3][index] = head_second - head_mean.square()

    return mu_z_post, var_z_post, nu_post, r_post


# ---------------------------------------------------------------------------
#  EvenProbit layer
# ---------------------------------------------------------------------------


def test_even_probit_forward_matches_cdf_variance_moments():
    mz = torch.tensor([[0.4, -0.3, -1.2, 0.9]], dtype=DOUBLE)
    Sz = torch.tensor([[0.5, 0.7, 0.2, 1.5]], dtype=DOUBLE)
    layer = EvenProbit(2, epsilon=EPSILON, kappa=KAPPA)

    ma, Sa = layer.forward(mz, Sz)
    h_mean, h_variance, cov_u_h = cdf_variance_moments(
        mz[..., 1::2], Sz[..., 1::2], epsilon=EPSILON, kappa=KAPPA
    )

    torch.testing.assert_close(ma[..., 1::2], h_mean, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(Sa[..., 1::2], h_variance, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(ma[..., 0::2], mz[..., 0::2], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(Sa[..., 0::2], Sz[..., 0::2], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(layer.cov_u_h, cov_u_h, rtol=1e-10, atol=1e-12)


def test_even_probit_forward_output_is_bounded_by_the_activation_range():
    mz = torch.tensor([[0.0, -6.0, 0.0, 6.0]], dtype=DOUBLE)
    Sz = torch.tensor([[1.0, 0.5, 1.0, 0.5]], dtype=DOUBLE)
    ma, Sa = EvenProbit(2, epsilon=EPSILON, kappa=KAPPA).forward(mz, Sz)

    assert bool((ma[..., 1::2] > EPSILON).all())
    assert bool((ma[..., 1::2] < EPSILON + KAPPA).all())
    assert bool((Sa[..., 1::2] >= 0.0).all())


def test_even_probit_forward_saturates_at_the_activation_bounds():
    # Phi underflows to exactly zero and rounds to exactly one past ~38 sigma,
    # so the open bounds close numerically. Both ends must stay clean.
    mz = torch.tensor([[0.0, -400.0, 0.0, 400.0]], dtype=DOUBLE)
    Sz = torch.tensor([[1.0, 0.5, 1.0, 0.5]], dtype=DOUBLE)
    ma, Sa = EvenProbit(2, epsilon=EPSILON, kappa=KAPPA).forward(mz, Sz)

    torch.testing.assert_close(
        ma[..., 1::2],
        torch.tensor([[EPSILON, EPSILON + KAPPA]], dtype=DOUBLE),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        Sa[..., 1::2], torch.zeros(1, 2, dtype=DOUBLE), rtol=1e-10, atol=1e-12
    )


def test_even_probit_caches_pre_activation_head_moments():
    layer = EvenProbit(1, epsilon=EPSILON, kappa=KAPPA)
    first_mz = torch.tensor([[0.1, 0.6]], dtype=DOUBLE)
    first_Sz = torch.tensor([[0.2, 0.4]], dtype=DOUBLE)
    layer.forward(first_mz, first_Sz)
    torch.testing.assert_close(layer.nu, first_mz[..., 1::2], rtol=1e-10, atol=1e-12)

    second_mz = torch.tensor([[0.1, -0.9]], dtype=DOUBLE)
    second_Sz = torch.tensor([[0.2, 1.1]], dtype=DOUBLE)
    layer.forward(second_mz, second_Sz)
    torch.testing.assert_close(layer.nu, second_mz[..., 1::2], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(layer.r, second_Sz[..., 1::2], rtol=1e-10, atol=1e-12)


def test_even_probit_backward_is_an_identity_passthrough():
    torch.manual_seed(0)
    layer = EvenProbit(3, epsilon=EPSILON, kappa=KAPPA)
    delta_ma = torch.randn(4, 6, dtype=DOUBLE)
    delta_Sa = torch.randn(4, 6, dtype=DOUBLE)

    out_ma, out_Sa = layer.backward(delta_ma, delta_Sa)
    assert out_ma is delta_ma
    assert out_Sa is delta_Sa


def test_even_probit_rejects_a_wrong_output_width():
    layer = EvenProbit(3, epsilon=EPSILON, kappa=KAPPA)
    with pytest.raises(ValueError, match="width"):
        layer.forward(torch.zeros(2, 4, dtype=DOUBLE), torch.zeros(2, 4, dtype=DOUBLE))
    with pytest.raises(ValueError, match="matching shapes"):
        layer.forward(torch.zeros(2, 6, dtype=DOUBLE), torch.zeros(2, 4, dtype=DOUBLE))


@pytest.mark.parametrize("epsilon,kappa", [(0.0, 1.0), (-1.0, 1.0), (0.1, 0.0), (0.1, -2.0)])
def test_even_probit_rejects_non_positive_constants(epsilon, kappa):
    with pytest.raises(ValueError):
        EvenProbit(2, epsilon=epsilon, kappa=kappa)


def test_even_probit_repr_reports_its_constants():
    text = repr(EvenProbit(4, epsilon=0.25, kappa=3.0))
    assert "EvenProbit" in text
    assert "half_width=4" in text
    assert "0.25" in text and "3.0" in text


# ---------------------------------------------------------------------------
#  Observation update: exact references
# ---------------------------------------------------------------------------


def test_cdf_innovation_matches_dense_numerical_integration():
    targets = torch.tensor([[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]], dtype=DOUBLE)
    mu_z = torch.tensor([[0.3, -0.7, 0.1], [1.4, -0.2, -1.1]], dtype=DOUBLE)
    var_z = torch.tensor([[0.5, 0.2, 1.3], [0.8, 0.05, 0.4]], dtype=DOUBLE)
    nu = torch.tensor([[0.2, -1.0, 0.6], [-0.4, 1.2, 0.0]], dtype=DOUBLE)
    r = torch.tensor([[0.6, 0.3, 1.0], [0.9, 0.15, 2.0]], dtype=DOUBLE)

    # Order 150 rather than the default: the widest state here, r = 2, needs it
    # (see test_cdf_innovation_quadrature_order_converges_monotonically), and
    # the point of this test is the integral identity, not the default order.
    delta_mu, delta_var = compute_cdf_tagiv_innovation(
        targets, mu_z, var_z, nu, r, epsilon=EPSILON, kappa=KAPPA, hermite_order=150
    )
    got = _posteriors_from_deltas(delta_mu, delta_var, mu_z, var_z, nu, r)
    want = _dense_reference(targets, mu_z, var_z, nu, r)

    for actual, expected in zip(got, want, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-13)


def test_cdf_innovation_returns_the_interleaved_layout():
    targets = torch.tensor([[1.0, -1.0]], dtype=DOUBLE)
    mu_z = torch.tensor([[0.2, -0.4]], dtype=DOUBLE)
    var_z = torch.tensor([[0.6, 0.3]], dtype=DOUBLE)
    nu = torch.tensor([[0.1, 0.5]], dtype=DOUBLE)
    r = torch.tensor([[0.7, 0.2]], dtype=DOUBLE)

    delta_mu, delta_var = compute_cdf_tagiv_innovation(
        targets, mu_z, var_z, nu, r, epsilon=EPSILON, kappa=KAPPA
    )
    assert delta_mu.shape == (1, 4)
    assert delta_var.shape == (1, 4)

    single = [
        compute_cdf_tagiv_innovation(
            targets[:, k : k + 1],
            mu_z[:, k : k + 1],
            var_z[:, k : k + 1],
            nu[:, k : k + 1],
            r[:, k : k + 1],
            epsilon=EPSILON,
            kappa=KAPPA,
        )
        for k in range(2)
    ]
    expected_mu = _interleave(
        torch.cat([part[0][:, 0::2] for part in single], dim=-1),
        torch.cat([part[0][:, 1::2] for part in single], dim=-1),
    )
    torch.testing.assert_close(delta_mu, expected_mu, rtol=1e-10, atol=1e-12)


def test_cdf_innovation_importance_sampling_monte_carlo():
    torch.manual_seed(1)
    targets = torch.tensor([[1.0]], dtype=DOUBLE)
    mu_z = torch.tensor([[0.3]], dtype=DOUBLE)
    var_z = torch.tensor([[0.5]], dtype=DOUBLE)
    nu = torch.tensor([[0.2]], dtype=DOUBLE)
    r = torch.tensor([[0.6]], dtype=DOUBLE)

    delta_mu, delta_var = compute_cdf_tagiv_innovation(
        targets, mu_z, var_z, nu, r, epsilon=EPSILON, kappa=KAPPA
    )
    mu_z_post, var_z_post, nu_post, r_post = _posteriors_from_deltas(
        delta_mu, delta_var, mu_z, var_z, nu, r
    )

    # Draw U from its prior and reweight by the likelihood N(y; mu, e + h(U)),
    # which is exactly the tilt omega_tilde. Z is then drawn from its exact
    # conditional given (y, U), so the same weights carry both heads.
    head = nu + r.sqrt() * torch.randn(2_000_000, 1, dtype=DOUBLE)
    noise = cdf_variance_activation(head, epsilon=EPSILON, kappa=KAPPA)
    total = var_z + noise
    residual = targets - mu_z
    weight = torch.exp(-0.5 * residual.square() / total) / total.sqrt()
    weight = weight / weight.sum()

    conditional_mean = mu_z + var_z * residual / total
    conditional_var = var_z * noise / total
    latent = conditional_mean + conditional_var.sqrt() * torch.randn(2_000_000, 1, dtype=DOUBLE)

    sampled_nu = (weight * head).sum(0)
    sampled_r = (weight * head.square()).sum(0) - sampled_nu.square()
    sampled_mu_z = (weight * latent).sum(0)
    sampled_var_z = (weight * latent.square()).sum(0) - sampled_mu_z.square()

    torch.testing.assert_close(sampled_nu, nu_post.reshape(-1), rtol=5e-3, atol=0)
    torch.testing.assert_close(sampled_r, r_post.reshape(-1), rtol=3e-2, atol=0)
    torch.testing.assert_close(sampled_mu_z, mu_z_post.reshape(-1), rtol=5e-3, atol=0)
    torch.testing.assert_close(sampled_var_z, var_z_post.reshape(-1), rtol=3e-2, atol=0)


def test_cdf_innovation_default_order_is_converged_for_a_narrow_head():
    # For r <~ 0.25 the likelihood factor is smooth on the scale of the node
    # spacing and the default order is already converged. The tolerance is on
    # the normalized deltas, which divide by r**2 and so amplify roundoff.
    targets = torch.tensor([[1.0, -1.0, -1.0]], dtype=DOUBLE)
    mu_z = torch.tensor([[0.4, -0.9, 0.2]], dtype=DOUBLE)
    var_z = torch.tensor([[0.7, 0.1, 1.1]], dtype=DOUBLE)
    nu = torch.tensor([[0.3, -0.6, 1.5]], dtype=DOUBLE)
    r = torch.tensor([[0.2, 0.25, 0.1]], dtype=DOUBLE)

    coarse = compute_cdf_tagiv_innovation(
        targets, mu_z, var_z, nu, r, epsilon=EPSILON, kappa=KAPPA, hermite_order=32
    )
    fine = compute_cdf_tagiv_innovation(
        targets, mu_z, var_z, nu, r, epsilon=EPSILON, kappa=KAPPA, hermite_order=64
    )
    for actual, expected in zip(coarse, fine, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)


def test_cdf_innovation_quadrature_order_converges_monotonically():
    """A wide head needs a higher order, and the error decreases geometrically.

    ``h`` traverses its whole range over roughly one unit of ``u``, so at
    ``r = 1.2`` the tilt varies on the scale of the node spacing and the
    default order 32 is only good to about ``1e-6`` relative. The default is
    not a universal setting; this pins the actual convergence so a change in
    the parameter range is caught rather than assumed away.
    """

    targets = torch.tensor([[1.0]], dtype=DOUBLE)
    mu_z = torch.tensor([[0.4]], dtype=DOUBLE)
    var_z = torch.tensor([[0.7]], dtype=DOUBLE)
    nu = torch.tensor([[0.3]], dtype=DOUBLE)
    r = torch.tensor([[1.2]], dtype=DOUBLE)

    def run(order):
        return compute_cdf_tagiv_innovation(
            targets, mu_z, var_z, nu, r, epsilon=EPSILON, kappa=KAPPA, hermite_order=order
        )

    reference = run(150)

    def relative_error(order):
        got = run(order)
        return max(
            float(((got[index] - reference[index]) / reference[index]).abs().max())
            for index in (0, 1)
        )

    errors = [relative_error(order) for order in (32, 64, 96, 128)]
    assert errors[0] < 1e-5
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1e-12

    coarse = run(96)
    fine = run(128)
    for actual, expected in zip(coarse, fine, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
#  Degeneracies and extremes
# ---------------------------------------------------------------------------


def test_cdf_innovation_degenerate_head_reduces_to_the_fixed_noise_update():
    targets = torch.tensor([[1.0, -1.0]], dtype=DOUBLE)
    mu_z = torch.tensor([[0.3, -0.8]], dtype=DOUBLE)
    var_z = torch.tensor([[0.5, 0.2]], dtype=DOUBLE)
    nu = torch.tensor([[0.4, -1.1]], dtype=DOUBLE)
    r = torch.zeros(1, 2, dtype=DOUBLE)

    delta_mu, delta_var = compute_cdf_tagiv_innovation(
        targets, mu_z, var_z, nu, r, epsilon=EPSILON, kappa=KAPPA
    )
    assert bool(torch.isfinite(delta_mu).all())
    assert bool(torch.isfinite(delta_var).all())

    total = var_z + cdf_variance_activation(nu, epsilon=EPSILON, kappa=KAPPA)
    torch.testing.assert_close(
        delta_mu[..., 0::2], (targets - mu_z) / total, rtol=1e-10, atol=1e-12
    )
    torch.testing.assert_close(delta_var[..., 0::2], -1.0 / total, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(
        delta_mu[..., 1::2], torch.zeros(1, 2, dtype=DOUBLE), rtol=1e-10, atol=1e-12
    )
    torch.testing.assert_close(
        delta_var[..., 1::2], torch.zeros(1, 2, dtype=DOUBLE), rtol=1e-10, atol=1e-12
    )


def test_cdf_innovation_survives_a_very_surprising_observation():
    targets = torch.tensor([[1.0, -1.0, 1.0]], dtype=DOUBLE)
    mu_z = torch.tensor([[-60.0, 80.0, 0.0]], dtype=DOUBLE)
    var_z = torch.tensor([[1e-6, 0.4, 1e3]], dtype=DOUBLE)
    nu = torch.tensor([[-12.0, 9.0, 0.0]], dtype=DOUBLE)
    r = torch.tensor([[4.0, 1e-8, 30.0]], dtype=DOUBLE)

    delta_mu, delta_var = compute_cdf_tagiv_innovation(
        targets, mu_z, var_z, nu, r, epsilon=EPSILON, kappa=KAPPA
    )
    assert bool(torch.isfinite(delta_mu).all())
    assert bool(torch.isfinite(delta_var).all())

    _, var_z_post, _, r_post = _posteriors_from_deltas(delta_mu, delta_var, mu_z, var_z, nu, r)
    assert bool((var_z_post >= 0.0).all())
    assert bool((r_post >= 0.0).all())
    assert bool((var_z_post <= var_z + 1e-9).all())
    assert bool((r_post <= r + 1e-9).all())


def test_cdf_innovation_a_matched_observation_still_informs_the_head():
    """``Delta = 0`` freezes the prediction mean but not the variance head.

    The likelihood ``N(0; 0, d(u))`` still prefers a small ``d``, so the tilt
    is a decreasing function of ``u`` and pulls ``nu`` down. It does *not*
    shrink ``r``: tilting a Gaussian by a monotone factor can widen it, and
    here it does, which is why the reference is the dense integral rather than
    a sign convention.
    """

    targets = torch.tensor([[1.0]], dtype=DOUBLE)
    mu_z = torch.tensor([[1.0]], dtype=DOUBLE)
    var_z = torch.tensor([[0.3]], dtype=DOUBLE)
    nu = torch.tensor([[0.0]], dtype=DOUBLE)
    r = torch.tensor([[1.0]], dtype=DOUBLE)

    delta_mu, delta_var = compute_cdf_tagiv_innovation(
        targets, mu_z, var_z, nu, r, epsilon=EPSILON, kappa=KAPPA, hermite_order=96
    )
    torch.testing.assert_close(
        delta_mu[..., 0::2], torch.zeros(1, 1, dtype=DOUBLE), rtol=1e-10, atol=1e-12
    )
    assert float(delta_mu[0, 1]) < 0.0

    got = _posteriors_from_deltas(delta_mu, delta_var, mu_z, var_z, nu, r)
    want = _dense_reference(targets, mu_z, var_z, nu, r)
    for actual, expected in zip(got, want, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epsilon": 0.0, "kappa": 1.0},
        {"epsilon": 0.1, "kappa": 0.0},
        {"epsilon": 0.1, "kappa": 1.0, "hermite_order": 3},
        {"epsilon": 0.1, "kappa": 1.0, "variance_floor": 0.0},
    ],
)
def test_cdf_innovation_rejects_invalid_constants(kwargs):
    ones = torch.ones(1, 1, dtype=DOUBLE)
    with pytest.raises(ValueError):
        compute_cdf_tagiv_innovation(ones, ones, ones, ones, ones, **kwargs)


def test_cdf_innovation_rejects_mismatched_shapes():
    ones = torch.ones(1, 2, dtype=DOUBLE)
    with pytest.raises(ValueError, match="matching shapes"):
        compute_cdf_tagiv_innovation(
            ones, ones, ones, ones, torch.ones(1, 3, dtype=DOUBLE), epsilon=EPSILON, kappa=KAPPA
        )


# ---------------------------------------------------------------------------
#  End-to-end step
# ---------------------------------------------------------------------------


@pytest.mark.cuda
def test_step_cdf_tagiv_updates_a_linear_head():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the triton Linear layer")
    torch.manual_seed(2)
    from triton_tagi import Linear, Sequential

    linear = Linear(8, 6, device="cuda")
    net = Sequential([linear, EvenProbit(3, epsilon=EPSILON, kappa=KAPPA)], device="cuda")
    before = linear.mw.clone()

    x_batch = torch.randn(16, 8, device="cuda")
    labels = torch.randint(0, 3, (16,), device="cuda")
    y_pred_mu, y_pred_var = net.step_cdf_tagiv(x_batch, labels, epsilon=EPSILON, kappa=KAPPA)

    assert y_pred_mu.shape == (16, 6)
    assert y_pred_var.shape == (16, 6)
    assert bool(torch.isfinite(y_pred_mu).all())
    assert not torch.allclose(before, linear.mw)
