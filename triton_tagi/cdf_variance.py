"""Positive, bounded aleatoric variance from a Gaussian CDF activation.

TAGI-V learns a heteroscedastic logit noise through a second output head. The
exponential activation of :class:`triton_tagi.layers.EvenExp` makes that noise
positive but unbounded, and a saturating or diverging variance head is a
recurring failure mode. Replacing the activation by a scaled Gaussian CDF

    ``V2bar_i = h(U_i) := epsilon + kappa Phi(U_i),   epsilon > 0, kappa > 0``

bounds the noise on both sides: ``epsilon < V2bar_i < epsilon + kappa`` for
every finite ``U_i``. Both constants have units of squared logits and are fixed
before calibration; the cap must accommodate the intended logit scale, since a
head pinned at ``epsilon + kappa`` has stopped adapting.

The activation is worth this module because its moments are *exact* for a
Gaussian ``U_i ~ N(nu_i, r_i)``. Writing ``a_i = nu_i / sqrt(1 + r_i)`` and
``rho_i = r_i / (1 + r_i)``,

    ``h_i    = E[V2bar_i]     = epsilon + kappa Phi(a_i)``
    ``b_i    = Var(V2bar_i)   = kappa^2 [Phi_2(a_i, a_i; rho_i) - Phi(a_i)^2]``
    ``c_i    = Cov(U_i, V2bar_i) = kappa r_i phi(a_i) / sqrt(1 + r_i)``

from normal probability augmentation and Stein's identity, and more generally
``Cov(T, V2bar_i) = kappa Cov(T, U_i) phi(a_i) / sqrt(1 + r_i)`` for any state
``T`` jointly Gaussian with ``U_i``. That last identity is the interface a TAGI
backward pass needs. The bivariate term is evaluated through
:func:`triton_tagi.hsm_calibration.probit_gaussian_moments`, which integrates
``Var[Phi(U)]`` as a positive integral rather than differencing
``Phi_2 - Phi^2``; the two are algebraically equal and the integral form keeps
full precision when ``Phi(a_i)`` is near zero or one.

``b_i`` is uncertainty *about* a variance. It is not a second variance to add
to ``e_i + h_i``: it enters fourth moments and nonlinear predictive averages
only. With ``V_i | U_i ~ N(0, h(U_i))``,

    ``Var(V_i) = h_i``,  ``Var(V_i^2) = 2 h_i^2 + 3 b_i``,
    ``Cov(V2bar_i, V_i^2) = b_i``,  ``Cov(U_i, V_i) = 0``.

The vanishing ``Cov(U_i, V_i)`` is why a purely linear Gaussian update on the
raw residual cannot identify this head on its own, and why the training channel
integrates ``U_i`` explicitly instead.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .hsm_calibration import DEFAULT_OWEN_ORDER, probit_gaussian_moments

_INV_SQRT_TWO_PI: float = 1.0 / math.sqrt(2.0 * math.pi)

# Below this prior variance the head is treated as deterministic: b_i is zero
# to rounding, so the bridge gain is undefined and is set to zero instead.
_DEGENERATE_VARIANCE: float = 1e-300


def cdf_variance_activation(u: Tensor, *, epsilon: float, kappa: float) -> Tensor:
    """Return ``h(u) = epsilon + kappa Phi(u)`` elementwise.

    This is the realisation used to generate a future positive variance. A
    Gaussian approximation to ``V2bar`` is never sampled in its place, since
    that would not preserve the ``(epsilon, epsilon + kappa)`` support.

    Args:
        u: Variance-head realisation.
        epsilon: Strictly positive variance floor.
        kappa: Strictly positive variance range.

    Returns:
        The aleatoric variance, in ``(epsilon, epsilon + kappa)``.
    """

    _check_constants(epsilon, kappa)
    return epsilon + kappa * torch.special.ndtr(u.double())


def cdf_variance_moments(
    nu: Tensor,
    r: Tensor,
    *,
    epsilon: float,
    kappa: float,
    order: int = DEFAULT_OWEN_ORDER,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the exact forward moments of ``V2bar = h(U)`` for ``U ~ N(nu, r)``.

    Args:
        nu: Variance-head prior means.
        r: Variance-head prior variances, non-negative.
        epsilon: Strictly positive variance floor.
        kappa: Strictly positive variance range.
        order: Gauss--Legendre order for the ``Var[Phi]`` integral.

    Returns:
        h_mean: ``E[V2bar] = epsilon + kappa Phi(a)``.
        h_variance: ``Var(V2bar) = kappa^2 Var[Phi(U)]``, uncertainty *about*
            the variance and never an additive noise term.
        cov_u_h: ``Cov(U, V2bar) = kappa r phi(a) / sqrt(1 + r)``.
    """

    _check_constants(epsilon, kappa)
    work_nu = nu.double()
    work_r = r.double().clamp_min(0.0)

    probit_mean, probit_variance = probit_gaussian_moments(work_nu, work_r, order=order)
    h_mean = epsilon + kappa * probit_mean
    h_variance = (kappa * kappa) * probit_variance

    root = (1.0 + work_r).sqrt()
    argument = work_nu / root
    density = _INV_SQRT_TWO_PI * torch.exp(-0.5 * argument * argument)
    cov_u_h = kappa * work_r * density / root
    return h_mean, h_variance, cov_u_h


def upstream_cdf_covariance(
    cov_upstream_u: Tensor,
    nu: Tensor,
    r: Tensor,
    *,
    kappa: float,
) -> Tensor:
    """Return ``Cov(T, V2bar)`` from ``Cov(T, U)`` for jointly Gaussian ``T``.

    Implements ``Cov(T, V2bar) = kappa Cov(T, U) phi(a) / sqrt(1 + r)`` with
    ``a = nu / sqrt(1 + r)``. This supplies the cross-covariance a TAGI
    backward pass needs to carry a variance-head observation into an upstream
    state or parameter block.

    Args:
        cov_upstream_u: ``Cov(T, U)`` for the upstream state ``T``.
        nu: Variance-head prior means, broadcastable to ``cov_upstream_u``.
        r: Variance-head prior variances, non-negative.
        kappa: Strictly positive variance range.

    Returns:
        ``Cov(T, V2bar)`` in float64.
    """

    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    work_r = r.double().clamp_min(0.0)
    root = (1.0 + work_r).sqrt()
    argument = nu.double() / root
    density = _INV_SQRT_TWO_PI * torch.exp(-0.5 * argument * argument)
    return kappa * cov_upstream_u.double() * density / root


def cdf_variance_bridge(
    nu: Tensor,
    r: Tensor,
    h_mean_post: Tensor,
    h_variance_post: Tensor,
    *,
    epsilon: float,
    kappa: float,
    order: int = DEFAULT_OWEN_ORDER,
) -> tuple[Tensor, Tensor]:
    """Project posterior ``V2bar`` moments back onto the Gaussian head ``U``.

    A legacy variance module that reports ``(h^+, b^+)`` rather than updating
    ``U`` directly is bridged by the Gaussian regression

        ``nu^+ = nu + (c / b) (h^+ - h)``,
        ``r^+  = r  + (c / b)^2 (b^+ - b)``,

    with ``(h, b, c)`` the prior moments of :func:`cdf_variance_moments`. This
    is a Gaussian projection, not an exact inverse of the CDF activation, and
    is unnecessary whenever the likelihood updates ``U`` directly -- which is
    what :func:`triton_tagi.update.observation.compute_cdf_tagiv_innovation`
    does. Where ``b`` underflows the head is deterministic and the gain is
    taken as zero rather than dividing by zero.

    Args:
        nu: Variance-head prior means.
        r: Variance-head prior variances.
        h_mean_post: Posterior ``E[V2bar]`` from the legacy module.
        h_variance_post: Posterior ``Var(V2bar)`` from the legacy module.
        epsilon: Strictly positive variance floor.
        kappa: Strictly positive variance range.
        order: Gauss--Legendre order for the prior moments.

    Returns:
        nu_post: Bridged posterior mean of ``U``.
        r_post: Bridged posterior variance of ``U``, clamped non-negative.
    """

    h_mean, h_variance, cov_u_h = cdf_variance_moments(
        nu, r, epsilon=epsilon, kappa=kappa, order=order
    )
    degenerate = h_variance <= _DEGENERATE_VARIANCE
    gain = torch.where(degenerate, torch.zeros_like(cov_u_h), cov_u_h / h_variance)

    nu_post = nu.double() + gain * (h_mean_post.double() - h_mean)
    r_post = r.double() + gain * gain * (h_variance_post.double() - h_variance)
    return nu_post, r_post.clamp_min(0.0)


def cdf_squared_error_moments(
    h_mean: Tensor,
    h_variance: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``Var(V^2)`` and ``Cov(V2bar, V^2)`` for the CDF head.

    For ``V | U ~ N(0, h(U))`` the fourth moment is ``E[V^4] = 3 (h^2 + b)``,
    so ``Var(V^2) = 2 h^2 + 3 b`` and ``Cov(V2bar, V^2) = b``. Substituting
    ``2 h^2`` for ``Var(V^2)`` would discard the variance-head uncertainty
    entirely.

    Args:
        h_mean: ``E[V2bar]``.
        h_variance: ``Var(V2bar)``.

    Returns:
        squared_variance: ``Var(V^2) = 2 h^2 + 3 b``.
        cross_covariance: ``Cov(V2bar, V^2) = b``.
    """

    mean = h_mean.double()
    variance = h_variance.double()
    return 2.0 * mean * mean + 3.0 * variance, variance


def _check_constants(epsilon: float, kappa: float) -> None:
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}")
