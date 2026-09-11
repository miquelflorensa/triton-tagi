"""Stable Gaussian kernels for the Laplace-Remax integral identities.

Remax probability moments follow from the Laplace identities ``1 / S =
int_0^inf exp(-t S) dt`` and ``1 / S^2 = int_0^inf t exp(-t S) dt`` applied to
the rectified sum ``S = sum_j [X_j]_+``. Because the classes are independent
given the scale, the integrand factorises into per-class kernels of the
rectified variable ``M = [X]_+`` for ``X ~ N(m, d^2)``::

    L(t; m, d) = E[exp(-t M)]
    D(t; m, d) = E[M exp(-t M)]
    F(t; m, d) = E[M^2 exp(-t M)]

The textbook forms of ``D`` and ``F`` combine a scaled normal CDF with a normal
density and lose all significance as ``t`` grows: both terms tend to the same
limit and cancel. Refining the ``t`` quadrature then makes the answer *worse*,
because the added nodes sit further into the regime where the cancellation is
total. Every kernel here is instead evaluated through

    ``E[M^n exp(-t M)] = d^n phi(alpha) I_n(beta)``,
    ``I_n(beta) = int_0^inf y^n exp(-y^2 / 2 + beta y) dy``,

with ``alpha = m / d`` and ``beta = alpha - d t``. Each ``I_n`` is a single
positive quantity, so no cancellation is possible in the kernels themselves.
The identity is exact: substituting ``x = d y`` in the defining integral gives

    ``E[M^n e^{-tM}] = d^n (2 pi)^{-1/2} e^{-alpha^2/2} int_0^inf y^n
      e^{-y^2/2 + (alpha - dt) y} dy``.

``I_n`` obeys ``I_n = beta I_{n-1} + (n - 1) I_{n-2}`` from integrating
``y^{n-1} (y - beta) exp(-y^2/2 + beta y)`` by parts, seeded by ``I_0 =
sqrt(pi / 2) erfcx(-beta / sqrt 2)`` and ``I_1 = 1 + beta I_0``. That recursion
is itself cancelling once ``beta`` is far negative -- ``I_1`` is of order
``beta^-2`` while ``1`` and ``beta I_0`` are both of order one -- so below
:data:`TAIL_CUTOFF` the kernels switch to the matching asymptotic series

    ``I_n(-B) = n! B^-(n+1) sum_k (-1)^k (n + 2k)! / [n! k! 2^k B^(2k)]``,

obtained by substituting ``y = s / B`` and expanding ``exp(-s^2 / 2 B^2)``; four
correction terms are carried, which is what makes the two branches agree to
``4e-11`` where they meet. Truncating at three terms leaves a ``7e-9`` step,
and moving the cutoff deeper instead only shifts the error onto the
increasingly cancelling recursion.

With both branches in place the Remax covariance row-sum identity holds to
``2e-12`` and stays there as the quadrature is refined; with either branch alone
it degrades by ten orders of magnitude between 160 and 640 nodes.

The kernels take ``m`` and ``d`` directly, so a caller supplies the calibrated
deviation scale and the variance-head realisation through ``d`` and never needs
to rescale a kernel afterwards.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import torch
from torch import Tensor

# The Laplace variable is mapped from the unit interval by t = y / [a (1 - y)],
# so a 160-node rule already resolves the whole positive half line.
DEFAULT_LAPLACE_ORDER: int = 160

# beta below this switches from the by-parts recursion to the asymptotic
# series. At -30 the two agree to 1e-10, which is where their errors cross:
# above it the series truncates too early, below it the recursion cancels.
TAIL_CUTOFF: float = -30.0

_INV_SQRT_TWO: float = 1.0 / math.sqrt(2.0)
_INV_SQRT_TWO_PI: float = 1.0 / math.sqrt(2.0 * math.pi)
_SQRT_HALF_PI: float = math.sqrt(0.5 * math.pi)


@functools.cache
def _legendre_unit_rule(order: int) -> tuple[Tensor, Tensor]:
    """Return Gauss--Legendre nodes and weights on ``[0, 1]``, CPU float64."""

    if order < 4:
        raise ValueError(f"quadrature order must be at least four, got {order}")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (
        torch.from_numpy(np.ascontiguousarray(0.5 * (nodes + 1.0))),
        torch.from_numpy(np.ascontiguousarray(0.5 * weights)),
    )


def laplace_rule(
    order: int = DEFAULT_LAPLACE_ORDER,
    *,
    map_scale: float = 1.0,
    reference: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return Laplace-variable nodes and weights on the positive half line.

    The map is ``t = y / [a (1 - y)]`` for ``y`` in ``(0, 1)``, with Jacobian
    ``dt = dy / [a (1 - y)^2]``, where ``a = map_scale`` is a typical rectified
    sum. The returned weights already carry that Jacobian, so a quadrature is
    ``sum_q weights[q] f(nodes[q])``.

    Args:
        order: Gauss--Legendre order.
        map_scale: Positive scale ``a`` of the map.
        reference: Optional tensor whose device the rule is placed on.

    Returns:
        nodes: Laplace variable ``t``, shape ``(order,)``, float64.
        weights: Quadrature weights including the Jacobian, same shape.
    """

    if map_scale <= 0.0:
        raise ValueError(f"map_scale must be positive, got {map_scale}")
    unit_nodes, unit_weights = _legendre_unit_rule(order)
    if reference is not None:
        unit_nodes = unit_nodes.to(reference.device)
        unit_weights = unit_weights.to(reference.device)
    complement = 1.0 - unit_nodes
    return (
        unit_nodes / (map_scale * complement),
        unit_weights / (map_scale * complement * complement),
    )


@functools.cache
def _hermite_rule(order: int) -> tuple[Tensor, Tensor]:
    """Return standard-normal nodes and weights summing to one, CPU float64."""

    if order < 4:
        raise ValueError(f"quadrature order must be at least four, got {order}")
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    return (
        torch.from_numpy(np.ascontiguousarray(nodes)),
        torch.from_numpy(np.ascontiguousarray(weights / math.sqrt(2.0 * math.pi))),
    )


def hermite_rule(
    order: int,
    *,
    reference: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return standard-normal Gauss--Hermite nodes and weights summing to one.

    A quadrature of ``f`` against ``N(m, v)`` is
    ``sum_j weights[j] f(m + sqrt(v) nodes[j])``.

    Args:
        order: Gauss--Hermite order, at least four.
        reference: Optional tensor whose device the rule is placed on.

    Returns:
        nodes: Standard-normal nodes, shape ``(order,)``, float64.
        weights: Weights summing to one, same shape.
    """

    nodes, weights = _hermite_rule(order)
    if reference is not None:
        return nodes.to(reference.device), weights.to(reference.device)
    return nodes, weights


def rectified_exponential_integrals(beta: Tensor, *, highest: int = 2) -> tuple[Tensor, ...]:
    """Return ``I_n(beta) = int_0^inf y^n exp(-y^2 / 2 + beta y) dy``.

    Uses the by-parts recursion ``I_n = beta I_{n-1} + (n - 1) I_{n-2}`` seeded
    by ``I_0 = sqrt(pi / 2) erfcx(-beta / sqrt 2)``, and the asymptotic series
    of the module docstring where ``beta < TAIL_CUTOFF``. Both branches are
    evaluated everywhere and selected pointwise, so the result is finite for
    every finite ``beta``.

    Args:
        beta: Kernel argument ``alpha - d t``, any shape.
        highest: Largest moment order ``n`` to return.

    Returns:
        A tuple ``(I_0, ..., I_highest)`` in float64, broadcast to ``beta``.
    """

    if highest < 0:
        raise ValueError(f"highest must be non-negative, got {highest}")
    work = beta.double()

    recursion = [_SQRT_HALF_PI * torch.special.erfcx(-work * _INV_SQRT_TWO)]
    if highest >= 1:
        recursion.append(1.0 + work * recursion[0])
    for order in range(2, highest + 1):
        recursion.append(work * recursion[order - 1] + (order - 1) * recursion[order - 2])

    # Clamp the series argument away from zero so the unselected branch stays
    # finite; torch.where evaluates both sides.
    reach = (-work).clamp_min(-TAIL_CUTOFF)
    inverse = 1.0 / reach
    square = inverse * inverse
    tail = work < TAIL_CUTOFF

    integrals = []
    for order in range(highest + 1):
        first = (order + 1) * (order + 2) / 2.0
        second = first * (order + 3) * (order + 4) / 4.0
        third = second * (order + 5) * (order + 6) / 6.0
        fourth = third * (order + 7) * (order + 8) / 8.0
        series = (
            math.factorial(order)
            * inverse ** (order + 1)
            * (1.0 - first * square + second * square**2 - third * square**3 + fourth * square**4)
        )
        integrals.append(torch.where(tail, series, recursion[order]))
    return tuple(integrals)


def scaled_rectified_integrals(
    alpha: Tensor,
    beta: Tensor,
    *,
    highest: int = 2,
) -> tuple[Tensor, ...]:
    """Return ``phi(alpha) I_n(beta)``, the form the kernels actually need.

    Carrying the product rather than its two factors is what keeps the kernels
    finite. When the Gaussian mean dominates its deviation, ``alpha`` and hence
    ``beta`` are large and positive: ``phi(alpha)`` underflows to zero while
    ``I_0 = sqrt(pi/2) erfcx(-beta / sqrt 2)`` overflows, so the factored form
    evaluates ``0 * inf``. The product never does, because

        ``phi(alpha) I_0(beta) = exp[(beta^2 - alpha^2) / 2] Phi(beta)``

    which is ``E[exp(-t M) 1{X > 0}]`` and therefore lies in ``[0, 1]``. The
    by-parts recursion carries over to the scaled quantities unchanged,

        ``phi I_1 = phi(alpha) + beta (phi I_0)``,
        ``phi I_n = beta (phi I_{n-1}) + (n - 1) (phi I_{n-2})``,

    and below :data:`TAIL_CUTOFF` the asymptotic series is scaled by
    ``phi(alpha)`` directly. Where ``phi(alpha)`` itself underflows the answer
    really is zero: ``beta < TAIL_CUTOFF`` with a large ``alpha`` forces a
    large ``t``, and ``E[M^n exp(-t M)]`` is then negligible.

    Args:
        alpha: ``mean / deviation``.
        beta: ``alpha - deviation t``, broadcastable against ``alpha``.
        highest: Largest moment order ``n`` to return.

    Returns:
        A tuple ``(phi I_0, ..., phi I_highest)`` in float64.
    """

    if highest < 0:
        raise ValueError(f"highest must be non-negative, got {highest}")
    work_alpha = alpha.double()
    work_beta = beta.double()
    density = _INV_SQRT_TWO_PI * torch.exp(-0.5 * work_alpha * work_alpha)

    exponent = 0.5 * (work_beta * work_beta - work_alpha * work_alpha)
    seed = torch.exp(torch.clamp(exponent + torch.special.log_ndtr(work_beta), min=-745.0, max=0.0))
    recursion = [seed]
    if highest >= 1:
        recursion.append(density + work_beta * seed)
    for order in range(2, highest + 1):
        recursion.append(work_beta * recursion[order - 1] + (order - 1) * recursion[order - 2])

    reach = (-work_beta).clamp_min(-TAIL_CUTOFF)
    inverse = 1.0 / reach
    square = inverse * inverse
    tail = work_beta < TAIL_CUTOFF

    integrals = []
    for order in range(highest + 1):
        first = (order + 1) * (order + 2) / 2.0
        second = first * (order + 3) * (order + 4) / 4.0
        third = second * (order + 5) * (order + 6) / 6.0
        fourth = third * (order + 7) * (order + 8) / 8.0
        series = (
            density
            * math.factorial(order)
            * inverse ** (order + 1)
            * (1.0 - first * square + second * square**2 - third * square**3 + fourth * square**4)
        )
        integrals.append(torch.where(tail, series, recursion[order]))
    return tuple(integrals)


def remax_kernels(
    t: Tensor,
    mean: Tensor,
    deviation: Tensor,
    *,
    highest: int = 2,
) -> tuple[Tensor, ...]:
    """Return the Laplace kernels of ``M = [X]_+`` for ``X ~ N(mean, deviation^2)``.

    With ``alpha = mean / deviation`` and ``beta = alpha - deviation t``,

        ``L(t) = Phi(-alpha) + phi(alpha) I_0(beta)``
        ``D(t) = deviation phi(alpha) I_1(beta)``
        ``F(t) = deviation^2 phi(alpha) I_2(beta)``

    where the ``Phi(-alpha)`` term of ``L`` is the atom at ``X <= 0``, on which
    ``exp(-t M) = 1``. The higher kernels carry no atom because ``M`` vanishes
    there. All arguments broadcast against one another.

    Args:
        t: Laplace variable, non-negative.
        mean: Gaussian mean ``m``.
        deviation: Gaussian standard deviation ``d``, strictly positive.
        highest: Largest kernel order; ``2`` returns ``(L, D, F)``.

    Returns:
        A tuple of kernels ``(L, D, F, ...)`` up to ``highest``, float64.
    """

    work_mean = mean.double()
    work_deviation = deviation.double()
    alpha = work_mean / work_deviation
    beta = alpha - work_deviation * t.double()

    integrals = scaled_rectified_integrals(alpha, beta, highest=highest)

    kernels = [torch.special.ndtr(-alpha) + integrals[0]]
    power = torch.ones_like(work_deviation)
    for order in range(1, highest + 1):
        power = power * work_deviation
        kernels.append(power * integrals[order])
    return tuple(kernels)


def rectified_zero_probability(mean: Tensor, deviation: Tensor) -> Tensor:
    """Return ``Pr(X <= 0) = Phi(-mean / deviation)`` for one class."""

    return torch.special.ndtr(-(mean.double() / deviation.double()))
