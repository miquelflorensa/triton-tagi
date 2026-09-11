"""Auxiliary Gaussian calibration channel for the shared Remax deviation scale.

Implements section "Learning the shared scale through an auxiliary channel" of
the CDF-TAGI-V/Remax joint calibration note. The predictive side lives in
:mod:`triton_tagi.cdf_remax`: a frozen network supplies, per calibration input,
the prediction head ``Z_i ~ N(mu_i, e_i)`` and the variance head
``U_i ~ N(nu_i, r_i)``, and one shared Gaussian log-scale multiplies the
combined deviation,

    ``X_{i,l} = mu_i + e^l [(Z_i - mu_i) + V_i]``,   ``L ~ N(lambda, q)``.

**Which channel learns what.** The Gaussian training channel updates the two
output heads and, through RTS, the trainable parameters. This module updates
``L`` and nothing else: the network enters only through the frozen forward
summaries ``(mu, e, nu, r)``, and neither ``Z`` nor ``U`` is a variable of the
auxiliary joint surrogate, so neither head receives a calibration update. The
calibration split must be disjoint from training, and the summaries must be
pre-update moments -- a post-update moment has already assimilated its label
and is not a calibration input.

From :func:`triton_tagi.cdf_remax.remax_scale_moments` the channel consumes

    ``p_i = E_L[m_i(L)]``,
    ``v_{A_i} = E_L[B_ii(L)] - p_i^2``,
    ``c_i = Cov(L, A_i) = E_L[(L - lambda) m_i(L)]``

with ``m_i(l) = E[A_i | L = l]`` the conditional Remax moments. The
cross-covariance is evaluated by the same scalar quadrature rather than by
differentiating an approximate moment formula, which would return an
approximate covariance rather than the covariance of the stated model. The
shared scale is integrated once for the complete class probability, after the
conditional class products have been formed; the reverse order is a different
(and wrong) model.

**The observed-class event channel.** With ``B_c = 1{C = c}``, the event
``B_c = 1`` has likelihood ``A_c`` conditionally on the latent probability
vector, so observing it is the same exact likelihood as observing ``C = c``.
The Bernoulli total-variance identity gives

    ``R_c = E[A_c (1 - A_c)] = p_c (1 - p_c) - v_{A_c}``,
    ``Var(B_c) = v_{A_c} + R_c = p_c (1 - p_c)``,

so the Gaussian surrogate ``Y_c = A_c + E_c`` with independent
``E_c ~ N(0, R_c)`` matches the mean, the variance and the covariance with
``L``. ``R_c`` is a moment-matching device and is *not* the TAGI-V logit error
``V_i``: that has already been integrated into ``A_c`` and must never be added
again. ``R_c`` is exactly the ``R`` of
:func:`triton_tagi.hsm_calibration.expected_bernoulli_variance`, which this
module reuses.

Conditioning the two-variable surrogate on the observed event gives the update
of :func:`remax_scale_event_update`. Its mean step is the exact one-step
categorical identity ``E[L | C = c] = E[L A_c] / p_c = lambda + c_c / p_c``;
its variance step is a Gaussian projection. Two stated alternatives share that
exact mean and differ only in the variance step:
:func:`remax_scale_full_update` projects the complete one-hot vector on the
simplex tangent space, and :func:`remax_scale_tilt_update` retains the exact
realised-label scalar moments by quadrature. All three replace the resulting
posterior by a Gaussian and continue, which is assumed-density filtering.

**Ordering.** For ``K > 2`` there is no tree-node factorization to exploit, so
:func:`calibrate_remax_log_scale_adf` performs exactly one observed-class
update per calibration sample and never assimilates the other ``K - 1`` zero
indicators as independent Bernoulli observations. Because the log-scale is
scalar, the order-independent batch fit of :func:`fit_remax_log_scale` is
preferred whenever the whole calibration split is available; the sequential
driver exists for a genuine stream, for the note's epoch-wise tracking option,
and as the ordering-sensitivity check the assumed-density approximation
requires.

**What is being fitted.** The batch objective fits probabilities *after* Remax
and after marginalizing uncertainty, ``-log E[A_c]``, never ``-E[log A_c]``:
the latter is a different objective and can be infinite, because an individual
Remax component has positive probability of being exactly zero. The fit is a
modular (cut) posterior conditioned on the frozen predictive summaries and is
not the joint weight-and-scale posterior, since in general

    ``prod_n E_q[p(c_n | x_n, theta, l)] != E_q[prod_n p(c_n | x_n, theta, l)]``

because a shared weight draw couples inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from .cdf_remax import (
    DEFAULT_HERMITE_ORDER,
    DEFAULT_SCALE_ORDER,
    remax_conditional_moments,
    remax_scale_moments,
)
from .hsm_calibration import (
    DEFAULT_GRID_REFINEMENTS,
    DEFAULT_GRID_SIZE,
    _grid_moments,
    _laplace_moments,
    _prior_grid_bounds,
    _refined_grid,
    expected_bernoulli_variance,
)
from .remax_kernels import DEFAULT_LAPLACE_ORDER, hermite_rule

# The note's proposed starting prior is lambda_0 = 0, q_0 = 0.5^2, centred on
# no rescaling. It is a stated regularization choice, not a data-supported
# universal default, and its sensitivity has to be reported.
DEFAULT_PRIOR_MEAN: float = 0.0
DEFAULT_PRIOR_VARIANCE: float = 0.25

# Variance-step variants of the auxiliary channel. Every one of them shares the
# exact one-step mean lambda + c_c / p_c.
REMAX_SCALE_METHODS: tuple[str, ...] = ("event", "full", "tilt")

# Batch-reference read-outs of the scalar calibration posterior.
REMAX_SCALE_FIT_METHODS: tuple[str, ...] = ("grid", "laplace")

# Peak working set inside cdf_remax is (rows, K, Q, J); eight million float64
# elements is sixty-four megabytes per working tensor.
DEFAULT_CHUNK_ELEMENTS: int = 8_000_000

_PROBABILITY_FLOOR: float = 1e-12
_LOG_FLOOR: float = 1e-300


# ──────────────────────────────────────────────────────────────────────────────
#  Log-scale belief
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LogScalePosterior:
    """Gaussian belief over the shared Remax log-scale ``L ~ N(lambda, q)``.

    The scale enters as ``s = exp(L) > 0`` multiplying the combined epistemic
    and aleatoric logit *deviation*, so positivity is structural and no
    posterior draw can flip the deviation's sign. Remax is positively
    homogeneous, so a global logit *gain* -- ordinary temperature scaling -- is
    exactly a no-op here; what is learnable is the logit mean relative to its
    uncertainty, and ``lambda = 0`` is the head exactly as trained.

    The CDF-head constants and the quadrature orders travel with the belief for
    the same reason :class:`triton_tagi.hsm_calibration.LogGainPosterior`
    carries ``sigma_v``: a scale fitted under one aleatoric activation, or
    under one set of quadrature rules, is not the same quantity as a scale
    fitted under another, and reading one under the other silently changes the
    model. :meth:`require_consistent` raises instead.

    Attributes:
        mean: Log-scale mean ``lambda``.
        variance: Log-scale variance ``q``, non-negative. Zero is a
            deterministic scale.
        epsilon: CDF-head variance floor the belief was fitted under.
        kappa: CDF-head variance range the belief was fitted under.
        laplace_order: Gauss--Legendre order over the Laplace variable.
        hermite_order: Gauss--Hermite order over each variance head.
        scale_order: Gauss--Hermite order over ``L`` itself.
        map_scale: Scale of the Laplace map ``t = y / [a (1 - y)]``.
        samples: Calibration labels that informed the belief, or None. The note
            asks for this alongside any empirical result: a single shared
            scalar can still be weakly identified when the likelihood is flat,
            and the reported ``q`` is then essentially the prior.
        method: Name of the recursion or read-out that produced the belief, or
            None for a stated prior. The sequential variants report the ``q``
            of the recursion they ran, which is not an exact posterior.
    """

    mean: float
    variance: float
    epsilon: float
    kappa: float
    laplace_order: int = DEFAULT_LAPLACE_ORDER
    hermite_order: int = DEFAULT_HERMITE_ORDER
    scale_order: int = DEFAULT_SCALE_ORDER
    map_scale: float = 1.0
    samples: float | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError(f"log-scale mean must be finite, got {self.mean}")
        if not math.isfinite(self.variance) or self.variance < 0.0:
            raise ValueError(
                f"log-scale variance must be finite and nonnegative, got {self.variance}"
            )
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be finite and positive, got {self.epsilon}")
        if not math.isfinite(self.kappa) or self.kappa <= 0.0:
            raise ValueError(f"kappa must be finite and positive, got {self.kappa}")
        for name in ("laplace_order", "hermite_order", "scale_order"):
            order = getattr(self, name)
            if int(order) < 4:
                raise ValueError(f"{name} must be at least four, got {order}")
        if not math.isfinite(self.map_scale) or self.map_scale <= 0.0:
            raise ValueError(f"map_scale must be finite and positive, got {self.map_scale}")

    @classmethod
    def prior(
        cls,
        *,
        epsilon: float,
        kappa: float,
        mean: float = DEFAULT_PRIOR_MEAN,
        variance: float = DEFAULT_PRIOR_VARIANCE,
        laplace_order: int = DEFAULT_LAPLACE_ORDER,
        hermite_order: int = DEFAULT_HERMITE_ORDER,
        scale_order: int = DEFAULT_SCALE_ORDER,
        map_scale: float = 1.0,
    ) -> LogScalePosterior:
        """Return the stated starting prior; ``mean = 0`` is the head as trained.

        The defaults are the note's ``lambda_0 = 0``, ``q_0 = 0.5^2``. For a
        posterior comparison at a fixed network checkpoint, every calibration
        fit should start from this same complete prior.
        """

        return cls(
            mean=float(mean),
            variance=float(variance),
            epsilon=float(epsilon),
            kappa=float(kappa),
            laplace_order=int(laplace_order),
            hermite_order=int(hermite_order),
            scale_order=int(scale_order),
            map_scale=float(map_scale),
        )

    @classmethod
    def unit(
        cls,
        *,
        epsilon: float,
        kappa: float,
        laplace_order: int = DEFAULT_LAPLACE_ORDER,
        hermite_order: int = DEFAULT_HERMITE_ORDER,
        scale_order: int = DEFAULT_SCALE_ORDER,
        map_scale: float = 1.0,
    ) -> LogScalePosterior:
        """Return the deterministic unit scale ``s = 1``, the uncalibrated head.

        This is the reference every calibrated belief has to beat: the frozen
        network read out with its trained deviation unscaled.
        """

        return cls.prior(
            epsilon=epsilon,
            kappa=kappa,
            mean=0.0,
            variance=0.0,
            laplace_order=laplace_order,
            hermite_order=hermite_order,
            scale_order=scale_order,
            map_scale=map_scale,
        )

    @property
    def scale_median(self) -> float:
        """Posterior median scale ``exp(lambda)``, the point estimate to predict at."""

        return math.exp(self.mean)

    @property
    def scale_mean(self) -> float:
        """Posterior mean scale ``exp(lambda + q / 2)``.

        Never substitute this, or ``E[e^{2L}]``, inside Remax in place of
        integrating the posterior: Remax is nonlinear, so the two differ.
        """

        return math.exp(self.mean + 0.5 * self.variance)

    def deterministic(self) -> LogScalePosterior:
        """Return the same mean with the variance set to zero.

        Reading the fitted scale as a point value isolates what the calibration
        uncertainty buys over predicting at ``m(lambda)``.
        """

        return replace(self, variance=0.0)

    def require_consistent(
        self,
        *,
        epsilon: float | None = None,
        kappa: float | None = None,
        laplace_order: int | None = None,
        hermite_order: int | None = None,
        scale_order: int | None = None,
        map_scale: float | None = None,
    ) -> None:
        """Raise unless the supplied provenance is the one the belief was fitted under.

        ``None`` means "take the belief's own value". Anything supplied and
        different is a silent model change and is rejected.
        """

        supplied = {
            "epsilon": epsilon,
            "kappa": kappa,
            "laplace_order": laplace_order,
            "hermite_order": hermite_order,
            "scale_order": scale_order,
            "map_scale": map_scale,
        }
        for name, value in supplied.items():
            if value is None:
                continue
            held = getattr(self, name)
            if type(held)(value) != held:
                raise ValueError(
                    f"the log-scale belief was fitted at {name}={held} and cannot be read "
                    f"at {name}={value}"
                )

    def resolve(
        self,
        *,
        epsilon: float | None = None,
        kappa: float | None = None,
        laplace_order: int | None = None,
        hermite_order: int | None = None,
        scale_order: int | None = None,
        map_scale: float | None = None,
    ) -> dict[str, Any]:
        """Return the belief's provenance after checking any supplied override."""

        self.require_consistent(
            epsilon=epsilon,
            kappa=kappa,
            laplace_order=laplace_order,
            hermite_order=hermite_order,
            scale_order=scale_order,
            map_scale=map_scale,
        )
        return {
            "epsilon": self.epsilon,
            "kappa": self.kappa,
            "laplace_order": self.laplace_order,
            "hermite_order": self.hermite_order,
            "scale_order": self.scale_order,
            "map_scale": self.map_scale,
        }

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly record of the belief and its provenance."""

        record: dict[str, Any] = {
            "log_scale_mean": self.mean,
            "log_scale_variance": self.variance,
            "scale_median": self.scale_median,
            "scale_mean": self.scale_mean,
            "epsilon": self.epsilon,
            "kappa": self.kappa,
            "laplace_order": self.laplace_order,
            "hermite_order": self.hermite_order,
            "scale_order": self.scale_order,
            "map_scale": self.map_scale,
        }
        if self.samples is not None:
            record["samples"] = self.samples
        if self.method is not None:
            record["method"] = self.method
        return record


# ──────────────────────────────────────────────────────────────────────────────
#  Surrogate moments and the consistency the projections need
# ──────────────────────────────────────────────────────────────────────────────


def remax_event_surrogate_variance(
    probabilities: Tensor,
    class_variance: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the moment-matching noise ``R_i`` and the indicator variance.

    Implements the Bernoulli total-variance split
    ``Var(B_i) = v_{A_i} + R_i = p_i (1 - p_i)``, with
    ``R_i = E[A_i (1 - A_i)]`` the average label randomness left conditionally
    on the latent Remax probability and ``v_{A_i}`` its epistemic dispersion.
    ``R_i`` is a moment-matching device for the Gaussian surrogate
    ``Y_i = A_i + E_i``; it is not the TAGI-V logit error ``V_i``, which has
    already been integrated into ``A_i`` and must not be added again.

    Only ``p_c`` and ``c_c`` are needed for the recursion itself. The second
    moment is what makes ``R_c`` interpretable and what reports predictive
    uncertainty.

    Args:
        probabilities: ``p``, shape ``(..., K)``.
        class_variance: ``v_A``, the diagonal of ``Sigma_A``, same shape.

    Returns:
        noise: ``R_i``, clamped at zero.
        indicator_variance: ``v_{A_i} + R_i``, which is ``p_i (1 - p_i)``
            whenever the moments are consistent.
    """

    work_probabilities = probabilities.double()
    work_variance = class_variance.double().clamp_min(0.0)
    noise = expected_bernoulli_variance(work_probabilities, work_variance)
    return noise, noise + work_variance


def remax_scale_consistency_slack(
    variance: float,
    probabilities: Tensor,
    cov_scale: Tensor,
) -> Tensor:
    """Return ``q p_i (1 - p_i) - c_i^2``, the Cauchy--Schwarz slack per class.

    Cauchy--Schwarz applied to ``(L, B_i)`` gives
    ``c_i^2 <= q p_i (1 - p_i)`` for exact moments, so the slack is
    non-negative. Approximate moments -- and every quadrature is approximate --
    must be checked, because a violated bound makes the event variance step
    remove more than the whole prior variance and makes the mean step exceed
    anything the model permits. The projections here clamp ``c_c`` to the bound
    rather than propagate an inconsistent moment.

    Args:
        variance: Current ``q``, non-negative.
        probabilities: ``p``, shape ``(..., K)``.
        cov_scale: ``c = Cov(L, A)``, same shape.

    Returns:
        The slack, same shape; negative entries flag inconsistent moments.
    """

    work_probabilities = probabilities.double()
    return variance * work_probabilities * (1.0 - work_probabilities) - cov_scale.double().square()


def _row_vector(values: Tensor, num_classes: int | None, name: str) -> list[float]:
    """Return one calibration row of a ``(K,)`` or ``(1, K)`` tensor as floats."""

    work = torch.as_tensor(values).double().reshape(-1)
    if work.numel() < 2:
        raise ValueError(f"{name} must hold at least two classes, got {work.numel()}")
    if num_classes is not None and work.numel() != num_classes:
        raise ValueError(f"{name} must have {num_classes} classes, got {work.numel()}")
    return work.tolist()


def _checked_state(
    mean: float,
    variance: float,
    variance_decrement_cap: float,
) -> None:
    if not math.isfinite(mean):
        raise ValueError(f"mean must be finite, got {mean}")
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError(f"variance must be finite and nonnegative, got {variance}")
    if not 0.0 < variance_decrement_cap < 1.0:
        raise ValueError(f"variance_decrement_cap must lie in (0, 1), got {variance_decrement_cap}")


def _floored(probability: float) -> float:
    """Return ``p`` floored away from both zero and one."""

    return min(max(probability, _PROBABILITY_FLOOR), 1.0 - _PROBABILITY_FLOOR)


def _consistent_covariance(variance: float, probability: float, covariance: float) -> float:
    """Return ``c_c`` clamped to the Cauchy--Schwarz bound ``sqrt(q p (1 - p))``."""

    bound = math.sqrt(max(variance * probability * (1.0 - probability), 0.0))
    return min(max(covariance, -bound), bound)


def _exact_mean_step(variance: float, probability: float, covariance: float) -> float:
    """Return the exact one-step increment ``c_c / p_c`` of ``E[L | C = c]``."""

    return _consistent_covariance(variance, probability, covariance) / probability


# ──────────────────────────────────────────────────────────────────────────────
#  The three stated variance steps
# ──────────────────────────────────────────────────────────────────────────────


def remax_scale_event_update(
    mean: float,
    variance: float,
    probabilities: Tensor,
    cov_scale: Tensor,
    observed: int,
    *,
    variance_decrement_cap: float = 0.999,
) -> tuple[float, float]:
    """Assimilate one observed class through the two-variable event surrogate.

    Conditioning

        ``[L, Y_c] ~ N([lambda, p_c], [[q, c_c], [c_c, p_c (1 - p_c)]])``

    on the observed event ``B_c = 1`` gives

        ``lambda+ = lambda + c_c / p_c``,
        ``q+_event = q - c_c^2 / [p_c (1 - p_c)]``

    where ``p_c (1 - p_c) = v_{A_c} + R_c`` is the total variance of the
    observed indicator. This is the closest match to the intended source.

    The mean step is *exact* for the categorical likelihood under the current
    working prior: ``E[L | C = c] = E[L A_c] / p_c = lambda + c_c / p_c``. The
    variance step is a Gaussian projection. It equals the exact posterior
    variance **averaged over the binary partition** ``{C = c, C != c}``, not the
    variance conditional on the realised class; those are different quantities.
    Replacing the result by a Gaussian and repeating it is therefore
    assumed-density filtering, and its ordering sensitivity has to be checked
    against the batch fit of :func:`fit_remax_log_scale`.

    Neither ``Z`` nor ``U`` is a variable of this surrogate, so neither output
    head is touched. The other ``K - 1`` zero indicators are not assimilated as
    independent Bernoulli observations.

    Args:
        mean: Current ``lambda``.
        variance: Current ``q``, non-negative.
        probabilities: ``p`` for this calibration row, shape ``(K,)`` or
            ``(1, K)``.
        cov_scale: ``c = Cov(L, A)`` for the same row, same shape.
        observed: Observed class index ``c``.
        variance_decrement_cap: Largest fraction of ``q`` one label may remove.
            Capping keeps the belief from freezing, which flooring a negative
            result at exactly zero would do.

    Returns:
        The updated ``lambda`` and ``q``.
    """

    _checked_state(mean, variance, variance_decrement_cap)
    row = _row_vector(probabilities, None, "probabilities")
    covariances = _row_vector(cov_scale, len(row), "cov_scale")
    if not 0 <= int(observed) < len(row):
        raise ValueError(f"observed class {observed} is out of range for {len(row)} classes")

    raw = row[int(observed)]
    if not math.isfinite(raw):
        return mean, variance
    probability = _floored(raw)
    total = probability * (1.0 - probability)
    covariance = covariances[int(observed)]
    if not math.isfinite(covariance):
        return mean, variance

    updated_mean = mean + _exact_mean_step(variance, probability, covariance)
    decrement = min(covariance * covariance / total, variance_decrement_cap * variance)
    return updated_mean, max(variance - decrement, 0.0)


def remax_scale_full_update(
    mean: float,
    variance: float,
    probabilities: Tensor,
    cov_scale: Tensor,
    observed: int,
    *,
    variance_decrement_cap: float = 0.999,
) -> tuple[float, float]:
    """Assimilate one observed class through the full categorical projection.

    With ``S = diag(p) - p p^T`` on the simplex tangent space and
    ``1^T c = 0``,

        ``lambda+ = lambda + c^T S^dagger (e_c - p) = lambda + c_c / p_c``,
        ``q+_full = q - c^T S^dagger c = q - sum_i c_i^2 / p_i``

    where ``e_c`` is a one-hot basis vector, distinct from the vector ``e`` of
    logit variances. The explicit sum is the pseudoinverse form whenever every
    ``p_i > 0``.

    The mean step is the same exact one-step mean as
    :func:`remax_scale_event_update`; only the variance decrement differs.
    Here ``q+_full = E_C[Var(L | C)]``, the posterior variance averaged over the
    full ``K``-way partition rather than the binary one, so it coincides with
    the event form at ``K = 2`` -- where ``c_2 = -c_1`` and ``p_2 = 1 - p_1``
    force the two decrements to agree -- and satisfies
    ``q+_full <= q+_event`` for ``K > 2``, the extra reduction coming from the
    distinctions the event channel throws away by pooling ``C != c``.

    This is **one joint categorical update**, not a product of ``K``
    independent binary updates: the latter would count the same label ``K``
    times and is not what the projection says.

    Args:
        mean: Current ``lambda``.
        variance: Current ``q``, non-negative.
        probabilities: ``p`` for this calibration row, shape ``(K,)`` or
            ``(1, K)``.
        cov_scale: ``c = Cov(L, A)`` for the same row, same shape.
        observed: Observed class index ``c``.
        variance_decrement_cap: Largest fraction of ``q`` one label may remove.

    Returns:
        The updated ``lambda`` and ``q``.
    """

    _checked_state(mean, variance, variance_decrement_cap)
    row = _row_vector(probabilities, None, "probabilities")
    covariances = _row_vector(cov_scale, len(row), "cov_scale")
    if not 0 <= int(observed) < len(row):
        raise ValueError(f"observed class {observed} is out of range for {len(row)} classes")

    raw = row[int(observed)]
    covariance = covariances[int(observed)]
    if not math.isfinite(raw) or not math.isfinite(covariance):
        return mean, variance
    probability = _floored(raw)

    updated_mean = mean + _exact_mean_step(variance, probability, covariance)

    quadratic = 0.0
    for entry, cross in zip(row, covariances, strict=True):
        if not math.isfinite(entry) or not math.isfinite(cross):
            return mean, variance
        quadratic += cross * cross / _floored(entry)
    decrement = min(quadratic, variance_decrement_cap * variance)
    return updated_mean, max(variance - decrement, 0.0)


def remax_scale_tilt_update(
    mean: float,
    variance: float,
    mu_z: Tensor,
    var_z: Tensor,
    nu: Tensor,
    r: Tensor,
    observed: int,
    *,
    epsilon: float,
    kappa: float,
    laplace_order: int = DEFAULT_LAPLACE_ORDER,
    hermite_order: int = DEFAULT_HERMITE_ORDER,
    scale_order: int = DEFAULT_SCALE_ORDER,
    map_scale: float = 1.0,
    variance_decrement_cap: float = 0.999,
) -> tuple[float, float]:
    """Assimilate one observed class through the exact scalar moment tilt.

    Because ``L`` is scalar, the realised-label posterior moments can be
    retained directly instead of projected:

        ``lambda+ = E_L[L m_c(L)] / p_c``,
        ``q+_tilt = E_L[L^2 m_c(L)] / p_c - (lambda+)^2``

    with ``p_c = E_L[m_c(L)]``. These are exact one-step integral identities
    under the current working prior, evaluated here by the same Gauss--Hermite
    rule over ``L`` that :func:`triton_tagi.cdf_remax.remax_scale_moments`
    uses, calling
    :func:`triton_tagi.cdf_remax.remax_conditional_moments` at each node. The
    mean is therefore algebraically identical to the event and full mean step:
    ``E_L[L m_c(L)] = lambda p_c + c_c``.

    Unlike the two projections, ``q+_tilt`` is ``Var(L | C = c)`` for the
    realised class and **may exceed** ``q``: an averaged conditional variance
    cannot grow, a single conditional one can. Only a floor is applied, never a
    cap from above. Replacing the resulting non-Gaussian posterior by a
    Gaussian and continuing is assumed-density filtering.

    Args:
        mean: Current ``lambda``.
        variance: Current ``q``, non-negative.
        mu_z: Prediction-head means for one row, shape ``(K,)`` or ``(1, K)``.
        var_z: Prediction-head variances, same shape.
        nu: Variance-head means, same shape.
        r: Variance-head variances, same shape.
        observed: Observed class index ``c``.
        epsilon: CDF-head variance floor.
        kappa: CDF-head variance range.
        laplace_order: Gauss--Legendre order over the Laplace variable.
        hermite_order: Gauss--Hermite order over each variance head.
        scale_order: Gauss--Hermite order over ``L``.
        map_scale: Scale of the Laplace map.
        variance_decrement_cap: Largest fraction of ``q`` one label may remove,
            applied as a floor so the belief cannot freeze.

    Returns:
        The updated ``lambda`` and ``q``.
    """

    _checked_state(mean, variance, variance_decrement_cap)
    values, weights, first = _scale_node_moments(
        _as_rows(mu_z),
        _as_rows(var_z),
        _as_rows(nu),
        _as_rows(r),
        scale_mean=mean,
        scale_variance=variance,
        epsilon=epsilon,
        kappa=kappa,
        laplace_order=laplace_order,
        hermite_order=hermite_order,
        scale_order=scale_order,
        map_scale=map_scale,
    )
    if first.shape[0] != 1:
        raise ValueError("remax_scale_tilt_update takes one calibration row at a time")
    num_classes = first.shape[1]
    if not 0 <= int(observed) < num_classes:
        raise ValueError(f"observed class {observed} is out of range for {num_classes} classes")

    tilt = weights * first[0, int(observed)]
    zeroth = float(tilt.sum())
    if not math.isfinite(zeroth):
        return mean, variance
    probability = _floored(zeroth)
    tilted_mean = float((tilt * values).sum()) / probability
    tilted_second = float((tilt * values * values).sum()) / probability
    if not math.isfinite(tilted_mean) or not math.isfinite(tilted_second):
        return mean, variance

    floor = (1.0 - variance_decrement_cap) * variance
    updated_variance = max(tilted_second - tilted_mean * tilted_mean, floor, 0.0)
    return tilted_mean, updated_variance


# ──────────────────────────────────────────────────────────────────────────────
#  Sequential assumed-density calibration
# ──────────────────────────────────────────────────────────────────────────────


def _as_rows(values: Tensor) -> Tensor:
    """Return a ``(K,)`` or ``(batch, K)`` tensor as float64 ``(batch, K)``."""

    work = torch.as_tensor(values).double()
    if work.dim() == 1:
        work = work.unsqueeze(0)
    if work.dim() != 2:
        raise ValueError(f"forward summaries must be (K,) or (batch, K), got {tuple(work.shape)}")
    return work


def _scale_node_moments(
    mu_z: Tensor,
    var_z: Tensor,
    nu: Tensor,
    r: Tensor,
    *,
    scale_mean: float,
    scale_variance: float,
    epsilon: float,
    kappa: float,
    laplace_order: int,
    hermite_order: int,
    scale_order: int,
    map_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the scale nodes, their weights, and ``m_i(l_h)`` at every node.

    The nodes are the same ``lambda + sqrt(q) x_h`` rule that
    :func:`triton_tagi.cdf_remax.remax_scale_moments` integrates ``L`` with, so
    any moment formed from the returned values reproduces that function's
    ``p`` and ``c`` exactly rather than approximately.
    """

    if scale_variance < 0.0:
        raise ValueError(f"scale_variance must be non-negative, got {scale_variance}")
    nodes, weights = hermite_rule(scale_order, reference=mu_z)
    deviation = math.sqrt(max(scale_variance, 0.0))
    values = scale_mean + deviation * nodes

    columns = []
    for value in values.tolist():
        first, _ = remax_conditional_moments(
            mu_z,
            var_z,
            nu,
            r,
            epsilon=epsilon,
            kappa=kappa,
            log_scale=value,
            laplace_order=laplace_order,
            hermite_order=hermite_order,
            map_scale=map_scale,
            cross_moments=False,
        )
        columns.append(first)
    return values, weights, torch.stack(columns, dim=-1)


def calibrate_remax_log_scale_adf(
    mu_z: Tensor,
    var_z: Tensor,
    nu: Tensor,
    r: Tensor,
    labels: Tensor,
    *,
    epsilon: float,
    kappa: float,
    prior_mean: float = DEFAULT_PRIOR_MEAN,
    prior_variance: float = DEFAULT_PRIOR_VARIANCE,
    initial_mean: float | None = None,
    method: str = "event",
    laplace_order: int = DEFAULT_LAPLACE_ORDER,
    hermite_order: int = DEFAULT_HERMITE_ORDER,
    scale_order: int = DEFAULT_SCALE_ORDER,
    map_scale: float = 1.0,
    process_variance: float = 0.0,
    variance_decrement_cap: float = 0.999,
    visit_order: Tensor | None = None,
) -> LogScalePosterior:
    """Calibrate the shared log-scale by sequential assumed-density filtering.

    One observed-class update per calibration sample, with the scale's
    predictive moments ``(p, c)`` recomputed from the current belief before the
    next sample is assimilated. For ``K > 2`` there is no tree-node
    factorization to exploit: the other ``K - 1`` zero indicators are **not**
    assimilated as independent Bernoulli observations, because they are not
    independent of the observed one and doing so would count the label ``K``
    times.

    Because the log-scale is scalar, the order-independent batch fit of
    :func:`fit_remax_log_scale` is preferred whenever the whole calibration
    split is available. This driver exists for a genuine stream, for the note's
    epoch-wise tracking option, and as the ordering-sensitivity check the
    assumed-density approximation requires -- reordering the stream changes the
    answer, and that is the diagnostic, not a bug. Do not replay a calibration
    set through it while treating its outcomes as new evidence.

    **The epoch-wise reset is a heuristic, not exact Bayes.** Passing
    ``initial_mean`` reproduces the intended source's tracking procedure:
    retain ``lambda`` from the previous validation pass and reset
    ``q <- prior_variance`` before the next one. After previous calibration
    labels have influenced the retained mean, resetting only the variance does
    **not** make reuse of those same labels a new independent Bayesian
    likelihood, and the returned ``q`` describes the chosen recursion rather
    than an exact posterior for a single fixed data set. For a fixed-checkpoint
    posterior comparison, leave ``initial_mean`` at ``None`` so every fit
    starts from the same complete prior ``(prior_mean, prior_variance)``.
    Adding ``process_variance`` is the separate, stated random-walk model
    ``q <- q + q_process`` for genuinely fresh streaming observations with a
    drifting scale; it is not the same operation as resetting to ``q_0``.

    Args:
        mu_z: Frozen calibration prediction-head means, shape ``(batch, K)``.
        var_z: Frozen prediction-head variances, same shape.
        nu: Frozen variance-head means, same shape.
        r: Frozen variance-head variances, same shape.
        labels: Observed class indices, shape ``(batch,)``.
        epsilon: CDF-head variance floor.
        kappa: CDF-head variance range.
        prior_mean: Prior ``lambda_0``; zero is the head as trained.
        prior_variance: Prior ``q_0``, and the value ``q`` is reset to when
            ``initial_mean`` is supplied.
        initial_mean: Log-scale mean retained from a previous epoch. ``None``
            starts from ``prior_mean``.
        method: One of :data:`REMAX_SCALE_METHODS`. ``event`` is the two-variable
            observed-class projection and the closest match to the intended
            source; ``full`` is the categorical tangent-space projection;
            ``tilt`` retains the exact realised-label scalar moments.
        laplace_order: Gauss--Legendre order over the Laplace variable.
        hermite_order: Gauss--Hermite order over each variance head.
        scale_order: Gauss--Hermite order over ``L``.
        map_scale: Scale of the Laplace map.
        process_variance: Added to ``q`` before every label, for the stated
            random-walk model.
        variance_decrement_cap: Largest fraction of ``q`` one label may remove.
        visit_order: Optional permutation of the sample index, so the same data
            can be replayed in a different order to measure the sensitivity.

    Returns:
        The filtered belief, carrying the label count and the method name.
    """

    if method not in REMAX_SCALE_METHODS:
        raise ValueError(f"method must be one of {REMAX_SCALE_METHODS}, got {method!r}")
    if not math.isfinite(process_variance) or process_variance < 0.0:
        raise ValueError(f"process_variance must be finite and nonnegative, got {process_variance}")
    if not math.isfinite(prior_variance) or prior_variance < 0.0:
        raise ValueError(f"prior_variance must be finite and nonnegative, got {prior_variance}")

    summaries = _checked_summaries(mu_z, var_z, nu, r)
    target = _checked_labels(labels, summaries[0])
    batch = target.shape[0]
    sequence = _checked_order(visit_order, batch, target.device)

    mean = float(prior_mean if initial_mean is None else initial_mean)
    variance = float(prior_variance)
    for row in sequence.tolist():
        variance += process_variance
        window = tuple(value[row : row + 1] for value in summaries)
        observed = int(target[row])
        if method == "tilt":
            mean, variance = remax_scale_tilt_update(
                mean,
                variance,
                *window,
                observed,
                epsilon=epsilon,
                kappa=kappa,
                laplace_order=laplace_order,
                hermite_order=hermite_order,
                scale_order=scale_order,
                map_scale=map_scale,
                variance_decrement_cap=variance_decrement_cap,
            )
            continue
        probabilities, _, cov_scale = remax_scale_moments(
            *window,
            scale_mean=mean,
            scale_variance=variance,
            epsilon=epsilon,
            kappa=kappa,
            laplace_order=laplace_order,
            hermite_order=hermite_order,
            scale_order=scale_order,
            map_scale=map_scale,
            cross_moments=False,
        )
        projection = remax_scale_event_update if method == "event" else remax_scale_full_update
        mean, variance = projection(
            mean,
            variance,
            probabilities,
            cov_scale,
            observed,
            variance_decrement_cap=variance_decrement_cap,
        )

    return LogScalePosterior(
        mean=mean,
        variance=variance,
        epsilon=float(epsilon),
        kappa=float(kappa),
        laplace_order=int(laplace_order),
        hermite_order=int(hermite_order),
        scale_order=int(scale_order),
        map_scale=float(map_scale),
        samples=float(batch),
        method=f"adf-{method}",
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Order-independent batch reference
# ──────────────────────────────────────────────────────────────────────────────


def _checked_summaries(
    mu_z: Tensor,
    var_z: Tensor,
    nu: Tensor,
    r: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return the four forward summaries as float64 ``(batch, K)`` tensors."""

    rows = tuple(_as_rows(value) for value in (mu_z, var_z, nu, r))
    shape = rows[0].shape
    names = ("mu_z", "var_z", "nu", "r")
    for name, value in zip(names, rows, strict=True):
        if value.shape != shape:
            raise ValueError(
                f"{name} must have the same shape as mu_z, got {tuple(value.shape)} "
                f"against {tuple(shape)}"
            )
    if shape[-1] < 2:
        raise ValueError(f"the forward summaries must hold at least two classes, got {shape[-1]}")
    if not bool(torch.isfinite(rows[0]).all()) or not bool(torch.isfinite(rows[2]).all()):
        raise ValueError("the head means must be finite")
    for name, value in (("var_z", rows[1]), ("r", rows[3])):
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError(f"{name} must be finite and nonnegative")
    return (rows[0], rows[1].clamp_min(0.0), rows[2], rows[3].clamp_min(0.0))


def _checked_labels(labels: Tensor, reference: Tensor) -> Tensor:
    target = torch.as_tensor(labels).reshape(-1).long().to(reference.device)
    if target.shape[0] != reference.shape[0]:
        raise ValueError("labels must have shape (batch,) matching the forward summaries")
    if bool(((target < 0) | (target >= reference.shape[-1])).any()):
        raise ValueError("labels contain an invalid class index")
    return target


def _checked_order(visit_order: Tensor | None, batch: int, device: torch.device) -> Tensor:
    if visit_order is None:
        return torch.arange(batch, device=device)
    sequence = torch.as_tensor(visit_order).reshape(-1).long().to(device)
    if sequence.shape[0] != batch:
        raise ValueError("visit_order must be a permutation of the calibration index")
    if bool(((sequence < 0) | (sequence >= batch)).any()):
        raise ValueError("visit_order must be a permutation of the calibration index")
    return sequence


def _checked_weights(sample_weight: Tensor | None, reference: Tensor) -> Tensor:
    if sample_weight is None:
        return torch.ones(reference.shape[0], dtype=torch.float64, device=reference.device)
    weights = torch.as_tensor(sample_weight).reshape(-1).double().to(reference.device)
    if weights.shape[0] != reference.shape[0]:
        raise ValueError("sample_weight must have one entry per calibration row")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("sample_weight must be finite and nonnegative")
    return weights


def _resolve_chunk_size(num_classes: int, laplace_order: int, hermite_order: int) -> int:
    per_row = max(1, num_classes * laplace_order * hermite_order)
    return max(1, DEFAULT_CHUNK_ELEMENTS // per_row)


def remax_scale_log_posterior_on_grid(
    mu_z: Tensor,
    var_z: Tensor,
    nu: Tensor,
    r: Tensor,
    labels: Tensor,
    *,
    epsilon: float,
    kappa: float,
    prior_mean: float = DEFAULT_PRIOR_MEAN,
    prior_variance: float = DEFAULT_PRIOR_VARIANCE,
    grid: Tensor | None = None,
    grid_bounds: tuple[float, float] | None = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    laplace_order: int = DEFAULT_LAPLACE_ORDER,
    hermite_order: int = DEFAULT_HERMITE_ORDER,
    map_scale: float = 1.0,
    sample_weight: Tensor | None = None,
    chunk_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate the scalar log calibration posterior on a grid of log-scales.

    Up to a constant,

        ``log q_cal(l) = log N(l; lambda_0, q_0)
                         + sum_n log m_{n,c_n}(l)``

    which is the negative of the note's objective
    ``J(l) = -sum_n log m_{n,c_n}(l) + (l - lambda_0)^2 / (2 q_0)``. Every
    calibration case enters through the exact conditional Remax moment
    ``m_{n,c_n}(l) = E[A_{c_n} | L = l]``, so the sum is the true categorical
    log-likelihood in ``l`` at the frozen summaries. Without the prior term it
    is held-out categorical negative log likelihood.

    The likelihood is ``log E[A_c]``, **not** ``E[log A_c]``: the latter is a
    different objective and can be ``-inf``, because an individual Remax
    component has positive probability of being exactly zero. It is also a
    modular (cut) likelihood conditioned on the frozen predictive summaries,
    treating calibration cases as conditionally independent, and is therefore
    not the joint weight-and-scale posterior.

    Being scalar, this evaluation is order-independent -- unlike the sequential
    recursion of :func:`calibrate_remax_log_scale_adf`, which is why it is the
    reference.

    Args:
        mu_z: Frozen calibration prediction-head means, shape ``(batch, K)``.
        var_z: Frozen prediction-head variances, same shape.
        nu: Frozen variance-head means, same shape.
        r: Frozen variance-head variances, same shape.
        labels: Observed class indices, shape ``(batch,)``.
        epsilon: CDF-head variance floor.
        kappa: CDF-head variance range.
        prior_mean: Prior ``lambda_0``.
        prior_variance: Prior ``q_0``, strictly positive.
        grid: Explicit grid of log-scales, shape ``(grid_size,)``. ``None``
            builds a uniform grid from ``grid_bounds``.
        grid_bounds: Inclusive interval for the built grid. ``None`` covers the
            prior, which keeps a data-free fit at its stated variance instead
            of truncating it.
        grid_size: Number of grid points, at least five.
        laplace_order: Gauss--Legendre order over the Laplace variable.
        hermite_order: Gauss--Hermite order over each variance head.
        map_scale: Scale of the Laplace map.
        sample_weight: Optional non-negative multiplicity of every calibration
            row, so a repeated identical row need be evaluated once: a weight
            of ``k`` contributes ``k log m_{n,c_n}(l)``, exactly as ``k``
            copies of that row would. ``None`` weights every row once.
        chunk_size: Rows per block; ``None`` derives one from the class count
            and the quadrature orders.

    Returns:
        grid: The grid the values were evaluated on, shape ``(grid_size,)``.
        log_posterior: Unnormalized values, same shape.
    """

    if not math.isfinite(prior_variance) or prior_variance <= 0.0:
        raise ValueError(f"prior_variance must be finite and positive, got {prior_variance}")
    summaries = _checked_summaries(mu_z, var_z, nu, r)
    target = _checked_labels(labels, summaries[0])
    weights = _checked_weights(sample_weight, summaries[0])
    batch, num_classes = summaries[0].shape
    device = summaries[0].device

    if grid is None:
        if grid_size < 5:
            raise ValueError(f"grid_size must be at least five, got {grid_size}")
        lower, upper = grid_bounds or _prior_grid_bounds(prior_mean, prior_variance)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("grid_bounds must be finite and increasing")
        grid = torch.linspace(lower, upper, grid_size, dtype=torch.float64, device=device)
    grid = torch.as_tensor(grid).to(device=device, dtype=torch.float64).reshape(-1)
    if grid.shape[0] < 5:
        raise ValueError("grid must hold at least five points")

    rows = chunk_size or _resolve_chunk_size(num_classes, laplace_order, hermite_order)
    if rows < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    log_likelihood = torch.zeros_like(grid)
    for index, value in enumerate(grid.tolist()):
        total = torch.zeros((), dtype=torch.float64, device=device)
        for start in range(0, batch, rows):
            stop = start + rows
            first, _ = remax_conditional_moments(
                summaries[0][start:stop],
                summaries[1][start:stop],
                summaries[2][start:stop],
                summaries[3][start:stop],
                epsilon=epsilon,
                kappa=kappa,
                log_scale=value,
                laplace_order=laplace_order,
                hermite_order=hermite_order,
                map_scale=map_scale,
                cross_moments=False,
            )
            picked = first.gather(1, target[start:stop, None]).squeeze(1)
            total = total + (weights[start:stop] * picked.clamp_min(_LOG_FLOOR).log()).sum()
        log_likelihood[index] = total

    penalty = -0.5 * (grid - prior_mean).square() / prior_variance
    return grid, log_likelihood + penalty


def fit_remax_log_scale(
    mu_z: Tensor,
    var_z: Tensor,
    nu: Tensor,
    r: Tensor,
    labels: Tensor,
    *,
    epsilon: float,
    kappa: float,
    prior_mean: float = DEFAULT_PRIOR_MEAN,
    prior_variance: float = DEFAULT_PRIOR_VARIANCE,
    method: str = "grid",
    grid_bounds: tuple[float, float] | None = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    refinements: int = DEFAULT_GRID_REFINEMENTS,
    laplace_order: int = DEFAULT_LAPLACE_ORDER,
    hermite_order: int = DEFAULT_HERMITE_ORDER,
    scale_order: int = DEFAULT_SCALE_ORDER,
    map_scale: float = 1.0,
    sample_weight: Tensor | None = None,
    chunk_size: int | None = None,
) -> LogScalePosterior:
    """Fit the shared log-scale on a held-out calibration split.

    The log-scale is a single scalar, so this batch calculation is
    order-independent and is preferred over the repeated assumed-density
    updates of :func:`calibrate_remax_log_scale_adf`, exactly as
    :func:`triton_tagi.hsm_calibration.fit_hsm_log_gain` is preferred for the
    hierarchical gain. ``method="laplace"`` returns the MAP estimate of the
    note's objective ``J(l) = -sum_n log m_{n,c_n}(l) + (l - lambda_0)^2 /
    (2 q_0)`` together with ``[J''(l_hat)]^{-1}``; ``method="grid"`` returns
    the normalized moments of the scalar posterior
    ``q_cal(l) ∝ N(l; lambda_0, q_0) prod_n m_{n,c_n}(l)``, which also captures
    skewness and multiple modes. That Laplace *approximation to the calibration
    posterior* is unrelated to the *Laplace transform* the Remax moments
    themselves are built on.

    Call this once with the network frozen and the calibration split disjoint
    from training. Refitting from the same stated prior at a new checkpoint is
    fine; replaying the same labels while carrying the previous posterior
    precision forward would count the data twice. The fit conditions on the
    frozen forward summaries and treats calibration cases as conditionally
    independent -- a modular cut posterior, not a joint re-inference of the
    network weights, since a shared weight draw couples inputs and
    ``prod_n E_q[p(c_n | .)] != E_q[prod_n p(c_n | .)]``.

    The objective fits ``-log E[A_c]``, the probability after Remax and after
    marginalizing uncertainty, never ``-E[log A_c]``, which can be infinite.
    Predict either at the point estimate ``m(lambda)`` or by integrating the
    posterior with
    :func:`triton_tagi.cdf_remax.remax_scale_moments`; substituting ``E[e^L]``
    or ``E[e^{2L}]`` inside Remax is not the same thing.

    Args:
        mu_z: Frozen calibration prediction-head means, shape ``(batch, K)``.
        var_z: Frozen prediction-head variances, same shape.
        nu: Frozen variance-head means, same shape.
        r: Frozen variance-head variances, same shape.
        labels: Observed class indices, shape ``(batch,)``.
        epsilon: CDF-head variance floor.
        kappa: CDF-head variance range.
        prior_mean: Prior ``lambda_0``; the note proposes zero, centred on no
            rescaling.
        prior_variance: Prior ``q_0``; the note proposes ``0.5^2``. This is a
            stated regularization choice, not a data-supported universal
            default, and its sensitivity should be reported when the likelihood
            is weak.
        method: One of :data:`REMAX_SCALE_FIT_METHODS`.
        grid_bounds: Inclusive interval for the coarse sweep. ``None`` covers
            the prior.
        grid_size: Number of grid points per sweep.
        refinements: Sweeps that re-centre and rescale the grid on the current
            mode. A sharp posterior needs them: a grid whose spacing exceeds
            the posterior standard deviation cannot resolve it, and a large
            calibration split lands in exactly that regime.
        laplace_order: Gauss--Legendre order over the Laplace variable.
        hermite_order: Gauss--Hermite order over each variance head.
        scale_order: Gauss--Hermite order over ``L``. The likelihood itself
            never integrates ``L``, so this only travels with the returned
            belief as the provenance any later read-out must match.
        map_scale: Scale of the Laplace map.
        sample_weight: Optional non-negative multiplicity of every calibration
            row, as in :func:`remax_scale_log_posterior_on_grid`.
        chunk_size: Rows per block.

    Returns:
        The fitted belief, carrying the label count and the method name.
    """

    if method not in REMAX_SCALE_FIT_METHODS:
        raise ValueError(f"method must be one of {REMAX_SCALE_FIT_METHODS}, got {method!r}")
    if refinements < 0:
        raise ValueError(f"refinements must be nonnegative, got {refinements}")

    def sweep(window: Tensor | None) -> tuple[Tensor, Tensor]:
        return remax_scale_log_posterior_on_grid(
            mu_z,
            var_z,
            nu,
            r,
            labels,
            epsilon=epsilon,
            kappa=kappa,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            grid=window,
            grid_bounds=grid_bounds,
            grid_size=grid_size,
            laplace_order=laplace_order,
            hermite_order=hermite_order,
            map_scale=map_scale,
            sample_weight=sample_weight,
            chunk_size=chunk_size,
        )

    grid, log_posterior = sweep(None)
    prior_deviation = math.sqrt(prior_variance)
    for _ in range(refinements):
        mode, curvature = _laplace_moments(grid[None, :], log_posterior[None, :])
        window = _refined_grid(mode, curvature, prior_deviation, int(grid.shape[0]))
        grid, log_posterior = sweep(window.reshape(-1))

    if method == "grid":
        mean, variance = _grid_moments(grid[None, :], log_posterior[None, :])
    else:
        mean, variance = _laplace_moments(grid[None, :], log_posterior[None, :])
        variance = variance.clamp(max=prior_variance)

    weights = _checked_weights(sample_weight, _as_rows(mu_z))
    return LogScalePosterior(
        mean=float(mean.reshape(-1)[0]),
        variance=float(variance.reshape(-1)[0].clamp_min(0.0)),
        epsilon=float(epsilon),
        kappa=float(kappa),
        laplace_order=int(laplace_order),
        hermite_order=int(hermite_order),
        scale_order=int(scale_order),
        map_scale=float(map_scale),
        samples=float(weights.sum()),
        method=method,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Scoring
# ──────────────────────────────────────────────────────────────────────────────


def remax_scale_negative_log_likelihood(
    mu_z: Tensor,
    var_z: Tensor,
    nu: Tensor,
    r: Tensor,
    labels: Tensor,
    posterior: LogScalePosterior,
    *,
    epsilon: float | None = None,
    kappa: float | None = None,
    laplace_order: int | None = None,
    hermite_order: int | None = None,
    scale_order: int | None = None,
    map_scale: float | None = None,
    sample_weight: Tensor | None = None,
    chunk_size: int | None = None,
) -> Tensor:
    """Return the mean categorical cross-entropy in nats of ``E[A_c]``.

    Scores a held-out split under the fitted belief, integrating the log-scale
    posterior rather than substituting a point value: the probabilities are
    ``p = E_L[m(L)]`` from
    :func:`triton_tagi.cdf_remax.remax_scale_moments`. Pass
    ``posterior.deterministic()`` to score the point estimate instead and see
    what the calibration uncertainty buys.

    This is ``-log E[A_c]``, not ``-E[log A_c]``. The second is a different
    quantity and can be infinite, because an individual Remax component has
    positive probability of being exactly zero.

    Any provenance argument left at ``None`` is taken from the belief; anything
    supplied that disagrees with it raises, so a scale fitted under one CDF
    head or one set of quadrature orders cannot be silently scored under
    another.

    Args:
        mu_z: Prediction-head means, shape ``(batch, K)``.
        var_z: Prediction-head variances, same shape.
        nu: Variance-head means, same shape.
        r: Variance-head variances, same shape.
        labels: Observed class indices, shape ``(batch,)``.
        posterior: The fitted log-scale belief.
        epsilon: CDF-head variance floor, or None for the belief's own.
        kappa: CDF-head variance range, or None for the belief's own.
        laplace_order: Gauss--Legendre order, or None.
        hermite_order: Gauss--Hermite order over the heads, or None.
        scale_order: Gauss--Hermite order over ``L``, or None.
        map_scale: Scale of the Laplace map, or None.
        sample_weight: Optional non-negative row multiplicities; the result is
            the weighted mean.
        chunk_size: Rows per block.

    Returns:
        The mean negative log likelihood, a scalar tensor.
    """

    provenance = posterior.resolve(
        epsilon=epsilon,
        kappa=kappa,
        laplace_order=laplace_order,
        hermite_order=hermite_order,
        scale_order=scale_order,
        map_scale=map_scale,
    )
    summaries = _checked_summaries(mu_z, var_z, nu, r)
    target = _checked_labels(labels, summaries[0])
    weights = _checked_weights(sample_weight, summaries[0])
    batch, num_classes = summaries[0].shape
    rows = chunk_size or _resolve_chunk_size(
        num_classes, provenance["laplace_order"], provenance["hermite_order"]
    )
    if rows < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    total = torch.zeros((), dtype=torch.float64, device=summaries[0].device)
    for start in range(0, batch, rows):
        stop = start + rows
        probabilities, _, _ = remax_scale_moments(
            summaries[0][start:stop],
            summaries[1][start:stop],
            summaries[2][start:stop],
            summaries[3][start:stop],
            scale_mean=posterior.mean,
            scale_variance=posterior.variance,
            epsilon=provenance["epsilon"],
            kappa=provenance["kappa"],
            laplace_order=provenance["laplace_order"],
            hermite_order=provenance["hermite_order"],
            scale_order=provenance["scale_order"],
            map_scale=provenance["map_scale"],
            cross_moments=False,
        )
        picked = probabilities.gather(1, target[start:stop, None]).squeeze(1)
        total = total - (weights[start:stop] * picked.clamp_min(_LOG_FLOOR).log()).sum()
    return total / weights.sum().clamp_min(_LOG_FLOOR)
