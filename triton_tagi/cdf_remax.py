"""Laplace-Remax moments under a random CDF variance head and a shared scale.

This module carries the predictive side of the TAGI-V/Remax calibration model.
A frozen network supplies, per input, the prediction head ``Z_i ~ N(mu_i, e_i)``
and the variance head ``U_i ~ N(nu_i, r_i)``; the aleatoric logit noise is
``V_i | U_i ~ N(0, h(U_i))`` with ``h`` the bounded CDF activation of
:mod:`triton_tagi.cdf_variance`. Calibration acts through one shared Gaussian
log-scale ``L ~ N(lambda, q)``::

    X_{i,l} = mu_i + e^l [(Z_i - mu_i) + V_i],   p = E[Remax(X_L)]

so ``s = e^l`` multiplies the combined epistemic and aleatoric *deviation*
while the predictive logit mean ``mu_i`` is held fixed.

**Why the scale sits there.** Remax is positively homogeneous, ``Remax(g x) =
Remax(x)`` for every ``g > 0`` -- including the all-zero event and a random
``g`` -- so an ordinary global logit gain, and therefore ordinary temperature
scaling, is exactly a no-op. Equivalently ``Remax(X_l) = Remax(e^{-l} mu +
(Z - mu) + V)``: the learnable quantity is the mean *relative to* its
uncertainty. The parameterisation ``s = e^L`` also keeps every posterior draw
positive, which clipping the mean of an unrestricted Gaussian gain would not.

**Two orderings that are not interchangeable.** Conditional on ``l``, the
classes are independent after each ``U_i`` is integrated out, so the Laplace
product over classes is valid. Integrating a *shared* ``L`` couples them even
when the logit covariances vanish. The class products must therefore be formed
conditionally on ``l`` and only then averaged over ``L`` -- never the reverse.
For the same reason each per-class kernel is averaged over its own ``U_i``
*before* the product across classes is taken, which is more faithful than
collapsing the head to ``h_i`` and treating the logit as exactly Gaussian, and
reduces a ``K``-dimensional integral to ``K`` scalar ones.

With ``d_i(u, l) = e^l sqrt(e_i + h(u))`` and the kernels of
:mod:`triton_tagi.remax_kernels`,

    ``L_i(t|l) = E_{U_i}[L(t; mu_i, d_i(U_i, l))]``  (and likewise D_i, F_i)
    ``pi_{0,i}(l) = E_{U_i}[Phi(-mu_i / d_i(U_i, l))]``
    ``P(t|l) = prod_j L_j(t|l)``,   ``pi_0(l) = prod_j pi_{0,j}(l)``

the Laplace identities give the exact conditional moments

    ``m_i(l)  = int_0^inf D_i P / L_i dt + pi_0(l) / K``
    ``B_ii(l) = int_0^inf t F_i P / L_i dt + pi_0(l) / K^2``
    ``B_ij(l) = int_0^inf t D_i D_j P / (L_i L_j) dt + pi_0(l) / K^2``

exact under the factorised output approximation and the hierarchical noise
model; only the quadrature is approximate. The ``pi_0 / K`` terms are the
uniform all-zero convention and are required even though ``E[[X_i]_+] > 0`` --
a positive expectation does not remove the atom at zero. Products divided by a
selected factor are written as stable log-differences in the code, never as a
literal ratio.

Costs follow the note: means and diagonal second moments are ``O(H Q J K)``
with ``H`` scale nodes, ``Q`` Laplace nodes and ``J`` variance-head nodes; the
full cross-moment block adds ``O(H Q K^2)`` and is therefore opt-in through
``cross_moments``.

The default orders hold ``sum_i p_i = 1`` to about ``1e-12`` for ``l`` in
``[-9, 6]``, which covers the note's proposed ``N(0, 0.5^2)`` scale prior many
times over. Beyond that the rectified sum spans so many orders of magnitude
that the Laplace rule under-resolves its own integrand -- at ``l = 9`` the
normalisation degrades to ``1e-2`` at 160 nodes and returns to ``3e-11`` at
640 -- so the orders must be rechecked whenever the parameter range widens,
and the normalisation is the cheapest thing to check.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .cdf_variance import cdf_variance_activation
from .remax_kernels import (
    DEFAULT_LAPLACE_ORDER,
    hermite_rule,
    laplace_rule,
    remax_kernels,
)

# Variance-head nodes. The integrand is a smooth bounded function of a
# standard normal, so a modest order converges to double precision.
DEFAULT_HERMITE_ORDER: int = 32

# Scale nodes for integrating L. The calibration posterior is scalar and
# near-Gaussian, so twenty nodes already agree with thirty-two to 1e-16.
DEFAULT_SCALE_ORDER: int = 20

# Peak working set is (batch, K, J, Q) float64 elements, materialised before
# the head average collapses the J axis. At K=100, J=32, Q=160 a single row
# costs 512k elements, so an unchunked batch of 2048 would ask for 8 GB per
# temporary. The batch is therefore split to hold each temporary near 128 MB.
DEFAULT_CHUNK_ELEMENTS: int = 16_000_000

_LOG_FLOOR: float = 1e-300


def _resolve_chunk(
    num_classes: int,
    laplace_order: int,
    hermite_order: int,
    chunk_size: int | None,
) -> int:
    """Return the batch chunk height that keeps one temporary near the budget."""

    if chunk_size is not None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        return chunk_size
    per_row = max(1, num_classes * laplace_order * hermite_order)
    return max(1, DEFAULT_CHUNK_ELEMENTS // per_row)


def remax_mixture_kernels(
    mu_z: Tensor,
    var_z: Tensor,
    nu: Tensor,
    r: Tensor,
    t: Tensor,
    *,
    epsilon: float,
    kappa: float,
    log_scale: float | Tensor = 0.0,
    hermite_order: int = DEFAULT_HERMITE_ORDER,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return the variance-mixture Laplace kernels at one log-scale.

    Each kernel is averaged over its own variance head before any product
    across classes is formed, which is the ordering the model requires.

    This is the unchunked primitive: its working set is ``(batch, K, J, Q)``
    before the head average collapses the ``J`` axis, so a caller with many
    rows or many classes must split the batch itself.
    :func:`remax_conditional_moments` does that already, and
    :func:`remax_zero_probability` avoids the Laplace axis entirely when only
    the all-zero atom is wanted.

    Args:
        mu_z: Prediction-head means ``mu``, shape ``(batch, K)``.
        var_z: Prediction-head variances ``e``, same shape.
        nu: Variance-head means, same shape.
        r: Variance-head variances, same shape.
        t: Laplace variable, shape ``(Q,)``.
        epsilon: CDF-head variance floor.
        kappa: CDF-head variance range.
        log_scale: Shared log-scale ``l``; a scalar, or one value per
            row of ``mu_z``.
        hermite_order: Gauss--Hermite order ``J`` over each head.

    Returns:
        kernel_l: ``L_i(t|l)``, shape ``(batch, K, Q)``.
        kernel_d: ``D_i(t|l)``, same shape.
        kernel_f: ``F_i(t|l)``, same shape.
        zero_probability: ``pi_{0,i}(l)``, shape ``(batch, K)``.
    """

    nodes, weights = hermite_rule(hermite_order, reference=mu_z)
    scale = torch.as_tensor(log_scale, dtype=torch.float64, device=mu_z.device).exp()
    if scale.dim() > 0:
        # A per-row scale becomes (batch, 1, 1) so it broadcasts over class and
        # head axes; a scalar broadcasts as it stands.
        scale = scale.reshape(-1, 1, 1)

    # (batch, K, 1) against (J,) so each class integrates its own head.
    head = nu.double().unsqueeze(-1) + r.double().clamp_min(0.0).sqrt().unsqueeze(-1) * nodes
    aleatoric = cdf_variance_activation(head, epsilon=epsilon, kappa=kappa)
    deviation = scale * (var_z.double().unsqueeze(-1) + aleatoric).sqrt()

    mean = mu_z.double().unsqueeze(-1).unsqueeze(-1)
    kernel_l, kernel_d, kernel_f = remax_kernels(
        t.double(), mean, deviation.unsqueeze(-1), highest=2
    )

    head_weights = weights.view(-1, 1)
    kernel_l = (kernel_l * head_weights).sum(dim=-2)
    kernel_d = (kernel_d * head_weights).sum(dim=-2)
    kernel_f = (kernel_f * head_weights).sum(dim=-2)

    alpha = mu_z.double().unsqueeze(-1) / deviation
    zero_probability = (torch.special.ndtr(-alpha) * weights).sum(dim=-1)
    return kernel_l, kernel_d, kernel_f, zero_probability


def remax_zero_probability(
    mu_z: Tensor,
    var_z: Tensor,
    nu: Tensor,
    r: Tensor,
    *,
    epsilon: float,
    kappa: float,
    log_scale: float | Tensor = 0.0,
    hermite_order: int = DEFAULT_HERMITE_ORDER,
) -> Tensor:
    """Return ``pi_{0,i}(l) = E_{U_i}[Phi(-mu_i / d_i(U_i, l))]`` per class.

    The atom where every rectified logit vanishes, and Remax falls back to the
    uniform vector, depends on no Laplace variable. Computing it through
    :func:`remax_mixture_kernels` would build the whole ``(batch, K, J, Q)``
    kernel block and discard it, so this evaluates the ``(batch, K, J)`` part
    alone -- cheaper by the Laplace order, which is what makes it usable as a
    diagnostic on a large class count.

    Args:
        mu_z: Prediction-head means, shape ``(batch, K)``.
        var_z: Prediction-head variances, same shape.
        nu: Variance-head means, same shape.
        r: Variance-head variances, same shape.
        epsilon: CDF-head variance floor.
        kappa: CDF-head variance range.
        log_scale: Shared log-scale ``l``; a scalar, or one value per row.
        hermite_order: Gauss--Hermite order over each variance head.

    Returns:
        Per-class zero probabilities, shape ``(batch, K)``. Their product over
        classes is the all-zero probability of a row.
    """

    nodes, weights = hermite_rule(hermite_order, reference=mu_z)
    scale = torch.as_tensor(log_scale, dtype=torch.float64, device=mu_z.device).exp()
    if scale.dim() > 0:
        scale = scale.reshape(-1, 1, 1)

    head = nu.double().unsqueeze(-1) + r.double().clamp_min(0.0).sqrt().unsqueeze(-1) * nodes
    aleatoric = cdf_variance_activation(head, epsilon=epsilon, kappa=kappa)
    deviation = scale * (var_z.double().unsqueeze(-1) + aleatoric).sqrt()
    alpha = mu_z.double().unsqueeze(-1) / deviation
    return (torch.special.ndtr(-alpha) * weights).sum(dim=-1)


def remax_conditional_moments(
    mu_z: Tensor,
    var_z: Tensor,
    nu: Tensor,
    r: Tensor,
    *,
    epsilon: float,
    kappa: float,
    log_scale: float | Tensor = 0.0,
    laplace_order: int = DEFAULT_LAPLACE_ORDER,
    hermite_order: int = DEFAULT_HERMITE_ORDER,
    map_scale: float = 1.0,
    cross_moments: bool = False,
    chunk_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the Remax probability moments ``m_i(l)`` and ``B(l)`` at one scale.

    Rows are independent given the scale, so the batch is split into chunks
    that keep the ``(chunk, K, J, Q)`` working set near a fixed budget. Without
    that split the peak allocation grows with both the batch and the class
    count, and a hundred-class batch of a couple of thousand rows asks for
    several gigabytes per temporary.

    Args:
        mu_z: Prediction-head means, shape ``(batch, K)``.
        var_z: Prediction-head variances, same shape.
        nu: Variance-head means, same shape.
        r: Variance-head variances, same shape.
        epsilon: CDF-head variance floor.
        kappa: CDF-head variance range.
        log_scale: Shared log-scale ``l``; a scalar, or one value per
            row of ``mu_z``.
        laplace_order: Gauss--Legendre order ``Q`` over the Laplace variable.
        hermite_order: Gauss--Hermite order ``J`` over each variance head.
        map_scale: Scale ``a`` of the map ``t = y / [a (1 - y)]``.
        cross_moments: When true return the full ``(batch, K, K)`` second
            moment block; otherwise only its diagonal, shape ``(batch, K)``.
        chunk_size: Rows evaluated at once. ``None`` derives it from
            :data:`DEFAULT_CHUNK_ELEMENTS` and the quadrature orders.

    Returns:
        first: ``m_i(l) = E[A_i | L = l]``, shape ``(batch, K)``.
        second: ``B_ij(l) = E[A_i A_j | L = l]``, shape ``(batch, K, K)`` when
            ``cross_moments`` else its diagonal ``(batch, K)``.
    """

    num_classes = mu_z.shape[-1]
    step = _resolve_chunk(num_classes, laplace_order, hermite_order, chunk_size)
    if mu_z.shape[0] > step:
        scale_rows = None
        if isinstance(log_scale, Tensor) and log_scale.dim() > 0:
            scale_rows = log_scale.reshape(-1)
        firsts, seconds = [], []
        for start in range(0, mu_z.shape[0], step):
            stop = start + step
            block_scale = log_scale if scale_rows is None else scale_rows[start:stop]
            block_first, block_second = remax_conditional_moments(
                mu_z[start:stop],
                var_z[start:stop],
                nu[start:stop],
                r[start:stop],
                epsilon=epsilon,
                kappa=kappa,
                log_scale=block_scale,
                laplace_order=laplace_order,
                hermite_order=hermite_order,
                map_scale=map_scale,
                cross_moments=cross_moments,
                chunk_size=step,
            )
            firsts.append(block_first)
            seconds.append(block_second)
        return torch.cat(firsts), torch.cat(seconds)

    t, quad_weights = laplace_rule(laplace_order, map_scale=map_scale, reference=mu_z)

    kernel_l, kernel_d, kernel_f, zero_probability = remax_mixture_kernels(
        mu_z,
        var_z,
        nu,
        r,
        t,
        epsilon=epsilon,
        kappa=kappa,
        log_scale=log_scale,
        hermite_order=hermite_order,
    )

    # Stable log-product; P / L_i is a log-difference, never a literal ratio.
    log_kernel_l = kernel_l.clamp_min(_LOG_FLOOR).log()
    log_product = log_kernel_l.sum(dim=1, keepdim=True)
    product_except = (log_product - log_kernel_l).exp()

    log_zero = zero_probability.clamp_min(_LOG_FLOOR).log()
    atom = log_zero.sum(dim=-1).exp()

    first = (quad_weights * kernel_d * product_except).sum(dim=-1)
    first = first + (atom / num_classes).unsqueeze(-1)

    weighted = quad_weights * t
    if not cross_moments:
        second = (weighted * kernel_f * product_except).sum(dim=-1)
        second = second + (atom / (num_classes * num_classes)).unsqueeze(-1)
        return first, second

    ratio_d = kernel_d / kernel_l.clamp_min(_LOG_FLOOR)
    second = torch.einsum(
        "bq,biq,bjq->bij", weighted * torch.exp(log_product).squeeze(1), ratio_d, ratio_d
    )
    diagonal = (weighted * kernel_f * product_except).sum(dim=-1)
    index = torch.arange(num_classes, device=mu_z.device)
    second[:, index, index] = diagonal
    second = second + (atom / (num_classes * num_classes)).view(-1, 1, 1)
    return first, second


def remax_scale_moments(
    mu_z: Tensor,
    var_z: Tensor,
    nu: Tensor,
    r: Tensor,
    *,
    scale_mean: float,
    scale_variance: float,
    epsilon: float,
    kappa: float,
    laplace_order: int = DEFAULT_LAPLACE_ORDER,
    hermite_order: int = DEFAULT_HERMITE_ORDER,
    scale_order: int = DEFAULT_SCALE_ORDER,
    map_scale: float = 1.0,
    cross_moments: bool = False,
    chunk_size: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Integrate the shared log-scale to get the predictive Remax moments.

    Implements ``p = E_L[m(L)]``, ``Sigma_A = E_L[B(L)] - p p^T`` and the
    calibration cross-covariance ``c_i = Cov(L, A_i) = E_L[(L - lambda)
    m_i(L)]``. The covariance is evaluated by the same quadrature rather than
    by differentiating an approximate moment formula, which would return an
    approximate covariance rather than the covariance of the stated model.

    Args:
        mu_z: Prediction-head means, shape ``(batch, K)``.
        var_z: Prediction-head variances, same shape.
        nu: Variance-head means, same shape.
        r: Variance-head variances, same shape.
        scale_mean: Current ``lambda``.
        scale_variance: Current ``q``, non-negative.
        epsilon: CDF-head variance floor.
        kappa: CDF-head variance range.
        laplace_order: Gauss--Legendre order over the Laplace variable.
        hermite_order: Gauss--Hermite order over each variance head.
        scale_order: Gauss--Hermite order ``H`` over ``L``.
        map_scale: Scale of the Laplace map.
        cross_moments: Return the full ``Sigma_A`` block rather than its
            diagonal.
        chunk_size: Rows evaluated at once; ``None`` derives it from the
            quadrature orders.

    Returns:
        probabilities: ``p``, shape ``(batch, K)``.
        covariance: ``Sigma_A``, shape ``(batch, K, K)`` when ``cross_moments``
            else its diagonal ``(batch, K)``.
        cov_scale: ``Cov(L, A_i)``, shape ``(batch, K)``.
    """

    if scale_variance < 0.0:
        raise ValueError(f"scale_variance must be non-negative, got {scale_variance}")

    nodes, weights = hermite_rule(scale_order, reference=mu_z)
    deviation = math.sqrt(max(scale_variance, 0.0))

    probabilities = torch.zeros_like(mu_z, dtype=torch.float64)
    cov_scale = torch.zeros_like(probabilities)
    second_shape = probabilities.shape + (mu_z.shape[-1],) if cross_moments else probabilities.shape
    second = torch.zeros(second_shape, dtype=torch.float64, device=mu_z.device)

    for node, weight in zip(nodes.tolist(), weights.tolist(), strict=True):
        value = scale_mean + deviation * node
        first_l, second_l = remax_conditional_moments(
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
            cross_moments=cross_moments,
            chunk_size=chunk_size,
        )
        probabilities = probabilities + weight * first_l
        second = second + weight * second_l
        cov_scale = cov_scale + weight * (value - scale_mean) * first_l

    if cross_moments:
        covariance = second - probabilities.unsqueeze(-1) * probabilities.unsqueeze(-2)
    else:
        covariance = (second - probabilities * probabilities).clamp_min(0.0)
    return probabilities, covariance, cov_scale


def remax_uncertainty_decomposition(
    mu_z: Tensor,
    var_z: Tensor,
    nu: Tensor,
    r: Tensor,
    *,
    scale_mean: float,
    scale_variance: float,
    epsilon: float,
    kappa: float,
    num_samples: int = 512,
    laplace_order: int = DEFAULT_LAPLACE_ORDER,
    map_scale: float = 1.0,
    sample_chunk: int = 32,
    chunk_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Split the class-indicator covariance into epistemic and aleatoric parts.

    Let ``Omega = (Z, U, L)`` and ``pi(Omega) = E_xi[Remax(X) | Omega]``, the
    class distribution with the aleatoric innovations averaged out but every
    epistemic state held fixed. Then

        ``Sigma_epi = Cov_Omega(pi(Omega))``
        ``Sigma_ale = E_Omega[diag(pi) - pi pi^T]``
        ``Sigma_B   = diag(p) - p p^T = Sigma_epi + Sigma_ale``

    for the one-hot class indicator ``B``. The aleatoric term absorbs both the
    Gaussian logit perturbation and the residual categorical randomness.

    Two traps this function exists to avoid. ``Var(A)`` for ``A =
    Remax(X_L)`` is *not* the epistemic part -- it already contains noise
    realisations -- so adding it to ``Sigma_B`` double counts. And the logit
    ratio ``e_i / (e_i + h_i)`` is not a recipe for splitting probability
    variance after the nonlinear normalisation. With every epistemic state
    fixed ``Sigma_epi`` is zero even though ``Var(A)`` stays positive.

    ``E_Omega[pi pi^T]`` is not delivered by the single-pass Laplace moments,
    so this is genuine nested integration. The epistemic space has ``2 K + 1``
    dimensions, which rules out a product quadrature grid, so ``Omega`` is
    drawn by unscrambled Sobol quasi-Monte Carlo and the Laplace kernels supply
    the inner aleatoric average exactly at each draw. Cost is therefore well
    above the ``O(H Q J K)`` of the predictive moments, and the result carries
    QMC error rather than quadrature error.

    Args:
        mu_z: Prediction-head means, shape ``(batch, K)``.
        var_z: Prediction-head variances, same shape.
        nu: Variance-head means, same shape.
        r: Variance-head variances, same shape.
        scale_mean: Calibration ``lambda``.
        scale_variance: Calibration ``q``, non-negative.
        epsilon: CDF-head variance floor.
        kappa: CDF-head variance range.
        num_samples: Sobol draws over ``Omega``.
        laplace_order: Gauss--Legendre order over the Laplace variable.
        map_scale: Scale of the Laplace map.
        sample_chunk: Draws evaluated per inner call, to bound memory.
        chunk_size: Rows evaluated at once inside each inner call.

    Returns:
        cov_epistemic: ``Sigma_epi``, shape ``(batch, K, K)``.
        cov_aleatoric: ``Sigma_ale``, shape ``(batch, K, K)``.
    """

    if num_samples < 2:
        raise ValueError(f"num_samples must be at least two, got {num_samples}")
    if scale_variance < 0.0:
        raise ValueError(f"scale_variance must be non-negative, got {scale_variance}")

    batch, num_classes = mu_z.shape
    device = mu_z.device
    work_mu = mu_z.double()
    epistemic_deviation = var_z.double().clamp_min(0.0).sqrt()
    head_deviation = r.double().clamp_min(0.0).sqrt()
    scale_deviation = math.sqrt(max(scale_variance, 0.0))

    engine = torch.quasirandom.SobolEngine(2 * num_classes + 1, scramble=False)
    uniform = engine.draw(num_samples).double().clamp_(1e-12, 1.0 - 1e-12).to(device)
    normal = torch.special.ndtri(uniform)

    mean_pi = torch.zeros(batch, num_classes, dtype=torch.float64, device=device)
    outer_pi = torch.zeros(batch, num_classes, num_classes, dtype=torch.float64, device=device)
    degenerate = torch.zeros(1, num_classes, dtype=torch.float64, device=device)

    for start in range(0, num_samples, sample_chunk):
        block = normal[start : start + sample_chunk]
        rows = block.shape[0]
        z_draw = block[:, :num_classes].unsqueeze(1)
        u_draw = block[:, num_classes : 2 * num_classes].unsqueeze(1)
        l_draw = scale_mean + scale_deviation * block[:, -1].view(rows, 1, 1)

        realised_z = work_mu + epistemic_deviation * z_draw
        realised_u = nu.double() + head_deviation * u_draw
        # X = mu + e^l [(Z - mu) + V]: with Z fixed the mean is shifted by the
        # scaled epistemic deviation and only the aleatoric noise is left, so
        # the prediction variance is zero and the head is a point mass.
        shifted = work_mu + l_draw.exp() * (realised_z - work_mu)

        inner, _ = remax_conditional_moments(
            shifted.reshape(-1, num_classes),
            degenerate.expand(rows * batch, num_classes),
            realised_u.expand(rows, batch, num_classes).reshape(-1, num_classes),
            degenerate.expand(rows * batch, num_classes),
            epsilon=epsilon,
            kappa=kappa,
            log_scale=l_draw.expand(rows, batch, 1).reshape(-1),
            laplace_order=laplace_order,
            hermite_order=4,
            map_scale=map_scale,
            cross_moments=False,
            chunk_size=chunk_size,
        )
        # The head is degenerate here, so the four-node Hermite rule over U is
        # exact: every node lands on nu itself and the weights sum to one. The
        # shared scale is passed through so it still multiplies the aleatoric
        # deviation, the only randomness left once Omega is fixed.
        inner = inner.view(rows, batch, num_classes)
        mean_pi = mean_pi + inner.sum(dim=0)
        outer_pi = outer_pi + torch.einsum("sbi,sbj->bij", inner, inner)

    mean_pi = mean_pi / num_samples
    outer_pi = outer_pi / num_samples

    cov_epistemic = outer_pi - mean_pi.unsqueeze(-1) * mean_pi.unsqueeze(-2)
    cov_aleatoric = torch.diag_embed(mean_pi) - outer_pi
    return cov_epistemic, cov_aleatoric


def remax_forward_diagnostics(
    mu_z: Tensor,
    var_z: Tensor,
    nu: Tensor,
    r: Tensor,
    *,
    scale_mean: float,
    scale_variance: float,
    epsilon: float,
    kappa: float,
    laplace_order: int = DEFAULT_LAPLACE_ORDER,
    hermite_order: int = DEFAULT_HERMITE_ORDER,
    scale_order: int = DEFAULT_SCALE_ORDER,
    map_scale: float = 1.0,
    saturation_tolerance: float = 1e-3,
) -> dict[str, Tensor]:
    """Return the forward diagnostics the formulation asks to be recorded.

    Three quantities are easy to compute and easy to forget, and each one
    silently invalidates a calibration study when it goes wrong.

    The all-zero probability is the mass on the atom where every rectified
    logit vanishes and Remax falls back to the uniform vector. It is a real
    part of the model rather than a numerical artefact, and a large value means
    the reported probabilities are substantially a uniform mixture.

    Head saturation is the fraction of output units whose expected aleatoric
    variance sits within ``saturation_tolerance * kappa`` of either end of
    ``(epsilon, epsilon + kappa)``. A saturated head has stopped adapting, so
    the cap was chosen badly for the intended logit scale.

    The quadrature refinement error compares the requested orders against
    doubled ones. It is the cheapest available evidence that the orders still
    suit the parameter range, which has to be rechecked whenever that range
    moves rather than assumed from a previous fit. The normalisation error is
    reported alongside it for the same reason.

    Args:
        mu_z: Prediction-head means, shape ``(batch, K)``.
        var_z: Prediction-head variances, same shape.
        nu: Variance-head means, same shape.
        r: Variance-head variances, same shape.
        scale_mean: Calibration ``lambda``.
        scale_variance: Calibration ``q``, non-negative.
        epsilon: CDF-head variance floor.
        kappa: CDF-head variance range.
        laplace_order: Gauss--Legendre order over the Laplace variable.
        hermite_order: Gauss--Hermite order over each variance head.
        scale_order: Gauss--Hermite order over ``L``.
        map_scale: Scale of the Laplace map.
        saturation_tolerance: Fraction of ``kappa`` counted as saturated.

    Returns:
        A mapping with ``probabilities`` and ``all_zero_probability`` per row,
        and the scalars ``head_saturation_fraction``,
        ``quadrature_refinement_error`` and ``simplex_mean_error``.
    """

    from .cdf_variance import cdf_variance_moments

    nodes, weights = hermite_rule(scale_order, reference=mu_z)
    deviation = math.sqrt(max(scale_variance, 0.0))

    atom = torch.zeros(mu_z.shape[0], dtype=torch.float64, device=mu_z.device)
    for node, weight in zip(nodes.tolist(), weights.tolist(), strict=True):
        zero_probability = remax_zero_probability(
            mu_z,
            var_z,
            nu,
            r,
            epsilon=epsilon,
            kappa=kappa,
            log_scale=scale_mean + deviation * node,
            hermite_order=hermite_order,
        )
        atom = atom + weight * zero_probability.prod(dim=-1)

    common = {
        "scale_mean": scale_mean,
        "scale_variance": scale_variance,
        "epsilon": epsilon,
        "kappa": kappa,
        "map_scale": map_scale,
    }
    coarse, _, _ = remax_scale_moments(
        mu_z,
        var_z,
        nu,
        r,
        laplace_order=laplace_order,
        hermite_order=hermite_order,
        scale_order=scale_order,
        **common,
    )
    refined, _, _ = remax_scale_moments(
        mu_z,
        var_z,
        nu,
        r,
        laplace_order=2 * laplace_order,
        hermite_order=2 * hermite_order,
        scale_order=scale_order,
        **common,
    )

    h_mean, _, _ = cdf_variance_moments(nu, r, epsilon=epsilon, kappa=kappa)
    reach = saturation_tolerance * kappa
    saturated = (h_mean - epsilon <= reach) | (epsilon + kappa - h_mean <= reach)

    return {
        "probabilities": coarse,
        "all_zero_probability": atom,
        "head_saturation_fraction": saturated.to(torch.float64).mean(),
        "quadrature_refinement_error": (coarse - refined).abs().max(),
        "simplex_mean_error": (coarse.sum(dim=-1) - 1.0).abs().max(),
    }
