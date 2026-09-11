"""Laplace-Remax moments under a random CDF variance head and a shared scale."""

from __future__ import annotations

import pytest
import torch

from triton_tagi.cdf_remax import (
    DEFAULT_CHUNK_ELEMENTS,
    _resolve_chunk,
    remax_conditional_moments,
    remax_mixture_kernels,
    remax_scale_moments,
    remax_uncertainty_decomposition,
)
from triton_tagi.cdf_variance import cdf_variance_activation, cdf_variance_moments
from triton_tagi.layers.remax import laplace_remax
from triton_tagi.remax_kernels import laplace_rule

DOUBLE = torch.float64
EPSILON, KAPPA = 0.02, 1.5

# The state the accompanying note uses for its numerical verification.
NOTE_MU = torch.tensor([[0.8, -0.3, 0.15]], dtype=DOUBLE)
NOTE_E = torch.tensor([[0.18, 0.09, 0.25]], dtype=DOUBLE)
NOTE_NU = torch.tensor([[-0.7, 0.1, -0.3]], dtype=DOUBLE)
NOTE_R = torch.tensor([[0.4, 0.2, 0.3]], dtype=DOUBLE)

# Published in the note's verification table for this state at s = 1.
NOTE_PROBABILITIES = (0.577012, 0.153584, 0.269404)
NOTE_ZERO_PROBABILITY = 0.038829


def _moments(**overrides: object) -> tuple[torch.Tensor, torch.Tensor]:
    keywords: dict[str, object] = {"epsilon": EPSILON, "kappa": KAPPA}
    keywords.update(overrides)
    return remax_conditional_moments(NOTE_MU, NOTE_E, NOTE_NU, NOTE_R, **keywords)  # type: ignore[arg-type]


def _sobol_reference(
    mu: torch.Tensor,
    var: torch.Tensor,
    nu: torch.Tensor,
    r: torch.Tensor,
    *,
    log_scale: float = 0.0,
    count: int = 1 << 19,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Deterministic quasi-Monte Carlo reference for the full hierarchy."""

    classes = mu.numel()
    engine = torch.quasirandom.SobolEngine(3 * classes, scramble=False)
    uniform = engine.draw(count).to(DOUBLE).clamp_(1e-12, 1.0 - 1e-12)
    normal = torch.special.ndtri(uniform)

    head = nu.view(-1) + r.view(-1).sqrt() * normal[:, :classes]
    aleatoric = cdf_variance_activation(head, epsilon=EPSILON, kappa=KAPPA)
    epistemic = var.view(-1).sqrt() * normal[:, classes : 2 * classes]
    noise = aleatoric.sqrt() * normal[:, 2 * classes :]

    logits = mu.view(-1) + torch.exp(torch.tensor(log_scale, dtype=DOUBLE)) * (epistemic + noise)
    rectified = logits.clamp_min(0.0)
    total = rectified.sum(dim=1, keepdim=True)
    probabilities = torch.where(
        total > 0.0, rectified / total.clamp_min(1e-300), torch.full_like(rectified, 1.0 / classes)
    )
    atom = float((total.squeeze(1) == 0.0).to(DOUBLE).mean())
    return probabilities.mean(dim=0), probabilities.pow(2).mean(dim=0), atom


# ──────────────────────────────────────────────────────────────────────────────
#  Published values and simplex structure
# ──────────────────────────────────────────────────────────────────────────────


def test_reproduces_the_published_predictive_probabilities() -> None:
    first, _ = _moments()
    for index, expected in enumerate(NOTE_PROBABILITIES):
        assert float(first[0, index]) == pytest.approx(expected, abs=5e-7)


def test_reproduces_the_published_all_zero_probability() -> None:
    t, _ = laplace_rule(160, reference=NOTE_MU)
    *_, zero_probability = remax_mixture_kernels(
        NOTE_MU, NOTE_E, NOTE_NU, NOTE_R, t, epsilon=EPSILON, kappa=KAPPA
    )
    assert float(zero_probability.prod()) == pytest.approx(NOTE_ZERO_PROBABILITY, abs=5e-7)


def test_moments_satisfy_the_simplex_identities() -> None:
    """``sum p = 1``, ``Sigma 1 = 0``, ``Sigma >= 0``, ``Sigma_ii <= p (1 - p)``."""

    first, second = _moments(cross_moments=True)
    covariance = second - first.unsqueeze(-1) * first.unsqueeze(-2)

    assert float(first.sum()) == pytest.approx(1.0, abs=1e-11)
    assert float(covariance.sum(dim=-1).abs().max()) < 1e-9
    assert float(torch.linalg.eigvalsh(covariance).min()) > -1e-9
    diagonal = torch.diagonal(covariance, dim1=1, dim2=2)
    assert bool((diagonal <= first * (1.0 - first) + 1e-12).all())
    assert bool((diagonal >= -1e-12).all())


def test_the_diagonal_agrees_with_the_full_cross_moment_block() -> None:
    _, diagonal = _moments(cross_moments=False)
    _, block = _moments(cross_moments=True)
    torch.testing.assert_close(
        diagonal, torch.diagonal(block, dim1=1, dim2=2), atol=1e-12, rtol=0.0
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Agreement with the existing fixed-noise path
# ──────────────────────────────────────────────────────────────────────────────


def test_a_degenerate_head_reduces_to_the_plain_laplace_remax_mean() -> None:
    """With ``r = 0`` the noise is fixed at ``h(nu)`` and the mixture collapses."""

    zero = torch.zeros_like(NOTE_R)
    h_mean, _, _ = cdf_variance_moments(NOTE_NU, zero, epsilon=EPSILON, kappa=KAPPA)
    first, _ = remax_conditional_moments(
        NOTE_MU, NOTE_E, NOTE_NU, zero, epsilon=EPSILON, kappa=KAPPA
    )
    reference, _, _ = laplace_remax(
        NOTE_MU, NOTE_E + h_mean, num_quad=160, full_jacobian=False, cap_jacobian=False
    )
    torch.testing.assert_close(first, reference, atol=1e-11, rtol=0.0)


def test_matches_a_quasi_monte_carlo_reference() -> None:
    first, second = _moments(cross_moments=False)
    ref_first, ref_second, _ = _sobol_reference(NOTE_MU, NOTE_E, NOTE_NU, NOTE_R)
    torch.testing.assert_close(first[0], ref_first, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(second[0], ref_second, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("log_scale", [-0.7, 0.4])
def test_matches_quasi_monte_carlo_away_from_unit_scale(log_scale: float) -> None:
    first, _ = _moments(log_scale=log_scale)
    ref_first, _, _ = _sobol_reference(NOTE_MU, NOTE_E, NOTE_NU, NOTE_R, log_scale=log_scale)
    torch.testing.assert_close(first[0], ref_first, atol=2e-3, rtol=2e-3)


# ──────────────────────────────────────────────────────────────────────────────
#  The invariance that dictates where the scale may act
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("gain", [0.25, 0.5, 2.0, 4.0])
def test_a_common_whole_logit_gain_leaves_remax_unchanged(gain: float) -> None:
    """``Remax(g x) = Remax(x)``: scaling mean, epistemic and aleatoric alike."""

    baseline, _ = _moments()
    scaled, _ = remax_conditional_moments(
        NOTE_MU * gain,
        NOTE_E * gain * gain,
        NOTE_NU,
        NOTE_R,
        epsilon=EPSILON * gain * gain,
        kappa=KAPPA * gain * gain,
    )
    torch.testing.assert_close(scaled, baseline, atol=1e-11, rtol=0.0)


def test_the_scale_is_unidentified_when_every_logit_mean_vanishes() -> None:
    """With ``mu = 0`` the model reduces to the invariant case."""

    zero_mean = torch.zeros_like(NOTE_MU)
    baseline, _ = remax_conditional_moments(
        zero_mean, NOTE_E, NOTE_NU, NOTE_R, epsilon=EPSILON, kappa=KAPPA
    )
    for log_scale in (-1.5, 0.8):
        moved, _ = remax_conditional_moments(
            zero_mean,
            NOTE_E,
            NOTE_NU,
            NOTE_R,
            epsilon=EPSILON,
            kappa=KAPPA,
            log_scale=log_scale,
        )
        torch.testing.assert_close(moved, baseline, atol=1e-11, rtol=0.0)


def test_the_scale_moves_probability_away_from_the_leading_class() -> None:
    previous = 1.0
    for log_scale in (-1.0, -0.5, 0.0, 0.5, 1.0):
        first, _ = _moments(log_scale=log_scale)
        leading = float(first[0, 0])
        assert leading < previous
        previous = leading


# ──────────────────────────────────────────────────────────────────────────────
#  Quadrature behaviour
# ──────────────────────────────────────────────────────────────────────────────


def test_moments_are_stable_under_quadrature_refinement() -> None:
    """Refining must not degrade the answer, which the naive kernels do."""

    baseline, _ = _moments(laplace_order=160, hermite_order=32)
    for laplace_order, hermite_order in ((256, 48), (320, 64), (640, 32)):
        refined, _ = _moments(laplace_order=laplace_order, hermite_order=hermite_order)
        assert float((refined - baseline).abs().max()) < 1e-9


def test_the_covariance_row_sum_survives_refinement() -> None:
    for laplace_order in (160, 320, 640):
        first, second = _moments(laplace_order=laplace_order, cross_moments=True)
        covariance = second - first.unsqueeze(-1) * first.unsqueeze(-2)
        assert float(covariance.sum(dim=-1).abs().max()) < 1e-9


@pytest.mark.parametrize("map_scale", [0.5, 1.0, 2.0])
def test_moments_are_insensitive_to_the_laplace_map_scale(map_scale: float) -> None:
    baseline, _ = _moments(map_scale=1.0)
    moved, _ = _moments(map_scale=map_scale)
    torch.testing.assert_close(moved, baseline, atol=1e-9, rtol=0.0)


def test_a_per_row_scale_matches_looping_over_scalars() -> None:
    values = [-0.6, 0.0, 0.9]
    batched_mu = NOTE_MU.repeat(3, 1)
    stacked, _ = remax_conditional_moments(
        batched_mu,
        NOTE_E.repeat(3, 1),
        NOTE_NU.repeat(3, 1),
        NOTE_R.repeat(3, 1),
        epsilon=EPSILON,
        kappa=KAPPA,
        log_scale=torch.tensor(values, dtype=DOUBLE),
    )
    for index, value in enumerate(values):
        single, _ = _moments(log_scale=value)
        torch.testing.assert_close(stacked[index], single[0], atol=1e-12, rtol=0.0)


def test_extreme_states_stay_on_the_simplex() -> None:
    generator = torch.Generator().manual_seed(17)
    mu = 6.0 * torch.randn(24, 5, generator=generator, dtype=DOUBLE)
    var = 1e-6 + 20.0 * torch.rand(24, 5, generator=generator, dtype=DOUBLE)
    nu = 8.0 * torch.randn(24, 5, generator=generator, dtype=DOUBLE)
    r = 30.0 * torch.rand(24, 5, generator=generator, dtype=DOUBLE)

    first, second = remax_conditional_moments(
        mu, var, nu, r, epsilon=EPSILON, kappa=KAPPA, cross_moments=True
    )
    assert bool(torch.isfinite(first).all()) and bool(torch.isfinite(second).all())
    assert bool((first >= -1e-12).all()) and bool((first <= 1.0 + 1e-12).all())
    torch.testing.assert_close(first.sum(dim=-1), torch.ones(24, dtype=DOUBLE), atol=1e-9, rtol=0.0)


# ──────────────────────────────────────────────────────────────────────────────
#  Integrating the shared scale
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("scale_mean", [-0.5, 0.0, 0.7])
def test_a_deterministic_scale_reproduces_the_conditional_moments(scale_mean: float) -> None:
    probabilities, _, cov_scale = remax_scale_moments(
        NOTE_MU,
        NOTE_E,
        NOTE_NU,
        NOTE_R,
        scale_mean=scale_mean,
        scale_variance=0.0,
        epsilon=EPSILON,
        kappa=KAPPA,
    )
    conditional, _ = _moments(log_scale=scale_mean)
    torch.testing.assert_close(probabilities, conditional, atol=1e-14, rtol=0.0)
    assert float(cov_scale.abs().max()) == 0.0


@pytest.mark.parametrize("scale_variance", [0.04, 0.25, 1.0])
def test_the_scale_covariance_respects_cauchy_schwarz(scale_variance: float) -> None:
    """``Cov(L, A_i)^2 <= q p_i (1 - p_i)`` for exact moments."""

    probabilities, _, cov_scale = remax_scale_moments(
        NOTE_MU,
        NOTE_E,
        NOTE_NU,
        NOTE_R,
        scale_mean=0.1,
        scale_variance=scale_variance,
        epsilon=EPSILON,
        kappa=KAPPA,
    )
    bound = scale_variance * probabilities * (1.0 - probabilities)
    assert bool(torch.isfinite(cov_scale).all())
    assert bool((cov_scale * cov_scale <= bound + 1e-15).all())
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("scale_order", [12, 20, 32])
def test_the_scale_integration_is_order_stable(scale_order: int) -> None:
    reference, _, ref_cov = remax_scale_moments(
        NOTE_MU,
        NOTE_E,
        NOTE_NU,
        NOTE_R,
        scale_mean=0.1,
        scale_variance=0.04,
        epsilon=EPSILON,
        kappa=KAPPA,
        scale_order=48,
    )
    value, _, cov = remax_scale_moments(
        NOTE_MU,
        NOTE_E,
        NOTE_NU,
        NOTE_R,
        scale_mean=0.1,
        scale_variance=0.04,
        epsilon=EPSILON,
        kappa=KAPPA,
        scale_order=scale_order,
    )
    assert torch.allclose(value, reference, atol=1e-12)
    assert torch.allclose(cov, ref_cov, atol=1e-12)


def test_a_very_broad_scale_prior_needs_more_laplace_nodes() -> None:
    """The default order is calibrated for the note's ``q_0 = 0.5^2`` prior.

    A prior variance of four puts five-sigma scale nodes at ``l = +/- 9``,
    where the rectified sum spans four orders of magnitude and the default
    160-node rule no longer resolves the Laplace integrand. This is a
    quadrature limit rather than a defect in the identities: refining restores
    the normalisation, which is exactly why the orders have to be rechecked
    whenever the parameter range changes.
    """

    arguments = {
        "scale_mean": 0.1,
        "scale_variance": 4.0,
        "epsilon": EPSILON,
        "kappa": KAPPA,
    }
    coarse, _, _ = remax_scale_moments(
        NOTE_MU, NOTE_E, NOTE_NU, NOTE_R, laplace_order=160, **arguments
    )
    refined, _, cov_scale = remax_scale_moments(
        NOTE_MU, NOTE_E, NOTE_NU, NOTE_R, laplace_order=640, **arguments
    )
    assert abs(float(coarse.sum()) - 1.0) > 1e-8
    assert float(refined.sum()) == pytest.approx(1.0, abs=1e-9)

    bound = 4.0 * refined * (1.0 - refined)
    assert bool((cov_scale * cov_scale <= bound + 1e-15).all())


def test_scale_moments_reject_a_negative_variance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        remax_scale_moments(
            NOTE_MU,
            NOTE_E,
            NOTE_NU,
            NOTE_R,
            scale_mean=0.0,
            scale_variance=-1.0,
            epsilon=EPSILON,
            kappa=KAPPA,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Epistemic and aleatoric separation
# ──────────────────────────────────────────────────────────────────────────────


def test_the_decomposition_sums_to_the_class_indicator_covariance() -> None:
    """``Sigma_epi + Sigma_ale = diag(p) - p p^T`` exactly, up to QMC error."""

    cov_epistemic, cov_aleatoric = remax_uncertainty_decomposition(
        NOTE_MU,
        NOTE_E,
        NOTE_NU,
        NOTE_R,
        scale_mean=0.1,
        scale_variance=0.04,
        epsilon=EPSILON,
        kappa=KAPPA,
        num_samples=4096,
        laplace_order=80,
    )
    probabilities, _, _ = remax_scale_moments(
        NOTE_MU,
        NOTE_E,
        NOTE_NU,
        NOTE_R,
        scale_mean=0.1,
        scale_variance=0.04,
        epsilon=EPSILON,
        kappa=KAPPA,
    )
    total = torch.diag_embed(probabilities) - probabilities.unsqueeze(-1) * probabilities.unsqueeze(
        -2
    )
    assert float((cov_epistemic + cov_aleatoric - total).abs().max()) < 5e-4


def test_both_decomposition_blocks_are_valid_covariances() -> None:
    cov_epistemic, cov_aleatoric = remax_uncertainty_decomposition(
        NOTE_MU,
        NOTE_E,
        NOTE_NU,
        NOTE_R,
        scale_mean=0.0,
        scale_variance=0.04,
        epsilon=EPSILON,
        kappa=KAPPA,
        num_samples=1024,
        laplace_order=80,
    )
    for block in (cov_epistemic, cov_aleatoric):
        assert float(torch.linalg.eigvalsh(block).min()) > -1e-12
        assert float(block.sum(dim=-1).abs().max()) < 1e-10


def test_fixing_every_epistemic_state_zeroes_only_the_epistemic_block() -> None:
    """The note's point: ``Var(A)`` stays positive while ``Sigma_epi`` vanishes."""

    zero = torch.zeros_like(NOTE_E)
    cov_epistemic, cov_aleatoric = remax_uncertainty_decomposition(
        NOTE_MU,
        zero,
        NOTE_NU,
        zero,
        scale_mean=0.0,
        scale_variance=0.0,
        epsilon=EPSILON,
        kappa=KAPPA,
        num_samples=256,
        laplace_order=80,
    )
    assert float(cov_epistemic.abs().max()) < 1e-14
    assert float(torch.diagonal(cov_aleatoric, dim1=1, dim2=2).sum()) > 0.1


def test_decomposition_rejects_a_degenerate_sample_count() -> None:
    with pytest.raises(ValueError, match="at least two"):
        remax_uncertainty_decomposition(
            NOTE_MU,
            NOTE_E,
            NOTE_NU,
            NOTE_R,
            scale_mean=0.0,
            scale_variance=0.0,
            epsilon=EPSILON,
            kappa=KAPPA,
            num_samples=1,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Batch chunking
# ──────────────────────────────────────────────────────────────────────────────


def test_the_derived_chunk_holds_the_working_set_near_its_budget() -> None:
    """Peak memory is ``(chunk, K, J, Q)``, which must not grow with ``K``.

    Without this the allocation scales with the class count as well as the
    batch, and a hundred-class batch of a couple of thousand rows asks for
    several gigabytes per temporary.
    """

    for num_classes in (10, 100, 1000):
        for laplace_order, hermite_order in ((160, 32), (80, 16)):
            chunk = _resolve_chunk(num_classes, laplace_order, hermite_order, None)
            assert chunk >= 1
            working_set = chunk * num_classes * laplace_order * hermite_order
            assert working_set <= DEFAULT_CHUNK_ELEMENTS


def test_an_explicit_chunk_size_is_honoured_and_validated() -> None:
    assert _resolve_chunk(100, 160, 32, 512) == 512
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        _resolve_chunk(10, 160, 32, 0)


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 16, 64])
def test_chunking_is_bit_exact_for_the_diagonal(chunk_size: int) -> None:
    generator = torch.Generator().manual_seed(41)
    rows = 16
    mu = torch.randn(rows, 4, generator=generator, dtype=DOUBLE)
    var = 0.05 + torch.rand(rows, 4, generator=generator, dtype=DOUBLE)
    nu = torch.randn(rows, 4, generator=generator, dtype=DOUBLE)
    r = 0.05 + torch.rand(rows, 4, generator=generator, dtype=DOUBLE)

    reference = remax_conditional_moments(
        mu, var, nu, r, epsilon=EPSILON, kappa=KAPPA, chunk_size=rows
    )
    chunked = remax_conditional_moments(
        mu, var, nu, r, epsilon=EPSILON, kappa=KAPPA, chunk_size=chunk_size
    )
    for value, expected in zip(chunked, reference, strict=True):
        assert float((value - expected).abs().max()) == 0.0


def test_chunking_is_bit_exact_for_the_cross_moment_block() -> None:
    generator = torch.Generator().manual_seed(42)
    rows = 13
    mu = torch.randn(rows, 5, generator=generator, dtype=DOUBLE)
    var = 0.05 + torch.rand(rows, 5, generator=generator, dtype=DOUBLE)
    nu = torch.randn(rows, 5, generator=generator, dtype=DOUBLE)
    r = 0.05 + torch.rand(rows, 5, generator=generator, dtype=DOUBLE)

    keywords = {"epsilon": EPSILON, "kappa": KAPPA, "cross_moments": True}
    reference = remax_conditional_moments(mu, var, nu, r, chunk_size=rows, **keywords)
    chunked = remax_conditional_moments(mu, var, nu, r, chunk_size=4, **keywords)
    for value, expected in zip(chunked, reference, strict=True):
        assert float((value - expected).abs().max()) == 0.0


def test_a_per_row_scale_is_sliced_with_its_own_rows_when_chunked() -> None:
    """A chunk must carry the scale values belonging to its rows, not row zero."""

    generator = torch.Generator().manual_seed(43)
    rows = 11
    mu = torch.randn(rows, 3, generator=generator, dtype=DOUBLE)
    var = 0.05 + torch.rand(rows, 3, generator=generator, dtype=DOUBLE)
    nu = torch.randn(rows, 3, generator=generator, dtype=DOUBLE)
    r = 0.05 + torch.rand(rows, 3, generator=generator, dtype=DOUBLE)
    log_scale = torch.linspace(-1.2, 1.2, rows, dtype=DOUBLE)

    keywords = {"epsilon": EPSILON, "kappa": KAPPA, "log_scale": log_scale}
    reference, _ = remax_conditional_moments(mu, var, nu, r, chunk_size=rows, **keywords)
    for chunk_size in (1, 2, 5):
        chunked, _ = remax_conditional_moments(mu, var, nu, r, chunk_size=chunk_size, **keywords)
        assert float((chunked - reference).abs().max()) == 0.0

    # And each row really does use its own scale, so the rows differ.
    assert float(reference.diff(dim=0).abs().max()) > 1e-6


def test_chunking_leaves_the_scale_integration_unchanged() -> None:
    generator = torch.Generator().manual_seed(44)
    rows = 9
    mu = torch.randn(rows, 4, generator=generator, dtype=DOUBLE)
    var = 0.05 + torch.rand(rows, 4, generator=generator, dtype=DOUBLE)
    nu = torch.randn(rows, 4, generator=generator, dtype=DOUBLE)
    r = 0.05 + torch.rand(rows, 4, generator=generator, dtype=DOUBLE)

    keywords = {
        "scale_mean": 0.1,
        "scale_variance": 0.04,
        "epsilon": EPSILON,
        "kappa": KAPPA,
    }
    reference = remax_scale_moments(mu, var, nu, r, chunk_size=rows, **keywords)
    chunked = remax_scale_moments(mu, var, nu, r, chunk_size=2, **keywords)
    for value, expected in zip(chunked, reference, strict=True):
        assert float((value - expected).abs().max()) == 0.0
