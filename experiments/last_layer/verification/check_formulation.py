"""Numerical verification of the CDF/TAGI-V/Remax joint-calibration formulation.

Referenced by ``CDF_TAGIV_Remax_joint_calibration.tex``, section "Numerical
verification of the formulation". Checks the variance-mixture kernels, the
exact CDF-head moments, the Gaussian training channel, the Remax scale
invariance, and the simplex covariance identities against Monte Carlo and
against independent dense integration.

Everything runs on the synthetic three-class state the note fixes, so the
output is directly comparable with its published table. Monte Carlo is a
verification reference for the deterministic equations, not a component of
them.

Run ``python -m experiments.last_layer.verification.check_formulation`` from
the repository root. ``--update-tex`` rewrites the table block inside the
``.tex`` between its ``% BEGIN NUMERICAL_TABLE`` markers; without it the
script only writes ``results.json`` and prints a comparison against the
note's published values.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from triton_tagi.cdf_remax import (
    remax_conditional_moments,
    remax_mixture_kernels,
    remax_scale_moments,
)
from triton_tagi.cdf_variance import cdf_variance_activation, cdf_variance_moments
from triton_tagi.remax_kernels import laplace_rule
from triton_tagi.remax_scale import fit_remax_log_scale
from triton_tagi.update.observation import compute_cdf_tagiv_innovation

DOUBLE = torch.float64

# The state fixed by the note.
MU = torch.tensor([[0.8, -0.3, 0.15]], dtype=DOUBLE)
VAR_Z = torch.tensor([[0.18, 0.09, 0.25]], dtype=DOUBLE)
NU = torch.tensor([[-0.7, 0.1, -0.3]], dtype=DOUBLE)
R = torch.tensor([[0.4, 0.2, 0.3]], dtype=DOUBLE)
EPSILON, KAPPA = 0.02, 1.5
LOG_SCALE = 0.0

MONTE_CARLO_DRAWS = 1_200_000
MONTE_CARLO_SEED = 20260910

COARSE = {"laplace_order": 160, "hermite_order": 32}
FINE = {"laplace_order": 256, "hermite_order": 48}

# Values printed in the note's table, for side-by-side comparison.
PUBLISHED: dict[str, str] = {
    "predictive_probabilities": "(0.577012, 0.153584, 0.269404)",
    "all_zero_probability": "0.038829",
    "max_probability_error_vs_mc": "1.02e-04",
    "max_covariance_error_vs_mc": "9.26e-05",
    "forward_quadrature_refinement_error": "4.78e-13",
    "simplex_mean_error": "3.55e-15",
    "covariance_row_sum_error": "2.62e-15",
    "whole_logit_gain_invariance_error": "3.33e-16",
    "batch_reference_scale": "2.0447",
    "population_nll_before": "1.081563",
    "population_nll_after": "1.054223",
}

# The scale the calibration labels are generated at, and their RNG seed.
GENERATING_SCALE = 1.8
CALIBRATION_LABELS = 3_000
CALIBRATION_SEED = 7891


def _monte_carlo_reference() -> tuple[torch.Tensor, torch.Tensor, float]:
    """Draw the full hierarchy and return probability moments and the atom."""

    generator = torch.Generator().manual_seed(MONTE_CARLO_SEED)
    classes = MU.numel()
    shape = (MONTE_CARLO_DRAWS, classes)

    head = NU + R.sqrt() * torch.randn(shape, generator=generator, dtype=DOUBLE)
    aleatoric = cdf_variance_activation(head, epsilon=EPSILON, kappa=KAPPA)
    epistemic = VAR_Z.sqrt() * torch.randn(shape, generator=generator, dtype=DOUBLE)
    noise = aleatoric.sqrt() * torch.randn(shape, generator=generator, dtype=DOUBLE)

    logits = MU + math.exp(LOG_SCALE) * (epistemic + noise)
    rectified = logits.clamp_min(0.0)
    total = rectified.sum(dim=1, keepdim=True)
    probabilities = torch.where(
        total > 0.0,
        rectified / total.clamp_min(1e-300),
        torch.full_like(rectified, 1.0 / classes),
    )
    second = probabilities.T @ probabilities / MONTE_CARLO_DRAWS
    atom = float((total.squeeze(1) == 0.0).to(DOUBLE).mean())
    return probabilities.mean(dim=0), second, atom


def _head_moment_checks() -> dict[str, Any]:
    """Exact CDF-head moments against Monte Carlo and against Stein's identity."""

    generator = torch.Generator().manual_seed(MONTE_CARLO_SEED + 1)
    sample = NU + R.sqrt() * torch.randn(
        (MONTE_CARLO_DRAWS, NU.numel()), generator=generator, dtype=DOUBLE
    )
    realised = cdf_variance_activation(sample, epsilon=EPSILON, kappa=KAPPA)
    h_mean, h_variance, cov_u_h = cdf_variance_moments(NU, R, epsilon=EPSILON, kappa=KAPPA)

    step = 1e-6
    upper, _, _ = cdf_variance_moments(NU + step, R, epsilon=EPSILON, kappa=KAPPA)
    lower, _, _ = cdf_variance_moments(NU - step, R, epsilon=EPSILON, kappa=KAPPA)
    stein = R * (upper - lower) / (2.0 * step)

    return {
        "h_mean": h_mean.squeeze(0).tolist(),
        "h_variance": h_variance.squeeze(0).tolist(),
        "cov_u_h": cov_u_h.squeeze(0).tolist(),
        "max_h_mean_error_vs_mc": float((h_mean - realised.mean(dim=0)).abs().max()),
        "max_h_variance_error_vs_mc": float((h_variance - realised.var(dim=0)).abs().max()),
        "max_cov_error_vs_stein": float((cov_u_h - stein).abs().max()),
        "bounded_below": bool((realised > EPSILON).all()),
        "bounded_above": bool((realised < EPSILON + KAPPA).all()),
    }


def _training_channel_checks() -> dict[str, Any]:
    """The Gaussian channel's posterior moments against dense integration."""

    targets = torch.tensor([[1.0, -1.0, -1.0]], dtype=DOUBLE)
    delta_mu, delta_var = compute_cdf_tagiv_innovation(
        targets, MU, VAR_Z, NU, R, epsilon=EPSILON, kappa=KAPPA, hermite_order=150
    )
    mu_post = MU + VAR_Z * delta_mu[:, 0::2]
    var_post = VAR_Z + VAR_Z * VAR_Z * delta_var[:, 0::2]
    nu_post = NU + R * delta_mu[:, 1::2]
    r_post = R + R * R * delta_var[:, 1::2]

    errors = []
    for index in range(MU.numel()):
        reference = _dense_channel_reference(index, float(targets[0, index]))
        computed = (
            float(nu_post[0, index]),
            float(r_post[0, index]),
            float(mu_post[0, index]),
            float(var_post[0, index]),
        )
        errors.extend(abs(a - b) for a, b in zip(computed, reference, strict=True))

    return {
        "signed_targets": targets.squeeze(0).tolist(),
        "nu_post": nu_post.squeeze(0).tolist(),
        "r_post": r_post.squeeze(0).tolist(),
        "mu_z_post": mu_post.squeeze(0).tolist(),
        "var_z_post": var_post.squeeze(0).tolist(),
        "max_posterior_error_vs_dense": max(errors),
        "all_variances_non_negative": bool((r_post >= 0.0).all() and (var_post >= 0.0).all()),
    }


def _dense_channel_reference(
    index: int,
    target: float,
    *,
    nodes: int = 400_001,
    reach: float = 14.0,
) -> tuple[float, float, float, float]:
    """Composite-Simpson posterior for one output unit, independent of the rule."""

    mean = float(MU[0, index])
    prediction = float(VAR_Z[0, index])
    head_mean = float(NU[0, index])
    head_var = float(R[0, index])
    deviation = math.sqrt(head_var)

    u = torch.linspace(
        head_mean - reach * deviation, head_mean + reach * deviation, nodes, dtype=DOUBLE
    )
    aleatoric = cdf_variance_activation(u, epsilon=EPSILON, kappa=KAPPA)
    total = prediction + aleatoric
    prior = torch.exp(-0.5 * (u - head_mean) ** 2 / head_var) / math.sqrt(2.0 * math.pi * head_var)
    likelihood = torch.exp(-0.5 * (target - mean) ** 2 / total) / torch.sqrt(2.0 * math.pi * total)
    weight = prior * likelihood

    rule = torch.full((nodes,), 2.0 / 3.0, dtype=DOUBLE)
    rule[1:-1:2] = 4.0 / 3.0
    rule[0] = rule[-1] = 1.0 / 3.0
    step = float(u[1] - u[0])

    mass = float((rule * weight).sum()) * step
    head_first = float((rule * weight * u).sum()) * step / mass
    head_second = float((rule * weight * u * u).sum()) * step / mass

    conditional_mean = mean + (prediction / total) * (target - mean)
    conditional_var = prediction * aleatoric / total
    prediction_first = float((rule * weight * conditional_mean).sum()) * step / mass
    prediction_second = (
        float((rule * weight * (conditional_var + conditional_mean**2)).sum()) * step / mass
    )
    return (
        head_first,
        head_second - head_first**2,
        prediction_first,
        prediction_second - prediction_first**2,
    )


def _population_nll(probabilities: torch.Tensor, log_scale: float) -> float:
    """Cross-entropy against the known generating probabilities.

    Evaluated against the generating vector rather than the realised label
    frequencies, so it measures the fit rather than the sample.
    """

    predicted, _ = remax_conditional_moments(
        MU, VAR_Z, NU, R, epsilon=EPSILON, kappa=KAPPA, log_scale=log_scale, **COARSE
    )
    return float(-(probabilities * predicted.squeeze(0).clamp_min(1e-300).log()).sum())


def _calibration_reference() -> dict[str, Any]:
    """Fit the shared scale from labels drawn at a known generating scale.

    This verifies the batch calibration reference of the note's objective, not
    the sequential recursion and not calibration on a trained network. Because
    every label shares one input state, the fit is passed class counts as
    sample weights, which is exactly equivalent to expanding the rows.
    """

    generating_log_scale = math.log(GENERATING_SCALE)
    generating, _ = remax_conditional_moments(
        MU,
        VAR_Z,
        NU,
        R,
        epsilon=EPSILON,
        kappa=KAPPA,
        log_scale=generating_log_scale,
        **COARSE,
    )
    generating = generating.squeeze(0)

    generator = torch.Generator().manual_seed(CALIBRATION_SEED)
    labels = torch.multinomial(
        generating, CALIBRATION_LABELS, replacement=True, generator=generator
    )
    counts = torch.bincount(labels, minlength=MU.numel()).to(DOUBLE)

    classes = MU.numel()
    posterior = fit_remax_log_scale(
        MU.repeat(classes, 1),
        VAR_Z.repeat(classes, 1),
        NU.repeat(classes, 1),
        R.repeat(classes, 1),
        torch.arange(classes),
        epsilon=EPSILON,
        kappa=KAPPA,
        prior_mean=0.0,
        prior_variance=0.25,
        method="grid",
        sample_weight=counts,
    )
    return {
        "generating_scale": GENERATING_SCALE,
        "generating_probabilities": generating.tolist(),
        "labels": CALIBRATION_LABELS,
        "label_seed": CALIBRATION_SEED,
        "label_counts": counts.tolist(),
        "batch_reference_log_scale": posterior.mean,
        "batch_reference_scale": math.exp(posterior.mean),
        "batch_reference_log_scale_deviation": math.sqrt(posterior.variance),
        "population_nll_before": _population_nll(generating, 0.0),
        "population_nll_after": _population_nll(generating, posterior.mean),
        "population_nll_at_generating_scale": _population_nll(generating, generating_log_scale),
    }


def _invariance_error() -> float:
    """``Remax(g x) = Remax(x)`` when the whole logit is scaled by ``g > 0``."""

    baseline, _ = remax_conditional_moments(
        MU, VAR_Z, NU, R, epsilon=EPSILON, kappa=KAPPA, **COARSE
    )
    worst = 0.0
    for gain in (0.25, 0.5, 2.0, 4.0):
        scaled, _ = remax_conditional_moments(
            MU * gain,
            VAR_Z * gain * gain,
            NU,
            R,
            epsilon=EPSILON * gain * gain,
            kappa=KAPPA * gain * gain,
            **COARSE,
        )
        worst = max(worst, float((scaled - baseline).abs().max()))
    return worst


def _zero_mean_invariance_error() -> float:
    """With every logit mean zero the scale is completely unidentified."""

    zero = torch.zeros_like(MU)
    baseline, _ = remax_conditional_moments(
        zero, VAR_Z, NU, R, epsilon=EPSILON, kappa=KAPPA, **COARSE
    )
    worst = 0.0
    for log_scale in (-1.5, -0.5, 0.8, 2.0):
        moved, _ = remax_conditional_moments(
            zero,
            VAR_Z,
            NU,
            R,
            epsilon=EPSILON,
            kappa=KAPPA,
            log_scale=log_scale,
            **COARSE,
        )
        worst = max(worst, float((moved - baseline).abs().max()))
    return worst


def _scale_sweep() -> dict[str, list[float]]:
    """Predictive probabilities against the shared scale, logit mean fixed."""

    sweep: dict[str, list[float]] = {}
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        first, _ = remax_conditional_moments(
            MU,
            VAR_Z,
            NU,
            R,
            epsilon=EPSILON,
            kappa=KAPPA,
            log_scale=math.log(scale),
            **COARSE,
        )
        sweep[f"{scale:g}"] = first.squeeze(0).tolist()
    return sweep


def run() -> dict[str, Any]:
    """Run every check and return the results."""

    first, second = remax_conditional_moments(
        MU, VAR_Z, NU, R, epsilon=EPSILON, kappa=KAPPA, cross_moments=True, **COARSE
    )
    covariance = second - first.unsqueeze(-1) * first.unsqueeze(-2)
    refined, _ = remax_conditional_moments(MU, VAR_Z, NU, R, epsilon=EPSILON, kappa=KAPPA, **FINE)

    t, _ = laplace_rule(COARSE["laplace_order"], reference=MU)
    *_, zero_probability = remax_mixture_kernels(
        MU, VAR_Z, NU, R, t, epsilon=EPSILON, kappa=KAPPA, hermite_order=COARSE["hermite_order"]
    )
    atom = float(zero_probability.prod())

    mc_first, mc_second, mc_atom = _monte_carlo_reference()
    mc_covariance = mc_second - torch.outer(mc_first, mc_first)

    scale_probabilities, scale_covariance, cov_scale = remax_scale_moments(
        MU,
        VAR_Z,
        NU,
        R,
        scale_mean=0.0,
        scale_variance=0.25,
        epsilon=EPSILON,
        kappa=KAPPA,
        cross_moments=True,
    )
    bound = 0.25 * scale_probabilities * (1.0 - scale_probabilities)
    calibration = _calibration_reference()

    return {
        "state": {
            "mu": MU.squeeze(0).tolist(),
            "var_z": VAR_Z.squeeze(0).tolist(),
            "nu": NU.squeeze(0).tolist(),
            "r": R.squeeze(0).tolist(),
            "epsilon": EPSILON,
            "kappa": KAPPA,
            "scale": math.exp(LOG_SCALE),
        },
        "quadrature": {"coarse": COARSE, "fine": FINE},
        "monte_carlo": {"draws": MONTE_CARLO_DRAWS, "seed": MONTE_CARLO_SEED},
        "predictive_probabilities": first.squeeze(0).tolist(),
        "all_zero_probability": atom,
        "all_zero_probability_mc": mc_atom,
        "max_probability_error_vs_mc": float((first.squeeze(0) - mc_first).abs().max()),
        "max_covariance_error_vs_mc": float((covariance.squeeze(0) - mc_covariance).abs().max()),
        "forward_quadrature_refinement_error": float((first - refined).abs().max()),
        "simplex_mean_error": abs(float(first.sum()) - 1.0),
        "covariance_row_sum_error": float(covariance.sum(dim=-1).abs().max()),
        "covariance_min_eigenvalue": float(torch.linalg.eigvalsh(covariance).min()),
        "covariance_diagonal_within_bernoulli_bound": bool(
            (torch.diagonal(covariance, dim1=1, dim2=2) <= first * (1.0 - first) + 1e-12).all()
        ),
        "whole_logit_gain_invariance_error": _invariance_error(),
        "zero_mean_scale_invariance_error": _zero_mean_invariance_error(),
        "scale_sweep": _scale_sweep(),
        "integrated_scale": {
            "prior_mean": 0.0,
            "prior_variance": 0.25,
            "probabilities": scale_probabilities.squeeze(0).tolist(),
            "cov_scale": cov_scale.squeeze(0).tolist(),
            "simplex_mean_error": abs(float(scale_probabilities.sum()) - 1.0),
            "covariance_row_sum_error": float(scale_covariance.sum(dim=-1).abs().max()),
            "cauchy_schwarz_satisfied": bool((cov_scale * cov_scale <= bound + 1e-15).all()),
        },
        "cdf_head": _head_moment_checks(),
        "training_channel": _training_channel_checks(),
        "calibration": calibration,
        "batch_reference_scale": calibration["batch_reference_scale"],
        "population_nll_before": calibration["population_nll_before"],
        "population_nll_after": calibration["population_nll_after"],
    }


def _table_rows(results: dict[str, Any]) -> list[tuple[str, str]]:
    def scientific(value: float) -> str:
        text = f"{value:.2e}"
        mantissa, exponent = text.split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"

    probabilities = ", ".join(f"{v:.6f}" for v in results["predictive_probabilities"])
    return [
        ("Predictive probability vector", f"$({probabilities})$"),
        ("All-zero probability", f"${results['all_zero_probability']:.6f}$"),
        (
            "Maximum probability error versus MC",
            scientific(results["max_probability_error_vs_mc"]),
        ),
        (
            "Maximum covariance error versus MC",
            scientific(results["max_covariance_error_vs_mc"]),
        ),
        (
            "Maximum training-channel error versus dense integration",
            scientific(results["training_channel"]["max_posterior_error_vs_dense"]),
        ),
        (
            "CDF-head mean / variance error versus MC",
            scientific(results["cdf_head"]["max_h_mean_error_vs_mc"])
            + " / "
            + scientific(results["cdf_head"]["max_h_variance_error_vs_mc"]),
        ),
        (
            "Head covariance error versus Stein identity",
            scientific(results["cdf_head"]["max_cov_error_vs_stein"]),
        ),
        (
            "Forward quadrature refinement error",
            scientific(results["forward_quadrature_refinement_error"]),
        ),
        (
            "Simplex mean / covariance row-sum error",
            scientific(results["simplex_mean_error"])
            + " / "
            + scientific(results["covariance_row_sum_error"]),
        ),
        (
            "Common whole-logit gain invariance error",
            scientific(results["whole_logit_gain_invariance_error"]),
        ),
        (
            "Zero-mean scale invariance error",
            scientific(results["zero_mean_scale_invariance_error"]),
        ),
        (
            f"Batch-reference scale from {results['calibration']['labels']:,} labels",
            f"${results['batch_reference_scale']:.4f}$ "
            f"(generating ${results['calibration']['generating_scale']}$)",
        ),
        (
            "Population NLL before / after scale fitting",
            f"${results['population_nll_before']:.6f}$ / ${results['population_nll_after']:.6f}$",
        ),
        (
            "Population NLL at the generating scale",
            f"${results['calibration']['population_nll_at_generating_scale']:.6f}$",
        ),
    ]


def render_table(results: dict[str, Any]) -> str:
    """Render the LaTeX table block for the note."""

    lines = [
        "\\begin{table}[H]",
        "\\centering\\small",
        "\\begin{tabular}{@{}p{.55\\textwidth}p{.40\\textwidth}@{}}",
        "\\toprule",
        "Quantity & Numerical result \\\\",
        "\\midrule",
    ]
    lines += [f"{name} & {value} \\\\" for name, value in _table_rows(results)]
    coarse = results["quadrature"]["coarse"]
    fine = results["quadrature"]["fine"]
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Verification for the specified state, produced by "
        "\\texttt{verification/check\\_formulation.py}. Monte Carlo uses "
        f"${results['monte_carlo']['draws'] / 1e6:.1f}\\times10^6$ draws with seed "
        f"{results['monte_carlo']['seed']}. Quadrature refinement compares "
        f"$(Q,J)=({coarse['laplace_order']},{coarse['hermite_order']})$ with "
        f"$({fine['laplace_order']},{fine['hermite_order']})$. The training-channel and "
        "head rows are checked against independent composite-Simpson integration and "
        "against Stein's identity rather than against sampling. Rows for the optional "
        "categorical-tilt route of Appendix~\\ref{S:training} are absent because that "
        "route is not implemented. Calibration labels use a separate seed, "
        f"{results['calibration']['label_seed']}, and the stated "
        "$\\Normal(0,0.5^2)$ log-scale prior; the population NLL is evaluated "
        "against the known generating probabilities, independently of the "
        "calibration-label frequencies.}",
        "\\label{T:checks}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def update_tex(path: Path, results: dict[str, Any]) -> bool:
    """Replace the marked table block in the note. Returns True if it changed."""

    source = path.read_text()
    pattern = re.compile(r"(% BEGIN NUMERICAL_TABLE\n).*?(\n% END NUMERICAL_TABLE)", re.DOTALL)
    if not pattern.search(source):
        raise ValueError(f"no NUMERICAL_TABLE markers found in {path}")
    updated = pattern.sub(
        lambda match: match.group(1) + render_table(results) + match.group(2), source
    )
    if updated == source:
        return False
    path.write_text(updated)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-tex",
        action="store_true",
        help="rewrite the table block inside the .tex note",
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "CDF_TAGIV_Remax_joint_calibration.tex",
        help="path to the note",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results.json",
        help="where to write the full results",
    )
    arguments = parser.parse_args()

    results = run()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2) + "\n")

    width = max(len(name) for name in PUBLISHED)
    print(f"{'quantity':<{width}}  {'computed':>26}  {'published':>26}")
    for key, published in PUBLISHED.items():
        value = results[key]
        computed = (
            "(" + ", ".join(f"{v:.6f}" for v in value) + ")"
            if isinstance(value, list)
            else (f"{value:.6f}" if abs(value) > 1e-4 else f"{value:.2e}")
        )
        print(f"{key:<{width}}  {computed:>26}  {published:>26}")

    print(f"\nfull results written to {arguments.output}")
    if arguments.update_tex:
        changed = update_tex(arguments.tex, results)
        print(f"{'updated' if changed else 'unchanged'}: {arguments.tex}")


if __name__ == "__main__":
    main()
