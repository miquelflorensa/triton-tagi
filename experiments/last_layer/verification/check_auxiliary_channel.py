"""Verification of the auxiliary Gaussian calibration channel.

Referenced by ``CDF_TAGIV_Remax_joint_calibration.tex``, paragraph
"Auxiliary-channel check". Verifies the three stated update variants for the
shared log-scale on the note's synthetic three-class state:

* the common posterior-mean identity, ``lambda^+ = lambda + c_c / p_c``, which
  equals ``E[L | C = c] = E[L A_c] / p_c`` and is exact for the categorical
  likelihood in all three variants;
* the observed-event expected-variance projection, ``q^+_event``;
* the full categorical projection ``q^+_full = q - sum_i c_i^2 / p_i``, which
  equals ``E_C[Var(L | C)]``, coincides with the event form at ``K = 2`` and is
  no larger than it for ``K > 2``;
* the exact scalar moment tilt ``q^+_tilt``, which is the variance for the
  *realised* label rather than an average over the label partition, and may
  therefore exceed ``q``;
* the Bernoulli total-variance split ``R_c + v_{A_c} = p_c (1 - p_c)``;
* the Cauchy--Schwarz consistency ``c_c^2 <= q p_c (1 - p_c)``.

These are one-step checks. They say nothing about the repeated-pass behaviour
of the sequential recursion, which is an adaptive filtering heuristic rather
than exact Bayes, and nothing about calibration on a trained network.

Run ``python -m experiments.last_layer.verification.check_auxiliary_channel``
from the repository root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from triton_tagi.cdf_remax import remax_scale_moments
from triton_tagi.hsm_calibration import expected_bernoulli_variance
from triton_tagi.remax_scale import (
    remax_scale_event_update,
    remax_scale_full_update,
    remax_scale_tilt_update,
)

DOUBLE = torch.float64

# The state fixed by the note.
MU = torch.tensor([[0.8, -0.3, 0.15]], dtype=DOUBLE)
VAR_Z = torch.tensor([[0.18, 0.09, 0.25]], dtype=DOUBLE)
NU = torch.tensor([[-0.7, 0.1, -0.3]], dtype=DOUBLE)
R = torch.tensor([[0.4, 0.2, 0.3]], dtype=DOUBLE)
EPSILON, KAPPA = 0.02, 1.5

# The note's auxiliary-channel check: prior N(0.1, 0.2^2), observed class two.
PRIOR_MEAN, PRIOR_VARIANCE = 0.1, 0.2**2
OBSERVED = 1  # zero-based index of the note's one-based "C = 2"

PUBLISHED: dict[str, float] = {
    "posterior_mean": 0.132152,
    "posterior_variance_event": 0.0397933,
    "posterior_variance_full": 0.0396814,
    "posterior_variance_tilt": 0.0385930,
}


def _moments(scale_order: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return remax_scale_moments(
        MU,
        VAR_Z,
        NU,
        R,
        scale_mean=PRIOR_MEAN,
        scale_variance=PRIOR_VARIANCE,
        epsilon=EPSILON,
        kappa=KAPPA,
        scale_order=scale_order,
        cross_moments=False,
    )


def _updates(scale_order: int) -> dict[str, float]:
    probabilities, class_variance, cov_scale = _moments(scale_order)
    row_p, row_c = probabilities[0], cov_scale[0]

    mean_event, variance_event = remax_scale_event_update(
        PRIOR_MEAN, PRIOR_VARIANCE, row_p, row_c, OBSERVED
    )
    mean_full, variance_full = remax_scale_full_update(
        PRIOR_MEAN, PRIOR_VARIANCE, row_p, row_c, OBSERVED
    )
    mean_tilt, variance_tilt = remax_scale_tilt_update(
        PRIOR_MEAN,
        PRIOR_VARIANCE,
        MU,
        VAR_Z,
        NU,
        R,
        OBSERVED,
        epsilon=EPSILON,
        kappa=KAPPA,
        scale_order=scale_order,
    )
    return {
        "posterior_mean": mean_event,
        "posterior_mean_full": mean_full,
        "posterior_mean_tilt": mean_tilt,
        "posterior_variance_event": variance_event,
        "posterior_variance_full": variance_full,
        "posterior_variance_tilt": variance_tilt,
        "probability": float(row_p[OBSERVED]),
        "cov_scale": float(row_c[OBSERVED]),
        "class_variance": float(class_variance[0, OBSERVED]),
    }


def run() -> dict[str, Any]:
    """Run every one-step check and return the results."""

    coarse = _updates(20)
    refined = _updates(32)

    probabilities, class_variance, cov_scale = _moments(20)
    row_p, row_v, row_c = probabilities[0], class_variance[0], cov_scale[0]
    surrogate = expected_bernoulli_variance(row_p, row_v)

    mean_spread = max(
        abs(coarse["posterior_mean"] - coarse["posterior_mean_full"]),
        abs(coarse["posterior_mean"] - coarse["posterior_mean_tilt"]),
    )
    refinement = max(abs(coarse[key] - refined[key]) for key in coarse)

    return {
        "state": {
            "mu": MU.squeeze(0).tolist(),
            "var_z": VAR_Z.squeeze(0).tolist(),
            "nu": NU.squeeze(0).tolist(),
            "r": R.squeeze(0).tolist(),
            "epsilon": EPSILON,
            "kappa": KAPPA,
        },
        "prior": {"mean": PRIOR_MEAN, "variance": PRIOR_VARIANCE},
        "observed_class_zero_based": OBSERVED,
        "predictive_probabilities": row_p.tolist(),
        "class_variances": row_v.tolist(),
        "cov_scale": row_c.tolist(),
        **{key: coarse[key] for key in PUBLISHED},
        "posterior_mean_full": coarse["posterior_mean_full"],
        "posterior_mean_tilt": coarse["posterior_mean_tilt"],
        "common_mean_identity_spread": mean_spread,
        "scale_quadrature_refinement_error": refinement,
        "full_projection_no_larger_than_event": (
            coarse["posterior_variance_full"] <= coarse["posterior_variance_event"]
        ),
        "full_projection_margin": (
            coarse["posterior_variance_event"] - coarse["posterior_variance_full"]
        ),
        "bernoulli_split_error": float((surrogate + row_v - row_p * (1.0 - row_p)).abs().max()),
        "expected_bernoulli_variance": surrogate.tolist(),
        "cauchy_schwarz_worst_ratio": float(
            (row_c * row_c / (PRIOR_VARIANCE * row_p * (1.0 - row_p))).max()
        ),
        "every_variance_non_negative": all(
            coarse[key] >= 0.0
            for key in (
                "posterior_variance_event",
                "posterior_variance_full",
                "posterior_variance_tilt",
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "auxiliary_results.json",
        help="where to write the full results",
    )
    arguments = parser.parse_args()

    results = run()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2) + "\n")

    width = max(len(name) for name in PUBLISHED)
    print(f"{'quantity':<{width}}  {'computed':>12}  {'published':>12}  {'diff':>9}")
    for key, published in PUBLISHED.items():
        value = results[key]
        print(f"{key:<{width}}  {value:>12.7f}  {published:>12.7f}  {abs(value - published):>9.1e}")

    print()
    print(f"common mean identity spread        {results['common_mean_identity_spread']:.2e}")
    print(f"scale quadrature refinement (20/32) {results['scale_quadrature_refinement_error']:.2e}")
    print(f"q+_full <= q+_event                {results['full_projection_no_larger_than_event']}")
    print(f"Bernoulli split R + v - p(1-p)      {results['bernoulli_split_error']:.2e}")
    print(f"Cauchy-Schwarz worst ratio         {results['cauchy_schwarz_worst_ratio']:.6f}")
    print(f"\nfull results written to {arguments.output}")


if __name__ == "__main__":
    main()
