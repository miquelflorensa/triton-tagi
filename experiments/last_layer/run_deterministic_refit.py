"""MAP softmax refit on frozen features: the missing deterministic reference.

Every TAGI last-layer head in this study is compared against the backbone's
own classifier, which was trained jointly with the representation under
augmentation. That is the wrong control for the question the study asks.
With the features frozen, deterministic softmax training is a convex problem
in ``(W, b)``, so the properly optimized deterministic reference is

    argmin_{W,b}  -sum_i log softmax_{y_i}(W h_i + b) + lam ||W||^2 ,

fit on the same 40,000 training features, with ``lam`` selected on the same
10,000-example validation split. Without it, an apparent TAGI improvement
cannot be separated from the benefit of simply re-optimizing the head on
non-augmented frozen features, from different regularization, or from a
temperature-like rescaling.

The runner reports four deterministic rungs: the original head, the original
head with a fitted temperature, the MAP refit, and the MAP refit with a fitted
temperature. Only the validation split is used for selection; the test split
is read once for the final numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import (  # noqa: E402
    agci_predictive_probs,
    classification_metrics,
    fit_softmax_temperature,
)
from triton_tagi.cifar_study import load_feature_shard  # noqa: E402

DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/last_layer/deterministic_refit"
DEFAULT_PENALTIES = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def probit_event_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    tau: float,
    num_quad: int,
) -> torch.Tensor:
    """Return ``log P(argmax(z + e) == y)`` for i.i.d. ``e ~ N(0, tau**2)``.

    This is the AGCI event likelihood evaluated at a deterministic utility
    vector. Conditioning on the winning noisy utility reduces the nominal
    ``K - 1`` dimensional integral to one dimension, so only ``O(QK)`` work is
    needed for the observed class rather than the ``O(QK**2)`` the full
    predictive requires. The integrand is a product of log-concave normal CDFs,
    so the log-likelihood is concave in the logits and the MAP fit below is a
    convex problem exactly as the softmax one is.
    """

    nodes, weights = np.polynomial.hermite.hermgauss(num_quad)
    node = torch.as_tensor(
        nodes * math.sqrt(2.0), device=logits.device, dtype=logits.dtype
    )
    log_weight = torch.as_tensor(
        np.log(weights) - 0.5 * math.log(math.pi),
        device=logits.device,
        dtype=logits.dtype,
    )
    rows = torch.arange(logits.shape[0], device=logits.device)
    margin = (logits[rows, labels].unsqueeze(-1) - logits) / tau
    log_cdf = torch.special.log_ndtr(margin.unsqueeze(1) + node.view(1, -1, 1))
    observed = torch.zeros_like(logits, dtype=torch.bool)
    observed[rows, labels] = True
    # The observed class is the conditioning variable, not a competitor.
    log_cdf = log_cdf.masked_fill(observed.unsqueeze(1), 0.0)
    return torch.logsumexp(log_weight.view(1, -1) + log_cdf.sum(-1), dim=-1)


def fit_map(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    penalty: float,
    steps: int,
    likelihood: str,
    tau: float,
    num_quad: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the L2-penalized MAP weights and bias under one likelihood.

    Both the softmax and the Gaussian-event (multinomial probit) negative log
    likelihoods are convex in ``(W, b)``, so full-batch L-BFGS reaches the
    global optimum and the result does not depend on initialization or example
    order. The penalty is a ridge on ``W`` only; the bias is left free so class
    prevalence is not shrunk toward uniform. The full-batch gradient is
    accumulated over chunks because the probit integrand is ``O(QK)`` per
    example and does not fit at once.
    """

    num_classes = int(labels.max().item()) + 1
    weight = torch.zeros(
        features.shape[1],
        num_classes,
        device=features.device,
        dtype=features.dtype,
        requires_grad=True,
    )
    bias = torch.zeros(
        num_classes, device=features.device, dtype=features.dtype, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        max_iter=steps,
        history_size=20,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    count = features.shape[0]

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=features.device, dtype=features.dtype)
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            logits = features[start:stop] @ weight + bias
            if likelihood == "softmax":
                part = torch.nn.functional.cross_entropy(
                    logits, labels[start:stop], reduction="sum"
                )
            else:
                part = -probit_event_logprob(
                    logits, labels[start:stop], tau=tau, num_quad=num_quad
                ).sum()
            (part / count).backward()
            total = total + part.detach() / count
        penalty_term = penalty * weight.square().sum()
        penalty_term.backward()
        return total + penalty_term.detach()

    optimizer.step(closure)
    return weight.detach(), bias.detach()


def evaluate(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    likelihood: str = "softmax",
    tau: float = 1.0,
    num_quad: int = 48,
    chunk_size: int = 512,
) -> dict[str, float]:
    """Score a deterministic head under its own predictive map."""

    if likelihood == "softmax":
        return classification_metrics(torch.softmax(logits.float(), dim=-1), labels)
    parts = [
        agci_predictive_probs(
            logits[start : start + chunk_size],
            torch.zeros_like(logits[start : start + chunk_size]),
            tau=tau,
            num_quad=num_quad,
        ).float().cpu()
        for start in range(0, logits.shape[0], chunk_size)
    ]
    return classification_metrics(torch.cat(parts), labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--penalties", nargs="+", type=float, default=list(DEFAULT_PENALTIES)
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--likelihood",
        choices=("softmax", "probit"),
        default="softmax",
        help=(
            "deterministic training likelihood; probit is the Gaussian "
            "argmax-event likelihood AGCI conditions on, which isolates the "
            "effect of the link from the effect of the posterior"
        ),
    )
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--num-quad", type=int, default=48)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="examples per gradient chunk; the probit fit costs O(QK) each",
    )
    parser.add_argument(
        "--eval-chunk-size",
        type=int,
        default=256,
        help="examples per predictive chunk; the probit predictive costs O(QK^2)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    feature_root = (
        args.feature_root
        if args.feature_root is not None
        else DEFAULT_STUDY_ROOT / "features" / args.dataset
    )
    train = load_feature_shard(feature_root / "train.pt")
    validation = load_feature_shard(feature_root / "validation.pt")
    test = load_feature_shard(feature_root / "test.pt")

    device = torch.device(args.device)
    # float64 keeps the convex solve exact; the layer is small enough that the
    # cost is irrelevant and it removes any doubt about the optimum.
    train_x = train["features"].to(device=device, dtype=torch.float64)
    train_y = train["labels"].to(device).long()
    validation_x = validation["features"].to(device=device, dtype=torch.float64)
    test_x = test["features"].to(device=device, dtype=torch.float64)

    scoring = dict(
        likelihood=args.likelihood,
        tau=args.tau,
        num_quad=args.num_quad,
        chunk_size=args.eval_chunk_size,
    )
    records = []
    for penalty in args.penalties:
        weight, bias = fit_map(
            train_x,
            train_y,
            penalty=penalty,
            steps=args.steps,
            likelihood=args.likelihood,
            tau=args.tau,
            num_quad=args.num_quad,
            chunk_size=args.chunk_size,
        )
        validation_metrics = evaluate(
            validation_x @ weight + bias, validation["labels"], **scoring
        )
        records.append(
            {
                "penalty": penalty,
                "validation": validation_metrics,
                "state": (weight, bias),
            }
        )
        print(
            f"lambda={penalty:<8g} val nll={validation_metrics['nll']:.4f} "
            f"acc={100 * validation_metrics['accuracy']:.2f}"
        )

    selected = min(records, key=lambda row: row["validation"]["nll"])
    weight, bias = selected.pop("state")
    for row in records:
        row.pop("state", None)
    print(f"selected lambda={selected['penalty']:g}")

    refit_validation_logits = validation_x @ weight + bias
    refit_test_logits = test_x @ weight + bias
    if args.likelihood == "softmax":
        refit_temperature = fit_softmax_temperature(
            refit_validation_logits.float().cpu(), validation["labels"]
        )
    else:
        # The probit head has no logits to rescale, so the scalar is fit on the
        # log of its own predictive, which is the same one-parameter family.
        probabilities = torch.cat(
            [
                agci_predictive_probs(
                    refit_validation_logits[start : start + args.eval_chunk_size],
                    torch.zeros_like(
                        refit_validation_logits[start : start + args.eval_chunk_size]
                    ),
                    tau=args.tau,
                    num_quad=args.num_quad,
                ).float().cpu()
                for start in range(
                    0, refit_validation_logits.shape[0], args.eval_chunk_size
                )
            ]
        )
        refit_temperature = fit_softmax_temperature(
            probabilities.clamp_min(1e-12).log(), validation["labels"]
        )
    original_temperature = fit_softmax_temperature(
        validation["logits"].float(), validation["labels"]
    )

    result = {
        "dataset": args.dataset,
        "likelihood": args.likelihood,
        "tau": args.tau,
        "selected_penalty": selected["penalty"],
        "penalty_sweep": records,
        "refit_temperature": float(refit_temperature),
        "original_temperature": float(original_temperature),
        "test": {
            "original_head": evaluate(test["logits"].float(), test["labels"]),
            "original_head_temperature_scaled": evaluate(
                test["logits"].float() / original_temperature, test["labels"]
            ),
            "map_refit": evaluate(refit_test_logits, test["labels"], **scoring),
            "map_refit_temperature_scaled": evaluate(
                refit_test_logits.float() / refit_temperature,
                test["labels"],
                **scoring,
            ),
        },
    }
    output = args.output / args.dataset / args.likelihood
    atomic_json(output / "result.json", result)
    torch.save(
        {"weight": weight.cpu(), "bias": bias.cpu(), "penalty": selected["penalty"]},
        output / "map_head.pt",
    )

    print(f"\n{args.dataset} clean test")
    header = f"{'rung':36s} {'top1':>6s} {'NLL':>8s} {'ECE':>7s} {'Brier':>7s}"
    print(header)
    for name, metrics in result["test"].items():
        print(
            f"{name:36s} {100 * metrics['accuracy']:6.2f} {metrics['nll']:8.4f} "
            f"{metrics['ece']:7.4f} {metrics['brier']:7.4f}"
        )
    print(f"\nwrote {output / 'result.json'}")


if __name__ == "__main__":
    main()
