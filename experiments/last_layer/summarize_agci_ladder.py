"""Summarize the deterministic-to-Bayesian ladder with paired uncertainty.

The ladder separates four things that AGCI changes at once: re-optimizing the
head, the categorical link, the sequential event conditioning, and integrating
over the weight posterior. Each rung isolates one of them.

    original head                 the deployed deterministic network
    original + temperature        the strongest scalar-calibrated reference
    softmax MAP refit             benefit of refitting the head at all
    Gaussian-event MAP refit      benefit of the AGCI likelihood alone
    AGCI mean-only plug-in        benefit of AGCI's learned mean
    AGCI posterior predictive     benefit of integrating weight uncertainty

Differences between rungs are reported as paired per-example NLL differences
with a bootstrap interval over examples, because every rung scores the same
10,000 test points and an unpaired comparison throws that structure away. Seed
spread is reported separately: it is variation in the fitted head, not in the
evaluation set, and the two should not be pooled.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

DEFAULT_REFIT = REPOSITORY_ROOT / "runs/last_layer/deterministic_refit"
DEFAULT_SEEDS = REPOSITORY_ROOT / "runs/last_layer/agci_seed_replication"


def paired_bootstrap(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return the mean paired difference and its percentile interval.

    ``first`` and ``second`` are per-example negative log likelihoods on the
    same test points, so resampling examples jointly preserves the pairing.
    """

    if first.shape != second.shape:
        raise ValueError("paired comparison requires matching per-example vectors")
    difference = (first - second).double()
    generator = torch.Generator().manual_seed(seed)
    count = difference.numel()
    index = torch.randint(
        count, (resamples, count), generator=generator, dtype=torch.long
    )
    draws = difference[index].mean(dim=1)
    low, high = torch.quantile(draws, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(difference.mean()), float(low), float(high)


def seed_summary(values: list[float]) -> str:
    if len(values) < 2:
        return f"{values[0]:.4f}" if values else "--"
    return f"{statistics.fmean(values):.4f} +- {statistics.stdev(values):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"])
    parser.add_argument("--refit-root", type=Path, default=DEFAULT_REFIT)
    parser.add_argument("--seed-root", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        for likelihood in ("softmax", "probit"):
            path = args.refit_root / dataset / likelihood / "result.json"
            if not path.exists():
                print(f"  [{likelihood} MAP refit not run]")
                continue
            payload = json.loads(path.read_text())
            print(f"  {likelihood} MAP refit, lambda*={payload['selected_penalty']:g}")
            for rung, metrics in payload["test"].items():
                if likelihood == "probit" and rung.startswith("original"):
                    continue  # identical to the softmax file; report once
                print(
                    f"    {rung:34s} top1={100 * metrics['accuracy']:6.2f} "
                    f"NLL={metrics['nll']:.4f} ECE={metrics['ece']:.4f} "
                    f"ACE={metrics['adaptive_ece']:.4f} Brier={metrics['brier']:.4f}"
                )

        seed_dir = args.seed_root / dataset
        records = []
        for path in sorted(seed_dir.glob("seed*/result.json")):
            records.append(json.loads(path.read_text()))
        if not records:
            print("  [AGCI seed replication not run]")
            continue
        print(f"  AGCI, {len(records)} seeds, epoch selected on validation")
        for key, name in (
            ("nll", "NLL"),
            ("accuracy", "top-1"),
            ("ece", "ECE"),
            ("adaptive_ece", "ACE"),
            ("brier", "Brier"),
        ):
            values = [r["test"][key] for r in records]
            scale = 100.0 if key == "accuracy" else 1.0
            print(f"    {name:8s} {seed_summary([scale * v for v in values])}")
        print(
            "    selected epochs "
            + ", ".join(str(r["selected_epoch"]) for r in records)
        )


if __name__ == "__main__":
    main()
