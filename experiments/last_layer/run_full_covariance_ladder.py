"""Does full feature covariance change the posterior-integration term?

The diagonal ladder found posterior integration worth exactly zero on CIFAR-10
and +0.0059 NLL on CIFAR-100, and attributed the difference to posterior
contraction: the CIFAR-10 diagonal posterior had collapsed to a delta, so there
was nothing to integrate over. That explanation makes a testable prediction. A
posterior that retains feature-direction covariance does not collapse the same
way, so if the explanation is right, integration should stop being free on
CIFAR-10 once the covariance is dense.

This runner fits :class:`FullCovarianceADFClassifier`, which keeps one dense
``(D + 1) x (D + 1)`` precision per class and replaces sites across epochs
rather than reconsuming evidence, and reports two rungs at every prior gain:

    posterior predictive   integrates the full output variance
    mean-only plug-in      the same mean with the variance set to zero

Their difference is the value of integration under this posterior, measured the
same way as in the diagonal ladder. The gain is swept and selected on the
validation split alone; the test split is read once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import classification_metrics  # noqa: E402
from triton_tagi.cifar_study import load_feature_shard, seed_everything  # noqa: E402
from triton_tagi.full_covariance_adf import (  # noqa: E402
    FullCovarianceADFClassifier,
)
from triton_tagi.multinomial_probit import (  # noqa: E402
    multinomial_probit_adf_predictive_probs,
)

DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/last_layer/full_covariance_ladder"
DEFAULT_GAINS = (0.003, 0.01, 0.03, 0.1, 0.3)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


@torch.no_grad()
def score(
    model: FullCovarianceADFClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Return predictive metrics, mean-only metrics, and mean output variance."""

    predictive, plug_in, variance_total = [], [], 0.0
    for start in range(0, features.shape[0], batch_size):
        chunk = features[start : start + batch_size]
        mean, variance = model.output_moments(chunk)
        variance_total += float(variance.mean()) * chunk.shape[0]
        predictive.append(
            multinomial_probit_adf_predictive_probs(
                mean, variance, probit_tau2=model.probit_tau2
            ).float().cpu()
        )
        # The plug-in rung deletes epistemic weight uncertainty and keeps the
        # same decision noise, so the two differ only in the integration.
        plug_in.append(
            multinomial_probit_adf_predictive_probs(
                mean, torch.zeros_like(variance), probit_tau2=model.probit_tau2
            ).float().cpu()
        )
    return (
        classification_metrics(torch.cat(predictive), labels),
        classification_metrics(torch.cat(plug_in), labels),
        variance_total / features.shape[0],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gains", nargs="+", type=float, default=list(DEFAULT_GAINS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=1024)
    parser.add_argument("--tau2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = args.study_root / "features" / args.dataset
    train = load_feature_shard(root / "train.pt")
    validation = load_feature_shard(root / "validation.pt")
    test = load_feature_shard(root / "test.pt")
    train_x = train["features"].float()
    train_y = train["labels"].long()

    records = []
    for gain in args.gains:
        seed_everything(args.seed)
        started = time.perf_counter()
        model = FullCovarianceADFClassifier(
            train_x.shape[1],
            int(train_y.max().item()) + 1,
            device=args.device,
            gain_w=gain,
            gain_b=gain,
            probit_tau2=args.tau2,
        )
        model.fit(
            train_x,
            train_y,
            batch_size=args.batch_size,
            seed=args.seed,
            epochs=args.epochs,
        )
        predictive, plug_in, variance = score(
            model,
            validation["features"].float(),
            validation["labels"],
            batch_size=args.prediction_batch_size,
        )
        records.append(
            {
                "gain": gain,
                "validation_predictive": predictive,
                "validation_plug_in": plug_in,
                "validation_output_variance": variance,
                "wall_s": time.perf_counter() - started,
                "model": model,
            }
        )
        print(
            f"gain={gain:<7g} predictive nll={predictive['nll']:.4f} "
            f"acc={100 * predictive['accuracy']:.2f}  |  plug-in nll={plug_in['nll']:.4f} "
            f"acc={100 * plug_in['accuracy']:.2f}  |  mean out var={variance:.4g}"
        )

    # Each rung selects its own gain on validation, because the prior that suits
    # a predictive that integrates variance need not suit one that ignores it.
    best_predictive = min(records, key=lambda r: r["validation_predictive"]["nll"])
    best_plug_in = min(records, key=lambda r: r["validation_plug_in"]["nll"])
    print(
        f"selected gain: predictive={best_predictive['gain']:g} "
        f"plug-in={best_plug_in['gain']:g}"
    )

    test_x, test_y = test["features"].float(), test["labels"]
    result = {"dataset": args.dataset, "epochs": args.epochs, "tau2": args.tau2}
    for tag, record in (("predictive", best_predictive), ("plug_in", best_plug_in)):
        predictive, plug_in, variance = score(
            record["model"], test_x, test_y, batch_size=args.prediction_batch_size
        )
        result[f"test_{tag}_selected"] = {
            "gain": record["gain"],
            "posterior_predictive": predictive,
            "mean_only_plug_in": plug_in,
            "mean_output_variance": variance,
        }
    for record in records:
        record.pop("model", None)
    result["gain_sweep"] = records

    atomic_json(args.output / args.dataset / "result.json", result)
    print(f"\n{args.dataset} clean test")
    for tag in ("predictive", "plug_in"):
        block = result[f"test_{tag}_selected"]
        print(f"  gain selected for {tag} = {block['gain']:g}")
        for name in ("posterior_predictive", "mean_only_plug_in"):
            metrics = block[name]
            print(
                f"    {name:22s} top1={100 * metrics['accuracy']:6.2f} "
                f"NLL={metrics['nll']:.4f} ECE={metrics['ece']:.4f} "
                f"Brier={metrics['brier']:.4f}"
            )
    print(f"\nwrote {args.output / args.dataset / 'result.json'}")


if __name__ == "__main__":
    main()
