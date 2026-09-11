"""Frozen last-layer ProbiTree on CIFAR-10: screen, then the full UQ report.

The direct-probit ``ProbiTree`` head was not part of the four-head
initialization study, so this is its own driver rather than a new column in
``run_study.py``'s grid: it reuses that study's cached features, batch size and
metric functions, but keeps its runs out of the study's run-hash space.

Axis 2 (gain) is swept as the plan specifies. ``probitree_r`` replaces axis 3:
it is the branch-noise *variance*, so it is the direct analogue of the fixed
``sigma_v`` the other fixed-noise heads take. ``mean_init="backbone"`` is
excluded because the head refuses it - a gate mapping cannot be read off
class-logit weights.

Usage:
    PYTHONPATH=. python experiments/last_layer/run_probitree_uq.py
    PYTHONPATH=. python experiments/last_layer/run_probitree_uq.py --epochs 5 --quick
    PYTHONPATH=. python experiments/last_layer/run_probitree_uq.py --dataset cifar100
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import torch

from triton_tagi.classification import TAGILastLayerClassifier
from triton_tagi.metrics import (
    classification_metrics,
    evaluate_ood_comprehensive,
    ood_detection_metrics_full,
    predictive_entropy,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FEATURES_ROOT = (
    REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k/features"
)
BATCH_SIZE = 256

# The four-head study's selected `hrc` cell per dataset, read off report.csv's
# confirm stage, so the baseline column is the established one rather than a
# guess. Those legacy cells omit `mean_init` from the run config - they predate
# axis 1 - so their init is `classification.py`'s default; `zero` is used here
# instead, for continuity with the CIFAR-10 row already recorded in the plan.
HRC_BASELINE = {
    "cifar10": {"gain_w": 0.3, "gain_b": 0.3, "sigma_v": 0.3, "mean_init": "zero"},
    "cifar100": {"gain_w": 0.3, "gain_b": 0.1, "sigma_v": 0.1, "mean_init": "zero"},
}

# Set by main(), so load() and the corruption sweep read one dataset's cache.
FEATURES = FEATURES_ROOT / "cifar10"


def load(name: str) -> dict:
    shard = torch.load(FEATURES / f"{name}.pt", map_location="cpu", weights_only=False)
    return {"features": shard["features"], "labels": shard["labels"]}


@torch.no_grad()
def predict_batches(classifier, features: torch.Tensor, batch_size: int):
    """Mirror ``run_study.predict_batches``: probabilities and a scalar epistemic score."""

    probability_parts, epistemic_parts = [], []
    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size].to(classifier.device)
        prediction = classifier.predict(batch)
        probability_parts.append(prediction.probabilities.cpu())
        if prediction.epistemic_variance is not None:
            epistemic_parts.append(
                prediction.epistemic_variance.reshape(batch.shape[0], -1).mean(1).cpu()
            )
    epistemic = torch.cat(epistemic_parts) if epistemic_parts else None
    return torch.cat(probability_parts), epistemic


def fit_one(train, validation, *, gain: float, r: float, mean_init: str, epochs: int, seed: int):
    classifier = TAGILastLayerClassifier(
        train["features"].shape[1],
        int(train["labels"].max()) + 1,
        device="cuda",
        head="probitree",
        probitree_r=r,
        gain_w=gain,
        gain_b=gain,
        mean_init=mean_init,
    )
    classifier.fit(
        train["features"],
        train["labels"],
        epochs=epochs,
        batch_size=BATCH_SIZE,
        seed=seed,
        validation=(validation["features"], validation["labels"]),
    )
    return classifier


def full_report(classifier, test, svhn) -> dict:
    """Return the accuracy / calibration / OOD families for one fitted head."""

    probabilities_id, epistemic_id = predict_batches(classifier, test["features"], BATCH_SIZE)
    probabilities_ood, epistemic_ood = predict_batches(classifier, svhn["features"], BATCH_SIZE)
    return {
        "classification": classification_metrics(probabilities_id, test["labels"]),
        "ood_svhn": evaluate_ood_comprehensive(
            probabilities_id,
            probabilities_ood,
            epistemic_id=epistemic_id,
            epistemic_ood=epistemic_ood,
        ),
    }


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def feature_energy_diagnostic(classifier, test, svhn) -> dict:
    """Ask what the native epistemic score is actually ranking.

    A frozen *deterministic* backbone feeds the head zero input variance, so
    the TAGI linear output variance collapses to ``Sz = ma^2 @ Sw + Sb`` - no
    ``Sa`` term survives. With a near-isotropic ``Sw`` that is a monotone
    function of the feature energy ``||f||^2``, which is a property of the
    setup rather than of the head or of anything training learned. This
    records the rank correlation and the AUROC that raw feature energy alone
    would earn, so the native-epistemic column can be read honestly.
    """

    _, epistemic_id = predict_batches(classifier, test["features"], BATCH_SIZE)
    _, epistemic_ood = predict_batches(classifier, svhn["features"], BATCH_SIZE)
    energy_id = (test["features"] ** 2).sum(1)
    energy_ood = (svhn["features"] ** 2).sum(1)
    return {
        "spearman_epistemic_vs_feature_energy": spearman(epistemic_id, energy_id),
        "auroc_native_epistemic": ood_detection_metrics_full(epistemic_id, epistemic_ood)["auroc"],
        "auroc_feature_energy_alone": ood_detection_metrics_full(energy_id, energy_ood)["auroc"],
        "mean_feature_energy_id": float(energy_id.mean()),
        "mean_feature_energy_ood": float(energy_ood.mean()),
    }


def corruption_energy_sweep(classifier, test) -> list[dict]:
    """Test the feature-energy account against the 75 cached corruption shards.

    If the native epistemic score is a monotone read of ``||f||^2``, then its
    AUROC against a shifted set is decided by the *sign* of that set's mean
    energy gap to the in-distribution test features, not by how shifted it is:
    sets whose features are larger than ID should score above 0.5 and sets
    whose features are smaller should score below, both for the same
    non-reason. SVHN happens to sit on the low side. Predictive entropy is
    swept alongside as the control that should track severity instead.
    """

    id_probabilities, id_epistemic = predict_batches(
        classifier, test["features"], BATCH_SIZE
    )
    id_entropy = predictive_entropy(id_probabilities)
    id_energy = (test["features"] ** 2).sum(1)

    rows = []
    for shard in sorted((FEATURES / "corruptions").glob("*.pt")):
        corruption, _, severity = shard.stem.rpartition("_")
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        features = payload["features"]
        probabilities, epistemic = predict_batches(classifier, features, BATCH_SIZE)
        energy = (features ** 2).sum(1)
        rows.append({
            "corruption": corruption,
            "severity": int(severity.lstrip("s")),
            "mean_feature_energy": float(energy.mean()),
            "energy_gap_vs_id": float(energy.mean() - id_energy.mean()),
            "accuracy": classification_metrics(probabilities, payload["labels"])["accuracy"],
            "auroc_native_epistemic": ood_detection_metrics_full(id_epistemic, epistemic)["auroc"],
            "auroc_entropy": ood_detection_metrics_full(
                id_entropy, predictive_entropy(probabilities)
            )["auroc"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="one cell, for a smoke test")
    parser.add_argument(
        "--gains", type=float, nargs="+", default=None,
        help="override the gain axis (default 0.03 0.1 0.3 1.0)",
    )
    parser.add_argument(
        "--r-grid", type=float, nargs="+", default=None,
        help="override the probitree_r axis (default 0.25 1.0). The gate margin is "
             "m / sqrt(r + v), so r trades off against gain and the useful range "
             "shifts with tree depth",
    )
    parser.add_argument(
        "--inits", nargs="+", choices=("random", "zero"), default=None,
        help="override the mean_init axis (default random zero)",
    )
    parser.add_argument(
        "--reuse-screen", action="store_true",
        help="keep the screen already in --output and refit only the selected cell",
    )
    parser.add_argument(
        "--corruption-sweep", action="store_true",
        help="sweep the 75 cached CIFAR-10-C shards for the feature-energy test",
    )
    parser.add_argument(
        "--output", default=None,
        help="defaults to experiments/last_layer/probitree_uq[_<dataset>].json",
    )
    args = parser.parse_args()

    global FEATURES
    FEATURES = FEATURES_ROOT / args.dataset
    if args.output is None:
        suffix = "" if args.dataset == "cifar10" else f"_{args.dataset}"
        args.output = f"experiments/last_layer/probitree_uq{suffix}.json"

    train, validation = load("train"), load("validation")
    test, svhn = load("test"), load("svhn")
    print(f"train={tuple(train['features'].shape)} val={tuple(validation['features'].shape)} "
          f"test={tuple(test['features'].shape)} svhn={tuple(svhn['features'].shape)}")

    gains = args.gains or ([0.3] if args.quick else [0.03, 0.1, 0.3, 1.0])
    noises = args.r_grid or ([1.0] if args.quick else [0.25, 1.0])
    inits = args.inits or (["zero"] if args.quick else ["random", "zero"])

    output = REPOSITORY_ROOT / args.output
    if args.reuse_screen:
        previous = json.loads(output.read_text())
        screen = previous["screen"]
        print(f"\n  reusing the {len(screen)}-cell screen already in {args.output}")
    else:
        screen = None

    print(f"\n  {'gain':>5} {'r':>5} {'init':>7} {'val NLL':>9} {'val acc':>8} "
          f"{'val ECE':>8} {'conf':>7} {'s':>6}")
    print("  " + "-" * 62)
    cells = [] if screen is None else screen
    for gain, r, mean_init in [] if screen is not None else itertools.product(gains, noises, inits):
        started = time.perf_counter()
        classifier = fit_one(
            train, validation, gain=gain, r=r, mean_init=mean_init,
            epochs=args.epochs, seed=args.seed,
        )
        probabilities, _ = predict_batches(classifier, validation["features"], BATCH_SIZE)
        metrics = classification_metrics(probabilities, validation["labels"])
        elapsed = time.perf_counter() - started
        # ECE and mean confidence are recorded per cell, not just NLL: this is a
        # calibration study, and on CIFAR-100 the NLL-best and accuracy-best
        # cells differ, so selection needs to be auditable against ECE.
        cell = {"gain": gain, "probitree_r": r, "mean_init": mean_init,
                "val_nll": metrics["nll"], "val_accuracy": metrics["accuracy"],
                "val_ece": metrics["ece"], "val_adaptive_ece": metrics["adaptive_ece"],
                "val_brier": metrics["brier"],
                "val_mean_confidence": metrics["mean_confidence"],
                "seconds": elapsed}
        cells.append(cell)
        print(f"  {gain:5.2f} {r:5.2f} {mean_init:>7} {metrics['nll']:9.4f} "
              f"{metrics['accuracy'] * 100:7.2f}% {metrics['ece']:8.4f} "
              f"{metrics['mean_confidence']:7.4f} {elapsed:6.1f}", flush=True)

    screen = cells
    best = min(screen, key=lambda cell: cell["val_nll"])
    print(f"\n  selected by val NLL: gain={best['gain']}, r={best['probitree_r']}, "
          f"init={best['mean_init']}")

    winner = fit_one(
        train, validation, gain=best["gain"], r=best["probitree_r"],
        mean_init=best["mean_init"], epochs=args.epochs, seed=args.seed,
    )
    probitree_report = full_report(winner, test, svhn)

    # The hrc baseline at the plan's established CIFAR-10 settings, so the UQ
    # columns - and the known worse-than-chance native epistemic AUROC - have
    # something to read against.
    num_classes = int(train["labels"].max()) + 1
    baseline_config = HRC_BASELINE[args.dataset]
    baseline = TAGILastLayerClassifier(
        train["features"].shape[1], num_classes, device="cuda", head="hrc",
        **baseline_config,
    )
    baseline.fit(
        train["features"], train["labels"], epochs=args.epochs,
        batch_size=BATCH_SIZE, seed=args.seed,
        validation=(validation["features"], validation["labels"]),
    )
    baseline_report = full_report(baseline, test, svhn)

    payload = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "hrc_baseline_config": baseline_config,
        "features": str(FEATURES.relative_to(REPOSITORY_ROOT)),
        "epochs": args.epochs,
        "seed": args.seed,
        "screen": screen,
        "selected": best,
        "probitree": probitree_report,
        "hrc_baseline": baseline_report,
        "feature_energy_diagnostic": {
            "probitree": feature_energy_diagnostic(winner, test, svhn),
            "hrc_baseline": feature_energy_diagnostic(baseline, test, svhn),
        },
    }
    if args.corruption_sweep:
        payload["corruption_energy_sweep"] = {
            "probitree": corruption_energy_sweep(winner, test),
            "hrc_baseline": corruption_energy_sweep(baseline, test),
        }
    output.write_text(json.dumps(payload, indent=2) + "\n")

    def show(title: str, report: dict) -> None:
        classification = report["classification"]
        print(f"\n  {title}")
        print(f"    acc {classification['accuracy'] * 100:.2f}%   "
              f"NLL {classification['nll']:.4f}   ECE {classification['ece']:.4f}   "
              f"adaECE {classification['adaptive_ece']:.4f}   "
              f"Brier {classification['brier']:.4f}")
        print(f"    AURC {classification['aurc']:.4f}   "
              f"conf {classification['mean_confidence']:.4f}")
        for score, values in report["ood_svhn"].items():
            print(f"    OOD/{score:24s} AUROC {values['auroc']:.4f}  "
                  f"AUPR-ood {values['aupr_ood']:.4f}  FPR95 {values['fpr95']:.4f}")

    show("ProbiTree", probitree_report)
    show(
        f"hrc baseline (gain_w {baseline_config['gain_w']}, "
        f"gain_b {baseline_config['gain_b']}, "
        f"sigma_v {baseline_config['sigma_v']}, {baseline_config['mean_init']})",
        baseline_report,
    )

    print("\n  what the native epistemic score is ranking")
    for name, diagnostic in payload["feature_energy_diagnostic"].items():
        print(f"    {name:14s} spearman(epistemic, ||f||^2) "
              f"{diagnostic['spearman_epistemic_vs_feature_energy']:+.5f}   "
              f"AUROC epistemic {diagnostic['auroc_native_epistemic']:.4f}   "
              f"AUROC ||f||^2 alone {diagnostic['auroc_feature_energy_alone']:.4f}")
    for name, rows in payload.get("corruption_energy_sweep", {}).items():
        larger = [r for r in rows if r["energy_gap_vs_id"] > 0]
        smaller = [r for r in rows if r["energy_gap_vs_id"] <= 0]
        corruption_set = "CIFAR-10-C" if args.dataset == "cifar10" else "CIFAR-100-C"
        print(f"\n  {corruption_set}, {name}: does the epistemic sign follow the energy gap?")
        for label, group in (("energy > ID", larger), ("energy < ID", smaller)):
            if not group:
                continue
            epistemic = sum(r["auroc_native_epistemic"] for r in group) / len(group)
            entropy = sum(r["auroc_entropy"] for r in group) / len(group)
            above = sum(r["auroc_native_epistemic"] > 0.5 for r in group)
            print(f"    {label:12s} n={len(group):2d}  mean AUROC epistemic {epistemic:.4f} "
                  f"(above 0.5 in {above}/{len(group)})   entropy {entropy:.4f}")

    print(f"\n  written to {args.output}")


if __name__ == "__main__":
    main()
