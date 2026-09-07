"""Core-Tail link against the Gaussian and logit links on frozen CIFAR features.

The Core-Tail link is designed to remove a specific empirical compromise:
section 13.5 of ``AGCI_LAST_LAYER_THEORY.md`` measures the logit link winning
on the easy bulk and the far tail and losing on near-misses, with the sign of
the net depending on how a dataset distributes mass across those strata.  The
Core-Tail link has probit sensitivity at a tie and the exact softmax tail, so
it should win, or at least not lose, in both strata at once.

This runner holds everything else fixed and changes only the categorical link:

    ``agci``        Gaussian decision noise, competitor quadrature
    ``gumbel_agci`` Gumbel decision noise integrated over the prior predictive
    ``logit_site``  softmax at the prior means, ordinary Gaussian site
    ``ct_agci``     Core-Tail at the prior means, ordinary Gaussian site

``logit_site`` is the controlled reference for ``ct_agci``: the two share every
line of the update except the link, so their difference is attributable to the
link alone and to nothing else.

Each link prefers a different prior, so the prior ratio ``kappa`` is swept for
each one separately and the comparison is made at each link's own optimum,
replicated across seeds.  Only the 10,000-example validation split is read.
The test and OOD splits are never touched.

The report adds the rank-stratified decomposition of validation NLL, using the
rank the frozen deterministic backbone's softmax assigns the observed class.
That stratification is the falsifiable part of the proposal: if the Core-Tail
link does not close the near-miss gap, the Gaussian advantage is not a property
of the static probability shape.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import (  # noqa: E402
    A_STAR,
    TAGILastLayerClassifier,
    agci_weight_gain_from_kappa,
    classification_metrics,
)
from triton_tagi.cifar_study import load_feature_shard, seed_everything  # noqa: E402

DEFAULT_STUDY_ROOT = (
    REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/last_layer/core_tail_cifar"

# Every head here is an argmax-event head with a gauge-fixed unit noise scale,
# so kappa is the whole prior description in all four cases.
HEADS = ("agci", "gumbel_agci", "logit_site", "ct_agci")
DEFAULT_KAPPAS = (0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
RANK_STRATA = ((0, 0, "rank 0"), (1, 4, "rank 1-4"), (5, 24, "rank 5-24"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def run_key(head: str, kappa: float, seed: int, a_star: float = A_STAR) -> str:
    """Return the resumable key for one configuration.

    The derived coefficient is the default, so its runs keep the shorter key
    and stay compatible with results written before the sweep existed.
    """

    suffix = "" if a_star == A_STAR else f"|a={a_star:g}"
    return f"{head}|kappa={kappa:g}|seed={seed}{suffix}"


@torch.no_grad()
def predict_probabilities(
    classifier: TAGILastLayerClassifier,
    features: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    parts = [
        classifier.predict(features[start : start + batch_size]).probabilities.cpu()
        for start in range(0, features.shape[0], batch_size)
    ]
    return torch.cat(parts)


def train_one(
    *,
    head: str,
    kappa: float,
    seed: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    feature_mean: torch.Tensor | None,
    gain_w: float,
    bias_gain: float,
    epochs: int,
    batch_size: int,
    prediction_batch_size: int,
    num_quad: int,
    device: str,
    a_star: float = A_STAR,
) -> dict:
    """Train one link at one prior ratio and return its per-epoch record."""

    seed_everything(seed)
    classifier = TAGILastLayerClassifier(
        input_dim=train_features.shape[1],
        num_classes=int(train_labels.max().item()) + 1,
        head=head,
        device=device,
        gain_w=gain_w,
        gain_b=bias_gain,
        feature_mean=None if feature_mean is None else feature_mean,
        agci_tau=1.0,
        agci_num_quad=num_quad,
        gumbel_seed=seed,
        core_tail_a_star=a_star,
    )

    generator = torch.Generator().manual_seed(seed)
    epochs_record: list[dict[str, float]] = []
    best: dict | None = None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(train_features.shape[0], generator=generator)
        for start in range(0, train_features.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            classifier.train_step(train_features[indices], train_labels[indices])
        probabilities = predict_probabilities(
            classifier, validation_features, prediction_batch_size
        )
        metrics = classification_metrics(probabilities, validation_labels)
        row = {
            "epoch": float(epoch),
            "val_nll": metrics["nll"],
            "val_accuracy": metrics["accuracy"],
            "val_ece": metrics["ece"],
            "val_brier": metrics["brier"],
            "val_confidence": metrics["mean_confidence"],
            "val_adaptive_ece": metrics["adaptive_ece"],
            "wall_s": time.perf_counter() - started,
        }
        epochs_record.append(row)
        if best is None or row["val_nll"] < best["val_nll"]:
            # Only the selected epoch's per-example log likelihoods are kept;
            # the rank decomposition needs them and nothing else does.
            best = dict(row)
            best["log_likelihood"] = (
                probabilities.gather(1, validation_labels[:, None].cpu())
                .squeeze(1)
                .clamp_min(1e-12)
                .log()
                .tolist()
            )
    assert best is not None
    return {
        "head": head,
        "kappa": kappa,
        "seed": seed,
        "a_star": a_star,
        "gain_w": gain_w,
        "epochs": epochs_record,
        "selected": best,
    }


def load_dataset(feature_root: Path, center: bool):
    train = load_feature_shard(feature_root / "train.pt")
    validation = load_feature_shard(feature_root / "validation.pt")
    train_features = train["features"].float()
    feature_mean = train_features.mean(dim=0) if center else None
    centered = train_features if feature_mean is None else train_features - feature_mean
    input_dim = train_features.shape[1]
    feature_energy = (centered.square().sum(dim=1).mean() / input_dim).item()
    return train, validation, feature_mean, feature_energy


def rank_decomposition(
    control_logits: torch.Tensor,
    labels: torch.Tensor,
    reference: list[float],
    candidate: list[float],
) -> list[dict]:
    """Return per-stratum mean NLL difference, positive favoring the candidate.

    Strata are defined by the rank the frozen deterministic backbone assigns
    the observed class, which is identical for every head and therefore an
    unbiased partition of the validation set.
    """

    order = control_logits.argsort(dim=1, descending=True)
    rank = (order == labels[:, None]).float().argmax(dim=1)
    reference_nll = -torch.tensor(reference, dtype=torch.float64)
    candidate_nll = -torch.tensor(candidate, dtype=torch.float64)
    difference = reference_nll - candidate_nll

    rows = []
    covered = torch.zeros_like(rank, dtype=torch.bool)
    for lower, upper, name in RANK_STRATA:
        mask = (rank >= lower) & (rank <= upper)
        covered |= mask
        if not bool(mask.any()):
            continue
        rows.append(
            {
                "stratum": name,
                "n": int(mask.sum()),
                # Contribution to the overall mean NLL difference, so the
                # strata sum to the total rather than each reporting a
                # conditional mean that cannot be recombined.
                "net": float(difference[mask].sum() / rank.shape[0]),
                "mean_difference": float(difference[mask].mean()),
            }
        )
    tail = ~covered
    if bool(tail.any()):
        rows.append(
            {
                "stratum": f"rank {RANK_STRATA[-1][1] + 1} and beyond",
                "n": int(tail.sum()),
                "net": float(difference[tail].sum() / rank.shape[0]),
                "mean_difference": float(difference[tail].mean()),
            }
        )
    rows.append(
        {
            "stratum": "total",
            "n": int(rank.shape[0]),
            "net": float(difference.mean()),
            "mean_difference": float(difference.mean()),
        }
    )
    return rows


def welch(first: list[float], second: list[float]) -> tuple[float, float]:
    """Return the mean difference ``first - second`` and its Welch t statistic."""

    difference = statistics.fmean(first) - statistics.fmean(second)
    if len(first) < 2 or len(second) < 2:
        return difference, float("nan")
    error = math.sqrt(
        statistics.variance(first) / len(first)
        + statistics.variance(second) / len(second)
    )
    return difference, (difference / error if error > 0.0 else float("inf"))


def command_run(args: argparse.Namespace) -> None:
    feature_root = (
        args.feature_root
        if args.feature_root is not None
        else DEFAULT_STUDY_ROOT / "features" / args.dataset
    )
    output = args.output / args.dataset
    results_path = output / "results.json"
    results = (
        json.loads(results_path.read_text()) if results_path.exists() else {}
    )

    train, validation, feature_mean, feature_energy = load_dataset(
        feature_root, args.center_features
    )
    device = args.device
    train_features = train["features"].float().to(device)
    train_labels = train["labels"].long().to(device)
    validation_features = validation["features"].float().to(device)
    validation_labels = validation["labels"].long()
    resolved_mean = None if feature_mean is None else feature_mean.to(device)
    input_dim = train_features.shape[1]

    plan: list[tuple[str, float, int]] = []
    for head in args.heads:
        for kappa in args.kappas:
            plan.append((head, kappa, args.seed))
    for head, kappa, seed in plan:
        key = run_key(head, kappa, seed)
        if key in results and not args.force:
            print(f"skip {key}")
            continue
        gain_w = agci_weight_gain_from_kappa(
            feature_energy, input_dim, kappa, tau=1.0, bias_gain=args.bias_gain
        )
        record = train_one(
            head=head,
            kappa=kappa,
            seed=seed,
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            feature_mean=resolved_mean,
            gain_w=gain_w,
            bias_gain=args.bias_gain,
            epochs=args.epochs,
            batch_size=args.batch_size,
            prediction_batch_size=args.prediction_batch_size,
            num_quad=args.num_quad,
            device=device,
        )
        results[key] = record
        atomic_json(results_path, results)
        selected = record["selected"]
        print(
            f"{key} epoch={selected['epoch']:.0f} nll={selected['val_nll']:.4f} "
            f"acc={selected['val_accuracy']:.4f} wall={selected['wall_s']:.1f}s"
        )

    # Coefficient sweep, which tests whether the derived a* is where the
    # empirical optimum actually sits or merely close to it.
    for a_star in args.a_stars:
        if a_star == A_STAR:
            continue
        for kappa in args.kappas:
            key = run_key("ct_agci", kappa, args.seed, a_star)
            if key in results and not args.force:
                print(f"skip {key}")
                continue
            gain_w = agci_weight_gain_from_kappa(
                feature_energy, input_dim, kappa, tau=1.0, bias_gain=args.bias_gain
            )
            record = train_one(
                head="ct_agci",
                kappa=kappa,
                seed=args.seed,
                train_features=train_features,
                train_labels=train_labels,
                validation_features=validation_features,
                validation_labels=validation_labels,
                feature_mean=resolved_mean,
                gain_w=gain_w,
                bias_gain=args.bias_gain,
                epochs=args.epochs,
                batch_size=args.batch_size,
                prediction_batch_size=args.prediction_batch_size,
                num_quad=args.num_quad,
                device=device,
                a_star=a_star,
            )
            results[key] = record
            atomic_json(results_path, results)
            print(f"{key} nll={record['selected']['val_nll']:.4f}")

    # Seed replication at each link's own optimum.
    if args.replicate_seeds > 1:
        for head in args.heads:
            candidates = [
                results[run_key(head, kappa, args.seed)]
                for kappa in args.kappas
                if run_key(head, kappa, args.seed) in results
            ]
            if not candidates:
                continue
            optimum = min(candidates, key=lambda row: row["selected"]["val_nll"])
            kappa = optimum["kappa"]
            for seed in range(args.seed + 1, args.seed + args.replicate_seeds):
                key = run_key(head, kappa, seed)
                if key in results and not args.force:
                    print(f"skip {key}")
                    continue
                gain_w = agci_weight_gain_from_kappa(
                    feature_energy, input_dim, kappa, tau=1.0, bias_gain=args.bias_gain
                )
                record = train_one(
                    head=head,
                    kappa=kappa,
                    seed=seed,
                    train_features=train_features,
                    train_labels=train_labels,
                    validation_features=validation_features,
                    validation_labels=validation_labels,
                    feature_mean=resolved_mean,
                    gain_w=gain_w,
                    bias_gain=args.bias_gain,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    prediction_batch_size=args.prediction_batch_size,
                    num_quad=args.num_quad,
                    device=device,
                )
                results[key] = record
                atomic_json(results_path, results)
                print(
                    f"{key} nll={record['selected']['val_nll']:.4f} "
                    f"acc={record['selected']['val_accuracy']:.4f}"
                )


def command_report(args: argparse.Namespace) -> None:
    lines = ["# The Core-Tail link on frozen CIFAR last layers", ""]
    lines.append(
        "The argmax-event construction and the prior parameterization are held "
        "fixed; only the categorical link changes. `logit_site` and `ct_agci` "
        "share every line of the update except the link, so their difference "
        "isolates it. Validation-only, best epoch by validation NLL. The CIFAR "
        "test splits are untouched."
    )
    lines.append("")

    for dataset in args.datasets:
        output = args.output / dataset
        results_path = output / "results.json"
        if not results_path.exists():
            continue
        every = json.loads(results_path.read_text())
        # Coefficient-sweep runs are reported separately; the link tables
        # compare each link at its own prior optimum with a* as derived.
        results = {
            key: row
            for key, row in every.items()
            if row.get("a_star", A_STAR) == A_STAR
        }
        feature_root = DEFAULT_STUDY_ROOT / "features" / dataset
        validation = load_feature_shard(feature_root / "validation.pt")
        control_logits = validation["logits"].float()
        labels = validation["labels"].long()

        heads = [head for head in HEADS if any(row["head"] == head for row in results.values())]
        kappas = sorted({row["kappa"] for row in results.values()})

        lines.append(f"## {dataset}: validation NLL by prior ratio")
        lines.append("")
        lines.append("| kappa | " + " | ".join(f"`{head}`" for head in heads) + " |")
        lines.append("|---|" + "---|" * len(heads))
        for kappa in kappas:
            cells = []
            for head in heads:
                key = run_key(head, kappa, args.seed)
                cells.append(
                    f"{results[key]['selected']['val_nll']:.4f}"
                    if key in results
                    else "--"
                )
            lines.append(f"| {kappa:g} | " + " | ".join(cells) + " |")
        lines.append("")

        optima: dict[str, dict] = {}
        replicates: dict[str, list[dict]] = {}
        for head in heads:
            candidates = [
                results[run_key(head, kappa, args.seed)]
                for kappa in kappas
                if run_key(head, kappa, args.seed) in results
            ]
            optimum = min(candidates, key=lambda row: row["selected"]["val_nll"])
            optima[head] = optimum
            replicates[head] = [
                row
                for row in results.values()
                if row["head"] == head and row["kappa"] == optimum["kappa"]
            ]

        lines.append("### Seed replication at each link's optimum")
        lines.append("")
        lines.append(
            "| link | kappa | n | NLL mean | NLL sd | top-1 mean | ACE mean | "
            "ECE mean | Brier mean | conf gap pp | epoch |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for head in heads:
            rows = replicates[head]
            nlls = [row["selected"]["val_nll"] for row in rows]
            accuracies = [row["selected"]["val_accuracy"] for row in rows]
            eces = [row["selected"]["val_adaptive_ece"] for row in rows]
            binned = [row["selected"]["val_ece"] for row in rows]
            briers = [row["selected"]["val_brier"] for row in rows]
            gaps = [
                row["selected"]["val_confidence"] - row["selected"]["val_accuracy"]
                for row in rows
            ]
            epochs = sorted({int(row["selected"]["epoch"]) for row in rows})
            sd = statistics.stdev(nlls) if len(nlls) > 1 else float("nan")
            lines.append(
                f"| `{head}` | {optima[head]['kappa']:g} | {len(rows)} | "
                f"{statistics.fmean(nlls):.4f} | {sd:.5f} | "
                f"{100 * statistics.fmean(accuracies):.2f}% | "
                f"{statistics.fmean(eces):.4f} | "
                f"{statistics.fmean(binned):.4f} | "
                f"{statistics.fmean(briers):.4f} | "
                f"{100 * statistics.fmean(gaps):+.2f} | "
                + ", ".join(str(value) for value in epochs)
                + " |"
            )
        lines.append("")

        if args.baseline in heads and args.candidate in heads:
            first = [row["selected"]["val_nll"] for row in replicates[args.baseline]]
            second = [row["selected"]["val_nll"] for row in replicates[args.candidate]]
            difference, statistic = welch(first, second)
            verdict = args.candidate if difference > 0 else args.baseline
            lines.append(
                f"`{args.baseline}` minus `{args.candidate}`: {difference:+.4f} NLL, "
                f"t {statistic:+.1f} -> **{verdict}** is better."
            )
            lines.append("")
            # The link cannot reorder classes within a row, so accuracy is
            # reported to show it does not move rather than to claim it does.
            lines.append(
                "| metric | "
                f"`{args.baseline}` | `{args.candidate}` | difference | t |"
            )
            lines.append("|---|---|---|---|---|")
            for name, key, scale, digits in (
                ("NLL", "val_nll", 1.0, 5),
                ("top-1 %", "val_accuracy", 100.0, 3),
                ("ACE", "val_adaptive_ece", 1.0, 5),
                ("ECE", "val_ece", 1.0, 5),
                ("Brier", "val_brier", 1.0, 5),
                ("conf gap pp", "val_confidence", 100.0, 3),
            ):
                first = [scale * row["selected"][key] for row in replicates[args.baseline]]
                second = [
                    scale * row["selected"][key] for row in replicates[args.candidate]
                ]
                if key == "val_confidence":
                    first = [
                        value - 100.0 * row["selected"]["val_accuracy"]
                        for value, row in zip(first, replicates[args.baseline], strict=True)
                    ]
                    second = [
                        value - 100.0 * row["selected"]["val_accuracy"]
                        for value, row in zip(second, replicates[args.candidate], strict=True)
                    ]
                delta, score = welch(first, second)
                lines.append(
                    f"| {name} | {statistics.fmean(first):.{digits}f} | "
                    f"{statistics.fmean(second):.{digits}f} | {delta:+.{digits}f} | "
                    f"{score:+.1f} |"
                )
            lines.append("")

        lines.append("### Validation NLL by control-softmax rank of the observed class")
        lines.append("")
        lines.append(
            f"Positive entries favor `{args.candidate}` over the named link. "
            "Entries are contributions to the overall mean difference, so each "
            "column sums to its total."
        )
        lines.append("")
        comparisons = [head for head in heads if head != args.candidate]
        strata = [name for _, _, name in RANK_STRATA]
        strata.append(f"rank {RANK_STRATA[-1][1] + 1} and beyond")
        strata.append("total")
        lines.append(
            "| stratum | n | " + " | ".join(f"vs `{head}`" for head in comparisons) + " |"
        )
        lines.append("|---|---|" + "---|" * len(comparisons))
        tables = {
            head: {
                row["stratum"]: row
                for row in rank_decomposition(
                    control_logits,
                    labels,
                    optima[head]["selected"]["log_likelihood"],
                    optima[args.candidate]["selected"]["log_likelihood"],
                )
            }
            for head in comparisons
        }
        for name in strata:
            present = [tables[head][name] for head in comparisons if name in tables[head]]
            if not present:
                continue
            count = present[0]["n"]
            cells = [
                f"{tables[head][name]['net']:+.4f}" if name in tables[head] else "--"
                for head in comparisons
            ]
            lines.append(f"| {name} | {count} | " + " | ".join(cells) + " |")
        lines.append("")

        coefficients = sorted({row["a_star"] for row in every.values() if "a_star" in row})
        if len(coefficients) > 1:
            lines.append("### Core-Tail coefficient sweep")
            lines.append("")
            lines.append(
                f"The derived coefficient is a* = {A_STAR:.6f}; a* = 0 is exactly "
                "`logit_site`. The sweep is reported only to test whether the "
                "derivation lands where the empirical optimum is, not to select "
                "a value."
            )
            lines.append("")
            sweep_kappas = sorted(
                {
                    row["kappa"]
                    for row in every.values()
                    if row["head"] == "ct_agci" and row.get("a_star") != A_STAR
                }
            )
            lines.append(
                "| a* | " + " | ".join(f"kappa {value:g}" for value in sweep_kappas) + " |"
            )
            lines.append("|---|" + "---|" * len(sweep_kappas))
            for a_star in coefficients:
                cells = []
                for kappa in sweep_kappas:
                    key = run_key("ct_agci", kappa, args.seed, a_star)
                    cells.append(
                        f"{every[key]['selected']['val_nll']:.4f}" if key in every else "--"
                    )
                marker = " (derived)" if a_star == A_STAR else ""
                lines.append(f"| {a_star:.4f}{marker} | " + " | ".join(cells) + " |")
            lines.append("")

    destination = args.output / "RESULTS.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    print(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    runner = subparsers.add_parser("run", help="sweep kappa for each link")
    runner.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    runner.add_argument("--feature-root", type=Path)
    runner.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    runner.add_argument(
        "--device",
        default="cuda",
        help=(
            "the Triton kernels launch on the current CUDA device, so an "
            "indexed device such as cuda:1 does not work; select a second "
            "GPU with CUDA_VISIBLE_DEVICES instead"
        ),
    )
    runner.add_argument("--heads", nargs="+", choices=HEADS, default=list(HEADS))
    runner.add_argument("--kappas", nargs="+", type=float, default=list(DEFAULT_KAPPAS))
    runner.add_argument(
        "--a-stars",
        nargs="+",
        type=float,
        default=[],
        help=(
            "extra Core-Tail coefficients to sweep alongside the derived "
            "value, which always runs; a_star = 0 is exactly the logit site"
        ),
    )
    runner.add_argument("--epochs", type=int, default=5)
    runner.add_argument("--batch-size", type=int, default=256)
    runner.add_argument("--prediction-batch-size", type=int, default=1024)
    runner.add_argument("--num-quad", type=int, default=48)
    runner.add_argument("--bias-gain", type=float, default=1.0)
    runner.add_argument("--seed", type=int, default=0)
    runner.add_argument(
        "--replicate-seeds",
        type=int,
        default=4,
        help="total seeds at each link's optimum, including the sweep seed",
    )
    runner.add_argument(
        "--no-center-features", dest="center_features", action="store_false"
    )
    runner.set_defaults(center_features=True)
    runner.add_argument("--force", action="store_true")
    runner.set_defaults(handler=command_run)

    reporter = subparsers.add_parser("report", help="write the markdown summary")
    reporter.add_argument(
        "--datasets", nargs="+", default=["cifar10", "cifar100"]
    )
    reporter.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    reporter.add_argument("--seed", type=int, default=0)
    reporter.add_argument("--candidate", default="ct_agci")
    reporter.add_argument("--baseline", default="logit_site")
    reporter.set_defaults(handler=command_report)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
