"""Hierarchical probit calibration on a frozen CIFAR-10 last layer.

Tests the calibration module of Goulet, Nguyen and Florensa-Montilla on the
hierarchical-probit head of ``run_hrc_calibration.py``. One head is trained per
tree labelling on the train split; everything else is post-hoc on a frozen
network, so every arm reads the same forward summaries and the comparison
isolates the calibration.

Two axes are crossed:

  * **Gain sharing.** One global gain, one per tree level, or one per internal
    node. A global gain is the positive-gain analogue of temperature scaling;
    the finer groupings are more flexible and need more calibration data.
  * **Calibration size.** The held-out split is subsampled, which is where the
    gain uncertainty stops being a rounding error and where deep per-node gains
    become weakly identified. The visit count of every group is reported with
    its posterior.

Ablations against those arms: the fitted log-gain read as a point value, which
drops the uncertainty and leaves ordinary probit temperature scaling; the
Laplace posterior in place of the normalized grid; sequential assumed-density
filtering in two stream orders, which is the ordering check the approximation
requires; the existing NLL-fitted latent scale of ``hrc_probit``; and softmax
temperature scaling on a backprop head.

The train split fits the heads, the validation split calibrates, and the test
split is read once per record.

Examples:
  python experiments/last_layer/run_hsm_calibration.py run
  python experiments/last_layer/run_hsm_calibration.py run --calibration-sizes 1000 10000
  python experiments/last_layer/run_hsm_calibration.py report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import (  # noqa: E402
    LogGainPosterior,
    TAGILastLayerClassifier,
    calibrate_hsm_log_gain_adf,
    classification_metrics,
    fit_hrc_log_tau,
    fit_hsm_log_gain,
    fit_softmax_temperature,
    gain_groups,
    hsm_class_moments,
    probability_simplex_deviation,
)
from triton_tagi.hrc_probit import hrc_log_probs  # noqa: E402

DEFAULT_FEATURE_ROOT = (
    REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k/features/cifar10"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "runs/last_layer/hsm_calibration/cifar10"

# Default head gain: the one run_hrc_calibration.py selected on CIFAR-10
# validation NLL for probit_full. It is a prior scale, not a dimensionless
# rate, so transferring it to another dataset is an assumption; --head-gain
# overrides it and stage_run records whatever was used.
HEAD_GAIN = 0.1
PRIOR_LOG_GAIN_VARIANCE = 4.0

# The gain is measured against the fixed noise of the branch channel, which
# hsm_calibration carries on the belief. The base HRC head observes +/-1 at
# each node under an observation noise, so that noise is its channel and
# lambda = 0 is the head as trained; the probit head fixes its latent unit
# instead, which is a channel of one.
#
# Two channels are available for the base HRC head, and they are different
# models rather than two parameterizations of one:
#
#   sigma_v     The noise the head was actually trained with. This is the
#               formulation of the paper instantiated on the base HRC: the
#               calibrated model and the training channel are the same model.
#   convention  cuTAGI's obs_to_class_probs reads every HRC head out at a
#               hard-coded alpha = 3 whatever it trained with, which is a
#               channel of 1/3. Kept because the earlier records were produced
#               against it and because it is what cuTAGI itself would report.
CONVENTION_ALPHA = {"hrc": 3.0, "hrc_probit": 1.0}
LATENT_SCALES = ("sigma_v", "convention")
CALIBRATION_SIZES = (100, 300, 1000, 3000, 10000)
SUBSAMPLE_SEEDS = (0, 1, 2)
SOFTMAX_LEARNING_RATE = 0.01
QUADRATURE_ORDER = 20
# None lets hsm_calibration size its blocks from the tree width, which is about
# four times faster than a fixed 2048 rows on a ten-leaf tree. Chunking is
# exact, so this changes only the runtime.
PREDICTION_CHUNK = None

REPORT_COLUMNS = (
    "arm",
    "role",
    "sharing",
    "method",
    "gain_uncertainty",
    "calibration_size",
    "records",
    "test_nll",
    "test_nll_sd",
    "test_accuracy",
    "test_brier",
    "test_ece",
    "test_adaptive_ece",
    "test_classwise_ece",
    "test_simplex_deviation",
    "test_epistemic_share",
    "log_gain_mean_first",
    "log_gain_sd_max",
    "visits_min",
    "calibration_nll",
)


# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationArm:
    """One way of turning a calibration split into a prediction-time gain."""

    name: str
    sharing: str | None
    method: str
    gain_uncertainty: bool
    role: str


CALIBRATION_ARMS = (
    CalibrationArm("uncalibrated", None, "none", False, "main"),
    CalibrationArm("hsm_global", "global", "grid", True, "main"),
    CalibrationArm("hsm_level", "level", "grid", True, "main"),
    CalibrationArm("hsm_node", "node", "grid", True, "main"),
    CalibrationArm("hsm_global_point", "global", "grid", False, "ablation"),
    CalibrationArm("hsm_level_point", "level", "grid", False, "ablation"),
    CalibrationArm("hsm_node_point", "node", "grid", False, "ablation"),
    CalibrationArm("hsm_global_laplace", "global", "laplace", True, "ablation"),
    CalibrationArm("hsm_global_adf", "global", "adf", True, "ablation"),
    CalibrationArm("hsm_global_adf_reversed", "global", "adf_reversed", True, "ablation"),
    CalibrationArm("hsm_node_adf", "node", "adf", True, "ablation"),
    CalibrationArm("hrc_fitted_scale", None, "nll_scale", False, "ablation"),
    CalibrationArm("softmax_temperature", None, "temperature", False, "ablation"),
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


# ──────────────────────────────────────────────────────────────────────────────
#  Data
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Split:
    features: Tensor
    labels: Tensor


def load_splits(feature_root: Path, device: str) -> dict[str, Split]:
    """Load the cached frozen ResNet-18 feature shards."""

    splits: dict[str, Split] = {}
    for name in ("train", "validation", "test"):
        payload = torch.load(feature_root / f"{name}.pt", map_location="cpu", weights_only=False)
        if "features" not in payload or "labels" not in payload:
            raise SystemExit(f"feature shard {name}.pt is missing features or labels")
        splits[name] = Split(
            payload["features"].to(device).contiguous(),
            payload["labels"].to(device).long().contiguous(),
        )
    return splits


def class_permutation(num_classes: int, seed: int, device: str) -> Tensor:
    """Return the class-to-leaf assignment; seed zero is the identity."""

    if seed == 0:
        return torch.arange(num_classes, device=device)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_classes, generator=generator).to(device)


def calibration_subset(size: int, total: int, seed: int) -> Tensor:
    """Return a seeded row index of the requested calibration size."""

    if size >= total:
        return torch.arange(total)
    generator = torch.Generator().manual_seed(1000 + seed)
    return torch.randperm(total, generator=generator)[:size].sort().values


# ──────────────────────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────────────────────


def train_probit_head(
    splits: dict[str, Split],
    permutation: Tensor,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    head_gain: float = HEAD_GAIN,
    head: str = "hrc_probit",
    sigma_v: float | None = None,
    prior_offsets: bool | None = None,
) -> TAGILastLayerClassifier:
    """Fit one hierarchical head on the proper K-leaf tree.

    Both heads train on the same tree so that the leaf products sum to one and
    the calibration never needs a normalizer. The Gaussian +/-1 head never sees
    a branch offset during training, so it defaults to the offset-free tree;
    the probit head models the offset and keeps it.

    Args:
        splits: Feature splits; only ``train`` is read.
        permutation: Class-to-leaf assignment.
        epochs: Training epochs.
        batch_size: Rows per update.
        seed: Seed for initialization and shuffling.
        device: Torch device.
        head_gain: Prior gain on the weights and bias.
        head: ``hrc_probit`` or ``hrc``.
        sigma_v: Observation noise for the Gaussian head; rejected by the
            probit head, which has no observation variance.
        prior_offsets: Branch offsets; ``None`` keeps them only for the probit
            head, which is the only one that trains against them.

    Returns:
        The fitted classifier.
    """

    if head not in CONVENTION_ALPHA:
        raise SystemExit(f"--head must be one of {sorted(CONVENTION_ALPHA)}")
    if head == "hrc" and sigma_v is None:
        raise SystemExit("the hrc head needs --sigma-v")
    if head == "hrc_probit" and sigma_v is not None:
        raise SystemExit("the hrc_probit head fixes its latent scale and takes no --sigma-v")
    offsets = (head == "hrc_probit") if prior_offsets is None else bool(prior_offsets)

    seed_everything(seed)
    classifier = TAGILastLayerClassifier(
        splits["train"].features.shape[1],
        int(permutation.shape[0]),
        head=head,
        hrc_tree="full",
        hrc_prior_offsets=offsets,
        device=device,
        gain_w=head_gain,
        gain_b=head_gain,
        sigma_v=sigma_v,
    )
    classifier.fit(
        splits["train"].features,
        permutation[splits["train"].labels],
        epochs=epochs,
        batch_size=batch_size,
        sigma_v=sigma_v,
        seed=seed,
        record_initial=False,
    )
    return classifier


def train_softmax_head(
    splits: dict[str, Split],
    *,
    num_classes: int,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> torch.nn.Linear:
    """Fit the conventional backprop cross-entropy reference head."""

    seed_everything(seed)
    head = torch.nn.Linear(splits["train"].features.shape[1], num_classes).to(device)
    optimizer = torch.optim.SGD(
        head.parameters(), lr=SOFTMAX_LEARNING_RATE, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    features, labels = splits["train"].features, splits["train"].labels
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(features.shape[0], generator=generator).to(device)
        for start in range(0, order.shape[0], batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(features[index]), labels[index])
            loss.backward()
            optimizer.step()
        scheduler.step()
    return head


# ──────────────────────────────────────────────────────────────────────────────
#  Calibration and scoring
# ──────────────────────────────────────────────────────────────────────────────


def restore_class_order(values: Tensor, permutation: Tensor) -> Tensor:
    """Map leaf-indexed columns back to class-indexed columns."""

    return values.index_select(1, permutation)


def probability_metrics(probabilities: Tensor, labels: Tensor) -> dict[str, float]:
    metrics = classification_metrics(probabilities, labels)
    keys = (
        "accuracy",
        "nll",
        "brier",
        "ece",
        "adaptive_ece",
        "classwise_ece",
        "mean_confidence",
    )
    scored = {key: metrics[key] for key in keys}
    scored["simplex_deviation"] = probability_simplex_deviation(probabilities)
    return scored


def channel_sigma_v(args: argparse.Namespace) -> float:
    """Return the fixed branch-channel noise this head is calibrated against.

    The probit head fixes its latent unit, so its channel is one whatever is
    asked for. The base HRC head has a real observation noise, and
    ``--latent-scale`` chooses whether to read it at that noise or at the
    ``1 / alpha`` cuTAGI hard-codes.
    """

    if args.head == "hrc_probit":
        return 1.0
    if args.latent_scale == "sigma_v":
        return float(args.sigma_v)
    return 1.0 / CONVENTION_ALPHA[args.head]


def uncalibrated_posterior(hrc, sigma_v: float, device: str) -> LogGainPosterior:
    """Return the deterministic belief that is the frozen head itself.

    ``lambda = 0`` with ``q = 0`` at the head's own channel: the class moments
    then reproduce its native readout to double precision, which is
    ``obs_to_class_probs`` at ``alpha = 1 / sigma_v`` for the base HRC head and
    the unit-scale probit readout for ``hrc_probit``.
    """

    return LogGainPosterior.prior(
        gain_groups(hrc, "global", device=device),
        mean=0.0,
        variance=0.0,
        sigma_v=sigma_v,
    )


def fit_log_gain(
    arm: CalibrationArm,
    moments: tuple[Tensor, Tensor],
    labels: Tensor,
    hrc,
    sigma_v: float,
) -> LogGainPosterior:
    """Return the log-gain belief this arm reads off the calibration split.

    The prior is centred at zero against the head's own channel, so a group
    with no calibration data falls back on the head exactly as trained.
    """

    if arm.method in ("grid", "laplace"):
        return fit_hsm_log_gain(
            *moments,
            labels,
            hrc,
            sharing=arm.sharing or "global",
            prior_mean=0.0,
            prior_variance=PRIOR_LOG_GAIN_VARIANCE,
            sigma_v=sigma_v,
            method=arm.method,
            chunk_size=PREDICTION_CHUNK,
        )
    reversed_stream = arm.method == "adf_reversed"
    order = torch.arange(labels.shape[0], device=labels.device)
    return calibrate_hsm_log_gain_adf(
        *moments,
        labels,
        hrc,
        sharing=arm.sharing or "global",
        prior_mean=0.0,
        prior_variance=PRIOR_LOG_GAIN_VARIANCE,
        sigma_v=sigma_v,
        order=QUADRATURE_ORDER,
        visit_order=order.flip(0) if reversed_stream else order,
    )


def score_hsm_arm(
    arm: CalibrationArm,
    posterior: LogGainPosterior,
    moments: dict[str, tuple[Tensor, Tensor]],
    splits: dict[str, Split],
    calibration_labels: Tensor,
    permutation: Tensor,
    hrc,
) -> dict[str, Any]:
    """Score one calibrated belief on the calibration and test splits."""

    belief = posterior if arm.gain_uncertainty else posterior.deterministic()
    record: dict[str, Any] = {
        "log_gain_mean": belief.mean.tolist(),
        "log_gain_variance": belief.variance.tolist(),
        "gain_median": belief.gain_median.tolist(),
        "visits": (belief.visits.tolist() if belief.visits is not None else None),
        "n_groups": belief.groups.n_groups,
    }
    calibration_class_moments = hsm_class_moments(
        *moments["calibration"], hrc, belief, order=QUADRATURE_ORDER, chunk_size=PREDICTION_CHUNK
    )
    picked = calibration_class_moments.mean.gather(1, calibration_labels[:, None])
    record["calibration_nll"] = float(-picked.clamp_min(1e-300).log().mean())

    class_moments = hsm_class_moments(
        *moments["test"], hrc, belief, order=QUADRATURE_ORDER, chunk_size=PREDICTION_CHUNK
    )
    probabilities = restore_class_order(class_moments.mean, permutation)
    scored = probability_metrics(probabilities, splits["test"].labels)
    record.update({f"test_{key}": value for key, value in scored.items()})
    # The epistemic share is v_Q / [mu_Q (1 - mu_Q)] at the predicted leaf: the
    # fraction of the class indicator's variance that is dispersion of the
    # latent probability rather than residual label randomness.
    top = class_moments.mean.argmax(dim=1, keepdim=True)
    top_mean = class_moments.mean.gather(1, top).squeeze(1)
    top_variance = class_moments.variance.gather(1, top).squeeze(1)
    total = (top_mean * (1.0 - top_mean)).clamp_min(1e-12)
    record["test_class_dispersion"] = float(top_variance.sqrt().mean())
    record["test_epistemic_share"] = float((top_variance / total).mean())
    record["test_partition_deviation"] = float(
        (class_moments.mean.sum(dim=-1) - 1.0).abs().max()
    )
    return record


def score_reference_arm(
    arm: CalibrationArm,
    moments: dict[str, tuple[Tensor, Tensor]],
    splits: dict[str, Split],
    calibration_labels: Tensor,
    calibration_index: Tensor,
    permutation: Tensor,
    hrc,
    softmax_logits: dict[str, Tensor],
    sigma_v: float,
) -> dict[str, Any]:
    """Score the two non-HSM references on the same calibration subsample.

    The tree arm is fitted on leaf-indexed labels and its scores are mapped
    back to class order; the softmax arm has no tree, so it is fitted on the
    class labels themselves.
    """

    record: dict[str, Any] = {}
    calibration_target = splits["validation"].labels[calibration_index]
    if arm.method == "nll_scale":
        log_tau = fit_hrc_log_tau(*moments["calibration"], calibration_labels, hrc)
        # hrc_probit's scale is tau; against a channel of sigma_v the same
        # readout is the gain sigma_v / tau, so the two are comparable only
        # after that shift.
        record["log_gain_mean"] = [math.log(sigma_v) - log_tau]
        record["log_gain_variance"] = [0.0]
        record["gain_median"] = [sigma_v * math.exp(-log_tau)]
        record["n_groups"] = 1
        scores = {
            name: restore_class_order(
                hrc_log_probs(*moments[name], hrc, log_tau=log_tau).exp(), permutation
            )
            for name in ("calibration", "test")
        }
    else:
        temperature = fit_softmax_temperature(
            softmax_logits["validation"][calibration_index], calibration_target
        )
        record["temperature"] = temperature
        record["n_groups"] = 1
        scores = {
            "calibration": torch.softmax(
                softmax_logits["validation"][calibration_index] / temperature, dim=1
            ),
            "test": torch.softmax(softmax_logits["test"] / temperature, dim=1),
        }
    picked = scores["calibration"].gather(1, calibration_target[:, None])
    record["calibration_nll"] = float(-picked.clamp_min(1e-300).log().mean())
    record.update(
        {
            f"test_{key}": value
            for key, value in probability_metrics(scores["test"], splits["test"].labels).items()
        }
    )
    return record


# ──────────────────────────────────────────────────────────────────────────────
#  Stages
# ──────────────────────────────────────────────────────────────────────────────


RECORD_IDENTITY = ("arm", "permutation_seed", "subsample_seed", "calibration_size")
# The head fields an appended run has to agree on. A record written before a
# field existed simply does not carry it, and is not evidence of a conflict.
APPEND_INVARIANTS = (
    "head",
    "head_gain",
    "sigma_v",
    "prior_offsets",
    "latent_scale",
    "channel_sigma_v",
    "epochs",
)


def load_previous_records(
    args: argparse.Namespace, offsets: bool, sigma_v: float
) -> list[dict[str, Any]]:
    """Return the records already in the output directory, checked for agreement.

    Appending is only meaningful when the existing records came off the same
    frozen head, so every head field both sides carry has to match.

    Args:
        args: Parsed arguments; ``output`` names the directory to extend.
        offsets: The branch-offset setting this run resolved to.
        sigma_v: The branch-channel noise this run resolved to.

    Returns:
        The existing records, or an empty list when there are none.

    Raises:
        SystemExit: If an existing record disagrees on a head field.
    """

    path = args.output / "records.json"
    if not path.exists():
        return []
    previous = json.loads(path.read_text())
    current = {
        "head": args.head,
        "head_gain": args.head_gain,
        "sigma_v": args.sigma_v,
        "prior_offsets": offsets,
        "latent_scale": args.latent_scale,
        "channel_sigma_v": sigma_v,
        "epochs": args.epochs,
    }
    for record in previous:
        for field in APPEND_INVARIANTS:
            if field in record and record[field] != current[field]:
                raise SystemExit(
                    f"--append refused: {path} holds {field}={record[field]!r} "
                    f"and this run would write {current[field]!r}"
                )
    unchecked = sorted(
        field for field in APPEND_INVARIANTS if not any(field in item for item in previous)
    )
    if unchecked:
        print(f"warning: {path} predates {unchecked}, so those cannot be checked for agreement")
    return previous


def merge_records(
    previous: list[dict[str, Any]], fresh: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return the previous records with every rerun cell replaced by a fresh one."""

    def identity(record: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(record[name] for name in RECORD_IDENTITY)

    replaced = {identity(record) for record in fresh}
    return [record for record in previous if identity(record) not in replaced] + fresh


def stage_run(args: argparse.Namespace) -> None:
    """Train one head per labelling and score every arm at every split size."""

    splits = load_splits(args.feature_root, args.device)
    num_classes = int(splits["train"].labels.max()) + 1
    validation_rows = splits["validation"].features.shape[0]
    sizes = [size for size in args.calibration_sizes if size <= validation_rows]
    if not sizes:
        raise SystemExit("no calibration size fits the validation split")

    sigma_v = channel_sigma_v(args)
    resolved_offsets = (args.head == "hrc_probit") if args.prior_offsets is None else bool(
        args.prior_offsets
    )
    previous = load_previous_records(args, resolved_offsets, sigma_v) if args.append else []
    if previous:
        print(f"appending to {len(previous)} existing records")

    records: list[dict[str, Any]] = []
    for permutation_seed in args.permutation_seeds:
        permutation = class_permutation(num_classes, permutation_seed, args.device)
        started = time.perf_counter()
        classifier = train_probit_head(
            splits,
            permutation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            head_gain=args.head_gain,
            head=args.head,
            sigma_v=args.sigma_v,
            prior_offsets=args.prior_offsets,
        )
        assert classifier.hrc is not None
        hrc = classifier.hrc
        training_seconds = time.perf_counter() - started
        node_moments = {
            name: classifier.hrc_node_moments(
                splits[name].features, batch_size=args.prediction_batch_size
            )
            for name in ("validation", "test")
        }
        softmax_head = train_softmax_head(
            splits,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        with torch.no_grad():
            softmax_logits = {
                name: softmax_head(splits[name].features).double()
                for name in ("validation", "test")
            }

        for size in sizes:
            seeds = SUBSAMPLE_SEEDS if size < validation_rows else (0,)
            for subsample_seed in seeds:
                index = calibration_subset(size, validation_rows, subsample_seed).to(args.device)
                calibration_moments = tuple(
                    tensor.index_select(0, index) for tensor in node_moments["validation"]
                )
                calibration_labels = permutation[splits["validation"].labels[index]]
                moments = {
                    "calibration": calibration_moments,
                    "test": node_moments["test"],
                }
                common = {
                    "permutation_seed": permutation_seed,
                    "subsample_seed": subsample_seed,
                    "calibration_size": int(index.shape[0]),
                    "num_classes": num_classes,
                    "tree_nodes": hrc.len,
                    "head": args.head,
                    "head_gain": args.head_gain,
                    "sigma_v": args.sigma_v,
                    "prior_offsets": classifier.hrc_prior_offsets,
                    "epochs": args.epochs,
                    "training_seconds": training_seconds,
                    "latent_scale": args.latent_scale,
                    "channel_sigma_v": sigma_v,
                    "convention_alpha": 1.0 / sigma_v,
                    "prior_log_gain_mean": 0.0,
                    "prior_log_gain_variance": PRIOR_LOG_GAIN_VARIANCE,
                }
                for arm in CALIBRATION_ARMS:
                    if arm.method == "none":
                        scored = score_hsm_arm(
                            arm,
                            uncalibrated_posterior(hrc, sigma_v, args.device),
                            moments,
                            splits,
                            calibration_labels,
                            permutation,
                            hrc,
                        )
                    elif arm.method in ("nll_scale", "temperature"):
                        scored = score_reference_arm(
                            arm,
                            moments,
                            splits,
                            calibration_labels,
                            index,
                            permutation,
                            hrc,
                            softmax_logits,
                            sigma_v,
                        )
                    else:
                        posterior = fit_log_gain(
                            arm, calibration_moments, calibration_labels, hrc, sigma_v
                        )
                        scored = score_hsm_arm(
                            arm,
                            posterior,
                            moments,
                            splits,
                            calibration_labels,
                            permutation,
                            hrc,
                        )
                    records.append(
                        {
                            **common,
                            "arm": arm.name,
                            "role": arm.role,
                            "sharing": arm.sharing or "none",
                            "method": arm.method,
                            "gain_uncertainty": arm.gain_uncertainty,
                            **scored,
                        }
                    )
                    print(
                        f"perm={permutation_seed} n_cal={index.shape[0]:>5d} "
                        f"sub={subsample_seed} {arm.name:24s} "
                        f"test_nll={records[-1]['test_nll']:.4f} "
                        f"acc={records[-1]['test_accuracy']:.4f} "
                        f"ece={records[-1]['test_ece']:.4f}",
                        flush=True,
                    )
        atomic_json(args.output / "records.json", merge_records(previous, records))
    print(f"\nwrote {args.output / 'records.json'}")


def summarize(values: list[float]) -> tuple[float | str, float | str]:
    """Return the mean and the sample standard deviation of a metric."""

    numeric = [value for value in values if isinstance(value, (int, float))]
    if not numeric:
        return "", ""
    centre = sum(numeric) / len(numeric)
    if len(numeric) < 2:
        return centre, ""
    spread = math.sqrt(sum((value - centre) ** 2 for value in numeric) / (len(numeric) - 1))
    return centre, spread


def stage_report(args: argparse.Namespace) -> None:
    """Aggregate the records over labellings and subsamples into one CSV."""

    path = args.output / "records.json"
    if not path.exists():
        raise SystemExit(f"run first: {path} is missing")
    records = json.loads(path.read_text())

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["arm"], record["calibration_size"]), []).append(record)

    rows: list[dict[str, Any]] = []
    for (arm, size), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        first = group[0]
        centre, spread = summarize([item["test_nll"] for item in group])
        row: dict[str, Any] = {
            "arm": arm,
            "role": first["role"],
            "sharing": first["sharing"],
            "method": first["method"],
            "gain_uncertainty": first["gain_uncertainty"],
            "calibration_size": size,
            "records": len(group),
            "test_nll": centre,
            "test_nll_sd": spread,
        }
        for name in (
            "test_accuracy",
            "test_brier",
            "test_ece",
            "test_adaptive_ece",
            "test_classwise_ece",
            "test_simplex_deviation",
            "test_epistemic_share",
            "calibration_nll",
        ):
            row[name] = summarize([item.get(name) for item in group])[0]
        beliefs = [item.get("log_gain_mean") for item in group if item.get("log_gain_mean")]
        variances = [
            item.get("log_gain_variance") for item in group if item.get("log_gain_variance")
        ]
        visits = [item.get("visits") for item in group if item.get("visits")]
        row["log_gain_mean_first"] = (
            sum(item[0] for item in beliefs) / len(beliefs) if beliefs else ""
        )
        row["log_gain_sd_max"] = (
            sum(math.sqrt(max(item)) for item in variances) / len(variances) if variances else ""
        )
        row["visits_min"] = sum(min(item) for item in visits) / len(visits) if visits else ""
        rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / "report.csv"
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REPORT_COLUMNS))
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in REPORT_COLUMNS} for row in rows)
    atomic_json(args.output / "report.json", rows)

    def cell(value: Any, width: int, spec: str) -> str:
        return format(value, spec) if isinstance(value, (int, float)) else "-".rjust(width)

    header = (
        f"{'n_cal':>6} {'arm':24s} {'test NLL':>9} {'+/-':>7} {'acc':>7} "
        f"{'ECE':>7} {'sd(L)':>7} {'epist':>7}"
    )
    for role in ("main", "ablation"):
        print(f"\n{role.upper()}")
        print(header)
        print("-" * len(header))
        for row in rows:
            if row["role"] != role:
                continue
            print(
                f"{row['calibration_size']:>6} {row['arm']:24s} "
                f"{cell(row['test_nll'], 9, '>9.4f')} "
                f"{cell(row['test_nll_sd'], 7, '>7.4f')} "
                f"{cell(row['test_accuracy'], 7, '>7.4f')} "
                f"{cell(row['test_ece'], 7, '>7.4f')} "
                f"{cell(row['log_gain_sd_max'], 7, '>7.4f')} "
                f"{cell(row['test_epistemic_share'], 7, '>7.4f')}"
            )
    print(f"\nwrote {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run", "report"])
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head-gain", type=float, default=HEAD_GAIN)
    parser.add_argument("--head", default="hrc_probit", choices=sorted(CONVENTION_ALPHA))
    parser.add_argument(
        "--sigma-v",
        type=float,
        default=None,
        help="Observation noise for the hrc head; the hrc_probit head takes none.",
    )
    parser.add_argument(
        "--latent-scale",
        default="sigma_v",
        choices=LATENT_SCALES,
        help=(
            "Fixed branch-channel noise the gain is measured against, for the "
            "hrc head. 'sigma_v' is the noise it was trained with, which is the "
            "paper's formulation on the base HRC head; 'convention' is 1 / 3, "
            "the alpha cuTAGI hard-codes into its readout. The hrc_probit head "
            "fixes its latent unit and ignores this."
        ),
    )
    parser.add_argument(
        "--prior-offsets",
        type=lambda value: value.lower() in ("1", "true", "yes"),
        default=None,
        help="Branch offsets; defaults to on for hrc_probit and off for hrc.",
    )
    parser.add_argument(
        "--calibration-sizes",
        type=int,
        nargs="+",
        default=list(CALIBRATION_SIZES),
        help="Held-out calibration sizes to sweep.",
    )
    parser.add_argument(
        "--permutation-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Tree labellings to average over; seed zero is the identity.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help=(
            "Merge into the records already in --output instead of replacing "
            "them, which is how a subset of --arms extends an existing sweep. "
            "Cells that repeat are overwritten by the fresh run."
        ),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help=(
            "Arms to run, by name; defaults to all. The three ADF arms are a "
            "sequential Python loop and dominate the runtime at large K, so "
            "dropping them is the usual way to shorten a sweep."
        ),
    )
    args = parser.parse_args()
    if args.arms is not None:
        known = {arm.name for arm in CALIBRATION_ARMS}
        unknown = sorted(set(args.arms) - known)
        if unknown:
            raise SystemExit(f"unknown arms {unknown}; choose from {sorted(known)}")
        selected = set(args.arms)
        globals()["CALIBRATION_ARMS"] = tuple(
            arm for arm in CALIBRATION_ARMS if arm.name in selected
        )
    {"run": stage_run, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
