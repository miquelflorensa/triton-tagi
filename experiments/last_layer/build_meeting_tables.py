"""Build the deliverable tables of MEETING_PLAN.md §5 from the runs on disk.

Every table degrades to an explicit "pending" note naming what is missing,
rather than silently omitting a row, so the file can be regenerated at any
point while the study is still running and always says how complete it is.

Usage:
  python experiments/last_layer/build_meeting_tables.py
  python experiments/last_layer/build_meeting_tables.py --output -   # stdout
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("study.json")

HEAD_ORDER = (
    "hrc",
    "hrc:full",
    "hrc:full+offsets",
    "remax_lognormal",
    "remax_laplace_diag",
    "logit_tagiv",
)
ARM_ORDER = ("random", "zero", "backbone")
DATASET_ORDER = ("cifar10", "cifar100")
CALIBRATION_ORDER = ("", "global", "level", "node")

# The ImageNet softmax reference, from MEETING_PLAN.md §5.
IMAGENET_REFERENCE = {"accuracy": 0.6976, "nll": 1.2469, "ece": 0.0263}


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def artifact_root(manifest: dict[str, Any]) -> Path:
    return REPOSITORY_ROOT / manifest["paths"]["artifacts"] / manifest["study_id"]


def default_hrc_tree(head: str) -> str:
    return "full" if head == "hrc_probit" else "padded"


def prior_offsets_on(value: Any) -> bool:
    """Whether a cell was trained with the full tree's readout offsets.

    Cells that predate the axis carry no value and were trained on the
    constructor default, which is offsets on. The report writes the field as a
    bool through JSON and as text through CSV, so both spellings are read.
    """

    if value is None or value == "":
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no"}


def arm_name(head: str, hrc_tree: Any, hrc_prior_offsets: Any = None) -> str:
    """The label a head is reported under, matching run_study.selection_arm."""

    tree = (hrc_tree or "").strip() if isinstance(hrc_tree, str) else ""
    if not tree or tree == default_hrc_tree(head):
        return head
    arm = f"{head}:{tree}"
    if tree == "full" and prior_offsets_on(hrc_prior_offsets):
        arm = f"{arm}+offsets"
    return arm


def screen_cells(manifest: dict[str, Any], stage: str = "init_screen") -> list[dict[str, Any]]:
    """Every completed screen cell, as config plus its final validation record."""

    root = artifact_root(manifest) / "heads" / stage
    cells = []
    for config_path in root.glob("*/*/*/config.json"):
        run_dir = config_path.parent
        if not (run_dir / "complete.json").exists():
            continue
        history_path = run_dir / "history.json"
        if not history_path.exists():
            continue
        config = json.loads(config_path.read_text())
        records = json.loads(history_path.read_text())
        final = [row for row in records if int(row["epoch"]) == int(config["epochs"])]
        if not final:
            continue
        zero = [row for row in records if int(row["epoch"]) == 0]
        # Epoch 0 is the prior and is not a trained model; the best trained
        # checkpoint is what "this cell can do" and is frequently epoch 1.
        trained = [row for row in records if int(row["epoch"]) > 0]
        cells.append(
            {
                "config": config,
                "arm": arm_name(
                    config["head"],
                    config.get("hrc_tree"),
                    config.get("hrc_prior_offsets"),
                ),
                "record": final[-1],
                "epoch0": zero[-1] if zero else None,
                "best_trained": (
                    min(trained, key=lambda row: row["val_nll"]) if trained else None
                ),
            }
        )
    return cells


def report_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    path = artifact_root(manifest) / "report.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def imagenet_rows() -> list[dict[str, Any]]:
    path = REPOSITORY_ROOT / "runs" / "imagenet" / "init_study" / "report.csv"
    if not path.exists():
        return []
    import csv

    with path.open() as handle:
        return list(csv.DictReader(handle))


def imagenet_cells(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    """The final-epoch record of every ImageNet run in one stage.

    The ImageNet driver writes one CSV row per (stage, config, seed, epoch),
    so the last epoch of each run is the cell result. Epoch 0 is the prior and
    is dropped for the same reason selection never picks it.
    """

    by_run: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("stage") != stage:
            continue
        try:
            epoch = float(row["epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        if epoch <= 0:
            continue
        key = row["run_dir"]
        if key not in by_run or epoch > float(by_run[key]["epoch"]):
            by_run[key] = row
    return list(by_run.values())


def table_imagenet(rows: list[dict[str, Any]]) -> str:
    """ImageNet-1k, accuracy and calibration only -- no OOD source is cached."""

    screen = imagenet_cells(rows, "screen")
    if not screen:
        return pending("The ImageNet arm", "no completed ImageNet runs")
    gains = sorted({float(c["gain_w"]) for c in screen})
    blocks = [
        "Best cell per (head, mean_init) by top-1, over the sigma_v axis. "
        f"Softmax reference: top-1 {IMAGENET_REFERENCE['accuracy']:.4f}, "
        f"NLL {IMAGENET_REFERENCE['nll']:.4f}, ECE {IMAGENET_REFERENCE['ece']:.4f}.",
        "",
        "| head | mean_init | " + " | ".join(f"gain {g:g}" for g in gains) + " |",
        "|---" * (len(gains) + 2) + "|",
    ]
    for head in ("hrc", "remax_lognormal", "remax_laplace_diag", "logit_tagiv"):
        for init in ARM_ORDER:
            row = []
            for gain in gains:
                group = [
                    c for c in screen
                    if c["head"] == head
                    and c["mean_init"] == init
                    and float(c["gain_w"]) == gain
                ]
                if not group:
                    row.append("—")
                    continue
                best = max(group, key=lambda c: float(c["val_accuracy"]))
                row.append(
                    f"{float(best['val_accuracy']):.4f} / {float(best['val_nll']):.2f}"
                )
            if all(entry == "—" for entry in row):
                continue
            blocks.append(f"| `{head}` | {init} | " + " | ".join(row) + " |")
    blocks += [
        "",
        "Cells are top-1 / NLL. **Mean confidence collapses at 1000 classes** for "
        "the remax heads -- they rank well and report near-uniform probability -- "
        "so accuracy and NLL rank the cells almost independently. Read both.",
    ]
    confirm = imagenet_cells(rows, "confirm")
    if not confirm:
        blocks += ["", pending("The ImageNet confirm rows", "needs `run --stage confirm`")]
    else:
        agg: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for cell in confirm:
            agg[(cell["head"], cell["mean_init"])].append(cell)
        blocks += [
            "",
            "**Confirm (4 epochs, 3 seeds):**",
            "",
            "| head | mean_init | seeds | top-1 | NLL | ECE |",
            "|---|---|---|---|---|---|",
        ]
        for key in sorted(agg):
            group = agg[key]
            def mean(field):
                return sum(float(c[field]) for c in group) / len(group)
            blocks.append(
                f"| `{key[0]}` | {key[1]} | {len(group)} | {mean('val_accuracy'):.4f} | "
                f"{mean('val_nll'):.4f} | {mean('val_ece'):.4f} |"
            )
    return "\n".join(blocks) + "\n"


def number(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def pending(what: str, needs: str) -> str:
    return f"*{what} is pending: {needs}.*\n"


def best_by_nll(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    return min(cells, key=lambda c: c["record"]["val_nll"]) if cells else None


# ──────────────────────────────────────────────────────────────────────────────
#  Tables
# ──────────────────────────────────────────────────────────────────────────────


def table_screen_init_delta(cells: list[dict[str, Any]]) -> str:
    """Table 2 at 20 epochs: the init arms, each at its own best gain/sigma_v."""

    if not cells:
        return pending("The screen initialization delta", "no completed init_screen cells")
    lines = [
        "| dataset | head | mean_init | cells | val top-1 | val NLL | val ECE | best gain | best sigma_v |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for dataset in DATASET_ORDER:
        for arm in HEAD_ORDER:
            for init in ARM_ORDER:
                group = [
                    c for c in cells
                    if c["config"]["dataset"] == dataset
                    and c["arm"] == arm
                    and c["config"].get("mean_init") == init
                ]
                winner = best_by_nll(group)
                if winner is None:
                    continue
                config, record = winner["config"], winner["record"]
                lines.append(
                    f"| {dataset} | `{arm}` | {init} | {len(group)} | "
                    f"{number(record['val_accuracy'])} | {number(record['val_nll'])} | "
                    f"{number(record['val_ece'])} | {config['gain_w']} | "
                    f"{config.get('sigma_v') if config.get('sigma_v') is not None else 'n/a'} |"
                )
    return "\n".join(lines) + "\n"


def table_axis_sensitivity(cells: list[dict[str, Any]], axis: str, values: list) -> str:
    """Tables 3 and 4: one metric against one axis, per head and init arm.

    Each entry is the best cell over the axes not shown, so the row reads as
    "what this arm achieves at this axis value", not a single arbitrary slice.
    """

    if not cells:
        return pending(f"The {axis} sensitivity", "no completed init_screen cells")
    blocks = []
    for dataset in DATASET_ORDER:
        present = [c for c in cells if c["config"]["dataset"] == dataset]
        if not present:
            continue
        blocks.append(f"\n**{dataset}** — validation NLL at 20 epochs\n")
        header = "| head | mean_init | " + " | ".join(f"{axis} {v}" for v in values) + " |"
        blocks.append(header)
        blocks.append("|---" * (len(values) + 2) + "|")
        for arm in HEAD_ORDER:
            for init in ARM_ORDER:
                row = []
                for value in values:
                    group = [
                        c for c in present
                        if c["arm"] == arm
                        and c["config"].get("mean_init") == init
                        and c["config"].get(axis) == value
                    ]
                    winner = best_by_nll(group)
                    row.append(number(winner["record"]["val_nll"]) if winner else "—")
                if all(entry == "—" for entry in row):
                    continue
                blocks.append(f"| `{arm}` | {init} | " + " | ".join(row) + " |")
    return "\n".join(blocks) + "\n"


def table_warm_start(cells: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    """The backbone arm before any training, reported as the reference it is.

    Epoch 0 of the ``backbone`` arm is the backbone's own trained ``fc`` read
    through the head's link, so it is not an untrained model -- it scores like
    the softmax baseline it copies. It is excluded from selection (a selected
    cell must have seen data) and reported here instead, because "the warm
    start beats every trained cell" is a finding about the heads, not about
    initialization, and the deck must not present it as the latter.

    It needs no seeds: at epoch 0 nothing has been trained, and the prior
    variances are a deterministic function of gain, so the row is exact.
    """

    if not cells:
        return pending("The warm-start reference", "no completed init_screen cells")
    baseline = {
        row["dataset"]: row for row in rows if row.get("head") == "pytorch_softmax"
    }
    lines = [
        "| dataset | head | epoch 0 top-1 | epoch 0 NLL | epoch 0 ECE | best trained NLL | trained at epoch |",
        "|---|---|---|---|---|---|---|",
    ]
    for dataset in DATASET_ORDER:
        for arm in HEAD_ORDER:
            warm = [
                c for c in cells
                if c["config"]["dataset"] == dataset
                and c["arm"] == arm
                and c["config"].get("mean_init") == "backbone"
            ]
            if not warm:
                continue
            # Epoch 0 varies across the grid only through the prior variance,
            # so report the best of them and the best trained cell beside it.
            warmest = min(
                (c for c in warm if c.get("epoch0")),
                key=lambda c: c["epoch0"]["val_nll"],
                default=None,
            )
            if warmest is None:
                continue
            record = warmest["epoch0"]
            trained = min(
                (c for c in warm if c.get("best_trained")),
                key=lambda c: c["best_trained"]["val_nll"],
                default=None,
            )
            if trained is None:
                continue
            lines.append(
                f"| {dataset} | `{arm}` | {number(record['val_accuracy'])} | "
                f"{number(record['val_nll'])} | {number(record['val_ece'])} | "
                f"{number(trained['best_trained']['val_nll'])} | "
                f"{int(trained['best_trained']['epoch'])} |"
            )
    for dataset, row in sorted(baseline.items()):
        lines.append(
            f"| {dataset} | `pytorch_softmax` (reference) | "
            f"{number(row.get('clean_svhn_classification_accuracy'))} | "
            f"{number(row.get('clean_svhn_classification_nll'))} | "
            f"{number(row.get('clean_svhn_classification_ece'))} | — | — |"
        )
    return "\n".join(lines) + "\n"


def _confirm_rows(rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("stage") == "init_confirm" and row.get("evaluation_kind") == kind
    ]


def _aggregate(rows: list[dict[str, Any]], metric: str) -> tuple[float, int] | None:
    values = [row[metric] for row in rows if isinstance(row.get(metric), (int, float))]
    if not values:
        return None
    return sum(values) / len(values), len(values)


def table_headline(rows: list[dict[str, Any]], imagenet: list[dict[str, Any]]) -> str:
    """Table 1: head x dataset against the softmax reference."""

    confirm = _confirm_rows(rows, "epoch_200")
    if not confirm:
        return pending(
            "Table 1 (headline)",
            "needs `run --stage init_confirm` then `evaluate --stage init_confirm`",
        )
    baseline = {
        row["dataset"]: row for row in rows if row.get("head") == "pytorch_softmax"
    }
    lines = [
        "| dataset | head | mean_init | seeds | top-1 | NLL | ECE | OOD AUROC (entropy) | OOD AUROC (native epi.) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in confirm:
        key = (
            row["dataset"],
            arm_name(row["head"], row.get("hrc_tree"), row.get("hrc_prior_offsets")),
            row.get("mean_init", ""),
            row.get("calibration", ""),
        )
        grouped[key].append(row)
    for (dataset, arm, init, calibration) in sorted(grouped):
        if calibration:
            continue  # calibrated rows are table 5
        group = grouped[(dataset, arm, init, calibration)]
        accuracy = _aggregate(group, "clean_svhn_classification_accuracy")
        lines.append(
            f"| {dataset} | `{arm}` | {init} | {accuracy[1] if accuracy else 0} | "
            f"{number(accuracy[0]) if accuracy else '—'} | "
            f"{number((_aggregate(group, 'clean_svhn_classification_nll') or [None])[0])} | "
            f"{number((_aggregate(group, 'clean_svhn_classification_ece') or [None])[0])} | "
            f"{number((_aggregate(group, 'clean_svhn_ood_entropy_auroc') or [None])[0])} | "
            f"{number((_aggregate(group, 'clean_svhn_ood_native_epistemic_auroc') or [None])[0])} |"
        )
    for dataset, row in sorted(baseline.items()):
        lines.append(
            f"| {dataset} | `pytorch_softmax` | — | 1 | "
            f"{number(row.get('clean_svhn_classification_accuracy'))} | "
            f"{number(row.get('clean_svhn_classification_nll'))} | "
            f"{number(row.get('clean_svhn_classification_ece'))} | "
            f"{number(row.get('clean_svhn_ood_entropy_auroc'))} | "
            f"{number(row.get('clean_svhn_ood_native_epistemic_auroc'))} |"
        )
    if imagenet:
        confirmed: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for cell in imagenet_cells(imagenet, "confirm"):
            confirmed[(cell["head"], cell["mean_init"])].append(cell)

        def imagenet_mean(group: list[dict[str, Any]], field: str) -> str:
            values = [float(c[field]) for c in group if c.get(field) not in (None, "")]
            return number(sum(values) / len(values)) if values else "—"

        for key in sorted(
            confirmed,
            key=lambda k: (
                HEAD_ORDER.index(k[0]) if k[0] in HEAD_ORDER else 9,
                ARM_ORDER.index(k[1]) if k[1] in ARM_ORDER else 9,
            ),
        ):
            group = confirmed[key]
            lines.append(
                f"| imagenet1k | `{key[0]}` | {key[1]} | {len(group)} | "
                f"{imagenet_mean(group, 'val_accuracy')} | "
                f"{imagenet_mean(group, 'val_nll')} | "
                f"{imagenet_mean(group, 'val_ece')} | n/a | n/a |"
            )
        lines.append(
            f"| imagenet1k | `pretrained_fc` | — | 1 | "
            f"{number(IMAGENET_REFERENCE['accuracy'])} | "
            f"{number(IMAGENET_REFERENCE['nll'])} | "
            f"{number(IMAGENET_REFERENCE['ece'])} | n/a | n/a |"
        )
    else:
        lines.append("")
        lines.append(pending("The ImageNet row", "the ImageNet arm has not been run"))
    lines.append("")
    lines.append(
        "ImageNet OOD cells are `n/a`, not blank: no ImageNet OOD set is cached, "
        "which was decided rather than overlooked (§6). The native-epistemic "
        "column is therefore a CIFAR-only result. That column is also **not "
        "comparable across heads**: it reduces each head's own output-space "
        "epistemic variance, which is node space for `hrc`, logit variance for "
        "`logit_tagiv`, and class space for the calibrated rows -- see the note "
        "under Table 6a. The ImageNet rows are **4 "
        "epochs and 3 seeds** against CIFAR's 200 epochs and 5 seeds, and its "
        "`pretrained_fc` reference is the teacher the features come from, so read "
        "down the ImageNet block, not across into the CIFAR rows."
    )
    return "\n".join(lines) + "\n"


def table_calibration(rows: list[dict[str, Any]]) -> str:
    """Table 5: calibration x initialization on the full-tree hrc arm."""

    calibrated = [
        row for row in rows
        if row.get("stage") == "init_confirm" and row.get("calibration")
    ]
    if not calibrated:
        return pending(
            "Table 5 (calibration x initialization)",
            "needs `calibrate --stage init_confirm`, which requires the full-tree "
            "hrc confirm runs",
        )
    uncalibrated = [
        row for row in _confirm_rows(rows, "epoch_200") if not row.get("calibration")
    ]
    lines = [
        "| dataset | head | mean_init | calibration | groups | NLL | ECE | top-1 | OOD AUROC (native epi.) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in calibrated + uncalibrated:
        arm = arm_name(row["head"], row.get("hrc_tree"), row.get("hrc_prior_offsets"))
        if ":full" not in arm:
            continue
        grouped[(row["dataset"], arm, row.get("mean_init", ""), row.get("calibration", ""))].append(row)
    for key in sorted(grouped, key=lambda k: (k[0], k[1], k[2], CALIBRATION_ORDER.index(k[3]) if k[3] in CALIBRATION_ORDER else 9)):
        dataset, arm, init, calibration = key
        group = grouped[key]
        lines.append(
            f"| {dataset} | `{arm}` | {init} | {calibration or 'uncalibrated'} | "
            f"{group[0].get('n_groups', '—')} | "
            f"{number((_aggregate(group, 'clean_svhn_classification_nll') or [None])[0])} | "
            f"{number((_aggregate(group, 'clean_svhn_classification_ece') or [None])[0])} | "
            f"{number((_aggregate(group, 'clean_svhn_classification_accuracy') or [None])[0])} | "
            f"{number((_aggregate(group, 'clean_svhn_ood_native_epistemic_auroc') or [None])[0])} |"
        )
    lines.append("")
    lines.append(
        "The gain belief is fitted on the full 10 000-row validation split, which "
        "is also the split that selected the cell — state that in the caption."
    )
    return "\n".join(lines) + "\n"



# ──────────────────────────────────────────────────────────────────────────────
#  Appendix — every combination on disk, nothing aggregated away
# ──────────────────────────────────────────────────────────────────────────────

# Accuracy, calibration and OOD for one evaluated cell. The OOD columns exist
# only for the confirm stage: `evaluate` scores SVHN and the corruption suite
# from a checkpoint, and the screen writes validation metrics only, so the
# screen grid below genuinely has no OOD to report rather than an empty column.
APPENDIX_METRICS = (
    ("top-1", "clean_svhn_classification_accuracy"),
    ("NLL", "clean_svhn_classification_nll"),
    ("ECE", "clean_svhn_classification_ece"),
    ("Brier", "clean_svhn_classification_brier"),
    ("AUROC ent", "clean_svhn_ood_entropy_auroc"),
    ("AUROC epi", "clean_svhn_ood_native_epistemic_auroc"),
    ("AUROC maxp", "clean_svhn_ood_negative_max_probability_auroc"),
    ("shift top-1", "corruption_macro_shift_classification_accuracy"),
    ("shift NLL", "corruption_macro_shift_classification_nll"),
    ("shift ECE", "corruption_macro_shift_classification_ece"),
    ("shift AUROC ent", "corruption_macro_ood_entropy_auroc"),
    ("shift AUROC epi", "corruption_macro_ood_native_epistemic_auroc"),
)

KIND_ORDER = ("epoch_200", "validation_selected")
KIND_LABEL = {"epoch_200": "epoch 200", "validation_selected": "best val"}


def axis_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):g}"


def table_all_confirmed(rows: list[dict[str, Any]]) -> str:
    """Appendix A: every confirmed combination, all three metric families.

    One row per (dataset, arm, mean_init, gain, sigma_v, calibration,
    checkpoint), averaged over the five seeds. This is the only stage that
    carries OOD, so it is the only place "accuracy, calibration and OOD for
    every combination" can be answered in full.
    """

    confirmed = [row for row in rows if row.get("stage") == "init_confirm"]
    if not confirmed:
        return pending("The per-combination appendix", "needs `evaluate --stage init_confirm`")
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in confirmed:
        grouped[(
            row["dataset"],
            arm_name(row["head"], row.get("hrc_tree"), row.get("hrc_prior_offsets")),
            row.get("mean_init", ""),
            row.get("gain_w"),
            row.get("sigma_v"),
            row.get("calibration", ""),
            row.get("evaluation_kind", ""),
        )].append(row)

    def sort_key(key: tuple) -> tuple:
        dataset, arm, init, gain, sigma, calibration, kind = key
        return (
            DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else 9,
            HEAD_ORDER.index(arm) if arm in HEAD_ORDER else 9,
            ARM_ORDER.index(init) if init in ARM_ORDER else 9,
            KIND_ORDER.index(kind) if kind in KIND_ORDER else 9,
            CALIBRATION_ORDER.index(calibration) if calibration in CALIBRATION_ORDER else 9,
        )

    header = (
        "| dataset | head | mean_init | gain | sigma_v | calibration | checkpoint | seeds | "
        + " | ".join(label for label, _ in APPENDIX_METRICS)
        + " |"
    )
    lines = [header, "|---" * (8 + len(APPENDIX_METRICS)) + "|"]
    for key in sorted(grouped, key=sort_key):
        dataset, arm, init, gain, sigma, calibration, kind = key
        group = grouped[key]
        epochs = sorted({int(row["epoch"]) for row in group if row.get("epoch") is not None})
        checkpoint = KIND_LABEL.get(kind, kind)
        if kind == "validation_selected" and epochs:
            span = f"{epochs[0]}" if len(epochs) == 1 else f"{epochs[0]}-{epochs[-1]}"
            checkpoint = f"best val (ep {span})"
        cells = [
            number((_aggregate(group, metric) or [None])[0])
            for _, metric in APPENDIX_METRICS
        ]
        lines.append(
            f"| {dataset} | `{arm}` | {init} | {axis_value(gain)} | {axis_value(sigma)} | "
            f"{calibration or 'uncalibrated'} | {checkpoint} | {len(group)} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "Every cell is the mean over the seeds in its row. `AUROC ent` / `epi` / "
        "`maxp` are SVHN-vs-test AUROC under the entropy, native-epistemic and "
        "negative-max-probability scores. The `shift` columns are CIFAR-10-C / "
        "CIFAR-100-C, macro-averaged over 15 corruptions x 5 severities: accuracy "
        "and calibration on the corrupted data, and AUROC for separating corrupted "
        "from clean. Note the corruption evaluation also reports a clean-side "
        "`corruption_macro_classification_*` block that duplicates the columns to "
        "its left -- it is the ID half of the same comparison, not a second result.",
        "",
        "Two rows per cell, because the heads disagree about when they are best: "
        "`epoch 200` is the protocol horizon and `best val` is the checkpoint "
        "`evaluate` picked on validation NLL. The gap between them is the "
        "peak-then-decay finding, not noise. **One `best val` row is epoch 0** "
        "(cifar10 `remax_lognormal` backbone, seed 4) -- the untrained warm start. "
        "Selection excludes epoch 0; the evaluator's own best-checkpoint search "
        "does not, so read that row as the prior, not as a trained result.",
    ]
    return "\n".join(lines) + "\n"


def table_all_screen(cells: list[dict[str, Any]]) -> str:
    """Appendix B: the complete screen grid -- accuracy and calibration only."""

    if not cells:
        return pending("The screen grid appendix", "no completed init_screen cells")
    lines = [
        "| dataset | head | mean_init | gain | sigma_v | epochs | top-1 | NLL | ECE | "
        "Brier | classwise ECE | mean conf. | AURC |",
        "|---" * 13 + "|",
    ]
    def sort_key(cell: dict[str, Any]) -> tuple:
        config = cell["config"]
        return (
            DATASET_ORDER.index(config["dataset"]) if config["dataset"] in DATASET_ORDER else 9,
            HEAD_ORDER.index(cell["arm"]) if cell["arm"] in HEAD_ORDER else 9,
            ARM_ORDER.index(config.get("mean_init", "")) if config.get("mean_init") in ARM_ORDER else 9,
            float(config.get("gain_w") or 0),
            float(config.get("sigma_v") or 0),
        )
    for cell in sorted(cells, key=sort_key):
        config, record = cell["config"], cell["record"]
        lines.append(
            f"| {config['dataset']} | `{cell['arm']}` | {config.get('mean_init', '—')} | "
            f"{axis_value(config.get('gain_w'))} | {axis_value(config.get('sigma_v'))} | "
            f"{config['epochs']} | {number(record['val_accuracy'])} | "
            f"{number(record['val_nll'])} | {number(record['val_ece'])} | "
            f"{number(record['val_brier'])} | {number(record['val_classwise_ece'])} | "
            f"{number(record['val_mean_confidence'])} | {number(record['val_aurc'])} |"
        )
    lines += [
        "",
        "The final epoch of every screen cell, one seed each. **No OOD columns**: "
        "the screen writes validation metrics only, and OOD is scored by "
        "`evaluate` from a checkpoint. The screen does keep its checkpoints "
        "(1824 of them, 435 MB), so `evaluate --stage init_screen` would fill this "
        "in for all 304 cells -- at the measured ~10 s per checkpoint that is "
        "about an hour, and it has not been run.",
    ]
    return "\n".join(lines) + "\n"


def table_all_imagenet(rows: list[dict[str, Any]]) -> str:
    """Appendix C: every ImageNet cell, screen and confirm."""

    if not rows:
        return pending("The ImageNet grid appendix", "no completed ImageNet runs")
    lines = [
        "| stage | head | mean_init | gain | sigma_v | seeds | epochs | top-1 | top-5 | "
        "NLL | ECE | Brier | mean conf. | AURC |",
        "|---" * 14 + "|",
    ]
    for stage in ("screen", "confirm"):
        grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for cell in imagenet_cells(rows, stage):
            grouped[(
                cell["head"],
                cell["mean_init"],
                cell["gain_w"],
                cell.get("sigma_v", ""),
            )].append(cell)
        def sort_key(key: tuple) -> tuple:
            head, init, gain, sigma = key
            return (
                HEAD_ORDER.index(head) if head in HEAD_ORDER else 9,
                ARM_ORDER.index(init) if init in ARM_ORDER else 9,
                float(gain or 0),
                float(sigma or 0),
            )
        for key in sorted(grouped, key=sort_key):
            head, init, gain, sigma = key
            group = grouped[key]
            def mean(field: str) -> str:
                values = [float(c[field]) for c in group if c.get(field) not in (None, "")]
                return number(sum(values) / len(values)) if values else "—"
            epochs = sorted({int(float(c["epoch"])) for c in group})
            lines.append(
                f"| {stage} | `{head}` | {init} | {axis_value(gain)} | {axis_value(sigma)} | "
                f"{len(group)} | {epochs[-1]} | {mean('val_accuracy')} | "
                f"{mean('val_top5_accuracy')} | {mean('val_nll')} | {mean('val_ece')} | "
                f"{mean('val_brier')} | {mean('val_mean_confidence')} | {mean('val_aurc')} |"
            )
    lines += [
        "",
        "No OOD columns anywhere on this dataset: no ImageNet OOD set is cached, "
        "which was decided rather than overlooked (§6). `remax_laplace_diag` "
        "trains at batch 134 against 256 for every other head -- its Laplace "
        "Jacobian is O(K^2) and OOMs at 1000 classes otherwise -- so its rows "
        "carry a batch-size confound the other heads do not.",
    ]
    return "\n".join(lines) + "\n"



# ──────────────────────────────────────────────────────────────────────────────
#  HSM post-hoc calibration — this study's axis 4, and the finished sweep
# ──────────────────────────────────────────────────────────────────────────────

HSM_ROOT = REPOSITORY_ROOT / "runs" / "last_layer" / "hsm_calibration"

# The order the calibration methods are argued in: the uncalibrated reference,
# the three gain-belief sharing levels that are this study's axis 4, then the
# ablations that say what the gain *belief* buys over a fitted point value, and
# finally the softmax temperature baseline that has to be beaten.
HSM_ARM_ORDER = (
    "uncalibrated",
    "hsm_global",
    "hsm_level",
    "hsm_node",
    "hsm_global_point",
    "hsm_level_point",
    "hsm_node_point",
    "hsm_global_laplace",
    "hsm_global_adf",
    "hsm_global_adf_reversed",
    "hsm_node_adf",
    "hrc_fitted_scale",
    "softmax_temperature",
)


def hsm_report(dataset: str, suffix: str = "") -> list[dict[str, Any]]:
    path = HSM_ROOT / f"{dataset}_base_hrc{suffix}" / "report.csv"
    if not path.exists():
        return []
    import csv

    with path.open() as handle:
        return list(csv.DictReader(handle))


def hsm_group_counts(dataset: str) -> dict[str, int]:
    """n_groups per sharing level, read from the sweep's own records."""

    path = HSM_ROOT / f"{dataset}_base_hrc" / "records.json"
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    for record in json.loads(path.read_text()):
        sharing = record.get("sharing")
        groups = record.get("n_groups")
        if sharing and isinstance(groups, int):
            counts[sharing] = max(counts.get(sharing, 0), groups)
    return counts


def confirm_group_counts(manifest: dict[str, Any]) -> dict[tuple[str, str], int]:
    """n_groups for the axis-4 fits, from the calibrated evaluation records.

    ``calibrate`` writes it into each evaluation JSON but the report schema does
    not carry it, so Table 5 printed an empty groups column. The count depends
    only on (dataset, sharing) -- it is a property of the tree -- so a
    disagreement would mean two different trees were calibrated under one label,
    which is worth seeing rather than averaging away.
    """

    counts: dict[tuple[str, str], set[int]] = defaultdict(set)
    root = artifact_root(manifest) / "evaluations"
    for path in root.glob("*/init_confirm_calibrated/*/*/*.json"):
        dataset = path.relative_to(root).parts[0]
        try:
            block = json.loads(path.read_text()).get("calibration") or {}
        except json.JSONDecodeError:
            continue
        sharing, groups = block.get("sharing"), block.get("n_groups")
        if sharing and isinstance(groups, int):
            counts[(dataset, sharing)].add(groups)
    return {key: sorted(values)[-1] for key, values in counts.items() if values}


def _best_key(entries: dict[str, float], lower_is_better: bool = True) -> str | None:
    if not entries:
        return None
    return min(entries, key=entries.get) if lower_is_better else max(entries, key=entries.get)


def table_hsm_axis4(rows: list[dict[str, Any]], groups: dict[tuple[str, str], int]) -> str:
    """Table 6a: which sharing level wins on the confirmed hrc:full cells."""

    confirmed = [
        row for row in rows
        if row.get("stage") == "init_confirm"
        and row.get("evaluation_kind") == "epoch_200"
        and ":full" in arm_name(
            row["head"], row.get("hrc_tree"), row.get("hrc_prior_offsets")
        )
    ]
    if not confirmed:
        return pending("Table 6a", "needs `calibrate --stage init_confirm`")
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    arms: list[str] = []
    for row in confirmed:
        arm = arm_name(row["head"], row.get("hrc_tree"), row.get("hrc_prior_offsets"))
        if arm not in arms:
            arms.append(arm)
        grouped[
            (row["dataset"], arm, row.get("mean_init", ""), row.get("calibration", ""))
        ].append(row)
    arms.sort(key=lambda arm: HEAD_ORDER.index(arm) if arm in HEAD_ORDER else 9)

    lines = [
        "| dataset | arm | mean_init | sharing | groups | NLL | ECE | top-1 | "
        "SVHN AUROC ent | SVHN AUROC epi | shift NLL | best NLL |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for dataset in DATASET_ORDER:
      for arm in arms:
        for init in ARM_ORDER:
            present = {
                calibration: grouped[(dataset, arm, init, calibration)]
                for calibration in CALIBRATION_ORDER
                if (dataset, arm, init, calibration) in grouped
            }
            if not present:
                continue
            nlls = {
                calibration: (_aggregate(group, "clean_svhn_classification_nll") or [None])[0]
                for calibration, group in present.items()
            }
            winner = _best_key({k: v for k, v in nlls.items() if v is not None})
            for calibration, group in present.items():
                count = (
                    str(groups.get((dataset, calibration), "—")) if calibration else "—"
                )
                lines.append(
                    f"| {dataset} | `{arm}` | {init} | {calibration or 'uncalibrated'} | "
                    f"{count} | "
                    f"{number(nlls[calibration])} | "
                    f"{number((_aggregate(group, 'clean_svhn_classification_ece') or [None])[0])} | "
                    f"{number((_aggregate(group, 'clean_svhn_classification_accuracy') or [None])[0])} | "
                    f"{number((_aggregate(group, 'clean_svhn_ood_entropy_auroc') or [None])[0])} | "
                    f"{number((_aggregate(group, 'clean_svhn_ood_native_epistemic_auroc') or [None])[0])} | "
                    f"{number((_aggregate(group, 'corruption_macro_shift_classification_nll') or [None])[0])} | "
                    f"{'**yes**' if calibration == winner else ''} |"
                )
    lines += [
        "",
        "Five seeds per row, gain fitted on the 10 000-row validation split. "
        "**The padded `hrc` arm has no rows here and cannot have any**: "
        "`gain_groups` refuses the padded tree, because it discards leaves that "
        "hold probability mass, so only the `hrc:full` arm is calibratable at all.",
        "",
        "**Do not read the native-epistemic column down this table.** The "
        "uncalibrated and calibrated rows do not score the same quantity. "
        "`run_study.predict_batches` reduces `ClassificationPrediction."
        "epistemic_variance`, which for `hrc` is the **node-space latent "
        "variance** -- `classification.py` leaves it at the network's raw output "
        "variance for this head -- while `calibrated_predict_batches` reduces "
        "`hsm_class_moments(...).variance`, the variance of the **class "
        "probabilities**. Its docstring claims the two compare like with like; "
        "the reduction matches, the quantity does not. So the 0.1052 -> 0.8962 "
        "jump on cifar10 is a change of score, not an effect of calibration. "
        "The like-for-like comparison is 6c, which scores probability-space "
        "dispersion throughout: there, calibration is worth ~nothing on cifar10 "
        "(0.907 uncalibrated vs 0.908 global at n=10000) and a real but modest "
        "gain on cifar100 (0.752 -> 0.776 level, 0.795 node). Table 1 and "
        "Appendix A carry the same node-space score in their uncalibrated rows, "
        "so that column ranks checkpoints within one head and is not comparable "
        "across heads or against 6c.",
    ]
    return "\n".join(lines) + "\n"


def table_hsm_sweep(metric: str, label: str, lower_is_better: bool = True) -> str:
    """Table 6b: every calibration method against the size of the fitting split."""

    blocks = []
    for dataset in DATASET_ORDER:
        rows = hsm_report(dataset)
        if not rows:
            continue
        counts = hsm_group_counts(dataset)
        sizes = sorted({int(row["calibration_size"]) for row in rows})
        table: dict[str, dict[int, float]] = defaultdict(dict)
        meta: dict[str, dict[str, str]] = {}
        for row in rows:
            value = row.get(metric)
            if value in (None, ""):
                continue
            table[row["arm"]][int(row["calibration_size"])] = float(value)
            meta[row["arm"]] = {
                "role": row["role"],
                "sharing": row["sharing"],
                "method": row["method"],
                "gain_uncertainty": row["gain_uncertainty"],
            }
        if not table:
            continue
        best = {
            size: _best_key(
                {arm: values[size] for arm, values in table.items() if size in values},
                lower_is_better,
            )
            for size in sizes
        }
        blocks += [
            f"\n**{dataset}** — test {label} against the number of rows the gain is fitted on\n",
            "| method | sharing | groups | gain belief | role | "
            + " | ".join(f"n={size}" for size in sizes)
            + " |",
            "|---" * (5 + len(sizes)) + "|",
        ]
        for arm in HSM_ARM_ORDER:
            if arm not in table:
                continue
            info = meta[arm]
            groups = counts.get(info["sharing"], "—") if info["sharing"] != "none" else "—"
            cells = []
            for size in sizes:
                if size not in table[arm]:
                    cells.append("—")
                    continue
                text = number(table[arm][size])
                cells.append(f"**{text}**" if best[size] == arm else text)
            blocks.append(
                f"| `{arm}` | {info['sharing']} | {groups} | "
                f"{'yes' if info['gain_uncertainty'] == 'True' else 'no'} | {info['role']} | "
                + " | ".join(cells)
                + " |"
            )
    if not blocks:
        return pending(f"The HSM {label} sweep", "runs/last_layer/hsm_calibration is missing")
    return "\n".join(blocks) + "\n"


def table_hsm_ood() -> str:
    """Table 6c: what each calibration method does to OOD detection."""

    blocks = []
    for dataset in DATASET_ORDER:
        rows = hsm_report(dataset, "_ood")
        if not rows:
            continue
        sizes = sorted({int(row["calibration_size"]) for row in rows})
        blocks += [
            f"\n**{dataset}** — SVHN AUROC, native-epistemic score / entropy score\n",
            "| method | sharing | " + " | ".join(f"n={size}" for size in sizes) + " |",
            "|---" * (2 + len(sizes)) + "|",
        ]
        indexed = {(row["arm"], int(row["calibration_size"])): row for row in rows}
        for arm in HSM_ARM_ORDER:
            present = [size for size in sizes if (arm, size) in indexed]
            if not present:
                continue
            cells = []
            for size in sizes:
                row = indexed.get((arm, size))
                if row is None:
                    cells.append("—")
                    continue
                cells.append(
                    f"{number(row['svhn_native_epistemic_auroc'], 3)} / "
                    f"{number(row['svhn_entropy_auroc'], 3)}"
                )
            blocks.append(
                f"| `{arm}` | {indexed[(arm, present[0])]['sharing']} | " + " | ".join(cells) + " |"
            )
    if not blocks:
        return pending("The HSM OOD sweep", "the *_base_hrc_ood reports are missing")
    blocks += [
        "",
        "The entropy score is nearly unmoved by calibration -- it reads the "
        "predictive distribution, which barely changes. The native-epistemic "
        "score is the one that needs the gain belief, and it is the column to "
        "point at when the question is what TAGI's own uncertainty adds.",
    ]
    return "\n".join(blocks) + "\n"


def build(manifest: dict[str, Any]) -> str:
    cells = screen_cells(manifest)
    rows = report_rows(manifest)
    imagenet = imagenet_rows()
    study = manifest["init_study"]

    counts = defaultdict(int)
    for cell in cells:
        counts[cell["config"]["dataset"]] += 1

    parts = [
        "# Deliverable tables — frozen last-layer initialization study",
        "",
        "Generated by `build_meeting_tables.py`. Table numbers follow "
        "`MEETING_PLAN.md` §5. Any table whose inputs are not on disk yet says "
        "so instead of quietly dropping rows.",
        "",
        "## Coverage",
        "",
        "| dataset | completed screen cells |",
        "|---|---|",
    ]
    for dataset in DATASET_ORDER:
        parts.append(f"| {dataset} | {counts[dataset]} |")
    parts += [
        f"| imagenet1k | {len(imagenet)} report rows |",
        "",
        "## Table 1 — head × dataset headline (full protocol)",
        "",
        table_headline(rows, imagenet),
        "## Table 2 — initialization delta",
        "",
        "### At 20 epochs (screen), each arm at its own best gain and sigma_v",
        "",
        table_screen_init_delta(cells),
        "### The warm start, before any training",
        "",
        table_warm_start(cells, rows),
        "## Table 3 — gain sensitivity",
        "",
        "Read this as *which prior width each arm wants*, not which arm wins: "
        "`random` improves as gain rises, `backbone` as it falls, and `zero` is "
        "flat. That interaction is why the two axes cannot be tuned separately.",
        "",
        table_axis_sensitivity(cells, "gain_w", study["tied_gains"]),
        "## Table 4 — sigma_v sensitivity",
        "",
        table_axis_sensitivity(cells, "sigma_v", study["sigma_v"]),
        "## Table 2b — ImageNet-1k (accuracy and calibration only)",
        "",
        table_imagenet(imagenet),
        "## Table 5 — calibration × initialization (full-tree hrc)",
        "",
        table_calibration(rows),
        "## Table 6 — post-hoc HSM calibration on HRC: which sharing level wins",
        "",
        "### 6a — this study's axis 4, on the confirmed `hrc:full` cells",
        "",
        table_hsm_axis4(rows, confirm_group_counts(manifest)),
        "### 6b — the finished sweep: every method against the fitting budget",
        "",
        table_hsm_sweep("test_nll", "NLL"),
        "### 6c — and what each method does to OOD detection",
        "",
        table_hsm_ood(),
        "### 6d — the same sweep on ECE, which tells a different story",
        "",
        table_hsm_sweep("test_ece", "ECE"),
        "## Appendix A — every confirmed combination: accuracy, calibration, OOD",
        "",
        table_all_confirmed(rows),
        "## Appendix B — the complete screen grid (accuracy and calibration)",
        "",
        table_all_screen(cells),
        "## Appendix C — the complete ImageNet grid",
        "",
        table_all_imagenet(imagenet),
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("MEETING_TABLES.md")),
        help="'-' writes to stdout",
    )
    args = parser.parse_args()
    text = build(load_manifest(args.manifest))
    if args.output == "-":
        print(text)
        return
    Path(args.output).write_text(text)
    print(f"wrote {args.output} ({len(text)} chars)")


if __name__ == "__main__":
    main()
