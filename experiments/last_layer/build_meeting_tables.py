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

HEAD_ORDER = ("hrc", "hrc:full", "remax_lognormal", "remax_laplace_diag", "logit_tagiv")
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


def arm_name(head: str, hrc_tree: Any) -> str:
    """The label a head is reported under, matching run_study.selection_arm."""

    tree = (hrc_tree or "").strip() if isinstance(hrc_tree, str) else ""
    if not tree or tree == default_hrc_tree(head):
        return head
    return f"{head}:{tree}"


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
        cells.append(
            {
                "config": config,
                "arm": arm_name(config["head"], config.get("hrc_tree")),
                "record": final[-1],
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
            arm_name(row["head"], row.get("hrc_tree")),
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
        "column is therefore a CIFAR-only result."
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
        arm = arm_name(row["head"], row.get("hrc_tree"))
        if not arm.endswith(":full"):
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
        "## Table 5 — calibration × initialization (full-tree hrc)",
        "",
        table_calibration(rows),
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
