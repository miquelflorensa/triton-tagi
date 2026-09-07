"""Print every ImageNet AGCI run as one comparable table.

Reads ``result.json`` (full 50k validation) from each run directory under the
runs root, falling back to the last ``history.json`` record (screening cohort)
for runs that skipped the final evaluation or are still evaluating. Cohort rows
are marked so they are never silently compared against full-validation rows.

    python experiments/imagenet/summarize_runs.py
    python experiments/imagenet/summarize_runs.py --sort nll
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPOSITORY_ROOT / "runs/imagenet/resnet18_agci"

# The frozen torchvision head over the same features, on all 50k images.
BASELINE = {
    "name": "softmax fc (control)",
    "source": "50k",
    "accuracy": 0.6976,
    "top5_accuracy": 0.89082,
    "nll": 1.24692,
    "ece": 0.026333,
    "gap": 0.022639,
    "brier": 0.41110,
    "aurc": 0.101396,
    "error_auroc": None,
}


def read_run(directory: Path) -> dict | None:
    """Return one comparable row for a run directory, or None if it has none."""

    result = directory / "result.json"
    if result.is_file():
        payload = json.loads(result.read_text())
        validation = payload.get("validation")
        if validation:
            variance = payload.get("output_variance", {})
            return {
                "name": directory.name,
                "source": "50k",
                "config": payload.get("config", {}),
                "gap": validation["mean_confidence"] - validation["accuracy"],
                "error_auroc": variance.get("error_auroc"),
                **validation,
            }

    history = directory / "history.json"
    if history.is_file():
        records = json.loads(history.read_text())
        if records:
            last = records[-1]
            return {
                "name": f"{directory.name} (ep {int(last['epoch'])})",
                "source": "10k",
                "config": {},
                "accuracy": last["val_accuracy"],
                "top5_accuracy": last["val_top5_accuracy"],
                "nll": last["val_nll"],
                "ece": last["val_ece"],
                "brier": last["val_brier"],
                "aurc": last["val_aurc"],
                "gap": last["val_mean_confidence"] - last["val_accuracy"],
                "error_auroc": last.get("val_variance_error_auroc"),
            }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--sort",
        choices=("accuracy", "nll", "ece", "name"),
        default="accuracy",
        help="accuracy sorts descending; nll and ece ascending",
    )
    args = parser.parse_args()

    rows = [row for d in sorted(args.root.iterdir()) if d.is_dir() and (row := read_run(d))]
    if not rows:
        print(f"no runs with results under {args.root}")
        return

    if args.sort == "accuracy":
        rows.sort(key=lambda r: -r["accuracy"])
    elif args.sort == "name":
        rows.sort(key=lambda r: r["name"])
    else:
        rows.sort(key=lambda r: r[args.sort])
    rows.insert(0, BASELINE)

    width = max(len(r["name"]) for r in rows) + 2
    header = (
        f"{'run':<{width}}{'set':>5}{'gain':>6}{'tau':>8}{'ep':>4}"
        f"{'top1':>8}{'top5':>8}{'nll':>9}{'ece%':>8}{'gap':>8}"
        f"{'brier':>8}{'aurc':>8}{'errAUC':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        config = row.get("config", {})
        gain = config.get("gain_w")
        tau = config.get("tau")
        epochs = config.get("epochs")
        auroc = row.get("error_auroc")
        print(
            f"{row['name']:<{width}}"
            f"{row['source']:>5}"
            f"{'' if gain is None else f'{gain:g}':>6}"
            f"{'' if tau is None else f'{tau:g}':>8}"
            f"{'' if epochs is None else epochs:>4}"
            f"{row['accuracy'] * 100:>8.2f}"
            f"{row['top5_accuracy'] * 100:>8.2f}"
            f"{row['nll']:>9.4f}"
            f"{row['ece'] * 100:>8.3f}"
            f"{row['gap'] * 100:>+8.2f}"
            f"{row['brier']:>8.4f}"
            f"{row['aurc']:>8.4f}"
            f"{'' if auroc is None else f'{auroc:.4f}':>8}"
        )
    print()
    print("set: 50k = full validation · 10k = screening cohort, not comparable to 50k rows")
    print("gap: mean confidence minus accuracy — negative is underconfident")


if __name__ == "__main__":
    main()
