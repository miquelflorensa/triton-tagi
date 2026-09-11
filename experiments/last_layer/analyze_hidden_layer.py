"""Summarize the hidden-layer arm against the single-layer head it replaces.

Selection follows the study protocol: within each (head, depth, mean_init) arm
the cell with the lowest validation NLL wins, and only that cell's test numbers
are reported. The deterministic MAP reference at each depth is the ceiling the
arm is measured against, so a row's headroom is `reference - test_accuracy` at
its own depth, not against the single-layer reference.

Usage:
    python experiments/last_layer/analyze_hidden_layer.py
    python experiments/last_layer/analyze_hidden_layer.py --select accuracy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path(__file__).resolve().parent / "hidden_layer_cifar100.json"


def depth_name(hidden_dims: list[int]) -> str:
    return "flat" if not hidden_dims else "x".join(str(width) for width in hidden_dims)


def select(rows: list[dict[str, Any]], criterion: str) -> dict[str, Any]:
    if criterion == "nll":
        return min(rows, key=lambda row: row["validation_nll"])
    return max(rows, key=lambda row: row["validation_accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--select", choices=("nll", "accuracy"), default="nll")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    runs = list(payload["runs"].values())
    references = payload["deterministic"]

    arms: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in runs:
        arms.setdefault(
            (row["head"], depth_name(row["hidden_dims"]), row["mean_init"]), []
        ).append(row)

    print(f"# {payload['dataset']} — TAGI hidden layer vs the single-layer head")
    print(f"\nSelection: lowest validation {args.select} within an arm. "
          f"{len(runs)} cells over {len(arms)} arms.\n")

    print("## Deterministic MAP reference on the same frozen features\n")
    print("| depth | val acc | test acc | test NLL |")
    print("|---|---|---|---|")
    for name, row in references.items():
        print(
            f"| {name} | {row['validation_accuracy']:.4f} | "
            f"{row['test_accuracy']:.4f} | {row['test_nll']:.4f} |"
        )

    print("\n## Selected cell per arm\n")
    print("| head | init | depth | gain | sigma_v | test acc | test NLL | test ECE "
          "| headroom | params |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    deltas: list[tuple[str, str, float, float]] = []
    for head, init in sorted({(key[0], key[2]) for key in arms}):
        chosen: dict[str, dict[str, Any]] = {}
        for depth in ("flat", *(name for name in references if name != "flat")):
            rows = arms.get((head, depth, init))
            if not rows:
                continue
            best = select(rows, args.select)
            chosen[depth] = best
            reference = references.get(depth, {}).get("test_accuracy")
            headroom = (
                "—" if reference is None else f"{reference - best['test_accuracy']:+.4f}"
            )
            print(
                f"| `{head}` | {init} | {depth} | {best['gain']} | {best['sigma_v']} | "
                f"{best['test_accuracy']:.4f} | {best['test_nll']:.4f} | "
                f"{best['test_ece']:.4f} | {headroom} | {best['parameters']:,} |"
            )
        if "flat" in chosen:
            for depth, best in chosen.items():
                if depth == "flat":
                    continue
                deltas.append(
                    (
                        f"{head} / {init} / {depth}",
                        "accuracy",
                        best["test_accuracy"] - chosen["flat"]["test_accuracy"],
                        best["test_nll"] - chosen["flat"]["test_nll"],
                    )
                )

    print("\n## What the hidden layer changed\n")
    print("| arm | test accuracy delta | test NLL delta |")
    print("|---|---|---|")
    for label, _, accuracy_delta, nll_delta in deltas:
        print(f"| {label} | {accuracy_delta:+.4f} | {nll_delta:+.4f} |")


if __name__ == "__main__":
    main()
