"""Extend the CIFAR-100 HRC screen into the (gain, sigma_v) corners it skipped.

The init study swept sigma_v in {0.05, 0.1, 0.3} and gain in {0.03..1.0}. The
older `screen` stage did reach sigma_v 0.01 and 0.03, but only on the padded
tree, and neither stage tried gain 3.0. This fills both edges for both trees
and all three mean inits, so the claim that the optimum is interior rests on a
grid that actually brackets it.

Writes to its own `init_screen_ext` stage: the finished study's screen, its
selection and its tables are untouched.

    python experiments/last_layer/run_hrc_grid_extension.py --dataset cifar100
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

_spec = importlib.util.spec_from_file_location("run_study", HERE / "run_study.py")
run_study = importlib.util.module_from_spec(_spec)
sys.modules["run_study"] = run_study
_spec.loader.exec_module(run_study)

STAGE = "init_screen_ext"

# The corners the two existing screens leave open. sigma_v 0.01/0.03 were only
# ever run on the padded tree; gain 3.0 was never run at all in this study.
NEW_SIGMA_V = [0.01, 0.03]
NEW_GAIN = [3.0]
OLD_SIGMA_V = [0.05, 0.1, 0.3]
OLD_GAIN = [0.03, 0.1, 0.3, 1.0]
MEAN_INIT = ["random", "zero", "backbone"]
TREES = [
    {"hrc_tree": "padded"},
    {"hrc_tree": "full", "hrc_prior_offsets": False},
]


def grid(sigma_v: list[float], gains: list[float]) -> list[dict]:
    """Every cell in (gain x sigma_v x mean_init x tree) not already on disk.

    The new region is the union of the new sigma_v rows at every gain and the
    new gain column at every sigma_v, minus nothing: their intersection is new
    on both axes and belongs in exactly once.
    """
    pairs = set(itertools.product(gains, sigma_v + OLD_SIGMA_V))
    pairs |= set(itertools.product(gains + OLD_GAIN, sigma_v))
    configs = []
    for (gain, sv), mean_init, tree in itertools.product(
        sorted(pairs), MEAN_INIT, TREES
    ):
        base = {"head": "hrc", "sigma_v": sv, "gain_w": gain, "gain_b": gain,
                "mean_init": mean_init}
        configs.append({**base, **tree})
    return configs


def report(manifest: dict, dataset: str) -> None:
    """New cells beside the best the finished screen found."""
    root = run_study.stage_root(manifest, STAGE, dataset)
    rows = []
    for path in sorted(root.glob("*/*/config.json")):
        run_dir = path.parent
        config = json.loads(path.read_text())
        history_path = run_dir / "history.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text())
        # Epoch 0 is the untrained prior, never a selectable cell.
        trained = [r for r in history if r["epoch"] > 0]
        if not trained:
            continue
        best = min(trained, key=lambda r: r["val_nll"])
        arm = "hrc:full" if config.get("hrc_tree") == "full" else "hrc"
        rows.append((best["val_nll"], arm, config["mean_init"], config["gain_w"],
                     config["sigma_v"], best["val_accuracy"], int(best["epoch"])))

    print(f"\n=== {dataset}: {len(rows)} extension cells, best NLL first ===")
    print(f"{'arm':10} {'init':9} {'gain':6} {'sigma_v':8} {'val_nll':8} {'val_acc':8} {'ep':>3}")
    for nll, arm, init, gain, sv, acc, ep in sorted(rows)[:20]:
        print(f"{arm:10} {init:9} {gain:<6} {sv:<8} {nll:.4f}   {acc:.4f}   {ep:3d}")

    incumbent = {
        "cifar100": ("hrc zero gain0.1 sv0.1", 1.3040, "hrc:full zero gain0.1 sv0.1", 1.3127),
        "cifar10": ("hrc zero gain0.3 sv0.3", 0.1662, "hrc:full random gain0.3 sv0.3", 0.1696),
    }.get(dataset)
    if incumbent and rows:
        best_new = min(rows)
        print(f"\nbest extension cell : {best_new[1]} {best_new[2]} gain{best_new[3]} "
              f"sv{best_new[4]} -> {best_new[0]:.4f}")
        print(f"finished screen best: {incumbent[0]} -> {incumbent[1]:.4f} "
              f"(padded) | {incumbent[2]} -> {incumbent[3]:.4f} (full)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(run_study.DEFAULT_MANIFEST))
    parser.add_argument("--dataset", default="cifar100", choices=("cifar10", "cifar100"))
    parser.add_argument("--sigma-v", type=float, nargs="+", default=NEW_SIGMA_V)
    parser.add_argument("--gains", type=float, nargs="+", default=NEW_GAIN)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    manifest = run_study.load_manifest(args.manifest)
    checkpoints = manifest["init_study"]["screen"]["checkpoints"]

    if not args.report_only:
        configs = grid(args.sigma_v, args.gains)
        print(f"=== {args.dataset}: {len(configs)} extension cells, "
              f"{args.epochs} epochs, seed {args.seed} ===", flush=True)
        for index, config in enumerate(configs, 1):
            print(f"[{index}/{len(configs)}] {config}", flush=True)
            run_study.run_configuration(
                manifest, args.dataset, STAGE, config, args.seed,
                args.epochs, checkpoints, args,
            )

    report(manifest, args.dataset)


if __name__ == "__main__":
    main()
