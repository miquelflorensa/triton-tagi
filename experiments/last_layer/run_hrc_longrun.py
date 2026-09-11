"""Isolated long-run convergence probe for the best CIFAR-100 `hrc:full` cell.

Every 200-epoch confirm run of this cell is flat over its last twenty epochs
(val NLL moves by 2e-4), and validation accuracy peaks around epoch 6. This
extends the same cell to a 10x budget and calibrates the final checkpoint, to
settle whether the CIFAR-100 gap against softmax is an unconverged head or a
ceiling.

It reuses `run_study.run_configuration` verbatim, so the only difference from
the confirm cell is `epochs`. Output goes to its own `init_longrun` stage; no
artifact of the finished study is read for writing or moved.

    python experiments/last_layer/run_hrc_longrun.py --epochs 2000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
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

from triton_tagi import TAGILastLayerClassifier, classification_metrics  # noqa: E402
from triton_tagi.cifar_study import load_feature_shard  # noqa: E402

STAGE = "init_longrun"
DATASET = "cifar100"

# The cell `select` chose for cifar100 hrc:full with offsets off, copied from
# heads/init_screen/cifar100/selection.json.
CELL = {
    "gain_b": 0.1,
    "gain_w": 0.1,
    "head": "hrc",
    "hrc_prior_offsets": False,
    "hrc_tree": "full",
    "mean_init": "zero",
    "sigma_v": 0.1,
}

# Log-spaced, so the whole trajectory is visible rather than only its tail.
# 200 is kept so the long run can be read against the confirm cell exactly.
CHECKPOINTS = [0, 1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200,
               300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000]


def checkpoints_for(epochs: int) -> list[int]:
    kept = [e for e in CHECKPOINTS if e <= epochs]
    if epochs not in kept:
        kept.append(epochs)
    return kept


def train(manifest: dict, epochs: int, seeds: list[int], args) -> None:
    for seed in seeds:
        run_study.run_configuration(
            manifest, DATASET, STAGE, dict(CELL), seed, epochs,
            checkpoints_for(epochs), args,
        )


def run_dirs(manifest: dict) -> list[Path]:
    root = run_study.stage_root(manifest, STAGE, DATASET)
    return sorted(p.parent for p in root.glob("*/*/config.json"))


def calibrate(manifest: dict, args) -> list[dict]:
    """Uncalibrated and HSM-calibrated test metrics at the final checkpoint.

    Mirrors `run_study.calibrate_stage` but scores the clean test split only:
    the corruption suite is what makes that stage slow, and convergence is the
    question here, not shift robustness.
    """
    features = run_study.feature_root(manifest, DATASET)
    validation = load_feature_shard(features / "validation.pt")
    test = load_feature_shard(features / "test.pt")
    results = []

    for run_dir in run_dirs(manifest):
        config = json.loads((run_dir / "config.json").read_text())
        epochs = int(config["epochs"])
        checkpoint = run_dir / "checkpoints" / f"epoch_{epochs:04d}.pt"
        if not checkpoint.exists():
            print(f"no final checkpoint in {run_dir}")
            continue

        classifier, _ = TAGILastLayerClassifier.load(checkpoint, device=args.device)
        probabilities, _ = run_study.predict_batches(
            classifier, test["features"], args.batch_size
        )
        metrics = classification_metrics(probabilities, test["labels"])
        results.append({
            "seed": config["seed"], "epochs": epochs, "calibration": "uncalibrated",
            "groups": None, **{k: float(v) for k, v in metrics.items()},
        })
        print(f"  uncalibrated  seed{config['seed']}  ep{epochs}  "
              f"nll={metrics['nll']:.4f}")

        for sharing in args.sharing:
            started = time.time()
            classifier, _ = TAGILastLayerClassifier.load(checkpoint, device=args.device)
            posterior = classifier.calibrate_hsm_log_gain(
                validation["features"], validation["labels"],
                sharing=sharing, method="grid", batch_size=args.batch_size,
            )
            probabilities, _ = run_study.calibrated_predict_batches(
                classifier, test["features"], args.batch_size, posterior
            )
            metrics = classification_metrics(probabilities, test["labels"])
            results.append({
                "seed": config["seed"], "epochs": epochs, "calibration": sharing,
                "groups": int(posterior.groups.n_groups),
                **{k: float(v) for k, v in metrics.items()},
            })
            print(f"  {sharing:12}  seed{config['seed']}  ep{epochs}  "
                  f"nll={metrics['nll']:.4f}  ({time.time() - started:.1f}s)")
    return results


def trajectory(manifest: dict) -> None:
    """Print the val curve at the checkpoint epochs, to read convergence off."""
    for run_dir in run_dirs(manifest):
        config = json.loads((run_dir / "config.json").read_text())
        history = json.loads((run_dir / "history.json").read_text())
        marks = set(checkpoints_for(int(config["epochs"])))
        print(f"\n  seed {config['seed']} — {config['epochs']} epochs")
        print(f"  {'epoch':>6} {'val_acc':>8} {'val_nll':>8} {'val_ece':>8} {'epistemic':>10}")
        for row in history:
            if int(row["epoch"]) in marks:
                print(f"  {int(row['epoch']):6d} {row['val_accuracy']:8.4f} "
                      f"{row['val_nll']:8.4f} {row['val_ece']:8.4f} "
                      f"{row['epistemic_mean']:10.2e}")
        tail = [r for r in history if r["epoch"] >= history[-1]["epoch"] - 20]
        print(f"  last 20 epochs: dNLL={tail[-1]['val_nll'] - tail[0]['val_nll']:+.6f}  "
              f"dACC={tail[-1]['val_accuracy'] - tail[0]['val_accuracy']:+.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(run_study.DEFAULT_MANIFEST))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sharing", nargs="+", default=["global", "level", "node"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    manifest = run_study.load_manifest(args.manifest)
    if not args.skip_train:
        print(f"=== train {DATASET} hrc:full zero gain=0.1 sigma_v=0.1 "
              f"— {args.epochs} epochs, seeds {args.seeds} ===", flush=True)
        train(manifest, args.epochs, args.seeds, args)

    print("\n=== validation trajectory ===")
    trajectory(manifest)

    print("\n=== test metrics at the final checkpoint ===", flush=True)
    results = calibrate(manifest, args)

    destination = run_study.stage_root(manifest, STAGE, DATASET) / "longrun_metrics.json"
    run_study.atomic_json(destination, results)
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
