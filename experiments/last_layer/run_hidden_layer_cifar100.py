"""Does a TAGI hidden layer close the frozen-head accuracy gap?

Every head in the last-layer study reads the frozen 512-d features through a
single Bayesian linear layer. On CIFAR-100 that head leaves a real gap: `hrc`
converges near 0.725 against a 0.767 deterministic reference, and
`remax_laplace_diag` collapses to 0.380 from a random init. The question here
is whether the gap is the *depth* of the Bayesian head rather than the head's
link function, so every arm is crossed with

    hidden_dims = ()        the study's single-layer head
    hidden_dims = (512,)    Linear(512, 512) + ReLU in front of it

on the same cached features, the same 40k/10k split, and the same screen
protocol (tied gains, 20 epochs, seed 0, selection on validation).

Read the framing before the numbers: the 1-layer architecture is provably
sufficient on these features, because a deterministic softmax over them reaches
0.767 with no hidden layer at all. A win here is therefore evidence about what
the TAGI update can reach, not about what the architecture can represent. The
``deterministic`` arm fits the matching MAP references at both depths so the
ceiling for each depth is measured rather than assumed.

Usage:
    python experiments/last_layer/run_hidden_layer_cifar100.py
    python experiments/last_layer/run_hidden_layer_cifar100.py --heads hrc --epochs 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi import (  # noqa: E402
    TAGILastLayerClassifier,
    classification_metrics,
)
from triton_tagi.cifar_study import load_feature_shard, seed_everything  # noqa: E402

DEFAULT_STUDY_ROOT = REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "hidden_layer_cifar100.json"

HEADS = ("hrc", "remax_lognormal", "remax_laplace_diag")
MEAN_INITS = ("random", "zero")
HIDDEN_ARMS = ((), (512,))
TIED_GAINS = (0.03, 0.1, 0.3, 1.0)
SIGMA_V = (0.03, 0.05, 0.1, 0.3)
PENALTIES = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_split(root: Path, name: str, device: torch.device) -> tuple[Tensor, Tensor]:
    shard = load_feature_shard(root / f"{name}.pt")
    return (
        shard["features"].to(device),
        shard["labels"].to(device).long(),
    )


def run_key(head: str, hidden: tuple[int, ...], mean_init: str, gain: float, sigma_v: float) -> str:
    width = "flat" if not hidden else "x".join(str(w) for w in hidden)
    return f"{head}|{width}|{mean_init}|g{gain}|s{sigma_v}"


# ── The deterministic ceiling at each depth ──────────────────────────────────


def fit_deterministic(
    train: tuple[Tensor, Tensor],
    validation: tuple[Tensor, Tensor],
    test: tuple[Tensor, Tensor],
    *,
    hidden: tuple[int, ...],
    num_classes: int,
    epochs: int,
    penalties: tuple[float, ...],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """MAP softmax over frozen features, at the same depth as the TAGI arm.

    With the features frozen the single-layer fit is convex, so this is the
    honest reference for that arm. The hidden arm is not convex, which is
    exactly why it is fit here rather than assumed to be at least as good.
    """

    features, labels = train
    best: dict[str, Any] | None = None
    for penalty in penalties:
        seed_everything(seed)
        widths = [features.shape[1], *hidden]
        layers: list[nn.Module] = []
        for fan_in, fan_out in zip(widths[:-1], widths[1:]):
            layers.extend([nn.Linear(fan_in, fan_out), nn.ReLU()])
        layers.append(nn.Linear(widths[-1], num_classes))
        model = nn.Sequential(*layers).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=penalty)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for _ in range(epochs):
            model.train()
            order = torch.randperm(features.shape[0], generator=generator).to(device)
            for start in range(0, order.numel(), 256):
                batch = order[start : start + 256]
                optimizer.zero_grad(set_to_none=True)
                criterion(model(features[batch]), labels[batch]).backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            scored = {
                name: classification_metrics(
                    model(split_features).softmax(dim=1), split_labels
                )
                for name, (split_features, split_labels) in (
                    ("validation", validation),
                    ("test", test),
                )
            }
        row = {"penalty": penalty, **{f"{k}_{m}": v for k, s in scored.items() for m, v in s.items()}}
        if best is None or row["validation_nll"] < best["validation_nll"]:
            best = row
    assert best is not None
    return best


# ── The TAGI arms ────────────────────────────────────────────────────────────


def run_cell(
    train: tuple[Tensor, Tensor],
    validation: tuple[Tensor, Tensor],
    test: tuple[Tensor, Tensor],
    *,
    head: str,
    hidden: tuple[int, ...],
    mean_init: str,
    gain: float,
    sigma_v: float,
    num_classes: int,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    features, labels = train
    seed_everything(seed)
    classifier = TAGILastLayerClassifier(
        features.shape[1],
        num_classes,
        head=head,
        device=device,
        hidden_dims=hidden,
        mean_init=mean_init,
        gain_w=gain,
        gain_b=gain,
        sigma_v=sigma_v,
    )
    started = time.perf_counter()
    history = classifier.fit(
        features,
        labels,
        epochs=epochs,
        batch_size=256,
        validation=validation,
        seed=seed,
    )
    scored = {
        name: classification_metrics(
            classifier.predict(split_features).probabilities, split_labels
        )
        for name, (split_features, split_labels) in (("validation", validation), ("test", test))
    }
    return {
        "head": head,
        "hidden_dims": list(hidden),
        "mean_init": mean_init,
        "gain": gain,
        "sigma_v": sigma_v,
        "epochs": epochs,
        "seed": seed,
        "wall_s": time.perf_counter() - started,
        "parameters": classifier.net.num_parameters(),
        "trajectory": [
            {"epoch": row["epoch"], "val_accuracy": row["val_accuracy"], "val_nll": row["val_nll"]}
            for row in history.records
        ],
        **{f"{split}_{name}": value for split, s in scored.items() for name, value in s.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar100")
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--heads", nargs="+", default=list(HEADS))
    parser.add_argument("--mean-inits", nargs="+", default=list(MEAN_INITS))
    parser.add_argument(
        "--hidden",
        nargs="+",
        default=["0", "512"],
        help="hidden widths to cross; 0 means the study's single-layer head",
    )
    parser.add_argument("--gains", nargs="+", type=float, default=list(TIED_GAINS))
    parser.add_argument("--sigma-v", nargs="+", type=float, default=list(SIGMA_V))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--deterministic-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-deterministic",
        action="store_true",
        help="reuse the references already in --output instead of refitting them",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    feature_root = args.study_root / "features" / args.dataset
    train = load_split(feature_root, "train", device)
    validation = load_split(feature_root, "validation", device)
    test = load_split(feature_root, "test", device)
    num_classes = int(train[1].max().item()) + 1
    hidden_arms = [() if width == "0" else (int(width),) for width in args.hidden]

    existing: dict[str, Any] = {}
    if args.output.exists():
        existing = json.loads(args.output.read_text())
    results: dict[str, Any] = existing.get("runs", {})
    references: dict[str, Any] = existing.get("deterministic", {})

    payload = {
        "dataset": args.dataset,
        "feature_root": str(feature_root),
        "num_classes": num_classes,
        "split": {name: int(tensors[1].numel()) for name, tensors in
                  (("train", train), ("validation", validation), ("test", test))},
        "protocol": {
            "epochs": args.epochs,
            "deterministic_epochs": args.deterministic_epochs,
            "seed": args.seed,
            "batch_size": 256,
            "selection": "lowest validation NLL within an arm; test read for the selected cell",
        },
        "deterministic": references,
        "runs": results,
    }

    if not args.skip_deterministic:
        for hidden in hidden_arms:
            name = "flat" if not hidden else "x".join(str(w) for w in hidden)
            if name in references:
                continue
            started = time.perf_counter()
            references[name] = fit_deterministic(
                train,
                validation,
                test,
                hidden=hidden,
                num_classes=num_classes,
                epochs=args.deterministic_epochs,
                penalties=PENALTIES,
                seed=args.seed,
                device=device,
            )
            print(
                f"deterministic {name:>8}  "
                f"val_acc={references[name]['validation_accuracy']:.4f} "
                f"test_acc={references[name]['test_accuracy']:.4f} "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
            atomic_json(args.output, payload)

    grid = list(
        itertools.product(args.heads, hidden_arms, args.mean_inits, args.gains, args.sigma_v)
    )
    print(f"{len(grid)} cells", flush=True)
    for index, (head, hidden, mean_init, gain, sigma_v) in enumerate(grid, start=1):
        key = run_key(head, hidden, mean_init, gain, sigma_v)
        if key in results:
            continue
        results[key] = run_cell(
            train,
            validation,
            test,
            head=head,
            hidden=hidden,
            mean_init=mean_init,
            gain=gain,
            sigma_v=sigma_v,
            num_classes=num_classes,
            epochs=args.epochs,
            seed=args.seed,
            device=device,
        )
        row = results[key]
        print(
            f"[{index:3d}/{len(grid)}] {key:<52} "
            f"val_acc={row['validation_accuracy']:.4f} val_nll={row['validation_nll']:.4f} "
            f"({row['wall_s']:.0f}s)",
            flush=True,
        )
        atomic_json(args.output, payload)

    atomic_json(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
