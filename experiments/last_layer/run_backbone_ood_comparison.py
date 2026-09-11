"""Does OOD detection depend on the backbone, holding everything else fixed?

The frozen last-layer study reads one representation: a CIFAR ResNet-18 trained
in-repo. Every OOD number it reports is therefore a statement about that
backbone as much as about the head, and nothing in the study separates the two.
This driver adds a second, architecturally unrelated backbone over the same
data, the same evaluation transform, and the same SVHN OOD source, so the
representation is the only thing that moves.

``cifar10_repvgg_a2`` is pretrained (``chenyaofo/pytorch-cifar-models``), so no
training is required. It reaches 0.9486 test accuracy against the study
ResNet-18's ~0.953 and pools to 1408 features rather than 512 -- close in
accuracy, far apart in representation geometry, which is the contrast worth
measuring.

Two caveats this script records rather than hides:

* the pretrained backbone saw all 50k CIFAR-10 training images, so the study's
  10k validation split is backbone-seen data for it and its in-distribution
  numbers are optimistic in a way the ResNet-18's are not. SVHN is unseen by
  both, so the OOD comparison itself is unaffected;
* the two backbones differ in accuracy by ~0.5 points, and OOD separability
  correlates with accuracy, so a small AUROC gap is not attributable to
  architecture on its own.

Stage ``extract`` caches features and logits; stage ``compare`` scores the
deterministic softmax OOD baselines for both backbones side by side.

Usage:
    python experiments/last_layer/run_backbone_ood_comparison.py extract
    python experiments/last_layer/run_backbone_ood_comparison.py compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from triton_tagi.cifar_study import (  # noqa: E402
    clean_datasets,
    extract_frozen_features,
    feature_metadata,
    get_spec,
    load_feature_shard,
    save_feature_shard,
    seed_everything,
    stratified_split,
    svhn_dataset,
)
from triton_tagi.metrics import (  # noqa: E402
    classification_metrics,
    evaluate_ood_comprehensive,
)

STUDY_ROOT = REPOSITORY_ROOT / "runs/last_layer/cifar_frozen_last_layer_v2_40k10k"
OUTPUT_ROOT = REPOSITORY_ROOT / "runs/last_layer/backbone_ood"
DATA_ROOT = REPOSITORY_ROOT / "data"

# Matches study.json's split block, so the two backbones partition identically.
SPLIT_SEED = 2026
VALIDATION_SIZE = 10_000

HUB_REPO = "chenyaofo/pytorch-cifar-models"


class HubBackbone(nn.Module):
    """Adapt a torch.hub CIFAR model to the study's feature/classifier contract.

    ``extract_frozen_features`` only needs ``forward_features`` and
    ``classifier``. Rather than hand-transcribe each architecture's forward
    pass -- which is how a silent off-by-one-layer bug gets in -- the final
    Linear is swapped for an Identity and the module's own forward becomes the
    feature extractor.
    """

    def __init__(self, entrypoint: str) -> None:
        super().__init__()
        self.entrypoint = entrypoint
        model = torch.hub.load(
            HUB_REPO, entrypoint, pretrained=True, trust_repo=True, verbose=False
        )
        name, final = self._final_linear(model)
        self._classifier = nn.Linear(final.in_features, final.out_features)
        self._classifier.load_state_dict(final.state_dict())
        setattr(model, name, nn.Identity())
        self.trunk = model
        self.feature_dim = final.in_features
        self.checkpoint = self._checkpoint_path(entrypoint)

    @staticmethod
    def _checkpoint_path(entrypoint: str) -> Path:
        """The downloaded weight file, so the shard records what it was built from."""

        cache = Path(torch.hub.get_dir()) / "checkpoints"
        matches = sorted(cache.glob(f"{entrypoint}-*.pt"))
        if not matches:
            raise FileNotFoundError(f"no cached weights matching {entrypoint}-*.pt under {cache}")
        return matches[-1]

    @staticmethod
    def _final_linear(model: nn.Module) -> tuple[str, nn.Linear]:
        candidates = [
            (name, module) for name, module in model.named_children() if isinstance(module, nn.Linear)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"expected exactly one top-level Linear to detach, found "
                f"{[name for name, _ in candidates]}; this adapter does not "
                "handle multi-layer classifier stacks"
            )
        return candidates[0]

    @property
    def classifier(self) -> nn.Linear:
        return self._classifier

    def forward_features(self, inputs: Tensor) -> Tensor:
        return torch.flatten(self.trunk(inputs), 1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.forward_features(inputs))


def feature_root(entrypoint: str) -> Path:
    return OUTPUT_ROOT / "features" / entrypoint


def extract(args: argparse.Namespace) -> None:
    seed_everything(SPLIT_SEED)
    device = torch.device(args.device)
    spec = get_spec("cifar10")
    backbone = HubBackbone(args.entrypoint).to(device).eval().requires_grad_(False)

    train_set, test_set = clean_datasets(
        "cifar10", DATA_ROOT, augment_train=False, download=not args.no_download
    )
    labels = torch.as_tensor(train_set.targets).long()
    train_indices, validation_indices = stratified_split(labels, VALIDATION_SIZE, SPLIT_SEED)

    root = feature_root(args.entrypoint)
    accuracies: dict[str, float] = {}

    train_all = extract_frozen_features(
        backbone, train_set, device=device, batch_size=args.batch_size, workers=args.workers
    )
    for name, indices in (("train", train_indices), ("validation", validation_indices)):
        tensors = {key: value[indices] for key, value in train_all.items()}
        accuracies[name] = float(
            tensors["logits"].argmax(1).eq(tensors["labels"]).float().mean()
        )
        save_feature_shard(
            root / f"{name}.pt",
            tensors,
            feature_metadata(
                dataset="cifar10",
                split=name,
                checkpoint=backbone.checkpoint,
                transform="dataset_evaluation_normalization",
                extra={"backbone": args.entrypoint, "feature_dim": backbone.feature_dim},
            ),
        )
        print(f"{name:<11} {tuple(tensors['features'].shape)} acc={accuracies[name]:.4f}", flush=True)

    for name, dataset in (
        ("test", test_set),
        ("svhn", svhn_dataset("cifar10", DATA_ROOT, download=not args.no_download)),
    ):
        tensors = extract_frozen_features(
            backbone, dataset, device=device, batch_size=args.batch_size, workers=args.workers
        )
        accuracies[name] = float(
            tensors["logits"].argmax(1).eq(tensors["labels"]).float().mean()
        )
        save_feature_shard(
            root / f"{name}.pt",
            tensors,
            feature_metadata(
                dataset="cifar10",
                split=name,
                checkpoint=backbone.checkpoint,
                transform="dataset_evaluation_normalization",
                extra={"backbone": args.entrypoint, "feature_dim": backbone.feature_dim},
            ),
        )
        print(f"{name:<11} {tuple(tensors['features'].shape)} acc={accuracies[name]:.4f}", flush=True)

    (root / "summary.json").write_text(
        json.dumps(
            {
                "backbone": args.entrypoint,
                "feature_dim": backbone.feature_dim,
                "split_seed": SPLIT_SEED,
                "validation_size": VALIDATION_SIZE,
                "accuracy": accuracies,
                "note": "svhn accuracy is meaningless; SVHN labels are not CIFAR classes",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {root}")


def deterministic_scores(root: Path) -> dict[str, Any]:
    """Softmax OOD baselines for one backbone's own frozen classifier."""

    test = load_feature_shard(root / "test.pt")
    svhn = load_feature_shard(root / "svhn.pt")
    id_probabilities = test["logits"].softmax(dim=1)
    ood_probabilities = svhn["logits"].softmax(dim=1)
    return {
        "feature_dim": int(test["features"].shape[1]),
        "id": classification_metrics(id_probabilities, test["labels"]),
        "ood": evaluate_ood_comprehensive(id_probabilities, ood_probabilities),
    }


def compare(args: argparse.Namespace) -> None:
    backbones = {
        "resnet18 (study, in-repo)": STUDY_ROOT / "features/cifar10",
        f"{args.entrypoint} (pretrained)": feature_root(args.entrypoint),
    }
    scored = {name: deterministic_scores(root) for name, root in backbones.items()}

    print("\n# CIFAR-10 -> SVHN, deterministic softmax baselines\n")
    print("Same data, same evaluation transform, same OOD source. Only the "
          "representation moves.\n")
    print("| backbone | feat dim | test acc | test NLL | test ECE |")
    print("|---|---|---|---|---|")
    for name, row in scored.items():
        print(
            f"| {name} | {row['feature_dim']} | {row['id']['accuracy']:.4f} | "
            f"{row['id']['nll']:.4f} | {row['id']['ece']:.4f} |"
        )

    print("\n| backbone | score | AUROC | AUPR-OOD | AUPR-ID | FPR95 |")
    print("|---|---|---|---|---|---|")
    for name, row in scored.items():
        for score_name, metrics in row["ood"].items():
            print(
                f"| {name} | {score_name} | {metrics['auroc']:.4f} | "
                f"{metrics['aupr_ood']:.4f} | {metrics['aupr_id']:.4f} | "
                f"{metrics['fpr95']:.4f} |"
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_ROOT / "deterministic_comparison.json"
    destination.write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "compare"))
    parser.add_argument("--entrypoint", default="cifar10_repvgg_a2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if args.command == "extract":
        extract(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
