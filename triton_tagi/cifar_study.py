"""Reproducible CIFAR feature extraction for frozen last-layer studies.

This module is intentionally not imported from triton_tagi.__init__: torchvision
is an optional example dependency, while the core TAGI package remains usable
without it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# cuBLAS reads this before its first handle is created. Set it at optional
# study-module import time, before importing/initializing torch CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18


CANONICAL_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)


@dataclass(frozen=True)
class CifarSpec:
    name: str
    num_classes: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    backbone_accuracy_gate: float


SPECS = {
    "cifar10": CifarSpec(
        "cifar10",
        10,
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
        0.90,
    ),
    "cifar100": CifarSpec(
        "cifar100",
        100,
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761),
        0.65,
    ),
}


def get_spec(dataset: str) -> CifarSpec:
    try:
        return SPECS[dataset.lower()]
    except KeyError as error:
        raise ValueError(f"dataset must be one of {sorted(SPECS)}") from error


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


class CifarResNet18(nn.Module):
    """Torchvision ResNet-18 with a CIFAR 3x3 stem and exposed features."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.network = resnet18(weights=None, num_classes=num_classes)
        self.network.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.network.maxpool = nn.Identity()

    @property
    def classifier(self) -> nn.Linear:
        return self.network.fc

    def forward_features(self, inputs: Tensor) -> Tensor:
        net = self.network
        x = net.conv1(inputs)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)
        x = net.layer4(x)
        x = net.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.forward_features(inputs))


def evaluation_transform(spec: CifarSpec):
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(spec.mean, spec.std)]
    )


def training_transform(spec: CifarSpec):
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ]
    )


def clean_datasets(
    dataset: str,
    data_root: str | Path,
    *,
    augment_train: bool,
    download: bool = True,
):
    spec = get_spec(dataset)
    dataset_type = datasets.CIFAR10 if spec.name == "cifar10" else datasets.CIFAR100
    train = dataset_type(
        data_root,
        train=True,
        download=download,
        transform=training_transform(spec) if augment_train else evaluation_transform(spec),
    )
    test = dataset_type(
        data_root,
        train=False,
        download=download,
        transform=evaluation_transform(spec),
    )
    return train, test


def svhn_dataset(dataset: str, data_root: str | Path, *, download: bool = True):
    spec = get_spec(dataset)
    return datasets.SVHN(
        data_root,
        split="test",
        download=download,
        transform=evaluation_transform(spec),
    )


class CifarCorruption(Dataset):
    """One canonical CIFAR-C corruption at one severity."""

    def __init__(
        self,
        root: str | Path,
        dataset: str,
        corruption: str,
        severity: int,
    ) -> None:
        spec = get_spec(dataset)
        if corruption not in CANONICAL_CORRUPTIONS:
            raise ValueError(f"unknown canonical corruption {corruption!r}")
        if severity not in range(1, 6):
            raise ValueError("severity must lie in [1, 5]")
        root = Path(root)
        image_path = root / f"{corruption}.npy"
        label_path = root / "labels.npy"
        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(
                f"expected CIFAR-C files {image_path} and {label_path}"
            )
        all_images = np.load(image_path, mmap_mode="r")
        all_labels = np.load(label_path, mmap_mode="r")
        start, stop = (severity - 1) * 10_000, severity * 10_000
        if all_images.shape[0] < stop:
            raise ValueError(f"{image_path} does not contain five 10k severity blocks")
        self.images = all_images[start:stop]
        self.labels = all_labels[:10_000]
        self.transform = evaluation_transform(spec)

    def __len__(self) -> int:
        return 10_000

    def __getitem__(self, index: int):
        return self.transform(np.asarray(self.images[index])), int(self.labels[index])


def stratified_split(labels: Tensor, validation_size: int, seed: int) -> tuple[Tensor, Tensor]:
    """Return deterministic train/validation indices with per-class allocation."""

    labels = labels.detach().cpu().long()
    if labels.dim() != 1 or not 0 < validation_size < labels.numel():
        raise ValueError("validation_size must lie between zero and the sample count")
    classes = labels.unique(sorted=True)
    generator = torch.Generator().manual_seed(seed)
    base, remainder = divmod(validation_size, classes.numel())
    validation_parts = []
    train_parts = []
    for offset, class_index in enumerate(classes.tolist()):
        indices = torch.nonzero(labels == class_index, as_tuple=False).flatten()
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        count = base + int(offset < remainder)
        validation_parts.append(indices[:count])
        train_parts.append(indices[count:])
    return torch.cat(train_parts).sort().values, torch.cat(validation_parts).sort().values


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


@torch.no_grad()
def extract_frozen_features(
    model: CifarResNet18,
    dataset: Dataset,
    *,
    device: str | torch.device,
    batch_size: int = 256,
    workers: int = 4,
) -> dict[str, Tensor]:
    """Extract CPU feature/logit/label tensors without autograd."""

    device = torch.device(device)
    model.to(device).eval().requires_grad_(False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    feature_parts, logit_parts, label_parts = [], [], []
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        features = model.forward_features(inputs)
        logits = model.classifier(features)
        feature_parts.append(features.cpu())
        logit_parts.append(logits.cpu())
        label_parts.append(torch.as_tensor(labels).long().cpu())
    return {
        "features": torch.cat(feature_parts),
        "logits": torch.cat(logit_parts),
        "labels": torch.cat(label_parts),
    }


def save_feature_shard(
    path: str | Path,
    tensors: dict[str, Tensor],
    metadata: dict[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **{key: value.detach().cpu().contiguous() for key, value in tensors.items()},
        "metadata": metadata,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_feature_shard(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"features", "logits", "labels", "metadata"}
    if not required.issubset(payload):
        raise ValueError(f"feature shard is missing {sorted(required - set(payload))}")
    if expected_fingerprint is not None:
        actual = payload["metadata"].get("fingerprint")
        if actual != expected_fingerprint:
            raise ValueError(
                f"feature fingerprint mismatch: expected {expected_fingerprint}, got {actual}"
            )
    return payload


def feature_metadata(
    *,
    dataset: str,
    split: str,
    checkpoint: str | Path,
    transform: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = {
        "dataset": dataset,
        "split": split,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "transform": transform,
        **(extra or {}),
    }
    return {**data, "fingerprint": stable_hash(data, length=32)}
