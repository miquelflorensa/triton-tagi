"""Shared deterministic CIFAR-10 ResNet-18 pieces for the interop examples."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision import datasets, transforms
from torchvision.models import resnet18

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class CifarResNet18(nn.Module):
    """Torchvision ResNet-18 with the conventional CIFAR 3x3 stem."""

    def __init__(self, num_classes: int = 10) -> None:
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


def feature_and_logits(model: nn.Module, inputs: Tensor) -> tuple[Tensor, Tensor]:
    """Representation callback accepted by ``FrozenTorchBackbone``."""

    if not isinstance(model, CifarResNet18):
        raise TypeError("This example callback expects CifarResNet18")
    features = model.forward_features(inputs)
    return features, model.classifier(features)


def evaluation_transform():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )


def training_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def cifar10_datasets(data_dir: str, *, augment_train: bool = True):
    train = datasets.CIFAR10(
        data_dir,
        train=True,
        download=True,
        transform=training_transform() if augment_train else evaluation_transform(),
    )
    test = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=evaluation_transform()
    )
    return train, test


def svhn_dataset(data_dir: str):
    # Use exactly the deterministic classifier's input preprocessing.
    return datasets.SVHN(
        data_dir, split="test", download=True, transform=evaluation_transform()
    )
