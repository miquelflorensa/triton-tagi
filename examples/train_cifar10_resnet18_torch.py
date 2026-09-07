"""Train a deterministic PyTorch ResNet-18 checkpoint for the TAGI example.

Usage:
    python examples/train_cifar10_resnet18_torch.py --epochs 100
    python examples/train_cifar10_resnet18_torch.py --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from cifar10_torch_common import CifarResNet18, cifar10_datasets


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        prediction = model(inputs.to(device)).argmax(1).cpu()
        correct += prediction.eq(labels).sum().item()
        total += labels.numel()
    return correct / total


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_set, test_set = cifar10_datasets(args.data_dir, augment_train=True)
    if args.smoke:
        train_set = Subset(train_set, range(min(1024, len(train_set))))
        test_set = Subset(test_set, range(min(512, len(test_set))))
        args.epochs = 1
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = CifarResNet18().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = samples = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
            samples += labels.numel()
        scheduler.step()
        accuracy = evaluate(model, test_loader, device)
        row = {"epoch": epoch, "train_loss": loss_sum / samples, "test_accuracy": accuracy}
        history.append(row)
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss={row['train_loss']:.4f}  "
            f"test_acc={accuracy:.2%}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "cifar_resnet18",
        "num_classes": 10,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "normalization": "cifar10",
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": args.epochs,
            "config": config,
        },
        output,
    )
    output.with_suffix(".json").write_text(
        json.dumps({"config": config, "history": history}, indent=2)
    )
    print(f"saved checkpoint: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/deterministic_cifar10_resnet18.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
