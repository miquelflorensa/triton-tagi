"""End-to-end TAGI ResNet-18 on CIFAR-10 with a direct-probit ``ProbiTree``.

The model has exactly nine Gaussian output scores for CIFAR-10. Training
observes the root-to-label paths of a minibatch and performs one TAGI backward
update from the assembled direct-probit ``g/h`` messages, summing the parameter
deltas over the batch as every other TAGI head here does. The branch-noise
variance ``r`` is fixed for initialization, training, and prediction.

Examples:
    python examples/cifar10_resnet18_probitree.py --smoke --no-augment
    python examples/cifar10_resnet18_probitree.py --epochs 100 --probitree-r 1
    python examples/cifar10_resnet18_probitree.py --epochs 20 --probitree-r 0.25
    python examples/cifar10_resnet18_probitree.py --batch-size 1  # old sequential mode
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from triton_tagi import (
    AvgPool2D,
    BatchNorm2D,
    Conv2D,
    Flatten,
    Linear,
    ProbiTree,
    ReLU,
    ResBlock,
    RunDir,
    Sequential,
    classification_metrics,
    probitree_log_probs,
    probitree_uniform_reference_means,
)

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(
    data_dir: str,
    device: torch.device,
    *,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load normalized CIFAR-10 tensors, optionally bounded for smoke tests."""

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)]
    )
    train_data = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    train_count = (
        len(train_data) if max_train_samples is None else min(max_train_samples, len(train_data))
    )
    test_count = (
        len(test_data) if max_test_samples is None else min(max_test_samples, len(test_data))
    )

    x_train = torch.stack([train_data[index][0] for index in range(train_count)]).to(device)
    y_train = torch.tensor([train_data[index][1] for index in range(train_count)], device=device)
    x_test = torch.stack([test_data[index][0] for index in range(test_count)]).to(device)
    y_test = torch.tensor([test_data[index][1] for index in range(test_count)], device=device)
    return x_train, y_train, x_test, y_test


def gpu_augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Apply random horizontal flip and reflected-pad crop on the GPU."""

    batch, channels, height, width = x.shape
    flip = torch.rand(batch, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    top = torch.randint(0, 2 * pad + 1, (batch,), device=x.device)
    left = torch.randint(0, 2 * pad + 1, (batch,), device=x.device)
    rows = top[:, None] + torch.arange(height, device=x.device)[None, :]
    columns = left[:, None] + torch.arange(width, device=x.device)[None, :]
    return padded[
        torch.arange(batch, device=x.device)[:, None, None, None],
        torch.arange(channels, device=x.device)[None, :, None, None],
        rows[:, None, :, None].expand(batch, channels, height, width),
        columns[:, None, None, :].expand(batch, channels, height, width),
    ]


def build_resnet18_probitree(
    tree: ProbiTree,
    device: torch.device,
    *,
    gain_w: float,
    gain_b: float,
) -> tuple[Sequential, Linear]:
    """Build the CIFAR ResNet-18 TAGI topology and return its score head."""

    kwargs = {"device": device, "gain_w": gain_w, "gain_b": gain_b}
    output_head = Linear(512, tree.num_gates, **kwargs)
    network = Sequential(
        [
            Conv2D(3, 64, 3, stride=1, padding=1, **kwargs),
            ReLU(),
            BatchNorm2D(64, **kwargs),
            ResBlock(64, 64, stride=1, **kwargs),
            ResBlock(64, 64, stride=1, **kwargs),
            ResBlock(64, 128, stride=2, **kwargs),
            ResBlock(128, 128, stride=1, **kwargs),
            ResBlock(128, 256, stride=2, **kwargs),
            ResBlock(256, 256, stride=1, **kwargs),
            ResBlock(256, 512, stride=2, **kwargs),
            ResBlock(512, 512, stride=1, **kwargs),
            AvgPool2D(4),
            Flatten(),
            output_head,
        ],
        device=device,
    )
    return network, output_head


@torch.no_grad()
def initialize_uniform_reference(
    network: Sequential,
    output_head: Linear,
    tree: ProbiTree,
    inputs: torch.Tensor,
    r: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set biases for uniform average predictions on representative inputs."""

    network.train()
    initial_mean, initial_variance = network.forward(inputs)
    reference_variance = initial_variance.double().mean(dim=0)
    target_mean = probitree_uniform_reference_means(tree, reference_variance, r)
    mean_correction = target_mean - initial_mean.double().mean(dim=0)
    assert output_head.mb is not None
    output_head.mb.add_(mean_correction.to(output_head.mb.dtype).reshape(1, -1))
    return target_mean, reference_variance


@torch.no_grad()
def evaluate(
    network: Sequential,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    tree: ProbiTree,
    r: float,
    batch_size: int,
) -> dict[str, float]:
    """Return classification metrics using analytical probit marginalization."""

    network.eval()
    probability_parts = []
    for start in range(0, len(inputs), batch_size):
        mean, variance = network.forward(inputs[start : start + batch_size])
        log_probabilities = probitree_log_probs(tree, mean, variance, r)
        deviation = torch.expm1(torch.logsumexp(log_probabilities, dim=1)).abs().max()
        if float(deviation) > 1e-8:
            raise ArithmeticError("ProbiTree probabilities do not sum to one")
        probability_parts.append(log_probabilities.exp().float().cpu())
    network.train()
    return classification_metrics(torch.cat(probability_parts), labels.cpu())


def train(
    network: Sequential,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    tree: ProbiTree,
    *,
    epochs: int,
    r: float,
    augment: bool,
    batch_size: int,
    evaluation_batch_size: int,
    checkpoint_interval: int,
    run: RunDir,
    config: dict,
) -> None:
    """Train on minibatched observations and checkpoint each requested epoch."""

    print(f"\n  {'Epoch':>5}  {'Train NLL':>10}  {'Test Acc':>9}  {'Test NLL':>9}  {'Time':>8}")
    print("  " + "─" * 55)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        permutation = torch.randperm(len(x_train), device=x_train.device)
        # Accumulate on the device: reading the scalar per step would stall the
        # CUDA stream once per update and dominate a launch-bound loop.
        evidence_sum = torch.zeros((), dtype=torch.float64, device=x_train.device)
        network.train()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            inputs = x_train[indices]
            if augment:
                inputs = gpu_augment(inputs)
            _, _, log_evidence = network.step_probitree(
                inputs,
                y_train[indices],
                tree,
                r,
            )
            evidence_sum += log_evidence.sum()

        if x_train.device.type == "cuda":
            torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        train_nll = -float(evidence_sum) / len(x_train)
        metrics = evaluate(network, x_test, y_test, tree, r, evaluation_batch_size)
        print(
            f"  {epoch:5d}  {train_nll:10.4f}  {metrics['accuracy'] * 100:8.2f}%"
            f"  {metrics['nll']:9.4f}  {wall_seconds:7.1f}s",
            flush=True,
        )
        run.append_metrics(
            epoch,
            train_pre_update_nll=train_nll,
            test_acc=metrics["accuracy"],
            test_nll=metrics["nll"],
            test_brier=metrics["brier"],
            test_ece=metrics["ece"],
            probitree_r=r,
            wall_s=wall_seconds,
        )
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            run.save_checkpoint(network, epoch, config)


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the Triton ResNet kernels")
    if args.probitree_r <= 0.0:
        raise ValueError("--probitree-r must be positive")
    if args.epochs < 1 or args.evaluation_batch_size < 1 or args.batch_size < 1:
        raise ValueError("epochs, batch size, and evaluation batch size must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    epochs = 1 if args.smoke else args.epochs
    max_train = args.smoke_train_samples if args.smoke else args.max_train_samples
    max_test = args.smoke_test_samples if args.smoke else args.max_test_samples
    tree = ProbiTree(10)

    print("=" * 72)
    print("  CIFAR-10 — end-to-end TAGI ResNet-18 — ProbiTree")
    print(f"  10 leaves, {tree.num_gates} Gaussian gates, fixed r={args.probitree_r:g}")
    print(f"  Training updates are minibatched ({args.batch_size} labeled images per TAGI update).")
    print("=" * 72)

    x_train, y_train, x_test, y_test = load_cifar10(
        args.data_dir,
        device,
        max_train_samples=max_train,
        max_test_samples=max_test,
    )
    print(f"  train={len(x_train):,}, test={len(x_test):,}, device={device}")

    network, output_head = build_resnet18_probitree(
        tree, device, gain_w=args.gain_w, gain_b=args.gain_b
    )
    reference_count = min(args.reference_samples, len(x_train))
    target_mean, reference_variance = initialize_uniform_reference(
        network,
        output_head,
        tree,
        x_train[:reference_count],
        args.probitree_r,
    )

    config = {
        "dataset": "cifar10",
        "arch": "resnet18_probitree",
        "optimizer": "tagi_direct_probit",
        "epochs": epochs,
        "batch_size": args.batch_size,
        "probitree_r": args.probitree_r,
        "probitree_tree": tree.to_dict(),
        "reference_variances": reference_variance.cpu().tolist(),
        "reference_target_means": target_mean.cpu().tolist(),
        "gain_w": args.gain_w,
        "gain_b": args.gain_b,
        "augment": args.augment,
        "seed": args.seed,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
    }
    run = RunDir("cifar10", "resnet18_probitree", "tagi", base=args.output_dir)
    run.save_config(config)
    print(f"  parameters={network.num_parameters():,}; results={run.path}")
    train(
        network,
        x_train,
        y_train,
        x_test,
        y_test,
        tree,
        epochs=epochs,
        r=args.probitree_r,
        augment=args.augment,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        checkpoint_interval=args.checkpoint_interval,
        run=run,
        config=config,
    )
    print(f"\n  Results saved under {run.path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--probitree-r", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gain-w", type=float, default=0.1)
    parser.add_argument("--gain-b", type=float, default=0.1)
    parser.add_argument("--reference-samples", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-samples", type=int, default=4)
    parser.add_argument("--smoke-test-samples", type=int, default=8)
    parser.set_defaults(augment=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
