"""End-to-end TAGI ResNet-18 on CIFAR-10 with the CDF-TAGI-V/Remax head.

The output layer is one interleaved ``2K`` Linear followed by
:class:`~triton_tagi.EvenProbit`, so the even stream carries the prediction
head ``Z_i`` and the odd stream the pre-activation variance head ``U_i`` whose
bounded CDF activation ``h(u) = epsilon + kappa Phi(u)`` is the learned
aleatoric logit noise. Training is one :meth:`Sequential.step_cdf_tagiv` per
minibatch, which runs the exact-quadrature channel: no Remax layer is in the
forward path, because class probabilities are not an activation for this head
but the analytic Laplace-Remax average over both heads and the shared
log-scale.

The shared positive deviation scale ``s = e^L`` is a *separate* inference
channel: it is fitted on a held-out split with the network frozen and never
touches the prediction or variance head. The frozen-feature study found it
worth 0.17 nats, so both the uncalibrated and the calibrated numbers are
reported every epoch.

Examples:
    python examples/cifar10_resnet18_cdf_remax.py --smoke --no-augment
    python examples/cifar10_resnet18_cdf_remax.py --epochs 50
    python examples/cifar10_resnet18_cdf_remax.py --epochs 20 --cdf-kappa 1.5 --cdf-epsilon 0.02
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from triton_tagi import (
    AvgPool2D,
    BatchNorm2D,
    Conv2D,
    EvenProbit,
    Flatten,
    Linear,
    ReLU,
    ResBlock,
    RunDir,
    Sequential,
    cdf_variance_moments,
    classification_metrics,
    fit_remax_log_scale,
    normalize_class_probabilities,
    remax_scale_moments,
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


def build_resnet18_cdf_remax(
    num_classes: int,
    device: torch.device,
    *,
    gain_w: float,
    gain_b: float,
    epsilon: float,
    kappa: float,
) -> tuple[Sequential, Linear, EvenProbit]:
    """Build the CIFAR ResNet-18 TAGI topology under an interleaved 2K head."""

    kwargs = {"device": device, "gain_w": gain_w, "gain_b": gain_b}
    output_head = Linear(512, 2 * num_classes, **kwargs)
    probit = EvenProbit(num_classes, epsilon=epsilon, kappa=kappa)
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
            probit,
        ],
        device=device,
    )
    return network, output_head, probit


@torch.no_grad()
def initialize_cdf_remax_prior(
    output_head: Linear,
    *,
    epsilon: float,
    kappa: float,
    aleatoric_init: float,
    v2bar_weight_var: float,
    v2bar_bias_var: float,
    zero_prediction_means: bool,
) -> float:
    """Start the variance head at ``h(nu) == aleatoric_init``.

    ``h`` is invertible on its attainable range, so the requested initial
    aleatoric variance is realised by the odd-slot bias mean
    ``nu = Phi^-1((v - epsilon) / kappa)``; the odd weight means are zeroed so
    that value holds at every input, and the TAGI-V prior variances supply the
    head's own uncertainty. This mirrors
    ``TAGILastLayerClassifier._initialize_cdf_remax_prior`` so the end-to-end
    and frozen-feature arms start from the same head prior.
    """

    if not epsilon < aleatoric_init < epsilon + kappa:
        raise ValueError(
            f"--cdf-aleatoric-init {aleatoric_init} is outside the attainable range "
            f"({epsilon}, {epsilon + kappa}) of the CDF variance activation"
        )
    quantile = (aleatoric_init - epsilon) / kappa
    nu = float(torch.special.ndtri(torch.tensor(quantile, dtype=torch.float64)))
    odd = slice(1, None, 2)
    assert output_head.mb is not None and output_head.Sb is not None
    if zero_prediction_means:
        # The class-symmetric prior: no prior committed to one random
        # classifier. On the frozen study this was worth +22 points at 100
        # classes and removed the gain axis entirely.
        even = slice(0, None, 2)
        output_head.mw[:, even].zero_()
        output_head.mb[:, even].zero_()
    output_head.mw[:, odd].zero_()
    output_head.Sw[:, odd].fill_(v2bar_weight_var)
    output_head.mb[:, odd].fill_(nu)
    output_head.Sb[:, odd].fill_(v2bar_bias_var)
    return nu


@torch.no_grad()
def forward_summaries(
    network: Sequential,
    probit: EvenProbit,
    inputs: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(mu_z, var_z, nu, r)``, the four tensors the head is read from.

    The even slots pass through ``EvenProbit`` untouched, so they are still the
    prediction head's own moments; the variance head's Gaussian survives only
    in the layer's forward cache, which is overwritten on every call and so is
    read per batch.
    """

    network.eval()
    means, variances, head_means, head_variances = [], [], [], []
    for start in range(0, len(inputs), batch_size):
        mean, variance = network.forward(inputs[start : start + batch_size])
        means.append(mean[..., 0::2].double())
        variances.append(variance[..., 0::2].double())
        assert probit.nu is not None and probit.r is not None
        head_means.append(probit.nu.clone())
        head_variances.append(probit.r.clone())
    network.train()
    return (
        torch.cat(means),
        torch.cat(variances),
        torch.cat(head_means),
        torch.cat(head_variances),
    )


@torch.no_grad()
def evaluate(
    network: Sequential,
    probit: EvenProbit,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    scale_mean: float,
    scale_variance: float,
    epsilon: float,
    kappa: float,
    laplace_order: int,
    hermite_order: int,
    batch_size: int,
    quadrature_batch_size: int,
    scale_order: int,
) -> dict[str, float]:
    """Score the analytic Laplace-Remax predictive under the current log-scale.

    The two batch sizes are not the same knob and must not be tied. The conv
    stack's im2col is the memory bound -- 2048 images through it asks for 4.5
    GB -- while the moment quadrature is launch-bound and wants as many rows at
    once as it can get. So the network is forwarded narrow, and the four small
    summary tensors it returns are integrated wide.
    """

    mu_z_all, var_z_all, nu_all, r_all = forward_summaries(network, probit, inputs, batch_size)
    probability_parts = []
    for start in range(0, len(mu_z_all), quadrature_batch_size):
        window = slice(start, start + quadrature_batch_size)
        probabilities, _, _ = remax_scale_moments(
            mu_z_all[window],
            var_z_all[window],
            nu_all[window],
            r_all[window],
            scale_mean=scale_mean,
            scale_variance=scale_variance,
            epsilon=epsilon,
            kappa=kappa,
            laplace_order=laplace_order,
            hermite_order=hermite_order,
            scale_order=scale_order,
            cross_moments=False,
        )
        probability_parts.append(normalize_class_probabilities(probabilities).float().cpu())
    return classification_metrics(torch.cat(probability_parts), labels.cpu())


def fit_log_scale(
    network: Sequential,
    probit: EvenProbit,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    kappa: float,
    laplace_order: int,
    hermite_order: int,
    scale_order: int,
    grid_size: int,
    refinements: int,
    batch_size: int,
    quadrature_batch_size: int,
):
    """Fit the shared log-scale on a split disjoint from training."""

    mu_z, var_z, nu, r = forward_summaries(network, probit, inputs, batch_size)
    return fit_remax_log_scale(
        mu_z,
        var_z,
        nu,
        r,
        labels.long(),
        epsilon=epsilon,
        kappa=kappa,
        method="grid",
        grid_size=grid_size,
        refinements=refinements,
        laplace_order=laplace_order,
        hermite_order=hermite_order,
        scale_order=scale_order,
        chunk_size=quadrature_batch_size,
    )


@torch.no_grad()
def head_occupancy(
    network: Sequential,
    probit: EvenProbit,
    inputs: torch.Tensor,
    *,
    epsilon: float,
    kappa: float,
    batch_size: int,
) -> dict[str, float]:
    """Report whether the bounded variance head has any room left to move."""

    _, var_z, nu, r = forward_summaries(network, probit, inputs, batch_size)
    h_mean, _, _ = cdf_variance_moments(nu, r, epsilon=epsilon, kappa=kappa)
    return {
        "aleatoric_median": float(h_mean.median()),
        # Where the head sits inside its own range, 0 at the floor, 1 at the cap.
        "range_position_median": float((h_mean.median() - epsilon) / kappa),
        "range_used": float((h_mean.max() - h_mean.min()) / kappa),
        "epistemic_median": float(var_z.median()),
    }


def train(
    network: Sequential,
    probit: EvenProbit,
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    epochs: int,
    epsilon: float,
    kappa: float,
    train_hermite_order: int,
    laplace_order: int,
    hermite_order: int,
    scale_order: int,
    grid_size: int,
    refinements: int,
    calibration_interval: int,
    augment: bool,
    batch_size: int,
    evaluation_batch_size: int,
    quadrature_batch_size: int,
    checkpoint_interval: int,
    run: RunDir,
    config: dict,
) -> None:
    """Train on minibatched observations, refitting the log-scale periodically.

    The scale fit costs about as much as two thirds of a training epoch, so it
    runs every ``calibration_interval`` epochs. On the epochs in between the
    standing posterior is carried forward and the calibrated columns are marked
    stale with ``*``: the scale is then a fit against an older checkpoint, not
    a wrong number, but it is not the fit this epoch would produce.
    """

    x_train, y_train = splits["train"]
    x_calibrate, y_calibrate = splits["calibrate"]
    x_test, y_test = splits["test"]
    quadrature = {
        "laplace_order": laplace_order,
        "hermite_order": hermite_order,
        "scale_order": scale_order,
    }

    header = (
        f"  {'Epoch':>5}  {'Test Acc':>9}  {'NLL raw':>8}  {'NLL cal':>9}  "
        f"{'ECE cal':>8}  {'scale s':>8}  {'h med':>7}  {'band':>6}  {'Time':>8}"
    )
    print(f"\n{header}", flush=True)
    print("  " + "─" * (len(header) - 2), flush=True)
    posterior = None
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        permutation = torch.randperm(len(x_train), device=x_train.device)
        network.train()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            inputs = x_train[indices]
            if augment:
                inputs = gpu_augment(inputs)
            network.step_cdf_tagiv(
                inputs,
                y_train[indices],
                epsilon=epsilon,
                kappa=kappa,
                hermite_order=train_hermite_order,
            )

        if x_train.device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        # Overwritten in place on a terminal; kept as its own line in a log file,
        # where "still training" and "hung" otherwise look identical.
        print(
            f"  {epoch:5d}  [trained in {train_seconds:6.1f}s, scoring]",
            end="\r" if sys.stdout.isatty() else "\n",
            flush=True,
        )

        refit = posterior is None or epoch % calibration_interval == 0 or epoch == epochs
        if refit:
            posterior = fit_log_scale(
                network,
                probit,
                x_calibrate,
                y_calibrate,
                epsilon=epsilon,
                kappa=kappa,
                grid_size=grid_size,
                refinements=refinements,
                batch_size=evaluation_batch_size,
                quadrature_batch_size=quadrature_batch_size,
                **quadrature,
            )
        shared = {
            "epsilon": epsilon,
            "kappa": kappa,
            "batch_size": evaluation_batch_size,
            "quadrature_batch_size": quadrature_batch_size,
            **quadrature,
        }
        raw = evaluate(
            network, probit, x_test, y_test, scale_mean=0.0, scale_variance=0.0, **shared
        )
        calibrated = evaluate(
            network,
            probit,
            x_test,
            y_test,
            scale_mean=float(posterior.mean),
            scale_variance=float(posterior.variance),
            **shared,
        )
        occupancy = head_occupancy(
            network,
            probit,
            x_test[: min(2048, len(x_test))],
            epsilon=epsilon,
            kappa=kappa,
            batch_size=evaluation_batch_size,
        )
        wall_seconds = time.perf_counter() - started
        stale = "" if refit else "*"
        print(
            f"  {epoch:5d}  {calibrated['accuracy'] * 100:8.2f}%  {raw['nll']:8.4f}"
            f"  {calibrated['nll']:8.4f}{stale:1s}  {calibrated['ece']:8.4f}"
            f"  {posterior.scale_median:8.4f}  {occupancy['aleatoric_median']:7.4f}"
            f"  {occupancy['range_position_median']:6.3f}  {wall_seconds:7.1f}s",
            flush=True,
        )
        run.append_metrics(
            epoch,
            test_acc=calibrated["accuracy"],
            test_nll_uncalibrated=raw["nll"],
            test_ece_uncalibrated=raw["ece"],
            test_nll=calibrated["nll"],
            test_ece=calibrated["ece"],
            test_brier=calibrated["brier"],
            test_top5=calibrated["top5_accuracy"],
            test_mean_confidence=calibrated["mean_confidence"],
            log_scale_mean=float(posterior.mean),
            log_scale_variance=float(posterior.variance),
            scale_median=posterior.scale_median,
            scale_refit=float(refit),
            train_s=train_seconds,
            wall_s=wall_seconds,
            **occupancy,
        )
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            run.save_checkpoint(network, epoch, config)


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the Triton ResNet kernels")
    if args.epochs < 1 or args.evaluation_batch_size < 1 or args.batch_size < 1:
        raise ValueError("epochs, batch size, and evaluation batch size must be positive")
    if args.calibration_samples < 1:
        raise ValueError("--calibration-samples must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    epochs = 1 if args.smoke else args.epochs
    max_train = args.smoke_train_samples if args.smoke else args.max_train_samples
    max_test = args.smoke_test_samples if args.smoke else args.max_test_samples
    num_classes = 10

    print("=" * 78)
    print("  CIFAR-10 — end-to-end TAGI ResNet-18 — CDF-TAGI-V / Remax")
    print(
        f"  interleaved 2K={2 * num_classes} head, aleatoric noise in "
        f"({args.cdf_epsilon:g}, {args.cdf_epsilon + args.cdf_kappa:g})"
    )
    print(f"  Training updates are minibatched ({args.batch_size} labeled images per TAGI update).")
    print("=" * 78)

    x_train, y_train, x_test, y_test = load_cifar10(
        args.data_dir,
        device,
        max_train_samples=max_train,
        max_test_samples=max_test,
    )
    calibration_count = min(args.calibration_samples, len(x_train) - 1)
    if args.smoke:
        calibration_count = min(calibration_count, max(1, len(x_train) // 2))
    # The scale is a held-out fit: the split must be disjoint from training, or
    # the calibration channel reads its own training data back.
    x_calibrate, y_calibrate = x_train[:calibration_count], y_train[:calibration_count]
    x_train, y_train = x_train[calibration_count:], y_train[calibration_count:]
    print(
        f"  train={len(x_train):,}, calibrate={len(x_calibrate):,}, "
        f"test={len(x_test):,}, device={device}"
    )

    network, output_head, probit = build_resnet18_cdf_remax(
        num_classes,
        device,
        gain_w=args.gain_w,
        gain_b=args.gain_b,
        epsilon=args.cdf_epsilon,
        kappa=args.cdf_kappa,
    )
    nu = initialize_cdf_remax_prior(
        output_head,
        epsilon=args.cdf_epsilon,
        kappa=args.cdf_kappa,
        aleatoric_init=args.cdf_aleatoric_init,
        v2bar_weight_var=args.v2bar_weight_var,
        v2bar_bias_var=args.v2bar_bias_var,
        zero_prediction_means=args.mean_init == "zero",
    )

    config = {
        "dataset": "cifar10",
        "arch": "resnet18_cdf_remax",
        "optimizer": "tagi_cdf_tagiv",
        "epochs": epochs,
        "batch_size": args.batch_size,
        "num_classes": num_classes,
        "cdf_epsilon": args.cdf_epsilon,
        "cdf_kappa": args.cdf_kappa,
        "cdf_aleatoric_init": args.cdf_aleatoric_init,
        "cdf_head_bias_mean": nu,
        "v2bar_weight_var": args.v2bar_weight_var,
        "v2bar_bias_var": args.v2bar_bias_var,
        "laplace_order": args.laplace_order,
        "hermite_order": args.hermite_order,
        "scale_order": args.scale_order,
        "train_hermite_order": args.train_hermite_order,
        "calibration_samples": calibration_count,
        "calibration_interval": args.calibration_interval,
        "grid_size": args.grid_size,
        "mean_init": args.mean_init,
        "gain_w": args.gain_w,
        "gain_b": args.gain_b,
        "augment": args.augment,
        "seed": args.seed,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
    }
    run = RunDir("cifar10", "resnet18_cdf_remax", "tagi", base=args.output_dir)
    run.save_config(config)
    print(f"  parameters={network.num_parameters():,}; results={run.path}")
    print(f"  head prior: h(nu)={args.cdf_aleatoric_init:g} at nu={nu:.4f}, mean_init={args.mean_init}")
    train(
        network,
        probit,
        {
            "train": (x_train, y_train),
            "calibrate": (x_calibrate, y_calibrate),
            "test": (x_test, y_test),
        },
        epochs=epochs,
        epsilon=args.cdf_epsilon,
        kappa=args.cdf_kappa,
        train_hermite_order=args.train_hermite_order,
        laplace_order=args.laplace_order,
        hermite_order=args.hermite_order,
        scale_order=args.scale_order,
        grid_size=args.grid_size,
        refinements=args.refinements,
        calibration_interval=args.calibration_interval,
        augment=args.augment,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        quadrature_batch_size=args.quadrature_batch_size,
        checkpoint_interval=args.checkpoint_interval,
        run=run,
        config=config,
    )
    print(f"\n  Results saved under {run.path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gain-w", type=float, default=0.1)
    parser.add_argument("--gain-b", type=float, default=0.1)
    parser.add_argument("--mean-init", choices=("random", "zero"), default="zero")
    # The frozen-feature CIFAR-10 selection: a tight noise band beats the
    # library default (0.02, 1.5), which is wide enough for 100-class logits.
    parser.add_argument("--cdf-epsilon", type=float, default=0.01)
    parser.add_argument("--cdf-kappa", type=float, default=0.05)
    parser.add_argument("--cdf-aleatoric-init", type=float, default=0.02)
    parser.add_argument("--v2bar-weight-var", type=float, default=1e-6)
    parser.add_argument("--v2bar-bias-var", type=float, default=1.0)
    parser.add_argument("--train-hermite-order", type=int, default=64)
    parser.add_argument("--laplace-order", type=int, default=80)
    parser.add_argument("--hermite-order", type=int, default=16)
    parser.add_argument("--scale-order", type=int, default=20)
    # 65 is run_cdf_remax_cifar.py's value. The library default 513 is
    # 8x the grid points for a scalar posterior and costs ~5 min per fit.
    parser.add_argument("--grid-size", type=int, default=65)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--calibration-samples", type=int, default=5000)
    parser.add_argument("--calibration-interval", type=int, default=5)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--quadrature-batch-size", type=int, default=2048)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-train-samples", type=int, default=64)
    parser.add_argument("--smoke-test-samples", type=int, default=32)
    parser.set_defaults(augment=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
