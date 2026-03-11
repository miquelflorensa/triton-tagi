# TAGI-Triton

**GPU-Accelerated Tractable Approximate Gaussian Inference for Bayesian Neural Networks**

TAGI-Triton is a high-performance library for training and running Bayesian neural networks (BNNs) using [Tractable Approximate Gaussian Inference (TAGI)](https://www.jmlr.org/papers/v22/20-1009.html). All heavy computations — variance propagation, Bayesian activations, parameter updates — are implemented as fused [Triton](https://triton-lang.org/) kernels for maximum GPU throughput.

> **Key idea:** Instead of backpropagation with point-estimate weights, TAGI propagates Gaussian distributions (mean + variance) through the network and performs closed-form Bayesian updates — no sampling, no variational bounds, no autograd.

---

## Features

| Feature | Description |
|---|---|
| **Fused Triton Kernels** | Variance-forward, backward-delta, ReLU moments, parameter updates — all in custom Triton kernels |
| **Modular Layer API** | `Linear`, `Conv2D`, `BatchNorm2D`, `ResBlock`, `AvgPool2D`, `Flatten` |
| **Bayesian Activations** | `ReLU`, `LeakyReLU`, `EvenSoftplus` — moment-propagation through nonlinearities |
| **Classification Heads** | `Remax` (softmax alternative) and `Bernoulli` (max-indicator via Gauss-Hermite quadrature) |
| **Shared Variance Layers** | `SharedVarLinear`, `SharedVarConv2D`, `SharedVarBatchNorm2D`, `SharedVarResBlock` — scalar variance per layer for regularization |
| **Optimizers** | Vanilla TAGI, `AdamTAGI`, `NadamTAGI` — gradient-free adaptive optimization |
| **Auto-Tune** | Automatic gain and σ_v selection via forward-only variance analysis + short training probes |
| **Monitoring** | `TAGIMonitor` — activation/parameter/signal-flow diagnostics with built-in plotting |
| **ResNet Support** | Full residual blocks with projection shortcuts (cuTAGI-compatible) |

---

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU
- PyTorch ≥ 2.0 (with CUDA)
- [Triton](https://triton-lang.org/) ≥ 2.0

### Setup

```bash
git clone https://github.com/miquelflorensa/triton-tagi.git
cd triton-tagi

pip install torch torchvision  # with CUDA support
pip install triton
pip install numpy matplotlib
```

---

## Quick Start

### MLP Classification (MNIST)

```python
import torch
from src import Sequential
from src.layers import Linear, ReLU, Remax

device = torch.device("cuda")

net = Sequential([
    Linear(784, 256, device=device),
    ReLU(),
    Linear(256, 128, device=device),
    ReLU(),
    Linear(128, 10, device=device),
    Remax(),
], device=device)

# Single training step
sigma_v = 0.01
y_pred_mu, y_pred_var = net.step(x_batch, y_batch, sigma_v)
```

### CNN Classification (CIFAR-10)

```python
from src import Sequential
from src.layers import Conv2D, ReLU, AvgPool2D, BatchNorm2D, Flatten, Linear, Remax

net = Sequential([
    Conv2D(3, 32, 5, stride=1, padding=2, device=device),
    ReLU(),
    BatchNorm2D(32, device=device),
    AvgPool2D(2),

    Conv2D(32, 64, 5, stride=1, padding=2, device=device),
    ReLU(),
    BatchNorm2D(64, device=device),
    AvgPool2D(2),

    Flatten(),
    Linear(64 * 8 * 8, 256, device=device),
    ReLU(),
    Linear(256, 10, device=device),
    Remax(),
], device=device)
```

### ResNet-18

```python
from src import Sequential
from src.layers import Conv2D, ReLU, BatchNorm2D, AvgPool2D, Flatten, Linear, Remax, ResBlock

net = Sequential([
    Conv2D(3, 64, 3, stride=1, padding=1, device=device),
    ReLU(),
    BatchNorm2D(64, device=device),

    ResBlock(64, 64, stride=1, device=device),
    ResBlock(64, 64, stride=1, device=device),
    ResBlock(64, 128, stride=2, device=device),   # projection shortcut
    ResBlock(128, 128, stride=1, device=device),
    ResBlock(128, 256, stride=2, device=device),
    ResBlock(256, 256, stride=1, device=device),
    ResBlock(256, 512, stride=2, device=device),
    ResBlock(512, 512, stride=1, device=device),

    AvgPool2D(4),
    Flatten(),
    Linear(512, 10, device=device),
    Remax(),
], device=device)
```

### Using an Optimizer

```python
from src import Sequential, AdamTAGI

net = Sequential([...], device=device)
opt = AdamTAGI(net)

for epoch in range(n_epochs):
    for xb, yb in batches:
        y_pred_mu, y_pred_var = opt.step(xb, yb, sigma_v)
```

---

## Architecture

```
src/
├── __init__.py              # Public API
├── network.py               # Sequential container (forward / step / train / eval)
├── optimizer.py             # AdamTAGI optimizer
├── nadam_optimizer.py       # NadamTAGI optimizer
├── auto_tune.py             # Automatic gain & σ_v selection
├── monitor.py               # TAGIMonitor, sweep_gains, sweep_sigma_v
├── param_init.py            # He / Xavier / Gaussian initialization
├── init.py                  # reinit_net, init_residual_aware
├── kernels/
│   └── common.py            # Low-level Triton kernel wrappers
├── layers/
│   ├── linear.py            # Bayesian fully-connected layer
│   ├── conv2d.py            # Bayesian Conv2D (im2col + fused matmul)
│   ├── batchnorm2d.py       # Bayesian Batch Normalization
│   ├── resblock.py          # Residual block (cuTAGI-compatible)
│   ├── relu.py              # Bayesian ReLU (moment propagation)
│   ├── leaky_relu.py        # Bayesian Leaky ReLU
│   ├── even_softplus.py     # Even Softplus activation
│   ├── remax.py             # Remax classification head
│   ├── bernoulli.py         # Bernoulli (max-indicator) classification
│   ├── avgpool2d.py         # Average Pooling 2D
│   ├── flatten.py           # Flatten layer
│   ├── shared_var_*.py      # Shared-variance variants of layers
│   └── __init__.py
└── update/
    ├── observation.py       # Output innovation (δ_μ, δ_S)
    ├── parameters.py        # Capped parameter update (cuTAGI-style)
    └── shared_var_parameters.py
```

---

## How TAGI Works

In a standard neural network, weights are point estimates updated via backpropagation. In TAGI, every weight and activation is a **Gaussian random variable** characterized by its mean (μ) and variance (σ²).

### Forward Pass (Moment Propagation)

For a linear layer $z = aW + b$:

$$\mu_z = \mu_a \, \mu_W + \mu_b$$

$$\sigma^2_z = \mu_a^2 \, \sigma^2_W + \sigma^2_a \, \mu_W^2 + \sigma^2_a \, \sigma^2_W + \sigma^2_b$$

### Backward Pass (Bayesian Update)

TAGI computes **output innovation** at the output layer:

$$\delta_\mu = \frac{y - \mu_z}{\sigma^2_z + \sigma^2_v}, \qquad \delta_\sigma = \frac{-1}{\sigma^2_z + \sigma^2_v}$$

These deltas propagate backward through each layer, updating both means and variances in closed form — no gradient computation required.

### Parameter Update (Capped)

Parameters are updated using a precision-space rule with adaptive capping to prevent overshooting:

$$\mu_W^{\text{new}} = \mu_W + \sigma^2_W \cdot \Delta_\mu$$

$$\sigma^{2,\text{new}}_W = \max\!\left(\sigma^2_W + (\sigma^2_W)^2 \cdot \Delta_\sigma,\; \epsilon\right)$$

---

## Training Scripts

| Script | Description |
|---|---|
| `train_mnist.py` | MNIST FNN: PyTorch vs Triton comparison |
| `train_mnist_cnn.py` | MNIST LeNet-style CNN |
| `train_cifar10.py` | CIFAR-10 3-layer CNN |
| `train_cifar10_3block.py` | CIFAR-10 extended CNN |
| `train_mnist_shared_var.py` | MNIST with shared-variance layers |
| `run_resnet18.py` | ResNet-18 on CIFAR-10 |
| `run_resnet18_cifar100.py` | ResNet-18 on CIFAR-100 |
| `run_cifar10_adam.py` | CIFAR-10 with AdamTAGI optimizer |
| `run_cifar10_nadam.py` | CIFAR-10 with NadamTAGI optimizer |
| `benchmark.py` | Wall-clock benchmarks (PyTorch vs Triton) |
| `test_auto_tune.py` | Auto-tune smoke test |

---

## Monitoring & Diagnostics

### TAGIMonitor

Track activation statistics, parameter health, and signal flow throughout training:

```python
from src.monitor import TAGIMonitor

monitor = TAGIMonitor(net, log_dir="run_logs")

for epoch in range(n_epochs):
    for xb, yb in batches:
        net.step(xb, yb, sigma_v)
    monitor.record(epoch, x_probe=x_train[:256])
    monitor.print_report()

monitor.plot("monitor.png")
```

### Gain Sweep

Find the best initialization gain by analyzing per-layer variance flow:

```python
from src.monitor import sweep_gains

sweep_gains(
    builder_fn=lambda gw: build_net(gain_w=gw),
    x_probe=x_train[:256],
    gains=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
    filename="gain_sweep.png",
)
```

### Auto-Tune

Automatically find the best gain and observation noise:

```python
from src.auto_tune import auto_tune

result = auto_tune(
    builder_fn=lambda gw, gb: build_net(gain_w=gw, gain_b=gb),
    x_probe=x_train[:512],
    y_probe=y_train_oh[:512],
    x_eval=x_test[:1000],
    y_eval=y_test_labels[:1000],
)
print(f"Best gain_w={result.gain_w}, gain_b={result.gain_b}, sigma_v={result.sigma_v}")
```

---

## Classification Heads

### Remax

A softmax alternative for Bayesian networks that operates in moment space. Computes output probabilities via ReLU normalization of the logit distribution — no sampling required.

```python
from src.layers import Remax
# Use as the final layer in a classification network
net = Sequential([..., Linear(256, 10, device=device), Remax()], device=device)
```

### Bernoulli

Max-indicator probabilities computed via Gauss-Hermite quadrature: $P_i = P(Z_i = \max_j Z_j)$.

```python
from src.layers import Bernoulli
net = Sequential([..., Linear(256, 10, device=device), Bernoulli(n_gh=32)], device=device)
```

---

## Shared Variance

Instead of maintaining one variance per parameter, shared-variance layers use a **single scalar variance** per layer for weights and biases. This acts as a natural regularizer and dramatically reduces the number of variance parameters:

```python
from src.layers import SharedVarLinear, SharedVarConv2D, SharedVarBatchNorm2D

net = Sequential([
    SharedVarConv2D(3, 32, 5, stride=1, padding=2, device=device),
    ReLU(),
    SharedVarBatchNorm2D(32, device=device),
    AvgPool2D(2),
    Flatten(),
    SharedVarLinear(32 * 16 * 16, 10, device=device),
    Remax(),
], device=device)
```

---

## Relation to cuTAGI

This library is a Triton-based reimplementation of [cuTAGI](https://github.com/lhnguyen102/cuTAGI), the reference C++/CUDA implementation of TAGI. Key design choices from cuTAGI are preserved:

- **Capped parameter updates** with batch-size-dependent cap factors
- **cuTAGI-style backward pass**: compute deltas first, then apply capped updates
- **ResBlock architecture**: identical to cuTAGI's `ResNetBlock` with projection shortcuts
- **BatchNorm**: running statistics with EMA, matching cuTAGI's normalization

The Triton implementation provides the same mathematical correctness with a more accessible Python-native codebase and automatic GPU kernel optimization.

---

## References

- Goulet, J.-A., Nguyen, L. H., & Amiri, S. (2021). *Tractable Approximate Gaussian Inference for Bayesian Neural Networks*. JMLR, 22(228), 1–23. [[paper]](https://www.jmlr.org/papers/v22/20-1009.html)
- cuTAGI: C++/CUDA implementation — [github.com/lhnguyen102/cuTAGI](https://github.com/lhnguyen102/cuTAGI)
- Triton: OpenAI's GPU programming language — [triton-lang.org](https://triton-lang.org/)

---

## License

See [LICENSE](cuTAGI/LICENSE) for details.
