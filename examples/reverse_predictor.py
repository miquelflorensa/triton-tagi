"""
Reverse Predictor — TAGI self-attention example — triton-tagi.

Replicates the ``examples/reverse_predictor.py`` example on cuTAGI's
``feat/attn-debug`` branch: given a random integer sequence, the model must
predict the same sequence in reverse. The architecture is a single-block
transformer with sinusoidal positional encoding, ``MultiheadAttentionV2``,
``RMSNorm``, and a linear head onto a hierarchical-softmax tree.

Usage:
    python examples/reverse_predictor.py
    python examples/reverse_predictor.py --num_epochs 20 --batch_size 32
    python examples/reverse_predictor.py --no_attn   (ablation: no attention block)
    python examples/reverse_predictor.py --help
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from triton_tagi import (
    Embedding,
    HierarchicalSoftmax,
    Linear,
    MultiheadAttentionV2,
    PositionalEncoding,
    RMSNorm,
    Sequential,
    class_to_obs,
    get_predicted_labels,
)


class ReverseDataset:
    """Random integer sequences paired with their reversed counterparts."""

    def __init__(self, vocab_size: int, seq_len: int, rng: np.random.Generator):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.rng = rng

    def next_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x, labels)``:
            x      : float32, shape (B, S, 1) — per-position class indices
            labels : int64,   shape (B*S,)    — reversed indices, flattened
        """
        x = self.rng.integers(self.vocab_size, size=(batch_size, self.seq_len))
        labels = np.flip(x, axis=1).reshape(-1).astype(np.int64)
        x = x.reshape(batch_size, self.seq_len, 1).astype(np.float32)
        return x, labels


def build_network(
    vocab_size: int,
    embed_dim: int,
    num_heads: int,
    seq_len: int,
    hrc: HierarchicalSoftmax,
    no_attn: bool,
    device: torch.device,
) -> Sequential:
    """Assemble the reverse-predictor network."""
    if no_attn:
        layers = [
            Embedding(vocab_size, embed_dim, input_size=seq_len, scale=1.0, device=device),
            Linear(embed_dim, hrc.len, device=device),
        ]
    else:
        layers = [
            Embedding(vocab_size, embed_dim, input_size=seq_len, scale=0.25, device=device),
            PositionalEncoding(embed_dim, device=device),
            MultiheadAttentionV2(
                embed_dim=embed_dim,
                num_heads=num_heads,
                seq_len=seq_len,
                bias=False,
                gain_weight=0.25,
                gain_bias=0.5,
                init_method="He",
                pos_emb="",
                use_causal_mask=False,
                device=device,
            ),
            RMSNorm([embed_dim], device=device),
            Linear(embed_dim, hrc.len, device=device),
        ]
    return Sequential(layers, device=device)


def train(
    net: Sequential,
    task: ReverseDataset,
    hrc: HierarchicalSoftmax,
    num_epochs: int,
    batch_size: int,
    steps_per_epoch: int,
    sigma_v: float,
    sigma_v_min: float,
    decay_factor: float,
    device: torch.device,
) -> None:
    """Run the training loop. Prints running error rate per epoch."""
    print(f"\n  {'Epoch':>5}  {'Error':>7}  {'sigma_v':>8}  {'Time':>7}")
    print("  " + "─" * 38)

    current_sigma_v = sigma_v
    net.train()
    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()
        errors: list[float] = []
        for _ in range(steps_per_epoch):
            x_np, labels_np = task.next_batch(batch_size)
            x = torch.from_numpy(x_np).to(device)
            labels = torch.from_numpy(labels_np).to(device)

            mu, var = net.step_hrc(x, labels, hrc, current_sigma_v)

            mu_flat = mu.reshape(-1, hrc.len)
            var_flat = var.reshape(-1, hrc.len)
            pred = get_predicted_labels(mu_flat, var_flat, hrc)
            errors.append((pred != labels).float().mean().item())

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        avg_err = sum(errors) / len(errors)
        print(f"  {epoch:5d}  {avg_err*100:6.2f}%  {current_sigma_v:8.3f}  {wall:6.2f}s")
        current_sigma_v = max(sigma_v_min, current_sigma_v * decay_factor)

    print("  " + "─" * 38)


def evaluate(
    net: Sequential,
    task: ReverseDataset,
    hrc: HierarchicalSoftmax,
    test_batch_size: int,
    device: torch.device,
    show: int = 5,
) -> float:
    """Run one evaluation batch; print a few examples and return accuracy."""
    net.eval()
    x_np, labels_np = task.next_batch(test_batch_size)
    x = torch.from_numpy(x_np).to(device)

    with torch.no_grad():
        mu, var = net.forward(x)

    mu_flat = mu.reshape(-1, hrc.len)
    var_flat = var.reshape(-1, hrc.len)
    pred = get_predicted_labels(mu_flat, var_flat, hrc).cpu().numpy()

    S = task.seq_len
    x_disp = x_np.squeeze(-1).astype(int)
    y_disp = labels_np.reshape(test_batch_size, S)
    pred_disp = pred.reshape(test_batch_size, S)

    print(f"\n  Test results (showing {min(show, test_batch_size)} of {test_batch_size}):")
    for i in range(min(show, test_batch_size)):
        print(f"    Input   : {x_disp[i].tolist()}")
        print(f"    Target  : {y_disp[i].tolist()}")
        print(f"    Predict : {pred_disp[i].tolist()}")

    acc = float((pred_disp == y_disp).mean())
    print(f"\n  Test accuracy: {acc*100:.2f}%")
    return acc


def main(
    num_epochs: int = 50,
    batch_size: int = 64,
    seq_len: int = 8,
    vocab_size: int = 8,
    embed_dim: int = 32,
    num_heads: int = 1,
    sigma_v: float = 4.5,
    sigma_v_min: float = 0.3,
    decay_factor: float = 1.0,
    steps_per_epoch: int = 100,
    no_attn: bool = False,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    print("=" * 56)
    print("  Reverse Predictor — TAGI self-attention — triton-tagi")
    print(f"  vocab={vocab_size}  seq_len={seq_len}  embed_dim={embed_dim}"
          f"  heads={num_heads}  no_attn={no_attn}")
    print("=" * 56)

    hrc = class_to_obs(vocab_size)
    task = ReverseDataset(vocab_size, seq_len, rng)

    net = build_network(vocab_size, embed_dim, num_heads, seq_len, hrc, no_attn, dev)
    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")

    train(
        net, task, hrc,
        num_epochs=num_epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        sigma_v=sigma_v,
        sigma_v_min=sigma_v_min,
        decay_factor=decay_factor,
        device=dev,
    )

    acc = evaluate(net, task, hrc, test_batch_size=256, device=dev, show=5)

    if not no_attn:
        scores = net.get_attention_scores()
        for idx, (mu_s, var_s) in scores.items():
            print(f"\n  Attention scores at layer {idx}: "
                  f"mu shape {tuple(mu_s.shape)}, sample [0, head 0]:")
            print(mu_s[0, 0].cpu().numpy())

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reverse Predictor with TAGI self-attention")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--sigma_v", type=float, default=4.5)
    parser.add_argument("--sigma_v_min", type=float, default=0.3)
    parser.add_argument("--decay_factor", type=float, default=1.0)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--no_attn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    main(**vars(args))
