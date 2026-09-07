"""Interoperability helpers for frozen deterministic PyTorch models.

triton-tagi does not translate an arbitrary ``torch.nn.Module`` into TAGI
layers. Callers expose penultimate features and logits, then train a small TAGI
classifier on either representation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .layers.linear import Linear

RepresentationFn = Callable[[nn.Module, Tensor], tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class TorchCheckpointInfo:
    """Metadata returned after loading a deterministic PyTorch checkpoint."""

    path: Path
    state_key: str | None
    metadata: dict[str, Any]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _is_state_dict(value: object) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and isinstance(tensor, Tensor)
        for key, tensor in value.items()
    )


def _select_state_dict(
    checkpoint: object, state_key: str | None
) -> tuple[Mapping[str, Tensor], str | None, dict[str, Any]]:
    if state_key is not None:
        if not isinstance(checkpoint, Mapping) or state_key not in checkpoint:
            raise KeyError(f"Checkpoint has no state dictionary at key {state_key!r}")
        state = checkpoint[state_key]
        if not _is_state_dict(state):
            raise TypeError(f"Checkpoint entry {state_key!r} is not a PyTorch state dictionary")
        return state, state_key, {str(k): v for k, v in checkpoint.items() if k != state_key}

    if _is_state_dict(checkpoint):
        return checkpoint, None, {}
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict"):
            state = checkpoint.get(key)
            if _is_state_dict(state):
                return state, key, {str(k): v for k, v in checkpoint.items() if k != key}
    raise ValueError(
        "Could not find a state dictionary. Pass a raw state_dict, use a "
        "'state_dict'/'model_state_dict' container, or specify state_key."
    )


def _strip_module_prefix(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return dict(state)


def load_torch_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    state_key: str | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> TorchCheckpointInfo:
    """Load a raw or conventionally wrapped PyTorch state dictionary."""

    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state, selected_key, metadata = _select_state_dict(checkpoint, state_key)
    incompatible = model.load_state_dict(_strip_module_prefix(state), strict=strict)
    model.to(device)
    return TorchCheckpointInfo(
        path=checkpoint_path,
        state_key=selected_key,
        metadata=metadata,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )


class FrozenTorchBackbone:
    """Read-only adapter exposing features, logits, and softmax probabilities."""

    def __init__(self, model: nn.Module, representation_fn: RepresentationFn) -> None:
        self.model = model
        self.representation_fn = representation_fn
        self.model.eval()
        self.model.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @torch.no_grad()
    def representations(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(features, logits)`` without constructing an autograd graph."""

        self.model.eval()
        features, logits = self.representation_fn(self.model, inputs.to(self.device))
        if features.shape[0] != inputs.shape[0] or logits.shape[0] != inputs.shape[0]:
            raise ValueError("Feature and logit batches must match the input batch size")
        if features.dim() != 2 or logits.dim() != 2:
            raise ValueError("Features and logits must both have shape (batch, dimensions)")
        return features.detach(), logits.detach()

    @torch.no_grad()
    def predict(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(features, logits, softmax_probabilities)``."""

        features, logits = self.representations(inputs)
        return features, logits, torch.softmax(logits, dim=-1)


def copy_torch_linear_(
    target: Linear,
    source: nn.Linear,
    *,
    weight_variance: float | None = None,
    bias_variance: float | None = None,
) -> Linear:
    """Warm-start TAGI linear means from ``torch.nn.Linear``."""

    expected = (source.in_features, source.out_features)
    if tuple(target.mw.shape) != expected:
        raise ValueError(f"Target shape {tuple(target.mw.shape)} does not match source {expected}")
    if source.bias is None and target.has_bias:
        raise ValueError("Cannot initialize a biased TAGI layer from a bias-free PyTorch layer")
    target.mw.copy_(source.weight.detach().T.to(target.device, target.mw.dtype))
    if target.has_bias and source.bias is not None:
        target.mb.copy_(source.bias.detach().reshape(1, -1).to(target.device, target.mb.dtype))
    if weight_variance is not None:
        if weight_variance < 0:
            raise ValueError("weight_variance must be nonnegative")
        target.Sw.fill_(weight_variance)
    if bias_variance is not None and target.Sb is not None:
        if bias_variance < 0:
            raise ValueError("bias_variance must be nonnegative")
        target.Sb.fill_(bias_variance)
    return target


def initialize_identity_linear_(
    target: Linear,
    *,
    weight_variance: float | None = None,
    bias_variance: float | None = None,
) -> Linear:
    """Initialize a square TAGI linear layer to the identity in mean."""

    if target.in_features != target.out_features:
        raise ValueError("Identity initialization requires equal input and output dimensions")
    target.mw.zero_()
    target.mw.diagonal().fill_(1.0)
    if target.mb is not None:
        target.mb.zero_()
    if weight_variance is not None:
        if weight_variance < 0:
            raise ValueError("weight_variance must be nonnegative")
        target.Sw.fill_(weight_variance)
    if bias_variance is not None and target.Sb is not None:
        if bias_variance < 0:
            raise ValueError("bias_variance must be nonnegative")
        target.Sb.fill_(bias_variance)
    return target
