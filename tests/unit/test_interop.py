"""Tests for deterministic PyTorch model interoperability."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from triton_tagi.interop import (
    FrozenTorchBackbone,
    copy_torch_linear_,
    initialize_identity_linear_,
    load_torch_checkpoint,
)
from triton_tagi.layers import Linear


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Linear(4, 3)
        self.classifier = nn.Linear(3, 2)

    def forward(self, x):
        return self.classifier(torch.relu(self.features(x)))


def _representations(model: TinyModel, x):
    features = torch.relu(model.features(x))
    return features, model.classifier(features)


@pytest.mark.parametrize("container_key", [None, "state_dict", "model_state_dict"])
def test_load_torch_checkpoint_conventions(tmp_path, container_key):
    torch.manual_seed(1)
    source = TinyModel()
    state = source.state_dict()
    payload = state if container_key is None else {container_key: state, "epoch": 7}
    path = tmp_path / "model.pt"
    torch.save(payload, path)

    target = TinyModel()
    info = load_torch_checkpoint(target, path)
    for actual, expected in zip(target.parameters(), source.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert info.state_key == container_key
    assert info.metadata == ({} if container_key is None else {"epoch": 7})


def test_load_strips_uniform_dataparallel_prefix(tmp_path):
    source = TinyModel()
    prefixed = {f"module.{key}": value for key, value in source.state_dict().items()}
    path = tmp_path / "model.pt"
    torch.save({"weights": prefixed}, path)
    target = TinyModel()
    load_torch_checkpoint(target, path, state_key="weights")
    torch.testing.assert_close(target.classifier.weight, source.classifier.weight)


def test_load_reports_non_strict_mismatch(tmp_path):
    model = TinyModel()
    state = model.state_dict()
    del state["classifier.bias"]
    path = tmp_path / "model.pt"
    torch.save(state, path)
    info = load_torch_checkpoint(TinyModel(), path, strict=False)
    assert info.missing_keys == ("classifier.bias",)


def test_frozen_backbone_is_read_only():
    model = TinyModel().train()
    adapter = FrozenTorchBackbone(model, _representations)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    features, logits, probabilities = adapter.predict(torch.randn(5, 4, requires_grad=True))
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert features.shape == (5, 3)
    assert logits.shape == probabilities.shape == (5, 2)
    assert not features.requires_grad and not logits.requires_grad
    torch.testing.assert_close(probabilities.sum(1), torch.ones(5))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_copy_torch_linear_transposes_and_sets_priors():
    source = nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.copy_(torch.arange(6.0).reshape(2, 3))
        source.bias.copy_(torch.tensor([4.0, 5.0]))
    target = Linear(3, 2, device="cpu")
    copy_torch_linear_(target, source, weight_variance=0.2, bias_variance=0.3)
    torch.testing.assert_close(target.mw, source.weight.T)
    torch.testing.assert_close(target.mb, source.bias[None])
    torch.testing.assert_close(target.Sw, torch.full_like(target.Sw, 0.2))
    torch.testing.assert_close(target.Sb, torch.full_like(target.Sb, 0.3))


def test_identity_initialization_preserves_default_prior():
    target = Linear(3, 3, device="cpu")
    variance = target.Sw.clone()
    initialize_identity_linear_(target)
    torch.testing.assert_close(target.mw, torch.eye(3))
    torch.testing.assert_close(target.mb, torch.zeros_like(target.mb))
    torch.testing.assert_close(target.Sw, variance)
    with pytest.raises(ValueError, match="equal"):
        initialize_identity_linear_(Linear(3, 2, device="cpu"))
