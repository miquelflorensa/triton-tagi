"""Tests for reproducible CIFAR last-layer study utilities."""

from __future__ import annotations

import pytest
import torch

from triton_tagi.cifar_study import (
    get_spec,
    load_feature_shard,
    save_feature_shard,
    stable_hash,
    stratified_split,
)


def test_stable_hash_is_order_independent_and_specs_are_explicit():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert get_spec("cifar10").num_classes == 10
    assert get_spec("cifar100").num_classes == 100
    with pytest.raises(ValueError, match="dataset"):
        get_spec("imagenet")


def test_stratified_split_is_deterministic_and_balanced():
    labels = torch.arange(100) % 10
    train_a, validation_a = stratified_split(labels, validation_size=20, seed=2026)
    train_b, validation_b = stratified_split(labels, validation_size=20, seed=2026)
    torch.testing.assert_close(train_a, train_b)
    torch.testing.assert_close(validation_a, validation_b)
    assert train_a.numel() == 80
    assert validation_a.numel() == 20
    counts = torch.bincount(labels[validation_a], minlength=10)
    torch.testing.assert_close(counts, torch.full((10,), 2))


def test_feature_shard_roundtrip_and_fingerprint_guard(tmp_path):
    path = tmp_path / "features.pt"
    tensors = {
        "features": torch.randn(4, 3),
        "logits": torch.randn(4, 2),
        "labels": torch.tensor([0, 1, 0, 1]),
    }
    metadata = {"fingerprint": "abc", "split": "validation"}
    save_feature_shard(path, tensors, metadata)
    payload = load_feature_shard(path, expected_fingerprint="abc")
    torch.testing.assert_close(payload["features"], tensors["features"])
    assert payload["metadata"] == metadata
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_feature_shard(path, expected_fingerprint="different")
