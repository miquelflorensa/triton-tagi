"""Checkpoint round trip for a reusable TAGI classification head."""

from __future__ import annotations

import pytest
import torch

from triton_tagi import RunDir, TAGILastLayerClassifier, load_model

pytestmark = pytest.mark.cuda


def test_remax_head_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(7)
    classifier = TAGILastLayerClassifier(4, 3, head="remax", device="cuda")
    values = torch.randn(8, 4)
    labels = torch.arange(8) % 3
    classifier.train_step(values, labels, sigma_v=0.05)
    expected = classifier.predict(values).probabilities
    config = {
        "input_dim": 4,
        "num_classes": 3,
        "arm": {"head": "remax"},
        "gain_w": 0.1,
        "gain_b": 0.1,
        "remax_approximation": "lognormal",
        "remax_jacobian": "diag",
        "remax_num_quad": 48,
    }
    run = RunDir("test", "head", "tagi", base=str(tmp_path))
    path = run.save_checkpoint(classifier.net, 1, config)

    def build(saved):
        return TAGILastLayerClassifier(
            saved["input_dim"],
            saved["num_classes"],
            head=saved["arm"]["head"],
            device="cuda",
            gain_w=saved["gain_w"],
            gain_b=saved["gain_b"],
        ).net

    restored, saved_config, epoch = load_model(path, build, device="cuda")
    actual, _ = restored.forward(values.cuda())
    torch.testing.assert_close(actual, expected)
    assert saved_config == config
    assert epoch == 1
