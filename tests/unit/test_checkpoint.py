"""Unit tests for triton_tagi.checkpoint (RunDir, save/load, metrics)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from triton_tagi.checkpoint import RunDir, _extract_net_state, _restore_net_state
from triton_tagi.layers import Linear, ReLU, BatchNorm2D, ResBlock, Flatten
from triton_tagi.network import Sequential


DEVICE = "cpu"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _small_mlp() -> Sequential:
    torch.manual_seed(0)
    return Sequential(
        [Linear(8, 4, device=DEVICE), ReLU(), Linear(4, 2, device=DEVICE)],
        device=DEVICE,
    )


# ---------------------------------------------------------------------------
#  RunDir directory structure
# ---------------------------------------------------------------------------


class TestRunDir:
    def test_directories_created(self, tmp_path):
        rd = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        assert rd.path.exists()
        assert rd.checkpoints.exists()
        assert rd.figures.exists()

    def test_name_format(self, tmp_path):
        rd = RunDir("cifar10", "resnet18", "adam", base=str(tmp_path))
        # name must start with dataset_arch_optimizer_
        assert rd.path.name.startswith("cifar10_resnet18_adam_")

    def test_repr(self, tmp_path):
        rd = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        assert "RunDir(" in repr(rd)

    def test_separate_runs_have_unique_paths(self, tmp_path):
        import time
        rd1 = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        time.sleep(1.1)  # guarantee different second-level timestamp
        rd2 = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        assert rd1.path != rd2.path


# ---------------------------------------------------------------------------
#  save_config
# ---------------------------------------------------------------------------


class TestSaveConfig:
    def test_writes_json(self, tmp_path):
        rd = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        config = {"n_epochs": 10, "batch_size": 64, "sigma_v": 0.01}
        rd.save_config(config)
        assert rd.config_json.exists()
        loaded = json.loads(rd.config_json.read_text())
        assert loaded == config

    def test_overwrites_on_second_call(self, tmp_path):
        rd = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        rd.save_config({"x": 1})
        rd.save_config({"x": 2})
        assert json.loads(rd.config_json.read_text())["x"] == 2


# ---------------------------------------------------------------------------
#  append_metrics
# ---------------------------------------------------------------------------


class TestAppendMetrics:
    def test_creates_file_with_header(self, tmp_path):
        rd = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        rd.append_metrics(1, train_loss=0.5, test_acc=0.8)
        assert rd.metrics_csv.exists()
        rows = list(csv.DictReader(rd.metrics_csv.open()))
        assert rows[0]["epoch"] == "1"
        assert "train_loss" in rows[0]
        assert "test_acc" in rows[0]

    def test_appends_multiple_rows(self, tmp_path):
        rd = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        for ep in range(1, 4):
            rd.append_metrics(ep, loss=float(ep))
        rows = list(csv.DictReader(rd.metrics_csv.open()))
        assert len(rows) == 3
        assert rows[-1]["epoch"] == "3"

    def test_header_written_only_once(self, tmp_path):
        rd = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        for ep in range(1, 5):
            rd.append_metrics(ep, acc=0.9)
        lines = rd.metrics_csv.read_text().splitlines()
        assert lines[0].startswith("epoch")
        assert len(lines) == 5  # 1 header + 4 data rows


# ---------------------------------------------------------------------------
#  Network state extraction and restoration
# ---------------------------------------------------------------------------


class TestNetState:
    def test_roundtrip_mlp(self):
        net = _small_mlp()
        original_mw = [
            layer.mw.clone()
            for layer in net.layers
            if hasattr(layer, "mw")
        ]
        state = _extract_net_state(net)
        # Corrupt parameters
        for layer in net.layers:
            if hasattr(layer, "mw"):
                layer.mw.fill_(999.0)
        _restore_net_state(net, state)
        restored_mw = [layer.mw for layer in net.layers if hasattr(layer, "mw")]
        for orig, res in zip(original_mw, restored_mw):
            torch.testing.assert_close(orig, res)

    def test_only_learnable_layers_in_state(self):
        net = _small_mlp()
        state = _extract_net_state(net)
        # ReLU (index 1) should not appear
        assert 1 not in state
        # Linear layers (indices 0, 2) should appear
        assert 0 in state
        assert 2 in state

    def test_state_contains_expected_keys(self):
        net = _small_mlp()
        state = _extract_net_state(net)
        for layer_state in state.values():
            assert "mw" in layer_state
            assert "Sw" in layer_state
            assert "mb" in layer_state
            assert "Sb" in layer_state

    def test_tensors_moved_to_cpu(self):
        net = _small_mlp()
        state = _extract_net_state(net)
        for layer_state in state.values():
            for t in layer_state.values():
                assert t.device.type == "cpu"


# ---------------------------------------------------------------------------
#  Checkpoint save / load round-trip
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    def test_save_creates_file(self, tmp_path):
        rd = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        net = _small_mlp()
        path = rd.save_checkpoint(net, epoch=5, config={"n_epochs": 10})
        assert path.exists()
        assert path.name == "epoch_0005.pt"

    def test_load_latest_when_path_none(self, tmp_path):
        rd = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        net = _small_mlp()
        for ep in (1, 3, 7):
            rd.save_checkpoint(net, epoch=ep, config={})
        epoch = rd.load_checkpoint(net)
        assert epoch == 7

    def test_raises_if_no_checkpoints(self, tmp_path):
        rd = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        net = _small_mlp()
        with pytest.raises(FileNotFoundError):
            rd.load_checkpoint(net)

    def test_weights_restored_correctly(self, tmp_path):
        rd = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        net = _small_mlp()
        original = {
            i: layer.mw.clone()
            for i, layer in enumerate(net.layers)
            if hasattr(layer, "mw")
        }
        rd.save_checkpoint(net, epoch=1, config={})
        # Corrupt weights
        for layer in net.layers:
            if hasattr(layer, "mw"):
                layer.mw.fill_(0.0)
        rd.load_checkpoint(net)
        for i, layer in enumerate(net.layers):
            if hasattr(layer, "mw"):
                torch.testing.assert_close(layer.mw, original[i])

    def test_config_stored_in_checkpoint(self, tmp_path):
        rd = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        net = _small_mlp()
        config = {"n_epochs": 50, "batch_size": 128}
        path = rd.save_checkpoint(net, epoch=1, config=config)
        ck = torch.load(path, weights_only=False)
        assert ck["config"] == config

    def test_epoch_stored_in_checkpoint(self, tmp_path):
        rd = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        net = _small_mlp()
        rd.save_checkpoint(net, epoch=42, config={})
        epoch = rd.load_checkpoint(net)
        assert epoch == 42

    def test_load_specific_path(self, tmp_path):
        rd = RunDir("mnist", "mlp", "tagi", base=str(tmp_path))
        net = _small_mlp()
        path3 = rd.save_checkpoint(net, epoch=3, config={})
        # Corrupt and save again at epoch 5
        for layer in net.layers:
            if hasattr(layer, "mw"):
                layer.mw.fill_(999.0)
        rd.save_checkpoint(net, epoch=5, config={})
        # Load epoch 3 explicitly — should NOT have corrupted weights
        net2 = _small_mlp()
        for layer in net2.layers:
            if hasattr(layer, "mw"):
                layer.mw.fill_(0.0)
        epoch = rd.load_checkpoint(net2, path=path3)
        assert epoch == 3
        for layer in net2.layers:
            if hasattr(layer, "mw"):
                assert not (layer.mw == 999.0).all()
