"""Config-level tests for the ImageNet arm of the initialization study.

The ImageNet driver runs unattended overnight over a 33 GB feature cache, so
its grid and selection logic are worth pinning down without touching a GPU.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import experiments.last_layer.run_imagenet_init_study as imagenet


@pytest.fixture
def manifest(tmp_path):
    return {
        "study_id": "imagenet_init_study",
        "paths": {"features": "features", "artifacts": str(tmp_path)},
        "last_layer": {"batch_size": 256, "validation_batch_size": 1024},
        "fixed_noise_heads": ["hrc", "remax_lognormal", "remax_laplace_diag"],
        "logit_heads": ["logit_tagiv"],
        "mean_init": ["random", "zero", "backbone"],
        "tied_gains": [0.1, 0.3, 1.0],
        "sigma_v": [0.1, 0.3],
        "screen": {"seeds": [0], "epochs": 1},
        "confirm": {"seeds": [0, 1, 2], "epochs": 4},
    }


def test_screen_grid_is_the_sixty_cells_the_plan_budgeted(manifest):
    configs = imagenet.screen_configs(manifest)
    assert len(configs) == 60

    fixed = [c for c in configs if c["head"] in manifest["fixed_noise_heads"]]
    logit = [c for c in configs if c["head"] == "logit_tagiv"]
    assert len(fixed) == 54  # 3 heads x 3 arms x 3 gains x 2 sigma_v
    assert len(logit) == 6  # 2 distinct arms x 3 gains

    # logit_tagiv zeroes its own latent means, so it has no distinct random arm.
    assert {c["mean_init"] for c in logit} == {"zero", "backbone"}
    assert all(c["sigma_v"] is None for c in logit)
    # ...while the fixed-noise heads carry all three arms and a real sigma_v.
    assert {c["mean_init"] for c in fixed} == {"random", "zero", "backbone"}
    assert {c["sigma_v"] for c in fixed} == {0.1, 0.3}
    assert all(c["gain_w"] == c["gain_b"] for c in configs)

    # Every cell is distinct: a repeat would silently skip as already complete.
    assert len({json.dumps(c, sort_keys=True) for c in configs}) == len(configs)


def _write_selection(manifest, selected):
    root = imagenet.stage_root(manifest, "screen")
    root.mkdir(parents=True, exist_ok=True)
    (root / "selection.json").write_text(json.dumps({"selected": selected}))


def test_confirm_pairs_each_selected_arm_with_random(manifest):
    _write_selection(
        manifest,
        {
            "hrc": {
                "config": {
                    "head": "hrc", "sigma_v": 0.1,
                    "gain_w": 0.3, "gain_b": 0.3, "mean_init": "zero",
                }
            },
            "remax_lognormal": {
                "config": {
                    "head": "remax_lognormal", "sigma_v": 0.3,
                    "gain_w": 1.0, "gain_b": 1.0, "mean_init": "backbone",
                }
            },
        },
    )
    configs = imagenet.confirm_configs(manifest)

    assert len(configs) == 4
    arms = {
        (c["head"], c["mean_init"]) for c in configs
    }
    assert arms == {
        ("hrc", "zero"),
        ("hrc", "random"),
        ("remax_lognormal", "backbone"),
        ("remax_lognormal", "random"),
    }
    # The random counterpart keeps the selected gain and noise, so the delta
    # isolates the initialization rather than mixing in another axis.
    hrc_random = next(
        c for c in configs if c["head"] == "hrc" and c["mean_init"] == "random"
    )
    assert hrc_random["gain_w"] == 0.3
    assert hrc_random["sigma_v"] == 0.1


def test_confirm_does_not_pair_a_head_with_no_distinct_random_arm(manifest):
    """logit_tagiv's random arm is bit-identical to zero, so it is not paired."""

    _write_selection(
        manifest,
        {
            "logit_tagiv": {
                "config": {
                    "head": "logit_tagiv", "sigma_v": None,
                    "gain_w": 0.3, "gain_b": 0.3, "mean_init": "zero",
                }
            }
        },
    )
    configs = imagenet.confirm_configs(manifest)
    assert configs == [
        {
            "head": "logit_tagiv", "sigma_v": None,
            "gain_w": 0.3, "gain_b": 0.3, "mean_init": "zero",
        }
    ]


def test_select_prefers_nll_within_one_point_of_the_best_accuracy(manifest):
    root = imagenet.stage_root(manifest, "screen")
    cells = [
        # (accuracy, nll) -- the top accuracy, but a poor NLL
        ("a", "zero", 0.700, 1.40),
        # within 1pp of the best accuracy and a much better NLL: the winner
        ("b", "backbone", 0.695, 1.10),
        # a better NLL still, but too far down on accuracy to be eligible
        ("c", "random", 0.640, 0.95),
    ]
    for name, mean_init, accuracy, nll in cells:
        run_dir = root / "hrc" / f"{name}_seed0"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "head": "hrc", "seed": 0, "sigma_v": 0.1,
                    "gain_w": 0.3, "gain_b": 0.3, "mean_init": mean_init,
                }
            )
        )
        (run_dir / "history.json").write_text(
            json.dumps(
                [
                    {"epoch": 0.0, "val_accuracy": 0.001, "val_nll": 9.0,
                     "val_brier": 1.0, "val_ece": 0.9},
                    {"epoch": 1.0, "val_accuracy": accuracy, "val_nll": nll,
                     "val_brier": 0.4, "val_ece": 0.05},
                ]
            )
        )

    imagenet.select_stage(SimpleNamespace(stage="screen"), manifest)

    selection = json.loads((root / "selection.json").read_text())
    assert selection["selected"]["hrc"]["config"]["mean_init"] == "backbone"
    # The epoch-0 prior is never selectable: it precedes any data.
    assert selection["selected"]["hrc"]["record"]["epoch"] == 1.0
