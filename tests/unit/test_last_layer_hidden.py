"""Tests for the hidden-layer option of the frozen last-layer classifier."""

from __future__ import annotations

import pytest
import torch

from triton_tagi.classification import TAGILastLayerClassifier
from triton_tagi.layers import Linear, ReLU

pytestmark = pytest.mark.cuda

FIXED_NOISE_HEADS = ("remax_lognormal", "remax_laplace_diag", "hrc")


def build(head: str, hidden_dims=(), *, num_classes: int = 10, input_dim: int = 6, **kwargs):
    torch.manual_seed(0)
    return TAGILastLayerClassifier(
        input_dim,
        num_classes,
        head=head,
        device="cuda",
        hidden_dims=hidden_dims,
        sigma_v=0.1,
        **kwargs,
    )


# ── Structure ────────────────────────────────────────────────────────────────


def test_default_is_the_single_layer_head():
    classifier = build("hrc")
    assert classifier.hidden_dims == ()
    assert classifier.hidden_linears == []
    assert [type(layer) for layer in classifier.net.layers] == [Linear]


@pytest.mark.parametrize("hidden_dims", [(8,), (8, 5)])
def test_each_width_contributes_a_linear_and_a_relu(hidden_dims):
    classifier = build("hrc", hidden_dims)
    expected = [Linear, ReLU] * len(hidden_dims) + [Linear]
    assert [type(layer) for layer in classifier.net.layers] == expected
    assert classifier.hidden_dims == hidden_dims
    assert [layer.out_features for layer in classifier.hidden_linears] == list(hidden_dims)


def test_the_stack_chains_fan_in_from_the_features_to_the_head():
    classifier = build("hrc", (8, 5), input_dim=6)
    assert [layer.in_features for layer in classifier.hidden_linears] == [6, 8]
    # input_dim stays the dimension of the frozen features the head is fed.
    assert classifier.input_dim == 6
    assert classifier.linear.in_features == 5


def test_the_head_keeps_its_own_output_width():
    plain = build("remax_lognormal")
    deep = build("remax_lognormal", (8,))
    assert deep.linear.out_features == plain.linear.out_features


def test_hidden_gains_default_to_the_output_gains_and_can_be_overridden():
    shared = build("hrc", (8,), gain_w=0.3, gain_b=0.2)
    assert (shared.hidden_gain_w, shared.hidden_gain_b) == (0.3, 0.2)
    split = build("hrc", (8,), gain_w=0.3, gain_b=0.2, hidden_gain_w=1.0, hidden_gain_b=0.5)
    assert (split.hidden_gain_w, split.hidden_gain_b) == (1.0, 0.5)
    assert split.hidden_linears[0].Sw.mean() > shared.hidden_linears[0].Sw.mean()


# ── Validation ───────────────────────────────────────────────────────────────


def test_nonpositive_hidden_width_is_rejected():
    with pytest.raises(ValueError, match="hidden width must be positive"):
        build("hrc", (8, 0))


def test_negative_hidden_gain_is_rejected():
    with pytest.raises(ValueError, match="hidden_gain_w and hidden_gain_b"):
        build("hrc", (8,), hidden_gain_w=-1.0)


def test_backbone_warm_start_is_rejected_behind_a_hidden_stack():
    weight = torch.randn(10, 6)
    bias = torch.randn(10)
    with pytest.raises(ValueError, match="the copy is undefined"):
        build("hrc", (8,), mean_init="backbone", backbone_fc=(weight, bias))


def test_heteroscedastic_logit_variance_head_is_rejected_behind_a_hidden_stack():
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="logit_variance_weight_share=0"):
        TAGILastLayerClassifier(
            6,
            10,
            head="logit_tagiv",
            device="cuda",
            hidden_dims=(8,),
            logit_variance_weight_share=0.5,
            logit_variance_feature_energy=6.0,
        )


# ── Behaviour ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("head", FIXED_NOISE_HEADS)
def test_every_fixed_noise_head_trains_through_the_hidden_stack(head):
    torch.manual_seed(0)
    features = torch.randn(256, 6, device="cuda")
    labels = torch.randint(0, 10, (256,), device="cuda")
    classifier = build(head, (16,))
    before = [layer.mw.clone() for layer in classifier.hidden_linears]
    for _ in range(5):
        classifier.train_step(features, labels)
    after = [layer.mw for layer in classifier.hidden_linears]
    for old, new in zip(before, after):
        assert not torch.allclose(old, new), "the hidden layer received no update"
    probabilities = classifier.predict(features).probabilities
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(1), torch.ones(256, device="cuda"), atol=1e-4)


def test_zero_mean_init_zeroes_the_head_and_leaves_the_hidden_stack_drawn():
    classifier = build("hrc", (16,), mean_init="zero")
    assert torch.count_nonzero(classifier.linear.mw) == 0
    assert torch.count_nonzero(classifier.hidden_linears[0].mw) > 0


# ── Checkpointing ────────────────────────────────────────────────────────────


def test_checkpoint_round_trips_the_hidden_stack(tmp_path):
    features = torch.randn(64, 6, device="cuda")
    labels = torch.randint(0, 10, (64,), device="cuda")
    classifier = build("remax_lognormal", (16, 8))
    for _ in range(3):
        classifier.train_step(features, labels)
    path = classifier.save(tmp_path / "head.pt")

    restored, _ = TAGILastLayerClassifier.load(path, device="cuda")
    assert restored.hidden_dims == (16, 8)
    for original, loaded in zip(classifier.hidden_linears, restored.hidden_linears):
        for name in ("mw", "Sw", "mb", "Sb"):
            torch.testing.assert_close(getattr(original, name), getattr(loaded, name))
    torch.testing.assert_close(
        classifier.predict(features).probabilities,
        restored.predict(features).probabilities,
    )


def test_a_single_layer_checkpoint_still_loads(tmp_path):
    classifier = build("hrc")
    path = classifier.save(tmp_path / "flat.pt")
    payload = torch.load(path, weights_only=False)
    # A checkpoint written before hidden_dims existed carries neither key.
    payload.pop("hidden_state")
    payload["config"].pop("hidden_dims")
    payload["config"].pop("hidden_gain_w")
    payload["config"].pop("hidden_gain_b")
    torch.save(payload, path)

    restored, _ = TAGILastLayerClassifier.load(path, device="cuda")
    assert restored.hidden_dims == ()
