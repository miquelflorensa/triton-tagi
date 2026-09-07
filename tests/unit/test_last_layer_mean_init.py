"""Tests for the mean_init axis of the last-layer initialization study."""

from __future__ import annotations

import pytest
import torch

from triton_tagi.classification import TAGILastLayerClassifier
from triton_tagi.hrc_softmax import class_to_obs, class_to_obs_full, project_classes_to_nodes

pytestmark = pytest.mark.cuda

FIXED_NOISE_HEADS = ("remax_lognormal", "remax_laplace_diag", "hrc")


def backbone(num_classes: int, input_dim: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(num_classes, input_dim, generator=generator),
        torch.randn(num_classes, generator=generator),
    )


def build(head: str, mean_init: str, *, num_classes: int = 10, input_dim: int = 6, **kwargs):
    torch.manual_seed(0)
    extra = {"backbone_fc": backbone(num_classes, input_dim)} if mean_init == "backbone" else {}
    if head == "logit_tagiv":
        # The heteroscedastic variance head needs the energy of the features it
        # will see before it can split its prior.
        extra["logit_variance_feature_energy"] = float(input_dim)
    return TAGILastLayerClassifier(
        input_dim,
        num_classes,
        head=head,
        device="cuda",
        mean_init=mean_init,
        sigma_v=None if head == "logit_tagiv" else 0.1,
        **extra,
        **kwargs,
    )


# ── Validation ────────────────────────────────────────────────────────────────


def test_unknown_mean_init_is_rejected_on_cuda():
    with pytest.raises(ValueError, match="Unknown mean_init"):
        TAGILastLayerClassifier(4, 3, head="hrc", device="cuda", mean_init="he")


def test_backbone_init_without_weights_is_rejected_on_cuda():
    with pytest.raises(ValueError, match="requires backbone_fc"):
        TAGILastLayerClassifier(4, 3, head="hrc", device="cuda", mean_init="backbone")


def test_backbone_weights_without_the_backbone_arm_are_rejected_on_cuda():
    with pytest.raises(ValueError, match="meaningless"):
        TAGILastLayerClassifier(4, 3, head="hrc", device="cuda", backbone_fc=backbone(3, 4))


def test_backbone_weights_must_match_the_head_shape_on_cuda():
    with pytest.raises(ValueError, match="num_classes, input_dim"):
        TAGILastLayerClassifier(
            6, 10, head="remax_lognormal", device="cuda",
            mean_init="backbone", backbone_fc=backbone(10, 5),
        )


def test_backbone_bias_must_match_the_head_shape_on_cuda():
    weight, _ = backbone(10, 6)
    with pytest.raises(ValueError, match=r"num_classes,\)"):
        TAGILastLayerClassifier(
            6, 10, head="remax_lognormal", device="cuda",
            mean_init="backbone", backbone_fc=(weight, torch.zeros(9)),
        )


# ── The three arms ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("head", (*FIXED_NOISE_HEADS, "logit_tagiv"))
def test_zero_arm_gives_exactly_zero_latent_means_on_cuda(head):
    classifier = build(head, "zero")
    stride = 2 if head == "logit_tagiv" else 1
    latent = slice(0, None, stride)
    assert torch.count_nonzero(classifier.linear.mw[:, latent]) == 0
    assert torch.count_nonzero(classifier.linear.mb[:, latent]) == 0


@pytest.mark.parametrize("head", (*FIXED_NOISE_HEADS, "logit_tagiv"))
def test_every_arm_leaves_the_prior_variances_identical_on_cuda(head):
    reference = build(head, "random")
    for mean_init in ("zero", "backbone"):
        other = build(head, mean_init)
        torch.testing.assert_close(other.linear.Sw, reference.linear.Sw)
        torch.testing.assert_close(other.linear.Sb, reference.linear.Sb)


def test_random_arm_is_the_untouched_status_quo_on_cuda():
    torch.manual_seed(0)
    explicit = TAGILastLayerClassifier(6, 10, head="hrc", device="cuda", mean_init="random")
    torch.manual_seed(0)
    default = TAGILastLayerClassifier(6, 10, head="hrc", device="cuda")
    torch.testing.assert_close(explicit.linear.mw, default.linear.mw)
    torch.testing.assert_close(explicit.linear.mb, default.linear.mb)
    assert torch.count_nonzero(default.linear.mw) > 0


@pytest.mark.parametrize("head", ("remax_lognormal", "remax_laplace_diag"))
def test_backbone_arm_copies_the_fc_layer_for_a_remax_head_on_cuda(head):
    weight, bias = backbone(10, 6)
    classifier = build(head, "backbone")
    torch.testing.assert_close(classifier.linear.mw, weight.T.cuda())
    torch.testing.assert_close(classifier.linear.mb, bias.reshape(1, -1).cuda())


def test_backbone_arm_projects_onto_tree_nodes_for_hrc_on_cuda():
    weight, bias = backbone(10, 6)
    classifier = build("hrc", "backbone")
    assert classifier.linear.mw.shape == (6, classifier.hrc.len)
    centered = weight - weight.mean(dim=0, keepdim=True)
    expected_w, expected_b = project_classes_to_nodes(
        classifier.hrc, centered.cuda(), (bias - bias.mean()).cuda()
    )
    torch.testing.assert_close(classifier.linear.mw, expected_w.T)
    torch.testing.assert_close(classifier.linear.mb, expected_b.reshape(1, -1))


def test_backbone_arm_writes_only_the_latent_channel_of_logit_tagiv_on_cuda():
    prior = build("logit_tagiv", "random")
    warm = build("logit_tagiv", "backbone")
    odd = slice(1, None, 2)
    # The variance stream keeps the prior its own initializer solved for.
    torch.testing.assert_close(warm.linear.mw[:, odd], prior.linear.mw[:, odd])
    torch.testing.assert_close(warm.linear.mb[:, odd], prior.linear.mb[:, odd])
    # The logit stream is a centered, logit_scale-normalized copy of the teacher.
    weight, bias = backbone(10, 6)
    torch.testing.assert_close(
        warm.linear.mw[:, 0::2], (weight - weight.mean(dim=0, keepdim=True)).T.cuda()
    )


def test_logit_tagiv_random_and_zero_arms_coincide_on_cuda():
    """The head zeroes its own latent means, so it has only two distinct arms."""

    random_arm = build("logit_tagiv", "random")
    zero_arm = build("logit_tagiv", "zero")
    torch.testing.assert_close(random_arm.linear.mw, zero_arm.linear.mw)
    torch.testing.assert_close(random_arm.linear.mb, zero_arm.linear.mb)


# ── Feature transform compensation ────────────────────────────────────────────


def test_backbone_arm_compensates_the_feature_transform_on_cuda():
    """The warm start must reproduce the backbone on the features it was fit to."""

    weight, bias = backbone(10, 6)
    features = torch.randn(4, 6, generator=torch.Generator().manual_seed(7))
    classifier = build(
        "remax_lognormal",
        "backbone",
        feature_mean=[0.5] * 6,
        feature_scale=2.0,
    )
    transformed = (features.cuda() - 0.5) / 2.0
    reproduced = transformed @ classifier.linear.mw + classifier.linear.mb
    expected = features.cuda() @ weight.T.cuda() + bias.cuda()
    torch.testing.assert_close(reproduced, expected, atol=1e-4, rtol=1e-4)


# ── Serialization ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mean_init", ("random", "zero", "backbone"))
def test_config_records_the_mean_init_arm_on_cuda(mean_init):
    assert build("hrc", mean_init).config()["mean_init"] == mean_init


def test_checkpoint_roundtrips_a_backbone_initialized_head_on_cuda(tmp_path):
    classifier = build("hrc", "backbone")
    path = classifier.save(tmp_path / "head.pt")
    restored, _ = TAGILastLayerClassifier.load(path, device="cuda")
    assert restored.mean_init == "backbone"
    assert restored.config()["mean_init"] == "backbone"
    torch.testing.assert_close(restored.linear.mw, classifier.linear.mw)
    torch.testing.assert_close(restored.linear.mb, classifier.linear.mb)


# ── The projection itself ─────────────────────────────────────────────────────


@pytest.mark.parametrize("num_classes", (10, 100))
@pytest.mark.parametrize("builder", (class_to_obs, class_to_obs_full))
def test_projection_scores_each_node_as_its_branch_contrast(num_classes, builder):
    hrc = builder(num_classes)
    weight, bias = backbone(num_classes, 5)
    node_weight, node_bias = project_classes_to_nodes(hrc, weight, bias)
    assert node_weight.shape == (hrc.len, 5)
    assert node_bias.shape == (hrc.len,)
    mask = hrc.path_mask()
    for node in range(hrc.len):
        branches: dict[bool, list[int]] = {True: [], False: []}
        for klass in range(num_classes):
            for level in range(hrc.n_obs):
                if mask[klass, level] == 0 or int(hrc.idx[klass, level]) - 1 != node:
                    continue
                branches[bool(hrc.obs[klass, level] > 0)].append(klass)
        expected = torch.zeros(5)
        for is_plus, classes in branches.items():
            if classes:
                expected += (1.0 if is_plus else -1.0) * weight[classes].mean(dim=0)
        torch.testing.assert_close(node_weight[node], expected, atol=1e-5, rtol=1e-5)


def test_projection_on_the_full_tree_is_invariant_to_the_class_gauge():
    hrc = class_to_obs_full(100)
    weight, bias = backbone(100, 5)
    shift = torch.randn(1, 5)
    torch.testing.assert_close(
        project_classes_to_nodes(hrc, weight, bias)[0],
        project_classes_to_nodes(hrc, weight + shift, bias + 2.0)[0],
        atol=1e-5,
        rtol=1e-5,
    )


def test_projection_rejects_a_class_count_the_tree_does_not_encode():
    with pytest.raises(ValueError, match="the tree encodes"):
        project_classes_to_nodes(class_to_obs(10), torch.randn(9, 5))
