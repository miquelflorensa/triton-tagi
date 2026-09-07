"""
triton-tagi: Tractable Approximate Gaussian Inference on Triton
================================================================

A minimal, GPU-accelerated Python reimplementation of cuTAGI
(https://github.com/lhnguyen102/cuTAGI) using fused Triton kernels.

The surface is deliberately small: the layer set mirrors what is needed to
reproduce cuTAGI's headline examples (regression, MNIST MLP/CNN, CIFAR-10 CNN
and ResNet-18) with numerical parity. Additional layers, optimizers, and
diagnostics live under ``_archive/`` at the repository root.

Modules
-------
- ``layers``  : Bayesian layers
- ``update``  : observation innovation and parameter update rules
- ``network`` : ``Sequential`` network builder
- ``kernels`` : low-level Triton kernels

Numerical precision
-------------------
TF32 matmul is disabled at import time. cuTAGI uses scalar FMA loops
(``__fmaf_rn``) with near-fp64 accuracy; leaving TF32 enabled in
PyTorch/Triton would introduce systematic ~1e-3 errors in the variance
forward pass and break numerical parity.
"""

import torch

torch.backends.cuda.matmul.allow_tf32 = False

from .base import Layer, LearnableLayer
from .classification import (
    ClassificationPrediction,
    TAGILastLayerClassifier,
    TrainingHistory,
    normalize_class_probabilities,
)
from .checkpoint import RunDir, load_model
from .hrc_probit import (
    DEFAULT_LOG_TAU,
    LogTauPosterior,
    alpha_from_log_tau,
    fit_hrc_log_tau,
    fit_hrc_log_tau_laplace,
    hrc_branch_log_probabilities,
    hrc_class_probabilities,
    hrc_log_partition_deviation,
    hrc_log_probs,
    hrc_log_tau_score_curvature,
    hrc_negative_log_likelihood,
    hrc_path_log_scores,
    log_tau_from_alpha,
    probit_log_tau_factor_score_curvature,
    update_log_tau_gaussian,
)
from .hrc_softmax import (
    HierarchicalSoftmax,
    class_to_obs,
    class_to_obs_full,
    get_predicted_labels,
    labels_to_hrc,
    labels_to_hrc_mask,
    obs_to_class_probs,
    obs_to_class_probs_tagiv,
)
from .hsm_calibration import (
    DEFAULT_GAIN_ORDER,
    DEFAULT_OWEN_ORDER,
    GAIN_SHARING,
    GainGroups,
    HsmClassMoments,
    HsmNodeMoments,
    LogGainPosterior,
    branch_codes,
    calibrate_hsm_log_gain_adf,
    custom_gain_groups,
    expected_bernoulli_variance,
    fit_hsm_log_gain,
    gain_groups,
    hsm_adf_projection,
    hsm_calibration_visits,
    hsm_class_moments,
    hsm_class_probabilities,
    hsm_cross_class_covariance,
    hsm_log_posterior_on_grid,
    hsm_negative_log_likelihood,
    hsm_node_moments,
    hsm_partition_deviation,
    node_depths,
    owens_t,
    probit_gaussian_moments,
)
from .inference_init import inference_init
from .interop import (
    FrozenTorchBackbone,
    TorchCheckpointInfo,
    copy_torch_linear_,
    initialize_identity_linear_,
    load_torch_checkpoint,
)
from .layers import (
    Add,
    AvgPool2D,
    BatchNorm2D,
    Conv2D,
    Embedding,
    EvenExp,
    EvenSoftplus,
    Flatten,
    LayerNorm,
    Linear,
    MaxPool2D,
    MultiheadAttentionV2,
    PositionalEncoding,
    RMSNorm,
    ReLU,
    Remax,
    ResBlock,
)
from .logit_tagiv import (
    LOGIT_VARIANCE_FLOOR,
    LogitCalibration,
    LogitUncertainty,
    center_logits,
    compute_logit_tagiv_innovation,
    fit_logit_calibration,
    logit_exp_variance_moments,
    logit_feature_energy,
    compute_logit_mean_innovation,
    compute_logit_replicate_innovation,
    gaussian_shrinkage_,
    logit_replicate_variance,
    power_compress_variance,
    replicate_log_variance_noise,
    replicate_log_variance_offset,
    logit_tagiv_predictive_nll,
    logit_tagiv_predictive_probs,
    logit_target_scale,
    logit_tagiv_uncertainty,
    prepare_logit_targets,
    split_logit_tagiv_outputs,
    standard_normal_base_samples,
    logit_variance_head_prior,
    logit_variance_prior_split,
)
from .metrics import (
    classwise_calibration_error,
    classification_metrics,
    epistemic_convergence,
    evaluate_ood,
    evaluate_ood_comprehensive,
    expected_calibration_error,
    fit_softmax_temperature,
    negative_max_probability,
    ood_detection_metrics,
    ood_detection_metrics_full,
    predictive_entropy,
    probability_simplex_deviation,
    selective_classification_metrics,
    training_required_epoch,
)
from .feature_support import (
    BayesianFeatureSupportGate,
    FeatureSupportPrediction,
)
from .network import Sequential
from .param_init import (
    gaussian_param_init,
    he_init,
    init_weight_bias_conv2d,
    init_weight_bias_linear,
    init_weight_bias_norm,
    xavier_init,
)

__version__ = "0.2.0"
__all__ = [
    # ABCs
    "Layer",
    "LearnableLayer",
    # Network
    "Sequential",
    "TAGILastLayerClassifier",
    "ClassificationPrediction",
    # Layers
    "TrainingHistory",
    "Add",
    "AvgPool2D",
    "BatchNorm2D",
    "Conv2D",
    "Embedding",
    "EvenExp",
    "EvenSoftplus",
    "Flatten",
    "LayerNorm",
    "Linear",
    "MaxPool2D",
    "MultiheadAttentionV2",
    "PositionalEncoding",
    "RMSNorm",
    "ReLU",
    "Remax",
    "ResBlock",
    # Parameter initialisation
    "he_init",
    "xavier_init",
    "gaussian_param_init",
    "init_weight_bias_linear",
    "init_weight_bias_conv2d",
    "init_weight_bias_norm",
    "inference_init",
    # Hierarchical softmax
    "HierarchicalSoftmax",
    "class_to_obs",
    "class_to_obs_full",
    "labels_to_hrc",
    "labels_to_hrc_mask",
    "obs_to_class_probs",
    "obs_to_class_probs_tagiv",
    "get_predicted_labels",
    # Calibrated hierarchical probit
    "DEFAULT_LOG_TAU",
    "LogTauPosterior",
    "alpha_from_log_tau",
    "log_tau_from_alpha",
    "hrc_branch_log_probabilities",
    "hrc_path_log_scores",
    "hrc_log_probs",
    "hrc_class_probabilities",
    "hrc_negative_log_likelihood",
    "hrc_log_partition_deviation",
    "fit_hrc_log_tau",
    "hrc_log_tau_score_curvature",
    "probit_log_tau_factor_score_curvature",
    "fit_hrc_log_tau_laplace",
    "update_log_tau_gaussian",
    # Hierarchical probit calibration with a positive uncertain gain
    "DEFAULT_GAIN_ORDER",
    "DEFAULT_OWEN_ORDER",
    "GAIN_SHARING",
    "GainGroups",
    "HsmNodeMoments",
    "HsmClassMoments",
    "LogGainPosterior",
    "owens_t",
    "probit_gaussian_moments",
    "expected_bernoulli_variance",
    "branch_codes",
    "node_depths",
    "gain_groups",
    "custom_gain_groups",
    "hsm_node_moments",
    "hsm_class_moments",
    "hsm_class_probabilities",
    "hsm_cross_class_covariance",
    "hsm_negative_log_likelihood",
    "hsm_partition_deviation",
    "hsm_calibration_visits",
    "hsm_log_posterior_on_grid",
    "fit_hsm_log_gain",
    "hsm_adf_projection",
    "calibrate_hsm_log_gain_adf",
    # Run management
    "RunDir",
    "load_model",
    "FrozenTorchBackbone",
    "TorchCheckpointInfo",
    "load_torch_checkpoint",
    "copy_torch_linear_",
    "initialize_identity_linear_",
    "normalize_class_probabilities",
    "classification_metrics",
    "expected_calibration_error",
    "fit_softmax_temperature",
    "classwise_calibration_error",
    "selective_classification_metrics",
    "probability_simplex_deviation",
    "predictive_entropy",
    "negative_max_probability",
    "ood_detection_metrics",
    "ood_detection_metrics_full",
    "evaluate_ood",
    "evaluate_ood_comprehensive",
    "training_required_epoch",
    "epistemic_convergence",
    "LOGIT_VARIANCE_FLOOR",
    "LogitCalibration",
    "LogitUncertainty",
    "center_logits",
    "compute_logit_tagiv_innovation",
    "compute_logit_mean_innovation",
    "compute_logit_replicate_innovation",
    "logit_replicate_variance",
    "gaussian_shrinkage_",
    "power_compress_variance",
    "replicate_log_variance_noise",
    "replicate_log_variance_offset",
    "fit_logit_calibration",
    "logit_exp_variance_moments",
    "logit_feature_energy",
    "logit_variance_prior_split",
    "logit_tagiv_predictive_nll",
    "logit_tagiv_predictive_probs",
    "logit_tagiv_uncertainty",
    "logit_target_scale",
    "logit_variance_head_prior",
    "prepare_logit_targets",
    "split_logit_tagiv_outputs",
    "standard_normal_base_samples",
    "BayesianFeatureSupportGate",
    "FeatureSupportPrediction",
]
