"""Reusable TAGI classification heads over frozen deterministic features."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

from .cdf_remax import (
    DEFAULT_HERMITE_ORDER as DEFAULT_CDF_HERMITE_ORDER,
)
from .cdf_remax import (
    DEFAULT_SCALE_ORDER as DEFAULT_CDF_SCALE_ORDER,
)
from .cdf_remax import (
    remax_scale_moments,
    remax_uncertainty_decomposition,
)
from .hrc_probit import (
    DEFAULT_LOG_TAU,
    fit_hrc_log_tau,
    hrc_log_probs,
    log_tau_from_alpha,
)
from .hrc_softmax import (
    HierarchicalSoftmax,
    class_to_obs,
    class_to_obs_full,
    obs_to_class_probs_tagiv,
    project_classes_to_nodes,
)
from .hsm_calibration import (
    DEFAULT_GAIN_ORDER,
    GainGroups,
    LogGainPosterior,
    fit_hsm_log_gain,
    hsm_class_probabilities,
)
from .layers import EvenExp, EvenProbit, EvenSoftplus, Linear, ReLU, Remax
from .logit_tagiv import (
    LOGIT_VARIANCE_FLOOR,
    LogitCalibration,
    fit_logit_calibration,
    gaussian_shrinkage_,
    logit_tagiv_predictive_probs,
    logit_variance_head_prior,
    logit_variance_prior_split,
    power_compress_variance,
    split_logit_tagiv_outputs,
    standard_normal_base_samples,
)
from .network import Sequential
from .probitree import (
    ProbiTree,
    probitree_log_probs,
    probitree_uniform_reference_means,
)
from .remax_kernels import DEFAULT_LAPLACE_ORDER
from .remax_scale import (
    DEFAULT_PRIOR_VARIANCE,
    REMAX_SCALE_FIT_METHODS,
    REMAX_SCALE_METHODS,
    LogScalePosterior,
    calibrate_remax_log_scale_adf,
    fit_remax_log_scale,
)
from .update.observation import categorical_predictive_probs

_FIXED_NOISE_HEADS = {
    "probit_ovr",
    "remax_lognormal",
    "remax_laplace",
    "remax_laplace_diag",
    "hrc",
}
_UNIT_PROBIT_HEADS = {"hrc_probit"}
_PROBITREE_HEADS = {"probitree"}
_HRC_HEADS = {"hrc", "hrc_probit", "hrc_tagiv"}
_TAGIV_HEADS = {"categorical_tagiv", "hrc_tagiv"}
_LOGIT_TARGET_HEADS = {"logit_tagiv"}
_CDF_REMAX_HEADS = {"cdf_remax"}
_MEAN_INIT_MODES = ("random", "zero", "backbone")

# CDF-Remax head defaults. epsilon and kappa carry units of squared logits and
# bound the learnable aleatoric noise to (epsilon, epsilon + kappa); the three
# quadrature orders are the converged defaults of :mod:`triton_tagi.cdf_remax`,
# and the training order is the one :meth:`Sequential.step_cdf_tagiv` uses.
DEFAULT_CDF_EPSILON: float = 0.02
DEFAULT_CDF_KAPPA: float = 1.5
DEFAULT_CDF_ALEATORIC_INIT: float = 0.1
DEFAULT_CDF_TRAIN_HERMITE_ORDER: int = 64

# Named once so the constructor can tell a value the caller passed from one it
# never touched, and reject the former for every head but cdf_remax.
_CDF_REMAX_DEFAULTS: dict[str, Any] = {
    "cdf_epsilon": DEFAULT_CDF_EPSILON,
    "cdf_kappa": DEFAULT_CDF_KAPPA,
    "cdf_aleatoric_init": DEFAULT_CDF_ALEATORIC_INIT,
    "cdf_laplace_order": DEFAULT_LAPLACE_ORDER,
    "cdf_hermite_order": DEFAULT_CDF_HERMITE_ORDER,
    "cdf_scale_order": DEFAULT_CDF_SCALE_ORDER,
    "cdf_train_hermite_order": DEFAULT_CDF_TRAIN_HERMITE_ORDER,
    "cdf_decomposition_samples": None,
}


def normalize_class_probabilities(probabilities: Tensor, eps: float = 1e-12) -> Tensor:
    """Clamp numerical negatives and normalize each class-probability row."""

    probabilities = probabilities.clamp_min(0.0)
    denominator = probabilities.sum(dim=-1, keepdim=True)
    if bool((denominator <= eps).any()):
        raise ValueError("At least one prediction has zero total probability mass")
    return probabilities / denominator.clamp_min(eps)


def _softplus_inverse(value: float) -> float:
    if value <= 0.0:
        raise ValueError("v2bar_init must be positive")
    return value if value > 20.0 else math.log(math.expm1(value))


@dataclass(frozen=True)
class ClassificationPrediction:
    probabilities: Tensor
    output_mean: Tensor
    output_variance: Tensor
    epistemic_variance: Tensor | None = None
    aleatoric_variance: Tensor | None = None
    diagnostics: dict[str, Tensor] | None = None


@dataclass(frozen=True)
class TrainingHistory:
    """Immutable epoch records returned by TAGILastLayerClassifier.fit."""

    records: tuple[dict[str, float], ...]

    def values(self, name: str) -> list[float]:
        return [record[name] for record in self.records if name in record]


class TAGILastLayerClassifier:
    """A trainable TAGI linear classification head over fixed representations.

    Supported heads are probit_ovr, remax_lognormal, remax_laplace_diag, hrc,
    hrc_probit, probitree, categorical_tagiv, hrc_tagiv, logit_tagiv, and
    cdf_remax. The legacy head="remax" spelling remains supported and is
    resolved from remax_approximation.

    head="hrc_probit" is the hierarchical-probit classifier. It defaults to the
    full K-leaf tree and has no free scale: unit latent probit noise fixes the
    latent gauge, the branch offsets follow from the class prior, the leaf
    probabilities sum to one without normalization, and nothing is calibrated
    after training. It therefore takes neither sigma_v nor a temperature.
    hrc_tree="padded", head="hrc", and calibrate_hrc_log_tau exist as
    ablations against it.

    head="probitree" is the exact-K direct-probit model. It has no gate
    offsets and uses the fixed observation-noise variance ``probitree_r``. Like
    every other head here it accepts minibatches, summing the per-sample gate
    messages into one capped parameter update. Its complete canonical class
    mapping is included in checkpoints.

    head="cdf_remax" is the CDF-TAGI-V/Remax head. One interleaved 2K layer
    carries the prediction head Z_i ~ N(mu_i, e_i) on the even slots and the
    variance head U_i ~ N(nu_i, r_i) on the odd ones, with bounded aleatoric
    logit noise h(u) = cdf_epsilon + cdf_kappa Phi(u). It appends no Remax
    layer: class probabilities are the analytic Laplace-Remax average over both
    heads and the shared log-scale L ~ N(cdf_scale_mean, cdf_scale_variance),
    computed in predict. The scale is fitted after training, and the
    uncalibrated head is cdf_scale_mean = cdf_scale_variance = 0, i.e. s = 1.
    It learns its observation variance and so takes no sigma_v, and its
    epistemic/aleatoric split costs a nested quasi-Monte Carlo integration, so
    it is opt-in through cdf_decomposition_samples.

    Every head except logit_tagiv trains on integer class labels. logit_tagiv
    regresses continuous teacher logits, so its train_step and fit take
    ``targets`` and use ``labels`` only for validation metrics.

    ``hidden_dims`` puts a TAGI hidden stack in front of the head: each width
    contributes ``Linear`` + ``ReLU`` (the exact-moment mixture ReLU), and the
    head's own output layer then reads hidden activations rather than the
    frozen features. The default ``()`` is the single-layer head every existing
    configuration uses, so nothing changes for them. Two consequences are worth
    stating: ``mean_init='backbone'`` is rejected, because a
    ``num_classes x input_dim`` fc cannot warm-start an output layer of a
    different fan-in; and ``mean_init='zero'`` zeroes only the output layer, so
    the hidden stack keeps its He draw and the first update reaches it only
    once the output weight means leave zero.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        head: str = "remax",
        device: str | torch.device = "cuda",
        hrc_tree: str = "auto",
        hrc_prior_offsets: bool = True,
        hrc_log_tau: float | None = None,
        gain_w: float = 0.1,
        gain_b: float = 0.1,
        hidden_dims: Sequence[int] = (),
        hidden_gain_w: float | None = None,
        hidden_gain_b: float | None = None,
        mean_init: str = "random",
        backbone_fc: tuple[Tensor, Tensor] | None = None,
        feature_mean: Tensor | list[float] | None = None,
        feature_scale: float = 1.0,
        sigma_v: float | None = None,
        probit_tau2: float = 1.0,
        probitree_r: float = 1.0,
        probitree_tree: ProbiTree | dict[str, Any] | None = None,
        probitree_reference_variances: Tensor | list[float] | None = None,
        remax_approximation: str = "lognormal",
        remax_jacobian: str = "diag",
        remax_num_quad: int = 48,
        v2bar_init: float = 0.01,
        v2bar_weight_var: float = 1e-6,
        v2bar_bias_var: float = 1.0,
        logit_variance_floor: float = LOGIT_VARIANCE_FLOOR,
        logit_aleatoric_init: float = 0.1,
        logit_variance_cv: float = 0.5,
        logit_variance_weight_share: float = 0.5,
        logit_variance_feature_energy: float | None = None,
        logit_scale: float = 1.0,
        logit_temperature: float = 1.0,
        logit_alpha: float = 1.0,
        logit_epistemic_scale: float = 1.0,
        logit_num_samples: int = 256,
        logit_seed: int = 0,
        logit_variance_power: float = 1.0,
        logit_variance_reference: float | None = None,
        cdf_epsilon: float = DEFAULT_CDF_EPSILON,
        cdf_kappa: float = DEFAULT_CDF_KAPPA,
        cdf_aleatoric_init: float = DEFAULT_CDF_ALEATORIC_INIT,
        cdf_laplace_order: int = DEFAULT_LAPLACE_ORDER,
        cdf_hermite_order: int = DEFAULT_CDF_HERMITE_ORDER,
        cdf_scale_order: int = DEFAULT_CDF_SCALE_ORDER,
        cdf_train_hermite_order: int = DEFAULT_CDF_TRAIN_HERMITE_ORDER,
        cdf_decomposition_samples: int | None = None,
    ) -> None:
        if input_dim < 1 or num_classes < 2:
            raise ValueError("input_dim must be positive and num_classes must be at least two")
        if gain_w < 0.0 or gain_b < 0.0:
            raise ValueError("gain_w and gain_b must be nonnegative")
        hidden_dims = tuple(int(width) for width in hidden_dims)
        if any(width < 1 for width in hidden_dims):
            raise ValueError("every hidden width must be positive")
        hidden_gain_w = gain_w if hidden_gain_w is None else float(hidden_gain_w)
        hidden_gain_b = gain_b if hidden_gain_b is None else float(hidden_gain_b)
        if hidden_gain_w < 0.0 or hidden_gain_b < 0.0:
            raise ValueError("hidden_gain_w and hidden_gain_b must be nonnegative")
        if mean_init not in _MEAN_INIT_MODES:
            raise ValueError(
                f"Unknown mean_init {mean_init!r}; expected one of {list(_MEAN_INIT_MODES)}"
            )
        if (mean_init == "backbone") != (backbone_fc is not None):
            raise ValueError(
                "mean_init='backbone' requires backbone_fc, and backbone_fc is "
                "meaningless for any other mean_init"
            )
        if hidden_dims and mean_init == "backbone":
            raise ValueError(
                "mean_init='backbone' warm-starts the output layer from a "
                "num_classes x input_dim fc; with hidden_dims the output layer "
                "reads hidden activations instead, so the copy is undefined"
            )
        if feature_scale <= 0.0 or not math.isfinite(feature_scale):
            raise ValueError("feature_scale must be finite and positive")
        if v2bar_weight_var < 0.0 or v2bar_bias_var < 0.0:
            raise ValueError("TAGI-V prior variances must be nonnegative")

        if head == "remax":
            head = "remax_lognormal" if remax_approximation == "lognormal" else "remax_laplace"
        elif head == "remax_lognormal":
            remax_approximation, remax_jacobian = "lognormal", "diag"
        elif head == "remax_laplace_diag":
            remax_approximation, remax_jacobian = "laplace", "diag"
        valid = (
            _FIXED_NOISE_HEADS
            | _UNIT_PROBIT_HEADS
            | _PROBITREE_HEADS
            | _TAGIV_HEADS
            | _LOGIT_TARGET_HEADS
            | _CDF_REMAX_HEADS
        )
        if head not in valid:
            raise ValueError(
                f"Unknown classification head {head!r}; expected one of {sorted(valid)}"
            )
        if head in _UNIT_PROBIT_HEADS and sigma_v is not None:
            raise ValueError(
                "hrc_probit fixes the latent probit scale at one and does not accept sigma_v"
            )
        if head in _PROBITREE_HEADS and sigma_v is not None:
            raise ValueError("probitree uses probitree_r and does not accept sigma_v")
        if not math.isfinite(probitree_r) or probitree_r <= 0.0:
            raise ValueError("probitree_r must be finite and positive")
        if head not in _PROBITREE_HEADS and probitree_tree is not None:
            raise ValueError("probitree_tree is only valid for head='probitree'")
        if head not in _PROBITREE_HEADS and probitree_reference_variances is not None:
            raise ValueError("probitree_reference_variances is only valid for head='probitree'")
        if head in _PROBITREE_HEADS and mean_init == "backbone":
            raise ValueError("probitree does not infer a gate mapping from class-logit weights")
        if head in _TAGIV_HEADS | _LOGIT_TARGET_HEADS and sigma_v is not None:
            raise ValueError("TAGI-V heads learn observation variance and do not accept sigma_v")
        if head in _CDF_REMAX_HEADS and sigma_v is not None:
            raise ValueError(
                "CDF-Remax learns its observation variance and does not accept sigma_v"
            )
        if sigma_v is not None and sigma_v <= 0.0:
            raise ValueError("sigma_v must be positive")
        if probit_tau2 < 0.0 or not math.isfinite(probit_tau2):
            raise ValueError("probit_tau2 must be finite and nonnegative")
        if hrc_tree not in {"auto", "padded", "full"}:
            raise ValueError("hrc_tree must be 'auto', 'padded', or 'full'")
        if hrc_tree == "auto":
            # The proposed hierarchical-probit head is defined on the full
            # K-leaf tree. The padded tree stays reachable as an ablation.
            hrc_tree = "full" if head == "hrc_probit" else "padded"
        if hrc_log_tau is not None and not math.isfinite(hrc_log_tau):
            raise ValueError("hrc_log_tau must be finite")
        if logit_variance_floor < 0.0 or not math.isfinite(logit_variance_floor):
            raise ValueError("logit_variance_floor must be finite and nonnegative")
        if logit_aleatoric_init <= logit_variance_floor:
            raise ValueError("logit_aleatoric_init must exceed logit_variance_floor")
        if logit_variance_cv <= 0.0 or not math.isfinite(logit_variance_cv):
            raise ValueError("logit_variance_cv must be finite and positive")
        if not math.isfinite(logit_variance_weight_share) or not (
            0.0 <= logit_variance_weight_share < 1.0
        ):
            raise ValueError("logit_variance_weight_share must lie in [0, 1)")
        if logit_variance_weight_share > 0.0 and head in _LOGIT_TARGET_HEADS:
            if hidden_dims:
                raise ValueError(
                    "logit_variance_feature_energy is the mean energy of the "
                    "inputs the output layer sees; with hidden_dims those are "
                    "hidden activations whose energy is not known before "
                    "construction, so the heteroscedastic variance head needs "
                    "logit_variance_weight_share=0"
                )
            if logit_variance_feature_energy is None:
                raise ValueError(
                    "a heteroscedastic variance head needs "
                    "logit_variance_feature_energy; compute it with "
                    "logit_feature_energy on the features the head will see, or "
                    "set logit_variance_weight_share=0 for a homoscedastic head"
                )
            if (
                not math.isfinite(logit_variance_feature_energy)
                or logit_variance_feature_energy <= 0.0
            ):
                raise ValueError("logit_variance_feature_energy must be finite and positive")
        if logit_scale <= 0.0 or not math.isfinite(logit_scale):
            raise ValueError("logit_scale must be finite and positive")
        if logit_temperature <= 0.0 or not math.isfinite(logit_temperature):
            raise ValueError("logit_temperature must be finite and positive")
        if logit_alpha < 0.0 or not math.isfinite(logit_alpha):
            raise ValueError("logit_alpha must be finite and nonnegative")
        if logit_epistemic_scale < 0.0 or not math.isfinite(logit_epistemic_scale):
            raise ValueError("logit_epistemic_scale must be finite and nonnegative")
        if logit_num_samples < 2:
            raise ValueError("logit_num_samples must be at least two")
        if not math.isfinite(logit_variance_power) or not 0.0 < logit_variance_power <= 1.0:
            raise ValueError("logit_variance_power must lie in (0, 1]")
        if logit_variance_power < 1.0 and logit_variance_reference is None:
            raise ValueError(
                "compressing the learned variance needs logit_variance_reference, "
                "the anchor the power transform holds fixed"
            )
        if logit_variance_reference is not None and (
            not math.isfinite(logit_variance_reference) or logit_variance_reference <= 0.0
        ):
            raise ValueError("logit_variance_reference must be finite and positive")
        cdf_keywords: dict[str, Any] = {
            "cdf_epsilon": cdf_epsilon,
            "cdf_kappa": cdf_kappa,
            "cdf_aleatoric_init": cdf_aleatoric_init,
            "cdf_laplace_order": cdf_laplace_order,
            "cdf_hermite_order": cdf_hermite_order,
            "cdf_scale_order": cdf_scale_order,
            "cdf_train_hermite_order": cdf_train_hermite_order,
            "cdf_decomposition_samples": cdf_decomposition_samples,
        }
        if head not in _CDF_REMAX_HEADS:
            supplied = sorted(
                name for name, value in cdf_keywords.items() if value != _CDF_REMAX_DEFAULTS[name]
            )
            if supplied:
                raise ValueError(f"{', '.join(supplied)} is only valid for head='cdf_remax'")
        if not math.isfinite(cdf_epsilon) or cdf_epsilon <= 0.0:
            raise ValueError("cdf_epsilon must be finite and positive")
        if not math.isfinite(cdf_kappa) or cdf_kappa <= 0.0:
            raise ValueError("cdf_kappa must be finite and positive")
        cdf_orders = (
            cdf_laplace_order,
            cdf_hermite_order,
            cdf_scale_order,
            cdf_train_hermite_order,
        )
        if min(cdf_orders) < 4:
            raise ValueError("every cdf_* quadrature order must be at least four")
        if cdf_decomposition_samples is not None and cdf_decomposition_samples < 2:
            raise ValueError("cdf_decomposition_samples must be None or at least two")
        if not cdf_epsilon < cdf_aleatoric_init < cdf_epsilon + cdf_kappa:
            raise ValueError(
                "cdf_aleatoric_init must lie strictly inside the attainable range "
                f"({cdf_epsilon}, {cdf_epsilon + cdf_kappa}) of the CDF variance "
                f"head, got {cdf_aleatoric_init}"
            )

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.head = head
        self.device = torch.device(device)
        self.gain_w = float(gain_w)
        self.gain_b = float(gain_b)
        self.mean_init = mean_init
        if feature_mean is None:
            self.feature_mean = None
        else:
            resolved_feature_mean = torch.as_tensor(
                feature_mean, device=self.device, dtype=torch.get_default_dtype()
            ).reshape(-1)
            if resolved_feature_mean.numel() != input_dim:
                raise ValueError("feature_mean must contain one value per input feature")
            if not torch.isfinite(resolved_feature_mean).all():
                raise ValueError("feature_mean must be finite")
            self.feature_mean = resolved_feature_mean.reshape(1, input_dim)
        self.feature_scale = float(feature_scale)
        self.sigma_v = sigma_v
        self.probit_tau2 = float(probit_tau2)
        self.probitree_r = float(probitree_r)
        self.remax_approximation = remax_approximation
        self.remax_jacobian = remax_jacobian
        self.remax_num_quad = int(remax_num_quad)
        self.v2bar_init = float(v2bar_init)
        self.v2bar_weight_var = float(v2bar_weight_var)
        self.v2bar_bias_var = float(v2bar_bias_var)
        self.logit_variance_floor = float(logit_variance_floor)
        self.logit_aleatoric_init = float(logit_aleatoric_init)
        self.logit_variance_cv = float(logit_variance_cv)
        self.logit_variance_weight_share = float(logit_variance_weight_share)
        self.logit_variance_feature_energy = (
            None if logit_variance_feature_energy is None else float(logit_variance_feature_energy)
        )
        self.logit_scale = float(logit_scale)
        self.logit_temperature = float(logit_temperature)
        self.logit_alpha = float(logit_alpha)
        self.logit_epistemic_scale = float(logit_epistemic_scale)
        self.logit_num_samples = int(logit_num_samples)
        self.logit_seed = int(logit_seed)
        self.logit_variance_power = float(logit_variance_power)
        self.logit_variance_reference = (
            None if logit_variance_reference is None else float(logit_variance_reference)
        )
        self._logit_base_samples: Tensor | None = None
        self.cdf_epsilon = float(cdf_epsilon)
        self.cdf_kappa = float(cdf_kappa)
        self.cdf_aleatoric_init = float(cdf_aleatoric_init)
        self.cdf_laplace_order = int(cdf_laplace_order)
        self.cdf_hermite_order = int(cdf_hermite_order)
        self.cdf_scale_order = int(cdf_scale_order)
        self.cdf_train_hermite_order = int(cdf_train_hermite_order)
        self.cdf_decomposition_samples = (
            None if cdf_decomposition_samples is None else int(cdf_decomposition_samples)
        )
        # Joint-calibration state for the shared log-scale ``L ~ N(lambda, q)``,
        # which a calibration pass assigns to. Zero mean and zero variance is
        # the *uncalibrated* head: the scale is then the degenerate ``s = e^0 =
        # 1``, so nothing multiplies the deviation and predict reproduces the
        # model as trained.
        self.cdf_scale_mean: float = 0.0
        self.cdf_scale_variance: float = 0.0
        self.cdf_scale_posterior: LogScalePosterior | None = None
        # Set when the head is assembled; predict reads the variance head's own
        # Gaussian ``(nu, r)`` out of this layer's forward cache, because the
        # post-activation moments do not determine it.
        self.cdf_head: EvenProbit | None = None

        self.hrc_tree = hrc_tree
        self.hrc_prior_offsets = bool(hrc_prior_offsets)
        self.hrc: HierarchicalSoftmax | None = None
        if head in _HRC_HEADS:
            if hrc_tree == "full":
                if head == "hrc_tagiv":
                    raise ValueError("hrc_tagiv currently supports the padded tree only")
                self.hrc = class_to_obs_full(num_classes, use_prior_offsets=hrc_prior_offsets)
            else:
                self.hrc = class_to_obs(num_classes)
        if isinstance(probitree_tree, dict):
            probitree_tree = ProbiTree.from_dict(probitree_tree)
        if probitree_tree is not None and not isinstance(probitree_tree, ProbiTree):
            raise ValueError("probitree_tree must be a ProbiTree or its serialized mapping")
        self.probitree: ProbiTree | None = (
            ProbiTree(num_classes)
            if head == "probitree" and probitree_tree is None
            else probitree_tree
        )
        if self.probitree is not None and self.probitree.num_classes != num_classes:
            raise ValueError("ProbiTree class count does not match num_classes")
        # Prediction-time latent probit scale. The proposed probit head keeps
        # the unit scale that fixes the latent gauge and is never fitted; the
        # legacy Gaussian head inherits cuTAGI's alpha = 3. Overriding this, or
        # calling calibrate_hrc_log_tau, is an ablation against that model.
        default_log_tau = log_tau_from_alpha(3.0) if head == "hrc" else DEFAULT_LOG_TAU
        self.hrc_log_tau = default_log_tau if hrc_log_tau is None else float(hrc_log_tau)
        # Filled in by calibrate_hsm_log_gain; None means the structural unit
        # gain, which is the uncalibrated head.
        self.hsm_posterior: LogGainPosterior | None = None
        self.last_pre_update_loss: float | None = None
        base_dim = (
            self.probitree.num_gates
            if self.probitree is not None
            else self.hrc.len
            if self.hrc is not None
            else num_classes
        )
        interleaved = head in _TAGIV_HEADS | _LOGIT_TARGET_HEADS | _CDF_REMAX_HEADS
        output_dim = 2 * base_dim if interleaved else base_dim
        self.hidden_dims = hidden_dims
        self.hidden_gain_w = hidden_gain_w
        self.hidden_gain_b = hidden_gain_b
        self.hidden_linears: list[Linear] = []
        layers: list = []
        fan_in = input_dim
        for width in hidden_dims:
            hidden = Linear(
                fan_in,
                width,
                device=self.device,
                gain_w=hidden_gain_w,
                gain_b=hidden_gain_b,
            )
            self.hidden_linears.append(hidden)
            layers.extend([hidden, ReLU()])
            fan_in = width
        self.linear = Linear(
            fan_in,
            output_dim,
            device=self.device,
            gain_w=gain_w,
            gain_b=gain_b,
        )
        layers.append(self.linear)
        if head.startswith("remax_"):
            layers.append(
                Remax(
                    approximation=remax_approximation,
                    jacobian=remax_jacobian,
                    num_quad=remax_num_quad,
                )
            )
        elif head in _TAGIV_HEADS:
            layers.append(EvenSoftplus(base_dim))
            self._initialize_tagiv_prior()
        elif head in _LOGIT_TARGET_HEADS:
            layers.append(EvenExp(base_dim))
            self._initialize_logit_tagiv_prior()
        elif head in _CDF_REMAX_HEADS:
            # No Remax layer follows. Class probabilities are not a forward
            # activation for this head: they are the analytic Laplace-Remax
            # average over the prediction head, the variance head and the
            # shared log-scale, which predict evaluates from the moments.
            layers.append(EvenProbit(base_dim, epsilon=self.cdf_epsilon, kappa=self.cdf_kappa))
            self.cdf_head = layers[-1]
            self._initialize_cdf_remax_prior()
        self._apply_mean_init(backbone_fc)
        self.probitree_reference_variances: list[float] | None = None
        if self.probitree is not None:
            self.initialize_probitree_uniform(probitree_reference_variances)
        self.net = Sequential(layers, device=self.device)

    def initialize_probitree_uniform(
        self, reference_variances: Tensor | list[float] | None = None
    ) -> Tensor:
        """Initialize gate biases for uniform leaves at reference variances.

        When omitted, the bias-prior variances are the reference. This makes
        the zero-input predictive distribution uniform; callers should pass
        representative forward variances when input-dependent variances make a
        more useful reference available.
        """

        if self.probitree is None:
            raise ValueError("uniform ProbiTree initialization requires head='probitree'")
        assert self.linear.mb is not None and self.linear.Sb is not None
        if reference_variances is None:
            resolved = self.linear.Sb.detach().reshape(-1).double()
        else:
            resolved = torch.as_tensor(
                reference_variances, dtype=torch.float64, device=self.device
            ).reshape(-1)
        means = probitree_uniform_reference_means(self.probitree, resolved, self.probitree_r)
        with torch.no_grad():
            self.linear.mb.copy_(means.to(self.linear.mb.dtype).reshape(1, -1))
        self.probitree_reference_variances = resolved.detach().cpu().tolist()
        return means

    def _initialize_tagiv_prior(self) -> None:
        odd = slice(1, None, 2)
        with torch.no_grad():
            self.linear.mw[:, odd].zero_()
            self.linear.Sw[:, odd].fill_(self.v2bar_weight_var)
            assert self.linear.mb is not None and self.linear.Sb is not None
            self.linear.mb[:, odd].fill_(_softplus_inverse(self.v2bar_init))
            self.linear.Sb[:, odd].fill_(self.v2bar_bias_var)

    def _initialize_logit_tagiv_prior(self) -> None:
        """Center the log-variance stream and zero the logit stream.

        The logit stream regresses centered teacher logits, so a class-symmetric
        zero-mean prior is the correct one rather than one random classifier.
        The log-variance stream splits the ``v_G`` that
        :func:`logit_variance_head_prior` solved for between its weights and its
        bias, so the total prior variance is ``v_G`` at the average feature
        energy while the weights still carry enough prior mass for the head to
        be input dependent at all.
        """

        mu_g, v_g = logit_variance_head_prior(
            aleatoric_init=self.logit_aleatoric_init,
            variance_floor=self.logit_variance_floor,
            coefficient_of_variation=self.logit_variance_cv,
        )
        weight_variance, bias_variance = logit_variance_prior_split(
            v_g,
            feature_energy=(
                1.0
                if self.logit_variance_feature_energy is None
                else self.logit_variance_feature_energy
            ),
            weight_share=self.logit_variance_weight_share,
        )
        odd = slice(1, None, 2)
        with torch.no_grad():
            self.linear.mw.zero_()
            assert self.linear.mb is not None and self.linear.Sb is not None
            self.linear.mb.zero_()
            self.linear.Sw[:, odd].fill_(weight_variance)
            self.linear.mb[:, odd].fill_(mu_g)
            self.linear.Sb[:, odd].fill_(bias_variance)

    def _initialize_cdf_remax_prior(self) -> None:
        """Start the CDF variance head at ``h(nu) == cdf_aleatoric_init``.

        The odd stream carries the *pre-activation* head ``U_i``, and the
        activation ``h(u) = epsilon + kappa Phi(u)`` is invertible on its
        attainable range, so the requested initial aleatoric variance ``v`` is
        realised by the bias mean

            ``nu = Phi^{-1}((v - epsilon) / kappa)``,

        which the constructor has already checked lies in ``(epsilon, epsilon +
        kappa)``. The odd weight means are zeroed so that value holds at every
        input, and the shared TAGI-V prior variances ``v2bar_weight_var`` and
        ``v2bar_bias_var`` supply the head's prior uncertainty, exactly as in
        :meth:`_initialize_tagiv_prior`. The even prediction stream is left for
        :meth:`_apply_mean_init` to set.

        The bias is stored in the layer's own float32, so ``h(nu)`` recovers
        ``cdf_aleatoric_init`` to that precision rather than exactly; the
        forward mean of the odd slot is ``E[h(U)]``, which differs from
        ``h(nu)`` by the prior variance ``r`` and is not the quantity set here.
        """

        quantile = (self.cdf_aleatoric_init - self.cdf_epsilon) / self.cdf_kappa
        nu = float(torch.special.ndtri(torch.tensor(quantile, dtype=torch.float64)))
        odd = slice(1, None, 2)
        with torch.no_grad():
            self.linear.mw[:, odd].zero_()
            self.linear.Sw[:, odd].fill_(self.v2bar_weight_var)
            assert self.linear.mb is not None and self.linear.Sb is not None
            self.linear.mb[:, odd].fill_(nu)
            self.linear.Sb[:, odd].fill_(self.v2bar_bias_var)

    def _backbone_class_weights(self, backbone_fc: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Validate a backbone ``fc`` layer and map it onto transformed features.

        The head does not see raw representations: it sees
        ``(a - feature_mean) / feature_scale``. Reproducing the backbone's own
        function of the raw features therefore needs the compensating affine
        map ``w -> feature_scale * w``, ``b -> b + w @ feature_mean``. Both
        reduce to a direct copy under the identity transform this study uses.
        """

        weight, bias = backbone_fc
        if weight.dim() != 2 or tuple(weight.shape) != (self.num_classes, self.input_dim):
            raise ValueError(
                "backbone_fc weight must have shape (num_classes, input_dim), "
                f"got {tuple(weight.shape)}"
            )
        dtype, device = self.linear.mw.dtype, self.device
        resolved_weight = weight.to(device=device, dtype=dtype)
        if bias is None:
            resolved_bias = torch.zeros(self.num_classes, device=device, dtype=dtype)
        else:
            if tuple(bias.shape) != (self.num_classes,):
                raise ValueError(
                    f"backbone_fc bias must have shape (num_classes,), got {tuple(bias.shape)}"
                )
            resolved_bias = bias.to(device=device, dtype=dtype)
        if not torch.isfinite(resolved_weight).all() or not torch.isfinite(resolved_bias).all():
            raise ValueError("backbone_fc must be finite")
        if self.feature_mean is not None:
            resolved_bias = resolved_bias + resolved_weight @ self.feature_mean.reshape(-1)
        return resolved_weight * self.feature_scale, resolved_bias

    def _apply_mean_init(self, backbone_fc: tuple[Tensor, Tensor] | None) -> None:
        """Set the prior weight means according to ``mean_init``.

        Runs after the head-specific prior initializers, and writes only the
        latent stream: on the interleaved heads the odd channels carry a
        variance prior that those initializers own, and it is left untouched.

        The three arms are

        ``random``
            A no-op, leaving the He draw the constructor made. This is the
            status quo, so every existing configuration is unchanged. Note that
            ``logit_tagiv`` zeroes its own latent means in
            :meth:`_initialize_logit_tagiv_prior`, by design, so ``random`` and
            ``zero`` coincide for that head.
        ``zero``
            A class-symmetric prior that commits to no particular classifier.
        ``backbone``
            A warm start from the trained ``fc`` layer the features came from.
            ``logit_tagiv`` routes through
            :meth:`initialize_mean_from_teacher`, which applies the centering
            and ``logit_scale`` division its targets were built with. The HRC
            heads score tree nodes rather than classes and so need the
            branch-contrast projection of
            :func:`~triton_tagi.hrc_softmax.project_classes_to_nodes`; that
            projection is taken on class-centered weights, which fixes the
            latent gauge on the padded tree's single-branch nodes and is a
            no-op on the full tree. Every other head is a direct copy.

        Prior variances are never touched by this method: all three arms leave
        ``Sw`` and ``Sb`` exactly as the gain determined them.
        """

        if self.mean_init == "random":
            return
        interleaved = self.head in _TAGIV_HEADS | _LOGIT_TARGET_HEADS | _CDF_REMAX_HEADS
        latent = slice(0, None, 2) if interleaved else slice(None)
        if self.mean_init == "zero":
            with torch.no_grad():
                self.linear.mw[:, latent] = 0.0
                if self.linear.mb is not None:
                    self.linear.mb[:, latent] = 0.0
            return

        assert backbone_fc is not None  # guaranteed by __init__
        weight, bias = self._backbone_class_weights(backbone_fc)
        if self.head in _LOGIT_TARGET_HEADS:
            self.initialize_mean_from_teacher(weight, bias, scale=self.logit_scale)
            return
        if self.hrc is not None:
            weight, bias = project_classes_to_nodes(
                self.hrc, weight - weight.mean(dim=0, keepdim=True), bias - bias.mean()
            )
        with torch.no_grad():
            self.linear.mw[:, latent] = weight.T
            if self.linear.mb is not None:
                self.linear.mb[:, latent] = bias.reshape(1, -1)

    def _compress_variance(self, aleatoric: Tensor) -> Tensor:
        if self.logit_variance_power >= 1.0 or self.logit_variance_reference is None:
            return aleatoric
        return power_compress_variance(
            aleatoric,
            gamma=self.logit_variance_power,
            reference=self.logit_variance_reference,
            variance_floor=self.logit_variance_floor,
        )

    def _resolve_logit_base_samples(self, dtype: torch.dtype) -> Tensor:
        cached = self._logit_base_samples
        if cached is None or cached.dtype != dtype or cached.device != self.device:
            cached = standard_normal_base_samples(
                self.logit_num_samples,
                self.num_classes,
                seed=self.logit_seed,
                device=self.device,
                dtype=dtype,
            )
            self._logit_base_samples = cached
        return cached

    def _hsm_channel_sigma_v(self) -> float:
        """Return the fixed branch-channel noise the log-gain is measured against.

        The base HRC head observes ``+/-1`` at each node under the observation
        noise it was trained with, so that noise is the channel scale and
        ``lambda = 0`` is the head unchanged. The hierarchical-probit head fixes
        its latent unit instead, which is a channel noise of one.

        Raises:
            ValueError: If the head is not hierarchical, or if the base HRC head
                carries no observation noise to calibrate against.
        """

        if self.hrc is None or self.head not in {"hrc", "hrc_probit"}:
            raise ValueError("hierarchical calibration requires the hrc or hrc_probit head")
        if self.head == "hrc_probit":
            return 1.0
        if self.sigma_v is None or self.sigma_v <= 0.0:
            raise ValueError("the hrc head needs the positive sigma_v it was trained with")
        return float(self.sigma_v)

    def _resolve_sigma_v(self, sigma_v: float | None) -> float:
        resolved = self.sigma_v if sigma_v is None else sigma_v
        if resolved is None or resolved <= 0.0:
            raise ValueError(f"head {self.head!r} requires a positive sigma_v")
        self.sigma_v = float(resolved)
        return float(resolved)

    def _transform_representations(self, representations: Tensor) -> Tensor:
        if representations.shape[-1] != self.input_dim:
            raise ValueError("representations have the wrong feature dimension")
        transformed = representations.to(device=self.device, dtype=self.linear.mw.dtype)
        if self.feature_mean is not None:
            transformed = transformed - self.feature_mean
        return transformed / self.feature_scale

    def train_step(
        self,
        representations: Tensor,
        labels: Tensor | None = None,
        sigma_v: float | None = None,
        *,
        targets: Tensor | None = None,
    ) -> None:
        """Apply one closed-form TAGI update to the last layer.

        Args:
            representations: Frozen features, shape (B, input_dim).
            labels: Integer class labels, shape (B,). Required by every head
                except logit_tagiv, which ignores them.
            sigma_v: Observation noise standard deviation, for fixed-noise heads.
            targets: Continuous teacher logits, shape (B, num_classes). Required
                by logit_tagiv and rejected by every other head.
        """

        x = self._transform_representations(representations)
        if self.head in _LOGIT_TARGET_HEADS:
            if sigma_v is not None:
                raise ValueError(
                    "logit_tagiv learns its observation variance and does not accept sigma_v"
                )
            if targets is None:
                raise ValueError("logit_tagiv trains on teacher logits; pass targets")
            resolved_targets = targets.to(device=self.device, dtype=x.dtype)
            if resolved_targets.shape != (x.shape[0], self.num_classes):
                raise ValueError("targets must have shape (batch, num_classes)")
            self.net.step_logit_tagiv(x, resolved_targets, self.logit_variance_floor)
            return
        if targets is not None:
            raise ValueError(f"head {self.head!r} trains on class labels, not on logit targets")
        if labels is None:
            raise ValueError(f"head {self.head!r} requires class labels")
        labels = labels.to(self.device).long()
        if self.head == "categorical_tagiv":
            if sigma_v is not None:
                raise ValueError("TAGI-V heads do not accept sigma_v")
            self.net.step_categorical(x, labels, self.num_classes)
            return
        if self.head == "hrc_tagiv":
            if sigma_v is not None:
                raise ValueError("TAGI-V heads do not accept sigma_v")
            assert self.hrc is not None
            self.net.step_hrc_tagiv(x, labels, self.hrc)
            return

        if self.head == "probitree":
            if sigma_v is not None:
                raise ValueError("probitree uses probitree_r and does not accept sigma_v")
            assert self.probitree is not None
            _, _, log_evidence = self.net.step_probitree(
                x, labels, self.probitree, self.probitree_r
            )
            self.last_pre_update_loss = -float(log_evidence.detach().mean().cpu())
            return

        if self.head == "hrc_probit":
            if sigma_v is not None:
                raise ValueError(
                    "hrc_probit fixes the latent probit scale at one and does not accept sigma_v"
                )
            assert self.hrc is not None
            self.net.step_hrc_probit(x, labels, self.hrc)
            return

        if self.head in _CDF_REMAX_HEADS:
            if sigma_v is not None:
                raise ValueError(
                    "CDF-Remax learns its observation variance and does not accept sigma_v"
                )
            self.net.step_cdf_tagiv(
                x,
                labels,
                epsilon=self.cdf_epsilon,
                kappa=self.cdf_kappa,
                hermite_order=self.cdf_train_hermite_order,
            )
            return

        resolved_sigma = self._resolve_sigma_v(sigma_v)
        if self.head == "hrc":
            assert self.hrc is not None
            self.net.step_hrc(x, labels, self.hrc, resolved_sigma)
            return
        if self.head == "probit_ovr":
            targets = -torch.ones(
                labels.shape[0], self.num_classes, device=self.device, dtype=x.dtype
            )
            targets.scatter_(1, labels.unsqueeze(1), 1.0)
        else:
            targets = torch.zeros(
                labels.shape[0], self.num_classes, device=self.device, dtype=x.dtype
            )
            targets.scatter_(1, labels.unsqueeze(1), 1.0)
        self.net.step(x, targets, resolved_sigma)

    @torch.no_grad()
    def predict(
        self,
        representations: Tensor,
        *,
        sigma_v: float | None = None,
    ) -> ClassificationPrediction:
        """Return class probabilities and the head's native uncertainty moments."""

        self.net.eval()
        mean, variance = self.net.forward(self._transform_representations(representations))
        aleatoric = None
        epistemic = variance
        diagnostics: dict[str, Tensor] = {}

        if self.head == "probit_ovr":
            resolved_sigma = self._resolve_sigma_v(sigma_v)
            scale = torch.sqrt(variance.clamp_min(0.0) + resolved_sigma**2)
            probabilities = 0.5 * (
                1.0 + torch.erf(mean / scale.clamp_min(1e-12) / math.sqrt(2.0))
            ).clamp_min(1e-12)
        elif self.head == "probitree":
            if sigma_v is not None:
                raise ValueError("probitree uses probitree_r and does not accept sigma_v")
            assert self.probitree is not None
            log_probabilities = probitree_log_probs(
                self.probitree, mean, variance, self.probitree_r
            )
            diagnostics["class_log_probabilities"] = log_probabilities
            partition_deviation = torch.expm1(torch.logsumexp(log_probabilities, dim=-1)).abs()
            diagnostics["class_probability_sum_deviation"] = partition_deviation
            if bool((partition_deviation > 1e-8).any()):
                raise ArithmeticError("ProbiTree class probabilities do not sum to one")
            probabilities = log_probabilities.exp().to(mean.dtype)
        elif self.head in {"hrc", "hrc_probit"}:
            if self.head == "hrc_probit" and sigma_v is not None:
                raise ValueError(
                    "hrc_probit fixes the latent probit scale at one and does not accept sigma_v"
                )
            assert self.hrc is not None
            # Log-space path products. The full tree of the proposed head sums
            # to one on its own, so no normalizer is applied there; the padded
            # ablation discards leaves that hold probability mass and needs one.
            log_probabilities = hrc_log_probs(
                mean, variance, self.hrc, log_tau=self.hrc_log_tau, normalize=None
            )
            diagnostics["class_log_probabilities"] = log_probabilities
            probabilities = log_probabilities.exp().to(mean.dtype)
        elif self.head == "categorical_tagiv":
            probabilities = categorical_predictive_probs(mean, variance, self.num_classes)
            epistemic = variance[..., 0::2]
            aleatoric = mean[..., 1::2].clamp_min(0.0)
            diagnostics["aleatoric_moment_variance"] = variance[..., 1::2]
        elif self.head == "logit_tagiv":
            logit_mean, logit_variance, aleatoric_variance, moment_variance = (
                split_logit_tagiv_outputs(mean, variance, variance_floor=self.logit_variance_floor)
            )
            aleatoric_variance = self._compress_variance(aleatoric_variance)
            probabilities = logit_tagiv_predictive_probs(
                logit_mean,
                logit_variance,
                aleatoric_variance,
                temperature=self.logit_temperature,
                alpha=self.logit_alpha,
                epistemic_scale=self.logit_epistemic_scale,
                base_samples=self._resolve_logit_base_samples(logit_mean.dtype),
            )
            epistemic = logit_variance
            aleatoric = aleatoric_variance
            diagnostics["logit_mean"] = logit_mean
            diagnostics["aleatoric_moment_variance"] = moment_variance
        elif self.head in _CDF_REMAX_HEADS:
            if sigma_v is not None:
                raise ValueError(
                    "CDF-Remax learns its observation variance and does not accept sigma_v"
                )
            assert self.cdf_head is not None
            nu, r = self.cdf_head.nu, self.cdf_head.r
            assert nu is not None and r is not None
            # The even slots pass through EvenProbit untouched, so they are
            # still the prediction head's pre-activation moments; the variance
            # head's own Gaussian is only in the layer's cache, since the
            # saturating activation does not determine it from its output.
            mu_z, var_z = mean[..., 0::2], variance[..., 0::2]
            probabilities, _, cov_scale = remax_scale_moments(
                mu_z,
                var_z,
                nu,
                r,
                scale_mean=self.cdf_scale_mean,
                scale_variance=self.cdf_scale_variance,
                epsilon=self.cdf_epsilon,
                kappa=self.cdf_kappa,
                laplace_order=self.cdf_laplace_order,
                hermite_order=self.cdf_hermite_order,
                scale_order=self.cdf_scale_order,
                cross_moments=False,
            )
            probabilities = probabilities.to(mean.dtype)
            diagnostics["cov_scale"] = cov_scale
            # There is deliberately no cheap fallback for the split. Var(A) for
            # A = Remax(X_L) already contains noise realisations and so is not
            # the epistemic part, and the logit ratio e_i / (e_i + h_i) stops
            # being a valid split once the nonlinear normalisation has acted --
            # with every epistemic state fixed the epistemic covariance is zero
            # while both of those stay positive. A wrong-but-cheap number is
            # worse than none, so the fields stay None until the caller pays
            # for the nested QMC integration.
            epistemic = None
            if self.cdf_decomposition_samples is not None:
                cov_epistemic, cov_aleatoric = remax_uncertainty_decomposition(
                    mu_z,
                    var_z,
                    nu,
                    r,
                    scale_mean=self.cdf_scale_mean,
                    scale_variance=self.cdf_scale_variance,
                    epsilon=self.cdf_epsilon,
                    kappa=self.cdf_kappa,
                    num_samples=self.cdf_decomposition_samples,
                    laplace_order=self.cdf_laplace_order,
                )
                epistemic = torch.diagonal(cov_epistemic, dim1=-2, dim2=-1)
                aleatoric = torch.diagonal(cov_aleatoric, dim1=-2, dim2=-1)
        elif self.head == "hrc_tagiv":
            assert self.hrc is not None
            probabilities = obs_to_class_probs_tagiv(mean, variance, self.hrc)
            epistemic = variance[..., 0::2]
            aleatoric = mean[..., 1::2].clamp_min(0.0)
            diagnostics["aleatoric_moment_variance"] = variance[..., 1::2]
        else:
            probabilities = mean

        # The shared row normalization is float32 hygiene applied to every
        # head, not part of any head's model. The hierarchical-probit head's
        # full tree already sums to one; hrc_log_partition_deviation measures
        # that before this step touches it.
        return ClassificationPrediction(
            normalize_class_probabilities(probabilities),
            mean,
            variance,
            epistemic,
            aleatoric,
            diagnostics or None,
        )

    def fit(
        self,
        features: Tensor,
        labels: Tensor | None = None,
        *,
        targets: Tensor | None = None,
        epochs: int,
        batch_size: int = 256,
        sigma_v: float | None = None,
        seed: int = 0,
        validation: tuple[Tensor, Tensor] | None = None,
        validation_batch_size: int | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_epochs: set[int] | None = None,
        record_initial: bool = True,
        callback: Callable[[int, TAGILastLayerClassifier, dict[str, float]], None] | None = None,
    ) -> TrainingHistory:
        """Train on cached features with deterministic shuffling and optional validation.

        Args:
            features: Frozen features, shape (N, input_dim).
            labels: Integer class labels, shape (N,). Required by every head
                except logit_tagiv, which trains on ``targets`` instead.
            targets: Continuous teacher logits, shape (N, num_classes), required
                by logit_tagiv and rejected by every other head.
        """

        if features.dim() != 2 or features.shape[1] != self.input_dim:
            raise ValueError("features must have shape (samples, input_dim)")
        if self.head in _LOGIT_TARGET_HEADS:
            if targets is None:
                raise ValueError("logit_tagiv trains on teacher logits; pass targets")
            if targets.dim() != 2 or targets.shape != (
                features.shape[0],
                self.num_classes,
            ):
                raise ValueError("targets must have shape (samples, num_classes)")
        else:
            if targets is not None:
                raise ValueError(f"head {self.head!r} trains on class labels, not on logit targets")
            if labels is None:
                raise ValueError(f"head {self.head!r} requires class labels")
        if labels is not None and (labels.dim() != 1 or labels.shape[0] != features.shape[0]):
            raise ValueError("labels must have shape (samples,)")
        if epochs < 0 or batch_size < 1:
            raise ValueError("epochs must be nonnegative and batch_size must be positive")
        if validation_batch_size is not None and validation_batch_size < 1:
            raise ValueError("validation_batch_size must be positive")
        if self.head in _UNIT_PROBIT_HEADS and sigma_v is not None:
            raise ValueError(
                "hrc_probit fixes the latent probit scale at one and does not accept sigma_v"
            )
        if self.head == "probitree" and sigma_v is not None:
            raise ValueError("probitree uses probitree_r and does not accept sigma_v")
        if self.head in _TAGIV_HEADS | _LOGIT_TARGET_HEADS and sigma_v is not None:
            raise ValueError("TAGI-V heads do not accept sigma_v")
        if self.head in _FIXED_NOISE_HEADS:
            self._resolve_sigma_v(sigma_v)

        from .metrics import classification_metrics

        checkpoints = set() if checkpoint_epochs is None else set(checkpoint_epochs)
        directory = None if checkpoint_dir is None else Path(checkpoint_dir)
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator().manual_seed(seed)
        records: list[dict[str, float]] = []
        started = perf_counter()

        def record(epoch: int, train_nll: float | None = None) -> None:
            row: dict[str, float] = {"epoch": float(epoch), "wall_s": perf_counter() - started}
            if train_nll is not None:
                row["train_pre_update_nll"] = train_nll
            if validation is not None:
                if validation_batch_size is None:
                    probabilities = self.predict(
                        validation[0], sigma_v=self.sigma_v
                    ).probabilities.cpu()
                else:
                    probability_parts = [
                        self.predict(
                            validation[0][start : start + validation_batch_size],
                            sigma_v=self.sigma_v,
                        ).probabilities.cpu()
                        for start in range(0, validation[0].shape[0], validation_batch_size)
                    ]
                    probabilities = torch.cat(probability_parts)
                metrics = classification_metrics(probabilities, validation[1].cpu())
                row.update({f"val_{key}": value for key, value in metrics.items()})
            records.append(row)
            if callback is not None:
                callback(epoch, self, row)
            if directory is not None and epoch in checkpoints:
                self.save(directory / f"epoch_{epoch:04d}.pt", metadata={"epoch": epoch})

        if record_initial:
            record(0)
        for epoch in range(1, epochs + 1):
            probitree_losses: list[float] = []
            permutation = torch.randperm(features.shape[0], generator=generator)
            for start in range(0, features.shape[0], batch_size):
                indices = permutation[start : start + batch_size]
                self.train_step(
                    features[indices],
                    None if labels is None else labels[indices],
                    self.sigma_v,
                    targets=None if targets is None else targets[indices],
                )
                if self.head == "probitree":
                    assert self.last_pre_update_loss is not None
                    probitree_losses.append(self.last_pre_update_loss)
            train_nll = sum(probitree_losses) / len(probitree_losses) if probitree_losses else None
            record(epoch, train_nll)
        return TrainingHistory(tuple(records))

    def reset_variance_head(self) -> None:
        """Restore the log-variance stream's prior, keeping the fitted mean.

        Between a mean warm-up and a variance fit this puts the AGVI gain back
        at its prior value so the variance head learns from residuals around a
        converged mean rather than around an untrained one.
        """

        if self.head not in _LOGIT_TARGET_HEADS:
            raise ValueError(f"head {self.head!r} has no logit TAGI-V variance stream")
        mu_g, v_g = logit_variance_head_prior(
            aleatoric_init=self.logit_aleatoric_init,
            variance_floor=self.logit_variance_floor,
            coefficient_of_variation=self.logit_variance_cv,
        )
        weight_variance, bias_variance = logit_variance_prior_split(
            v_g,
            feature_energy=(
                1.0
                if self.logit_variance_feature_energy is None
                else self.logit_variance_feature_energy
            ),
            weight_share=self.logit_variance_weight_share,
        )
        odd = slice(1, None, 2)
        with torch.no_grad():
            self.linear.mw[:, odd].zero_()
            assert self.linear.mb is not None and self.linear.Sb is not None
            self.linear.Sw[:, odd].fill_(weight_variance)
            self.linear.mb[:, odd].fill_(mu_g)
            self.linear.Sb[:, odd].fill_(bias_variance)

    def initialize_mean_from_teacher(
        self,
        weight: Tensor,
        bias: Tensor | None = None,
        *,
        scale: float = 1.0,
        center: bool = True,
    ) -> None:
        """Set the latent stream to a teacher's classifier, centered and rescaled.

        The distillation target ``(l - mean_k l) / scale`` is exactly this affine
        map of the same features, so starting here makes the mean warm-up an
        assimilation rather than a re-derivation: the innovations begin near
        zero, the means stay put, and the pass serves only to contract the
        parameter covariance. That matters wherever relearning the teacher from
        a zero mean does not converge to it.

        Args:
            weight: Teacher classifier weight, shape (num_classes, input_dim) as
                :class:`torch.nn.Linear` stores it.
            bias: Teacher classifier bias, shape (num_classes,).
            scale: The normalization the targets were divided by.
            center: Subtract the class mean, matching centered targets.
        """

        if self.head not in _LOGIT_TARGET_HEADS:
            raise ValueError(f"head {self.head!r} has no logit TAGI-V mean stream")
        if weight.dim() != 2 or weight.shape != (self.num_classes, self.input_dim):
            raise ValueError("weight must have shape (num_classes, input_dim)")
        if bias is not None and bias.shape != (self.num_classes,):
            raise ValueError("bias must have shape (num_classes,)")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive")

        resolved_weight = weight.to(device=self.device, dtype=self.linear.mw.dtype)
        resolved_bias = (
            torch.zeros(self.num_classes, device=self.device, dtype=self.linear.mw.dtype)
            if bias is None
            else bias.to(device=self.device, dtype=self.linear.mw.dtype)
        )
        if center:
            resolved_weight = resolved_weight - resolved_weight.mean(dim=0, keepdim=True)
            resolved_bias = resolved_bias - resolved_bias.mean()
        with torch.no_grad():
            self.linear.mw[:, 0::2] = resolved_weight.T / scale
            assert self.linear.mb is not None
            self.linear.mb[:, 0::2] = resolved_bias.reshape(1, -1) / scale

    def _shuffled_batches(self, count: int, batch_size: int, seed: int):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(count, generator=generator)
        for start in range(0, count, batch_size):
            yield permutation[start : start + batch_size]

    def fit_mean(
        self,
        features: Tensor,
        targets: Tensor,
        *,
        epochs: int,
        batch_size: int = 256,
        observation_variance: float = 0.01,
        seed: int = 0,
    ) -> None:
        """Warm up the latent stream alone under a fixed observation variance.

        Args:
            features: Frozen features, shape (N, input_dim).
            targets: Logit targets, shape (N, num_classes); the mean over
                augmentation repeats when the head will later model their spread.
            epochs: Passes over the data.
            batch_size: Rows per update.
            observation_variance: The fixed ``sigma_v**2``.
            seed: Shuffling seed.
        """

        if self.head not in _LOGIT_TARGET_HEADS:
            raise ValueError(f"head {self.head!r} has no logit TAGI-V mean stream")
        if features.dim() != 2 or features.shape[1] != self.input_dim:
            raise ValueError("features must have shape (samples, input_dim)")
        if targets.shape != (features.shape[0], self.num_classes):
            raise ValueError("targets must have shape (samples, num_classes)")
        if epochs < 0:
            raise ValueError("epochs must be nonnegative")
        for epoch in range(epochs):
            for index in self._shuffled_batches(features.shape[0], batch_size, seed + epoch):
                x = self._transform_representations(features[index])
                self.net.step_logit_tagiv_mean(
                    x,
                    targets[index].to(device=self.device, dtype=x.dtype),
                    observation_variance,
                )

    def fit_variance(
        self,
        features: Tensor,
        *,
        targets: Tensor | None = None,
        residual_variance: Tensor | None = None,
        repeats: int | None = None,
        epochs: int,
        batch_size: int = 256,
        seed: int = 0,
        weight_shrinkage: float = 0.0,
    ) -> None:
        """Train the variance stream alone, with the fitted mean held fixed.

        Pass ``targets`` to run the single-observation AGVI update on residuals
        around the current mean, or ``residual_variance`` with ``repeats`` to run
        the replicate-aware update, whose observation of the noise never passes
        through the mean head and is therefore immune to mean-head error.

        Args:
            features: Frozen features, shape (N, input_dim).
            targets: Logit observations, shape (N, num_classes).
            residual_variance: Unbiased sample variance over repeats, shape
                (N, num_classes).
            repeats: The replicate count ``M`` behind ``residual_variance``.
            epochs: Passes over the data.
            batch_size: Rows per update.
            seed: Shuffling seed.
            weight_shrinkage: Precision of a zero-mean Gaussian prior reapplied
                to the variance head's weights after every batch. It bounds how
                much across-input spread the log-variance develops; the bias is
                left free so the overall level is unaffected.
        """

        if self.head not in _LOGIT_TARGET_HEADS:
            raise ValueError(f"head {self.head!r} has no logit TAGI-V variance stream")
        if (targets is None) == (residual_variance is None):
            raise ValueError("pass exactly one of targets or residual_variance")
        if features.dim() != 2 or features.shape[1] != self.input_dim:
            raise ValueError("features must have shape (samples, input_dim)")
        if epochs < 0:
            raise ValueError("epochs must be nonnegative")
        observations = targets if targets is not None else residual_variance
        assert observations is not None
        if observations.shape != (features.shape[0], self.num_classes):
            raise ValueError("observations must have shape (samples, num_classes)")
        if residual_variance is not None and (repeats is None or repeats < 2):
            raise ValueError("replicate-aware updates need repeats of at least two")

        for epoch in range(epochs):
            for index in self._shuffled_batches(features.shape[0], batch_size, seed + epoch):
                x = self._transform_representations(features[index])
                batch = observations[index].to(device=self.device, dtype=x.dtype)
                if targets is not None:
                    self.net.step_logit_tagiv(
                        x, batch, self.logit_variance_floor, update_mean=False
                    )
                else:
                    assert repeats is not None
                    self.net.step_logit_tagiv_replicates(
                        x, batch, repeats, self.logit_variance_floor
                    )
                if weight_shrinkage > 0.0:
                    with torch.no_grad():
                        odd = slice(1, None, 2)
                        gaussian_shrinkage_(
                            self.linear.mw[:, odd], self.linear.Sw[:, odd], weight_shrinkage
                        )

    @torch.no_grad()
    def logit_moments(
        self, features: Tensor, *, batch_size: int | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return the logit TAGI-V predictive moments over cached features.

        Args:
            features: Frozen features, shape (N, input_dim).
            batch_size: Rows per forward pass; ``None`` runs one pass.

        Returns:
            logit_mean: Latent logit means, shape (N, num_classes).
            epistemic_variance: Epistemic logit variances, shape (N, num_classes).
            aleatoric_variance: Learned observation variances ``s_min2 + E[exp G]``,
                shape (N, num_classes).
        """

        if self.head not in _LOGIT_TARGET_HEADS:
            raise ValueError(f"head {self.head!r} has no logit TAGI-V moments")
        if features.dim() != 2 or features.shape[1] != self.input_dim:
            raise ValueError("features must have shape (samples, input_dim)")
        self.net.eval()
        step = features.shape[0] if batch_size is None else batch_size
        if step < 1:
            raise ValueError("batch_size must be positive")
        means, epistemics, aleatorics = [], [], []
        for start in range(0, features.shape[0], step):
            mean, variance = self.net.forward(
                self._transform_representations(features[start : start + step])
            )
            logit_mean, logit_variance, aleatoric, _ = split_logit_tagiv_outputs(
                mean, variance, variance_floor=self.logit_variance_floor
            )
            means.append(logit_mean)
            epistemics.append(logit_variance)
            aleatorics.append(self._compress_variance(aleatoric))
        return torch.cat(means), torch.cat(epistemics), torch.cat(aleatorics)

    @torch.no_grad()
    def calibrate(
        self,
        features: Tensor,
        labels: Tensor,
        *,
        mode: str = "joint",
        alpha: float | None = None,
        iterations: int = 32,
        rounds: int = 2,
        batch_size: int | None = None,
    ) -> LogitCalibration:
        """Fit the post-hoc temperature and aleatoric multiplier, then store them.

        Args:
            features: Calibration features, shape (N, input_dim).
            labels: Calibration labels, shape (N,).
            mode: ``"temperature"`` fits ``T`` alone at the current or supplied
                ``alpha``; ``"joint"`` also selects ``alpha``, ``0`` included.
            alpha: Multiplier used by ``mode="temperature"`` and as the joint
                search's starting point; ``None`` keeps the current value.
            iterations: Golden-section iterations per one-dimensional search.
            rounds: Coordinate-descent rounds in joint mode.
            batch_size: Rows per forward pass when computing the moments.

        Returns:
            The selected :class:`~triton_tagi.logit_tagiv.LogitCalibration`,
            whose temperature and multiplier are also stored on the classifier.
        """

        logit_mean, epistemic, aleatoric = self.logit_moments(features, batch_size=batch_size)
        calibration = fit_logit_calibration(
            logit_mean,
            epistemic,
            aleatoric,
            labels.to(self.device).long(),
            mode=mode,
            alpha=self.logit_alpha if alpha is None else float(alpha),
            epistemic_scale=self.logit_epistemic_scale,
            iterations=iterations,
            rounds=rounds,
            base_samples=self._resolve_logit_base_samples(logit_mean.dtype),
        )
        self.logit_temperature = calibration.temperature
        self.logit_alpha = calibration.alpha
        return calibration

    @torch.no_grad()
    def hrc_node_moments(
        self,
        features: Tensor,
        *,
        batch_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return cached tree-node moments for an HRC head, shape (N, hrc.len)."""

        if self.hrc is None or self.head not in {"hrc", "hrc_probit"}:
            raise ValueError("hrc_node_moments requires the hrc or hrc_probit head")
        if features.dim() != 2 or features.shape[1] != self.input_dim:
            raise ValueError("features must have shape (samples, input_dim)")
        self.net.eval()
        rows = features.shape[0] if batch_size is None else max(1, int(batch_size))
        mean_parts, variance_parts = [], []
        for start in range(0, features.shape[0], rows):
            batch = self._transform_representations(features[start : start + rows])
            mean, variance = self.net.forward(batch)
            mean_parts.append(mean)
            variance_parts.append(variance)
        return torch.cat(mean_parts), torch.cat(variance_parts)

    def calibrate_hrc_log_tau(
        self,
        features: Tensor,
        labels: Tensor,
        *,
        log_tau_bounds: tuple[float, float] = (-5.0, 5.0),
        grid_size: int = 33,
        iterations: int = 60,
        batch_size: int | None = None,
    ) -> float:
        """Fit and store a prediction-time latent probit log-scale.

        Ablation only. The proposed hierarchical-probit head has no free
        scale: its unit latent noise is an identifiability convention, and
        calling this replaces that convention with a fitted quantity. Use it to
        measure what a post-hoc scale would buy, not as part of training.

        The network is frozen, so the scale is identified relative to the
        output scale it was trained at. Call this on a held-out calibration
        split and report on an untouched test split.

        Args:
            features: Calibration features, shape (N, input_dim).
            labels: Calibration labels, shape (N,).
            log_tau_bounds: Search interval for ``log_tau``.
            grid_size: Grid points used to bracket the minimum.
            iterations: Golden-section refinement steps.
            batch_size: Rows per forward pass when caching the node moments.

        Returns:
            The fitted ``log_tau``, also stored on the classifier.
        """

        assert self.hrc is not None
        mean, variance = self.hrc_node_moments(features, batch_size=batch_size)
        self.hrc_log_tau = fit_hrc_log_tau(
            mean,
            variance,
            labels.to(self.device).long(),
            self.hrc,
            log_tau_bounds=log_tau_bounds,
            grid_size=grid_size,
            iterations=iterations,
            normalize=True,
        )
        return self.hrc_log_tau

    def calibrate_hsm_log_gain(
        self,
        features: Tensor,
        labels: Tensor,
        *,
        sharing: str | GainGroups = "global",
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        method: str = "grid",
        batch_size: int | None = None,
    ) -> LogGainPosterior:
        """Fit a Gaussian belief over the positive branch gain on held-out data.

        This is the hierarchical probit calibration of Goulet, Nguyen and
        Florensa-Montilla. Unlike :meth:`calibrate_hrc_log_tau`, which replaces
        the structural unit scale by a fitted point value, it keeps the gain
        positive and retains its uncertainty, so prediction integrates the gain
        instead of plugging one in. Call it on a calibration split disjoint from
        the training data with the network frozen, and read the returned belief
        through :func:`triton_tagi.hsm_calibration.hsm_class_moments`.

        Both hierarchical heads are supported and each supplies its own fixed
        channel noise: the observation noise it trained with for the base HRC
        head, and one for the unit-latent probit head. ``prior_mean = 0`` is
        therefore the frozen head unchanged in either case.

        Args:
            features: Calibration features, shape (N, input_dim).
            labels: Calibration leaf labels, shape (N,).
            sharing: ``global``, ``level``, ``node``, or an explicit grouping.
            prior_mean: Prior mean of every group log-gain; zero is the
                uncalibrated head.
            prior_variance: Prior variance of every group log-gain.
            method: ``grid`` or ``laplace``.
            batch_size: Rows per forward pass when caching the node moments.

        Returns:
            The fitted belief, also stored as ``self.hsm_posterior``.
        """

        sigma_v = self._hsm_channel_sigma_v()
        assert self.hrc is not None
        mean, variance = self.hrc_node_moments(features, batch_size=batch_size)
        self.hsm_posterior = fit_hsm_log_gain(
            mean,
            variance,
            labels.to(self.device).long(),
            self.hrc,
            sharing=sharing,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            sigma_v=sigma_v,
            method=method,
            chunk_size=batch_size,
        )
        return self.hsm_posterior

    @torch.no_grad()
    def hsm_probabilities(
        self,
        features: Tensor,
        *,
        posterior: LogGainPosterior | None = None,
        order: int = DEFAULT_GAIN_ORDER,
        batch_size: int | None = None,
    ) -> Tensor:
        """Return calibrated class probabilities, shape (N, n_classes).

        With no fitted belief this is the frozen head read through the
        calibration machinery: for the base HRC head it reproduces
        :func:`triton_tagi.hrc_softmax.obs_to_class_probs` at
        ``alpha = 1 / sigma_v``, and for the probit head the unit-scale readout.

        Args:
            features: Input features, shape (N, input_dim).
            posterior: Log-gain belief; defaults to the stored calibration, or
                to the deterministic unit gain when none has been fitted. A
                belief fitted against a different channel noise is rejected
                rather than silently rescaled.
            order: Gauss--Hermite order for the log-gain integral.
            batch_size: Rows per block.

        Returns:
            Probabilities that sum to one per row on the proper K-leaf tree.
        """

        sigma_v = self._hsm_channel_sigma_v()
        assert self.hrc is not None
        belief = posterior if posterior is not None else self.hsm_posterior
        if belief is None:
            belief = LogGainPosterior.unit(self.hrc, sigma_v=sigma_v)
        if belief.sigma_v != sigma_v:
            raise ValueError(
                f"the belief was fitted at sigma_v={belief.sigma_v} and this head "
                f"reads out at sigma_v={sigma_v}"
            )
        mean, variance = self.hrc_node_moments(features, batch_size=batch_size)
        return hsm_class_probabilities(
            mean, variance, self.hrc, belief, order=order, chunk_size=batch_size
        )

    @torch.no_grad()
    def cdf_remax_moments(
        self, features: Tensor, *, batch_size: int | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the frozen forward summaries the CDF-Remax head calibrates on.

        The calibration channel sees the network only through these four
        tensors, which is what keeps it a separate inference channel: it
        updates the shared scale and never the prediction or variance head.

        Args:
            features: Frozen features, shape (N, input_dim).
            batch_size: Rows per forward pass; ``None`` runs one pass.

        Returns:
            mu_z: Prediction-head means, shape (N, num_classes).
            var_z: Prediction-head variances, same shape.
            nu: Variance-head means, same shape.
            r: Variance-head variances, same shape.
        """

        if self.head not in _CDF_REMAX_HEADS:
            raise ValueError(f"head {self.head!r} has no CDF-Remax forward summaries")
        if features.dim() != 2 or features.shape[1] != self.input_dim:
            raise ValueError("features must have shape (samples, input_dim)")
        assert self.cdf_head is not None
        self.net.eval()
        step = features.shape[0] if batch_size is None else batch_size
        if step < 1:
            raise ValueError("batch_size must be positive")

        means, variances, head_means, head_variances = [], [], [], []
        for start in range(0, features.shape[0], step):
            mean, variance = self.net.forward(
                self._transform_representations(features[start : start + step])
            )
            # Even slots pass through EvenProbit, so they are still the
            # prediction head's own moments; the variance head's Gaussian
            # survives only in the layer's cache.
            means.append(mean[..., 0::2].double())
            variances.append(variance[..., 0::2].double())
            head_means.append(self.cdf_head.nu.clone())
            head_variances.append(self.cdf_head.r.clone())
        return (
            torch.cat(means),
            torch.cat(variances),
            torch.cat(head_means),
            torch.cat(head_variances),
        )

    def calibrate_remax_log_scale(
        self,
        features: Tensor,
        labels: Tensor,
        *,
        prior_mean: float = 0.0,
        prior_variance: float = DEFAULT_PRIOR_VARIANCE,
        method: str = "grid",
        grid_size: int = 513,
        refinements: int = 2,
        initial_mean: float | None = None,
        process_variance: float = 0.0,
        batch_size: int | None = None,
        apply: bool = True,
    ) -> LogScalePosterior:
        """Fit the shared Remax deviation scale on a held-out calibration split.

        Since ``Remax(g x) = Remax(x)`` for every ``g > 0``, an ordinary global
        logit gain is exactly a no-op and ordinary temperature scaling cannot
        act here. What is fitted instead is the positive deviation scale
        ``s = e^L`` of

            ``X_{i,L} = mu_i + e^L [(Z_i - mu_i) + V_i]``,

        which rescales the combined epistemic and aleatoric deviation while
        holding the predictive logit mean fixed, so the learnable quantity is
        the mean relative to its uncertainty. ``prior_mean = 0`` is the frozen
        head unchanged.

        Call this with the network frozen on a split disjoint from the training
        data. Post-update training moments are not valid calibration inputs,
        because they have already assimilated the label.

        Two families of ``method`` are available and they answer different
        questions. ``grid`` and ``laplace`` fit the order-independent batch
        posterior of the note's objective and are the default, because the
        scale is scalar and a batch fit does not depend on the order the
        calibration labels happen to arrive in. ``event``, ``full`` and
        ``tilt`` run the sequential assumed-density recursion instead, one
        observed-class update per sample; they reproduce the source note's
        algorithm and are ordering-sensitive.

        ``initial_mean`` and ``process_variance`` apply to the sequential
        methods only. Passing ``initial_mean`` retains a mean across epochs
        while the variance resets to ``prior_variance``, which is the note's
        epoch-wise tracking heuristic and not exact Bayes: once earlier labels
        have moved the retained mean, resetting only the variance does not make
        replaying them a new independent likelihood. ``process_variance`` is
        the stated random-walk alternative for genuinely fresh observations.

        Args:
            features: Calibration features, shape (N, input_dim).
            labels: Calibration class labels, shape (N,).
            prior_mean: Prior mean of the log-scale; zero is no rescaling.
            prior_variance: Prior variance of the log-scale.
            method: One of ``grid``, ``laplace``, ``event``, ``full``, ``tilt``.
            grid_size: Grid points per sweep for the batch methods. The scale
                is scalar and its posterior smooth, so a coarse grid with
                refinements matches a fine one: 65 points over three sweeps
                reproduces the 513-point answer to ``1e-15`` at an eighth of
                the cost, which matters once the class count is large.
            refinements: Refinement sweeps for the batch methods.
            initial_mean: Retained mean for the sequential methods.
            process_variance: Random-walk increment for the sequential methods.
            batch_size: Rows per forward pass, and the fit's chunk size.
            apply: Store the result on the classifier so that
                :meth:`predict` integrates it.

        Returns:
            The fitted belief, also stored as ``self.cdf_scale_posterior`` when
            ``apply`` is set.
        """

        if method not in set(REMAX_SCALE_FIT_METHODS) | set(REMAX_SCALE_METHODS):
            expected = sorted(set(REMAX_SCALE_FIT_METHODS) | set(REMAX_SCALE_METHODS))
            raise ValueError(f"Unknown method {method!r}; expected one of {expected}")
        sequential = method in REMAX_SCALE_METHODS
        if sequential and (grid_size != 513 or refinements != 2):
            raise ValueError(
                "grid_size and refinements apply only to the batch methods "
                f"{list(REMAX_SCALE_FIT_METHODS)}"
            )
        if not sequential and (initial_mean is not None or process_variance != 0.0):
            raise ValueError(
                "initial_mean and process_variance apply only to the sequential "
                f"methods {list(REMAX_SCALE_METHODS)}"
            )

        mu_z, var_z, nu, r = self.cdf_remax_moments(features, batch_size=batch_size)
        targets = labels.to(self.device).long()
        shared = {
            "epsilon": self.cdf_epsilon,
            "kappa": self.cdf_kappa,
            "prior_mean": prior_mean,
            "prior_variance": prior_variance,
            "laplace_order": self.cdf_laplace_order,
            "hermite_order": self.cdf_hermite_order,
            "scale_order": self.cdf_scale_order,
        }
        if sequential:
            posterior = calibrate_remax_log_scale_adf(
                mu_z,
                var_z,
                nu,
                r,
                targets,
                method=method,
                initial_mean=initial_mean,
                process_variance=process_variance,
                **shared,
            )
        else:
            posterior = fit_remax_log_scale(
                mu_z,
                var_z,
                nu,
                r,
                targets,
                method=method,
                grid_size=grid_size,
                refinements=refinements,
                chunk_size=batch_size,
                **shared,
            )

        if apply:
            self.cdf_scale_posterior = posterior
            self.cdf_scale_mean = float(posterior.mean)
            self.cdf_scale_variance = float(posterior.variance)
        return posterior

    def config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "head": self.head,
            "gain_w": self.gain_w,
            "gain_b": self.gain_b,
            "hidden_dims": list(self.hidden_dims),
            "hidden_gain_w": self.hidden_gain_w,
            "hidden_gain_b": self.hidden_gain_b,
            "mean_init": self.mean_init,
            "feature_mean": (
                None
                if self.feature_mean is None
                else self.feature_mean.squeeze(0).detach().cpu().tolist()
            ),
            "feature_scale": self.feature_scale,
            "sigma_v": self.sigma_v,
            "probit_tau2": self.probit_tau2,
            "probitree_r": self.probitree_r,
            "probitree_tree": (None if self.probitree is None else self.probitree.to_dict()),
            "probitree_reference_variances": self.probitree_reference_variances,
            "hrc_tree": self.hrc_tree,
            "hrc_prior_offsets": self.hrc_prior_offsets,
            "hrc_log_tau": self.hrc_log_tau,
            "remax_approximation": self.remax_approximation,
            "remax_jacobian": self.remax_jacobian,
            "remax_num_quad": self.remax_num_quad,
            "v2bar_init": self.v2bar_init,
            "v2bar_weight_var": self.v2bar_weight_var,
            "v2bar_bias_var": self.v2bar_bias_var,
            "logit_variance_floor": self.logit_variance_floor,
            "logit_aleatoric_init": self.logit_aleatoric_init,
            "logit_variance_cv": self.logit_variance_cv,
            "logit_variance_weight_share": self.logit_variance_weight_share,
            "logit_variance_feature_energy": self.logit_variance_feature_energy,
            "logit_scale": self.logit_scale,
            "logit_temperature": self.logit_temperature,
            "logit_alpha": self.logit_alpha,
            "logit_epistemic_scale": self.logit_epistemic_scale,
            "logit_num_samples": self.logit_num_samples,
            "logit_seed": self.logit_seed,
            "logit_variance_power": self.logit_variance_power,
            "logit_variance_reference": self.logit_variance_reference,
            "cdf_epsilon": self.cdf_epsilon,
            "cdf_kappa": self.cdf_kappa,
            "cdf_aleatoric_init": self.cdf_aleatoric_init,
            "cdf_laplace_order": self.cdf_laplace_order,
            "cdf_hermite_order": self.cdf_hermite_order,
            "cdf_scale_order": self.cdf_scale_order,
            "cdf_train_hermite_order": self.cdf_train_hermite_order,
            "cdf_decomposition_samples": self.cdf_decomposition_samples,
            "cdf_scale_mean": self.cdf_scale_mean,
            "cdf_scale_variance": self.cdf_scale_variance,
        }

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        """Save a self-contained last-layer checkpoint."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = {
            name: getattr(self.linear, name).detach().cpu().clone()
            for name in ("mw", "Sw", "mb", "Sb")
            if getattr(self.linear, name) is not None
        }
        hidden_state = [
            {
                name: getattr(layer, name).detach().cpu().clone()
                for name in ("mw", "Sw", "mb", "Sb")
                if getattr(layer, name) is not None
            }
            for layer in self.hidden_linears
        ]
        torch.save(
            {
                "config": self.config(),
                "state": state,
                "hidden_state": hidden_state,
                "metadata": metadata or {},
            },
            destination,
        )
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cuda",
    ) -> tuple[TAGILastLayerClassifier, dict[str, Any]]:
        """Restore a classifier and return it with user metadata."""

        payload = torch.load(path, map_location=device, weights_only=False)
        config = dict(payload["config"])
        if config.get("head") == "hrc_probit":
            saved_scale = config.get("sigma_v")
            if saved_scale is not None and not math.isclose(float(saved_scale), 1.0):
                raise ValueError("hrc_probit checkpoint used a non-unit latent probit scale")
            # Migrate checkpoints from the initial implementation, which
            # serialized the structural unit scale as if it were tunable.
            config["sigma_v"] = None
        # mean_init only ever shaped the prior means, which the saved state is
        # about to overwrite, and the backbone tensors it needed are not part
        # of the checkpoint. Rebuild on the no-op arm and restore the recorded
        # value, so config() still reports how the run was initialized.
        saved_mean_init = config.pop("mean_init", "random")
        # The CDF-Remax log-scale is fitted after training rather than
        # configured, so like mean_init it is restored by assignment. A
        # checkpoint written before the head existed simply has no scale, and
        # falls back to the uncalibrated s = 1.
        saved_scale_mean = config.pop("cdf_scale_mean", 0.0)
        saved_scale_variance = config.pop("cdf_scale_variance", 0.0)
        classifier = cls(**config, device=device)
        for name, value in payload["state"].items():
            getattr(classifier.linear, name).copy_(value.to(classifier.device))
        hidden_state = payload.get("hidden_state", [])
        if len(hidden_state) != len(classifier.hidden_linears):
            raise ValueError(
                f"checkpoint carries {len(hidden_state)} hidden layers but the "
                f"config declares {len(classifier.hidden_linears)}"
            )
        for layer, tensors in zip(classifier.hidden_linears, hidden_state):
            for name, value in tensors.items():
                getattr(layer, name).copy_(value.to(classifier.device))
        classifier.mean_init = saved_mean_init
        classifier.cdf_scale_mean = float(saved_scale_mean)
        classifier.cdf_scale_variance = float(saved_scale_variance)
        return classifier, dict(payload.get("metadata", {}))
