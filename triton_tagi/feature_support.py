"""Bayesian feature-support evidence for open-set classification.

The closed-set TAGI classifier answers which known class generated an input.
This module supplies the separate model-comparison question required for OOD
detection: whether a frozen representation is better explained by the known
class mixture or by a class-agnostic background distribution.

Both hypotheses use objective-Bayes posterior predictive Student-t densities.
The known-class hypothesis retains one full covariance per class.  The
background is isotropic, deliberately discarding class and directional
structure; it is the rotation-invariant maximum-entropy alternative determined
by the global mean and residual energy.  No OOD examples, thresholds,
bandwidths, covariance shrinkage, or learned mixing coefficients are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def _validate_features(features: Tensor) -> None:
    if features.dim() != 2 or not features.is_floating_point():
        raise ValueError("features must be a floating-point N x D tensor")
    if not features.shape[0] or not features.shape[1]:
        raise ValueError("features must be non-empty")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("features must be finite")


def _stable_cholesky(matrix: Tensor) -> Tensor:
    """Cholesky factor with scale-aware roundoff protection only."""

    factor, info = torch.linalg.cholesky_ex(matrix)
    if not bool(info.any()):
        return factor
    dimension = matrix.shape[-1]
    scale = matrix.diagonal(dim1=-2, dim2=-1).mean().clamp_min(
        torch.finfo(matrix.dtype).tiny
    )
    identity = torch.eye(dimension, dtype=matrix.dtype, device=matrix.device)
    jitter = torch.finfo(matrix.dtype).eps * dimension * scale
    for _ in range(8):
        factor, info = torch.linalg.cholesky_ex(matrix + jitter * identity)
        if not bool(info.any()):
            return factor
        jitter = jitter * 10.0
    raise ValueError("feature scatter is singular beyond numerical roundoff")


@dataclass(frozen=True)
class FeatureSupportPrediction:
    """Domain evidence and the resulting open-set categorical distribution."""

    log_id_evidence: Tensor
    log_background_evidence: Tensor
    log_bayes_factor: Tensor
    id_probability: Tensor
    ood_probability: Tensor
    probabilities: Tensor | None = None


class BayesianFeatureSupportGate:
    """Objective-Bayes support model for a fixed feature representation.

    The per-class posterior predictive follows from the independence Jeffreys
    prior ``p(mu, Sigma) proportional to |Sigma|^(-(D+1)/2)``.  It is defined
    when every class contains more than ``D`` observations.  Class weights use
    the observed frequencies.  The background predictive integrates an
    unknown mean and shared isotropic variance under ``p(mu, sigma2) ∝ 1/sigma2``.

    Equal prior odds are used for the ID/background model comparison.  The log
    Bayes factor is retained because probabilities can legitimately saturate
    when hundreds of feature dimensions contribute evidence.
    """

    def __init__(
        self,
        *,
        classes: Tensor,
        class_means: Tensor,
        class_scale_cholesky: Tensor,
        class_df: Tensor,
        class_log_normalizer: Tensor,
        background_mean: Tensor,
        background_scale: Tensor,
        background_df: Tensor,
        background_log_normalizer: Tensor,
    ) -> None:
        self.classes = classes
        self.class_means = class_means
        self.class_scale_cholesky = class_scale_cholesky
        self.class_df = class_df
        self.class_log_normalizer = class_log_normalizer
        self.background_mean = background_mean
        self.background_scale = background_scale
        self.background_df = background_df
        self.background_log_normalizer = background_log_normalizer

    @property
    def device(self) -> torch.device:
        return self.class_means.device

    @property
    def dtype(self) -> torch.dtype:
        return self.class_means.dtype

    @property
    def num_features(self) -> int:
        return self.class_means.shape[1]

    @property
    def num_classes(self) -> int:
        return self.class_means.shape[0]

    @classmethod
    @torch.no_grad()
    def fit(
        cls,
        features: Tensor,
        labels: Tensor,
        *,
        device: str | torch.device | None = None,
    ) -> "BayesianFeatureSupportGate":
        """Fit posterior predictives using only labeled ID features."""

        _validate_features(features)
        labels = labels.reshape(-1)
        if labels.numel() != features.shape[0]:
            raise ValueError("labels must contain one entry per feature row")
        if labels.dtype == torch.bool or labels.is_floating_point():
            raise ValueError("labels must be integer class indices")
        target_device = torch.device(device) if device is not None else features.device
        work = features.to(device=target_device, dtype=torch.float64)
        labels = labels.to(device=target_device, dtype=torch.long)
        classes = labels.unique(sorted=True)
        if classes.numel() < 2:
            raise ValueError("support modeling requires at least two classes")

        sample_count, dimension = work.shape
        means = []
        factors = []
        degrees = []
        normalizers = []
        for class_index in classes:
            class_features = work[labels == class_index]
            count = class_features.shape[0]
            if count <= dimension:
                raise ValueError(
                    "objective-Bayes full-covariance prediction requires more "
                    f"samples than dimensions per class; got {count} <= {dimension}"
                )
            mean = class_features.mean(dim=0)
            centered = class_features - mean
            scatter = centered.mT @ centered
            df = count - dimension
            predictive_scale = scatter * ((count + 1.0) / (count * df))
            factor = _stable_cholesky(predictive_scale)
            log_determinant = 2.0 * factor.diagonal().log().sum()
            log_prior = math.log(count / sample_count)
            log_normalizer = (
                torch.lgamma(work.new_tensor((df + dimension) / 2.0))
                - torch.lgamma(work.new_tensor(df / 2.0))
                - 0.5
                * (
                    dimension * math.log(df * math.pi)
                    + log_determinant
                )
                + log_prior
            )
            means.append(mean)
            factors.append(factor)
            degrees.append(work.new_tensor(float(df)))
            normalizers.append(log_normalizer)

        background_mean = work.mean(dim=0)
        residual = work - background_mean
        residual_energy = residual.square().sum()
        background_df = work.new_tensor(float((sample_count - 1) * dimension))
        background_scale = residual_energy * (
            (sample_count + 1.0)
            / (sample_count * (sample_count - 1.0) * dimension)
        )
        background_log_normalizer = (
            torch.lgamma((background_df + dimension) / 2.0)
            - torch.lgamma(background_df / 2.0)
            - 0.5
            * dimension
            * torch.log(background_df * math.pi * background_scale)
        )
        return cls(
            classes=classes,
            class_means=torch.stack(means),
            class_scale_cholesky=torch.stack(factors),
            class_df=torch.stack(degrees),
            class_log_normalizer=torch.stack(normalizers),
            background_mean=background_mean,
            background_scale=background_scale,
            background_df=background_df,
            background_log_normalizer=background_log_normalizer,
        )

    def to(self, device: str | torch.device) -> "BayesianFeatureSupportGate":
        """Return a copy on ``device`` while preserving double precision."""

        state = {
            key: value.to(device)
            for key, value in self.state_dict().items()
        }
        return self.from_state_dict(state)

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "classes": self.classes,
            "class_means": self.class_means,
            "class_scale_cholesky": self.class_scale_cholesky,
            "class_df": self.class_df,
            "class_log_normalizer": self.class_log_normalizer,
            "background_mean": self.background_mean,
            "background_scale": self.background_scale,
            "background_df": self.background_df,
            "background_log_normalizer": self.background_log_normalizer,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor]) -> "BayesianFeatureSupportGate":
        required = {
            "classes",
            "class_means",
            "class_scale_cholesky",
            "class_df",
            "class_log_normalizer",
            "background_mean",
            "background_scale",
            "background_df",
            "background_log_normalizer",
        }
        missing = required - set(state)
        if missing:
            raise ValueError(f"support-gate state is missing {sorted(missing)}")
        return cls(**{key: state[key] for key in required})

    @torch.no_grad()
    def log_evidence(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Return known-class-mixture and background log evidence."""

        _validate_features(features)
        if features.shape[1] != self.num_features:
            raise ValueError(
                f"expected {self.num_features} features, got {features.shape[1]}"
            )
        work = features.to(device=self.device, dtype=self.dtype)
        dimension = self.num_features
        class_parts = []
        for index in range(self.num_classes):
            residual = work - self.class_means[index]
            whitened = torch.linalg.solve_triangular(
                self.class_scale_cholesky[index],
                residual.mT,
                upper=False,
            ).mT
            quadratic = whitened.square().sum(dim=1)
            df = self.class_df[index]
            class_parts.append(
                self.class_log_normalizer[index]
                - 0.5 * (df + dimension) * torch.log1p(quadratic / df)
            )
        log_id = torch.logsumexp(torch.stack(class_parts, dim=1), dim=1)
        background_quadratic = (
            (work - self.background_mean).square().sum(dim=1)
            / self.background_scale
        )
        log_background = self.background_log_normalizer - 0.5 * (
            self.background_df + dimension
        ) * torch.log1p(background_quadratic / self.background_df)
        return log_id, log_background

    @torch.no_grad()
    def predict(
        self,
        features: Tensor,
        conditional_probabilities: Tensor | None = None,
    ) -> FeatureSupportPrediction:
        """Compute equal-prior domain odds and optional K+1 probabilities."""

        log_id, log_background = self.log_evidence(features)
        domain_log_probabilities = torch.log_softmax(
            torch.stack((log_id, log_background), dim=1), dim=1
        )
        id_probability = domain_log_probabilities[:, 0].exp()
        ood_probability = domain_log_probabilities[:, 1].exp()
        probabilities = None
        if conditional_probabilities is not None:
            if conditional_probabilities.shape != (features.shape[0], self.num_classes):
                raise ValueError(
                    "conditional probabilities must have shape N x num_classes"
                )
            conditional = conditional_probabilities.to(
                device=self.device, dtype=self.dtype
            )
            if bool((conditional < 0).any()) or not torch.allclose(
                conditional.sum(dim=1),
                torch.ones(features.shape[0], device=self.device, dtype=self.dtype),
                atol=1e-6,
                rtol=1e-6,
            ):
                raise ValueError("conditional probabilities must be normalized")
            probabilities = torch.cat(
                (conditional * id_probability[:, None], ood_probability[:, None]),
                dim=1,
            )
        return FeatureSupportPrediction(
            log_id_evidence=log_id,
            log_background_evidence=log_background,
            log_bayes_factor=log_id - log_background,
            id_probability=id_probability,
            ood_probability=ood_probability,
            probabilities=probabilities,
        )
