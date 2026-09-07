"""Directional full-feature-covariance TAGI last layer.

This experimental classifier keeps one dense Gaussian posterior over the
feature-to-utility weights of each class. Multinomial-probit ADF supplies a
Gaussian marginal site for every class event, and those sites are incorporated
in parameter natural form. Further epochs use block-EP site replacement rather
than consuming the same evidence repeatedly.

Cross-class parameter covariance is projected away, but feature-direction
covariance is retained exactly within every class.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from .multinomial_probit import (
    multinomial_probit_adf_event,
    multinomial_probit_adf_predictive_probs,
)
from .param_init import init_weight_bias_linear

_EPS = 1e-10


@dataclass(frozen=True)
class FullCovarianceADFPrediction:
    probabilities: Tensor
    output_mean: Tensor
    output_variance: Tensor
    contrast_logdet: Tensor


def _contrast_basis(
    num_classes: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    difference = torch.eye(num_classes, device=device, dtype=dtype)[:, :-1]
    difference[-1, :] = -1.0
    basis, _ = torch.linalg.qr(difference, mode="reduced")
    return basis


def contrast_covariance_logdet(variance: Tensor, eps: float = _EPS) -> Tensor:
    """Log determinant of diagonal output covariance in class-difference space."""

    if variance.dim() != 2 or not variance.is_floating_point():
        raise ValueError("variance must have floating-point shape N x K")
    if bool((variance < 0).any()) or not bool(torch.isfinite(variance).all()):
        raise ValueError("variance must be finite and nonnegative")
    num_classes = variance.shape[1]
    if num_classes < 2:
        raise ValueError("at least two classes are required")
    basis = _contrast_basis(
        num_classes, device=variance.device, dtype=variance.dtype
    )
    projected = torch.einsum("ki,bk,kj->bij", basis, variance, basis)
    identity = torch.eye(
        num_classes - 1, device=variance.device, dtype=variance.dtype
    )
    _, logdet = torch.linalg.slogdet(projected + eps * identity)
    return logdet


@torch.no_grad()
def multinomial_probit_epistemic_mutual_information(
    output_mean: Tensor,
    output_variance: Tensor,
    *,
    probit_tau2: float = 1.0,
    num_samples: int = 32,
) -> Tensor:
    """Deterministic Sobol estimate of probit BALD epistemic information."""

    if output_mean.shape != output_variance.shape or output_mean.dim() != 2:
        raise ValueError("output moments must have matching N x K shapes")
    if num_samples < 2:
        raise ValueError("num_samples must be at least two")
    if probit_tau2 <= 0.0 or not math.isfinite(probit_tau2):
        raise ValueError("BALD requires finite positive probit_tau2")
    probabilities = multinomial_probit_adf_predictive_probs(
        output_mean, output_variance, probit_tau2=probit_tau2
    )
    total_entropy = -(
        probabilities * probabilities.clamp_min(1e-30).log()
    ).sum(dim=1)

    sobol = torch.quasirandom.SobolEngine(
        dimension=output_mean.shape[1], scramble=False
    )
    uniform = sobol.draw(num_samples).to(
        device=output_mean.device, dtype=output_mean.dtype
    )
    tiny = torch.finfo(output_mean.dtype).eps
    standard_normal = math.sqrt(2.0) * torch.erfinv(
        2.0 * uniform.clamp(tiny, 1.0 - tiny) - 1.0
    )
    latent = output_mean[:, None, :] + output_variance.clamp_min(0.0).sqrt()[
        :, None, :
    ] * standard_normal[None, :, :]
    conditional = multinomial_probit_adf_predictive_probs(
        latent.reshape(-1, output_mean.shape[1]),
        torch.zeros_like(latent).reshape(-1, output_mean.shape[1]),
        probit_tau2=probit_tau2,
    ).reshape(output_mean.shape[0], num_samples, output_mean.shape[1])
    conditional_entropy = -(
        conditional * conditional.clamp_min(1e-30).log()
    ).sum(dim=2).mean(dim=1)
    return (total_entropy - conditional_entropy).clamp_min(0.0)


class FullCovarianceADFClassifier:
    """Multinomial-probit ADF with full feature covariance per class."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        device: str | torch.device = "cuda",
        gain_w: float = 0.1,
        gain_b: float = 0.1,
        probit_tau2: float = 1.0,
    ) -> None:
        if input_dim < 1 or num_classes < 2:
            raise ValueError("input_dim must be positive and num_classes at least two")
        if gain_w <= 0.0 or gain_b <= 0.0:
            raise ValueError("full-covariance prior gains must be positive")
        if probit_tau2 < 0.0 or not math.isfinite(probit_tau2):
            raise ValueError("probit_tau2 must be finite and nonnegative")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.parameter_dim = input_dim + 1
        self.device = torch.device(device)
        self.probit_tau2 = float(probit_tau2)
        mean_w, variance_w, mean_b, variance_b = init_weight_bias_linear(
            input_dim,
            num_classes,
            gain_w=gain_w,
            gain_b=gain_b,
            device=self.device,
        )
        parameter_mean = torch.cat((mean_w, mean_b), dim=0).mT.double()
        parameter_variance = torch.cat((variance_w, variance_b), dim=0).mT.double()
        self.prior_precision = torch.diag_embed(parameter_variance.reciprocal())
        self.prior_natural_mean = torch.bmm(
            self.prior_precision, parameter_mean.unsqueeze(-1)
        ).squeeze(-1)
        self.precision = self.prior_precision.clone()
        self.natural_mean = self.prior_natural_mean.clone()
        self.parameter_mean = parameter_mean
        self.precision_cholesky = torch.linalg.cholesky(self.precision)
        self.samples_seen = 0

    def _augment(self, features: Tensor) -> Tensor:
        if features.dim() != 2 or features.shape[1] != self.input_dim:
            raise ValueError("features must have shape N x input_dim")
        work = features.to(device=self.device, dtype=torch.float64)
        return torch.cat(
            (work, torch.ones(work.shape[0], 1, device=self.device, dtype=work.dtype)),
            dim=1,
        )

    def _refresh(self) -> None:
        self.precision = 0.5 * (
            self.precision + self.precision.transpose(-1, -2)
        )
        factor, info = torch.linalg.cholesky_ex(self.precision)
        if bool(info.any()):
            raise RuntimeError("full-covariance posterior precision lost positivity")
        self.precision_cholesky = factor
        self.parameter_mean = torch.cholesky_solve(
            self.natural_mean.unsqueeze(-1), factor
        ).squeeze(-1)

    @torch.no_grad()
    def output_moments(self, features: Tensor) -> tuple[Tensor, Tensor]:
        design = self._augment(features)
        mean = design @ self.parameter_mean.mT
        right_hand_side = design.mT.unsqueeze(0).expand(
            self.num_classes, -1, -1
        )
        whitened = torch.linalg.solve_triangular(
            self.precision_cholesky, right_hand_side, upper=False
        )
        variance = whitened.square().sum(dim=1).mT.clamp_min(_EPS)
        return mean, variance

    def _event_sites(
        self, features: Tensor, labels: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
        design = self._augment(features)
        labels = labels.to(device=self.device, dtype=torch.long).reshape(-1)
        if labels.numel() != design.shape[0]:
            raise ValueError("labels must contain one value per feature row")
        prior_mean, prior_variance = self.output_moments(features)
        _, posterior_mean, posterior_covariance = multinomial_probit_adf_event(
            prior_mean,
            prior_variance,
            labels,
            probit_tau2=self.probit_tau2,
        )
        posterior_variance = posterior_covariance.diagonal(dim1=-2, dim2=-1)
        prior_variance = prior_variance.clamp_min(_EPS)
        posterior_variance = posterior_variance.clamp_min(_EPS)
        site_precision = (
            posterior_variance.reciprocal() - prior_variance.reciprocal()
        ).clamp_min(0.0)
        site_natural_mean = (
            posterior_mean / posterior_variance
            - prior_mean / prior_variance
        )
        statistics = {
            "site_precision_mean": site_precision.mean().item(),
            "site_precision_max": site_precision.max().item(),
            "output_variance_mean": prior_variance.mean().item(),
        }
        return design, site_precision, site_natural_mean, statistics

    def _apply_sites(
        self,
        design: Tensor,
        site_precision: Tensor,
        site_natural_mean: Tensor,
        *,
        sign: float,
    ) -> None:
        for class_index in range(self.num_classes):
            weighted_design = design * site_precision[:, class_index].sqrt()[:, None]
            self.precision[class_index].add_(
                weighted_design.mT @ weighted_design, alpha=sign
            )
        self.natural_mean.add_(site_natural_mean.mT @ design, alpha=sign)

    @torch.no_grad()
    def train_step(self, features: Tensor, labels: Tensor) -> dict[str, float]:
        """Consume one previously unseen minibatch through parallel ADF sites."""

        design, site_precision, site_natural_mean, statistics = self._event_sites(
            features, labels
        )
        self._apply_sites(
            design, site_precision, site_natural_mean, sign=1.0
        )
        self.samples_seen += design.shape[0]
        self._refresh()
        return statistics

    @torch.no_grad()
    def fit(
        self,
        features: Tensor,
        labels: Tensor,
        *,
        batch_size: int = 256,
        seed: int = 0,
        epochs: int = 1,
        epoch_callback: Callable[
            [int, "FullCovarianceADFClassifier", dict[str, float]], None
        ]
        | None = None,
    ) -> list[dict[str, float]]:
        """Run block EP with explicit site replacement after the first pass."""

        if self.samples_seen:
            raise RuntimeError("full-covariance ADF fit can consume the dataset only once")
        if batch_size < 1 or epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        generator = torch.Generator().manual_seed(seed)
        site_precision = torch.zeros(
            features.shape[0],
            self.num_classes,
            device=self.device,
            dtype=torch.float64,
        )
        site_natural_mean = torch.zeros_like(site_precision)
        records = []
        for epoch in range(1, epochs + 1):
            order = torch.randperm(features.shape[0], generator=generator)
            epoch_statistics = []
            for start in range(0, order.numel(), batch_size):
                indices = order[start : start + batch_size]
                device_indices = indices.to(self.device)
                batch_features = features[indices]
                batch_labels = labels[indices]
                design = self._augment(batch_features)
                old_precision = site_precision[device_indices]
                old_natural_mean = site_natural_mean[device_indices]
                if epoch > 1:
                    self._apply_sites(
                        design,
                        old_precision,
                        old_natural_mean,
                        sign=-1.0,
                    )
                    self._refresh()
                (
                    design,
                    new_precision,
                    new_natural_mean,
                    statistics,
                ) = self._event_sites(batch_features, batch_labels)
                self._apply_sites(
                    design, new_precision, new_natural_mean, sign=1.0
                )
                site_precision[device_indices] = new_precision
                site_natural_mean[device_indices] = new_natural_mean
                self._refresh()
                epoch_statistics.append(statistics)
            record = {
                "epoch": float(epoch),
                "site_precision_mean": sum(
                    item["site_precision_mean"] for item in epoch_statistics
                ) / len(epoch_statistics),
                "site_precision_max": max(
                    item["site_precision_max"] for item in epoch_statistics
                ),
                "output_variance_mean": sum(
                    item["output_variance_mean"] for item in epoch_statistics
                ) / len(epoch_statistics),
            }
            records.append(record)
            if epoch_callback is not None:
                epoch_callback(epoch, self, record)
        self.samples_seen = features.shape[0]
        self.site_precision = site_precision
        self.site_natural_mean = site_natural_mean
        return records

    @torch.no_grad()
    def predict(self, features: Tensor) -> FullCovarianceADFPrediction:
        mean, variance = self.output_moments(features)
        probabilities = multinomial_probit_adf_predictive_probs(
            mean, variance, probit_tau2=self.probit_tau2
        )
        return FullCovarianceADFPrediction(
            probabilities=probabilities,
            output_mean=mean,
            output_variance=variance,
            contrast_logdet=contrast_covariance_logdet(variance),
        )

    def state_dict(self) -> dict[str, Tensor | int | float]:
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "probit_tau2": self.probit_tau2,
            "prior_precision": self.prior_precision,
            "prior_natural_mean": self.prior_natural_mean,
            "precision": self.precision,
            "natural_mean": self.natural_mean,
            "samples_seen": self.samples_seen,
            **(
                {
                    "site_precision": self.site_precision,
                    "site_natural_mean": self.site_natural_mean,
                }
                if hasattr(self, "site_precision")
                else {}
            ),
        }
