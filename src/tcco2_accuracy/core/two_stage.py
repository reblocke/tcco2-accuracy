"""Two-stage (zone + reflex) strategy calculations for TcCO2 screening."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .utils import (
    _likelihood_ratio_from_log_probabilities,
    quantile_key,
    safe_ratio,
    validate_params_df,
)


@dataclass(frozen=True)
class TwoStagePolicy:
    """Definition of the TcCO2 zone-based reflex strategy."""

    lower: float
    upper: float
    true_threshold: float

    def __post_init__(self) -> None:
        if self.lower >= self.upper:
            raise ValueError("Two-stage policy lower bound must be < upper bound.")


def two_stage_zone_probabilities(
    paco2_values: Sequence[float],
    delta: float,
    sd_total: float,
    policy: TwoStagePolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-record probabilities of falling in each TcCO2 zone.

    TcCO2 is modeled as Normal(PaCO2 - delta, sd_total), so the zone probabilities
    integrate measurement error analytically using the Normal CDF.
    """

    paco2_values = np.asarray(paco2_values, dtype=float)
    if paco2_values.size == 0:
        raise ValueError("PaCO2 values must be non-empty.")
    if sd_total <= 0:
        raise ValueError("Total SD must be positive.")

    mean_tcco2 = paco2_values - float(delta)
    lower_z = (float(policy.lower) - mean_tcco2) / float(sd_total)
    upper_z = (float(policy.upper) - mean_tcco2) / float(sd_total)
    log_zone1, log_zone2, log_zone3 = _two_stage_log_probabilities(lower_z, upper_z)
    zone1 = np.exp(log_zone1)
    zone2 = np.exp(log_zone2)
    zone3 = np.exp(log_zone3)
    return zone1, zone2, zone3


def two_stage_metrics(
    paco2_values: Sequence[float],
    delta: float,
    sd_total: float,
    policy: TwoStagePolicy,
) -> dict[str, float]:
    """Compute expected two-stage strategy metrics for one parameter draw."""

    paco2_values = np.asarray(paco2_values, dtype=float)
    zone1, zone2, zone3 = two_stage_zone_probabilities(paco2_values, delta, sd_total, policy)
    mean_tcco2 = paco2_values - float(delta)
    lower_z = (float(policy.lower) - mean_tcco2) / float(sd_total)
    upper_z = (float(policy.upper) - mean_tcco2) / float(sd_total)
    log_zone1, log_zone2, log_zone3 = _two_stage_log_probabilities(lower_z, upper_z)

    total = float(paco2_values.size)
    pos_mask = paco2_values >= float(policy.true_threshold)
    neg_mask = ~pos_mask
    prevalence = pos_mask.mean()
    neg_prevalence = 1 - prevalence

    zone1_prob = zone1.mean()
    zone2_prob = zone2.mean()
    zone3_prob = zone3.mean()

    zone1_pos = zone1[pos_mask].sum() / total
    zone2_pos = zone2[pos_mask].sum() / total
    zone3_pos = zone3[pos_mask].sum() / total
    zone1_neg = zone1[neg_mask].sum() / total
    zone2_neg = zone2[neg_mask].sum() / total
    zone3_neg = zone3[neg_mask].sum() / total

    zone1_pos_rate = safe_ratio(zone1_pos, prevalence)
    zone2_pos_rate = safe_ratio(zone2_pos, prevalence)
    zone3_pos_rate = safe_ratio(zone3_pos, prevalence)
    zone1_neg_rate = safe_ratio(zone1_neg, neg_prevalence)
    zone2_neg_rate = safe_ratio(zone2_neg, neg_prevalence)
    zone3_neg_rate = safe_ratio(zone3_neg, neg_prevalence)

    # Likelihood ratios summarize how much each zone shifts the pretest odds.
    # Log probabilities keep extreme yet representable tail ratios finite.
    zone1_lr = _likelihood_ratio_from_log_probabilities(log_zone1[pos_mask], log_zone1[neg_mask])
    zone2_lr = _likelihood_ratio_from_log_probabilities(log_zone2[pos_mask], log_zone2[neg_mask])
    zone3_lr = _likelihood_ratio_from_log_probabilities(log_zone3[pos_mask], log_zone3[neg_mask])

    # Post-test probabilities combine zone frequency with pretest prevalence.
    zone1_post = safe_ratio(zone1_pos, zone1_prob)
    zone2_post = safe_ratio(zone2_pos, zone2_prob)
    zone3_post = safe_ratio(zone3_pos, zone3_prob)

    # Reflex ABG removes zone 2 errors; residual misclassification remains in zones 1 and 3.
    residual_fn = zone1_pos
    residual_fp = zone3_neg
    residual_total = residual_fn + residual_fp

    return {
        "prevalence": float(prevalence),
        "zone1_prob": float(zone1_prob),
        "zone2_prob": float(zone2_prob),
        "zone3_prob": float(zone3_prob),
        "zone1_pos_rate": float(zone1_pos_rate),
        "zone2_pos_rate": float(zone2_pos_rate),
        "zone3_pos_rate": float(zone3_pos_rate),
        "zone1_neg_rate": float(zone1_neg_rate),
        "zone2_neg_rate": float(zone2_neg_rate),
        "zone3_neg_rate": float(zone3_neg_rate),
        "zone1_lr": float(zone1_lr),
        "zone2_lr": float(zone2_lr),
        "zone3_lr": float(zone3_lr),
        "zone1_post_prob": float(zone1_post),
        "zone2_post_prob": float(zone2_post),
        "zone3_post_prob": float(zone3_post),
        "reflex_fraction": float(zone2_prob),
        "residual_fn": float(residual_fn),
        "residual_fp": float(residual_fp),
        "residual_misclass": float(residual_total),
        "residual_fn_per_1000": float(residual_fn * 1000),
        "residual_fp_per_1000": float(residual_fp * 1000),
        "residual_misclass_per_1000": float(residual_total * 1000),
    }


def _two_stage_log_probabilities(
    lower_z: np.ndarray,
    upper_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stable log probabilities for the three ordered normal zones."""

    log_zone1 = stats.norm.logcdf(lower_z)
    log_zone3 = stats.norm.logsf(upper_z)
    log_larger = np.empty_like(lower_z, dtype=float)
    log_smaller = np.empty_like(lower_z, dtype=float)

    # Above zero, subtract survival functions so two nearly-one CDFs are never
    # differenced. Elsewhere, the CDF difference is stable on its log scale.
    positive = lower_z >= 0
    log_larger[positive] = stats.norm.logsf(lower_z[positive])
    log_smaller[positive] = stats.norm.logsf(upper_z[positive])
    log_larger[~positive] = stats.norm.logcdf(upper_z[~positive])
    log_smaller[~positive] = stats.norm.logcdf(lower_z[~positive])
    log_zone2 = _log_probability_difference(log_larger, log_smaller)

    endpoint_difference = log_larger - log_smaller
    endpoint_scale = np.maximum.reduce(
        [np.ones_like(log_larger), np.abs(log_larger), np.abs(log_smaller)]
    )
    # Route only endpoint differences comparable to floating-point roundoff to
    # quadrature. A sqrt(eps) rule is too broad in deep tails, where the log
    # endpoint scale is large but stable subtraction remains preferable over a
    # fixed-order density integral across a rapidly changing interval.
    cancellation_prone = endpoint_difference <= 64 * np.finfo(float).eps * endpoint_scale
    use_quadrature = cancellation_prone | ~np.isfinite(log_zone2)
    if np.any(use_quadrature):
        log_zone2[use_quadrature] = _log_normal_interval_probability(
            lower_z[use_quadrature],
            upper_z[use_quadrature],
        )
    return log_zone1, log_zone2, log_zone3


def _log_probability_difference(log_larger: np.ndarray, log_smaller: np.ndarray) -> np.ndarray:
    """Return log(exp(log_larger) - exp(log_smaller)) elementwise."""

    log_larger = np.asarray(log_larger, dtype=float)
    log_smaller = np.asarray(log_smaller, dtype=float)
    difference = np.minimum(log_smaller - log_larger, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = log_larger + np.log(-np.expm1(difference))
    result = np.where(np.isneginf(log_smaller), log_larger, result)
    return result


def _log_normal_interval_probability(
    lower_z: np.ndarray,
    upper_z: np.ndarray,
) -> np.ndarray:
    """Return log normal interval mass when endpoint logs are indistinguishable.

    Fixed Gauss-Legendre quadrature evaluates the positive density over the
    interval directly. This fallback is used only when log-CDF/log-SF endpoint
    subtraction has collapsed, which occurs for valid extremely narrow zones.
    """

    lower_z = np.asarray(lower_z, dtype=float)
    upper_z = np.asarray(upper_z, dtype=float)
    half_width = (upper_z - lower_z) / 2
    midpoint = lower_z + half_width
    nodes, weights = np.polynomial.legendre.leggauss(8)
    evaluation_points = midpoint[:, None] + half_width[:, None] * nodes[None, :]
    log_weighted_density = stats.norm.logpdf(evaluation_points) + np.log(weights)[None, :]
    return np.log(half_width) + np.logaddexp.reduce(log_weighted_density, axis=1)


def summarize_two_stage_draws(
    paco2_values: Sequence[float],
    params_df: pd.DataFrame,
    policy: TwoStagePolicy,
    quantiles: Sequence[float] = (0.025, 0.5, 0.975),
    n_draws: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Summarize two-stage metrics across bootstrap parameter draws."""

    params = validate_params_df(params_df)
    rng = np.random.default_rng(seed)
    if n_draws is not None and n_draws < params.shape[0]:
        chosen = rng.choice(params.index.to_numpy(), size=n_draws, replace=True)
        params = params.loc[chosen].reset_index(drop=True)

    rows: list[dict[str, float]] = []
    for params_row in params.itertuples(index=False):
        delta = float(getattr(params_row, "delta"))
        sigma2 = float(getattr(params_row, "sigma2"))
        tau2 = float(getattr(params_row, "tau2"))
        sd_total = math.sqrt(sigma2 + tau2)
        rows.append(two_stage_metrics(paco2_values, delta, sd_total, policy))

    metrics = pd.DataFrame(rows)
    quantiles = tuple(float(q) for q in quantiles)
    summary: dict[str, float] = {}
    for column in metrics.columns:
        values = metrics[column].to_numpy(dtype=float)
        summary.update(
            {
                f"{column}_{quantile_key('', q).lstrip('_')}": float(np.quantile(values, q))
                for q in quantiles
            }
        )
    return pd.DataFrame([summary])
