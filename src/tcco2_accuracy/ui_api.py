"""Compute-layer helpers for TcCO2 → PaCO2 UI inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from ._params import select_conway_studies_for_subgroup, select_group_params
from ._posterior import (
    extract_params,
    mixture_quantiles,
    mixture_survival,
    normalize_log_weights,
    prior_log_weights,
    select_param_draws,
    validate_prior,
    weighted_quantile,
)
from .bootstrap import bootstrap_conway_parameters
from .data import prepare_conway_meta_inputs

SubgroupLabel = Literal["pft", "ed_inp", "icu", "all"]
InferenceMode = Literal["prior_weighted", "likelihood_only"]


@dataclass(frozen=True)
class PredictionResult:
    subgroup: SubgroupLabel
    tcco2: float
    threshold: float
    mode: InferenceMode
    interval: float
    paco2_q_low: float
    paco2_median: float
    paco2_q_high: float
    p_ge_threshold: float
    decision_label: str
    p_true_positive: float
    p_false_positive: float
    p_true_negative: float
    p_false_negative: float
    paco2_bin: np.ndarray
    posterior_prob: np.ndarray
    prior_prob: np.ndarray | None
    posterior_cdf: np.ndarray
    likelihood_prob: np.ndarray | None = None


def predict_paco2_from_tcco2(
    tcco2: float,
    subgroup: SubgroupLabel,
    threshold: float = 45.0,
    mode: InferenceMode = "prior_weighted",
    interval: float = 0.95,
    params_draws: pd.DataFrame | None = None,
    paco2_prior_values: np.ndarray | None = None,
    paco2_prior_weights: np.ndarray | None = None,
    n_param_draws: int | None = None,
    seed: int | None = None,
    bin_width: float = 1.0,
) -> PredictionResult:
    """Return posterior summary and histogram for a single TcCO2 value."""

    if params_draws is None:
        raise ValueError("params_draws must be provided for UI inference.")
    if interval <= 0 or interval >= 1:
        raise ValueError("interval must be between 0 and 1.")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")

    params = select_group_params(params_draws, subgroup, validate=True, reset_index=True)
    rng = np.random.default_rng(seed)
    params = select_param_draws(params, n_draws=n_param_draws, rng=rng)
    deltas, sd_total = extract_params(params)
    lower_q = (1 - interval) / 2
    upper_q = 1 - lower_q
    quantiles = (lower_q, 0.5, upper_q)

    if mode == "prior_weighted":
        paco2_values = validate_prior(paco2_prior_values)
        prior_weights = _validate_prior_weights(paco2_prior_weights, paco2_values)
        # Prior-weighted updates the empirical PaCO2 pretest prior with the TcCO2 likelihood.
        log_weights = prior_log_weights(float(tcco2), paco2_values, deltas, sd_total)
        if prior_weights is not None:
            with np.errstate(divide="ignore"):
                log_prior_weights = np.where(prior_weights > 0, np.log(prior_weights), -np.inf)
            log_weights = log_weights + log_prior_weights
        weights = normalize_log_weights(log_weights)
        # Prediction interval (PI) reflects parameter uncertainty + measurement variability.
        paco2_interval = weighted_quantile(paco2_values, weights, quantiles)
        # Threshold probability is posterior mass at/above the chosen hypercapnia cutoff.
        p_ge_threshold = float(np.sum(weights[paco2_values >= float(threshold)]))
        paco2_bin, posterior_prob, prior_prob, likelihood_prob = _posterior_histogram(
            tcco2=float(tcco2),
            deltas=deltas,
            sd_total=sd_total,
            bin_width=float(bin_width),
            mode=mode,
            paco2_values=paco2_values,
            prior_weights=prior_weights,
            weights=weights,
        )
    elif mode == "likelihood_only":
        means = float(tcco2) + deltas
        # Prediction interval (PI) reflects parameter uncertainty + measurement variability.
        paco2_interval = mixture_quantiles(quantiles, means, sd_total)
        # Threshold probability is posterior mass at/above the chosen hypercapnia cutoff.
        p_ge_threshold = float(mixture_survival(np.array([float(threshold)]), means, sd_total)[0])
        paco2_bin, posterior_prob, prior_prob, likelihood_prob = _posterior_histogram(
            tcco2=float(tcco2),
            deltas=deltas,
            sd_total=sd_total,
            bin_width=float(bin_width),
            mode=mode,
        )
    else:
        raise ValueError(f"Unknown inference mode: {mode}")

    decision_label = "positive" if float(tcco2) >= float(threshold) else "negative"
    p_true_positive = p_ge_threshold if decision_label == "positive" else 0.0
    p_false_positive = (1 - p_ge_threshold) if decision_label == "positive" else 0.0
    p_true_negative = (1 - p_ge_threshold) if decision_label == "negative" else 0.0
    p_false_negative = p_ge_threshold if decision_label == "negative" else 0.0
    posterior_cdf = np.cumsum(posterior_prob)

    return PredictionResult(
        subgroup=subgroup,
        tcco2=float(tcco2),
        threshold=float(threshold),
        mode=mode,
        interval=float(interval),
        paco2_q_low=float(paco2_interval[0]),
        paco2_median=float(paco2_interval[1]),
        paco2_q_high=float(paco2_interval[2]),
        p_ge_threshold=float(p_ge_threshold),
        decision_label=decision_label,
        p_true_positive=float(p_true_positive),
        p_false_positive=float(p_false_positive),
        p_true_negative=float(p_true_negative),
        p_false_negative=float(p_false_negative),
        paco2_bin=paco2_bin,
        posterior_prob=posterior_prob,
        prior_prob=prior_prob,
        likelihood_prob=likelihood_prob,
        posterior_cdf=posterior_cdf,
    )


def build_subgroup_bootstrap_draws(
    studies: pd.DataFrame,
    subgroup: SubgroupLabel,
    n_boot: int,
    seed: int | None,
    bootstrap_mode: str,
) -> pd.DataFrame:
    """Return bootstrap parameter draws for one UI subgroup selection."""

    subset = select_conway_studies_for_subgroup(studies, subgroup)
    conway_inputs = prepare_conway_meta_inputs(subset)
    return bootstrap_conway_parameters(
        conway_inputs,
        n_boot=n_boot,
        seed=seed,
        bootstrap_mode=bootstrap_mode,
        truncate_tau2=True,
    )


def _posterior_histogram(
    tcco2: float,
    deltas: np.ndarray,
    sd_total: np.ndarray,
    bin_width: float,
    mode: InferenceMode,
    paco2_values: np.ndarray | None = None,
    prior_weights: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return histogram bins and posterior/prior/scaled-likelihood probabilities."""

    if mode == "prior_weighted":
        if paco2_values is None or weights is None:
            raise ValueError("Prior-weighted histogram requires paco2 values and weights.")
        min_value = float(np.min(paco2_values))
        max_value = float(np.max(paco2_values))
    else:
        max_sd = float(np.max(sd_total))
        min_value = float(np.min(tcco2 + deltas) - 6 * max_sd)
        max_value = float(np.max(tcco2 + deltas) + 6 * max_sd)

    min_center = math.floor(min_value / bin_width) * bin_width
    max_center = math.ceil(max_value / bin_width) * bin_width
    if max_center <= min_center:
        max_center = min_center + bin_width
    if not np.isfinite(min_center) or not np.isfinite(max_center) or max_center <= min_center:
        raise ValueError("Invalid histogram bounds for posterior.")
    paco2_bin = np.arange(min_center, max_center + bin_width, bin_width)
    bin_edges = np.concatenate([paco2_bin - bin_width / 2, [paco2_bin[-1] + bin_width / 2]])

    if mode == "prior_weighted":
        weighted_hist = np.histogram(paco2_values, bins=bin_edges, weights=weights)[0]
        total_weight = float(np.sum(weighted_hist))
        if total_weight <= 0:
            raise ValueError("Posterior weights must be positive for histogram.")
        posterior_prob = weighted_hist / total_weight
        prior_hist = np.histogram(paco2_values, bins=bin_edges, weights=prior_weights)[0]
        if prior_weights is None:
            prior_hist = np.histogram(paco2_values, bins=bin_edges)[0]
        prior_prob = prior_hist / float(np.sum(prior_hist))
        likelihood_prob = _scaled_likelihood_prob(
            tcco2=tcco2,
            deltas=deltas,
            sd_total=sd_total,
            paco2_bin=paco2_bin,
            bin_width=bin_width,
        )
        return paco2_bin, posterior_prob, prior_prob, likelihood_prob

    densities = stats.norm.pdf(
        paco2_bin[:, None], loc=tcco2 + deltas[None, :], scale=sd_total[None, :]
    )
    mixture_density = np.mean(densities, axis=1)
    posterior_prob = mixture_density * bin_width
    mass = float(np.sum(posterior_prob))
    if mass <= 0:
        raise ValueError("Posterior density must integrate to positive mass.")
    posterior_prob = posterior_prob / mass
    return paco2_bin, posterior_prob, None, None


def _validate_prior_weights(
    prior_weights: np.ndarray | None,
    paco2_values: np.ndarray,
) -> np.ndarray | None:
    """Return normalized prior weights aligned to PaCO2 support, when provided."""

    if prior_weights is None:
        return None
    weights = np.asarray(prior_weights, dtype=float)
    if weights.shape != paco2_values.shape:
        raise ValueError("Prior weights must match prior values.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Prior weights must be finite.")
    if np.any(weights < 0):
        raise ValueError("Prior weights must be non-negative.")
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("Prior weights must contain positive mass.")
    return weights / total


def _scaled_likelihood_prob(
    tcco2: float,
    deltas: np.ndarray,
    sd_total: np.ndarray,
    paco2_bin: np.ndarray,
    bin_width: float,
) -> np.ndarray:
    """Return a normalized likelihood curve on the posterior histogram grid."""

    densities = stats.norm.pdf(
        tcco2,
        loc=paco2_bin[:, None] - deltas[None, :],
        scale=sd_total[None, :],
    )
    likelihood_prob = np.mean(densities, axis=1) * bin_width
    mass = float(np.sum(likelihood_prob))
    if mass <= 0:
        raise ValueError("Scaled likelihood must integrate to positive mass.")
    return likelihood_prob / mass
