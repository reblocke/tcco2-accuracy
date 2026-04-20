"""Internal posterior mixture and prior-weighting utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .utils import threshold_label


def prior_log_weights(
    tcco2: float,
    paco2_values: np.ndarray,
    deltas: np.ndarray,
    sd_total: np.ndarray,
) -> np.ndarray:
    """Return log posterior weights for empirical PaCO2 prior values."""

    if paco2_values.size == 0:
        raise ValueError("Prior distribution must be non-empty.")
    if deltas.size == 0:
        raise ValueError("Parameter draws must be non-empty.")
    chunk_size = 5000
    log_weights = np.empty(paco2_values.size, dtype=float)
    for start in range(0, paco2_values.size, chunk_size):
        stop = min(start + chunk_size, paco2_values.size)
        chunk = paco2_values[start:stop]
        loc = chunk[:, None] - deltas[None, :]
        logpdf = stats.norm.logpdf(tcco2, loc=loc, scale=sd_total[None, :])
        log_weights[start:stop] = special.logsumexp(logpdf, axis=1) - math.log(deltas.size)
    return log_weights


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Normalize log weights to probability weights."""

    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise ValueError("Prior log-weights are all non-finite.")
    normalized = log_weights.copy()
    normalized[~finite] = -np.inf
    normalized = normalized - special.logsumexp(normalized)
    return np.exp(normalized)


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Iterable[float],
) -> np.ndarray:
    """Return weighted empirical quantiles."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        raise ValueError("Values must be non-empty.")
    if weights.size != values.size:
        raise ValueError("Weights must match values.")
    if not np.any(weights > 0):
        raise ValueError("Weights must contain positive entries.")
    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = weights[sorter]
    cumulative = np.cumsum(weights_sorted)
    cumulative = cumulative / cumulative[-1]
    quantile_values = np.asarray(list(quantiles), dtype=float)
    indices = np.searchsorted(cumulative, quantile_values, side="left")
    indices = np.clip(indices, 0, values_sorted.size - 1)
    return values_sorted[indices]


def mixture_survival(
    thresholds: np.ndarray,
    means: np.ndarray,
    sd_total: np.ndarray,
) -> np.ndarray:
    """Return mixture survival probabilities at thresholds."""

    z_scores = (thresholds[:, None] - means[None, :]) / sd_total[None, :]
    return np.mean(1 - stats.norm.cdf(z_scores), axis=1)


def mixture_quantiles(
    quantiles: Iterable[float],
    means: np.ndarray,
    sd_total: np.ndarray,
) -> np.ndarray:
    """Return quantiles of an equally weighted normal mixture."""

    return np.array([mixture_quantile(float(q), means, sd_total) for q in quantiles])


def mixture_quantile(
    quantile: float,
    means: np.ndarray,
    sd_total: np.ndarray,
) -> float:
    """Return one quantile of an equally weighted normal mixture."""

    if not 0 < quantile < 1:
        raise ValueError("Quantile must be between 0 and 1.")
    means = np.asarray(means, dtype=float)
    sd_total = np.asarray(sd_total, dtype=float)
    max_sd = np.max(sd_total)
    if max_sd <= 0:
        raise ValueError("Total SD must be positive.")
    lower = np.min(means) - 8 * max_sd
    upper = np.max(means) + 8 * max_sd
    cdf_lower = mixture_cdf(lower, means, sd_total)
    cdf_upper = mixture_cdf(upper, means, sd_total)
    if not (cdf_lower <= quantile <= cdf_upper):
        span = 10 * max_sd
        for _ in range(10):
            if cdf_lower > quantile:
                lower -= span
                cdf_lower = mixture_cdf(lower, means, sd_total)
            if cdf_upper < quantile:
                upper += span
                cdf_upper = mixture_cdf(upper, means, sd_total)
            if cdf_lower <= quantile <= cdf_upper:
                break
        else:
            raise ValueError("Failed to bracket mixture quantile.")
    return float(
        optimize.brentq(
            lambda x: mixture_cdf(x, means, sd_total) - quantile,
            lower,
            upper,
        )
    )


def mixture_cdf(value: float, means: np.ndarray, sd_total: np.ndarray) -> float:
    """Return the CDF of an equally weighted normal mixture."""

    z_scores = (value - means) / sd_total
    return float(np.mean(stats.norm.cdf(z_scores)))


def threshold_prob_map(thresholds: Sequence[float], probs: Sequence[float]) -> dict[str, float]:
    """Return threshold probability columns keyed with stable threshold labels."""

    return {
        f"p_ge_{threshold_label(threshold)}": float(prob)
        for threshold, prob in zip(thresholds, probs)
    }


def select_param_draws(
    params: pd.DataFrame,
    n_draws: int | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample parameter draws with replacement when requested."""

    if n_draws is None or n_draws >= params.shape[0]:
        return params.reset_index(drop=True)
    chosen = rng.choice(params.index.to_numpy(), size=int(n_draws), replace=True)
    return params.loc[chosen].reset_index(drop=True)


def extract_params(params: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return delta and total SD arrays from validated parameter draws."""

    deltas = params["delta"].to_numpy(dtype=float)
    sigma2 = params["sigma2"].to_numpy(dtype=float)
    tau2 = params["tau2"].to_numpy(dtype=float)
    sd_total = np.sqrt(sigma2 + tau2)
    if np.any(sd_total <= 0):
        raise ValueError("Total SD must be positive for inference.")
    return deltas, sd_total


def validate_prior(paco2_prior: np.ndarray | None) -> np.ndarray:
    """Return finite PaCO2 prior values or raise for missing/empty priors."""

    if paco2_prior is None:
        raise ValueError("Prior distribution must be provided.")
    prior = np.asarray(paco2_prior, dtype=float)
    prior = prior[np.isfinite(prior)]
    if prior.size == 0:
        raise ValueError("Prior distribution must be non-empty.")
    return prior
