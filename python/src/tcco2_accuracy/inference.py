"""Inverse inference utilities for TcCO2 → PaCO2."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .data import PACO2_SUBGROUP_ORDER, prepare_paco2_distribution
from .simulation import DEFAULT_CLASSIFICATION_THRESHOLDS, PACO2_TO_CONWAY_GROUP


DEFAULT_INFERENCE_QUANTILES: tuple[float, ...] = (0.025, 0.5, 0.975)


def infer_paco2(
    tcco2_values: Sequence[float],
    params: pd.DataFrame,
    thresholds: Sequence[float] = DEFAULT_CLASSIFICATION_THRESHOLDS,
    paco2_prior: np.ndarray | None = None,
    use_prior: bool = False,
    seed: int | None = None,
    n_draws: int | None = None,
) -> pd.DataFrame:
    """Return PaCO2 posterior summaries for given TcCO2 values."""

    if use_prior and paco2_prior is None:
        raise ValueError("Prior-weighted inference requires paco2_prior values.")
    params = _validate_params(params)
    rng = np.random.default_rng(seed)
    params = _select_param_draws(params, n_draws=n_draws, rng=rng)
    deltas, sd_total = _extract_params(params)
    paco2_prior_values = _validate_prior(paco2_prior) if use_prior else None
    thresholds = [float(value) for value in thresholds]

    rows: list[dict[str, float]] = []
    for tcco2 in tcco2_values:
        if use_prior:
            summary = _prior_weighted_summary(
                float(tcco2),
                paco2_prior_values,
                deltas,
                sd_total,
                thresholds,
            )
        else:
            summary = _likelihood_summary(float(tcco2), deltas, sd_total, thresholds)
        summary["tcco2"] = float(tcco2)
        rows.append(summary)

    return pd.DataFrame(rows)


def infer_paco2_by_subgroup(
    tcco2_values: Sequence[float],
    paco2_data: pd.DataFrame,
    params: pd.DataFrame,
    thresholds: Sequence[float] = DEFAULT_CLASSIFICATION_THRESHOLDS,
    use_prior: bool = False,
    seed: int | None = None,
    n_draws: int | None = None,
) -> pd.DataFrame:
    """Return posterior summaries for each PaCO2 subgroup."""

    prepared = paco2_data if "subgroup" in paco2_data.columns else prepare_paco2_distribution(paco2_data)
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    available_groups = set(params["group"]) if "group" in params.columns else set()

    for subgroup in PACO2_SUBGROUP_ORDER:
        paco2_values = prepared.loc[prepared["subgroup"] == subgroup, "paco2"].to_numpy(dtype=float)
        if paco2_values.size == 0:
            continue
        if "group" in params.columns:
            group_key = subgroup if subgroup in available_groups else PACO2_TO_CONWAY_GROUP.get(subgroup, subgroup)
            group_params = params[params["group"] == group_key]
        else:
            group_params = params
        if group_params.empty:
            continue
        group_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        summaries = infer_paco2(
            tcco2_values,
            group_params,
            thresholds=thresholds,
            paco2_prior=paco2_values if use_prior else None,
            use_prior=use_prior,
            seed=group_seed,
            n_draws=n_draws,
        )
        summaries.insert(0, "group", subgroup)
        frames.append(summaries)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _likelihood_summary(
    tcco2: float,
    deltas: np.ndarray,
    sd_total: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, float]:
    means = tcco2 + deltas
    quantiles = _mixture_quantiles(DEFAULT_INFERENCE_QUANTILES, means, sd_total)
    summary = {
        "paco2_q025": float(quantiles[0]),
        "paco2_q500": float(quantiles[1]),
        "paco2_q975": float(quantiles[2]),
    }
    probs = _mixture_survival(np.asarray(thresholds, dtype=float), means, sd_total)
    summary.update(_threshold_prob_map(thresholds, probs))
    return summary


def _prior_weighted_summary(
    tcco2: float,
    paco2_prior: np.ndarray,
    deltas: np.ndarray,
    sd_total: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, float]:
    paco2_values = np.asarray(paco2_prior, dtype=float)
    log_weights = _prior_log_weights(tcco2, paco2_values, deltas, sd_total)
    weights = _normalize_log_weights(log_weights)
    quantiles = _weighted_quantile(paco2_values, weights, DEFAULT_INFERENCE_QUANTILES)
    summary = {
        "paco2_q025": float(quantiles[0]),
        "paco2_q500": float(quantiles[1]),
        "paco2_q975": float(quantiles[2]),
    }
    probs = [float(np.sum(weights[paco2_values >= threshold])) for threshold in thresholds]
    summary.update(_threshold_prob_map(thresholds, probs))
    return summary


def _prior_log_weights(
    tcco2: float,
    paco2_values: np.ndarray,
    deltas: np.ndarray,
    sd_total: np.ndarray,
) -> np.ndarray:
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


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise ValueError("Prior log-weights are all non-finite.")
    normalized = log_weights.copy()
    normalized[~finite] = -np.inf
    normalized = normalized - special.logsumexp(normalized)
    return np.exp(normalized)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Iterable[float],
) -> np.ndarray:
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


def _mixture_survival(
    thresholds: np.ndarray,
    means: np.ndarray,
    sd_total: np.ndarray,
) -> np.ndarray:
    z_scores = (thresholds[:, None] - means[None, :]) / sd_total[None, :]
    return np.mean(1 - stats.norm.cdf(z_scores), axis=1)


def _mixture_quantiles(
    quantiles: Iterable[float],
    means: np.ndarray,
    sd_total: np.ndarray,
) -> np.ndarray:
    return np.array([
        _mixture_quantile(float(q), means, sd_total) for q in quantiles
    ])


def _mixture_quantile(
    quantile: float,
    means: np.ndarray,
    sd_total: np.ndarray,
) -> float:
    if not 0 < quantile < 1:
        raise ValueError("Quantile must be between 0 and 1.")
    means = np.asarray(means, dtype=float)
    sd_total = np.asarray(sd_total, dtype=float)
    max_sd = np.max(sd_total)
    if max_sd <= 0:
        raise ValueError("Total SD must be positive.")
    lower = np.min(means) - 8 * max_sd
    upper = np.max(means) + 8 * max_sd
    cdf_lower = _mixture_cdf(lower, means, sd_total)
    cdf_upper = _mixture_cdf(upper, means, sd_total)
    if not (cdf_lower <= quantile <= cdf_upper):
        span = 10 * max_sd
        for _ in range(10):
            if cdf_lower > quantile:
                lower -= span
                cdf_lower = _mixture_cdf(lower, means, sd_total)
            if cdf_upper < quantile:
                upper += span
                cdf_upper = _mixture_cdf(upper, means, sd_total)
            if cdf_lower <= quantile <= cdf_upper:
                break
        else:
            raise ValueError("Failed to bracket mixture quantile.")
    return float(
        optimize.brentq(
            lambda x: _mixture_cdf(x, means, sd_total) - quantile,
            lower,
            upper,
        )
    )


def _mixture_cdf(value: float, means: np.ndarray, sd_total: np.ndarray) -> float:
    z_scores = (value - means) / sd_total
    return float(np.mean(stats.norm.cdf(z_scores)))


def _threshold_prob_map(thresholds: Sequence[float], probs: Sequence[float]) -> dict[str, float]:
    return {
        f"p_ge_{_threshold_label(threshold)}": float(prob)
        for threshold, prob in zip(thresholds, probs)
    }


def _threshold_label(value: float) -> str:
    label = f"{value:g}".replace(".", "p")
    return label.replace("-", "m")


def _select_param_draws(
    params: pd.DataFrame,
    n_draws: int | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if n_draws is None or n_draws >= params.shape[0]:
        return params.reset_index(drop=True)
    chosen = rng.choice(params.index.to_numpy(), size=int(n_draws), replace=True)
    return params.loc[chosen].reset_index(drop=True)


def _extract_params(params: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    deltas = params["delta"].to_numpy(dtype=float)
    sigma2 = params["sigma2"].to_numpy(dtype=float)
    tau2 = params["tau2"].to_numpy(dtype=float)
    sd_total = np.sqrt(sigma2 + tau2)
    if np.any(sd_total <= 0):
        raise ValueError("Total SD must be positive for inference.")
    return deltas, sd_total


def _validate_params(params: pd.DataFrame) -> pd.DataFrame:
    missing = {"delta", "sigma2", "tau2"} - set(params.columns)
    if missing:
        raise ValueError(f"Missing parameter columns: {sorted(missing)}")
    return params


def _validate_prior(paco2_prior: np.ndarray | None) -> np.ndarray:
    if paco2_prior is None:
        raise ValueError("Prior distribution must be provided.")
    prior = np.asarray(paco2_prior, dtype=float)
    prior = prior[np.isfinite(prior)]
    if prior.size == 0:
        raise ValueError("Prior distribution must be non-empty.")
    return prior
