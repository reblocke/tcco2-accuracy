"""Inverse inference utilities for TcCO2 → PaCO2."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._params import select_group_params
from ._posterior import (
    extract_params as _extract_params,
)
from ._posterior import (
    mixture_quantiles as _mixture_quantiles,
)
from ._posterior import (
    mixture_survival as _mixture_survival,
)
from ._posterior import (
    normalize_log_weights as _normalize_log_weights,
)
from ._posterior import (
    prior_log_weights as _prior_log_weights,
)
from ._posterior import (
    select_param_draws as _select_param_draws,
)
from ._posterior import (
    threshold_prob_map as _threshold_prob_map,
)
from ._posterior import (
    validate_prior as _validate_prior,
)
from ._posterior import (
    weighted_quantile as _weighted_quantile,
)
from .constants import PACO2_SUBGROUP_ORDER
from .paco2 import prepare_paco2_distribution
from .simulation import DEFAULT_CLASSIFICATION_THRESHOLDS
from .utils import validate_params_df

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
    params = validate_params_df(params)
    rng = np.random.default_rng(seed)
    # Bootstrap draws encode epistemic uncertainty; sd_total captures measurement error + heterogeneity.
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

    prepared = (
        paco2_data if "subgroup" in paco2_data.columns else prepare_paco2_distribution(paco2_data)
    )
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []

    for subgroup in PACO2_SUBGROUP_ORDER:
        paco2_values = prepared.loc[prepared["subgroup"] == subgroup, "paco2"].to_numpy(dtype=float)
        if paco2_values.size == 0:
            continue
        group_params = select_group_params(params, subgroup)
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
    # Mixture quantiles form the prediction interval (PI), not a CI.
    quantiles = _mixture_quantiles(DEFAULT_INFERENCE_QUANTILES, means, sd_total)
    summary = {
        "paco2_q025": float(quantiles[0]),
        "paco2_q500": float(quantiles[1]),
        "paco2_q975": float(quantiles[2]),
    }
    # Posterior mass at/above the threshold P(PaCO2 ≥ threshold).
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
    # Weighted quantiles form the prediction interval (PI), not a CI.
    quantiles = _weighted_quantile(paco2_values, weights, DEFAULT_INFERENCE_QUANTILES)
    summary = {
        "paco2_q025": float(quantiles[0]),
        "paco2_q500": float(quantiles[1]),
        "paco2_q975": float(quantiles[2]),
    }
    # Posterior mass at/above the threshold P(PaCO2 ≥ threshold).
    probs = [float(np.sum(weights[paco2_values >= threshold])) for threshold in thresholds]
    summary.update(_threshold_prob_map(thresholds, probs))
    return summary
