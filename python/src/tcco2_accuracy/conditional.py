"""Conditional misclassification curve calculations."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .utils import validate_params_df


BIN_METHODS = ("round", "floor", "cut")


def conditional_classification_curves(
    paco2_values: Sequence[float],
    params_df: pd.DataFrame,
    threshold: float,
    bin_width: float = 1.0,
    bin_method: str = "round",
    quantiles: Iterable[float] = (0.025, 0.5, 0.975),
    n_draws: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Return conditional TN/FP/FN/TP probability curves by PaCO2 bin."""

    paco2_values = np.asarray(paco2_values, dtype=float)
    if paco2_values.size == 0:
        raise ValueError("PaCO2 values must be non-empty.")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")
    if bin_method not in BIN_METHODS:
        raise ValueError(f"Unknown bin_method: {bin_method}")

    params = validate_params_df(params_df)
    rng = np.random.default_rng(seed)
    if n_draws is not None and n_draws < params.shape[0]:
        chosen = rng.choice(params.index.to_numpy(), size=n_draws, replace=True)
        params = params.loc[chosen].reset_index(drop=True)
    deltas = params["delta"].to_numpy(dtype=float)
    sd_total = np.sqrt(params["sigma2"].to_numpy(dtype=float) + params["tau2"].to_numpy(dtype=float))
    if np.any(sd_total <= 0):
        raise ValueError("Total SD must be positive for conditional curves.")

    binned = _bin_paco2(paco2_values, bin_width=bin_width, bin_method=bin_method)
    counts = pd.Series(binned).value_counts().sort_index()
    total = float(paco2_values.size)

    quantiles = tuple(float(q) for q in quantiles)
    quantile_labels = [_quantile_label(q) for q in quantiles]
    rows: list[dict[str, float | int | str]] = []

    for paco2_bin, count in counts.items():
        # TcCO2 = PaCO2 - d with d~Normal(delta, sd_total).
        # P(TcCO2 >= T | PaCO2=p) = P(d <= p - T) = Phi((p - T - delta)/sd_total).
        z_scores = (float(paco2_bin) - float(threshold) - deltas) / sd_total
        p_test_pos = stats.norm.cdf(z_scores)
        # Conditional TN/FP/FN/TP switch on whether true PaCO2 is below/above threshold.
        if paco2_bin < threshold:
            fp = p_test_pos
            tn = 1 - p_test_pos
            tp = np.zeros_like(p_test_pos)
            fn = np.zeros_like(p_test_pos)
        else:
            tp = p_test_pos
            fn = 1 - p_test_pos
            tn = np.zeros_like(p_test_pos)
            fp = np.zeros_like(p_test_pos)
        row: dict[str, float | int | str] = {
            "threshold": float(threshold),
            "paco2_bin": float(paco2_bin),
            "count": int(count),
            "weight": float(count / total),
        }
        _append_quantiles(row, "tn", tn, quantiles, quantile_labels)
        _append_quantiles(row, "fp", fp, quantiles, quantile_labels)
        _append_quantiles(row, "fn", fn, quantiles, quantile_labels)
        _append_quantiles(row, "tp", tp, quantiles, quantile_labels)
        rows.append(row)

    return pd.DataFrame(rows)


def _bin_paco2(values: np.ndarray, bin_width: float, bin_method: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if bin_method == "round":
        return np.round(values / bin_width) * bin_width
    if bin_method == "floor":
        return np.floor(values / bin_width) * bin_width
    min_edge = math.floor(values.min() / bin_width) * bin_width
    max_edge = math.ceil(values.max() / bin_width) * bin_width
    edges = np.arange(min_edge, max_edge + bin_width, bin_width)
    labels = edges[:-1]
    return pd.cut(values, bins=edges, right=False, include_lowest=True, labels=labels).to_numpy(dtype=float)


def _append_quantiles(
    row: dict[str, float | int | str],
    prefix: str,
    values: np.ndarray,
    quantiles: Sequence[float],
    quantile_labels: Sequence[str],
) -> None:
    quantile_values = np.quantile(values, quantiles)
    for label, quantile_value in zip(quantile_labels, quantile_values):
        row[f"{prefix}_{label}"] = float(quantile_value)


def _quantile_label(quantile: float) -> str:
    if math.isclose(quantile, 0.5):
        return "q50"
    return f"q{int(round(quantile * 1000)):03d}"
