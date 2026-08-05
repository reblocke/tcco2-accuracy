"""Conditional misclassification curve calculations."""

from __future__ import annotations

import math
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .utils import validate_params_df
from .validate_inputs import validate_paco2_values, validate_threshold

BIN_METHODS = ("round", "floor", "cut")


def conditional_classification_curves(
    paco2_values: Sequence[float],
    params_df: pd.DataFrame,
    threshold: float,
    bin_width: float = 1.0,
    bin_method: str = "cut",
    quantiles: Iterable[float] = (0.025, 0.5, 0.975),
    n_draws: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Return conditional TN/FP/FN/TP probability curves by PaCO2 bin."""

    paco2_values = validate_paco2_values(paco2_values)
    threshold = validate_threshold(threshold)
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
    sd_total = np.sqrt(
        params["sigma2"].to_numpy(dtype=float) + params["tau2"].to_numpy(dtype=float)
    )
    if np.any(sd_total <= 0):
        raise ValueError("Total SD must be positive for conditional curves.")

    bin_lower, bin_upper = _bin_paco2(
        paco2_values,
        bin_width=bin_width,
        bin_method=bin_method,
    )
    counts = pd.Series(bin_lower).value_counts().sort_index()
    total = float(paco2_values.size)

    quantiles = tuple(float(q) for q in quantiles)
    quantile_labels = [_quantile_label(q) for q in quantiles]
    rows: list[dict[str, float | int | str]] = []

    for paco2_bin, count in counts.items():
        in_bin = bin_lower == float(paco2_bin)
        values = paco2_values[in_bin]
        tn, fp, fn, tp = _conditional_components(
            values,
            deltas,
            sd_total,
            threshold=float(threshold),
        )
        row: dict[str, float | int | str] = {
            "threshold": float(threshold),
            "paco2_bin": float(paco2_bin),
            "paco2_bin_upper": float(bin_upper[in_bin][0]),
            "count": int(count),
            "weight": float(count / total),
        }
        _append_quantiles(row, "tn", tn, quantiles, quantile_labels)
        _append_quantiles(row, "fp", fp, quantiles, quantile_labels)
        _append_quantiles(row, "fn", fn, quantiles, quantile_labels)
        _append_quantiles(row, "tp", tp, quantiles, quantile_labels)
        rows.append(row)

    return pd.DataFrame(rows)


def _bin_paco2(
    values: np.ndarray,
    bin_width: float,
    bin_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return display-bin lower and upper edges.

    ``cut`` and legacy ``floor`` use conventional half-open bins. Legacy
    ``round`` retains nearest-center grouping, but exposes the corresponding
    half-open edges instead of allowing the rounded label to define truth.
    """

    values = np.asarray(values, dtype=float)
    lower = np.full(values.shape, np.nan, dtype=float)
    upper = np.full(values.shape, np.nan, dtype=float)
    finite_values, inverse = np.unique(values[np.isfinite(values)], return_inverse=True)
    width = Decimal(str(float(bin_width)))
    half_width = width / 2
    unique_lower: list[float] = []
    unique_upper: list[float] = []
    for value in finite_values:
        decimal_value = Decimal(str(float(value)))
        quotient = decimal_value / width
        if bin_method == "round":
            # Half-up midpoint assignment ensures every value belongs to the
            # reported half-open interval; ties-to-even can put an exact
            # midpoint on the excluded upper edge.
            center_index = quotient.to_integral_value(rounding=ROUND_HALF_UP)
            decimal_lower = center_index * width - half_width
        else:
            lower_index = quotient.to_integral_value(rounding=ROUND_FLOOR)
            decimal_lower = lower_index * width
        unique_lower.append(float(decimal_lower))
        unique_upper.append(float(decimal_lower + width))

    finite_mask = np.isfinite(values)
    lower[finite_mask] = np.asarray(unique_lower, dtype=float)[inverse]
    upper[finite_mask] = np.asarray(unique_upper, dtype=float)[inverse]
    return lower, upper


def _conditional_components(
    paco2_values: np.ndarray,
    deltas: np.ndarray,
    sd_total: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-draw TN/FP/FN/TP mass conditional on one display bin."""

    count = float(paco2_values.size)
    tn = np.zeros_like(deltas, dtype=float)
    fp = np.zeros_like(deltas, dtype=float)
    fn = np.zeros_like(deltas, dtype=float)
    tp = np.zeros_like(deltas, dtype=float)

    # Bound peak memory for restricted distributions while retaining vectorized
    # arithmetic across parameter draws.
    chunk_size = 1000
    for start in range(0, paco2_values.size, chunk_size):
        values = paco2_values[start : start + chunk_size]
        # TcCO2 = PaCO2 - d. P(TcCO2 >= T | PaCO2=p) is the survival
        # probability at (T - (p - delta)) / sd_total.
        z_scores = (float(threshold) - values[:, None] + deltas[None, :]) / sd_total[None, :]
        p_test_pos = stats.norm.sf(z_scores)
        p_test_neg = stats.norm.cdf(z_scores)
        truth_pos = values >= float(threshold)
        truth_neg = ~truth_pos
        tn += p_test_neg[truth_neg].sum(axis=0)
        fp += p_test_pos[truth_neg].sum(axis=0)
        fn += p_test_neg[truth_pos].sum(axis=0)
        tp += p_test_pos[truth_pos].sum(axis=0)

    return tn / count, fp / count, fn / count, tp / count


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
