"""Shared helper utilities for TcCO2 accuracy workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def threshold_label(threshold: float) -> str:
    """Return a label-safe string for threshold-based column names."""

    label = f"{threshold:g}".replace(".", "p")
    return label.replace("-", "m")


def quantile_key(prefix: str, quantile: float) -> str:
    """Return a standardized quantile column key.

    Args:
        prefix: Column prefix (e.g., "d" or "paco2").
        quantile: Quantile in [0, 1].

    Returns:
        Column key formatted as "{prefix}_qXYZ" with XYZ in permille.
    """

    label = f"q{int(round(quantile * 1000)):03d}"
    return f"{prefix}_{label}" if prefix else label


def n_draws_per_group(params_df: pd.DataFrame) -> int:
    """Return the maximum number of parameter draws per group."""

    if "group" not in params_df.columns:
        return int(params_df.shape[0])
    counts = params_df.groupby("group").size()
    return int(counts.max()) if not counts.empty else 0


def validate_params_df(params_df: pd.DataFrame) -> pd.DataFrame:
    """Validate bootstrap parameter draws.

    Required columns: delta, sigma2, tau2.
    Ensures values are numeric, finite, and variances are non-negative.
    """

    required = {"delta", "sigma2", "tau2"}
    missing = required - set(params_df.columns)
    if missing:
        raise ValueError(f"Missing parameter columns: {sorted(missing)}")

    for column in required:
        values = pd.to_numeric(params_df[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite values in parameter column: {column}")

    for column in ("sigma2", "tau2"):
        values = pd.to_numeric(params_df[column], errors="coerce").to_numpy(dtype=float)
        if np.any(values < 0):
            raise ValueError(f"Negative values in variance column: {column}")

    return params_df


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator/denominator or NaN if denominator is non-positive."""

    return float(numerator / denominator) if denominator > 0 else float("nan")
