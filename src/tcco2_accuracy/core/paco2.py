"""Pure PaCO2 distribution preparation and binned-prior helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_PACO2_QUANTILES,
    PACO2_PRIOR_GROUPS,
    PACO2_PRIOR_REQUIRED_COLUMNS,
    PACO2_REQUIRED_COLUMNS,
    PACO2_SUBGROUP_ORDER,
)
from .utils import quantile_key


def prepare_paco2_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """Filter PaCO2 rows and assign subgroup labels."""

    validate_paco2_columns(data)
    filtered = data.loc[data["paco2"].notna()].copy()
    filtered["subgroup"] = assign_paco2_subgroup(filtered)
    return filtered


def assign_paco2_subgroup(data: pd.DataFrame) -> pd.Series:
    """Assign mutually exclusive PaCO2 subgroup labels."""

    validate_paco2_columns(data)
    is_amb = data["is_amb"].fillna(0).astype(int)
    is_emer = data["is_emer"].fillna(0).astype(int)
    is_inp = data["is_inp"].fillna(0).astype(int)
    cc_time = data["cc_time"].fillna(0).astype(int)

    pft_mask = is_amb == 1
    icu_mask = (is_inp == 1) & (cc_time == 1) & (is_emer == 0) & (is_amb == 0)
    ed_inp_mask = (is_emer == 1) | (is_inp == 1)

    subgroup = pd.Series(
        np.select([pft_mask, icu_mask, ed_inp_mask], ["pft", "icu", "ed_inp"], default=pd.NA),
        index=data.index,
        dtype="object",
    )
    if subgroup.isna().any():
        raise ValueError("Unclassified PaCO2 records after subgroup assignment.")
    return subgroup


def paco2_subgroup_summary(
    data: pd.DataFrame,
    quantiles: Sequence[float] = DEFAULT_PACO2_QUANTILES,
) -> pd.DataFrame:
    """Summarize subgroup counts and PaCO2 quantiles."""

    if "subgroup" in data.columns:
        prepared = data.loc[data["paco2"].notna()].copy()
    else:
        prepared = prepare_paco2_distribution(data)

    quantile_list = list(quantiles)
    quantile_columns = [quantile_key("paco2", q) for q in quantile_list]
    rows: list[dict[str, float | int | str]] = []
    for group in PACO2_SUBGROUP_ORDER:
        subset = prepared[prepared["subgroup"] == group]
        if subset.empty:
            continue
        q_values = subset["paco2"].quantile(quantile_list, interpolation="linear")
        row: dict[str, float | int | str] = {"group": group, "count": int(subset.shape[0])}
        for q in quantile_list:
            row[quantile_key("paco2", q)] = float(q_values.loc[q])
        rows.append(row)

    return pd.DataFrame(rows, columns=["group", "count", *quantile_columns])


def build_paco2_prior_bins(
    data: pd.DataFrame,
    bin_width: float = 1.0,
) -> pd.DataFrame:
    """Return binned PaCO2 priors for each subgroup plus pooled "all"."""

    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")
    prepared = data if "subgroup" in data.columns else prepare_paco2_distribution(data)
    frames: list[pd.DataFrame] = []
    binned_counts: dict[str, pd.Series] = {}
    for group in PACO2_SUBGROUP_ORDER:
        values = prepared.loc[prepared["subgroup"] == group, "paco2"].to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError(f"No PaCO2 values available for subgroup '{group}'.")
        bins = np.round(values / bin_width) * bin_width
        counts = pd.Series(bins).value_counts().sort_index()
        total = float(counts.sum())
        frame = pd.DataFrame(
            {
                "group": group,
                "paco2_bin": counts.index.astype(float),
                "count": counts.to_numpy(dtype=int),
                "weight": counts.to_numpy(dtype=float) / total,
            }
        )
        frames.append(frame)
        binned_counts[group] = counts

    all_counts: pd.Series | None = None
    for counts in binned_counts.values():
        all_counts = counts if all_counts is None else all_counts.add(counts, fill_value=0)
    if all_counts is None:
        raise ValueError("Unable to pool PaCO2 prior bins across subgroups.")
    all_total = float(all_counts.sum())
    all_frame = pd.DataFrame(
        {
            "group": "all",
            "paco2_bin": all_counts.index.astype(float),
            "count": all_counts.to_numpy(dtype=int),
            "weight": all_counts.to_numpy(dtype=float) / all_total,
        }
    )
    frames.append(all_frame)

    result = pd.concat(frames, ignore_index=True)
    return validate_paco2_prior_bins(result)


def validate_paco2_prior_bins(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the browser/offline binned PaCO2 prior schema."""

    prior = data.copy()
    if "group" not in prior.columns and "subgroup" in prior.columns:
        prior = prior.rename(columns={"subgroup": "group"})
    missing = PACO2_PRIOR_REQUIRED_COLUMNS - set(prior.columns)
    if missing:
        raise ValueError(f"Missing prior bin columns: {sorted(missing)}")
    prior["group"] = prior["group"].astype(str).str.strip().str.lower()
    prior["paco2_bin"] = pd.to_numeric(prior["paco2_bin"], errors="coerce")
    prior["count"] = pd.to_numeric(prior["count"], errors="coerce")
    prior["weight"] = pd.to_numeric(prior["weight"], errors="coerce")
    if not np.all(np.isfinite(prior["paco2_bin"])):
        raise ValueError("Non-finite PaCO2 bin values in prior.")
    if not np.all(np.isfinite(prior["count"])):
        raise ValueError("Non-finite counts in prior.")
    if not np.all(np.isfinite(prior["weight"])):
        raise ValueError("Non-finite weights in prior.")
    if np.any(prior["count"] < 0):
        raise ValueError("Prior counts must be non-negative.")
    if np.any(prior["weight"] < 0):
        raise ValueError("Prior weights must be non-negative.")
    groups = set(prior["group"])
    missing_groups = set(PACO2_PRIOR_GROUPS) - groups
    if missing_groups:
        raise ValueError(f"Prior bins missing groups: {sorted(missing_groups)}")
    weight_sums = prior.groupby("group")["weight"].sum()
    if not np.allclose(weight_sums.to_numpy(dtype=float), 1.0, atol=1e-6):
        raise ValueError("Prior weights must sum to 1 within each group.")
    return prior


def prior_values_from_bins(prior_bins: pd.DataFrame, group: str) -> np.ndarray:
    """Expand binned PaCO2 prior counts into empirical prior values."""

    subset = prior_bins.loc[prior_bins["group"] == group]
    if subset.empty:
        raise ValueError(f"No binned priors available for group '{group}'.")
    return np.repeat(
        subset["paco2_bin"].to_numpy(dtype=float),
        subset["count"].to_numpy(dtype=int),
    )


def validate_paco2_columns(data: pd.DataFrame) -> None:
    """Validate columns required to assign PaCO2 analysis subgroups."""

    missing = PACO2_REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing PaCO2 columns: {sorted(missing)}")
