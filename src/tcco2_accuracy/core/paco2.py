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
from .validate_inputs import validate_paco2_values


def prepare_paco2_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """Filter PaCO2 rows and assign subgroup labels."""

    if "paco2" not in data.columns:
        raise ValueError("Missing PaCO2 columns: ['paco2']")
    if "subgroup" in data.columns:
        prepared = _retained_paco2_rows(data)
        subgroup = prepared["subgroup"].astype("string").str.strip().str.lower()
        if subgroup.isna().any() or subgroup.eq("").any():
            raise ValueError("Prepared PaCO2 subgroup labels must be non-empty.")
        invalid = sorted(set(subgroup) - set(PACO2_SUBGROUP_ORDER))
        if invalid:
            raise ValueError(
                "Prepared PaCO2 subgroup labels must be one of "
                f"{list(PACO2_SUBGROUP_ORDER)}; found {invalid}."
            )
        prepared["subgroup"] = subgroup
        return prepared
    validate_paco2_columns(data)
    filtered = _retained_paco2_rows(data)
    filtered["subgroup"] = assign_paco2_subgroup(filtered)
    return filtered


def assign_paco2_subgroup(data: pd.DataFrame) -> pd.Series:
    """Assign mutually exclusive PaCO2 subgroup labels."""

    validate_paco2_columns(data)
    is_amb = _binary_assignment_flag(data, "is_amb")
    is_emer = _binary_assignment_flag(data, "is_emer")
    is_inp = _binary_assignment_flag(data, "is_inp")
    cc_time = _binary_assignment_flag(data, "cc_time")

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
    prepared = prepare_paco2_distribution(data)
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
    prior["group"] = prior["group"].astype("string").str.strip().str.lower()
    if prior["group"].isna().any() or prior["group"].eq("").any():
        raise ValueError("Prior group labels must be non-empty.")
    prior["paco2_bin"] = pd.to_numeric(prior["paco2_bin"], errors="coerce")
    prior["weight"] = pd.to_numeric(prior["weight"], errors="coerce")
    if "count" in prior.columns:
        prior["count"] = pd.to_numeric(prior["count"], errors="coerce")
    prior["paco2_bin"] = validate_paco2_values(prior["paco2_bin"])
    if not np.all(np.isfinite(prior["weight"])):
        raise ValueError("Non-finite weights in prior.")
    if "count" in prior.columns:
        if not np.all(np.isfinite(prior["count"])):
            raise ValueError("Non-finite counts in prior.")
        if np.any(prior["count"] < 0):
            raise ValueError("Prior counts must be non-negative.")
    if np.any(prior["weight"] < 0):
        raise ValueError("Prior weights must be non-negative.")
    groups = set(prior["group"])
    allowed_groups = set(PACO2_PRIOR_GROUPS)
    invalid_groups = groups - allowed_groups
    if invalid_groups:
        raise ValueError(f"Unknown prior groups: {sorted(invalid_groups)}")
    missing_groups = allowed_groups - groups
    if missing_groups:
        raise ValueError(f"Prior bins missing groups: {sorted(missing_groups)}")
    weight_sums = prior.groupby("group")["weight"].sum()
    if not np.allclose(weight_sums.to_numpy(dtype=float), 1.0, atol=1e-6):
        raise ValueError("Prior weights must sum to 1 within each group.")
    return prior


def prior_distribution_from_bins(
    prior_bins: pd.DataFrame, group: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return discrete prior support and normalized mass for one PaCO2 group."""

    subset = prior_bins.loc[prior_bins["group"] == group].sort_values("paco2_bin")
    if subset.empty:
        raise ValueError(f"No binned priors available for group '{group}'.")
    values = validate_paco2_values(subset["paco2_bin"])
    if "count" in subset.columns:
        counts = subset["count"].to_numpy(dtype=float)
        total_count = float(np.sum(counts))
        if total_count <= 0:
            raise ValueError(f"Prior counts must be positive for group '{group}'.")
        weights = counts / total_count
    else:
        weights = subset["weight"].to_numpy(dtype=float)
        total_weight = float(np.sum(weights))
        if total_weight <= 0:
            raise ValueError(f"Prior weights must be positive for group '{group}'.")
        weights = weights / total_weight
    return values, weights


def prior_values_from_bins(prior_bins: pd.DataFrame, group: str) -> np.ndarray:
    """Expand binned PaCO2 prior counts into empirical prior values."""

    subset = prior_bins.loc[prior_bins["group"] == group]
    if subset.empty:
        raise ValueError(f"No binned priors available for group '{group}'.")
    if "count" not in subset.columns:
        raise ValueError("Cannot expand weight-only priors without a count column.")
    values = validate_paco2_values(subset["paco2_bin"])
    return np.repeat(
        values,
        subset["count"].to_numpy(dtype=int),
    )


def validate_paco2_columns(data: pd.DataFrame) -> None:
    """Validate columns required to assign PaCO2 analysis subgroups."""

    missing = PACO2_REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing PaCO2 columns: {sorted(missing)}")


def _retained_paco2_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Drop genuinely missing PaCO2 rows and validate every retained value."""

    retained = data.loc[data["paco2"].notna()].copy()
    retained["paco2"] = validate_paco2_values(retained["paco2"])
    return retained


def _binary_assignment_flag(data: pd.DataFrame, column: str) -> pd.Series:
    """Return one PaCO2 assignment flag without truncating nonbinary values."""

    raw = data[column]
    values = pd.to_numeric(raw, errors="coerce")
    if (raw.notna() & values.isna()).any():
        raise ValueError(f"PaCO2 assignment flag `{column}` must be numeric and binary (0/1).")
    values = values.fillna(0)
    if not np.all(np.isfinite(values)) or not values.isin((0, 1)).all():
        raise ValueError(f"PaCO2 assignment flag `{column}` must be binary (0/1).")
    return values.astype(int)
