"""Internal helpers for Conway parameter group routing."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import pandas as pd

from .constants import CONWAY_SUBGROUP_FLAGS
from .utils import validate_params_df

PACO2_TO_CONWAY_GROUP: dict[str, str] = {
    "pft": "lft",
    "ed_inp": "arf",
    "icu": "icu",
    # "All" uses Conway main-analysis parameters (all studies).
    "all": "main",
}

ParameterFallback = Literal["error", "main"]


def resolve_conway_group(
    subgroup: str,
    available_groups: Iterable[object] | None = None,
    *,
    map_all_to_main: bool = False,
) -> str:
    """Resolve a PaCO2 subgroup label to the matching Conway parameter group."""

    subgroup_key = str(subgroup)
    if map_all_to_main and subgroup_key == "all":
        return "main"
    if available_groups is not None:
        available = {str(group) for group in available_groups}
        if subgroup_key in available:
            return subgroup_key
    return PACO2_TO_CONWAY_GROUP.get(subgroup_key, subgroup_key)


def select_group_params(
    params: pd.DataFrame,
    subgroup: str,
    *,
    validate: bool = False,
    reset_index: bool = False,
    fallback: ParameterFallback = "error",
    map_all_to_main: bool = False,
) -> pd.DataFrame:
    """Return parameter rows for a PaCO2 subgroup with explicit routing provenance.

    Grouped parameter tables fail closed when the requested group is unavailable.
    Callers may explicitly request the pooled ``main`` group as a fallback; rows
    from unrelated groups are never silently combined. A table without a
    ``group`` column is treated as one deliberately supplied model.
    """

    if fallback not in {"error", "main"}:
        raise ValueError(f"Unknown parameter fallback policy: {fallback}")

    selected = validate_params_df(params) if validate else params
    if "group" not in selected.columns:
        return _annotate_selection(
            selected,
            requested_group=str(subgroup),
            parameter_group_used="single_model",
            reset_index=reset_index,
        )

    group_values = selected["group"].astype(str)
    group_key = resolve_conway_group(
        subgroup,
        available_groups=group_values,
        map_all_to_main=map_all_to_main,
    )
    group_params = selected[group_values == group_key]
    if group_params.empty:
        available = sorted(set(group_values))
        if fallback == "main":
            group_key = "main"
            group_params = selected[group_values == group_key]
            if group_params.empty:
                raise ValueError(
                    "Parameter fallback requested group 'main', but it is unavailable; "
                    f"requested subgroup='{subgroup}', available groups={available}."
                )
        else:
            raise ValueError(
                "No parameters found for requested subgroup "
                f"'{subgroup}' (resolved group '{group_key}'); available groups={available}. "
                "Supply the required group or explicitly set fallback='main'."
            )
    return _annotate_selection(
        group_params,
        requested_group=str(subgroup),
        parameter_group_used=group_key,
        reset_index=reset_index,
    )


def select_conway_studies_for_subgroup(studies: pd.DataFrame, subgroup: str) -> pd.DataFrame:
    """Return canonical Conway study rows for a PaCO2 subgroup."""

    group_key = resolve_conway_group(subgroup)
    if group_key == "main":
        subset = studies
    else:
        flag = CONWAY_SUBGROUP_FLAGS.get(group_key)
        if flag is None:
            allowed = sorted({"main", *CONWAY_SUBGROUP_FLAGS})
            raise ValueError(
                f"Unknown Conway subgroup '{subgroup}' (resolved group '{group_key}'); "
                f"expected one of {allowed} or a mapped PaCO2 subgroup."
            )
        subset = studies[studies[flag].astype(bool)]
    if subset.empty:
        raise ValueError(f"No studies available for Conway group '{group_key}'.")
    return subset


def _maybe_reset_index(frame: pd.DataFrame, reset_index: bool) -> pd.DataFrame:
    return frame.reset_index(drop=True) if reset_index else frame


def _annotate_selection(
    frame: pd.DataFrame,
    *,
    requested_group: str,
    parameter_group_used: str,
    reset_index: bool,
) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["requested_group"] = requested_group
    annotated["parameter_group_used"] = parameter_group_used
    return _maybe_reset_index(annotated, reset_index)
