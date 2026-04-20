"""Internal helpers for Conway parameter group routing."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

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
    warn_on_fallback: bool = True,
    map_all_to_main: bool = False,
) -> pd.DataFrame:
    """Return parameter rows for a PaCO2 subgroup, falling back to all rows."""

    selected = validate_params_df(params) if validate else params
    if "group" not in selected.columns:
        return _maybe_reset_index(selected, reset_index)

    group_values = selected["group"].astype(str)
    group_key = resolve_conway_group(
        subgroup,
        available_groups=group_values,
        map_all_to_main=map_all_to_main,
    )
    group_params = selected[group_values == group_key]
    if group_params.empty:
        if warn_on_fallback:
            warnings.warn(
                f"No parameters found for subgroup '{subgroup}'; falling back to all params.",
                UserWarning,
            )
        return _maybe_reset_index(selected, reset_index)
    return _maybe_reset_index(group_params, reset_index)


def select_conway_studies_for_subgroup(studies: pd.DataFrame, subgroup: str) -> pd.DataFrame:
    """Return canonical Conway study rows for a PaCO2 subgroup."""

    group_key = resolve_conway_group(subgroup)
    flag = CONWAY_SUBGROUP_FLAGS.get(group_key)
    if flag is None:
        subset = studies
    else:
        subset = studies[studies[flag].astype(bool)]
    if subset.empty:
        raise ValueError(f"No studies available for Conway group '{group_key}'.")
    return subset


def _maybe_reset_index(frame: pd.DataFrame, reset_index: bool) -> pd.DataFrame:
    return frame.reset_index(drop=True) if reset_index else frame
