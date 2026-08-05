from __future__ import annotations

import pandas as pd
import pytest

from tcco2_accuracy.core._params import select_group_params


def _grouped_params() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["main", "lft", "arf", "icu"],
            "delta": [0.0, 1.0, 2.0, 3.0],
            "sigma2": [1.0, 1.0, 1.0, 1.0],
            "tau2": [0.0, 0.0, 0.0, 0.0],
        }
    )


def test_select_group_params_maps_group_and_records_provenance_without_mutation() -> None:
    params = _grouped_params()

    selected = select_group_params(params, "pft", reset_index=True)

    assert selected["group"].tolist() == ["lft"]
    assert selected["requested_group"].tolist() == ["pft"]
    assert selected["parameter_group_used"].tolist() == ["lft"]
    assert "requested_group" not in params.columns
    assert "parameter_group_used" not in params.columns


def test_select_group_params_missing_group_fails_closed_by_default() -> None:
    with pytest.raises(ValueError, match="resolved group 'unknown'.*available groups"):
        select_group_params(_grouped_params(), "unknown")


def test_select_group_params_explicit_main_fallback_selects_main_only() -> None:
    selected = select_group_params(_grouped_params(), "unknown", fallback="main")

    assert selected["group"].tolist() == ["main"]
    assert selected["requested_group"].tolist() == ["unknown"]
    assert selected["parameter_group_used"].tolist() == ["main"]


def test_select_group_params_explicit_main_fallback_requires_main() -> None:
    params = _grouped_params().query("group != 'main'")

    with pytest.raises(ValueError, match="fallback requested group 'main'.*unavailable"):
        select_group_params(params, "unknown", fallback="main")


def test_select_group_params_rejects_unknown_fallback_policy() -> None:
    with pytest.raises(ValueError, match="Unknown parameter fallback policy"):
        select_group_params(_grouped_params(), "pft", fallback="all")  # type: ignore[arg-type]


def test_select_group_params_accepts_explicit_ungrouped_single_model() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [1.0], "tau2": [0.0]})

    selected = select_group_params(params, "pft")

    assert selected["requested_group"].tolist() == ["pft"]
    assert selected["parameter_group_used"].tolist() == ["single_model"]
