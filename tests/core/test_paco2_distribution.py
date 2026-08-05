from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcco2_accuracy.data import (
    DEFAULT_PACO2_QUANTILES,
    INSILICO_PACO2_PATH,
    PACO2_REQUIRED_COLUMNS,
    PACO2_SUBGROUP_ORDER,
    load_paco2_distribution,
    paco2_subgroup_summary,
    prepare_paco2_distribution,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "paco2_distribution_summary.csv"


@pytest.fixture()
def paco2_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "paco2": [40.0, 42.0, 45.0, 50.0, 55.0, 60.0],
            "is_amb": [1, 1, 0, 0, 0, 0],
            "is_emer": [0, 0, 1, 0, 0, 1],
            "is_inp": [0, 0, 1, 1, 1, 0],
            "cc_time": [0, 0, 0, 1, 1, 0],
        }
    )


def test_paco2_required_columns(paco2_data: pd.DataFrame) -> None:
    missing = PACO2_REQUIRED_COLUMNS - set(paco2_data.columns)
    assert not missing


def test_paco2_subgroup_membership_exclusive(paco2_data: pd.DataFrame) -> None:
    prepared = prepare_paco2_distribution(paco2_data)
    counts = prepared["subgroup"].value_counts()
    assert set(counts.index).issubset(set(PACO2_SUBGROUP_ORDER))
    assert prepared["subgroup"].notna().all()
    assert counts.sum() == prepared.shape[0]


def test_paco2_units_mmhg(paco2_data: pd.DataFrame) -> None:
    summary = paco2_subgroup_summary(paco2_data, quantiles=DEFAULT_PACO2_QUANTILES)
    medians = summary.set_index("group")["paco2_q500"]
    assert medians.loc["pft"] == pytest.approx(41.0)
    assert medians.loc["ed_inp"] == pytest.approx(52.5)
    assert medians.loc["icu"] == pytest.approx(52.5)


def test_paco2_distribution_summary_matches_fixture() -> None:
    if not INSILICO_PACO2_PATH.exists():
        pytest.skip("In-silico PaCO2 .dta not available.")
    summary = paco2_subgroup_summary(load_paco2_distribution(), quantiles=DEFAULT_PACO2_QUANTILES)
    expected = pd.read_csv(FIXTURE_PATH)
    pd.testing.assert_frame_equal(summary, expected, check_exact=False, atol=0.01)


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, -np.inf, "not-numeric"])
def test_paco2_distribution_rejects_invalid_retained_values(
    paco2_data: pd.DataFrame, value: object
) -> None:
    paco2_data["paco2"] = paco2_data["paco2"].astype(object)
    paco2_data.loc[0, "paco2"] = value

    with pytest.raises(ValueError, match="PaCO2 values"):
        prepare_paco2_distribution(paco2_data)


def test_paco2_distribution_drops_genuinely_missing_values(
    paco2_data: pd.DataFrame,
) -> None:
    paco2_data.loc[0, "paco2"] = np.nan

    prepared = prepare_paco2_distribution(paco2_data)

    assert prepared.shape[0] == paco2_data.shape[0] - 1
    assert prepared["paco2"].notna().all()


def test_paco2_distribution_has_no_hard_upper_limit(paco2_data: pd.DataFrame) -> None:
    paco2_data.loc[0, "paco2"] = 500.0

    prepared = prepare_paco2_distribution(paco2_data)

    assert 500.0 in prepared["paco2"].to_numpy()


@pytest.mark.parametrize("subgroup", [None, "", "unknown"])
def test_prepared_paco2_distribution_rejects_invalid_retained_subgroups(
    subgroup: str | None,
) -> None:
    prepared = pd.DataFrame({"paco2": [40.0], "subgroup": [subgroup]})

    with pytest.raises(ValueError, match="subgroup labels"):
        prepare_paco2_distribution(prepared)


def test_prepared_paco2_distribution_normalizes_valid_labels_after_missing_drop() -> None:
    prepared = pd.DataFrame(
        {
            "paco2": [40.0, np.nan],
            "subgroup": [" PFT ", None],
        }
    )

    result = prepare_paco2_distribution(prepared)

    assert result[["paco2", "subgroup"]].to_dict(orient="records") == [
        {"paco2": 40.0, "subgroup": "pft"}
    ]


@pytest.mark.parametrize("value", [1.5, "invalid", np.inf])
def test_paco2_distribution_rejects_nonbinary_assignment_flags(
    paco2_data: pd.DataFrame, value: object
) -> None:
    paco2_data["is_amb"] = paco2_data["is_amb"].astype(object)
    paco2_data.loc[0, "is_amb"] = value

    with pytest.raises(ValueError, match="is_amb.*binary"):
        prepare_paco2_distribution(paco2_data)


def test_paco2_distribution_treats_missing_assignment_flag_as_false(
    paco2_data: pd.DataFrame,
) -> None:
    paco2_data.loc[0, "is_amb"] = np.nan

    with pytest.raises(ValueError, match="Unclassified"):
        prepare_paco2_distribution(paco2_data)
