from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcco2_accuracy.core._params import select_conway_studies_for_subgroup
from tcco2_accuracy.core.conway_meta import prepare_conway_inputs
from tcco2_accuracy.data import CONWAY_DATA_PATH, prepare_conway_meta_inputs
from tcco2_accuracy.validate_inputs import (
    validate_conway_meta_inputs_df,
    validate_conway_studies_df,
    validate_thresholds,
)


def _valid_studies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_id": ["Study A", "Study B"],
            "bias": [1.2, -0.5],
            "sd": [2.0, 3.0],
            "s2": [4.0, 9.0],
            "n_pairs": [4, 6],
            "n_participants": [2, 3],
            "c": [2.0, 2.0],
            "is_icu": [1, 0],
            "is_arf": [1, 1],
            "is_lft": [0, 1],
        }
    )


def test_validate_conway_studies_ok() -> None:
    if not Path(CONWAY_DATA_PATH).exists():
        pytest.skip("Canonical Conway study table missing.")
    df = pd.read_csv(CONWAY_DATA_PATH)
    validate_conway_studies_df(df)


def test_validate_conway_studies_invalid() -> None:
    df = pd.DataFrame(
        {
            "study_id": ["bad"],
            "bias": [1.2],
            "sd": [1.5],
            "n_pairs": [-5],
            "n_participants": [10],
            "is_icu": [0],
            "is_arf": [0],
            "is_lft": [0],
        }
    )
    with pytest.raises(ValueError, match="n_pairs"):
        validate_conway_studies_df(df)

    df_missing = df.drop(columns=["sd"])
    with pytest.raises(ValueError, match="sd|s2"):
        validate_conway_studies_df(df_missing)


def test_validate_conway_studies_rejects_empty_table() -> None:
    with pytest.raises(ValueError, match="at least one study"):
        validate_conway_studies_df(_valid_studies().iloc[0:0])


@pytest.mark.parametrize(
    ("study_ids", "message"),
    [
        (["Study A", "   "], "non-empty"),
        (["Study A", " Study A "], "unique"),
        (["Study A", None], "non-empty"),
    ],
)
def test_validate_conway_studies_rejects_invalid_identifiers(
    study_ids: list[str | None], message: str
) -> None:
    studies = _valid_studies()
    studies["study_id"] = study_ids

    with pytest.raises(ValueError, match=message):
        validate_conway_studies_df(studies)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("n_participants", 1, "greater than 1"),
        ("n_participants", 2.5, "integer counts greater than 1"),
        ("n_pairs", 4.5, "positive integer counts"),
        ("n_pairs", 1, "greater than or equal"),
        ("c", np.nan, "Non-finite"),
        ("c", 0.5, "greater than or equal to 1"),
        ("c", 3.0, "n_pairs / n_participants"),
    ],
)
def test_validate_conway_studies_rejects_invalid_counts(
    column: str, value: float, message: str
) -> None:
    studies = _valid_studies()
    studies[column] = studies[column].astype(float)
    studies.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_conway_studies_df(studies)


def test_validate_conway_studies_requires_consistent_positive_variances() -> None:
    studies = _valid_studies()
    studies.loc[0, "s2"] = 4.01
    with pytest.raises(ValueError, match=r"sd\^2"):
        validate_conway_studies_df(studies)

    studies = _valid_studies()
    studies.loc[0, "s2"] = -1.0
    with pytest.raises(ValueError, match="positive"):
        validate_conway_studies_df(studies)

    studies = _valid_studies()
    studies.loc[0, "s2"] = 4.0 + 4.0e-10
    validate_conway_studies_df(studies)


@pytest.mark.parametrize("omitted_column", ["sd", "s2"])
def test_validate_conway_studies_accepts_one_variance_representation(
    omitted_column: str,
) -> None:
    validate_conway_studies_df(_valid_studies().drop(columns=omitted_column))


def test_prepare_conway_inputs_derives_omitted_repeated_measure_count() -> None:
    analysis = prepare_conway_meta_inputs(_valid_studies().drop(columns="c"))

    prepared = prepare_conway_inputs(analysis)

    np.testing.assert_allclose(prepared["c"], prepared["n"] / prepared["n_2"])


def test_prepare_conway_inputs_normalizes_valid_numeric_strings() -> None:
    analysis = prepare_conway_meta_inputs(_valid_studies())
    numeric_columns = ("bias", "sd", "s2", "n", "n_2", "c")
    for column in numeric_columns:
        analysis[column] = analysis[column].astype(str)

    prepared = prepare_conway_inputs(analysis)

    assert all(pd.api.types.is_numeric_dtype(prepared[column]) for column in numeric_columns)
    assert np.isfinite(prepared[["s2_adj", "v_bias", "logs2", "v_logs2"]]).all().all()


def test_validate_conway_studies_permits_overlapping_subgroups() -> None:
    studies = _valid_studies()
    studies.loc[0, ["is_icu", "is_arf", "is_lft"]] = 1

    validate_conway_studies_df(studies)

    analysis = prepare_conway_meta_inputs(studies)
    for subgroup in ("icu", "arf", "lft"):
        selected = select_conway_studies_for_subgroup(analysis, subgroup)
        assert "Study A" in selected["study"].to_numpy()


@pytest.mark.parametrize("value", [np.nan, 1.5, "not-a-flag"])
def test_validate_conway_studies_rejects_invalid_subgroup_flags(value: object) -> None:
    studies = _valid_studies()
    studies["is_icu"] = studies["is_icu"].astype(object)
    studies.loc[0, "is_icu"] = value

    with pytest.raises(ValueError, match="is_icu"):
        validate_conway_studies_df(studies)


def test_prepare_conway_inputs_defensively_rejects_invalid_analysis_form() -> None:
    analysis = _valid_studies().rename(
        columns={"study_id": "study", "n_pairs": "n", "n_participants": "n_2"}
    )
    analysis.loc[1, "study"] = " Study A "

    with pytest.raises(ValueError, match="unique"):
        validate_conway_meta_inputs_df(analysis)
    with pytest.raises(ValueError, match="unique"):
        prepare_conway_inputs(analysis)


def test_validate_conway_meta_inputs_rejects_empty_subgroup() -> None:
    analysis = _valid_studies().rename(
        columns={"study_id": "study", "n_pairs": "n", "n_participants": "n_2"}
    )

    with pytest.raises(ValueError, match="at least one study"):
        validate_conway_meta_inputs_df(analysis.iloc[0:0])


def test_validate_thresholds_rejects_empty_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="At least one"):
        validate_thresholds([])
    for threshold in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="threshold"):
            validate_thresholds([threshold])


def test_validate_thresholds_has_no_unsupported_upper_bound() -> None:
    assert validate_thresholds([500.0]) == (500.0,)
