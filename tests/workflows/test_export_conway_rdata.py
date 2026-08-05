from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.export_conway_rdata import (
    _extract_study_ids,
    _finalize_table,
    _load_counts_table,
)
from tcco2_accuracy.validate_inputs import validate_conway_studies_df


def _valid_export_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_id": ["Study A"],
            "bias": [1.0],
            "sd": [2.0],
            "s2": [4.0],
            "n_pairs": [4.0],
            "n_participants": [2.0],
            "c": [2.0],
            "is_icu": [0],
            "is_arf": [0],
            "is_lft": [1],
        }
    )


def test_export_conway_rdata_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("pyreadr")
    root = Path(__file__).resolve().parents[2]
    rdata_path = root / "Data" / "data.Rdata"
    if not rdata_path.exists():
        pytest.skip("Conway data.Rdata not available.")
    script_path = root / "scripts" / "export_conway_rdata.py"
    if not script_path.exists():
        pytest.skip("Export script not available.")

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input",
            str(rdata_path),
            "--out-dir",
            str(tmp_path),
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    csv_path = tmp_path / "conway_studies.csv"
    df = pd.read_csv(csv_path)
    validate_conway_studies_df(df)
    assert df.shape[0] >= 75
    assert df["study_id"].nunique() >= 75
    assert df["bias"].notna().all()
    assert df["s2"].notna().all()
    assert df["n_pairs"].notna().all()
    assert df["n_participants"].notna().all()

    kim_row = df.loc[df["study_id"] == "Kim 2014 (hypotensive)"]
    assert not kim_row.empty
    assert int(kim_row.iloc[0]["is_arf"]) == 1

    bolliger_row = df.loc[df["study_id"] == "Bolliger 2007 (TOSCA - ICU)"]
    if not bolliger_row.empty:
        assert int(bolliger_row.iloc[0]["is_icu"]) == 1
        assert bolliger_row[["n_pairs", "n_participants", "c"]].notna().all().all()


@pytest.mark.parametrize("strict", [True, False])
def test_export_rejects_noninteger_counts_before_cast(strict: bool) -> None:
    canonical = _valid_export_table()
    canonical.loc[0, "n_pairs"] = 4.2

    with pytest.raises(ValueError, match="integer counts"):
        _finalize_table(canonical, strict=strict, allow_missing_counts=True)


@pytest.mark.parametrize("study_id", [None, "   "])
def test_rdata_study_id_extraction_rejects_missing_or_blank(
    study_id: str | None,
) -> None:
    source = pd.DataFrame({"study": [study_id]})

    with pytest.raises(ValueError, match="missing|blank"):
        _extract_study_ids(source, "main", strict=False)


@pytest.mark.parametrize("study_id", [None, "   "])
def test_count_table_rejects_missing_or_blank_study_ids(
    tmp_path: Path, study_id: str | None
) -> None:
    path = tmp_path / "counts.csv"
    pd.DataFrame(
        {
            "study_id": [study_id],
            "n_pairs": [4],
            "n_participants": [2],
            "c": [2.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing|blank"):
        _load_counts_table(path, strict=False)


def test_export_study_ids_are_normalized_before_duplicate_check(tmp_path: Path) -> None:
    path = tmp_path / "counts.csv"
    pd.DataFrame(
        {
            "study_id": ["Study A", " Study A "],
            "n_pairs": [4, 4],
            "n_participants": [2, 2],
            "c": [2.0, 2.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Duplicate study IDs"):
        _load_counts_table(path, strict=False)
