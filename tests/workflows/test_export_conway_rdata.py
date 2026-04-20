from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tcco2_accuracy.validate_inputs import validate_conway_studies_df


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
