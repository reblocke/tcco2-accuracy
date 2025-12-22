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
    rdata_path = root / "Data" / "Conway Thorax supplement and code" / "data.Rdata"
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
    assert df.shape[0] > 50
    assert df["study_id"].nunique() >= 50
