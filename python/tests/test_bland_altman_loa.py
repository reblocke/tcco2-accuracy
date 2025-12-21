from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tcco2_accuracy.bland_altman import loa_bounds


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "conway_table1.csv"


def _load_main_analysis_row() -> dict[str, str]:
    with FIXTURE_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["population"] == "Main analysis":
                return row
    raise ValueError("Main analysis row missing from fixture")


def test_loa_matches_conway_main_analysis() -> None:
    row = _load_main_analysis_row()
    delta = float(row["bias"])
    sigma = float(row["sd"])
    tau2 = float(row["tau2"])
    expected_lower = float(row["loa_l"])
    expected_upper = float(row["loa_u"])

    lower, upper = loa_bounds(delta, sigma, tau2)

    assert lower == pytest.approx(expected_lower, abs=0.1)
    assert upper == pytest.approx(expected_upper, abs=0.1)
