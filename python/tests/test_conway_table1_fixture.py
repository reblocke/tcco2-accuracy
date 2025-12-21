from __future__ import annotations

import csv
import math
from pathlib import Path


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "conway_table1.csv"
EXPECTED_COLUMNS = [
    "population",
    "studies",
    "n_pairs",
    "n_participants",
    "bias",
    "sd",
    "tau2",
    "loa_l",
    "loa_u",
    "ci_l",
    "ci_u",
]


def _load_fixture() -> tuple[list[str], list[dict[str, str]]]:
    with FIXTURE_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return reader.fieldnames or [], rows


def test_conway_table1_fixture_columns() -> None:
    fieldnames, rows = _load_fixture()
    assert fieldnames == EXPECTED_COLUMNS
    assert rows


def test_conway_table1_fixture_loa_bounds_present() -> None:
    _, rows = _load_fixture()
    required_populations = {
        "Main analysis",
        "ICU",
        "Acute respiratory failure",
        "Outpatients requiring lung function tests",
    }
    assert required_populations.issubset({row["population"] for row in rows})
    for row in rows:
        loa_l = float(row["loa_l"])
        loa_u = float(row["loa_u"])
        assert math.isfinite(loa_l)
        assert math.isfinite(loa_u)
