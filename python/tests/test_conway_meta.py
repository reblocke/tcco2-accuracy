from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tcco2_accuracy.conway_meta import conway_group_summary
from tcco2_accuracy.data import load_conway_group


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "conway_table1.csv"
GROUP_MAP = {
    "Main analysis": "main",
    "ICU": "icu",
    "Acute respiratory failure": "arf",
    "Outpatients requiring lung function tests": "lft",
}


def _load_fixture() -> dict[str, dict[str, str]]:
    with FIXTURE_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["population"]: row for row in reader}


@pytest.mark.parametrize("population", list(GROUP_MAP))
def test_conway_meta_matches_table1(population: str) -> None:
    fixture = _load_fixture()[population]
    data = load_conway_group(GROUP_MAP[population])
    summary = conway_group_summary(data)

    assert summary.bias == pytest.approx(float(fixture["bias"]), abs=0.1)
    assert summary.sd == pytest.approx(float(fixture["sd"]), abs=0.1)
    assert summary.tau2 == pytest.approx(float(fixture["tau2"]), abs=0.1)
    assert summary.loa_l == pytest.approx(float(fixture["loa_l"]), abs=0.1)
    assert summary.loa_u == pytest.approx(float(fixture["loa_u"]), abs=0.1)
    assert summary.ci_l == pytest.approx(float(fixture["ci_l"]), abs=0.1)
    assert summary.ci_u == pytest.approx(float(fixture["ci_u"]), abs=0.1)
    assert summary.n_pairs == int(fixture["n_pairs"])
    assert summary.n_participants == int(fixture["n_participants"])
