"""I/O helpers for Conway meta-analysis inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
CONWAY_DATA_PATH = REPO_ROOT / "Conway Meta" / "data.dta"

CONWAY_SUBGROUPS: dict[str, list[str]] = {
    "icu": [
        "Baulig 2007",
        "Bendjelid 2005",
        "Berlowitz 2011",
        "Bolliger 2007 (TOSCA - ICU)",
        "Chakravarthy 2010",
        "Chhajed 2012",
        "Henao-Brasseur 2016",
        "Hinkelbein 2008",
        "Hirabayashi 2009 (ventilated)",
        "Johnson 2008",
        "Rodriguez 2006",
        "Roediger 2011 (ear)",
        "Rosier 2014",
        "Senn 2005",
        "vanOppen 2015",
        "Vivien 2006",
    ],
    "arf": [
        "Bobbia 2015",
        "Delerme 2012",
        "Gancel 2011",
        "Kelly 2011",
        "Kim 2014 (normotensive)",
        "Kim 2014 (hypotensive)",
        "Lermuzeaux 2016",
        "McVicar 2009",
        "Nicolini 2011",
        "Perrin 2011",
        "Peschanski 2016",
        "Piquilloud 2013",
        "Ruiz 2016",
        "Storre 2007",
    ],
    "lft": [
        "Chhajed 2010",
        "Domingo 2006",
        "Maniscalco 2008",
        "Ekkerkamp 2015",
    ],
}

EXTRA_STUDIES: dict[str, dict[str, float | str]] = {
    "Bolliger 2007 (TOSCA - ICU)": {
        "study": "Bolliger 2007 (TOSCA - ICU)",
        "n": 49.0,
        "n_2": 49.0,
        "c": 1.0,
        "bias": -2.175,
        "lower95": -11.55,
        "upper95": 7.2,
        "s2": 22.878651,
    }
}


def load_conway_data(path: Path | None = None) -> pd.DataFrame:
    """Return the Conway study-level dataset."""

    return pd.read_stata(path or CONWAY_DATA_PATH)


def load_conway_group(group: str, path: Path | None = None) -> pd.DataFrame:
    """Load a Conway subgroup by name."""

    key = group.strip().lower()
    if key in {"main", "all"}:
        return load_conway_data(path)
    if key not in CONWAY_SUBGROUPS:
        raise ValueError(f"Unknown Conway subgroup: {group}")
    data = load_conway_data(path)
    data = data[data["study"].isin(CONWAY_SUBGROUPS[key])]
    return _with_extra_studies(data, CONWAY_SUBGROUPS[key])


def _with_extra_studies(data: pd.DataFrame, studies: Iterable[str]) -> pd.DataFrame:
    missing = [
        EXTRA_STUDIES[name]
        for name in studies
        if name in EXTRA_STUDIES and name not in set(data["study"])
    ]
    if not missing:
        return data
    return pd.concat([data, pd.DataFrame(missing)], ignore_index=True)
