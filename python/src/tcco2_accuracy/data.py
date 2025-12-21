"""I/O helpers for Conway meta-analysis inputs and PaCO2 distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .utils import quantile_key


REPO_ROOT = Path(__file__).resolve().parents[3]
CONWAY_DATA_PATH = REPO_ROOT / "Conway Meta" / "data.dta"
INSILICO_PACO2_PATH = REPO_ROOT / "Data" / "In Silico TCCO2 Database.dta"

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

PACO2_REQUIRED_COLUMNS = {"paco2", "is_amb", "is_emer", "is_inp", "cc_time"}
PACO2_SUBGROUP_ORDER = ("pft", "ed_inp", "icu")
DEFAULT_PACO2_QUANTILES: tuple[float, ...] = (0.025, 0.25, 0.5, 0.75, 0.975)


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


def load_paco2_distribution(path: Path | None = None) -> pd.DataFrame:
    """Return the in-silico PaCO2 distribution."""

    return pd.read_stata(path or INSILICO_PACO2_PATH, convert_categoricals=False)


def prepare_paco2_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """Filter PaCO2 rows and assign subgroup labels."""

    _validate_paco2_columns(data)
    filtered = data.loc[data["paco2"].notna()].copy()
    filtered["subgroup"] = assign_paco2_subgroup(filtered)
    return filtered


def assign_paco2_subgroup(data: pd.DataFrame) -> pd.Series:
    """Assign mutually exclusive PaCO2 subgroup labels."""

    _validate_paco2_columns(data)
    is_amb = data["is_amb"].fillna(0).astype(int)
    is_emer = data["is_emer"].fillna(0).astype(int)
    is_inp = data["is_inp"].fillna(0).astype(int)
    cc_time = data["cc_time"].fillna(0).astype(int)

    pft_mask = is_amb == 1
    icu_mask = (is_inp == 1) & (cc_time == 1) & (is_emer == 0) & (is_amb == 0)
    ed_inp_mask = (is_emer == 1) | (is_inp == 1)

    subgroup = pd.Series(
        np.select([pft_mask, icu_mask, ed_inp_mask], ["pft", "icu", "ed_inp"], default=pd.NA),
        index=data.index,
        dtype="object",
    )
    if subgroup.isna().any():
        raise ValueError("Unclassified PaCO2 records after subgroup assignment.")
    return subgroup


def paco2_subgroup_summary(
    data: pd.DataFrame,
    quantiles: Sequence[float] = DEFAULT_PACO2_QUANTILES,
) -> pd.DataFrame:
    """Summarize subgroup counts and PaCO2 quantiles."""

    if "subgroup" in data.columns:
        prepared = data.loc[data["paco2"].notna()].copy()
    else:
        prepared = prepare_paco2_distribution(data)

    quantile_list = list(quantiles)
    quantile_columns = [quantile_key("paco2", q) for q in quantile_list]
    rows: list[dict[str, float | int | str]] = []
    for group in PACO2_SUBGROUP_ORDER:
        subset = prepared[prepared["subgroup"] == group]
        if subset.empty:
            continue
        q_values = subset["paco2"].quantile(quantile_list, interpolation="linear")
        row: dict[str, float | int | str] = {"group": group, "count": int(subset.shape[0])}
        for q in quantile_list:
            row[quantile_key("paco2", q)] = float(q_values.loc[q])
        rows.append(row)

    return pd.DataFrame(rows, columns=["group", "count", *quantile_columns])
def _validate_paco2_columns(data: pd.DataFrame) -> None:
    missing = PACO2_REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing PaCO2 columns: {sorted(missing)}")
