"""I/O helpers for Conway meta-analysis inputs and PaCO2 distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .utils import quantile_key
from .validate_inputs import validate_conway_studies_df


REPO_ROOT = Path(__file__).resolve().parents[3]
CONWAY_DATA_PATH = REPO_ROOT / "Data" / "conway_studies.csv"
CONWAY_XLSX_PATH = REPO_ROOT / "Data" / "conway_studies.xlsx"
CONWAY_LEGACY_DTA_PATH = REPO_ROOT / "Data" / "data.dta"
INSILICO_PACO2_PATH = REPO_ROOT / "Data" / "In Silico TCCO2 Database.dta"

CONWAY_SUBGROUP_FLAGS = {
    "icu": "is_icu",
    "arf": "is_arf",
    "lft": "is_lft",
}

PACO2_REQUIRED_COLUMNS = {"paco2", "is_amb", "is_emer", "is_inp", "cc_time"}
PACO2_SUBGROUP_ORDER = ("pft", "ed_inp", "icu")
DEFAULT_PACO2_QUANTILES: tuple[float, ...] = (0.025, 0.25, 0.5, 0.75, 0.975)


def load_conway_studies(path: Path | None = None) -> pd.DataFrame:
    """Return the canonical Conway study input table."""

    if path is None:
        if CONWAY_DATA_PATH.exists():
            path = CONWAY_DATA_PATH
        elif CONWAY_XLSX_PATH.exists():
            path = CONWAY_XLSX_PATH
        else:
            path = CONWAY_LEGACY_DTA_PATH
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(path)
    elif path.suffix.lower() == ".dta":
        data = pd.read_stata(path)
    else:
        raise ValueError(f"Unsupported Conway table format: {path.suffix}")

    canonical = _canonicalize_conway_table(data)
    validate_conway_studies_df(canonical)
    return canonical


def load_conway_data(path: Path | None = None) -> pd.DataFrame:
    """Return Conway study-level inputs for meta-analysis."""

    studies = load_conway_studies(path)
    analysis = _to_conway_analysis(studies)
    analysis.attrs["group"] = "main"
    return analysis


def prepare_conway_meta_inputs(studies: pd.DataFrame) -> pd.DataFrame:
    """Prepare meta-analysis columns from a canonical study table."""

    canonical = _canonicalize_conway_table(studies)
    validate_conway_studies_df(canonical)
    return _to_conway_analysis(canonical)


def load_conway_group(group: str, path: Path | None = None) -> pd.DataFrame:
    """Load a Conway subgroup by name."""

    key = group.strip().lower()
    studies = load_conway_studies(path)
    if key in {"main", "all"}:
        analysis = _to_conway_analysis(studies)
        analysis.attrs["group"] = "main"
        return analysis
    if key not in CONWAY_SUBGROUP_FLAGS:
        raise ValueError(f"Unknown Conway subgroup: {group}")
    flag = CONWAY_SUBGROUP_FLAGS[key]
    subset = studies[studies[flag].astype(bool)]
    analysis = _to_conway_analysis(subset)
    analysis.attrs["group"] = key
    return analysis


def _canonicalize_conway_table(data: pd.DataFrame) -> pd.DataFrame:
    canonical = data.copy()
    rename_map = {
        "study": "study_id",
        "n": "n_pairs",
        "n_2": "n_participants",
        "icu1": "is_icu",
        "icu_group": "is_icu",
        "respiratory_lft": "is_lft",
        "respiratory_lft_6": "is_lft",
        "ed_arf_7": "is_arf",
        "ed_arf": "is_arf",
        "ed_inp_group": "is_arf",
        "pft_group": "is_lft",
    }
    canonical = canonical.rename(columns={k: v for k, v in rename_map.items() if k in canonical.columns})

    for flag in CONWAY_SUBGROUP_FLAGS.values():
        if flag in canonical.columns:
            canonical[flag] = canonical[flag].fillna(0).astype(int)

    if "sd" not in canonical.columns and "s2" in canonical.columns:
        canonical["sd"] = np.sqrt(pd.to_numeric(canonical["s2"], errors="coerce"))
    if "s2" not in canonical.columns and "sd" in canonical.columns:
        canonical["s2"] = pd.to_numeric(canonical["sd"], errors="coerce") ** 2

    if "study_id" in canonical.columns:
        # Strip cohort qualifiers so main-analysis counts align to Table 1 citations.
        canonical["study_base"] = (
            canonical["study_id"]
            .astype(str)
            .str.replace(r"\s*\([^)]*\)\s*$", "", regex=True)
            .str.strip()
        )

    return canonical


def _to_conway_analysis(studies: pd.DataFrame) -> pd.DataFrame:
    data = studies.copy()
    rename_map = {
        "study_id": "study",
        "n_pairs": "n",
        "n_participants": "n_2",
    }
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})
    if "s2" not in data.columns:
        data["s2"] = pd.to_numeric(data["sd"], errors="coerce") ** 2
    return data


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
