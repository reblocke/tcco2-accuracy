"""I/O helpers for Conway meta-analysis inputs and PaCO2 distributions."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .core.constants import (
    CONWAY_SUBGROUP_FLAGS,
    DEFAULT_PACO2_QUANTILES,
    PACO2_PRIOR_GROUPS,
    PACO2_PRIOR_REQUIRED_COLUMNS,
    PACO2_REQUIRED_COLUMNS,
    PACO2_SUBGROUP_ORDER,
)
from .core.paco2 import (
    assign_paco2_subgroup,
    build_paco2_prior_bins,
    paco2_subgroup_summary,
    prepare_paco2_distribution,
    prior_values_from_bins,
    validate_paco2_columns,
    validate_paco2_prior_bins,
)
from .core.validate_inputs import validate_conway_studies_df

__all__ = [
    "CONWAY_DATA_PATH",
    "CONWAY_LEGACY_DTA_PATH",
    "CONWAY_SUBGROUP_FLAGS",
    "CONWAY_XLSX_PATH",
    "DEFAULT_PACO2_QUANTILES",
    "INSILICO_PACO2_FALLBACK_PATHS",
    "INSILICO_PACO2_PATH",
    "PACO2_PRIOR_BINS_PATH",
    "PACO2_PRIOR_BINS_XLSX_PATH",
    "PACO2_PRIOR_GROUPS",
    "PACO2_PRIOR_REQUIRED_COLUMNS",
    "PACO2_REQUIRED_COLUMNS",
    "PACO2_SUBGROUP_ORDER",
    "PriorLoadError",
    "PriorLoadResult",
    "assign_paco2_subgroup",
    "build_paco2_prior_bins",
    "load_conway_data",
    "load_conway_group",
    "load_conway_studies",
    "load_default_paco2_prior",
    "load_paco2_distribution",
    "load_paco2_prior",
    "load_paco2_prior_bins",
    "load_paco2_prior_bins_bytes",
    "paco2_subgroup_summary",
    "prepare_conway_meta_inputs",
    "prepare_paco2_distribution",
    "prior_values_from_bins",
    "validate_paco2_columns",
    "validate_paco2_prior_bins",
]


def _resolve_repo_root() -> Path:
    """Return the repository root without depending on the working directory."""

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "Data").exists():
            return parent
    return current.parents[3]


REPO_ROOT = _resolve_repo_root()
CONWAY_DATA_PATH = REPO_ROOT / "Data" / "conway_studies.csv"
CONWAY_XLSX_PATH = REPO_ROOT / "Data" / "conway_studies.xlsx"
CONWAY_LEGACY_DTA_PATH = REPO_ROOT / "Data" / "data.dta"
INSILICO_PACO2_PATH = REPO_ROOT / "Data" / "In Silico TCCO2 Database.dta"
INSILICO_PACO2_FALLBACK_PATHS = (REPO_ROOT / "Data" / "in_silico_tcco2_db.dta",)
PACO2_PRIOR_BINS_PATH = REPO_ROOT / "Data" / "paco2_prior_bins.csv"
PACO2_PRIOR_BINS_XLSX_PATH = REPO_ROOT / "Data" / "paco2_prior_bins.xlsx"


@dataclass(frozen=True)
class PriorLoadError:
    """Structured error for missing PaCO2 prior data."""

    message: str
    paths_checked: tuple[Path, ...]


@dataclass(frozen=True)
class PriorLoadResult:
    """PaCO2 prior load result for UI inference."""

    values: np.ndarray | None
    bins: pd.DataFrame | None
    source: str | None
    paths_checked: tuple[Path, ...]
    error: PriorLoadError | None = None


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
    canonical = canonical.rename(
        columns={k: v for k, v in rename_map.items() if k in canonical.columns}
    )

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

    if path is None:
        for candidate in (INSILICO_PACO2_PATH, *INSILICO_PACO2_FALLBACK_PATHS):
            if candidate.exists():
                path = candidate
                break
        else:
            path = INSILICO_PACO2_PATH
    return pd.read_stata(path, convert_categoricals=False)


def load_paco2_prior_bins(path: Path) -> pd.DataFrame:
    """Load a binned PaCO2 prior table from CSV/XLSX."""

    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported prior bin format: {path.suffix}")
    return validate_paco2_prior_bins(data)


def load_paco2_prior_bins_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load a binned PaCO2 prior from uploaded bytes."""

    buffer = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        data = pd.read_csv(buffer)
    elif filename.lower().endswith((".xlsx", ".xls")):
        data = pd.read_excel(buffer)
    else:
        raise ValueError("Uploaded prior must be CSV/XLSX.")
    return validate_paco2_prior_bins(data)


def load_default_paco2_prior(
    subgroup: str,
    bins_path: Path | None = None,
) -> np.ndarray:
    """Load the repo-shipped binned prior for a subgroup.

    The default bins CSV exists to keep the UI and CI portable without
    depending on the full in-silico .dta.
    """

    bins_path = bins_path or PACO2_PRIOR_BINS_PATH
    if not bins_path.exists():
        raise FileNotFoundError(f"Default PaCO2 prior bins not found: {bins_path}")
    bins = load_paco2_prior_bins(bins_path)
    # "all" is a pooled prior weighted by subgroup sample sizes.
    return prior_values_from_bins(bins, subgroup.strip().lower())


def load_paco2_prior(
    subgroup: str,
    uploaded_bytes: bytes | None = None,
    uploaded_name: str | None = None,
    default_bins_path: Path | None = None,
    insilico_path: Path | None = None,
) -> PriorLoadResult:
    """Load PaCO2 prior values with UI-friendly precedence and metadata."""

    subgroup_key = subgroup.strip().lower()
    if subgroup_key not in PACO2_PRIOR_GROUPS:
        raise ValueError(f"Unknown PaCO2 prior group: {subgroup}")
    paths_checked: list[Path] = []

    if uploaded_bytes is not None:
        bins = load_paco2_prior_bins_bytes(uploaded_bytes, uploaded_name or "prior.csv")
        values = prior_values_from_bins(bins, subgroup_key)
        return PriorLoadResult(
            values=values,
            bins=bins,
            source="uploaded",
            paths_checked=tuple(paths_checked),
        )

    bins_path = default_bins_path or PACO2_PRIOR_BINS_PATH
    paths_checked.append(bins_path)
    if bins_path.exists():
        # Binned weights represent the empirical PaCO2 pretest distribution.
        bins = load_paco2_prior_bins(bins_path)
        # "all" is a pooled prior weighted by subgroup sample sizes.
        values = prior_values_from_bins(bins, subgroup_key)
        return PriorLoadResult(
            values=values,
            bins=bins,
            source="default_bins",
            paths_checked=tuple(paths_checked),
        )

    insilico_path = insilico_path or INSILICO_PACO2_PATH
    paths_checked.append(insilico_path)
    if insilico_path.exists():
        prepared = prepare_paco2_distribution(load_paco2_distribution(insilico_path))
        if subgroup_key == "all":
            # Pooled prior uses all subgroups weighted by their sample sizes.
            values = prepared["paco2"].to_numpy(dtype=float)
        else:
            values = prepared.loc[prepared["subgroup"] == subgroup_key, "paco2"].to_numpy(
                dtype=float
            )
        if values.size == 0:
            raise ValueError(f"No PaCO2 values found for subgroup '{subgroup_key}'.")
        return PriorLoadResult(
            values=values,
            bins=None,
            source="insilico_dta",
            paths_checked=tuple(paths_checked),
        )

    message = "PaCO2 prior data not found. Provide a binned prior CSV or the in-silico database."
    error = PriorLoadError(message=message, paths_checked=tuple(paths_checked))
    return PriorLoadResult(
        values=None,
        bins=None,
        source=None,
        paths_checked=tuple(paths_checked),
        error=error,
    )


_validate_paco2_prior_bins = validate_paco2_prior_bins
_prior_values_from_bins = prior_values_from_bins
_validate_paco2_columns = validate_paco2_columns
