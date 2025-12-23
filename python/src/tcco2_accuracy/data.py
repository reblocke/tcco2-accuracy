"""I/O helpers for Conway meta-analysis inputs and PaCO2 distributions."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .utils import quantile_key
from .validate_inputs import validate_conway_studies_df


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
PACO2_PRIOR_BINS_PATH = REPO_ROOT / "Data" / "paco2_prior_bins.csv"
PACO2_PRIOR_BINS_XLSX_PATH = REPO_ROOT / "Data" / "paco2_prior_bins.xlsx"

CONWAY_SUBGROUP_FLAGS = {
    "icu": "is_icu",
    "arf": "is_arf",
    "lft": "is_lft",
}

PACO2_REQUIRED_COLUMNS = {"paco2", "is_amb", "is_emer", "is_inp", "cc_time"}
PACO2_SUBGROUP_ORDER = ("pft", "ed_inp", "icu")
PACO2_PRIOR_GROUPS = ("pft", "ed_inp", "icu", "all")
PACO2_PRIOR_REQUIRED_COLUMNS = {"group", "paco2_bin", "count", "weight"}
DEFAULT_PACO2_QUANTILES: tuple[float, ...] = (0.025, 0.25, 0.5, 0.75, 0.975)


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


def build_paco2_prior_bins(
    data: pd.DataFrame,
    bin_width: float = 1.0,
) -> pd.DataFrame:
    """Return binned PaCO2 priors for each subgroup plus pooled "all".

    The binned prior is shipped with the repo for portability in Streamlit
    deployments; it captures the empirical pretest PaCO2 distribution without
    requiring the full in-silico .dta at runtime.
    """

    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")
    prepared = data if "subgroup" in data.columns else prepare_paco2_distribution(data)
    frames: list[pd.DataFrame] = []
    binned_counts: dict[str, pd.Series] = {}
    for group in PACO2_SUBGROUP_ORDER:
        values = prepared.loc[prepared["subgroup"] == group, "paco2"].to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError(f"No PaCO2 values available for subgroup '{group}'.")
        bins = np.round(values / bin_width) * bin_width
        counts = pd.Series(bins).value_counts().sort_index()
        total = float(counts.sum())
        frame = pd.DataFrame(
            {
                "group": group,
                "paco2_bin": counts.index.astype(float),
                "count": counts.to_numpy(dtype=int),
                # Weights encode the empirical PaCO2 pretest distribution per subgroup.
                "weight": counts.to_numpy(dtype=float) / total,
            }
        )
        frames.append(frame)
        binned_counts[group] = counts

    # Pool subgroup bins weighted by subgroup sample sizes (equivalent to pooling raw data).
    all_counts: pd.Series | None = None
    for counts in binned_counts.values():
        all_counts = counts if all_counts is None else all_counts.add(counts, fill_value=0)
    if all_counts is None:
        raise ValueError("Unable to pool PaCO2 prior bins across subgroups.")
    all_total = float(all_counts.sum())
    all_frame = pd.DataFrame(
        {
            "group": "all",
            "paco2_bin": all_counts.index.astype(float),
            "count": all_counts.to_numpy(dtype=int),
            "weight": all_counts.to_numpy(dtype=float) / all_total,
        }
    )
    frames.append(all_frame)

    result = pd.concat(frames, ignore_index=True)
    return _validate_paco2_prior_bins(result)


def load_paco2_prior_bins(path: Path) -> pd.DataFrame:
    """Load a binned PaCO2 prior table from CSV/XLSX."""

    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported prior bin format: {path.suffix}")
    return _validate_paco2_prior_bins(data)


def load_paco2_prior_bins_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load a binned PaCO2 prior from uploaded bytes."""

    buffer = io.BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        data = pd.read_csv(buffer)
    elif filename.lower().endswith((".xlsx", ".xls")):
        data = pd.read_excel(buffer)
    else:
        raise ValueError("Uploaded prior must be CSV/XLSX.")
    return _validate_paco2_prior_bins(data)


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
        values = _prior_values_from_bins(bins, subgroup_key)
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
        values = _prior_values_from_bins(bins, subgroup_key)
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
            values = prepared.loc[prepared["subgroup"] == subgroup_key, "paco2"].to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError(f"No PaCO2 values found for subgroup '{subgroup_key}'.")
        return PriorLoadResult(
            values=values,
            bins=None,
            source="insilico_dta",
            paths_checked=tuple(paths_checked),
        )

    message = (
        "PaCO2 prior data not found. Provide a binned prior CSV or the in-silico database."
    )
    error = PriorLoadError(message=message, paths_checked=tuple(paths_checked))
    return PriorLoadResult(
        values=None,
        bins=None,
        source=None,
        paths_checked=tuple(paths_checked),
        error=error,
    )


def _validate_paco2_prior_bins(data: pd.DataFrame) -> pd.DataFrame:
    prior = data.copy()
    if "group" not in prior.columns and "subgroup" in prior.columns:
        prior = prior.rename(columns={"subgroup": "group"})
    missing = PACO2_PRIOR_REQUIRED_COLUMNS - set(prior.columns)
    if missing:
        raise ValueError(f"Missing prior bin columns: {sorted(missing)}")
    prior["group"] = prior["group"].astype(str).str.strip().str.lower()
    prior["paco2_bin"] = pd.to_numeric(prior["paco2_bin"], errors="coerce")
    prior["count"] = pd.to_numeric(prior["count"], errors="coerce")
    prior["weight"] = pd.to_numeric(prior["weight"], errors="coerce")
    if not np.all(np.isfinite(prior["paco2_bin"])):
        raise ValueError("Non-finite PaCO2 bin values in prior.")
    if not np.all(np.isfinite(prior["count"])):
        raise ValueError("Non-finite counts in prior.")
    if not np.all(np.isfinite(prior["weight"])):
        raise ValueError("Non-finite weights in prior.")
    if np.any(prior["count"] < 0):
        raise ValueError("Prior counts must be non-negative.")
    if np.any(prior["weight"] < 0):
        raise ValueError("Prior weights must be non-negative.")
    groups = set(prior["group"])
    missing_groups = set(PACO2_PRIOR_GROUPS) - groups
    if missing_groups:
        raise ValueError(f"Prior bins missing groups: {sorted(missing_groups)}")
    weight_sums = prior.groupby("group")["weight"].sum()
    if not np.allclose(weight_sums.to_numpy(dtype=float), 1.0, atol=1e-6):
        raise ValueError("Prior weights must sum to 1 within each group.")
    return prior


def _prior_values_from_bins(prior_bins: pd.DataFrame, group: str) -> np.ndarray:
    subset = prior_bins.loc[prior_bins["group"] == group]
    if subset.empty:
        raise ValueError(f"No binned priors available for group '{group}'.")
    return np.repeat(
        subset["paco2_bin"].to_numpy(dtype=float),
        subset["count"].to_numpy(dtype=int),
    )


def _validate_paco2_columns(data: pd.DataFrame) -> None:
    missing = PACO2_REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing PaCO2 columns: {sorted(missing)}")
