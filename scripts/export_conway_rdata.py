"""Export Conway `data.Rdata` into canonical Conway study tables.

The Conway R markdown notes that study-level inputs live in named objects
(`main`, `ICU`, `ARF`, `LFT`) with bias/S2 columns; counts (n/n_2/c) are
instead stored in the published Stata file. We merge these sources into the
canonical study table used by the Python pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RDATA_CANDIDATES = (
    REPO_ROOT / "Data" / "data.Rdata",
    REPO_ROOT / "Data" / "Conway Thorax supplement and code" / "data.Rdata",
)

REQUIRED_RDATA_OBJECTS = ("main", "ICU", "ARF", "LFT")
ROWNAME_COLUMNS = ("row.names", "row_names", "rownames")

FALLBACK_STUDIES = {
    # ICU-only Bolliger row is excluded from main but needed for Table 1 counts.
    "Bolliger 2007 (TOSCA - ICU)": {
        "source_object": "ICU",
        "n_pairs": 49,
        "n_participants": 49,
        "c": 1.0,
    }
}


def _default_rdata_path() -> Path:
    for candidate in DEFAULT_RDATA_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_RDATA_CANDIDATES[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Conway data.Rdata to CSV/XLSX.")
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_rdata_path(),
        help="Path to Conway data.Rdata.",
    )
    parser.add_argument(
        "--dta",
        type=Path,
        default=None,
        help="Optional path to Conway data.dta (counts).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "Data",
        help="Output directory for canonical tables.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail on missing required fields (default: True).",
    )
    parser.add_argument(
        "--allow-missing-counts",
        action="store_true",
        help="Allow missing counts even in strict mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    objects = _load_rdata_objects(args.input)
    rdata_frames = _require_rdata_frames(objects, REQUIRED_RDATA_OBJECTS, args.strict)

    main_frame = rdata_frames["main"]
    canonical = _build_main_table(main_frame, args.strict)
    canonical = _ensure_fallback_studies(canonical, rdata_frames, args.strict)

    # Counts live in data.dta rather than the RData objects.
    counts = _load_counts_table(_discover_dta_path(args.input, args.dta), args.strict)
    canonical = _merge_counts(
        canonical,
        counts,
        strict=args.strict,
        allow_missing=args.allow_missing_counts,
    )

    canonical = _apply_subgroup_flags(canonical, rdata_frames)
    canonical = _finalize_table(
        canonical,
        strict=args.strict,
        allow_missing_counts=args.allow_missing_counts,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "conway_studies.csv"
    xlsx_path = out_dir / "conway_studies.xlsx"
    for path in (csv_path, xlsx_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace.")

    canonical.to_csv(csv_path, index=False)
    canonical.to_excel(xlsx_path, index=False)


def _load_rdata_objects(path: Path) -> dict[str, object]:
    try:
        import pyreadr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("pyreadr is required to export Conway data.Rdata.") from exc

    if not path.exists():
        raise FileNotFoundError(f"RData file not found: {path}")
    result = pyreadr.read_r(path)
    if not result:
        raise ValueError(f"No objects found in {path}")
    return dict(result)


def _describe_objects(objects: dict[str, object]) -> str:
    lines: list[str] = []
    for key, value in objects.items():
        if isinstance(value, pd.DataFrame):
            columns = ", ".join(str(col) for col in value.columns)
            lines.append(f"{key}: [{columns}]")
        else:
            lines.append(f"{key}: <{type(value).__name__}>")
    return "; ".join(lines)


def _require_rdata_frames(
    objects: dict[str, object],
    required: Iterable[str],
    strict: bool,
) -> dict[str, pd.DataFrame]:
    missing = [name for name in required if name not in objects]
    if missing:
        message = (
            "Required RData objects are missing: "
            f"{missing}. Available: {_describe_objects(objects)}"
        )
        _ensure(False, message, strict)
    frames: dict[str, pd.DataFrame] = {}
    for name in required:
        value = objects.get(name)
        if not isinstance(value, pd.DataFrame):
            message = (
                f"RData object '{name}' is not a dataframe. "
                f"Available: {_describe_objects(objects)}"
            )
            _ensure(False, message, strict)
            continue
        frames[name] = value
    if len(frames) != len(required):
        raise ValueError("Missing required RData dataframes; see warnings above.")
    return frames


def _index_looks_like_labels(index: pd.Index) -> bool:
    if isinstance(index, pd.RangeIndex):
        return False
    if pd.api.types.is_string_dtype(index):
        return True
    return all(isinstance(value, str) for value in index)


def _find_column(data: pd.DataFrame, target: str) -> str | None:
    target = target.lower()
    for col in data.columns:
        if str(col).strip().lower() == target:
            return col
    return None


def _extract_study_ids(data: pd.DataFrame, object_key: str, strict: bool) -> pd.Series:
    # pyreadr may expose row names as a column or index, so prefer those labels.
    for col in ROWNAME_COLUMNS:
        if col in data.columns:
            values = data[col]
            break
    else:
        if _index_looks_like_labels(data.index):
            values = pd.Series(data.index, index=data.index)
        else:
            study_col = _find_column(data, "study") or _find_column(data, "study_id")
            if study_col is None:
                message = (
                    f"Unable to locate study identifiers in RData object '{object_key}'. "
                    f"Columns: {list(data.columns)}"
                )
                _ensure(False, message, strict)
                raise ValueError(message)
            values = data[study_col]
    return values.astype(str).map(_normalize_study_id)


def _normalize_study_id(value: str) -> str:
    return str(value).strip()


def _build_main_table(main: pd.DataFrame, strict: bool) -> pd.DataFrame:
    study_ids = _extract_study_ids(main, "main", strict)
    bias_col = _find_column(main, "bias")
    s2_col = _find_column(main, "s2")
    if bias_col is None or s2_col is None:
        message = (
            "main dataframe must include bias and S2 columns as documented in "
            "TcCO2 meta-analysis.Rmd. "
            f"Columns: {list(main.columns)}"
        )
        _ensure(False, message, strict)
        raise ValueError(message)

    bias = pd.to_numeric(main[bias_col], errors="coerce")
    s2 = pd.to_numeric(main[s2_col], errors="coerce")
    sd = np.sqrt(s2)

    _validate_bias_s2(bias, s2, strict)

    return pd.DataFrame(
        {
            "study_id": study_ids,
            "bias": bias,
            "sd": sd,
            "s2": s2,
        }
    )


def _ensure_fallback_studies(
    canonical: pd.DataFrame,
    rdata_frames: dict[str, pd.DataFrame],
    strict: bool,
) -> pd.DataFrame:
    existing = set(canonical["study_id"])
    additions: list[pd.DataFrame] = []
    for study_id, spec in FALLBACK_STUDIES.items():
        if study_id in existing:
            continue
        source_key = spec["source_object"]
        source = rdata_frames.get(source_key)
        if source is None:
            message = (
                f"Fallback study '{study_id}' requires '{source_key}' object in RData."
            )
            _ensure(False, message, strict)
            continue
        bias, s2 = _lookup_bias_s2(source, study_id, source_key, strict)
        additions.append(
            pd.DataFrame(
                {
                    "study_id": [study_id],
                    "bias": [bias],
                    "sd": [np.sqrt(s2)],
                    "s2": [s2],
                }
            )
        )
    if additions:
        canonical = pd.concat([canonical, *additions], ignore_index=True)
    return canonical


def _lookup_bias_s2(
    data: pd.DataFrame,
    study_id: str,
    object_key: str,
    strict: bool,
) -> tuple[float, float]:
    study_ids = _extract_study_ids(data, object_key, strict)
    mask = study_ids == study_id
    if not mask.any():
        message = (
            f"Study '{study_id}' not found in RData object '{object_key}'. "
            f"Available: {sorted(study_ids.unique())[:5]}"
        )
        _ensure(False, message, strict)
        raise ValueError(message)
    bias_col = _find_column(data, "bias")
    s2_col = _find_column(data, "s2")
    if bias_col is None or s2_col is None:
        message = (
            f"RData object '{object_key}' must include bias and S2 columns. "
            f"Columns: {list(data.columns)}"
        )
        _ensure(False, message, strict)
        raise ValueError(message)
    bias = float(pd.to_numeric(data.loc[mask, bias_col], errors="coerce").iloc[0])
    s2 = float(pd.to_numeric(data.loc[mask, s2_col], errors="coerce").iloc[0])
    _validate_bias_s2(pd.Series([bias]), pd.Series([s2]), strict)
    return bias, s2


def _validate_bias_s2(bias: pd.Series, s2: pd.Series, strict: bool) -> None:
    if not np.all(np.isfinite(bias.to_numpy())):
        _ensure(False, "Non-finite bias values detected in RData.", strict)
    if not np.all(np.isfinite(s2.to_numpy())):
        _ensure(False, "Non-finite S2 values detected in RData.", strict)
    if np.any(s2.to_numpy() <= 0):
        _ensure(False, "S2 values must be positive in RData.", strict)


def _discover_dta_path(rdata_path: Path, explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    else:
        candidates.append(rdata_path.with_name("data.dta"))
        candidates.append(REPO_ROOT / "Conway Meta" / "data.dta")
        candidates.append(REPO_ROOT / "Data" / "Conway Thorax supplement and code" / "data.dta")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    message = "Unable to locate data.dta. Looked in: " + ", ".join(
        str(path) for path in candidates
    )
    raise FileNotFoundError(message)


def _load_counts_table(path: Path, strict: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"data.dta not found: {path}")
    data = pd.read_stata(path)
    required = {"study", "n", "n_2", "c"}
    missing = required - set(data.columns)
    if missing:
        message = f"data.dta is missing required columns: {sorted(missing)}"
        _ensure(False, message, strict)
        raise ValueError(message)
    # Stata counts are the source of truth for n/n_2/c.
    counts = data.loc[:, ["study", "n", "n_2", "c"]].copy()
    counts["study_id"] = counts["study"].astype(str).map(_normalize_study_id)
    counts = counts.rename(columns={"n": "n_pairs", "n_2": "n_participants"})
    counts = counts.drop(columns="study")
    if counts["study_id"].duplicated().any():
        message = "Duplicate study IDs detected in data.dta."
        _ensure(False, message, strict)
    return counts


def _merge_counts(
    canonical: pd.DataFrame,
    counts: pd.DataFrame,
    strict: bool,
    allow_missing: bool,
) -> pd.DataFrame:
    canonical = canonical.copy()
    canonical["_study_key"] = canonical["study_id"].map(_normalize_study_id)
    counts = counts.copy()
    counts["_study_key"] = counts["study_id"].map(_normalize_study_id)
    merged = canonical.merge(
        counts.drop(columns=["study_id"]),
        on="_study_key",
        how="left",
    )
    merged = merged.drop(columns=["_study_key"])

    for study_id, spec in FALLBACK_STUDIES.items():
        mask = merged["study_id"] == study_id
        if not mask.any():
            continue
        # Fill counts for known missing studies from the supplemental ICU row.
        for column in ("n_pairs", "n_participants", "c"):
            if merged.loc[mask, column].isna().any():
                merged.loc[mask, column] = spec[column]

    missing_counts = merged["n_pairs"].isna() | merged["n_participants"].isna() | merged["c"].isna()
    if missing_counts.any():
        missing_ids = sorted(merged.loc[missing_counts, "study_id"].unique())
        message = f"Missing counts for study_id values: {missing_ids}"
        if strict and not allow_missing:
            raise ValueError(message)
        if allow_missing:
            print(f"Warning: {message}", file=sys.stderr)

    return merged


def _apply_subgroup_flags(
    canonical: pd.DataFrame, rdata_frames: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    canonical = canonical.copy()
    # Subgroup membership comes from RData objects because Stata flags are incomplete.
    icu_ids = set(_extract_study_ids(rdata_frames["ICU"], "ICU", True))
    arf_ids = set(_extract_study_ids(rdata_frames["ARF"], "ARF", True))
    lft_ids = set(_extract_study_ids(rdata_frames["LFT"], "LFT", True))

    canonical["is_icu"] = canonical["study_id"].isin(icu_ids).astype(int)
    canonical["is_arf"] = canonical["study_id"].isin(arf_ids).astype(int)
    canonical["is_lft"] = canonical["study_id"].isin(lft_ids).astype(int)
    return canonical


def _finalize_table(
    canonical: pd.DataFrame,
    strict: bool,
    allow_missing_counts: bool,
) -> pd.DataFrame:
    canonical = canonical.copy()
    for column in ("n_pairs", "n_participants"):
        numeric = pd.to_numeric(canonical[column], errors="coerce")
        if allow_missing_counts:
            canonical[column] = numeric.round().astype("Int64")
        else:
            canonical[column] = numeric.round().astype(int)
    canonical["c"] = pd.to_numeric(canonical["c"], errors="coerce")

    _validate_bias_s2(canonical["bias"], canonical["s2"], strict)

    if strict and not allow_missing_counts:
        if canonical[["n_pairs", "n_participants", "c"]].isna().any().any():
            raise ValueError("Counts contain missing values after merge.")
        if (canonical[["n_pairs", "n_participants", "c"]] <= 0).any().any():
            raise ValueError("Counts must be positive.")

    canonical = canonical.sort_values("study_id").reset_index(drop=True)
    ordered = [
        "study_id",
        "bias",
        "sd",
        "s2",
        "n_pairs",
        "n_participants",
        "c",
        "is_icu",
        "is_arf",
        "is_lft",
    ]
    return canonical[ordered]


def _ensure(condition: bool, message: str, strict: bool) -> None:
    if condition:
        return
    if strict:
        raise ValueError(message)
    print(f"Warning: {message}", file=sys.stderr)


if __name__ == "__main__":
    main()
