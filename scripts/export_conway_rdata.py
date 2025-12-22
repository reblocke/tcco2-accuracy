"""Export Conway `data.Rdata` into canonical Conway study tables.

Uses the `main` data frame (Conway primary study-level dataset) from
`data.Rdata` (stored here under `Data/Conway Thorax supplement and code/data.Rdata`).
Columns are mapped as follows:
- study_id: `study`
- bias: `bias` (PaCO2 - TcCO2)
- sd/s2: within-study SD/variance of differences (`s2` -> `sd = sqrt(s2)`)
- n_pairs: `n`
- n_participants: `n_2`
- c: repeated measures per participant (`c` if present; otherwise n/n_2)
- subgroup flags: `icu1` → is_icu, `ed_arf_7` → is_arf, `respiratory_lft_6` → is_lft
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RDATA = REPO_ROOT / "Data" / "Conway Thorax supplement and code" / "data.Rdata"

EXTRA_STUDIES = {
    "Bolliger 2007 (TOSCA - ICU)": {
        "study_id": "Bolliger 2007 (TOSCA - ICU)",
        "bias": -2.175,
        "sd": np.sqrt(22.878651),
        "s2": 22.878651,
        "n_pairs": 49,
        "n_participants": 49,
        "c": 1.0,
        "is_icu": 1,
        "is_arf": 0,
        "is_lft": 0,
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Conway data.Rdata to CSV/XLSX.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RDATA,
        help="Path to Conway data.Rdata.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = _load_rdata(args.input)
    canonical = _build_canonical_table(data)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "conway_studies.csv"
    xlsx_path = out_dir / "conway_studies.xlsx"
    for path in (csv_path, xlsx_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace.")

    canonical.to_csv(csv_path, index=False)
    canonical.to_excel(xlsx_path, index=False)


def _load_rdata(path: Path) -> pd.DataFrame:
    try:
        import pyreadr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("pyreadr is required to export Conway data.Rdata.") from exc

    if not path.exists():
        raise FileNotFoundError(f"RData file not found: {path}")
    result = pyreadr.read_r(path)
    if not result:
        raise ValueError(f"No objects found in {path}")
    if "main" in result:
        data = result["main"]
    else:
        key = next(iter(result.keys()))
        data = result[key]
    data = data.copy()
    data.columns = [str(col).strip().lower() for col in data.columns]
    return data


def _first_column(data: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in data.columns:
            return candidate
    return None


def _build_canonical_table(data: pd.DataFrame) -> pd.DataFrame:
    study_col = _first_column(data, ["study", "study_id"])
    n_col = _first_column(data, ["n", "n_pairs"])
    n2_col = _first_column(data, ["n_2", "n_participants"])
    s2_col = _first_column(data, ["s2", "s2_unb"])
    bias_col = _first_column(data, ["bias"])

    if not study_col or not n_col or not n2_col or not s2_col or not bias_col:
        raise ValueError("RData missing required columns (study, n, n_2, bias, s2).")

    is_icu_col = _first_column(data, ["icu1", "icu_group"])
    is_arf_col = _first_column(data, ["ed_arf_7", "ed_arf", "ed_inp_group"])
    is_lft_col = _first_column(data, ["respiratory_lft_6", "respiratory_lft", "pft_group"])

    canonical = pd.DataFrame(
        {
            "study_id": data[study_col].astype(str),
            "bias": pd.to_numeric(data[bias_col], errors="coerce"),
            "s2": pd.to_numeric(data[s2_col], errors="coerce"),
            "n_pairs": pd.to_numeric(data[n_col], errors="coerce"),
            "n_participants": pd.to_numeric(data[n2_col], errors="coerce"),
            "c": pd.to_numeric(data.get("c"), errors="coerce"),
            "is_icu": pd.to_numeric(data[is_icu_col], errors="coerce") if is_icu_col else 0,
            "is_arf": pd.to_numeric(data[is_arf_col], errors="coerce") if is_arf_col else 0,
            "is_lft": pd.to_numeric(data[is_lft_col], errors="coerce") if is_lft_col else 0,
        }
    )
    canonical["c"] = canonical["c"].fillna(canonical["n_pairs"] / canonical["n_participants"])
    canonical["sd"] = np.sqrt(canonical["s2"])
    for flag in ("is_icu", "is_arf", "is_lft"):
        canonical[flag] = canonical[flag].fillna(0).astype(int)

    for study_id, row in EXTRA_STUDIES.items():
        if study_id not in set(canonical["study_id"]):
            canonical = pd.concat([canonical, pd.DataFrame([row])], ignore_index=True)

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


if __name__ == "__main__":
    main()
