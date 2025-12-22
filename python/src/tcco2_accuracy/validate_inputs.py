"""Validation helpers for Conway study input tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_STUDY_COLUMNS = {"study_id", "bias", "n_pairs", "n_participants"}
SUBGROUP_FLAG_COLUMNS = ("is_icu", "is_arf", "is_lft")


def validate_conway_studies_df(df: pd.DataFrame) -> None:
    """Validate the canonical Conway study input table.

    Required columns: study_id, bias, n_pairs, n_participants, and either sd or s2.
    Subgroup flags (is_icu, is_arf, is_lft) must be present and boolean-like.
    """

    missing = REQUIRED_STUDY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required Conway columns: {sorted(missing)}")

    if "sd" not in df.columns and "s2" not in df.columns:
        raise ValueError("Conway study table must include `sd` or `s2`.")

    for column in ("bias", "sd", "s2", "n_pairs", "n_participants", "c"):
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite values detected in `{column}`.")
        if column in {"sd", "s2", "n_pairs", "n_participants", "c"}:
            if np.any(values <= 0):
                raise ValueError(f"Column `{column}` must be positive.")

    n_pairs = pd.to_numeric(df["n_pairs"], errors="coerce").to_numpy(dtype=float)
    n_participants = pd.to_numeric(df["n_participants"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isclose(n_pairs, np.round(n_pairs))):
        raise ValueError("`n_pairs` must be integer-valued.")
    if not np.all(np.isclose(n_participants, np.round(n_participants))):
        raise ValueError("`n_participants` must be integer-valued.")

    if df["study_id"].isna().any():
        raise ValueError("`study_id` must be non-empty for all rows.")
    if df["study_id"].duplicated().any():
        raise ValueError("`study_id` must be unique for each study.")

    missing_flags = [col for col in SUBGROUP_FLAG_COLUMNS if col not in df.columns]
    if missing_flags:
        raise ValueError(f"Missing subgroup flag columns: {missing_flags}")

    for column in SUBGROUP_FLAG_COLUMNS:
        raw = df[column]
        if raw.isna().any():
            raise ValueError(f"Subgroup flag `{column}` must not be missing.")
        values = raw.dropna().unique()
        allowed = {0, 1, True, False}
        if not set(values).issubset(allowed):
            raise ValueError(
                f"Subgroup flag `{column}` must be boolean-like (0/1/True/False)."
            )
