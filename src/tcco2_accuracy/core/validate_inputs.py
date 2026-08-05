"""Validation helpers for scientific inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

REQUIRED_STUDY_COLUMNS = {"study_id", "bias", "n_pairs", "n_participants"}
SUBGROUP_FLAG_COLUMNS = ("is_icu", "is_arf", "is_lft")
CONWAY_RELATION_RTOL = 1e-10
CONWAY_RELATION_ATOL = 1e-12


def validate_conway_studies_df(df: pd.DataFrame) -> None:
    """Validate the canonical Conway study input table.

    Required columns: study_id, bias, n_pairs, n_participants, and either sd or s2.
    Subgroup flags (is_icu, is_arf, is_lft) must be present and boolean-like. Flags
    are evaluated independently, so a study may belong to more than one subgroup.
    """

    missing = REQUIRED_STUDY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required Conway columns: {sorted(missing)}")

    if "sd" not in df.columns and "s2" not in df.columns:
        raise ValueError("Conway study table must include `sd` or `s2`.")
    if df.empty:
        raise ValueError("Conway study table must contain at least one study.")

    study_ids = df["study_id"].astype("string").str.strip()
    if study_ids.isna().any() or study_ids.eq("").any():
        raise ValueError("`study_id` must be non-empty for all rows.")
    if study_ids.duplicated().any():
        raise ValueError("`study_id` must be unique after trimming whitespace.")

    _finite_numeric_values(df, "bias")
    sd = _finite_numeric_values(df, "sd") if "sd" in df.columns else None
    s2 = _finite_numeric_values(df, "s2") if "s2" in df.columns else None
    _validate_variances(sd=sd, s2=s2)

    n_pairs = _finite_numeric_values(df, "n_pairs")
    n_participants = _finite_numeric_values(df, "n_participants")
    _validate_counts(n_pairs, n_participants, "n_pairs", "n_participants")

    if "c" in df.columns:
        c = _finite_numeric_values(df, "c")
        _validate_repeated_measure_count(c, n_pairs, n_participants)

    missing_flags = [col for col in SUBGROUP_FLAG_COLUMNS if col not in df.columns]
    if missing_flags:
        raise ValueError(f"Missing subgroup flag columns: {missing_flags}")

    for column in SUBGROUP_FLAG_COLUMNS:
        raw = df[column]
        if raw.isna().any():
            raise ValueError(f"Subgroup flag `{column}` must not be missing.")
        values = raw.unique()
        allowed = {0, 1, True, False}
        if not set(values).issubset(allowed):
            raise ValueError(f"Subgroup flag `{column}` must be boolean-like (0/1/True/False).")


def validate_conway_meta_inputs_df(df: pd.DataFrame) -> None:
    """Defensively validate analysis-form Conway inputs before equation use."""

    required = {"study", "bias", "s2", "n", "n_2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing Conway meta-analysis columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Conway subgroup analysis must contain at least one study.")

    _finite_numeric_values(df, "bias")
    s2 = _finite_numeric_values(df, "s2")
    sd = _finite_numeric_values(df, "sd") if "sd" in df.columns else None
    _validate_variances(sd=sd, s2=s2)

    n_pairs = _finite_numeric_values(df, "n")
    n_participants = _finite_numeric_values(df, "n_2")
    _validate_counts(n_pairs, n_participants, "n", "n_2")

    if "c" in df.columns:
        c = _finite_numeric_values(df, "c")
        _validate_repeated_measure_count(c, n_pairs, n_participants)

    study_ids = df["study"].astype("string").str.strip()
    if study_ids.isna().any() or study_ids.eq("").any():
        raise ValueError("Conway meta-analysis `study` identifiers must be non-empty.")
    if study_ids.duplicated().any():
        raise ValueError("Conway meta-analysis `study` identifiers must be unique after trimming.")


def validate_paco2_values(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return non-empty, finite, positive PaCO2 values without imposing an upper bound."""

    try:
        result = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("PaCO2 values must be numeric.") from exc
    if result.size == 0:
        raise ValueError("PaCO2 values must be non-empty.")
    if not np.all(np.isfinite(result)):
        raise ValueError("PaCO2 values must be finite.")
    if np.any(result <= 0):
        raise ValueError("PaCO2 values must be positive.")
    return result


def validate_threshold(value: float) -> float:
    """Return one finite, positive PaCO2 threshold."""

    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("PaCO2 threshold must be numeric.") from exc
    if not np.isfinite(threshold):
        raise ValueError("PaCO2 threshold must be finite.")
    if threshold <= 0:
        raise ValueError("PaCO2 threshold must be positive.")
    return threshold


def validate_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    """Return supplied PaCO2 thresholds after scalar validation."""

    thresholds = tuple(validate_threshold(value) for value in values)
    if not thresholds:
        raise ValueError("At least one PaCO2 threshold is required.")
    return thresholds


def _finite_numeric_values(df: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values detected in `{column}`.")
    return values


def _validate_variances(*, sd: np.ndarray | None, s2: np.ndarray | None) -> None:
    if sd is not None and np.any(sd <= 0):
        raise ValueError("Column `sd` must be finite and positive.")
    if s2 is not None and np.any(s2 <= 0):
        raise ValueError("Column `s2` must be finite and positive.")
    if sd is not None and s2 is not None:
        consistent = np.isclose(
            sd**2,
            s2,
            rtol=CONWAY_RELATION_RTOL,
            atol=CONWAY_RELATION_ATOL,
        )
        if not np.all(consistent):
            raise ValueError(
                "Columns `sd` and `s2` must satisfy sd^2 approximately equal to s2 "
                f"(rtol={CONWAY_RELATION_RTOL:g}, atol={CONWAY_RELATION_ATOL:g})."
            )


def _validate_counts(
    n_pairs: np.ndarray,
    n_participants: np.ndarray,
    pairs_label: str,
    participants_label: str,
) -> None:
    if np.any(n_pairs <= 0) or not np.all(n_pairs == np.floor(n_pairs)):
        raise ValueError(f"`{pairs_label}` must contain positive integer counts.")
    if np.any(n_participants <= 1) or not np.all(n_participants == np.floor(n_participants)):
        raise ValueError(f"`{participants_label}` must contain integer counts greater than 1.")
    if np.any(n_pairs < n_participants):
        raise ValueError(
            f"`{pairs_label}` must be greater than or equal to `{participants_label}`."
        )


def _validate_repeated_measure_count(
    c: np.ndarray,
    n_pairs: np.ndarray,
    n_participants: np.ndarray,
) -> None:
    if np.any(c < 1):
        raise ValueError("Column `c` must be greater than or equal to 1.")
    expected = n_pairs / n_participants
    if not np.all(
        np.isclose(
            c,
            expected,
            rtol=CONWAY_RELATION_RTOL,
            atol=CONWAY_RELATION_ATOL,
        )
    ):
        raise ValueError(
            "Column `c` must equal n_pairs / n_participants "
            f"(rtol={CONWAY_RELATION_RTOL:g}, atol={CONWAY_RELATION_ATOL:g})."
        )
