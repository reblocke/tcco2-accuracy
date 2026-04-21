"""JSON-serializable browser contract for the static TcCO2 app."""

from __future__ import annotations

from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

from .data import load_paco2_prior_bins_bytes, prior_distribution_from_bins
from .ui_api import build_subgroup_bootstrap_draws, predict_paco2_from_tcco2
from .utils import validate_params_df
from .validate_inputs import validate_conway_studies_df

DEFAULT_N_BOOT = 1000
DEFAULT_BOOTSTRAP_MODE = "cluster_plus_withinstudy"


def compute_ui_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute a serializable UI result from a browser payload."""

    subgroup = str(payload.get("subgroup", "all")).strip().lower()
    mode = str(payload.get("mode", "prior_weighted")).strip().lower()
    params = _params_from_payload(payload, subgroup)

    prior_values = None
    prior_weights = None
    prior_source = "not_used"
    if mode == "prior_weighted":
        prior_bins = _required_frame(payload, "prior_bins")
        prior_values, prior_weights = prior_distribution_from_bins(prior_bins, subgroup)
        prior_source = str(payload.get("prior_source") or "provided_bins")

    result = predict_paco2_from_tcco2(
        tcco2=_float_payload(payload, "tcco2", 50.0),
        subgroup=subgroup,  # type: ignore[arg-type]
        threshold=_float_payload(payload, "threshold", 45.0),
        mode=mode,  # type: ignore[arg-type]
        interval=_float_payload(payload, "interval", 0.95),
        params_draws=params,
        paco2_prior_values=prior_values,
        paco2_prior_weights=prior_weights,
        n_param_draws=_optional_int_payload(payload, "n_param_draws"),
        seed=_optional_int_payload(payload, "seed"),
        bin_width=_float_payload(payload, "bin_width", 1.0),
    )

    return {
        "subgroup": result.subgroup,
        "tcco2": result.tcco2,
        "threshold": result.threshold,
        "mode": result.mode,
        "interval": result.interval,
        "paco2_q_low": result.paco2_q_low,
        "paco2_median": result.paco2_median,
        "paco2_q_high": result.paco2_q_high,
        "p_ge_threshold": result.p_ge_threshold,
        "decision_label": result.decision_label,
        "p_true_positive": result.p_true_positive,
        "p_false_positive": result.p_false_positive,
        "p_true_negative": result.p_true_negative,
        "p_false_negative": result.p_false_negative,
        "paco2_bin": _array_to_list(result.paco2_bin),
        "posterior_prob": _array_to_list(result.posterior_prob),
        "prior_prob": None if result.prior_prob is None else _array_to_list(result.prior_prob),
        "likelihood_prob": None
        if result.likelihood_prob is None
        else _array_to_list(result.likelihood_prob),
        "posterior_cdf": _array_to_list(result.posterior_cdf),
        "metadata": {
            "params_source": "payload_params"
            if _has_frame_payload(payload, "params")
            else "computed",
            "prior_source": prior_source,
            "n_params": int(params.shape[0]),
        },
    }


def build_bootstrap_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return serializable bootstrap draws for browser-side recompute flows."""

    subgroup = str(payload.get("subgroup", "all")).strip().lower()
    studies = _required_frame(payload, "study")
    params = build_subgroup_bootstrap_draws(
        studies=studies,
        subgroup=subgroup,  # type: ignore[arg-type]
        n_boot=_int_payload(payload, "n_boot", DEFAULT_N_BOOT),
        seed=_optional_int_payload(payload, "seed"),
        bootstrap_mode=str(payload.get("bootstrap_mode") or DEFAULT_BOOTSTRAP_MODE),
    )
    return {
        "subgroup": subgroup,
        "n_rows": int(params.shape[0]),
        "params": _json_records(params),
    }


def _params_from_payload(payload: dict[str, Any], subgroup: str) -> pd.DataFrame:
    params = _optional_frame(payload, "params")
    if params is not None:
        return validate_params_df(params).reset_index(drop=True)

    studies = _required_frame(payload, "study")
    validate_conway_studies_df(studies)
    return build_subgroup_bootstrap_draws(
        studies=studies,
        subgroup=subgroup,  # type: ignore[arg-type]
        n_boot=_int_payload(payload, "n_boot", DEFAULT_N_BOOT),
        seed=_optional_int_payload(payload, "seed"),
        bootstrap_mode=str(payload.get("bootstrap_mode") or DEFAULT_BOOTSTRAP_MODE),
    )


def _required_frame(payload: dict[str, Any], name: str) -> pd.DataFrame:
    frame = _optional_frame(payload, name)
    if frame is None:
        raise ValueError(f"Missing required browser payload table: {name}")
    return frame


def _optional_frame(payload: dict[str, Any], name: str) -> pd.DataFrame | None:
    records = payload.get(f"{name}_records")
    if records is not None:
        return pd.DataFrame(list(records))

    csv_text = payload.get(f"{name}_csv")
    if csv_text is not None:
        if not str(csv_text).strip():
            raise ValueError(f"Empty CSV payload for table: {name}")
        if name == "prior_bins":
            return load_paco2_prior_bins_bytes(str(csv_text).encode("utf-8"), "prior.csv")
        return pd.read_csv(StringIO(str(csv_text)))

    return None


def _has_frame_payload(payload: dict[str, Any], name: str) -> bool:
    return payload.get(f"{name}_records") is not None or payload.get(f"{name}_csv") is not None


def _float_payload(payload: dict[str, Any], name: str, default: float) -> float:
    value = payload.get(name, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed


def _int_payload(payload: dict[str, Any], name: str, default: int) -> int:
    value = _optional_int_payload(payload, name)
    return default if value is None else value


def _optional_int_payload(payload: dict[str, Any], name: str) -> int | None:
    value = payload.get(name)
    if value in (None, "", 0, "0"):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _array_to_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.replace({np.nan: None}).to_dict(orient="records"):
        records.append({str(key): value for key, value in row.items()})
    return records
