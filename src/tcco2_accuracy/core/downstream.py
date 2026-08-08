"""Pure, in-memory helpers for draw-aligned downstream analysis.

Only the workflow module exposes results. It returns aggregate summaries and a
non-sensitive manifest; patient identifiers, ordering values, target records,
bins, counts, weights, and replicate-level values remain private to a running
process and are never written or returned by the public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import PACO2_SUBGROUP_ORDER
from .inference import infer_paco2
from .paco2 import prepare_paco2_distribution
from .simulation import expected_classification_metrics
from .two_stage import TwoStagePolicy, two_stage_metrics
from .utils import validate_params_df
from .validate_inputs import validate_threshold

__all__ = ["DownstreamAnalysisConfig", "PatientInputColumns"]

DOWNSTREAM_GROUPS: tuple[str, ...] = (*PACO2_SUBGROUP_ORDER, "all")
TARGET_RESAMPLING_POLICY = "ordinary_patient_cluster_with_bounded_degenerate_redraw"
MAX_TARGET_RESAMPLE_ATTEMPTS = 100
MAX_DEGENERATE_PROPOSAL_FRACTION = 0.01
CORE_METRICS: tuple[str, ...] = (
    "prevalence",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "lr_pos",
    "lr_neg",
    "tp_rate",
    "fp_rate",
    "tn_rate",
    "fn_rate",
    "misclass_rate",
)
TWO_STAGE_METRICS: tuple[str, ...] = (
    "zone1_prob",
    "zone2_prob",
    "zone3_prob",
    "zone1_lr",
    "zone2_lr",
    "zone3_lr",
    "zone1_post_prob",
    "zone2_post_prob",
    "zone3_post_prob",
    "reflex_fraction",
    "residual_misclass",
)
PREDICTION_METRICS: tuple[str, ...] = (
    "paco2_pi_lower",
    "paco2_pi_median",
    "paco2_pi_upper",
    "paco2_ge_threshold_probability",
)
_PARAMETER_GROUPS = {"pft": "lft", "ed_inp": "arf", "icu": "icu", "all": "main"}
_OUTPUT_METRIC_NAMES = {
    "tp_rate": "tp_probability",
    "fp_rate": "fp_probability",
    "tn_rate": "tn_probability",
    "fn_rate": "fn_probability",
    "misclass_rate": "misclassification_probability",
    "zone1_prob": "zone1_probability",
    "zone2_prob": "zone2_probability",
    "zone3_prob": "zone3_probability",
    "zone1_post_prob": "zone1_posterior_probability",
    "zone2_post_prob": "zone2_posterior_probability",
    "zone3_post_prob": "zone3_posterior_probability",
    "residual_misclass": "residual_misclassification_probability",
}


@dataclass(frozen=True)
class PatientInputColumns:
    """Required caller-supplied columns for patient-level downstream analysis."""

    patient_id: str = "patient_id"
    encounter_id: str = "encounter_id"
    encounter_order: str = "encounter_order"
    measurement_order: str = "measurement_order"


@dataclass(frozen=True)
class DownstreamAnalysisConfig:
    """Fixed downstream-analysis choices with explicit sensitivity switches."""

    true_threshold: float = 45.0
    two_stage_lower: float = 40.0
    two_stage_upper: float = 50.0
    tcco2_values: tuple[float, ...] = (35.0, 40.0, 45.0, 50.0, 55.0)
    measurement_policy: Literal["index", "all_measurements"] = "index"
    support: Literal["all", "central_95"] = "all"
    parameter_mapping: Literal["setting_specific", "pooled_main"] = "setting_specific"

    def __post_init__(self) -> None:
        threshold = validate_threshold(self.true_threshold)
        lower = float(self.two_stage_lower)
        upper = float(self.two_stage_upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("Two-stage boundaries must be finite and strictly ordered.")
        tcco2_values = tuple(float(value) for value in self.tcco2_values)
        if not tcco2_values or not np.all(np.isfinite(tcco2_values)):
            raise ValueError("Prediction TcCO2 values must be a non-empty finite sequence.")
        if len(set(tcco2_values)) != len(tcco2_values):
            raise ValueError("Prediction TcCO2 values must be unique.")
        if self.measurement_policy not in {"index", "all_measurements"}:
            raise ValueError(f"Unknown measurement policy: {self.measurement_policy}")
        if self.support not in {"all", "central_95"}:
            raise ValueError(f"Unknown PaCO2 support policy: {self.support}")
        if self.parameter_mapping not in {"setting_specific", "pooled_main"}:
            raise ValueError(f"Unknown parameter mapping: {self.parameter_mapping}")
        object.__setattr__(self, "true_threshold", threshold)
        object.__setattr__(self, "two_stage_lower", lower)
        object.__setattr__(self, "two_stage_upper", upper)
        object.__setattr__(self, "tcco2_values", tcco2_values)


@dataclass(frozen=True)
class _JointDraws:
    """Private compact draw arrays retained only during one workflow invocation."""

    core: dict[tuple[str, str, str], np.ndarray]
    prediction: dict[tuple[str, str, float, str, str], np.ndarray]
    two_stage: dict[tuple[str, str, str], np.ndarray]
    target_resampling: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class _TargetPopulation:
    """Patient clusters for ordinary cluster-bootstrap resampling."""

    clusters: tuple[np.ndarray, ...]


def _run_draw_aligned_downstream(
    patient_data: pd.DataFrame,
    params: pd.DataFrame,
    *,
    config: DownstreamAnalysisConfig,
    columns: PatientInputColumns,
    seed: int,
) -> _JointDraws:
    """Pair private agreement and patient bootstrap draws without exposing raw results."""

    target_groups = _prepare_target_groups(patient_data, columns=columns, config=config)
    parameter_groups = _prepare_parameter_groups(params, config=config)
    n_boot = _shared_replicate_count(parameter_groups)
    draws = _empty_joint_draws(parameter_groups, n_boot=n_boot, config=config)
    rng = np.random.default_rng(seed)
    policy = TwoStagePolicy(
        lower=config.two_stage_lower,
        upper=config.two_stage_upper,
        true_threshold=config.true_threshold,
    )

    for group in DOWNSTREAM_GROUPS:
        target_population = target_groups[group]
        parameter_rows, parameter_group = parameter_groups[group]
        rejected_proposals = 0
        total_proposals = 0
        for replicate, parameter_row in enumerate(parameter_rows.itertuples(index=False)):
            target_values, rejected, proposals = _resample_patient_clusters(
                target_population,
                rng=rng,
                threshold=config.true_threshold,
                group=group,
                replicate=replicate,
            )
            rejected_proposals += rejected
            total_proposals += proposals
            delta = float(parameter_row.delta)
            sigma2 = float(parameter_row.sigma2)
            tau2 = float(parameter_row.tau2)
            sd_total = _total_sd(sigma2, tau2, group=group, replicate=replicate)

            core_metrics = expected_classification_metrics(
                target_values,
                delta=delta,
                sd_total=sd_total,
                threshold_value=config.true_threshold,
            )
            _require_finite_metrics(core_metrics, CORE_METRICS, group=group, replicate=replicate)
            _store_metric_values(
                draws.core,
                key_prefix=(group, parameter_group),
                values=core_metrics,
                metrics=CORE_METRICS,
                replicate=replicate,
            )

            two_stage = two_stage_metrics(
                target_values, delta=delta, sd_total=sd_total, policy=policy
            )
            _require_finite_metrics(two_stage, TWO_STAGE_METRICS, group=group, replicate=replicate)
            _store_metric_values(
                draws.two_stage,
                key_prefix=(group, parameter_group),
                values=two_stage,
                metrics=TWO_STAGE_METRICS,
                replicate=replicate,
            )

            one_draw = pd.DataFrame({"delta": [delta], "sigma2": [sigma2], "tau2": [tau2]})
            for mode, use_prior in (("likelihood_only", False), ("prior_weighted", True)):
                prediction = infer_paco2(
                    config.tcco2_values,
                    one_draw,
                    thresholds=(config.true_threshold,),
                    paco2_prior=target_values if use_prior else None,
                    use_prior=use_prior,
                )
                for prediction_row in prediction.itertuples(index=False):
                    values = {
                        "paco2_pi_lower": float(prediction_row.paco2_q025),
                        "paco2_pi_median": float(prediction_row.paco2_q500),
                        "paco2_pi_upper": float(prediction_row.paco2_q975),
                        "paco2_ge_threshold_probability": float(
                            getattr(
                                prediction_row, f"p_ge_{_threshold_label(config.true_threshold)}"
                            )
                        ),
                    }
                    _require_finite_metrics(
                        values, PREDICTION_METRICS, group=group, replicate=replicate
                    )
                    _store_metric_values(
                        draws.prediction,
                        key_prefix=(group, parameter_group, float(prediction_row.tcco2), mode),
                        values=values,
                        metrics=PREDICTION_METRICS,
                        replicate=replicate,
                    )

        rejected_fraction = rejected_proposals / total_proposals
        if rejected_fraction > MAX_DEGENERATE_PROPOSAL_FRACTION:
            raise ValueError(
                "Degenerate patient-bootstrap proposals exceeded the allowed 1% within-setting "
                f"fraction; group='{group}', rejected_fraction={rejected_fraction:.6f}. "
                "The cohort is too sparse or imbalanced for the requested class-conditional metrics."
            )
        draws.target_resampling[group] = (rejected_proposals, total_proposals)

    return draws


def _empty_joint_draws(
    parameter_groups: Mapping[str, tuple[pd.DataFrame, str]],
    *,
    n_boot: int,
    config: DownstreamAnalysisConfig,
) -> _JointDraws:
    core: dict[tuple[str, str, str], np.ndarray] = {}
    prediction: dict[tuple[str, str, float, str, str], np.ndarray] = {}
    two_stage: dict[tuple[str, str, str], np.ndarray] = {}
    for group in DOWNSTREAM_GROUPS:
        parameter_group = parameter_groups[group][1]
        for metric in CORE_METRICS:
            core[(group, parameter_group, metric)] = np.empty(n_boot, dtype=float)
        for metric in TWO_STAGE_METRICS:
            two_stage[(group, parameter_group, metric)] = np.empty(n_boot, dtype=float)
        for tcco2 in config.tcco2_values:
            for mode in ("likelihood_only", "prior_weighted"):
                for metric in PREDICTION_METRICS:
                    prediction[(group, parameter_group, tcco2, mode, metric)] = np.empty(
                        n_boot, dtype=float
                    )
    return _JointDraws(
        core=core,
        prediction=prediction,
        two_stage=two_stage,
        target_resampling={},
    )


def _store_metric_values(
    store: dict[tuple[object, ...], np.ndarray],
    *,
    key_prefix: tuple[object, ...],
    values: Mapping[str, float],
    metrics: Sequence[str],
    replicate: int,
) -> None:
    for metric in metrics:
        store[(*key_prefix, metric)][replicate] = float(values[metric])


def _summarize_joint_draws(
    draws: _JointDraws,
    *,
    config: DownstreamAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize private arrays into aggregate 2.5/50/97.5 percentile tables."""

    core_rows = [
        _summary_row(
            values,
            requested_group=group,
            parameter_group_used=parameter_group,
            true_threshold=config.true_threshold,
            metric=_output_metric_name(metric),
        )
        for (group, parameter_group, metric), values in sorted(draws.core.items())
    ]
    prediction_rows = [
        _summary_row(
            values,
            requested_group=group,
            parameter_group_used=parameter_group,
            true_threshold=config.true_threshold,
            tcco2=tcco2,
            mode=mode,
            metric=_output_metric_name(metric),
        )
        for (group, parameter_group, tcco2, mode, metric), values in sorted(
            draws.prediction.items()
        )
    ]
    two_stage_rows = [
        _summary_row(
            values,
            requested_group=group,
            parameter_group_used=parameter_group,
            true_threshold=config.true_threshold,
            zone_lower=config.two_stage_lower,
            zone_upper=config.two_stage_upper,
            metric=_output_metric_name(metric),
        )
        for (group, parameter_group, metric), values in sorted(draws.two_stage.items())
    ]
    return (
        pd.DataFrame(
            core_rows,
            columns=(
                "requested_group",
                "parameter_group_used",
                "true_threshold",
                "metric",
                "bootstrap_q025",
                "bootstrap_q500",
                "bootstrap_q975",
            ),
        ),
        pd.DataFrame(
            prediction_rows,
            columns=(
                "requested_group",
                "parameter_group_used",
                "true_threshold",
                "tcco2",
                "mode",
                "metric",
                "bootstrap_q025",
                "bootstrap_q500",
                "bootstrap_q975",
            ),
        ),
        pd.DataFrame(
            two_stage_rows,
            columns=(
                "requested_group",
                "parameter_group_used",
                "true_threshold",
                "zone_lower",
                "zone_upper",
                "metric",
                "bootstrap_q025",
                "bootstrap_q500",
                "bootstrap_q975",
            ),
        ),
    )


def _summary_row(values: np.ndarray, **keys: float | str) -> dict[str, float | str]:
    low, estimate, high = np.quantile(values, (0.025, 0.5, 0.975))
    return {
        **keys,
        "bootstrap_q025": float(low),
        "bootstrap_q500": float(estimate),
        "bootstrap_q975": float(high),
    }


def _monte_carlo_stability(
    primary: _JointDraws,
    repeat: _JointDraws,
    *,
    repeat_seed: int,
) -> pd.DataFrame:
    """Compare private independent runs using batch-quantile MCSE estimates."""

    frames = [
        _stability_store(
            "core",
            primary.core,
            repeat.core,
            key_names=("requested_group", "parameter_group_used", "metric"),
            repeat_seed=repeat_seed,
        ),
        _stability_store(
            "prediction",
            primary.prediction,
            repeat.prediction,
            key_names=("requested_group", "parameter_group_used", "tcco2", "mode", "metric"),
            repeat_seed=repeat_seed,
        ),
        _stability_store(
            "two_stage",
            primary.two_stage,
            repeat.two_stage,
            key_names=("requested_group", "parameter_group_used", "metric"),
            repeat_seed=repeat_seed,
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def _stability_store(
    analysis: str,
    primary: Mapping[tuple[object, ...], np.ndarray],
    repeat: Mapping[tuple[object, ...], np.ndarray],
    *,
    key_names: Sequence[str],
    repeat_seed: int,
) -> pd.DataFrame:
    if set(primary) != set(repeat):
        raise ValueError(f"Independent {analysis} runs do not report matching result components.")
    rows: list[dict[str, float | str | bool]] = []
    for key in sorted(primary):
        primary_summary = _quantile_summary_with_mcse(primary[key])
        repeat_summary = _quantile_summary_with_mcse(repeat[key])
        key_values = dict(zip(key_names, key))
        metric = str(key_values["metric"])
        key_values["metric"] = _output_metric_name(metric)
        for component in ("bootstrap_q025", "bootstrap_q500", "bootstrap_q975"):
            primary_value, primary_mcse = primary_summary[component]
            repeat_value, repeat_mcse = repeat_summary[component]
            combined_mcse = float(np.hypot(primary_mcse, repeat_mcse))
            difference = abs(primary_value - repeat_value)
            precision = _reporting_precision(metric, primary_value)
            rows.append(
                {
                    "analysis": analysis,
                    **key_values,
                    "repeat_seed": repeat_seed,
                    "component": component,
                    "primary": primary_value,
                    "repeat": repeat_value,
                    "primary_mcse": primary_mcse,
                    "repeat_mcse": repeat_mcse,
                    "combined_mcse": combined_mcse,
                    "difference": difference,
                    "reporting_precision": precision,
                    "within_2_mcse": bool(difference <= 2 * combined_mcse),
                    "mcse_passed": bool(combined_mcse <= precision / 10),
                }
            )
    return pd.DataFrame(rows)


def _quantile_summary_with_mcse(values: np.ndarray) -> dict[str, tuple[float, float]]:
    if values.size < 20:
        raise ValueError(
            "Monte Carlo stability requires at least 20 draws for batch MCSE estimation."
        )
    batches = np.array_split(values, 20)
    result: dict[str, tuple[float, float]] = {}
    for component, quantile in (
        ("bootstrap_q025", 0.025),
        ("bootstrap_q500", 0.5),
        ("bootstrap_q975", 0.975),
    ):
        batch_estimates = np.asarray([np.quantile(batch, quantile) for batch in batches])
        result[component] = (
            float(np.quantile(values, quantile)),
            float(np.std(batch_estimates, ddof=1) / np.sqrt(len(batches))),
        )
    return result


def _is_stable(stability: pd.DataFrame) -> bool:
    """Return whether every aggregate component meets the MCSE precision gate."""

    return not stability.empty and bool(stability["mcse_passed"].all())


def _prepare_target_groups(
    patient_data: pd.DataFrame,
    *,
    columns: PatientInputColumns,
    config: DownstreamAnalysisConfig,
) -> dict[str, _TargetPopulation]:
    prepared = prepare_paco2_distribution(patient_data)
    records = _normalize_patient_schema(prepared, columns=columns)
    if config.measurement_policy == "index":
        group_records = {
            group: _select_index_records(
                records.loc[records["subgroup"] == group].copy(),
                columns=columns,
            )
            for group in PACO2_SUBGROUP_ORDER
        }
        group_records["all"] = _select_index_records(records, columns=columns)
    else:
        group_records = {
            group: records.loc[records["subgroup"] == group].copy()
            for group in PACO2_SUBGROUP_ORDER
        }
        group_records["all"] = records.copy()
    groups: dict[str, _TargetPopulation] = {}
    for group, frame in group_records.items():
        supported = _apply_support(frame, support=config.support)
        if supported.empty:
            raise ValueError(f"No PaCO2 records remain for downstream group '{group}'.")
        values = supported["paco2"].to_numpy(dtype=float)
        positive = values >= config.true_threshold
        if not positive.any() or positive.all():
            raise ValueError(
                "Each downstream group must contain PaCO2 values both below and at/above "
                f"{config.true_threshold:g} mmHg before bootstrap resampling; group='{group}'."
            )
        groups[group] = _target_population(
            supported,
            patient_column=columns.patient_id,
        )
    return groups


def _validate_patient_schema(data: pd.DataFrame, *, columns: PatientInputColumns) -> None:
    required = {
        columns.patient_id,
        columns.encounter_id,
        columns.encounter_order,
        columns.measurement_order,
        "paco2",
        "subgroup",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required patient-level columns: {missing}")
    for column in (columns.patient_id, columns.encounter_id):
        values = data[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"Patient-level column '{column}' must be nonblank.")
    _order_values(data[columns.encounter_order], columns.encounter_order)
    _order_values(data[columns.measurement_order], columns.measurement_order)


def _normalize_patient_schema(
    data: pd.DataFrame,
    *,
    columns: PatientInputColumns,
) -> pd.DataFrame:
    """Return an internal copy with canonical identifiers and unique measurement keys."""

    _validate_patient_schema(data, columns=columns)
    normalized = data.copy()
    for column in (columns.patient_id, columns.encounter_id):
        normalized[column] = normalized[column].astype("string").str.strip()
    measurement_key = pd.DataFrame(
        {
            "_patient_id": normalized[columns.patient_id],
            "_encounter_id": normalized[columns.encounter_id],
            "_measurement_order": _order_values(
                normalized[columns.measurement_order], columns.measurement_order
            ),
        }
    )
    if measurement_key.duplicated(keep=False).any():
        raise ValueError(
            "Patient-level rows must have unique patient/encounter/measurement-order keys."
        )
    return normalized


def _select_index_records(data: pd.DataFrame, *, columns: PatientInputColumns) -> pd.DataFrame:
    """Select one earliest eligible record per patient without an arbitrary tie-breaker."""

    records = data.copy()
    records["_encounter_order"] = _order_values(
        records[columns.encounter_order], columns.encounter_order
    )
    records["_measurement_order"] = _order_values(
        records[columns.measurement_order], columns.measurement_order
    )
    patient = columns.patient_id
    encounter = columns.encounter_id
    encounter_orders = records.groupby([patient, encounter], sort=False)[
        "_encounter_order"
    ].nunique()
    if (encounter_orders != 1).any():
        raise ValueError("Each patient/encounter pair must have one consistent encounter_order.")

    earliest_encounter_order = records.groupby(patient, sort=False)["_encounter_order"].transform(
        "min"
    )
    earliest_encounters = records.loc[
        records["_encounter_order"] == earliest_encounter_order
    ].copy()
    if earliest_encounters.groupby(patient, sort=False)[encounter].nunique().gt(1).any():
        raise ValueError(
            "Encounter order does not uniquely identify the earliest eligible encounter."
        )

    earliest_measurement_order = earliest_encounters.groupby(patient, sort=False)[
        "_measurement_order"
    ].transform("min")
    index_records = earliest_encounters.loc[
        earliest_encounters["_measurement_order"] == earliest_measurement_order
    ].copy()
    if index_records.groupby(patient, sort=False).size().gt(1).any():
        raise ValueError(
            "Measurement order does not uniquely identify the earliest eligible PaCO2 value."
        )
    return index_records.drop(columns=["_encounter_order", "_measurement_order"])


def _order_values(values: pd.Series, column: str) -> pd.Series:
    """Normalize an all-numeric or all-datetime order field; reject mixed input."""

    if values.isna().any():
        raise ValueError(f"Ordering column '{column}' must be finite numeric or valid datetime.")
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        numeric_values = numeric.to_numpy(dtype=float)
        if not np.isfinite(numeric_values).all():
            raise ValueError(
                f"Ordering column '{column}' must be finite numeric or valid datetime."
            )
        return numeric.astype(float)
    if numeric.notna().any():
        raise ValueError(
            f"Ordering column '{column}' must use only finite numeric values or only valid datetimes."
        )
    parsed = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
    if parsed.isna().any():
        raise ValueError(f"Ordering column '{column}' must be finite numeric or valid datetime.")
    return pd.Series(parsed.astype("int64"), index=values.index, dtype="int64")


def _apply_support(data: pd.DataFrame, *, support: str) -> pd.DataFrame:
    if support == "all":
        return data.copy()
    lower, upper = data["paco2"].quantile([0.025, 0.975], interpolation="linear")
    return data.loc[(data["paco2"] >= lower) & (data["paco2"] <= upper)].copy()


def _prepare_parameter_groups(
    params: pd.DataFrame,
    *,
    config: DownstreamAnalysisConfig,
) -> dict[str, tuple[pd.DataFrame, str]]:
    params = validate_params_df(params).copy()
    if "group" not in params.columns or "replicate" not in params.columns:
        raise ValueError(
            "Joint downstream analysis requires grouped parameter draws with replicate IDs."
        )
    params["group"] = params["group"].astype(str)
    params["replicate"] = pd.to_numeric(params["replicate"], errors="coerce")
    if params["replicate"].isna().any() or not np.all(params["replicate"] % 1 == 0):
        raise ValueError("Parameter replicate IDs must be integers.")
    result: dict[str, tuple[pd.DataFrame, str]] = {}
    for group in DOWNSTREAM_GROUPS:
        parameter_group = (
            "main" if config.parameter_mapping == "pooled_main" else _PARAMETER_GROUPS[group]
        )
        subset = params.loc[params["group"] == parameter_group].copy()
        if subset.empty:
            raise ValueError(
                f"No parameter draws for downstream group '{group}' using '{parameter_group}'."
            )
        subset = subset.sort_values("replicate").reset_index(drop=True)
        expected = np.arange(subset.shape[0])
        actual = subset["replicate"].to_numpy(dtype=int)
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"Parameter draws for '{parameter_group}' must contain one row for every "
                "replicate starting at zero."
            )
        result[group] = (subset, parameter_group)
    return result


def _shared_replicate_count(parameter_groups: Mapping[str, tuple[pd.DataFrame, str]]) -> int:
    counts = {group: len(frame) for group, (frame, _) in parameter_groups.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Parameter groups have unequal replicate counts: {counts}")
    return next(iter(counts.values()))


def _target_population(
    records: pd.DataFrame,
    *,
    patient_column: str,
) -> _TargetPopulation:
    clusters: list[np.ndarray] = []
    for _, frame in records.groupby(patient_column, sort=False):
        values = frame["paco2"].to_numpy(dtype=float)
        clusters.append(values)
    if not clusters:
        raise ValueError("Patient-cluster target population must be non-empty.")
    return _TargetPopulation(clusters=tuple(clusters))


def _resample_patient_clusters(
    population: _TargetPopulation,
    *,
    rng: np.random.Generator,
    threshold: float,
    group: str,
    replicate: int,
) -> tuple[np.ndarray, int, int]:
    """Return one ordinary patient-cluster sample with both truth classes."""

    clusters = population.clusters
    if not clusters:
        raise ValueError("Patient-cluster target population must be non-empty.")
    for attempt in range(1, MAX_TARGET_RESAMPLE_ATTEMPTS + 1):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        values = np.concatenate([clusters[index] for index in chosen])
        positive = values >= threshold
        if positive.any() and not positive.all():
            return values, attempt - 1, attempt
    raise ValueError(
        "Unable to obtain a patient-bootstrap replicate containing both truth classes within "
        f"{MAX_TARGET_RESAMPLE_ATTEMPTS} attempts; group='{group}', replicate={replicate}. "
        "The cohort is too sparse or imbalanced for the requested class-conditional metrics."
    )


def _total_sd(sigma2: float, tau2: float, *, group: str, replicate: int) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        total_variance = float(np.add(sigma2, tau2))
    if not np.isfinite(total_variance) or total_variance <= 0:
        raise ValueError(
            "Total downstream variance must be finite and strictly positive; "
            f"group='{group}', replicate={replicate}."
        )
    return float(np.sqrt(total_variance))


def _require_finite_metrics(
    values: Mapping[str, float], metrics: Sequence[str], *, group: str, replicate: int
) -> None:
    invalid = [metric for metric in metrics if not np.isfinite(values[metric])]
    if invalid:
        raise ValueError(
            "Non-estimable downstream metric(s) in a patient bootstrap draw; "
            f"group='{group}', replicate={replicate}, metrics={invalid}."
        )


def _reporting_precision(metric: str, value: float) -> float:
    if "paco2_pi" in metric:
        return 0.1
    if "lr" not in metric:
        return 0.001
    magnitude = abs(value)
    if magnitude < 100:
        return 0.01
    return float(10 ** (np.floor(np.log10(magnitude)) - 1))


def _threshold_label(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "p").replace("-", "m")


def _output_metric_name(metric: str) -> str:
    return _OUTPUT_METRIC_NAMES.get(metric, metric)
