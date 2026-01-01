"""Workflow helpers for manuscript-ready reporting outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..bland_altman import loa_bounds
from ..conditional import conditional_classification_curves
from ..data import PACO2_SUBGROUP_ORDER, load_paco2_distribution, prepare_paco2_distribution
from ..inference import infer_paco2, infer_paco2_by_subgroup
from ..simulation import (
    DEFAULT_CLASSIFICATION_THRESHOLDS,
    DEFAULT_SUMMARY_QUANTILES,
    PACO2_TO_CONWAY_GROUP,
    simulate_forward,
    simulate_forward_metrics,
    summarize_simulation_metrics,
)
from ..two_stage import TwoStagePolicy, summarize_two_stage_draws
from ..utils import quantile_key, threshold_label, validate_params_df
from . import bootstrap as bootstrap_workflow


MANUSCRIPT_GROUP_ORDER = ("pft", "ed_inp", "icu", "all")
MANUSCRIPT_GROUP_LABELS: dict[str, str] = {
    "pft": "Ambulatory (PFT)",
    "ed_inp": "ED/Inpatient",
    "icu": "ICU",
    "all": "Overall",
}
PARAM_GROUP_MAP: dict[str, str] = {"lft": "pft", "arf": "ed_inp", "icu": "icu", "main": "all"}


@dataclass(frozen=True)
class ManuscriptWorkflowResult:
    parameters: pd.DataFrame
    table1: pd.DataFrame
    confusion_matrix: pd.DataFrame
    two_stage_summary: pd.DataFrame
    table2: pd.DataFrame
    prediction_intervals: pd.DataFrame
    paco2_bins: pd.DataFrame
    misclassification_curves: pd.DataFrame
    markdown: dict[str, str]
    snippets: str
    invariants: dict[str, float | int | str]


def run_manuscript_outputs(
    params: pd.DataFrame | None = None,
    paco2_data: pd.DataFrame | None = None,
    paco2_path: Path | None = None,
    conway_path: Path | None = None,
    thresholds: Sequence[float] = DEFAULT_CLASSIFICATION_THRESHOLDS,
    true_threshold: float = 45.0,
    two_stage_lower: float = 40.0,
    two_stage_upper: float = 50.0,
    tcco2_values: Sequence[float] = (35.0, 40.0, 45.0, 50.0, 55.0),
    mode: str = "analytic",
    seed: int | None = None,
    n_boot: int = 1000,
    bootstrap_mode: str = "cluster_plus_withinstudy",
    n_draws: int | None = None,
    n_mc: int | None = None,
    include_prior: bool = True,
    out_dir: Path | None = None,
) -> ManuscriptWorkflowResult:
    """Generate manuscript-ready tables, figures, and result snippets."""

    if params is None:
        params = bootstrap_workflow.run_bootstrap(
            n_boot=n_boot,
            seed=seed,
            conway_path=conway_path,
            bootstrap_mode=bootstrap_mode,
        ).draws
    if paco2_data is None:
        paco2_data = load_paco2_distribution(paco2_path)
    prepared = prepare_paco2_distribution(paco2_data)

    thresholds = _normalize_thresholds(thresholds, true_threshold)
    rng = np.random.default_rng(seed)
    simulation_seed = _next_seed(rng)
    inference_seed = _next_seed(rng)
    two_stage_seed = _next_seed(rng)
    conditional_seed = _next_seed(rng)

    # Manuscript parameter table: bootstrap uncertainty in δ/σ²/τ² and LoA.
    parameters = _summarize_parameters(params)
    parameters_md = _format_parameters_table(parameters)

    # Table 1: forward simulation operating characteristics with bootstrap CI.
    sim_metrics = _simulate_with_all(
        prepared,
        params,
        thresholds=thresholds,
        mode=mode,
        seed=simulation_seed,
        n_draws=n_draws,
        n_mc=n_mc,
    )
    sim_summary = summarize_simulation_metrics(sim_metrics)
    paco2_stats = _paco2_stats(prepared, true_threshold=true_threshold)
    table1 = _build_table1(sim_summary, paco2_stats, true_threshold=true_threshold)
    table1_md = _format_table1(table1, true_threshold=true_threshold)
    confusion_matrix = _build_confusion_matrix(sim_summary, true_threshold=true_threshold)
    confusion_md = _format_confusion_matrix(confusion_matrix, true_threshold=true_threshold)

    # Table 2: two-stage zone/reflex strategy metrics and interval LRs.
    policy = TwoStagePolicy(lower=two_stage_lower, upper=two_stage_upper, true_threshold=true_threshold)
    two_stage_summary = _build_two_stage_summary(
        prepared,
        params,
        policy=policy,
        n_draws=n_draws,
        seed=two_stage_seed,
    )
    two_stage_md = _format_two_stage_summary(two_stage_summary, policy)
    table2 = _build_two_stage_table(two_stage_summary)
    table2_md = _format_table2(table2, policy)

    # Table 3: TcCO2 → PaCO2 prediction intervals with likelihood/prior modes.
    prediction_intervals = _build_prediction_table(
        prepared,
        params,
        tcco2_values=tcco2_values,
        thresholds=thresholds,
        seed=inference_seed,
        n_draws=n_draws,
        include_prior=include_prior,
    )
    table3_md = _format_table3(prediction_intervals, threshold=true_threshold)

    # Figure data: PaCO2 distributions and misclassification vs true PaCO2.
    paco2_bins = _build_paco2_bins(
        prepared,
        true_threshold=true_threshold,
        borderline_lower=two_stage_lower,
        borderline_upper=two_stage_upper,
    )
    misclassification_curves = _build_misclassification_curves(
        prepared,
        params,
        threshold=true_threshold,
        n_draws=n_draws,
        seed=conditional_seed,
    )

    snippets = _build_results_snippets(
        parameters=parameters,
        table1=table1,
        two_stage=table2,
        prediction_intervals=prediction_intervals,
        true_threshold=true_threshold,
        two_stage_lower=two_stage_lower,
        two_stage_upper=two_stage_upper,
    )

    markdown = {
        "parameters": parameters_md,
        "table1": table1_md,
        "confusion_matrix": confusion_md,
        "two_stage_summary": two_stage_md,
        "table2": table2_md,
        "table3": table3_md,
        "snippets": snippets,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(out_dir / "manuscript_parameters.csv", parameters)
        _write_text(out_dir / "manuscript_parameters.md", parameters_md)
        _write_csv(out_dir / "manuscript_table1.csv", table1)
        _write_text(out_dir / "manuscript_table1.md", table1_md)
        _write_csv(out_dir / "manuscript_confusion_matrix.csv", confusion_matrix)
        _write_text(out_dir / "manuscript_confusion_matrix.md", confusion_md)
        _write_csv(out_dir / "two_stage_summary.csv", two_stage_summary)
        _write_text(out_dir / "two_stage_summary.md", two_stage_md)
        _write_csv(out_dir / "manuscript_table2_two_stage.csv", table2)
        _write_text(out_dir / "manuscript_table2_two_stage.md", table2_md)
        _write_csv(out_dir / "manuscript_table3_prediction_intervals.csv", prediction_intervals)
        _write_text(out_dir / "manuscript_table3_prediction_intervals.md", table3_md)
        _write_csv(out_dir / "figure_paco2_distribution_bins.csv", paco2_bins)
        _write_csv(out_dir / "figure_misclassification_vs_paco2.csv", misclassification_curves)
        _write_text(out_dir / "manuscript_results_snippets.md", snippets)

    invariants = {
        "groups": int(table1["group"].nunique()) if not table1.empty else 0,
        "thresholds": ",".join(f"{value:g}" for value in thresholds),
        "true_threshold": float(true_threshold),
        "tcco2_values": ",".join(f"{value:g}" for value in tcco2_values),
    }
    return ManuscriptWorkflowResult(
        parameters=parameters,
        table1=table1,
        confusion_matrix=confusion_matrix,
        two_stage_summary=two_stage_summary,
        table2=table2,
        prediction_intervals=prediction_intervals,
        paco2_bins=paco2_bins,
        misclassification_curves=misclassification_curves,
        markdown=markdown,
        snippets=snippets,
        invariants=invariants,
    )


def _normalize_thresholds(thresholds: Sequence[float], true_threshold: float) -> list[float]:
    values = [float(value) for value in thresholds]
    if true_threshold not in values:
        values.append(float(true_threshold))
    return values


def _next_seed(rng: np.random.Generator | None) -> int | None:
    if rng is None:
        return None
    return int(rng.integers(0, np.iinfo(np.uint32).max))


def _summarize_parameters(params: pd.DataFrame) -> pd.DataFrame:
    params = validate_params_df(params).copy()
    if "group" in params.columns:
        params["group"] = params["group"].map(PARAM_GROUP_MAP).fillna(params["group"])
    else:
        params["group"] = "all"

    if "sd_total" not in params.columns:
        params["sd_total"] = np.sqrt(
            params["sigma2"].to_numpy(dtype=float) + params["tau2"].to_numpy(dtype=float)
        )
    if "loa_l" not in params.columns or "loa_u" not in params.columns:
        sd_within = np.sqrt(params["sigma2"].to_numpy(dtype=float))
        tau2 = params["tau2"].to_numpy(dtype=float)
        loa_bounds_arr = [
            loa_bounds(delta, within, between)
            for delta, within, between in zip(params["delta"].to_numpy(), sd_within, tau2)
        ]
        params["loa_l"] = [bounds[0] for bounds in loa_bounds_arr]
        params["loa_u"] = [bounds[1] for bounds in loa_bounds_arr]

    params["loa_width"] = params["loa_u"] - params["loa_l"]
    metrics = ["delta", "sigma2", "tau2", "sd_total", "loa_l", "loa_u", "loa_width"]
    summary = params.groupby("group")[metrics].quantile(DEFAULT_SUMMARY_QUANTILES).unstack(-1)
    summary.columns = [
        f"{metric}_{quantile_key('', quantile).lstrip('_')}" for metric, quantile in summary.columns
    ]
    summary = summary.reset_index()
    summary["label"] = summary["group"].map(MANUSCRIPT_GROUP_LABELS).fillna(summary["group"])
    return _order_groups(summary)


def _simulate_with_all(
    paco2_data: pd.DataFrame,
    params: pd.DataFrame,
    thresholds: Sequence[float],
    mode: str,
    seed: int | None,
    n_draws: int | None,
    n_mc: int | None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = simulate_forward(
        paco2_data,
        params,
        thresholds=thresholds,
        mode=mode,
        seed=_next_seed(rng),
        n_draws=n_draws,
        n_mc=n_mc,
    )

    paco2_values = paco2_data["paco2"].to_numpy(dtype=float)
    available_groups = set(params["group"]) if "group" in params.columns else set()
    group_params = _select_group_params(params, "all", available_groups)
    if n_draws is not None and n_draws < group_params.shape[0]:
        chosen = rng.choice(group_params.index.to_numpy(), size=n_draws, replace=True)
        group_params = group_params.loc[chosen].reset_index(drop=True)
    all_metrics = simulate_forward_metrics(
        paco2_values,
        group_params,
        thresholds=thresholds,
        mode=mode,
        seed=_next_seed(rng),
        n_mc=n_mc,
    )
    if "group" in all_metrics.columns:
        all_metrics = all_metrics.rename(columns={"group": "param_group"})
    all_metrics.insert(0, "group", "all")
    return pd.concat([metrics, all_metrics], ignore_index=True)


def _paco2_stats(prepared: pd.DataFrame, true_threshold: float) -> pd.DataFrame:
    patient_col = next(
        (col for col in ("patient_id", "subject_id", "patient", "subject") if col in prepared.columns),
        None,
    )
    rows: list[dict[str, float | int | str]] = []
    for group in MANUSCRIPT_GROUP_ORDER:
        subset = prepared if group == "all" else prepared[prepared["subgroup"] == group]
        if subset.empty:
            continue
        paco2_values = subset["paco2"].to_numpy(dtype=float)
        row: dict[str, float | int | str] = {
            "group": group,
            "n_encounters": int(subset.shape[0]),
            "n_patients": int(subset[patient_col].nunique()) if patient_col is not None else float("nan"),
            "paco2_q250": float(np.quantile(paco2_values, 0.25)),
            "paco2_q500": float(np.quantile(paco2_values, 0.5)),
            "paco2_q750": float(np.quantile(paco2_values, 0.75)),
            "prevalence": float(np.mean(paco2_values >= true_threshold)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _build_table1(sim_summary: pd.DataFrame, paco2_stats: pd.DataFrame, true_threshold: float) -> pd.DataFrame:
    summary = sim_summary.loc[sim_summary["threshold"] == float(true_threshold)].copy()
    summary = summary.merge(paco2_stats, on="group", how="left")
    summary["label"] = summary["group"].map(MANUSCRIPT_GROUP_LABELS).fillna(summary["group"])
    ordered = _order_groups(summary)
    metric_columns = [
        "prevalence",
        "sensitivity",
        "specificity",
        "lr_pos",
        "lr_neg",
        "ppv",
        "npv",
        "fp_rate",
        "fn_rate",
        "misclass_rate",
        "fp_per_1000",
        "fn_per_1000",
        "misclass_per_1000",
    ]
    quantile_columns = [
        f"{metric}_{suffix}"
        for metric in metric_columns
        for suffix in ("q025", "q500", "q975")
        if f"{metric}_{suffix}" in ordered.columns
    ]
    base_columns = [
        "group",
        "label",
        "threshold",
        "n_encounters",
        "n_patients",
        "paco2_q250",
        "paco2_q500",
        "paco2_q750",
    ]
    keep = [col for col in base_columns + quantile_columns if col in ordered.columns]
    return ordered[keep]


def _build_confusion_matrix(sim_summary: pd.DataFrame, true_threshold: float) -> pd.DataFrame:
    summary = sim_summary.loc[sim_summary["threshold"] == float(true_threshold)].copy()
    summary["label"] = summary["group"].map(MANUSCRIPT_GROUP_LABELS).fillna(summary["group"])
    ordered = _order_groups(summary)
    metrics = ["tp_per_1000", "fp_per_1000", "fn_per_1000", "tn_per_1000"]
    quantile_columns = [
        f"{metric}_{suffix}"
        for metric in metrics
        for suffix in ("q025", "q500", "q975")
        if f"{metric}_{suffix}" in ordered.columns
    ]
    base_columns = ["group", "label", "threshold"]
    keep = [col for col in base_columns + quantile_columns if col in ordered.columns]
    return ordered[keep]


def _build_two_stage_summary(
    prepared: pd.DataFrame,
    params: pd.DataFrame,
    policy: TwoStagePolicy,
    n_draws: int | None,
    seed: int | None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    available_groups = set(params["group"]) if "group" in params.columns else set()
    frames: list[pd.DataFrame] = []

    for subgroup in PACO2_SUBGROUP_ORDER:
        paco2_values = prepared.loc[prepared["subgroup"] == subgroup, "paco2"].to_numpy(dtype=float)
        if paco2_values.size == 0:
            continue
        group_params = _select_group_params(params, subgroup, available_groups)
        summary = summarize_two_stage_draws(
            paco2_values,
            group_params,
            policy=policy,
            n_draws=n_draws,
            seed=_next_seed(rng),
        )
        summary.insert(0, "group", subgroup)
        frames.append(summary)

    paco2_values = prepared["paco2"].to_numpy(dtype=float)
    all_params = _select_group_params(params, "all", available_groups)
    all_summary = summarize_two_stage_draws(
        paco2_values,
        all_params,
        policy=policy,
        n_draws=n_draws,
        seed=_next_seed(rng),
    )
    all_summary.insert(0, "group", "all")
    frames.append(all_summary)

    summary = pd.concat(frames, ignore_index=True)
    summary["label"] = summary["group"].map(MANUSCRIPT_GROUP_LABELS).fillna(summary["group"])
    return _order_groups(summary)


def _build_two_stage_table(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.copy()


def _build_prediction_table(
    prepared: pd.DataFrame,
    params: pd.DataFrame,
    tcco2_values: Sequence[float],
    thresholds: Sequence[float],
    seed: int | None,
    n_draws: int | None,
    include_prior: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    available_groups = set(params["group"]) if "group" in params.columns else set()
    likelihood = infer_paco2_by_subgroup(
        tcco2_values,
        prepared,
        params,
        thresholds=thresholds,
        use_prior=False,
        seed=_next_seed(rng),
        n_draws=n_draws,
    )
    prior = None
    if include_prior:
        prior = infer_paco2_by_subgroup(
            tcco2_values,
            prepared,
            params,
            thresholds=thresholds,
            use_prior=True,
            seed=_next_seed(rng),
            n_draws=n_draws,
        )

    all_paco2 = prepared["paco2"].to_numpy(dtype=float)
    all_params = _select_group_params(params, "all", available_groups)
    all_likelihood = infer_paco2(
        tcco2_values,
        all_params,
        thresholds=thresholds,
        use_prior=False,
        seed=_next_seed(rng),
        n_draws=n_draws,
    )
    all_likelihood.insert(0, "group", "all")
    likelihood = pd.concat([likelihood, all_likelihood], ignore_index=True)

    if include_prior:
        all_prior = infer_paco2(
            tcco2_values,
            all_params,
            thresholds=thresholds,
            paco2_prior=all_paco2,
            use_prior=True,
            seed=_next_seed(rng),
            n_draws=n_draws,
        )
        all_prior.insert(0, "group", "all")
        prior = pd.concat([prior, all_prior], ignore_index=True)

    likelihood = _order_groups(likelihood)
    likelihood = _rename_mode_columns(likelihood, "likelihood")
    if prior is None:
        combined = likelihood
    else:
        prior = _order_groups(prior)
        prior = _rename_mode_columns(prior, "prior")
        combined = likelihood.merge(prior, on=["group", "tcco2"], how="left")
    group_values = combined["group"].astype(str)
    combined["label"] = group_values.map(MANUSCRIPT_GROUP_LABELS).fillna(group_values)
    return combined


def _build_paco2_bins(
    prepared: pd.DataFrame,
    true_threshold: float,
    borderline_lower: float,
    borderline_upper: float,
    bin_width: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group in MANUSCRIPT_GROUP_ORDER:
        subset = prepared if group == "all" else prepared[prepared["subgroup"] == group]
        if subset.empty:
            continue
        values = subset["paco2"].to_numpy(dtype=float)
        bins = np.round(values / bin_width) * bin_width
        counts = pd.Series(bins).value_counts().sort_index()
        total = float(values.size)
        for bin_center, count in counts.items():
            weight = count / total
            rows.append(
                {
                    "group": group,
                    "bin_center": float(bin_center),
                    "count": int(count),
                    "weight": float(weight),
                    "density": float(weight / bin_width),
                    "true_threshold": float(true_threshold),
                    "borderline_lower": float(borderline_lower),
                    "borderline_upper": float(borderline_upper),
                }
            )
    return pd.DataFrame(rows)


def _build_misclassification_curves(
    prepared: pd.DataFrame,
    params: pd.DataFrame,
    threshold: float,
    n_draws: int | None,
    seed: int | None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    available_groups = set(params["group"]) if "group" in params.columns else set()
    frames: list[pd.DataFrame] = []
    for subgroup in PACO2_SUBGROUP_ORDER:
        paco2_values = prepared.loc[prepared["subgroup"] == subgroup, "paco2"].to_numpy(dtype=float)
        if paco2_values.size == 0:
            continue
        group_params = _select_group_params(params, subgroup, available_groups)
        curves = conditional_classification_curves(
            paco2_values,
            group_params,
            threshold=float(threshold),
            quantiles=DEFAULT_SUMMARY_QUANTILES,
            n_draws=n_draws,
            seed=_next_seed(rng),
        )
        curves.insert(0, "group", subgroup)
        frames.append(curves)

    all_paco2 = prepared["paco2"].to_numpy(dtype=float)
    all_params = _select_group_params(params, "all", available_groups)
    all_curves = conditional_classification_curves(
        all_paco2,
        all_params,
        threshold=float(threshold),
        quantiles=DEFAULT_SUMMARY_QUANTILES,
        n_draws=n_draws,
        seed=_next_seed(rng),
    )
    all_curves.insert(0, "group", "all")
    frames.append(all_curves)

    curves = pd.concat(frames, ignore_index=True)
    for label in ("q025", "q50", "q975"):
        curves[f"misclass_{label}"] = curves[f"fp_{label}"] + curves[f"fn_{label}"]
    return curves


def _select_group_params(
    params: pd.DataFrame,
    subgroup: str,
    available_groups: set[str],
) -> pd.DataFrame:
    if "group" not in params.columns:
        return params
    if subgroup == "all":
        group_key = "main"
    else:
        group_key = subgroup if subgroup in available_groups else PACO2_TO_CONWAY_GROUP.get(subgroup, subgroup)
    subset = params[params["group"] == group_key]
    if subset.empty:
        return params
    return subset


def _order_groups(df: pd.DataFrame) -> pd.DataFrame:
    if "group" not in df.columns:
        return df
    ordered = df.copy()
    ordered["group"] = pd.Categorical(ordered["group"], categories=MANUSCRIPT_GROUP_ORDER, ordered=True)
    sort_columns = ["group"]
    if "tcco2" in ordered.columns:
        sort_columns.append("tcco2")
    return ordered.sort_values(sort_columns).reset_index(drop=True)


def _rename_mode_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = frame.copy()
    for column in list(renamed.columns):
        if column in {"group", "tcco2"}:
            continue
        renamed = renamed.rename(columns={column: f"{prefix}_{column}"})
    return renamed


def _format_interval(row: pd.Series, metric: str, precision: int) -> str:
    values = [
        row.get(f"{metric}_q500"),
        row.get(f"{metric}_q025"),
        row.get(f"{metric}_q975"),
    ]
    if _has_nan(values):
        return "NA"
    return (
        f"{_format_value(values[0], precision)} "
        f"[{_format_value(values[1], precision)}, {_format_value(values[2], precision)}]"
    )


def _format_triplet(median: float, low: float, high: float, precision: int) -> str:
    values = [median, low, high]
    if _has_nan(values):
        return "NA"
    return (
        f"{_format_value(median, precision)} "
        f"[{_format_value(low, precision)}, {_format_value(high, precision)}]"
    )


def _format_iqr(median: float, low: float, high: float, precision: int) -> str:
    values = [median, low, high]
    if _has_nan(values):
        return "NA"
    return (
        f"{_format_value(median, precision)} "
        f"[{_format_value(low, precision)}, {_format_value(high, precision)}]"
    )


def _format_int(value: float | int | None) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{int(value)}"


def _format_parameters_table(parameters: pd.DataFrame) -> str:
    lines = [
        "# Error-model parameters used",
        "",
        "Median with 95% uncertainty interval (bootstrap percentile).",
        "",
    ]
    headers = [
        "Setting",
        "Bias δ",
        "σ² (within)",
        "τ² (between)",
        "SD total",
        "LoA L",
        "LoA U",
        "LoA width",
    ]
    rows = []
    for _, row in parameters.iterrows():
        rows.append(
            [
                row["label"],
                _format_interval(row, "delta", 2),
                _format_interval(row, "sigma2", 2),
                _format_interval(row, "tau2", 2),
                _format_interval(row, "sd_total", 2),
                _format_interval(row, "loa_l", 2),
                _format_interval(row, "loa_u", 2),
                _format_interval(row, "loa_width", 2),
            ]
        )
    lines.append(_markdown_table(headers, rows))
    return "\n".join(lines)


def _format_table1(table1: pd.DataFrame, true_threshold: float) -> str:
    lines = [
        "# Cohort + operating characteristics by setting",
        "",
        "Median with 95% CI (bootstrap percentile) for TcCO2 classification metrics.",
        f"Threshold for TcCO2 ≥ {true_threshold:g} mmHg.",
        "",
    ]
    headers = [
        "Setting",
        "N encounters",
        "N patients",
        "PaCO2 median [IQR]",
        f"P(PaCO2≥{true_threshold:g})",
        "Sensitivity",
        "Specificity",
        "LR+",
        "LR-",
        "PPV",
        "NPV",
        "FP rate",
        "FN rate",
        "Misclass rate",
        "FP/1000",
        "FN/1000",
        "Misclass/1000",
    ]
    rows = []
    for _, row in table1.iterrows():
        rows.append(
            [
                row["label"],
                _format_int(row.get("n_encounters")),
                _format_int(row.get("n_patients")),
                _format_iqr(row["paco2_q500"], row["paco2_q250"], row["paco2_q750"], 1),
                _format_interval(row, "prevalence", 3),
                _format_interval(row, "sensitivity", 3),
                _format_interval(row, "specificity", 3),
                _format_interval(row, "lr_pos", 2),
                _format_interval(row, "lr_neg", 2),
                _format_interval(row, "ppv", 3),
                _format_interval(row, "npv", 3),
                _format_interval(row, "fp_rate", 3),
                _format_interval(row, "fn_rate", 3),
                _format_interval(row, "misclass_rate", 3),
                _format_interval(row, "fp_per_1000", 1),
                _format_interval(row, "fn_per_1000", 1),
                _format_interval(row, "misclass_per_1000", 1),
            ]
        )
    lines.append(_markdown_table(headers, rows))
    return "\n".join(lines)


def _format_confusion_matrix(confusion: pd.DataFrame, true_threshold: float) -> str:
    lines = [
        "# Confusion matrix (expected per 1000 tested)",
        "",
        "Median with 95% CI (bootstrap percentile).",
        f"Threshold for TcCO2 ≥ {true_threshold:g} mmHg.",
        "",
    ]
    headers = ["Setting", "TP/1000", "FP/1000", "FN/1000", "TN/1000"]
    rows = []
    for _, row in confusion.iterrows():
        rows.append(
            [
                row["label"],
                _format_interval(row, "tp_per_1000", 1),
                _format_interval(row, "fp_per_1000", 1),
                _format_interval(row, "fn_per_1000", 1),
                _format_interval(row, "tn_per_1000", 1),
            ]
        )
    lines.append(_markdown_table(headers, rows))
    return "\n".join(lines)


def _format_two_stage_summary(summary: pd.DataFrame, policy: TwoStagePolicy) -> str:
    lines = [
        "# Two-stage strategy summary",
        "",
        "Median with 95% CI (bootstrap percentile).",
        f"Zones: <{policy.lower:g}, {policy.lower:g}–{policy.upper:g}, >{policy.upper:g} mmHg.",
        f"True hypercapnia threshold: {policy.true_threshold:g} mmHg.",
        "",
    ]
    headers = [
        "Setting",
        "Zone1 P",
        "Zone2 P",
        "Zone3 P",
        "LR Zone1",
        "LR Zone2",
        "LR Zone3",
        "P(pos|Zone1)",
        "P(pos|Zone2)",
        "P(pos|Zone3)",
        "Reflex fraction",
        "Residual misclass/1000",
    ]
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            [
                row["label"],
                _format_interval(row, "zone1_prob", 3),
                _format_interval(row, "zone2_prob", 3),
                _format_interval(row, "zone3_prob", 3),
                _format_interval(row, "zone1_lr", 2),
                _format_interval(row, "zone2_lr", 2),
                _format_interval(row, "zone3_lr", 2),
                _format_interval(row, "zone1_post_prob", 3),
                _format_interval(row, "zone2_post_prob", 3),
                _format_interval(row, "zone3_post_prob", 3),
                _format_interval(row, "reflex_fraction", 3),
                _format_interval(row, "residual_misclass_per_1000", 1),
            ]
        )
    lines.append(_markdown_table(headers, rows))
    return "\n".join(lines)


def _format_table2(table2: pd.DataFrame, policy: TwoStagePolicy) -> str:
    lines = [
        "# Two-stage strategy (manuscript table)",
        "",
        "Median with 95% CI (bootstrap percentile).",
        f"Zones: <{policy.lower:g}, {policy.lower:g}–{policy.upper:g}, >{policy.upper:g} mmHg.",
        f"True hypercapnia threshold: {policy.true_threshold:g} mmHg.",
        "",
    ]
    headers = [
        "Setting",
        "Zone1 P",
        "Zone2 P",
        "Zone3 P",
        "LR Zone1",
        "LR Zone2",
        "LR Zone3",
        "P(pos|Zone1)",
        "P(pos|Zone2)",
        "P(pos|Zone3)",
        "Reflex fraction",
        "Residual misclass/1000",
    ]
    rows = []
    for _, row in table2.iterrows():
        rows.append(
            [
                row["label"],
                _format_interval(row, "zone1_prob", 3),
                _format_interval(row, "zone2_prob", 3),
                _format_interval(row, "zone3_prob", 3),
                _format_interval(row, "zone1_lr", 2),
                _format_interval(row, "zone2_lr", 2),
                _format_interval(row, "zone3_lr", 2),
                _format_interval(row, "zone1_post_prob", 3),
                _format_interval(row, "zone2_post_prob", 3),
                _format_interval(row, "zone3_post_prob", 3),
                _format_interval(row, "reflex_fraction", 3),
                _format_interval(row, "residual_misclass_per_1000", 1),
            ]
        )
    lines.append(_markdown_table(headers, rows))
    return "\n".join(lines)


def _format_table3(prediction_intervals: pd.DataFrame, threshold: float) -> str:
    label = threshold_label(threshold)
    lines = [
        "# TcCO2 → PaCO2 prediction intervals",
        "",
        "Median with 95% prediction interval (PI).",
        f"P(PaCO2≥{threshold:g}) reported for each mode.",
        "",
    ]
    headers = [
        "Setting",
        "TcCO2",
        "Likelihood PaCO2 median [PI]",
        f"Likelihood P(PaCO2≥{threshold:g})",
        "Prior-weighted PaCO2 median [PI]",
        f"Prior-weighted P(PaCO2≥{threshold:g})",
    ]
    rows = []
    for _, row in prediction_intervals.iterrows():
        rows.append(
            [
                row["label"],
                f"{row['tcco2']:.0f}",
                _format_triplet(
                    row.get("likelihood_paco2_q500"),
                    row.get("likelihood_paco2_q025"),
                    row.get("likelihood_paco2_q975"),
                    2,
                ),
                _format_number(row.get(f"likelihood_p_ge_{label}"), 3),
                _format_triplet(
                    row.get("prior_paco2_q500"),
                    row.get("prior_paco2_q025"),
                    row.get("prior_paco2_q975"),
                    2,
                ),
                _format_number(row.get(f"prior_p_ge_{label}"), 3),
            ]
        )
    lines.append(_markdown_table(headers, rows))
    return "\n".join(lines)


def _format_number(value: float | int | None, precision: int) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{precision}f}"


def _format_value(value: float | int | None, precision: int) -> str:
    if pd.isna(value):
        return "NA"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{precision}f}"


def _build_results_snippets(
    parameters: pd.DataFrame,
    table1: pd.DataFrame,
    two_stage: pd.DataFrame,
    prediction_intervals: pd.DataFrame,
    true_threshold: float,
    two_stage_lower: float,
    two_stage_upper: float,
) -> str:
    snippets = ["# Manuscript results snippets", ""]

    param_lines = []
    for _, row in parameters.iterrows():
        param_lines.append(
            f"{row['label']}: δ={_format_interval(row, 'delta', 2)} mmHg; "
            f"σ²={_format_interval(row, 'sigma2', 2)}, τ²={_format_interval(row, 'tau2', 2)}; "
            f"LoA { _format_interval(row, 'loa_l', 2) } to { _format_interval(row, 'loa_u', 2) }."
        )
    snippets.extend(
        [
            "## Error-model parameters used",
            "95% uncertainty interval (bootstrap percentile).",
            " ".join(param_lines),
            "",
        ]
    )

    op_lines = []
    for _, row in table1.iterrows():
        op_lines.append(
            f"{row['label']}: Se={_format_interval(row, 'sensitivity', 3)}, "
            f"Sp={_format_interval(row, 'specificity', 3)}, "
            f"LR+={_format_interval(row, 'lr_pos', 2)}, LR-={_format_interval(row, 'lr_neg', 2)}; "
            f"FP/1000={_format_interval(row, 'fp_per_1000', 1)}, "
            f"FN/1000={_format_interval(row, 'fn_per_1000', 1)}."
        )
    snippets.extend(
        [
            "## Operating characteristics",
            "95% CI (bootstrap percentile); TcCO2 threshold aligns to PaCO2 ≥"
            f" {true_threshold:g} mmHg.",
            " ".join(op_lines),
            "",
        ]
    )

    two_stage_lines = []
    for _, row in two_stage.iterrows():
        two_stage_lines.append(
            f"{row['label']}: zones P(<{two_stage_lower:g})={_format_interval(row, 'zone1_prob', 3)}, "
            f"P({two_stage_lower:g}–{two_stage_upper:g})={_format_interval(row, 'zone2_prob', 3)}, "
            f"P(>{two_stage_upper:g})={_format_interval(row, 'zone3_prob', 3)}; "
            f"LRs={_format_interval(row, 'zone1_lr', 2)}/"
            f"{_format_interval(row, 'zone2_lr', 2)}/"
            f"{_format_interval(row, 'zone3_lr', 2)}; "
            f"residual misclass/1000={_format_interval(row, 'residual_misclass_per_1000', 1)}."
        )
    snippets.extend(
        [
            "## Two-stage strategy",
            "95% CI (bootstrap percentile).",
            " ".join(two_stage_lines),
            "",
        ]
    )

    pred_lines = []
    overall = prediction_intervals[prediction_intervals["group"] == "all"]
    for _, row in overall.iterrows():
        pred_lines.append(
            f"TcCO2 {row['tcco2']:.0f}: likelihood { _format_triplet(row.get('likelihood_paco2_q500'), row.get('likelihood_paco2_q025'), row.get('likelihood_paco2_q975'), 2) }, "
            f"P(PaCO2≥{true_threshold:g})={_format_number(row.get(f'likelihood_p_ge_{threshold_label(true_threshold)}'), 3)}; "
            f"prior-weighted { _format_triplet(row.get('prior_paco2_q500'), row.get('prior_paco2_q025'), row.get('prior_paco2_q975'), 2) }, "
            f"P(PaCO2≥{true_threshold:g})={_format_number(row.get(f'prior_p_ge_{threshold_label(true_threshold)}'), 3)}."
        )
    snippets.extend(
        [
            "## Prediction intervals",
            "95% prediction intervals (PI) shown for likelihood-only and prior-weighted modes.",
            " ".join(pred_lines) if pred_lines else "No prediction rows available.",
        ]
    )

    return "\n".join(snippets)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _has_nan(values: Sequence[float | int | None]) -> bool:
    return any(pd.isna(value) for value in values)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
