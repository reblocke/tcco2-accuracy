"""I/O utilities for artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_group_draws
from .conway_meta import conway_group_summary
from .core._params import ParameterFallback
from .data import load_conway_group
from .simulation import (
    DEFAULT_CLASSIFICATION_THRESHOLDS,
    DEFAULT_SUMMARY_QUANTILES,
    simulate_forward,
    summarize_simulation_metrics,
)

CONWAY_GROUPS: dict[str, str] = {
    "main": "main",
    "icu": "icu",
    "arf": "arf",
    "lft": "lft",
}


def build_bootstrap_params(
    n_boot: int = 1000,
    seed: int = 202401,
    bootstrap_mode: str = "cluster_plus_withinstudy",
) -> pd.DataFrame:
    """Generate bootstrap draws for Conway subgroups."""

    group_data = [(name, load_conway_group(key)) for name, key in CONWAY_GROUPS.items()]
    return bootstrap_group_draws(
        group_data,
        n_boot=n_boot,
        seed=seed,
        truncate_tau2=True,
        bootstrap_mode=bootstrap_mode,
    )


def write_bootstrap_params(path: Path, params: pd.DataFrame) -> None:
    """Write bootstrap draws to CSV or Parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        params.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        params.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported bootstrap params format: {path.suffix}")


def bootstrap_loa_summary(
    params: pd.DataFrame,
    conway_path: Path | None = None,
    *,
    data_by_group: Iterable[tuple[str, pd.DataFrame]] | None = None,
    truncate_tau2: bool = True,
) -> pd.DataFrame:
    """Summarize bootstrap LoA bounds versus corrected analytic CIs.

    When ``data_by_group`` is supplied, analytic comparators are calculated from
    those exact group frames. Direct callers may omit it to retain path-backed
    loading of the canonical Conway groups.
    """

    rows: list[dict[str, float | str]] = []
    bootstrap_mode = _extract_bootstrap_mode(params)
    method_version = _extract_single_value(params, "agreement_method_version")
    results_status = _extract_single_value(params, "results_status")
    analytic_groups = (
        [
            (group_name, load_conway_group(group_key, path=conway_path))
            for group_name, group_key in CONWAY_GROUPS.items()
        ]
        if data_by_group is None
        else list(data_by_group)
    )
    for group_name, group_data in analytic_groups:
        subset = params[params["group"] == group_name]
        if subset.empty:
            continue
        loa_l_q = subset["loa_l"].quantile([0.025, 0.5, 0.975])
        loa_u_q = subset["loa_u"].quantile([0.025, 0.5, 0.975])
        summary = conway_group_summary(group_data, truncate_tau2=truncate_tau2)
        bootstrap_outer_width = float(loa_u_q.loc[0.975] - loa_l_q.loc[0.025])
        corrected_analytic_outer_width = float(summary.ci_u - summary.ci_l)
        width_ratio = (
            bootstrap_outer_width / corrected_analytic_outer_width
            if np.isfinite(corrected_analytic_outer_width) and corrected_analytic_outer_width != 0
            else float("nan")
        )
        rows.append(
            {
                "group": group_name,
                "loa_l_q025": float(loa_l_q.loc[0.025]),
                "loa_l_q50": float(loa_l_q.loc[0.5]),
                "loa_l_q975": float(loa_l_q.loc[0.975]),
                "loa_u_q025": float(loa_u_q.loc[0.025]),
                "loa_u_q50": float(loa_u_q.loc[0.5]),
                "loa_u_q975": float(loa_u_q.loc[0.975]),
                "corrected_analytic_loa_l": summary.loa_l,
                "corrected_analytic_loa_u": summary.loa_u,
                "corrected_analytic_ci_l": summary.ci_l,
                "corrected_analytic_ci_u": summary.ci_u,
                "bootstrap_outer_width": bootstrap_outer_width,
                "corrected_analytic_outer_width": corrected_analytic_outer_width,
                "width_ratio": float(width_ratio),
                "width_gap": float(corrected_analytic_outer_width - bootstrap_outer_width),
                "n_boot": int(subset.shape[0]),
                **({"bootstrap_mode": bootstrap_mode} if bootstrap_mode is not None else {}),
                **(
                    {"agreement_method_version": method_version}
                    if method_version is not None
                    else {}
                ),
                **({"results_status": results_status} if results_status is not None else {}),
            }
        )

    return pd.DataFrame(rows)


def format_bootstrap_summary(
    summary: pd.DataFrame,
    n_boot: int,
    seed: int,
    bootstrap_mode: str,
) -> str:
    """Return a markdown summary of bootstrap LoA spread."""

    lines = [
        "# Bootstrap LoA spread summary",
        "",
        f"Bootstrap draws: {n_boot} per subgroup (seed={seed}).",
        f"Bootstrap mode: {bootstrap_mode}.",
    ]
    method_version = _extract_single_value(summary, "agreement_method_version")
    results_status = _extract_single_value(summary, "results_status")
    if method_version is not None:
        lines.append(f"Agreement method version: `{method_version}`.")
    if results_status is not None:
        lines.append(f"Results status: `{results_status}`.")
    lines.extend(
        [
            "",
            "LoA bounds shown as 2.5/50/97.5% bootstrap quantiles;",
            "corrected analytic CI shown as outer CI bounds from the same method revision.",
            "",
            "| Group | LoA L q2.5 | LoA L q50 | LoA L q97.5 | LoA U q2.5 | LoA U q50 | LoA U q97.5 | Corrected analytic CI L | Corrected analytic CI U | Width ratio | Width gap |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group"]),
                    f"{row['loa_l_q025']:.2f}",
                    f"{row['loa_l_q50']:.2f}",
                    f"{row['loa_l_q975']:.2f}",
                    f"{row['loa_u_q025']:.2f}",
                    f"{row['loa_u_q50']:.2f}",
                    f"{row['loa_u_q975']:.2f}",
                    f"{row['corrected_analytic_ci_l']:.2f}",
                    f"{row['corrected_analytic_ci_u']:.2f}",
                    f"{row['width_ratio']:.2f}",
                    f"{row['width_gap']:.2f}",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("Width interpretation (bootstrap vs corrected analytic outer CI):")
    for _, row in summary.iterrows():
        interpretation = _interpret_width_ratio(float(row["width_ratio"]))
        lines.append(f"- {row['group']}: {interpretation}.")

    return "\n".join(lines)


def _extract_bootstrap_mode(params: pd.DataFrame) -> str | None:
    return _extract_single_value(params, "bootstrap_mode")


def _extract_single_value(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    values = pd.Series(frame[column]).dropna().astype(str).unique()
    if values.size != 1:
        return None
    return str(values[0])


def _interpret_width_ratio(width_ratio: float) -> str:
    if not np.isfinite(width_ratio):
        return "ratio unavailable"
    if width_ratio < 0.8:
        return "materially narrower than the corrected analytic CI"
    if width_ratio <= 1.2:
        return "comparable to the corrected analytic CI"
    return "wider than the corrected analytic CI"


def build_simulation_summary(
    params: pd.DataFrame,
    paco2_data: pd.DataFrame,
    thresholds: Sequence[float] = DEFAULT_CLASSIFICATION_THRESHOLDS,
    mode: str = "analytic",
    seed: int | None = None,
    n_draws: int | None = None,
    n_mc: int | None = None,
    fallback: ParameterFallback = "error",
) -> pd.DataFrame:
    """Generate forward simulation summaries by subgroup."""

    metrics = simulate_forward(
        paco2_data,
        params,
        thresholds=thresholds,
        mode=mode,
        seed=seed,
        n_draws=n_draws,
        n_mc=n_mc,
        fallback=fallback,
    )
    return summarize_simulation_metrics(metrics, quantiles=DEFAULT_SUMMARY_QUANTILES)


def format_simulation_summary(
    summary: pd.DataFrame,
    thresholds: Sequence[float],
    n_boot: int,
    mode: str,
) -> str:
    """Return a markdown summary of forward simulation outputs."""

    if summary.empty:
        return "# Forward simulation summary\n\nNo simulation rows available."

    threshold_label = ", ".join(f"{value:.0f}" for value in thresholds)
    lines = [
        "# Forward simulation summary",
        "",
        f"Bootstrap draws: {n_boot} per subgroup.",
        f"Mode: {mode}. Thresholds (mmHg): {threshold_label}.",
        f"Parameter routing: {_format_parameter_routes(summary)}.",
        "",
        "Median values shown with [2.5%, 97.5%] bootstrap intervals.",
        "",
        "## d distribution + LoA",
        "",
        "| Group | d mean | d SD | d q2.5 | d q97.5 | LoA L | LoA U |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    summary_no_threshold = summary.drop_duplicates(subset=["group"])
    for _, row in summary_no_threshold.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group"]),
                    _format_interval(row, "d_mean", precision=2),
                    _format_interval(row, "d_sd", precision=2),
                    _format_interval(row, "d_q025", precision=2),
                    _format_interval(row, "d_q975", precision=2),
                    _format_interval(row, "loa_l", precision=2),
                    _format_interval(row, "loa_u", precision=2),
                ]
            )
            + " |"
        )

    classification_metrics = [
        ("prevalence", "Prevalence", 3),
        ("sensitivity", "Sensitivity", 3),
        ("specificity", "Specificity", 3),
        ("ppv", "PPV", 3),
        ("npv", "NPV", 3),
        ("accuracy", "Accuracy", 3),
    ]
    if "lr_pos_q500" in summary.columns and "lr_neg_q500" in summary.columns:
        classification_metrics.extend([("lr_pos", "LR+", 2), ("lr_neg", "LR-", 2)])
    lines.extend(
        [
            "",
            "## Classification metrics",
            "",
            "| Group | Threshold | "
            + " | ".join(metric[1] for metric in classification_metrics)
            + " |",
            "| --- | --- | " + " | ".join(["---"] * len(classification_metrics)) + " |",
        ]
    )

    for _, row in summary.iterrows():
        formatted_metrics = [
            _format_interval(row, metric, precision=precision)
            for metric, _, precision in classification_metrics
        ]
        lines.append(
            "| "
            + " | ".join([str(row["group"]), f"{row['threshold']:.0f}", *formatted_metrics])
            + " |"
        )

    if "misclass_rate_q500" in summary.columns:
        lines.extend(
            [
                "",
                "## Misclassification burden",
                "",
                "| Group | Threshold | FP rate | FN rate | Misclass rate | FP/1000 | FN/1000 | Misclass/1000 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for _, row in summary.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["group"]),
                        f"{row['threshold']:.0f}",
                        _format_interval(row, "fp_rate", precision=3),
                        _format_interval(row, "fn_rate", precision=3),
                        _format_interval(row, "misclass_rate", precision=3),
                        _format_interval(row, "fp_per_1000", precision=1),
                        _format_interval(row, "fn_per_1000", precision=1),
                        _format_interval(row, "misclass_per_1000", precision=1),
                    ]
                )
                + " |"
            )

    return "\n".join(lines)


def _format_parameter_routes(frame: pd.DataFrame) -> str:
    if not {"requested_group", "parameter_group_used"}.issubset(frame.columns):
        return "unavailable"
    routes = frame[["requested_group", "parameter_group_used"]].drop_duplicates()
    return ", ".join(
        f"{row.requested_group}->{row.parameter_group_used}" for row in routes.itertuples()
    )


def _format_interval(row: pd.Series, metric: str, precision: int) -> str:
    return (
        f"{row[f'{metric}_q500']:.{precision}f} "
        f"[{row[f'{metric}_q025']:.{precision}f}, {row[f'{metric}_q975']:.{precision}f}]"
    )
