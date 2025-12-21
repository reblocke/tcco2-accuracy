"""I/O utilities for artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_group_draws
from .conway_meta import conway_group_summary
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


def bootstrap_loa_summary(params: pd.DataFrame, conway_path: Path | None = None) -> pd.DataFrame:
    """Summarize bootstrap LoA bounds versus Conway CIs."""

    rows: list[dict[str, float | str]] = []
    bootstrap_mode = _extract_bootstrap_mode(params)
    for group_name, group_key in CONWAY_GROUPS.items():
        subset = params[params["group"] == group_name]
        if subset.empty:
            continue
        loa_l_q = subset["loa_l"].quantile([0.025, 0.5, 0.975])
        loa_u_q = subset["loa_u"].quantile([0.025, 0.5, 0.975])
        summary = conway_group_summary(load_conway_group(group_key, path=conway_path))
        bootstrap_outer_width = float(loa_u_q.loc[0.975] - loa_l_q.loc[0.025])
        conway_outer_width = float(summary.ci_u - summary.ci_l)
        width_ratio = (
            bootstrap_outer_width / conway_outer_width
            if np.isfinite(conway_outer_width) and conway_outer_width != 0
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
                "conway_loa_l": summary.loa_l,
                "conway_loa_u": summary.loa_u,
                "conway_ci_l": summary.ci_l,
                "conway_ci_u": summary.ci_u,
                "bootstrap_outer_width": bootstrap_outer_width,
                "conway_outer_width": conway_outer_width,
                "width_ratio": float(width_ratio),
                "width_gap": float(conway_outer_width - bootstrap_outer_width),
                "n_boot": int(subset.shape[0]),
                **({"bootstrap_mode": bootstrap_mode} if bootstrap_mode is not None else {}),
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
        "",
        "LoA bounds shown as 2.5/50/97.5% bootstrap quantiles;",
        "Conway CI shown as reported outer CI bounds.",
        "",
        "| Group | LoA L q2.5 | LoA L q50 | LoA L q97.5 | LoA U q2.5 | LoA U q50 | LoA U q97.5 | Conway CI L | Conway CI U | Width ratio | Width gap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

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
                    f"{row['conway_ci_l']:.2f}",
                    f"{row['conway_ci_u']:.2f}",
                    f"{row['width_ratio']:.2f}",
                    f"{row['width_gap']:.2f}",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("Width interpretation (bootstrap vs Conway outer CI):")
    for _, row in summary.iterrows():
        interpretation = _interpret_width_ratio(float(row["width_ratio"]))
        lines.append(f"- {row['group']}: {interpretation}.")

    return "\n".join(lines)


def _extract_bootstrap_mode(params: pd.DataFrame) -> str | None:
    if "bootstrap_mode" not in params.columns:
        return None
    modes = pd.Series(params["bootstrap_mode"]).dropna().unique()
    if modes.size != 1:
        return None
    return str(modes[0])


def _interpret_width_ratio(width_ratio: float) -> str:
    if not np.isfinite(width_ratio):
        return "ratio unavailable"
    if width_ratio < 0.8:
        return "materially narrower than Conway CI"
    if width_ratio <= 1.2:
        return "comparable to Conway CI"
    return "wider than Conway CI"


def build_simulation_summary(
    params: pd.DataFrame,
    paco2_data: pd.DataFrame,
    thresholds: Sequence[float] = DEFAULT_CLASSIFICATION_THRESHOLDS,
    mode: str = "analytic",
    seed: int | None = None,
    n_draws: int | None = None,
    n_mc: int | None = None,
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

    lines.extend(
        [
            "",
            "## Classification metrics",
            "",
            "| Group | Threshold | Prevalence | Sensitivity | Specificity | PPV | NPV | Accuracy |",
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
                    _format_interval(row, "prevalence", precision=3),
                    _format_interval(row, "sensitivity", precision=3),
                    _format_interval(row, "specificity", precision=3),
                    _format_interval(row, "ppv", precision=3),
                    _format_interval(row, "npv", precision=3),
                    _format_interval(row, "accuracy", precision=3),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def _format_interval(row: pd.Series, metric: str, precision: int) -> str:
    return (
        f"{row[f'{metric}_q500']:.{precision}f} "
        f"[{row[f'{metric}_q025']:.{precision}f}, {row[f'{metric}_q975']:.{precision}f}]"
    )
