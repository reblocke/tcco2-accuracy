"""Workflow helpers for TcCO2 → PaCO2 inference demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..data import PACO2_SUBGROUP_ORDER, load_paco2_distribution, prepare_paco2_distribution
from ..inference import infer_paco2_by_subgroup
from ..simulation import DEFAULT_CLASSIFICATION_THRESHOLDS
from ..utils import n_draws_per_group, threshold_label
from . import bootstrap as bootstrap_workflow


DEFAULT_TCCO2_VALUES: tuple[float, ...] = (35.0, 45.0, 55.0)


@dataclass(frozen=True)
class InferenceWorkflowResult:
    likelihood: pd.DataFrame
    prior: pd.DataFrame | None
    summary: pd.DataFrame
    invariants: dict[str, float | int | str]
    markdown: str


def run_inference_demo(
    tcco2_values: Sequence[float] = DEFAULT_TCCO2_VALUES,
    params: pd.DataFrame | None = None,
    paco2_data: pd.DataFrame | None = None,
    paco2_path: Path | None = None,
    conway_path: Path | None = None,
    thresholds: Sequence[float] = DEFAULT_CLASSIFICATION_THRESHOLDS,
    seed: int | None = None,
    n_boot: int = 1000,
    n_draws: int | None = None,
    include_prior: bool = True,
    out_dir: Path | None = None,
) -> InferenceWorkflowResult:
    """Run inference demo tables for TcCO2 → PaCO2.

    Reads:
        - Conway bootstrap parameters (if ``params`` is not provided).
        - PaCO2 distribution data (if ``paco2_data`` is not provided).

    Writes:
        - ``inference_demo.md`` in ``out_dir`` when provided.

    Returns:
        ``InferenceWorkflowResult`` with likelihood-only and optional prior-weighted
        summaries. Output frames include ``group``, ``tcco2``, PaCO2 quantiles, and
        threshold exceedance probabilities.

    Determinism:
        Fully deterministic for fixed ``seed`` and parameter draws; formatting
        requires a single threshold value.
    """

    if params is None:
        params = bootstrap_workflow.run_bootstrap(n_boot=n_boot, seed=seed, conway_path=conway_path).draws
    if paco2_data is None:
        paco2_data = load_paco2_distribution(paco2_path)
    likelihood = infer_paco2_by_subgroup(
        tcco2_values,
        paco2_data,
        params,
        thresholds=thresholds,
        use_prior=False,
        seed=seed,
        n_draws=n_draws,
    )
    prior = None
    if include_prior:
        prior = infer_paco2_by_subgroup(
            tcco2_values,
            paco2_data,
            params,
            thresholds=thresholds,
            use_prior=True,
            seed=seed,
            n_draws=n_draws,
        )
    summary = _combine_modes(likelihood, prior)
    markdown = format_inference_demo(
        likelihood,
        prior,
        thresholds=thresholds,
        n_boot=n_draws_per_group(params),
        n_draws=n_draws,
        seed=seed,
        paco2_data=paco2_data,
    )
    if out_dir is not None:
        _write_text(Path(out_dir) / "inference_demo.md", markdown)
    invariants = {
        "groups": int(summary["group"].nunique()) if not summary.empty else 0,
        "tcco2_values": ",".join(f"{value:g}" for value in tcco2_values),
        "include_prior": str(include_prior).lower(),
    }
    return InferenceWorkflowResult(
        likelihood=likelihood,
        prior=prior,
        summary=summary,
        invariants=invariants,
        markdown=markdown,
    )


def format_inference_demo(
    likelihood: pd.DataFrame,
    prior: pd.DataFrame | None,
    thresholds: Sequence[float],
    n_boot: int,
    n_draws: int | None,
    seed: int | None,
    paco2_data: pd.DataFrame,
) -> str:
    threshold_list = list(thresholds)
    if not threshold_list:
        raise ValueError("Inference thresholds must be non-empty.")
    if len(threshold_list) != 1:
        raise ValueError(
            "format_inference_demo supports one threshold; pass a single threshold or extend formatting."
        )
    threshold = threshold_list[0]
    threshold_col = f"p_ge_{threshold_label(threshold)}"
    prepared = paco2_data if "subgroup" in paco2_data.columns else prepare_paco2_distribution(paco2_data)
    prior_counts = prepared.groupby("subgroup")["paco2"].count()
    counts = ", ".join(
        f"{group} (n={int(prior_counts.get(group, 0))})" for group in PACO2_SUBGROUP_ORDER
    )
    seed_label = "none" if seed is None else str(seed)
    draw_label = "full bootstrap set per subgroup" if n_draws is None else f"{n_draws} draws per subgroup"
    lines = [
        "# TcCO2 → PaCO2 inference demo",
        "",
        f"Bootstrap draws: {n_boot} per subgroup (seed={seed_label}).",
        f"Parameter draws: {draw_label}.",
        f"Subgroup priors: empirical PaCO2 distributions for {counts}.",
        "Interval type: 95% prediction interval (PI), not CI.",
        "",
        "## Likelihood-only (bootstrap mixture)",
        "",
        f"| Group | TcCO2 | PaCO2 median [PI] | P(PaCO2≥{threshold:g}) |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in likelihood.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group"]),
                    f"{row['tcco2']:.0f}",
                    _format_interval(row),
                    f"{row[threshold_col]:.3f}",
                ]
            )
            + " |"
        )
    if prior is None:
        return "\n".join(lines)
    lines.extend(
        [
            "",
            "## Prior-weighted (empirical PaCO2 prior)",
            "",
            f"| Group | TcCO2 | PaCO2 median [PI] | P(PaCO2≥{threshold:g}) |",
            "| --- | --- | --- | --- |",
        ]
    )
    for _, row in prior.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group"]),
                    f"{row['tcco2']:.0f}",
                    _format_interval(row),
                    f"{row[threshold_col]:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _combine_modes(likelihood: pd.DataFrame, prior: pd.DataFrame | None) -> pd.DataFrame:
    frames = [likelihood.assign(mode="likelihood")]
    if prior is not None:
        frames.append(prior.assign(mode="prior"))
    return pd.concat(frames, ignore_index=True)


def _format_interval(row: pd.Series) -> str:
    return f"{row['paco2_q500']:.2f} [{row['paco2_q025']:.2f}, {row['paco2_q975']:.2f}]"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
