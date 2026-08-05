"""Workflow helpers for conditional misclassification curves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .._files import write_text
from .._params import ParameterFallback, select_group_params
from ..conditional import conditional_classification_curves
from ..data import PACO2_SUBGROUP_ORDER, load_paco2_distribution, prepare_paco2_distribution
from ..simulation import DEFAULT_SUMMARY_QUANTILES
from ..utils import n_draws_per_group, threshold_label
from . import bootstrap as bootstrap_workflow
from ._private_output import require_private_output_path


@dataclass(frozen=True)
class ConditionalWorkflowResult:
    curves: pd.DataFrame
    invariants: dict[str, float | int | str]
    markdown: str


def run_conditional_classification(
    params: pd.DataFrame | None = None,
    paco2_data: pd.DataFrame | None = None,
    paco2_path: Path | None = None,
    conway_path: Path | None = None,
    threshold: float = 45.0,
    bin_width: float = 1.0,
    bin_method: str = "cut",
    seed: int | None = None,
    n_boot: int = 1000,
    bootstrap_mode: str = "cluster_plus_withinstudy",
    n_draws: int | None = None,
    out_dir: Path | None = None,
    fallback: ParameterFallback = "error",
) -> ConditionalWorkflowResult:
    """Run conditional TN/FP/FN/TP curves by PaCO2 bin.

    Reads:
        - Conway bootstrap parameters (if ``params`` is not provided).
        - PaCO2 distribution data from explicit ``paco2_data`` or ``paco2_path``.

    Writes:
        - ``conditional_classification_t{threshold}.csv`` and ``.md`` in ``out_dir`` when provided.

    Notes:
        ``bootstrap_mode`` controls how parameter uncertainty is propagated into
        the conditional misclassification curves.
    """

    if out_dir is not None:
        out_dir = require_private_output_path(out_dir)

    if paco2_data is None and paco2_path is None:
        raise ValueError("Provide paco2_data or an explicit private paco2_path.")
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

    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for subgroup in PACO2_SUBGROUP_ORDER:
        paco2_values = prepared.loc[prepared["subgroup"] == subgroup, "paco2"].to_numpy(dtype=float)
        if paco2_values.size == 0:
            continue
        group_params = select_group_params(params, subgroup, fallback=fallback)
        group_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        curves = conditional_classification_curves(
            paco2_values,
            group_params,
            threshold=threshold,
            bin_width=bin_width,
            bin_method=bin_method,
            quantiles=DEFAULT_SUMMARY_QUANTILES,
            n_draws=n_draws,
            seed=group_seed,
        )
        curves.insert(0, "group", subgroup)
        curves.insert(1, "requested_group", str(group_params["requested_group"].iloc[0]))
        curves.insert(
            2,
            "parameter_group_used",
            str(group_params["parameter_group_used"].iloc[0]),
        )
        frames.append(curves)

    curves = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    markdown = format_conditional_summary(
        curves,
        threshold=threshold,
        bin_width=bin_width,
        bin_method=bin_method,
        n_boot=n_draws_per_group(params) if params is not None else 0,
        seed=seed,
        bootstrap_mode=bootstrap_mode,
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        token = threshold_label(threshold)
        curves.to_csv(out_dir / f"conditional_classification_t{token}.csv", index=False)
        write_text(out_dir / f"conditional_classification_t{token}.md", markdown)

    invariants = {
        "groups": int(curves["group"].nunique()) if not curves.empty else 0,
        "num_bins": int(curves["paco2_bin"].nunique()) if not curves.empty else 0,
        "threshold": float(threshold),
        "bootstrap_mode": bootstrap_mode,
    }
    return ConditionalWorkflowResult(curves=curves, invariants=invariants, markdown=markdown)


def format_conditional_summary(
    curves: pd.DataFrame,
    threshold: float,
    bin_width: float,
    bin_method: str,
    n_boot: int,
    seed: int | None,
    bootstrap_mode: str,
) -> str:
    seed_label = "none" if seed is None else str(seed)
    lines = [
        "# Conditional misclassification curves",
        "",
        f"Threshold (mmHg): {threshold:g}.",
        f"Bin width: {bin_width:g} ({bin_method}).",
        f"Bootstrap draws: {n_boot} per subgroup (seed={seed_label}).",
        f"Bootstrap mode: {bootstrap_mode}.",
        f"Parameter routing: {_format_parameter_routes(curves)}.",
        "",
        "Each row corresponds to a half-open PaCO2 bin [paco2_bin, paco2_bin_upper) "
        "with empirical count/weight.",
        "Truth and test-positive probability are calculated from each original unbinned PaCO2 "
        "value before display-bin aggregation.",
        "TN/FP/FN/TP columns report bootstrap quantiles of conditional probabilities.",
    ]
    if curves.empty:
        lines.append("")
        lines.append("No conditional curve rows available.")
        return "\n".join(lines)
    lines.extend(
        [
            "",
            "Columns: group, requested_group, parameter_group_used, threshold, "
            "paco2_bin, paco2_bin_upper, count, weight,",
            "tn_q025/tn_q50/tn_q975, fp_q025/fp_q50/fp_q975,",
            "fn_q025/fn_q50/fn_q975, tp_q025/tp_q50/tp_q975.",
        ]
    )
    return "\n".join(lines)


def _format_parameter_routes(frame: pd.DataFrame) -> str:
    if frame.empty or not {"requested_group", "parameter_group_used"}.issubset(frame.columns):
        return "unavailable"
    routes = frame[["requested_group", "parameter_group_used"]].drop_duplicates()
    return ", ".join(
        f"{row.requested_group}->{row.parameter_group_used}" for row in routes.itertuples()
    )
