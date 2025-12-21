"""Workflow helpers for conditional misclassification curves."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..conditional import conditional_classification_curves
from ..data import PACO2_SUBGROUP_ORDER, load_paco2_distribution, prepare_paco2_distribution
from ..simulation import DEFAULT_SUMMARY_QUANTILES, PACO2_TO_CONWAY_GROUP
from ..utils import n_draws_per_group, threshold_label
from . import bootstrap as bootstrap_workflow


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
    bin_method: str = "round",
    seed: int | None = None,
    n_boot: int = 1000,
    bootstrap_mode: str = "cluster_plus_withinstudy",
    n_draws: int | None = None,
    out_dir: Path | None = None,
) -> ConditionalWorkflowResult:
    """Run conditional TN/FP/FN/TP curves by PaCO2 bin.

    Reads:
        - Conway bootstrap parameters (if ``params`` is not provided).
        - PaCO2 distribution data (if ``paco2_data`` is not provided).

    Writes:
        - ``conditional_classification_t{threshold}.csv`` and ``.md`` in ``out_dir`` when provided.

    Notes:
        ``bootstrap_mode`` controls how parameter uncertainty is propagated into
        the conditional misclassification curves.
    """

    if params is None:
        params = bootstrap_workflow.run_bootstrap(
            n_boot=n_boot,
            seed=seed,
            conway_path=conway_path,
            bootstrap_mode=bootstrap_mode,
        ).draws
    if paco2_data is None:
        paco2_data = load_paco2_distribution(paco2_path)
    prepared = paco2_data if "subgroup" in paco2_data.columns else prepare_paco2_distribution(paco2_data)

    rng = np.random.default_rng(seed)
    available_groups = set(params["group"]) if "group" in params.columns else set()
    frames: list[pd.DataFrame] = []
    for subgroup in PACO2_SUBGROUP_ORDER:
        paco2_values = prepared.loc[prepared["subgroup"] == subgroup, "paco2"].to_numpy(dtype=float)
        if paco2_values.size == 0:
            continue
        if "group" in params.columns:
            group_key = subgroup if subgroup in available_groups else PACO2_TO_CONWAY_GROUP.get(subgroup, subgroup)
            group_params = params[params["group"] == group_key]
        else:
            group_params = params
        if group_params.empty:
            warnings.warn(
                f"No parameters found for subgroup '{subgroup}'; falling back to all params.",
                UserWarning,
            )
            group_params = params
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
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        token = threshold_label(threshold)
        curves.to_csv(out_dir / f"conditional_classification_t{token}.csv", index=False)
        _write_text(out_dir / f"conditional_classification_t{token}.md", markdown)

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
        "",
        "Each row corresponds to a PaCO2 bin with empirical count/weight.",
        "TN/FP/FN/TP columns report bootstrap quantiles of conditional probabilities.",
    ]
    if curves.empty:
        lines.append("")
        lines.append("No conditional curve rows available.")
        return "\n".join(lines)
    lines.extend(
        [
            "",
            "Columns: group, threshold, paco2_bin, count, weight,",
            "tn_q025/tn_q50/tn_q975, fp_q025/fp_q50/fp_q975,",
            "fn_q025/fn_q50/fn_q975, tp_q025/tp_q50/tp_q975.",
        ]
    )
    return "\n".join(lines)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
