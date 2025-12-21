"""Workflow helpers for forward simulation summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..data import load_paco2_distribution
from ..io import (
    DEFAULT_CLASSIFICATION_THRESHOLDS,
    build_simulation_summary,
    format_simulation_summary,
)
from . import bootstrap as bootstrap_workflow


@dataclass(frozen=True)
class SimulationWorkflowResult:
    summary: pd.DataFrame
    invariants: dict[str, float | int | str]
    markdown: str


def run_forward_simulation_summary(
    params: pd.DataFrame | None = None,
    paco2_data: pd.DataFrame | None = None,
    paco2_path: Path | None = None,
    conway_path: Path | None = None,
    thresholds: Sequence[float] = DEFAULT_CLASSIFICATION_THRESHOLDS,
    mode: str = "analytic",
    seed: int | None = None,
    n_boot: int = 1000,
    n_draws: int | None = None,
    n_mc: int | None = None,
    out_dir: Path | None = None,
) -> SimulationWorkflowResult:
    if params is None:
        params = bootstrap_workflow.run_bootstrap(n_boot=n_boot, seed=seed, conway_path=conway_path).draws
    if paco2_data is None:
        paco2_data = load_paco2_distribution(paco2_path)
    summary = build_simulation_summary(
        params,
        paco2_data,
        thresholds=thresholds,
        mode=mode,
        seed=seed,
        n_draws=n_draws,
        n_mc=n_mc,
    )
    n_boot_per_group = _n_boot_per_group(params)
    markdown = format_simulation_summary(summary, thresholds=thresholds, n_boot=n_boot_per_group, mode=mode)
    if out_dir is not None:
        _write_text(Path(out_dir) / "simulation_summary.md", markdown)
    invariants = {
        "groups": int(summary["group"].nunique()) if not summary.empty else 0,
        "thresholds": ",".join(f"{value:g}" for value in thresholds),
        "mode": mode,
    }
    return SimulationWorkflowResult(summary=summary, invariants=invariants, markdown=markdown)


def _n_boot_per_group(params: pd.DataFrame) -> int:
    if "group" not in params.columns:
        return int(params.shape[0])
    counts = params.groupby("group").size()
    return int(counts.max()) if not counts.empty else 0


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
