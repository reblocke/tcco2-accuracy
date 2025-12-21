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
from ..utils import n_draws_per_group
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
    """Run forward simulation summaries for TcCO2 accuracy.

    Reads:
        - Conway bootstrap parameters (if ``params`` is not provided).
        - PaCO2 distribution data (if ``paco2_data`` is not provided).

    Writes:
        - ``simulation_summary.md`` in ``out_dir`` when provided.

    Returns:
        ``SimulationWorkflowResult`` with quantile summaries per subgroup and
        threshold. Output includes d moments, LoA bounds, and classification metrics.

    Determinism:
        Deterministic for fixed ``seed`` and parameter draws; Monte Carlo mode
        uses ``seed`` for sampling.
    """

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
    n_boot_per_group = n_draws_per_group(params)
    markdown = format_simulation_summary(summary, thresholds=thresholds, n_boot=n_boot_per_group, mode=mode)
    if out_dir is not None:
        _write_text(Path(out_dir) / "simulation_summary.md", markdown)
    invariants = {
        "groups": int(summary["group"].nunique()) if not summary.empty else 0,
        "thresholds": ",".join(f"{value:g}" for value in thresholds),
        "mode": mode,
    }
    return SimulationWorkflowResult(summary=summary, invariants=invariants, markdown=markdown)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
