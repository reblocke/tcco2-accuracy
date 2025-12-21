"""Workflow helpers for Conway bootstrap draws."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..bootstrap import bootstrap_group_draws
from ..data import load_conway_group
from ..io import CONWAY_GROUPS, bootstrap_loa_summary, format_bootstrap_summary, write_bootstrap_params


@dataclass(frozen=True)
class BootstrapWorkflowResult:
    draws: pd.DataFrame
    summary: pd.DataFrame
    invariants: dict[str, float | int]
    markdown: str


def run_bootstrap(
    n_boot: int = 1000,
    seed: int = 202401,
    conway_path: Path | None = None,
    groups: dict[str, str] | None = None,
    data_by_group: Iterable[tuple[str, pd.DataFrame]] | None = None,
    study_id: str = "study",
    truncate_tau2: bool = True,
    bootstrap_mode: str = "cluster_plus_withinstudy",
    out_dir: Path | None = None,
) -> BootstrapWorkflowResult:
    """Run Conway bootstrap draws and summaries.

    Reads:
        - Conway study-level data from ``conway_path`` when provided, otherwise
          the bundled `Conway Meta/data.dta`.

    Writes:
        - ``bootstrap_params.csv`` and ``bootstrap_summary.md`` in ``out_dir`` when provided.

    Returns:
        ``BootstrapWorkflowResult`` with bootstrap draws (columns include
        ``replicate``, ``group``, ``delta``, ``sigma2``, ``tau2``) plus summary
        quantiles versus Conway LoA bounds.

    Notes:
        ``bootstrap_mode`` controls whether within-study estimation uncertainty
        is injected in addition to cluster resampling.

    Determinism:
        Deterministic for fixed ``seed``; tau2 is truncated at zero by default
        to maintain non-negative between-study variance in bootstrap draws.
    """

    group_map = groups or CONWAY_GROUPS
    if data_by_group is None:
        data_by_group = [
            (group_name, load_conway_group(group_key, path=conway_path))
            for group_name, group_key in group_map.items()
        ]
    draws = bootstrap_group_draws(
        data_by_group,
        n_boot=n_boot,
        seed=seed,
        study_id=study_id,
        truncate_tau2=truncate_tau2,
        bootstrap_mode=bootstrap_mode,
    )
    summary = bootstrap_loa_summary(draws, conway_path=conway_path)
    markdown = format_bootstrap_summary(
        summary,
        n_boot=n_boot,
        seed=seed,
        bootstrap_mode=bootstrap_mode,
    )
    if out_dir is not None:
        out_dir = Path(out_dir)
        write_bootstrap_params(out_dir / "bootstrap_params.csv", draws)
        _write_text(out_dir / "bootstrap_summary.md", markdown)
    invariants = {
        "groups": int(summary.shape[0]),
        "n_boot": int(n_boot),
        "draws": int(draws.shape[0]),
        "bootstrap_mode": bootstrap_mode,
    }
    return BootstrapWorkflowResult(draws=draws, summary=summary, invariants=invariants, markdown=markdown)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
