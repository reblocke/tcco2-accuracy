"""Workflow helpers for PaCO2 distribution summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..data import (
    DEFAULT_PACO2_QUANTILES,
    PACO2_SUBGROUP_ORDER,
    load_paco2_distribution,
    paco2_subgroup_summary,
    prepare_paco2_distribution,
)


@dataclass(frozen=True)
class Paco2WorkflowResult:
    summary: pd.DataFrame
    data: pd.DataFrame
    invariants: dict[str, float | int]
    markdown: str


def run_paco2_summary(
    paco2_path: Path | None = None,
    paco2_data: pd.DataFrame | None = None,
    quantiles: Sequence[float] = DEFAULT_PACO2_QUANTILES,
    out_dir: Path | None = None,
) -> Paco2WorkflowResult:
    provided_data = paco2_data is not None
    if paco2_data is None:
        paco2_data = load_paco2_distribution(paco2_path)
    prepared = prepare_paco2_distribution(paco2_data)
    summary = paco2_subgroup_summary(prepared, quantiles=quantiles)
    if paco2_path is not None:
        source = str(paco2_path)
    elif provided_data:
        source = "in-memory"
    else:
        source = "Data/In Silico TCCO2 Database.dta"
    markdown = format_paco2_summary(summary, source=source)
    if out_dir is not None:
        _write_text(Path(out_dir) / "paco2_distribution_summary.md", markdown)
    counts = prepared["subgroup"].value_counts()
    invariants = {
        "total": int(prepared.shape[0]),
        **{f"{group}_count": int(counts.get(group, 0)) for group in PACO2_SUBGROUP_ORDER},
    }
    return Paco2WorkflowResult(summary=summary, data=prepared, invariants=invariants, markdown=markdown)


def format_paco2_summary(summary: pd.DataFrame, source: str) -> str:
    quantile_columns = [col for col in summary.columns if col.startswith("paco2_q")]
    quantile_columns.sort(key=_quantile_sort_key)
    quantile_labels = [_quantile_label(col) for col in quantile_columns]
    lines = [
        "# PaCO2 distribution summary",
        "",
        f"Source: `{source}` (non-missing PaCO2 rows).",
        "",
        "| Group | Count | " + " | ".join(quantile_labels) + " |",
        "| --- | --- | " + " | ".join(["---"] * len(quantile_labels)) + " |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [str(row["group"]), f"{int(row['count'])}"]
                + [f"{row[column]:.2f}" for column in quantile_columns]
            )
            + " |"
        )
    return "\n".join(lines)


def _quantile_sort_key(column: str) -> int:
    return int(column.split("q")[1])


def _quantile_label(column: str) -> str:
    value = int(column.split("q")[1]) / 10
    label = f"{value:g}"
    return f"PaCO2 q{label}"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
