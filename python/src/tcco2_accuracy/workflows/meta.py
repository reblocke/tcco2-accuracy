"""Workflow helpers for Conway meta-analysis checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..conway_meta import conway_group_summary
from ..data import load_conway_group
from ..io import CONWAY_GROUPS


META_GROUP_LABELS: dict[str, str] = {
    "main": "Main analysis",
    "icu": "ICU",
    "arf": "Acute respiratory failure",
    "lft": "Outpatients requiring lung function tests",
}


@dataclass(frozen=True)
class MetaWorkflowResult:
    summary: pd.DataFrame
    invariants: dict[str, float | int | str]
    markdown: str


def run_meta_checks(
    conway_path: Path | None = None,
    groups: dict[str, str] | None = None,
    data_by_group: Iterable[tuple[str, pd.DataFrame]] | None = None,
    out_dir: Path | None = None,
) -> MetaWorkflowResult:
    """Run Conway meta-analysis checks by subgroup.

    Reads:
        - Conway study-level data from ``conway_path`` when provided, otherwise
          the bundled `Conway Meta/data.dta`.

    Writes:
        - ``meta_loa_check.md`` in ``out_dir`` when provided.

    Returns:
        ``MetaWorkflowResult`` containing a summary DataFrame with columns
        ``group``, ``population``, ``studies``, ``n_pairs``, ``n_participants``,
        ``bias``, ``sd``, ``tau2``, ``loa_l``, ``loa_u``, ``ci_l``, and ``ci_u``.

    Determinism:
        Deterministic; no random sampling is used.
    """

    group_map = groups or CONWAY_GROUPS
    provided_groups = data_by_group is not None
    if data_by_group is None:
        data_by_group = [
            (group_name, load_conway_group(group_key, path=conway_path))
            for group_name, group_key in group_map.items()
        ]
    rows: list[dict[str, float | int | str]] = []
    for group_name, group_data in data_by_group:
        summary = conway_group_summary(group_data)
        rows.append(
            {
                "group": group_name,
                "population": META_GROUP_LABELS.get(group_name, group_name),
                "studies": summary.studies,
                "n_pairs": summary.n_pairs,
                "n_participants": summary.n_participants,
                "bias": summary.bias,
                "sd": summary.sd,
                "tau2": summary.tau2,
                "loa_l": summary.loa_l,
                "loa_u": summary.loa_u,
                "ci_l": summary.ci_l,
                "ci_u": summary.ci_u,
            }
        )

    summary_frame = pd.DataFrame(
        rows,
        columns=[
            "group",
            "population",
            "studies",
            "n_pairs",
            "n_participants",
            "bias",
            "sd",
            "tau2",
            "loa_l",
            "loa_u",
            "ci_l",
            "ci_u",
        ],
    )
    invariants = _meta_invariants(summary_frame)
    if conway_path is not None:
        source = str(conway_path)
    elif provided_groups:
        source = "in-memory"
    else:
        source = "Conway Meta/data.dta"
    markdown = format_meta_summary(summary_frame, source=source)
    if out_dir is not None:
        _write_text(Path(out_dir) / "meta_loa_check.md", markdown)
    return MetaWorkflowResult(summary=summary_frame, invariants=invariants, markdown=markdown)


def format_meta_summary(summary: pd.DataFrame, source: str) -> str:
    lines = [
        "# Meta-analysis LoA Check",
        "",
        f"Source: `{source}`.",
        "- Formula: SD_total = sqrt(sigma^2 + tau^2); LoA = delta ± 2 * SD_total.",
        "",
        "| Population | Bias | SD | Tau2 | LoA L | LoA U | CI L | CI U |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["population"]),
                    f"{row['bias']:.2f}",
                    f"{row['sd']:.2f}",
                    f"{row['tau2']:.2f}",
                    f"{row['loa_l']:.2f}",
                    f"{row['loa_u']:.2f}",
                    f"{row['ci_l']:.2f}",
                    f"{row['ci_u']:.2f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _meta_invariants(summary: pd.DataFrame) -> dict[str, float | int | str]:
    if summary.empty:
        return {"groups": 0, "max_loa_abs_error": float("nan")}
    sd_total = np.sqrt(summary["sd"] ** 2 + summary["tau2"])
    loa_l_expected = summary["bias"] - 2 * sd_total
    loa_u_expected = summary["bias"] + 2 * sd_total
    loa_residuals = np.concatenate(
        [
            (loa_l_expected - summary["loa_l"]).to_numpy(),
            (loa_u_expected - summary["loa_u"]).to_numpy(),
        ]
    )
    max_abs_error = float(np.max(np.abs(loa_residuals)))
    return {
        "groups": int(summary.shape[0]),
        "max_loa_abs_error": max_abs_error,
        "total_pairs": int(summary["n_pairs"].sum()),
    }


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
