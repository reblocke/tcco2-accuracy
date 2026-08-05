"""Workflow helpers for Conway meta-analysis checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .._files import write_text
from ..conway_meta import AGREEMENT_METHOD_VERSION, RESULTS_STATUS, conway_group_summary
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
    source_label: str | None = None,
    published_comparator_path: Path | None = None,
    published_comparator_source: str | None = None,
    out_dir: Path | None = None,
) -> MetaWorkflowResult:
    """Run Conway meta-analysis checks by subgroup.

    Reads:
        - Conway study-level data from ``conway_path`` when provided, otherwise
          the bundled `Data/conway_studies.csv`.

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
        resolved_data_by_group = [
            (group_name, load_conway_group(group_key, path=conway_path))
            for group_name, group_key in group_map.items()
        ]
    else:
        resolved_data_by_group = list(data_by_group)
    if not resolved_data_by_group:
        raise ValueError("At least one non-empty Conway subgroup analysis is required.")
    rows: list[dict[str, float | int | str]] = []
    for group_name, group_data in resolved_data_by_group:
        summary = conway_group_summary(group_data, truncate_tau2=True)
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
    if source_label is not None:
        source = source_label
    elif conway_path is not None:
        source = str(conway_path)
    elif provided_groups:
        source = "in-memory"
    else:
        source = "Data/conway_studies.csv"
    published_comparator = (
        _load_published_comparator(published_comparator_path)
        if published_comparator_path is not None
        else None
    )
    markdown = format_meta_summary(
        summary_frame,
        source=source,
        published_comparator=published_comparator,
        published_source=(
            published_comparator_source
            or (str(published_comparator_path) if published_comparator_path else "")
        ),
    )
    if out_dir is not None:
        write_text(Path(out_dir) / "meta_loa_check.md", markdown)
    return MetaWorkflowResult(summary=summary_frame, invariants=invariants, markdown=markdown)


def format_meta_summary(
    summary: pd.DataFrame,
    source: str,
    published_comparator: pd.DataFrame | None = None,
    published_source: str = "Conway et al., Thorax 2019, Table 1",
    agreement_method_version: str = AGREEMENT_METHOD_VERSION,
    results_status: str = RESULTS_STATUS,
) -> str:
    lines = [
        "# Meta-analysis LoA Check",
        "",
        f"Source: `{source}`.",
        f"Agreement method version: `{agreement_method_version}`.",
        f"Results status: `{results_status}`.",
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
    if published_comparator is not None:
        lines.extend(
            _format_published_comparison(
                summary,
                published_comparator=published_comparator,
                published_source=published_source,
            )
        )
    return "\n".join(lines)


def _load_published_comparator(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Published Conway comparator not found: {path}")
    comparator = pd.read_csv(path)
    required = {"population", "bias", "sd", "tau2", "loa_l", "loa_u", "ci_l", "ci_u"}
    missing = required - set(comparator.columns)
    if missing:
        raise ValueError(f"Published Conway comparator is missing columns: {sorted(missing)}")
    return comparator


def _format_published_comparison(
    corrected: pd.DataFrame,
    published_comparator: pd.DataFrame,
    published_source: str,
) -> list[str]:
    metrics = (
        ("bias", "Bias"),
        ("sd", "SD"),
        ("tau2", "Tau2"),
        ("loa_l", "LoA L"),
        ("loa_u", "LoA U"),
        ("ci_l", "CI L"),
        ("ci_u", "CI U"),
    )
    published_by_population = published_comparator.set_index("population")
    lines = [
        "",
        "## Corrected versus published/legacy comparator",
        "",
        f"Published comparator: `{published_source}`.",
        (
            "Published values are rounded to one decimal place; delta is corrected minus published "
            "and therefore includes source-rounding differences."
        ),
    ]
    for _, corrected_row in corrected.iterrows():
        population = str(corrected_row["population"])
        if population not in published_by_population.index:
            continue
        published_row = published_by_population.loc[population]
        lines.extend(
            [
                "",
                f"### {population}",
                "",
                "| Metric | Corrected | Published/legacy | Delta |",
                "| --- | --- | --- | --- |",
            ]
        )
        for column, label in metrics:
            corrected_value = float(corrected_row[column])
            published_value = float(published_row[column])
            lines.append(
                f"| {label} | {corrected_value:.2f} | {published_value:.2f} | "
                f"{corrected_value - published_value:+.2f} |"
            )
    return lines


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
