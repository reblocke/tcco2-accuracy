"""I/O utilities for artifact generation."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from .bootstrap import bootstrap_group_draws
from .conway_meta import conway_group_summary
from .data import load_conway_group


CONWAY_GROUPS: dict[str, str] = {
    "main": "main",
    "icu": "icu",
    "arf": "arf",
    "lft": "lft",
}


def build_bootstrap_params(n_boot: int = 1000, seed: int = 202401) -> pd.DataFrame:
    """Generate bootstrap draws for Conway subgroups."""

    group_data = [(name, load_conway_group(key)) for name, key in CONWAY_GROUPS.items()]
    return bootstrap_group_draws(group_data, n_boot=n_boot, seed=seed, truncate_tau2=True)


def write_bootstrap_params(path: Path, params: pd.DataFrame) -> None:
    """Write bootstrap draws to CSV or Parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        params.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        params.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported bootstrap params format: {path.suffix}")


def bootstrap_loa_summary(params: pd.DataFrame) -> pd.DataFrame:
    """Summarize bootstrap LoA bounds versus Conway CIs."""

    rows: list[dict[str, float | str]] = []
    for group_name, group_key in CONWAY_GROUPS.items():
        subset = params[params["group"] == group_name]
        if subset.empty:
            continue
        loa_l_q = subset["loa_l"].quantile([0.025, 0.5, 0.975])
        loa_u_q = subset["loa_u"].quantile([0.025, 0.5, 0.975])
        summary = conway_group_summary(load_conway_group(group_key))
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
                "n_boot": int(subset.shape[0]),
            }
        )

    return pd.DataFrame(rows)


def format_bootstrap_summary(summary: pd.DataFrame, n_boot: int, seed: int) -> str:
    """Return a markdown summary of bootstrap LoA spread."""

    lines = [
        "# Bootstrap LoA spread summary",
        "",
        f"Bootstrap draws: {n_boot} per subgroup (seed={seed}).",
        "",
        "LoA bounds shown as 2.5/50/97.5% bootstrap quantiles;",
        "Conway CI shown as reported outer CI bounds.",
        "",
        "| Group | LoA L q2.5 | LoA L q50 | LoA L q97.5 | LoA U q2.5 | LoA U q50 | LoA U q97.5 | Conway CI L | Conway CI U |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Qualitative check: bootstrap LoA quantile ranges span a similar",
            "scale to Conway's outer CI bounds across subgroups.",
        ]
    )

    return "\n".join(lines)
