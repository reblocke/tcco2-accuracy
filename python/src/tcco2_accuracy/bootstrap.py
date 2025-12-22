"""Bootstrap utilities for Conway meta-analysis parameters."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .bland_altman import loa_bounds
from .conway_meta import loa_summary, prepare_conway_inputs


BOOTSTRAP_MODES = ("cluster_only", "cluster_plus_withinstudy")


def bootstrap_conway_parameters(
    data: pd.DataFrame,
    n_boot: int,
    seed: int | None = None,
    study_id: str = "study",
    truncate_tau2: bool = True,
    bootstrap_mode: str = "cluster_only",
) -> pd.DataFrame:
    """Return bootstrap draws for delta, sigma^2, tau^2, and LoA.

    ``bootstrap_mode="cluster_plus_withinstudy"`` adds within-study perturbations
    on top of cluster resampling to propagate finite-sample uncertainty.
    """

    if bootstrap_mode not in BOOTSTRAP_MODES:
        raise ValueError(f"Unknown bootstrap_mode: {bootstrap_mode}")

    if study_id not in data.columns:
        raise KeyError(f"Missing study identifier column: {study_id}")

    inputs = prepare_conway_inputs(data)
    study_ids = inputs[study_id].dropna().unique()
    if study_ids.size == 0:
        raise ValueError("No studies available for bootstrap")

    cluster_map = {study: inputs[inputs[study_id] == study] for study in study_ids}
    rng = np.random.default_rng(seed)
    draws: list[dict[str, float | int]] = []

    for replicate in range(n_boot):
        sampled_ids = rng.choice(study_ids, size=study_ids.size, replace=True)
        boot = pd.concat([cluster_map[study] for study in sampled_ids], ignore_index=True)
        # Cluster bootstrap resampling captures between-study selection variability.
        if bootstrap_mode == "cluster_plus_withinstudy":
            # Within-study perturbations propagate finite-sample uncertainty in bias/log-variance inputs.
            # We assume independence between bias and log-variance perturbations per study.
            bias = rng.normal(boot["bias"].to_numpy(), np.sqrt(boot["v_bias"].to_numpy()))
            logs2 = rng.normal(boot["logs2"].to_numpy(), np.sqrt(boot["v_logs2"].to_numpy()))
        else:
            bias = boot["bias"]
            logs2 = boot["logs2"]
        loa = loa_summary(
            bias,
            boot["v_bias"],
            logs2,
            boot["v_logs2"],
            truncate_tau2=truncate_tau2,
        )
        # Truncation is a stability choice for simulation draws, not Table 1 reproduction.
        tau2 = max(0.0, loa.tau2) if truncate_tau2 else loa.tau2
        sigma2 = loa.sd**2
        sd_total = math.sqrt(sigma2 + tau2)
        loa_l, loa_u = loa_bounds(loa.bias, loa.sd, tau2)
        draws.append(
            {
                "replicate": replicate,
                "studies": study_ids.size,
                "delta": loa.bias,
                "sigma2": sigma2,
                "tau2": tau2,
                "sd_total": sd_total,
                "loa_l": loa_l,
                "loa_u": loa_u,
                "bootstrap_mode": bootstrap_mode,
            }
        )

    return pd.DataFrame(draws)


def bootstrap_group_draws(
    data_by_group: Iterable[tuple[str, pd.DataFrame]],
    n_boot: int,
    seed: int | None = None,
    study_id: str = "study",
    truncate_tau2: bool = True,
    bootstrap_mode: str = "cluster_only",
) -> pd.DataFrame:
    """Return bootstrap draws for multiple groups."""

    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for group_name, group_data in data_by_group:
        group_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        draws = bootstrap_conway_parameters(
            group_data,
            n_boot=n_boot,
            seed=group_seed,
            study_id=study_id,
            truncate_tau2=truncate_tau2,
            bootstrap_mode=bootstrap_mode,
        )
        draws.insert(0, "group", group_name)
        frames.append(draws)

    return pd.concat(frames, ignore_index=True)
