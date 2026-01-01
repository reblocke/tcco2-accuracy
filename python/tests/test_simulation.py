from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcco2_accuracy.data import (
    PACO2_PRIOR_BINS_PATH,
    PACO2_SUBGROUP_ORDER,
    load_paco2_prior_bins,
)
from tcco2_accuracy.simulation import (
    expected_classification_metrics,
    simulate_forward,
    simulate_forward_metrics,
    summarize_simulation_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_PATH = REPO_ROOT / "artifacts" / "bootstrap_params.csv"


def _sample_prior_values(
    bins: pd.DataFrame,
    group: str,
    n_samples: int,
    random_state: np.random.Generator,
) -> np.ndarray:
    subset = bins.loc[bins["group"] == group]
    values = subset["paco2_bin"].to_numpy(dtype=float)
    weights = subset["weight"].to_numpy(dtype=float)
    return random_state.choice(values, size=n_samples, replace=True, p=weights)


def _sample_paco2_prior(seed: int = 202401, n_samples: int = 200) -> pd.DataFrame:
    bins = load_paco2_prior_bins(PACO2_PRIOR_BINS_PATH)
    random_state = np.random.default_rng(seed)
    frames = []
    for group in PACO2_SUBGROUP_ORDER:
        values = _sample_prior_values(bins, group, n_samples, random_state)
        frames.append(pd.DataFrame({"paco2": values, "subgroup": group}))
    return pd.concat(frames, ignore_index=True)


def test_fixed_parameter_d_moments_match_delta_and_variance() -> None:
    paco2_values = np.array([35.0, 45.0, 55.0])
    params = pd.DataFrame({"delta": [1.5], "sigma2": [4.0], "tau2": [1.0]})

    metrics = simulate_forward_metrics(paco2_values, params, thresholds=[45.0], mode="analytic")

    result = metrics.iloc[0]
    assert result["d_mean"] == pytest.approx(1.5)
    assert result["d_sd"] ** 2 == pytest.approx(5.0)


def test_bootstrap_intervals_are_non_degenerate() -> None:
    paco2_data = _sample_paco2_prior()
    params = pd.read_csv(BOOTSTRAP_PATH)

    metrics = simulate_forward(paco2_data, params, thresholds=[45.0], mode="analytic", seed=202401, n_draws=50)
    summary = summarize_simulation_metrics(metrics)

    for _, row in summary.iterrows():
        assert row["d_mean_q025"] < row["d_mean_q975"]


def test_expected_classification_metrics_handles_prevalence_extremes() -> None:
    paco2_values = np.array([40.0, 41.0, 42.0])

    high_threshold = expected_classification_metrics(paco2_values, delta=0.0, sd_total=2.0, threshold_value=60.0)
    assert high_threshold["prevalence"] == pytest.approx(0.0)
    assert np.isnan(high_threshold["sensitivity"])
    assert np.isfinite(high_threshold["specificity"])

    low_threshold = expected_classification_metrics(paco2_values, delta=0.0, sd_total=2.0, threshold_value=30.0)
    assert low_threshold["prevalence"] == pytest.approx(1.0)
    assert np.isfinite(low_threshold["sensitivity"])
    assert np.isnan(low_threshold["specificity"])


def test_expected_classification_metrics_lr_handles_zero_denominator() -> None:
    paco2_values = np.array([30.0, 60.0])
    metrics = expected_classification_metrics(
        paco2_values,
        delta=0.0,
        sd_total=1e-6,
        threshold_value=45.0,
    )

    assert metrics["lr_pos"] == np.inf
    assert metrics["lr_neg"] == pytest.approx(0.0)


def test_simulation_missing_group_params_falls_back() -> None:
    paco2_data = pd.DataFrame(
        {"paco2": [35.0, 45.0, 55.0], "subgroup": ["pft", "ed_inp", "icu"]}
    )
    params = pd.DataFrame(
        {"group": ["main"], "delta": [1.0], "sigma2": [4.0], "tau2": [0.0]}
    )

    with pytest.warns(UserWarning, match="No parameters found"):
        metrics = simulate_forward(paco2_data, params, thresholds=[45.0], mode="analytic")

    assert set(metrics["group"]) == {"pft", "ed_inp", "icu"}
