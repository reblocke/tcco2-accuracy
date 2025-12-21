from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcco2_accuracy.data import load_paco2_distribution, prepare_paco2_distribution
from tcco2_accuracy.simulation import simulate_forward, simulate_forward_metrics, summarize_simulation_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_PATH = REPO_ROOT / "artifacts" / "bootstrap_params.csv"


def test_fixed_parameter_d_moments_match_delta_and_variance() -> None:
    paco2_values = np.array([35.0, 45.0, 55.0])
    params = pd.DataFrame({"delta": [1.5], "sigma2": [4.0], "tau2": [1.0]})

    metrics = simulate_forward_metrics(paco2_values, params, thresholds=[45.0], mode="analytic")

    result = metrics.iloc[0]
    assert result["d_mean"] == pytest.approx(1.5)
    assert result["d_sd"] ** 2 == pytest.approx(5.0)


def test_bootstrap_intervals_are_non_degenerate() -> None:
    paco2_data = prepare_paco2_distribution(load_paco2_distribution())
    params = pd.read_csv(BOOTSTRAP_PATH)

    metrics = simulate_forward(paco2_data, params, thresholds=[45.0], mode="analytic", seed=202401, n_draws=50)
    summary = summarize_simulation_metrics(metrics)

    for _, row in summary.iterrows():
        assert row["d_mean_q025"] < row["d_mean_q975"]
