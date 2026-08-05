from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tcco2_accuracy.data import (
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
SYNTHETIC_PRIOR_PATH = REPO_ROOT / "tests" / "fixtures" / "synthetic_paco2_prior.csv"


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
    bins = load_paco2_prior_bins(SYNTHETIC_PRIOR_PATH)
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

    metrics = simulate_forward(
        paco2_data, params, thresholds=[45.0], mode="analytic", seed=202401, n_draws=50
    )
    summary = summarize_simulation_metrics(metrics)

    for _, row in summary.iterrows():
        assert row["d_mean_q025"] < row["d_mean_q975"]


def test_expected_classification_metrics_handles_prevalence_extremes() -> None:
    paco2_values = np.array([40.0, 41.0, 42.0])

    high_threshold = expected_classification_metrics(
        paco2_values, delta=0.0, sd_total=2.0, threshold_value=60.0
    )
    assert high_threshold["prevalence"] == pytest.approx(0.0)
    assert np.isnan(high_threshold["sensitivity"])
    assert np.isfinite(high_threshold["specificity"])

    low_threshold = expected_classification_metrics(
        paco2_values, delta=0.0, sd_total=2.0, threshold_value=30.0
    )
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


@pytest.mark.parametrize("z", [8.0, 12.0])
def test_expected_classification_metrics_preserves_extreme_tail_lrs(z: float) -> None:
    paco2_values = np.array([45.0 - z, 45.0 + z])

    metrics = expected_classification_metrics(
        paco2_values,
        delta=0.0,
        sd_total=1.0,
        threshold_value=45.0,
    )

    expected_lr_pos = stats.norm.sf(-z) / stats.norm.sf(z)
    expected_lr_neg = stats.norm.cdf(-z) / stats.norm.cdf(z)
    assert np.isfinite(metrics["lr_pos"])
    assert metrics["lr_pos"] == pytest.approx(expected_lr_pos, rel=1e-13)
    assert metrics["lr_neg"] == pytest.approx(expected_lr_neg, rel=1e-13, abs=0.0)
    assert metrics["fp_rate"] > 0
    assert metrics["fn_rate"] > 0


def test_expected_classification_lr_uses_log_tails_beyond_probability_underflow() -> None:
    metrics = expected_classification_metrics(
        np.array([44.0, 45.0]),
        delta=40.0,
        sd_total=1.0,
        threshold_value=45.0,
    )

    expected_log_lr = stats.norm.logsf(40.0) - stats.norm.logsf(41.0)
    assert metrics["fp_rate"] == 0.0
    assert metrics["tp_rate"] == 0.0
    assert np.isfinite(metrics["lr_pos"])
    assert metrics["lr_pos"] == pytest.approx(np.exp(expected_log_lr), rel=1e-13)


def test_simulation_missing_group_params_fails_closed() -> None:
    paco2_data = pd.DataFrame({"paco2": [35.0, 45.0, 55.0], "subgroup": ["pft", "ed_inp", "icu"]})
    params = pd.DataFrame({"group": ["main"], "delta": [1.0], "sigma2": [4.0], "tau2": [0.0]})

    with pytest.raises(ValueError, match="No parameters found for requested subgroup 'pft'"):
        simulate_forward(paco2_data, params, thresholds=[45.0], mode="analytic")


def test_simulation_explicit_main_fallback_records_provenance() -> None:
    paco2_data = pd.DataFrame({"paco2": [35.0, 45.0, 55.0], "subgroup": ["pft", "ed_inp", "icu"]})
    params = pd.DataFrame({"group": ["main"], "delta": [1.0], "sigma2": [4.0], "tau2": [0.0]})

    metrics = simulate_forward(
        paco2_data,
        params,
        thresholds=[45.0],
        mode="analytic",
        fallback="main",
    )

    assert set(metrics["group"]) == {"pft", "ed_inp", "icu"}
    assert set(metrics["requested_group"]) == {"pft", "ed_inp", "icu"}
    assert set(metrics["parameter_group_used"]) == {"main"}


@pytest.mark.parametrize("threshold", [0.0, -1.0, np.nan, np.inf])
def test_expected_classification_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        expected_classification_metrics(
            [40.0, 50.0], delta=0.0, sd_total=2.0, threshold_value=threshold
        )


@pytest.mark.parametrize("paco2", [0.0, -1.0, np.nan, np.inf])
def test_expected_classification_rejects_invalid_paco2(paco2: float) -> None:
    with pytest.raises(ValueError, match="PaCO2 values"):
        expected_classification_metrics([paco2], delta=0.0, sd_total=2.0, threshold_value=45.0)


def test_simulation_rejects_empty_threshold_sequence() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [4.0], "tau2": [0.0]})

    with pytest.raises(ValueError, match="At least one"):
        simulate_forward_metrics([40.0], params, thresholds=[], mode="analytic")
