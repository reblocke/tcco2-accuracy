from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tcco2_accuracy.data import load_paco2_prior
from tcco2_accuracy.ui_api import predict_paco2_from_tcco2

ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_PRIOR_PATH = ROOT / "tests" / "fixtures" / "synthetic_paco2_prior.csv"


def test_single_draw_likelihood_matches_normal() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [4.0], "tau2": [0.0]})

    result = predict_paco2_from_tcco2(
        tcco2=45.0,
        subgroup="pft",
        threshold=45.0,
        mode="likelihood_only",
        params_draws=params,
        interval=0.95,
    )

    mean = 45.0
    sd = 2.0
    assert result.paco2_median == pytest.approx(mean)
    assert result.paco2_q_low == pytest.approx(stats.norm.ppf(0.025, loc=mean, scale=sd))
    assert result.paco2_q_high == pytest.approx(stats.norm.ppf(0.975, loc=mean, scale=sd))
    assert result.p_ge_threshold == pytest.approx(0.5)


def test_ui_api_defaults_to_likelihood_only_without_prior() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [4.0], "tau2": [0.0]})

    result = predict_paco2_from_tcco2(
        tcco2=45.0,
        subgroup="pft",
        threshold=45.0,
        params_draws=params,
    )

    assert result.mode == "likelihood_only"
    assert result.prior_prob is None


def test_ui_api_prior_weighted_requires_explicit_prior() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [4.0], "tau2": [0.0]})

    with pytest.raises(ValueError, match="requires an explicitly supplied.*prior"):
        predict_paco2_from_tcco2(
            tcco2=45.0,
            subgroup="pft",
            threshold=45.0,
            mode="prior_weighted",
            params_draws=params,
        )


def test_prior_weighting_moves_threshold_probability() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [4.0], "tau2": [0.0]})
    prior_values = np.array([43.0, 46.0, 47.0, 48.0, 48.0])

    likelihood = predict_paco2_from_tcco2(
        tcco2=45.0,
        subgroup="pft",
        threshold=45.0,
        mode="likelihood_only",
        params_draws=params,
    )
    weighted = predict_paco2_from_tcco2(
        tcco2=45.0,
        subgroup="pft",
        threshold=45.0,
        mode="prior_weighted",
        params_draws=params,
        paco2_prior_values=prior_values,
    )

    assert weighted.p_ge_threshold > likelihood.p_ge_threshold


def test_posterior_histogram_conservation() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [4.0], "tau2": [0.0]})
    prior_values = np.array([40.0, 45.0, 50.0, 55.0])

    result = predict_paco2_from_tcco2(
        tcco2=45.0,
        subgroup="pft",
        threshold=45.0,
        mode="prior_weighted",
        params_draws=params,
        paco2_prior_values=prior_values,
    )

    assert np.sum(result.posterior_prob) == pytest.approx(1.0, abs=1e-10)
    assert np.all((result.posterior_prob >= 0) & (result.posterior_prob <= 1))
    assert result.prior_prob is not None
    assert np.sum(result.prior_prob) == pytest.approx(1.0, abs=1e-10)
    assert np.all((result.prior_prob >= 0) & (result.prior_prob <= 1))
    assert result.likelihood_prob is not None
    assert result.likelihood_prob.shape == result.paco2_bin.shape
    assert np.sum(result.likelihood_prob) == pytest.approx(1.0, abs=1e-10)
    assert np.all(np.isfinite(result.likelihood_prob))
    assert np.all(result.likelihood_prob >= 0)
    assert result.paco2_q_low >= np.min(result.paco2_bin)
    assert result.paco2_q_high <= np.max(result.paco2_bin)


def test_single_draw_scaled_likelihood_matches_normal_shape() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [4.0], "tau2": [0.0]})
    prior_values = np.array([40.0, 45.0, 50.0, 55.0])

    result = predict_paco2_from_tcco2(
        tcco2=45.0,
        subgroup="pft",
        threshold=45.0,
        mode="prior_weighted",
        params_draws=params,
        paco2_prior_values=prior_values,
    )

    expected = stats.norm.pdf(result.paco2_bin, loc=45.0, scale=2.0)
    expected = expected / expected.sum()

    assert result.likelihood_prob is not None
    assert result.likelihood_prob == pytest.approx(expected)


def test_weighted_prior_matches_equivalent_count_expansion() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [4.0], "tau2": [0.0]})
    expanded_prior = np.array([40.0, 42.0, 42.0, 46.0])

    expanded = predict_paco2_from_tcco2(
        tcco2=42.0,
        subgroup="pft",
        threshold=45.0,
        mode="prior_weighted",
        params_draws=params,
        paco2_prior_values=expanded_prior,
    )
    weighted = predict_paco2_from_tcco2(
        tcco2=42.0,
        subgroup="pft",
        threshold=45.0,
        mode="prior_weighted",
        params_draws=params,
        paco2_prior_values=np.array([40.0, 42.0, 46.0]),
        paco2_prior_weights=np.array([0.25, 0.5, 0.25]),
    )

    assert weighted.paco2_median == pytest.approx(expanded.paco2_median)
    assert weighted.p_ge_threshold == pytest.approx(expanded.p_ge_threshold)
    assert weighted.posterior_prob == pytest.approx(expanded.posterior_prob)
    assert weighted.prior_prob == pytest.approx(expanded.prior_prob)


def test_likelihood_only_result_omits_redundant_likelihood_curve() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [4.0], "tau2": [0.0]})

    result = predict_paco2_from_tcco2(
        tcco2=45.0,
        subgroup="pft",
        threshold=45.0,
        mode="likelihood_only",
        params_draws=params,
    )

    assert result.likelihood_prob is None


@pytest.mark.parametrize("threshold", [0.0, -1.0, np.nan, np.inf])
def test_ui_api_rejects_invalid_threshold(threshold: float) -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [4.0], "tau2": [0.0]})

    with pytest.raises(ValueError, match="threshold"):
        predict_paco2_from_tcco2(
            tcco2=45.0,
            subgroup="pft",
            threshold=threshold,
            mode="likelihood_only",
            params_draws=params,
        )


def test_decision_label_probabilities() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [4.0], "tau2": [0.0]})
    prior_values = np.array([40.0, 45.0, 50.0])

    positive = predict_paco2_from_tcco2(
        tcco2=46.0,
        subgroup="pft",
        threshold=45.0,
        mode="prior_weighted",
        params_draws=params,
        paco2_prior_values=prior_values,
    )
    assert positive.p_true_positive + positive.p_false_positive == pytest.approx(1.0)
    assert positive.p_true_negative == 0.0
    assert positive.p_false_negative == 0.0

    negative = predict_paco2_from_tcco2(
        tcco2=44.0,
        subgroup="pft",
        threshold=45.0,
        mode="prior_weighted",
        params_draws=params,
        paco2_prior_values=prior_values,
    )
    assert negative.p_true_negative + negative.p_false_negative == pytest.approx(1.0)
    assert negative.p_true_positive == 0.0
    assert negative.p_false_positive == 0.0


def test_all_setting_maps_to_main_params() -> None:
    params = pd.DataFrame(
        {
            "group": ["main", "lft", "arf", "icu"],
            "delta": [0.0, 10.0, 20.0, 30.0],
            "sigma2": [1.0, 1.0, 1.0, 1.0],
            "tau2": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = predict_paco2_from_tcco2(
        tcco2=50.0,
        subgroup="all",
        threshold=45.0,
        mode="likelihood_only",
        params_draws=params,
    )

    assert result.subgroup == "all"
    assert result.paco2_median == pytest.approx(50.0, abs=1e-6)


def test_ui_api_inference_smoke_all() -> None:
    params_path = ROOT / "artifacts" / "bootstrap_params.csv"
    params = pd.read_csv(params_path)

    prior_result = load_paco2_prior(
        "all",
        uploaded_bytes=SYNTHETIC_PRIOR_PATH.read_bytes(),
        uploaded_name=SYNTHETIC_PRIOR_PATH.name,
    )
    assert prior_result.error is None
    assert prior_result.values is not None

    result = predict_paco2_from_tcco2(
        tcco2=50.0,
        subgroup="all",
        threshold=45.0,
        mode="prior_weighted",
        params_draws=params,
        paco2_prior_values=prior_result.values,
        paco2_prior_weights=prior_result.weights,
    )

    assert np.isfinite(result.paco2_q_low)
    assert np.isfinite(result.paco2_q_high)
    assert result.paco2_q_low < result.paco2_q_high
    assert 0.0 <= result.p_ge_threshold <= 1.0
