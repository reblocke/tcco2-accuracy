from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tcco2_accuracy.data import (
    PACO2_PRIOR_BINS_PATH,
    load_paco2_distribution,
    load_paco2_prior,
    prepare_paco2_distribution,
)
from tcco2_accuracy.ui_api import predict_paco2_from_tcco2


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
    assert result.paco2_q_low >= np.min(result.paco2_bin)
    assert result.paco2_q_high <= np.max(result.paco2_bin)


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
    root = Path(__file__).resolve().parents[2]
    params_path = root / "artifacts" / "bootstrap_params.csv"
    params = pd.read_csv(params_path)

    prior_result = load_paco2_prior("all", default_bins_path=PACO2_PRIOR_BINS_PATH)
    assert prior_result.error is None
    assert prior_result.values is not None

    result = predict_paco2_from_tcco2(
        tcco2=50.0,
        subgroup="all",
        threshold=45.0,
        mode="prior_weighted",
        params_draws=params,
        paco2_prior_values=prior_result.values,
    )

    assert np.isfinite(result.paco2_q_low)
    assert np.isfinite(result.paco2_q_high)
    assert result.paco2_q_low < result.paco2_q_high
    assert 0.0 <= result.p_ge_threshold <= 1.0


def test_inference_demo_regression_optional() -> None:
    root = Path(__file__).resolve().parents[2]
    demo_path = root / "artifacts" / "inference_demo.md"
    params_path = root / "artifacts" / "bootstrap_params.csv"
    paco2_path = root / "Data" / "In Silico TCCO2 Database.dta"
    if not (demo_path.exists() and params_path.exists() and paco2_path.exists()):
        pytest.skip("Required demo artifacts or data not available.")

    demo_value = _extract_demo_probability(demo_path)
    if demo_value is None:
        pytest.skip("Unable to parse inference_demo.md output.")

    params = pd.read_csv(params_path)
    paco2_data = prepare_paco2_distribution(load_paco2_distribution(paco2_path))
    paco2_values = paco2_data.loc[paco2_data["subgroup"] == "pft", "paco2"].to_numpy()

    result = predict_paco2_from_tcco2(
        tcco2=45.0,
        subgroup="pft",
        threshold=45.0,
        mode="prior_weighted",
        params_draws=params,
        paco2_prior_values=paco2_values,
    )

    assert result.p_ge_threshold == pytest.approx(demo_value, abs=0.05)


def _extract_demo_probability(path: Path) -> float | None:
    lines = path.read_text().splitlines()
    in_prior_section = False
    pattern = re.compile(r"\|\s+pft\s+\|\s+45\s+\|.*\|\s+([0-9.]+)\s+\|")
    for line in lines:
        if line.strip().startswith("## Prior-weighted"):
            in_prior_section = True
            continue
        if in_prior_section and line.strip().startswith("## "):
            break
        if in_prior_section:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    return None
