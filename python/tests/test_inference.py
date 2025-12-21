from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tcco2_accuracy.inference import infer_paco2


def test_likelihood_only_matches_normal() -> None:
    params = pd.DataFrame({"delta": [2.0], "sigma2": [4.0], "tau2": [0.0]})

    result = infer_paco2([40.0], params, thresholds=[45.0])

    row = result.iloc[0]
    mean = 42.0
    sd = 2.0
    assert row["paco2_q500"] == pytest.approx(mean)
    assert row["paco2_q025"] == pytest.approx(stats.norm.ppf(0.025, loc=mean, scale=sd))
    assert row["paco2_q975"] == pytest.approx(stats.norm.ppf(0.975, loc=mean, scale=sd))
    expected_prob = 1 - stats.norm.cdf(45.0, loc=mean, scale=sd)
    assert row["p_ge_45"] == pytest.approx(expected_prob)


def test_likelihood_monotonicity() -> None:
    params = pd.DataFrame(
        {
            "delta": [-1.0, 1.0],
            "sigma2": [1.0, 1.0],
            "tau2": [0.0, 0.0],
        }
    )

    result = infer_paco2([30.0, 40.0, 50.0], params, thresholds=[45.0])

    medians = result["paco2_q500"].to_numpy()
    probs = result["p_ge_45"].to_numpy()
    assert np.all(np.diff(medians) > 0)
    assert np.all(np.diff(probs) > 0)


def test_prior_weighted_symmetry_keeps_center() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [1.0], "tau2": [0.0]})
    prior = np.array([30.0, 40.0, 50.0])

    result = infer_paco2([40.0], params, thresholds=[40.0], paco2_prior=prior, use_prior=True)

    row = result.iloc[0]
    assert row["paco2_q500"] == pytest.approx(40.0)
    assert row["p_ge_40"] > 0.5
