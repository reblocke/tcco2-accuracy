from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.reference.conway_reference import (
    corrected_reference,
    prepare_reference_inputs,
)


def test_single_study_matches_hand_calculation_with_repeated_measures() -> None:
    data = pd.DataFrame(
        {
            "study": ["hand"],
            "n": [12.0],
            "n_2": [4.0],
            "bias": [1.25],
            "s2": [9.0],
        }
    )
    inputs = prepare_reference_inputs(data)
    summary = corrected_reference(data)
    expected_s2_adjusted = 11.0
    expected_sigma2 = expected_s2_adjusted * np.exp(1 / 3)

    assert inputs["s2_adjusted"] == pytest.approx([expected_s2_adjusted], abs=1e-12)
    assert inputs["v_bias"] == pytest.approx([expected_s2_adjusted / 4], abs=1e-12)
    assert inputs["log_sigma2"] == pytest.approx(
        [np.log(expected_s2_adjusted) + 1 / 3],
        abs=1e-12,
    )
    assert inputs["var_log_sigma2"] == pytest.approx([2 / 3], abs=1e-12)
    assert summary["bias"] == pytest.approx(1.25, abs=1e-12)
    assert summary["sigma2"] == pytest.approx(expected_sigma2, abs=1e-12)
    assert summary["tau2"] == pytest.approx(0.0, abs=1e-12)
    assert summary["loa_l"] == pytest.approx(1.25 - 2 * np.sqrt(expected_sigma2), abs=1e-12)
    assert summary["loa_u"] == pytest.approx(1.25 + 2 * np.sqrt(expected_sigma2), abs=1e-12)
    assert np.isnan([summary["ci_l"], summary["ci_u"]]).all()
