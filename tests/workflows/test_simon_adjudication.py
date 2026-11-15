import math

import pytest

from tcco2_accuracy.workflows.future_setting_data import (
    load_ledger,
    prepare_summaries,
    select_units,
)


def test_simon_steady_state_observation_in_overall_likelihood():
    # Simon 2003, Fig. 6: 135 steady-state Tc−A differences, 15 patients.
    main = select_units(load_ledger(), "main").reset_index(drop=True)
    [i] = main.index[main.study_id == "Simon 2003 (steady state)"]
    assert (main.at[i, "n_pairs"], main.at[i, "participants"]) == (135, 15)
    assert (len(main), main.publication_id.nunique()) == (75, 73)

    data = prepare_summaries(main)
    mmhg_per_kpa = 7.50062
    pooled_sd = (2.81 - (-1.03)) * mmhg_per_kpa / (2 * 1.96)
    # Nine pairs per participant on average. The Conway factor is 1 + 8/126.
    adjusted_variance = pooled_sd**2 * (1 + 8 / 126)
    assert data["bias"][i] == pytest.approx(-0.89 * mmhg_per_kpa)
    assert data["ell"][i] == pytest.approx(math.log(adjusted_variance) + 1 / 14)
    assert data["v_bias"][i] == pytest.approx(adjusted_variance / 15)
    assert data["v_ell"][i] == pytest.approx(2 / 14)
