from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tcco2_accuracy.workflows import bootstrap, manuscript


def test_manuscript_workflow_smoke(tmp_path: Path) -> None:
    seed = 123
    n_boot = 25
    data_by_group = [
        ("main", _synthetic_conway_group("main", -0.2)),
        ("icu", _synthetic_conway_group("icu", -0.5)),
        ("arf", _synthetic_conway_group("arf", 1.1)),
        ("lft", _synthetic_conway_group("lft", -0.1)),
    ]
    boot_result = bootstrap.run_bootstrap(n_boot=n_boot, seed=seed, data_by_group=data_by_group)

    paco2_data = _synthetic_paco2_data()
    result = manuscript.run_manuscript_outputs(
        params=boot_result.draws,
        paco2_data=paco2_data,
        seed=seed,
        n_draws=10,
        out_dir=tmp_path,
    )

    expected_files = [
        "manuscript_parameters.csv",
        "manuscript_table1.csv",
        "manuscript_table2_two_stage.csv",
        "manuscript_table3_prediction_intervals.csv",
        "figure_paco2_distribution_bins.csv",
        "figure_misclassification_vs_paco2.csv",
        "manuscript_results_snippets.md",
    ]
    for filename in expected_files:
        assert (tmp_path / filename).exists()

    params = pd.read_csv(tmp_path / "manuscript_parameters.csv")
    assert {"group", "delta_q500", "sigma2_q500", "tau2_q500"}.issubset(params.columns)
    assert params[["delta_q500", "sigma2_q500", "tau2_q500"]].notna().all().all()

    table1 = pd.read_csv(tmp_path / "manuscript_table1.csv")
    assert {"sensitivity_q500", "specificity_q500", "lr_pos_q500"}.issubset(table1.columns)
    assert table1[["sensitivity_q500", "specificity_q500"]].notna().all().all()

    table2 = pd.read_csv(tmp_path / "manuscript_table2_two_stage.csv")
    assert {"zone1_prob_q500", "zone2_prob_q500", "zone3_prob_q500"}.issubset(table2.columns)

    table3 = pd.read_csv(tmp_path / "manuscript_table3_prediction_intervals.csv")
    assert {"likelihood_paco2_q500", "prior_paco2_q500"}.issubset(table3.columns)
    assert table3[["likelihood_paco2_q500", "prior_paco2_q500"]].notna().all().all()

    assert "Error-model parameters used" in result.snippets


def _synthetic_conway_group(group_name: str, offset: float) -> pd.DataFrame:
    bias = np.array([offset - 0.2, offset, offset + 0.15])
    return pd.DataFrame(
        {
            "study": [f"{group_name}_a", f"{group_name}_b", f"{group_name}_c"],
            "n": [20.0, 25.0, 30.0],
            "n_2": [20.0, 25.0, 30.0],
            "bias": bias,
            "s2": [4.0, 5.5, 6.0],
        }
    )


def _synthetic_paco2_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "paco2": [35.0, 40.0, 50.0, 42.0, 47.0, 55.0, 30.0, 60.0, 38.0],
            "is_amb": [1, 1, 1, 0, 0, 0, 0, 0, 0],
            "is_emer": [0, 0, 0, 1, 1, 1, 0, 0, 0],
            "is_inp": [0, 0, 0, 1, 1, 1, 1, 1, 1],
            "cc_time": [0, 0, 0, 0, 0, 0, 1, 1, 1],
        }
    )
