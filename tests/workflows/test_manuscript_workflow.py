from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcco2_accuracy.core.conway_meta import AGREEMENT_METHOD_VERSION, RESULTS_STATUS
from tcco2_accuracy.reporting import manuscript as manuscript_reporting
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
    assert {"requested_group", "parameter_group_used"}.issubset(table1.columns)
    assert table1[["sensitivity_q500", "specificity_q500"]].notna().all().all()

    table2 = pd.read_csv(tmp_path / "manuscript_table2_two_stage.csv")
    assert {"zone1_prob_q500", "zone2_prob_q500", "zone3_prob_q500"}.issubset(table2.columns)
    assert {"requested_group", "parameter_group_used"}.issubset(table2.columns)

    table3 = pd.read_csv(tmp_path / "manuscript_table3_prediction_intervals.csv")
    assert {"likelihood_paco2_q500", "prior_paco2_q500"}.issubset(table3.columns)
    assert {"requested_group", "parameter_group_used"}.issubset(table3.columns)
    assert table3[["likelihood_paco2_q500", "prior_paco2_q500"]].notna().all().all()

    for name in (
        "table1",
        "confusion_matrix",
        "two_stage_summary",
        "table2",
        "table3",
        "snippets",
    ):
        assert "Parameter routing:" in result.markdown[name]
    assert "Error-model parameters used" in result.snippets


def test_manuscript_parameters_only_never_loads_paco2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_by_group = [
        ("main", _synthetic_conway_group("main", -0.2)),
        ("icu", _synthetic_conway_group("icu", -0.5)),
        ("arf", _synthetic_conway_group("arf", 1.1)),
        ("lft", _synthetic_conway_group("lft", -0.1)),
    ]
    draws = bootstrap.run_bootstrap(n_boot=4, seed=123, data_by_group=data_by_group).draws

    def _deny_restricted_load(*args, **kwargs):
        raise AssertionError("parameter-only reporting attempted to load restricted PaCO2 data")

    monkeypatch.setattr(manuscript_reporting, "load_paco2_distribution", _deny_restricted_load)
    result = manuscript.run_manuscript_parameters(params=draws, out_dir=tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "manuscript_parameters.csv",
        "manuscript_parameters.md",
    }
    assert result.parameters["agreement_method_version"].unique().tolist() == [
        AGREEMENT_METHOD_VERSION
    ]
    assert result.parameters["results_status"].unique().tolist() == [RESULTS_STATUS]
    assert AGREEMENT_METHOD_VERSION in result.markdown
    assert RESULTS_STATUS in result.markdown


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
