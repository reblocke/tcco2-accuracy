from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tcco2_accuracy.bootstrap import bootstrap_conway_parameters
from tcco2_accuracy.conditional import conditional_classification_curves
from tcco2_accuracy.conway_meta import conway_group_summary
from tcco2_accuracy.data import CONWAY_DATA_PATH, load_conway_group


def test_hybrid_bootstrap_widens_delta_and_loa() -> None:
    data = _synthetic_bootstrap_data()
    draws_cluster = bootstrap_conway_parameters(
        data,
        n_boot=400,
        seed=123,
        bootstrap_mode="cluster_only",
    )
    draws_hybrid = bootstrap_conway_parameters(
        data,
        n_boot=400,
        seed=123,
        bootstrap_mode="cluster_plus_withinstudy",
    )

    sd_cluster = np.std(draws_cluster["delta"].to_numpy(), ddof=0)
    sd_hybrid = np.std(draws_hybrid["delta"].to_numpy(), ddof=0)
    assert sd_hybrid > sd_cluster + 0.01

    width_cluster = draws_cluster["loa_u"].quantile(0.975) - draws_cluster["loa_l"].quantile(0.025)
    width_hybrid = draws_hybrid["loa_u"].quantile(0.975) - draws_hybrid["loa_l"].quantile(0.025)
    assert width_hybrid > width_cluster * 1.10


def test_hybrid_bootstrap_real_data_alignment() -> None:
    if not CONWAY_DATA_PATH.exists():
        pytest.skip("Conway meta-analysis data missing.")
    for group in ("main", "lft"):
        data = load_conway_group(group)
        draws_cluster = bootstrap_conway_parameters(
            data,
            n_boot=400,
            seed=321,
            bootstrap_mode="cluster_only",
        )
        draws_hybrid = bootstrap_conway_parameters(
            data,
            n_boot=400,
            seed=321,
            bootstrap_mode="cluster_plus_withinstudy",
        )
        conway = conway_group_summary(data)
        width_cluster = draws_cluster["loa_u"].quantile(0.975) - draws_cluster["loa_l"].quantile(
            0.025
        )
        width_hybrid = draws_hybrid["loa_u"].quantile(0.975) - draws_hybrid["loa_l"].quantile(0.025)
        ratio_cluster = width_cluster / (conway.ci_u - conway.ci_l)
        ratio_hybrid = width_hybrid / (conway.ci_u - conway.ci_l)
        assert np.isfinite(ratio_cluster)
        assert np.isfinite(ratio_hybrid)
        # Monte Carlo noise and large tau2 can make ratios nearly identical or slightly reversed.
        # The synthetic test above carries the strict widening assertion.
        assert ratio_hybrid >= ratio_cluster * 0.99


def test_conditional_curves_single_draw() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [1.0], "tau2": [0.0]})
    paco2_values = np.array([44.0, 45.0, 46.0])
    curves = conditional_classification_curves(paco2_values, params, threshold=45.0)

    row_45 = curves.loc[curves["paco2_bin"] == 45.0].iloc[0]
    assert np.isclose(row_45["tp_q50"], 0.5, atol=1e-6)
    assert np.isclose(row_45["fn_q50"], 0.5, atol=1e-6)
    assert np.isclose(row_45["tn_q50"], 0.0, atol=1e-12)
    assert np.isclose(row_45["fp_q50"], 0.0, atol=1e-12)

    row_44 = curves.loc[curves["paco2_bin"] == 44.0].iloc[0]
    expected_fp = stats.norm.cdf(-1)
    assert np.isclose(row_44["fp_q50"], expected_fp, atol=1e-6)
    assert np.isclose(row_44["tn_q50"], stats.norm.sf(-1), atol=1e-6)


def test_conditional_probabilities_sum_to_one() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [1.0], "tau2": [0.0]})
    paco2_values = np.array([44.0, 45.0, 46.0])
    curves = conditional_classification_curves(paco2_values, params, threshold=45.0)

    for _, row in curves.iterrows():
        total = row["tn_q50"] + row["fp_q50"] + row["fn_q50"] + row["tp_q50"]
        assert np.isclose(total, 1.0, atol=1e-8)
        assert 0.0 <= row["tn_q50"] <= 1.0
        assert 0.0 <= row["fp_q50"] <= 1.0
        assert 0.0 <= row["fn_q50"] <= 1.0
        assert 0.0 <= row["tp_q50"] <= 1.0


def test_conditional_branching_rules() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [1.0], "tau2": [0.0]})
    paco2_values = np.array([44.0, 45.0, 46.0])
    curves = conditional_classification_curves(paco2_values, params, threshold=45.0)

    below = curves[curves["paco2_bin"] < 45.0]
    above = curves[curves["paco2_bin"] >= 45.0]
    assert np.isclose(below["tp_q50"], 0.0, atol=1e-12).all()
    assert np.isclose(below["fn_q50"], 0.0, atol=1e-12).all()
    assert np.isclose(above["tn_q50"], 0.0, atol=1e-12).all()
    assert np.isclose(above["fp_q50"], 0.0, atol=1e-12).all()


def test_conditional_truth_and_probabilities_use_original_values_before_binning() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [1.0], "tau2": [0.0]})
    paco2_values = np.array([44.49, 44.50, 44.99, 45.00, 45.01])

    curves = conditional_classification_curves(paco2_values, params, threshold=45.0)

    assert curves[["paco2_bin", "paco2_bin_upper", "count"]].to_dict("records") == [
        {"paco2_bin": 44.0, "paco2_bin_upper": 45.0, "count": 3},
        {"paco2_bin": 45.0, "paco2_bin_upper": 46.0, "count": 2},
    ]
    below = curves.loc[curves["paco2_bin"] == 44.0].iloc[0]
    above = curves.loc[curves["paco2_bin"] == 45.0].iloc[0]
    expected_fp = np.mean(stats.norm.sf(45.0 - paco2_values[:3]))
    expected_tp = np.mean(stats.norm.sf(45.0 - paco2_values[3:]))
    assert below["fp_q50"] == pytest.approx(expected_fp)
    assert below["tn_q50"] == pytest.approx(1.0 - expected_fp)
    assert below["fn_q50"] == 0.0
    assert below["tp_q50"] == 0.0
    assert above["tp_q50"] == pytest.approx(expected_tp)
    assert above["fn_q50"] == pytest.approx(1.0 - expected_tp)
    assert above["tn_q50"] == 0.0
    assert above["fp_q50"] == 0.0
    for _, row in curves.iterrows():
        total = row["tn_q50"] + row["fp_q50"] + row["fn_q50"] + row["tp_q50"]
        assert total == pytest.approx(1.0, abs=1e-15)


def test_legacy_round_bins_do_not_determine_conditional_truth() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [1.0], "tau2": [0.0]})
    paco2_values = np.array([44.99, 45.00, 45.01])

    curves = conditional_classification_curves(
        paco2_values,
        params,
        threshold=45.0,
        bin_method="round",
    )

    row = curves.iloc[0]
    assert row["paco2_bin"] == pytest.approx(44.5)
    assert row["paco2_bin_upper"] == pytest.approx(45.5)
    assert row["tn_q50"] + row["fp_q50"] == pytest.approx(1 / 3)
    assert row["fn_q50"] + row["tp_q50"] == pytest.approx(2 / 3)
    expected_tp = np.sum(stats.norm.sf(45.0 - paco2_values[1:])) / 3
    assert row["tp_q50"] == pytest.approx(expected_tp)
    total = row["tn_q50"] + row["fp_q50"] + row["fn_q50"] + row["tp_q50"]
    assert total == pytest.approx(1.0, abs=1e-15)


def test_legacy_round_midpoints_belong_to_reported_half_open_bins() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [1.0], "tau2": [0.0]})

    curves = conditional_classification_curves(
        np.array([44.5, 45.5]),
        params,
        threshold=45.0,
        bin_method="round",
    )

    assert curves[["paco2_bin", "paco2_bin_upper", "count"]].to_dict("records") == [
        {"paco2_bin": 44.5, "paco2_bin_upper": 45.5, "count": 1},
        {"paco2_bin": 45.5, "paco2_bin_upper": 46.5, "count": 1},
    ]


@pytest.mark.parametrize("bin_method", ["cut", "floor"])
def test_decimal_width_exact_edges_remain_in_the_upper_half_open_bin(bin_method: str) -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [1.0], "tau2": [0.0]})
    below = np.nextafter(45.3, -np.inf)

    curves = conditional_classification_curves(
        np.array([below, 45.3, np.nextafter(45.3, np.inf)]),
        params,
        threshold=45.0,
        bin_width=0.1,
        bin_method=bin_method,
    )

    assert curves[["paco2_bin", "paco2_bin_upper", "count"]].to_dict("records") == [
        {"paco2_bin": 45.2, "paco2_bin_upper": 45.3, "count": 1},
        {"paco2_bin": 45.3, "paco2_bin_upper": 45.4, "count": 2},
    ]


def test_half_open_bins_keep_exact_upper_edge_values() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [1.0], "tau2": [0.0]})

    curves = conditional_classification_curves(
        np.array([44.0, 45.0]),
        params,
        threshold=45.0,
    )

    assert curves["count"].sum() == 2
    assert list(curves["paco2_bin"]) == [44.0, 45.0]


def test_conditional_negative_tail_remains_positive_at_z12() -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [1.0], "tau2": [0.0]})

    curves = conditional_classification_curves(
        np.array([33.0, 57.0]),
        params,
        threshold=45.0,
    )

    positive = curves.loc[curves["paco2_bin"] == 57.0].iloc[0]
    negative = curves.loc[curves["paco2_bin"] == 33.0].iloc[0]
    assert positive["fn_q50"] == pytest.approx(1.776482112077654e-33, rel=1e-14, abs=0.0)
    assert negative["fp_q50"] == pytest.approx(1.776482112077654e-33, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("threshold", [0.0, -1.0, np.nan, np.inf])
def test_conditional_rejects_invalid_threshold(threshold: float) -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [1.0], "tau2": [0.0]})

    with pytest.raises(ValueError, match="threshold"):
        conditional_classification_curves([40.0], params, threshold=threshold)


@pytest.mark.parametrize("paco2", [0.0, -1.0, np.nan, np.inf])
def test_conditional_rejects_invalid_paco2(paco2: float) -> None:
    params = pd.DataFrame({"delta": [0.0], "sigma2": [1.0], "tau2": [0.0]})

    with pytest.raises(ValueError, match="PaCO2 values"):
        conditional_classification_curves([paco2], params, threshold=45.0)


def _synthetic_bootstrap_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "n": [12.0, 14.0, 10.0, 16.0, 18.0, 20.0],
            "n_2": [12.0, 14.0, 10.0, 16.0, 18.0, 20.0],
            "bias": [-1.5, -0.8, -0.1, 0.3, 0.9, 1.4],
            "s2": [4.5, 5.0, 4.0, 6.5, 5.5, 6.0],
        }
    )
