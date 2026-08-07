from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.reference.conway_reference import (
    agreement_reference_from_inputs,
    corrected_reference,
    dersimonian_laird_tau2,
    prepare_reference_inputs,
)


def test_reference_module_is_independent_of_production_package() -> None:
    module_path = Path(__file__).with_name("conway_reference.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert "tcco2_accuracy" not in imported_roots
    assert "tests" not in imported_roots


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


def test_zero_heterogeneity_retains_raw_diagnostic_but_uses_zero_variance() -> None:
    bias = np.zeros(3)
    v_bias = np.full(3, 0.2)
    log_sigma2 = np.log(np.full(3, 4.0))
    var_log_sigma2 = np.full(3, 0.1)

    raw_tau2 = dersimonian_laird_tau2(bias, v_bias)
    summary = agreement_reference_from_inputs(bias, v_bias, log_sigma2, var_log_sigma2)

    assert raw_tau2 == pytest.approx(-0.2, abs=1e-12)
    assert summary["tau2_raw"] == pytest.approx(raw_tau2, abs=1e-12)
    assert summary["tau2"] == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(
        [
            summary["loa_l"],
            summary["loa_u"],
            summary["ci_l_mod"],
            summary["ci_u_mod"],
            summary["ci_l"],
            summary["ci_u"],
        ]
    ).all()


def test_near_zero_positive_heterogeneity_remains_finite() -> None:
    target_tau2 = 1e-10
    v_bias = np.full(3, 0.2)
    bias_extent = np.sqrt(v_bias[0] + target_tau2)
    summary = agreement_reference_from_inputs(
        np.array([-bias_extent, 0.0, bias_extent]),
        v_bias,
        np.log(np.full(3, 4.0)),
        np.full(3, 0.1),
    )

    assert summary["tau2_raw"] == pytest.approx(target_tau2, rel=0, abs=1e-15)
    assert summary["tau2"] == pytest.approx(target_tau2, rel=0, abs=1e-15)
    assert np.isfinite(
        [
            summary["loa_l"],
            summary["loa_u"],
            summary["ci_l_mod"],
            summary["ci_u_mod"],
            summary["ci_l"],
            summary["ci_u"],
        ]
    ).all()


def test_reference_natural_log_matches_coherent_base10_conversion() -> None:
    bias = np.array([-1.0, 0.5, 1.5])
    v_bias = np.array([0.2, 0.3, 0.4])
    variances = np.array([4.0, 9.0, 16.0])
    corrections = np.array([1 / 19, 1 / 23, 1 / 29])
    var_log_sigma2 = np.array([2 / 19, 2 / 23, 2 / 29])
    natural = np.log(variances) + corrections
    ln10 = np.log(10.0)
    base10 = np.log10(variances) + corrections / ln10

    natural_summary = agreement_reference_from_inputs(
        bias,
        v_bias,
        natural,
        var_log_sigma2,
    )
    converted_summary = agreement_reference_from_inputs(
        bias,
        v_bias,
        base10 * ln10,
        var_log_sigma2,
    )

    for field in natural_summary:
        assert converted_summary[field] == pytest.approx(
            natural_summary[field],
            rel=0,
            abs=1e-12,
        )


def test_reference_rejects_mixed_logarithm_mutant() -> None:
    s2 = 9.0
    n_participants = 4.0
    corrected = agreement_reference_from_inputs(
        np.array([0.0]),
        np.array([s2 / n_participants]),
        np.array([np.log(s2) + 1 / (n_participants - 1)]),
        np.array([2 / (n_participants - 1)]),
    )
    mixed_log_mutant_sigma2 = np.exp(np.log10(s2) + 1 / (n_participants - 1))

    assert corrected["sigma2"] == pytest.approx(s2 * np.exp(1 / 3), abs=1e-12)
    assert corrected["sigma2"] != pytest.approx(mixed_log_mutant_sigma2, rel=1e-6)


def test_reference_rejects_log_scale_tau2_coefficient_mutant() -> None:
    bias = np.array([-2.0, 0.0, 2.0])
    v_bias = np.full(3, 0.5)
    log_sigma2 = np.log(np.full(3, 4.0))
    var_log_sigma2 = np.full(3, 0.1)
    summary = agreement_reference_from_inputs(bias, v_bias, log_sigma2, var_log_sigma2)

    total_variance = summary["sigma2"] + summary["tau2"]
    base_terms = 1 / np.sum(1 / (v_bias + summary["tau2"])) + summary[
        "sigma2"
    ] ** 2 / total_variance * (0.1 / 3)
    direct_scale = base_terms + summary["var_tau2"] / total_variance
    log_scale_mutant = base_terms + summary["tau2"] ** 2 / total_variance * summary["var_tau2"]

    assert summary["tau2"] == pytest.approx(3.5, abs=1e-12)
    assert summary["var_loa_model"] == pytest.approx(direct_scale, abs=1e-12)
    assert summary["var_loa_model"] != pytest.approx(log_scale_mutant, rel=1e-6)
