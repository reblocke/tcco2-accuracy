from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tcco2_accuracy.conway_meta import (
    conway_group_summary,
    loa_summary,
    prepare_conway_inputs,
    random_effects_meta,
)
from tcco2_accuracy.core.conway_meta import _loa_variance_components
from tcco2_accuracy.data import load_conway_group

CORRECTED_TARGETS = {
    "main": (
        -0.113811361575801,
        4.02746808637818,
        8.85951341077083,
        -11.2800649532579,
        11.0524422301063,
    ),
    "icu": (
        -0.596087967926880,
        4.69358324490349,
        1.88742995591776,
        -12.3538952915815,
        11.1617193557278,
    ),
    "arf": (
        1.69183584577254,
        4.89029630952780,
        3.15749561582049,
        -12.3965448415699,
        15.7802165331150,
    ),
    "lft": (
        -0.046951514427622,
        2.78249509283438,
        1.44665912497586,
        -11.1629173900598,
        11.0690143612045,
    ),
}


def _reference_random_effects(
    effect: np.ndarray,
    variance: np.ndarray,
    *,
    truncate_tau2: bool = False,
) -> tuple[float, float, float, float]:
    """Independent test implementation of the random-effects calculations."""

    weights_fixed = 1 / variance
    sum_weights_fixed = np.sum(weights_fixed)
    mean_fixed = np.sum(effect * weights_fixed) / sum_weights_fixed
    q_stat = np.sum(weights_fixed * (effect - mean_fixed) ** 2)
    dl_denominator = sum_weights_fixed - np.sum(weights_fixed**2) / sum_weights_fixed
    tau2 = (q_stat - (effect.size - 1)) / dl_denominator
    if truncate_tau2:
        tau2 = max(0.0, tau2)

    weights_random = 1 / (variance + tau2)
    sum_weights_random = np.sum(weights_random)
    mean_random = np.sum(effect * weights_random) / sum_weights_random
    var_model = 1 / sum_weights_random
    var_robust = (
        (effect.size / (effect.size - 1))
        * np.sum(weights_random**2 * (effect - mean_random) ** 2)
        / sum_weights_random**2
    )
    return float(mean_random), float(tau2), float(var_model), float(var_robust)


def _corrected_reference(data: pd.DataFrame) -> dict[str, float]:
    """Evaluate Tipton-Shuster Eqs. 4.5, 4.13, and 4.16 without production helpers."""

    n_pairs = data["n"].to_numpy(dtype=float)
    n_participants = data["n_2"].to_numpy(dtype=float)
    repeated = data["c"].fillna(data["n"] / data["n_2"]).to_numpy(dtype=float)
    s2_adjusted = data["s2"].to_numpy(dtype=float) * (1 + (repeated - 1) / (n_pairs - repeated))
    v_bias = s2_adjusted / n_participants
    log_sigma2 = np.log(s2_adjusted) + 1 / (n_participants - 1)
    var_log_sigma2 = 2 / (n_participants - 1)

    bias, tau2, var_bias, var_bias_robust = _reference_random_effects(
        data["bias"].to_numpy(dtype=float),
        v_bias,
    )
    pooled_log_sigma2, _, var_log_sigma2_pool, var_log_sigma2_robust = _reference_random_effects(
        log_sigma2, var_log_sigma2
    )
    sigma2 = float(np.exp(pooled_log_sigma2))
    total_variance = sigma2 + tau2
    var_tau2 = 2 / np.sum((v_bias + tau2) ** -2)
    b_sigma2 = sigma2**2 / total_variance
    b_tau2 = 1 / total_variance
    var_loa_robust = var_bias_robust + b_sigma2 * var_log_sigma2_robust + b_tau2 * var_tau2
    loa_half_width = 2 * np.sqrt(total_variance)
    tcrit = stats.t.ppf(0.975, data.shape[0] - 1)
    ci_half_width = tcrit * np.sqrt(var_loa_robust)
    return {
        "bias": bias,
        "sd": float(np.sqrt(sigma2)),
        "tau2": tau2,
        "loa_l": bias - loa_half_width,
        "loa_u": bias + loa_half_width,
        "ci_l": bias - loa_half_width - ci_half_width,
        "ci_u": bias + loa_half_width + ci_half_width,
        "var_model": var_bias,
        "var_log_sigma2_model": var_log_sigma2_pool,
    }


@pytest.mark.parametrize("group", CORRECTED_TARGETS)
def test_corrected_public_groups_match_independent_reference(group: str) -> None:
    data = load_conway_group(group)
    summary = conway_group_summary(data)
    reference = _corrected_reference(data)

    for field in ("bias", "sd", "tau2", "loa_l", "loa_u", "ci_l", "ci_u"):
        assert getattr(summary, field) == pytest.approx(reference[field], abs=1e-10)

    expected = CORRECTED_TARGETS[group]
    actual = (summary.bias, summary.sd, summary.tau2, summary.ci_l, summary.ci_u)
    assert actual == pytest.approx(expected, abs=1e-10)


def test_bolliger_row_uses_natural_log_hand_calculation() -> None:
    data = load_conway_group("main")
    bolliger = data.loc[data["study"] == "Bolliger 2007 (TOSCA - ICU)"].copy()
    inputs = prepare_conway_inputs(bolliger)
    row = bolliger.iloc[0]
    expected_log_sigma2 = np.log(row["s2"]) + 1 / (row["n_2"] - 1)
    expected_sigma2 = np.exp(expected_log_sigma2)
    legacy_mixed_base_sigma2 = np.exp(np.log10(row["s2"]) + 1 / (row["n_2"] - 1))

    loa = loa_summary(
        inputs["bias"],
        inputs["v_bias"],
        inputs["logs2"],
        inputs["v_logs2"],
        truncate_tau2=True,
    )

    assert inputs["logs2"].iloc[0] == pytest.approx(expected_log_sigma2, abs=1e-12)
    assert loa.sd**2 == pytest.approx(expected_sigma2, abs=1e-12)
    assert expected_sigma2 != pytest.approx(legacy_mixed_base_sigma2, rel=0.5)


def test_natural_log_matches_coherently_converted_base10_inputs() -> None:
    data = pd.DataFrame(
        {
            "study": ["a", "b", "c"],
            "n": [20.0, 24.0, 30.0],
            "n_2": [20.0, 24.0, 30.0],
            "bias": [-1.0, 0.5, 1.5],
            "s2": [4.0, 9.0, 16.0],
        }
    )
    inputs = prepare_conway_inputs(data)
    ln10 = np.log(10.0)
    base10_logs2 = np.log10(inputs["s2_adj"]) + 1 / ((inputs["n_2"] - 1) * ln10)
    base10_v_logs2 = 2 / ((inputs["n_2"] - 1) * ln10**2)
    natural = loa_summary(
        inputs["bias"],
        inputs["v_bias"],
        inputs["logs2"],
        inputs["v_logs2"],
        truncate_tau2=True,
    )
    converted = loa_summary(
        inputs["bias"],
        inputs["v_bias"],
        base10_logs2 * ln10,
        base10_v_logs2 * ln10**2,
        truncate_tau2=True,
    )

    for field in natural.__dataclass_fields__:
        assert getattr(natural, field) == pytest.approx(getattr(converted, field), abs=1e-12)


def test_positive_tau2_uses_direct_scale_variance_coefficient() -> None:
    bias = np.array([-2.0, 0.0, 2.0])
    v_bias = np.full(3, 0.5)
    logs2 = np.log(np.array([4.0, 5.0, 6.0]))
    v_logs2 = np.full(3, 0.1)

    loa = loa_summary(bias, v_bias, logs2, v_logs2, truncate_tau2=True)
    expected_tau2 = 3.5
    expected_sigma2 = float(np.cbrt(4 * 5 * 6))
    total_variance = expected_sigma2 + expected_tau2
    var_tau2 = 2 / np.sum((v_bias + expected_tau2) ** -2)
    direct_tau2_contribution = var_tau2 / total_variance
    equivalent_log_tau2_contribution = (expected_tau2**2 / total_variance) * (
        var_tau2 / expected_tau2**2
    )
    expected_var_loa = (
        1 / np.sum(1 / (v_bias + expected_tau2))
        + expected_sigma2**2 / total_variance * (0.1 / 3)
        + direct_tau2_contribution
    )
    tcrit = stats.t.ppf(0.975, 2)
    inferred_var_loa = ((loa.loa_l - loa.ci_l_mod) / tcrit) ** 2

    assert loa.tau2 == pytest.approx(expected_tau2, abs=1e-12)
    assert direct_tau2_contribution == pytest.approx(equivalent_log_tau2_contribution, abs=1e-12)
    assert inferred_var_loa == pytest.approx(expected_var_loa, abs=1e-12)
    assert inferred_var_loa != pytest.approx(
        expected_var_loa - direct_tau2_contribution + expected_tau2**2 / total_variance * var_tau2,
        rel=1e-3,
    )


def test_default_summaries_truncate_negative_raw_tau2_before_loa_calculations() -> None:
    data = pd.DataFrame(
        {
            "study": ["a", "b", "c"],
            "n": [20.0, 20.0, 20.0],
            "n_2": [20.0, 20.0, 20.0],
            "bias": [0.0, 0.0, 0.0],
            "s2": [4.0, 4.0, 4.0],
        }
    )
    inputs = prepare_conway_inputs(data)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = random_effects_meta(inputs["bias"], inputs["v_bias"])
    with np.errstate(divide="raise", invalid="raise"):
        loa = loa_summary(
            inputs["bias"],
            inputs["v_bias"],
            inputs["logs2"],
            inputs["v_logs2"],
        )
        group = conway_group_summary(data)

    assert raw.tau2 == pytest.approx(-0.2)
    assert loa.tau2 == pytest.approx(0.0)
    assert group.tau2 == pytest.approx(0.0)
    assert np.isfinite(
        [
            loa.bias,
            loa.sd,
            loa.loa_l,
            loa.loa_u,
            loa.ci_l_mod,
            loa.ci_u_mod,
            loa.ci_l_rve,
            loa.ci_u_rve,
            group.loa_l,
            group.loa_u,
            group.ci_l,
            group.ci_u,
        ]
    ).all()
    assert loa.loa_l == pytest.approx(loa.bias - 2 * loa.sd)
    assert loa.loa_u == pytest.approx(loa.bias + 2 * loa.sd)


def test_loa_variance_components_reject_invalid_denominators() -> None:
    with pytest.raises(ValueError, match=r"sigma2 \+ tau2"):
        _loa_variance_components(1.0, -1.0, np.array([2.0]))
    with pytest.raises(ValueError, match=r"v_bias \+ tau2"):
        _loa_variance_components(2.0, -1.0, np.array([1.0, 2.0]))


def test_agreement_summary_is_equivariant_to_unit_rescaling() -> None:
    data = pd.DataFrame(
        {
            "study": ["a", "b", "c", "d"],
            "n": [20.0, 25.0, 30.0, 35.0],
            "n_2": [20.0, 25.0, 30.0, 35.0],
            "bias": [-2.0, -0.5, 1.0, 2.5],
            "s2": [4.0, 6.0, 9.0, 12.0],
        }
    )
    scale = 7.0
    rescaled = data.copy()
    rescaled["bias"] *= scale
    rescaled["s2"] *= scale**2

    original_summary = conway_group_summary(data, truncate_tau2=True)
    rescaled_summary = conway_group_summary(rescaled, truncate_tau2=True)

    for field in ("bias", "sd", "loa_l", "loa_u", "ci_l", "ci_u"):
        assert getattr(rescaled_summary, field) == pytest.approx(
            getattr(original_summary, field) * scale,
            rel=1e-12,
        )
    assert rescaled_summary.tau2 == pytest.approx(original_summary.tau2 * scale**2, rel=1e-12)


def test_single_study_meta_edge_case() -> None:
    data = pd.DataFrame(
        {
            "study": ["solo"],
            "n": [10.0],
            "n_2": [10.0],
            "bias": [1.2],
            "s2": [4.0],
        }
    )
    inputs = prepare_conway_inputs(data)
    meta = random_effects_meta(inputs["bias"], inputs["v_bias"])
    loa = loa_summary(inputs["bias"], inputs["v_bias"], inputs["logs2"], inputs["v_logs2"])

    assert meta.studies == 1
    assert meta.tau2 == pytest.approx(0.0)
    assert meta.var_robust == pytest.approx(meta.var_model)
    assert np.isfinite(meta.mean)
    assert loa.studies == 1
    assert np.isfinite([loa.bias, loa.sd, loa.loa_l, loa.loa_u]).all()
    assert np.isnan(loa.ci_l_mod)
    assert np.isnan(loa.ci_u_mod)
    assert np.isnan(loa.ci_l_rve)
    assert np.isnan(loa.ci_u_rve)
