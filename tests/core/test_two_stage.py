from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from tcco2_accuracy.core.two_stage import _two_stage_log_probabilities
from tcco2_accuracy.two_stage import TwoStagePolicy, two_stage_metrics, two_stage_zone_probabilities


def test_two_stage_zone_probabilities_sum_to_one() -> None:
    paco2_values = np.array([35.0, 45.0, 55.0])
    policy = TwoStagePolicy(lower=40.0, upper=50.0, true_threshold=45.0)

    zone1, zone2, zone3 = two_stage_zone_probabilities(
        paco2_values, delta=1.0, sd_total=3.0, policy=policy
    )

    total = zone1 + zone2 + zone3
    assert np.allclose(total, 1.0, atol=1e-8)


def test_two_stage_post_test_probabilities_in_bounds() -> None:
    paco2_values = np.array([35.0, 45.0, 55.0, 60.0])
    policy = TwoStagePolicy(lower=40.0, upper=50.0, true_threshold=45.0)

    metrics = two_stage_metrics(paco2_values, delta=1.0, sd_total=4.0, policy=policy)

    for key in ("zone1_post_prob", "zone2_post_prob", "zone3_post_prob"):
        value = metrics[key]
        if np.isfinite(value):
            assert 0 <= value <= 1


def test_two_stage_probabilities_preserve_extreme_upper_tail_and_interval() -> None:
    policy = TwoStagePolicy(lower=28.0, upper=32.0, true_threshold=25.0)

    zone1, zone2, zone3 = two_stage_zone_probabilities(
        np.array([20.0]),
        delta=0.0,
        sd_total=1.0,
        policy=policy,
    )

    assert zone1[0] == pytest.approx(stats.norm.cdf(8.0), rel=1e-14)
    assert zone2[0] == pytest.approx(stats.norm.sf(8.0) - stats.norm.sf(12.0), rel=1e-14, abs=0.0)
    assert zone3[0] == pytest.approx(1.776482112077654e-33, rel=1e-14, abs=0.0)
    assert zone1[0] + zone2[0] + zone3[0] == pytest.approx(1.0, abs=1e-15)


def test_two_stage_narrow_ordered_zone_retains_positive_probability() -> None:
    policy = TwoStagePolicy(lower=0.0, upper=1e-16, true_threshold=1.0)

    _, zone2, _ = two_stage_zone_probabilities(
        np.array([1.0]),
        delta=1.0,
        sd_total=1.0,
        policy=policy,
    )

    assert zone2[0] > 0.0
    assert zone2[0] == pytest.approx(stats.norm.pdf(0.0) * 1e-16, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("lower_z", [1.0, -12.0])
def test_two_stage_adjacent_float_zone_uses_direct_interval_mass(lower_z: float) -> None:
    upper_z = np.nextafter(lower_z, np.inf)
    policy = TwoStagePolicy(lower=lower_z, upper=upper_z, true_threshold=1.0)

    _, zone2, _ = two_stage_zone_probabilities(
        np.array([10.0]),
        delta=10.0,
        sd_total=1.0,
        policy=policy,
    )

    expected = stats.norm.pdf(lower_z) * (upper_z - lower_z)
    assert zone2[0] > 0.0
    assert zone2[0] == pytest.approx(expected, rel=5e-14, abs=0.0)


def test_two_stage_far_tail_with_substantial_log_difference_avoids_quadrature() -> None:
    lower_z = np.array([100_000.0])
    upper_z = lower_z + 0.0006705523
    log_larger = stats.norm.logsf(lower_z)
    log_smaller = stats.norm.logsf(upper_z)
    expected = log_larger + np.log(-np.expm1(log_smaller - log_larger))

    _, actual, _ = _two_stage_log_probabilities(lower_z, upper_z)

    assert log_larger[0] - log_smaller[0] > 1.0
    assert actual[0] == pytest.approx(expected[0], abs=1e-8)


def test_two_stage_extreme_tail_lr_is_not_subtraction_infinity() -> None:
    policy = TwoStagePolicy(lower=40.0, upper=45.0, true_threshold=45.0)

    metrics = two_stage_metrics(
        np.array([33.0, 57.0]),
        delta=0.0,
        sd_total=1.0,
        policy=policy,
    )

    expected_zone3_lr = stats.norm.sf(-12.0) / stats.norm.sf(12.0)
    assert np.isfinite(metrics["zone3_lr"])
    assert metrics["zone3_lr"] == pytest.approx(expected_zone3_lr, rel=1e-13)


def test_two_stage_lr_uses_log_tails_beyond_probability_underflow() -> None:
    policy = TwoStagePolicy(lower=42.0, upper=45.0, true_threshold=45.0)

    metrics = two_stage_metrics(
        np.array([44.0, 45.0]),
        delta=40.0,
        sd_total=1.0,
        policy=policy,
    )

    expected_log_lr = stats.norm.logsf(40.0) - stats.norm.logsf(41.0)
    assert metrics["zone3_prob"] == 0.0
    assert np.isfinite(metrics["zone3_lr"])
    assert metrics["zone3_lr"] == pytest.approx(np.exp(expected_log_lr), rel=1e-13)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (np.nan, 2.0),
        (1.0, np.inf),
        (-np.inf, 2.0),
        (1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_two_stage_policy_rejects_nonfinite_or_unordered_boundaries(
    lower: float, upper: float
) -> None:
    with pytest.raises(ValueError, match="boundar|lower bound"):
        TwoStagePolicy(lower=lower, upper=upper, true_threshold=45.0)


@pytest.mark.parametrize("threshold", [0.0, -1.0, np.nan, np.inf])
def test_two_stage_policy_rejects_invalid_true_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        TwoStagePolicy(lower=40.0, upper=50.0, true_threshold=threshold)


def test_two_stage_policy_permits_ordered_negative_boundaries() -> None:
    policy = TwoStagePolicy(lower=-2.0, upper=-1.0, true_threshold=1.0)

    zone1, zone2, zone3 = two_stage_zone_probabilities(
        [1.0], delta=2.0, sd_total=1.0, policy=policy
    )

    assert zone1[0] + zone2[0] + zone3[0] == pytest.approx(1.0)


def test_two_stage_policy_normalizes_numeric_scalar_inputs() -> None:
    policy = TwoStagePolicy(lower="40", upper="50", true_threshold="45")  # type: ignore[arg-type]

    assert policy == TwoStagePolicy(lower=40.0, upper=50.0, true_threshold=45.0)


@pytest.mark.parametrize("paco2", [0.0, -1.0, np.nan, np.inf])
def test_two_stage_rejects_invalid_paco2_values(paco2: float) -> None:
    policy = TwoStagePolicy(lower=40.0, upper=50.0, true_threshold=45.0)

    with pytest.raises(ValueError, match="PaCO2 values"):
        two_stage_zone_probabilities([paco2], delta=0.0, sd_total=1.0, policy=policy)
