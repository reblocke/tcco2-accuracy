from __future__ import annotations

import numpy as np

from tcco2_accuracy.two_stage import TwoStagePolicy, two_stage_metrics, two_stage_zone_probabilities


def test_two_stage_zone_probabilities_sum_to_one() -> None:
    paco2_values = np.array([35.0, 45.0, 55.0])
    policy = TwoStagePolicy(lower=40.0, upper=50.0, true_threshold=45.0)

    zone1, zone2, zone3 = two_stage_zone_probabilities(paco2_values, delta=1.0, sd_total=3.0, policy=policy)

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
