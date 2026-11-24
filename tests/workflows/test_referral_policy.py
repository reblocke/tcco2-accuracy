"""Acceptance cases for whole-mmHg displays and fixed two-step policies."""

import numpy as np
import pytest
from scipy.stats import norm

from tcco2_accuracy.core.future_setting import PredictiveKernel
from tcco2_accuracy.core.referral_policy import (
    DISPLAY,
    class_joint,
    class_policy,
    display_policy,
    evaluate_policy,
    risk_bounds,
)


def test_rounded_forward_law_and_inverse_agree_with_independent_normal_truth():
    a = np.array([37.4, 45.0, 54.8])
    w = np.array([0.3, 0.5, 0.2])
    theta = np.zeros((4, 16, 5))
    theta[:2, :, 0] = 3
    theta[2:, :, 0] = -7
    theta[..., 1] = np.log(9.0)
    joint = class_joint(PredictiveKernel(theta), a, w)
    edges = np.r_[-np.inf, DISPLAY - 0.5, 100.5, np.inf]
    for chain in range(4):
        truth = np.diff(norm.cdf(edges[:, None], loc=a-theta[chain, 0, 0], scale=3), axis=0)
        for c, mask in enumerate((a < 45, a >= 45)):
            np.testing.assert_allclose(joint[chain, 0, c], truth[:, mask] @ w[mask], atol=2e-15)
    q, _, _ = risk_bounds(joint, joint)
    at = 45 - DISPLAY[0]
    sub = joint[:, :, :, at + 1]
    assert q[at] == pytest.approx(sub[:, :, 1].mean() / sub.sum(axis=2).mean())
    assert abs(q[at] - (sub[:, :, 1] / sub.sum(axis=2)).mean()) > 0.01
    metrics = evaluate_policy(joint, (39, 51))
    np.testing.assert_allclose(metrics["efficiency"] + metrics["referral"], 1)
    np.testing.assert_allclose(metrics["negative"] + metrics["positive"], metrics["efficiency"])
    np.testing.assert_allclose(metrics["FN"], joint[:, :, 1, 1:21].sum(axis=-1))
    np.testing.assert_allclose(metrics["FP"], joint[:, :, 0, 32:-1].sum(axis=-1))
    assert np.all(metrics["off_grid"] <= metrics["referral"])


def test_prefix_suffix_and_numerical_holds_do_not_bridge_isolated_qualifiers():
    q = np.linspace(0.01, 0.99, len(DISPLAY))
    q[:4] = [0.01, 0.02, 0.051, 0.03]
    lo, hi = q.copy(), q.copy()
    hi[1] = 0.05001
    assert display_policy(lo, hi, 0.05)[0] == 20
    assert display_policy(lo, hi, 0.005) == (19, 101)
    wide = display_policy(lo, hi, 0.1)
    narrow = display_policy(lo, hi, 0.05)
    assert wide[0] >= narrow[0] and wide[1] <= narrow[1]


def test_class_denominators_optimum_and_empty_call_regions():
    # Negative/positive cohort masses at displays 20, 21, and 100.
    x = np.zeros((4, 16, 2, 83))
    x[:, :, 0, [1, 2, -2]] = [0.42, 0.02, 0.01]
    x[:, :, 1, [1, 2, -2]] = [0.01, 0.20, 0.34]
    cut = class_policy(x.mean(axis=(0, 1)), 0.05)
    m = evaluate_policy(x, cut)
    np.testing.assert_allclose(m['efficiency'], 0.78)
    assert np.all(m['FNR'] <= 0.05) and np.all(m['FPR'] <= 0.05)
    none = evaluate_policy(x, (19, 101))
    assert np.all(none['referral'] == 1)
    assert np.isnan(none['negative_wrong']).all() and np.isnan(none['positive_wrong']).all()
    assert np.all(none['FN'] + none['FP'] == 0)
