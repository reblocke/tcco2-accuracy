"""Acceptance cases for whole-mmHg displays and fixed two-step policies."""

import json
from pathlib import Path

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


def test_referral_cache_and_certificate_require_the_current_methods(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/manuscript"))
    import referral_analysis
    import referral_report

    from tcco2_accuracy.core import future_setting

    store = tmp_path / "store"
    cache = tmp_path / "cache"
    store.mkdir()
    cache.mkdir()
    prior = store / "all-weights.csv"
    prior.write_text("a,w\n40,0.5\n50,0.5\n")
    (store / "cohort.json").write_text(
        json.dumps({"all": {"prior_sha256": referral_analysis.digest(prior)}})
    )
    (store / "manifest.json").write_text("{}")
    theta = np.zeros((4, 4, 5))
    monkeypatch.setattr(
        referral_analysis, "classification_draws", lambda *a, **kw: [("all", theta)]
    )
    source = tmp_path / "future_setting.py"
    source.write_bytes(Path(future_setting.__file__).read_bytes())
    monkeypatch.setattr(future_setting, "__file__", str(source))

    def generate(order):
        referral_analysis.run(store, store, cache, "all", 4, order, 1)

    receipts = {}
    for order in (32, 64):
        generate(order)
        receipts[str(order)] = json.loads((cache / f"all-4-{order}.json").read_bytes())
    cert = cache / "all-bin-certificate.npz"
    q = np.full(81, 0.5)
    np.savez(cert, q=q, lower=q, upper=q, quadrature=np.zeros_like(q), per_chain=4)
    precision_script = Path(referral_report.__file__).with_name("referral_precision.py")
    entry = {
        "output": referral_analysis.digest(cert),
        "sources": receipts,
        "precision_code": referral_analysis.digest(precision_script),
    }
    generate(32)
    referral_report.load_group(cache, "all", entry)

    legacy = {k: v for k, v in entry.items() if k != "precision_code"}
    for stale in (legacy, dict(entry, precision_code="0" * 64)):
        with pytest.raises(ValueError, match="precision implementation"):
            referral_report.load_group(cache, "all", stale)

    source.write_bytes(source.read_bytes() + b"\n# synthetic kernel revision\n")
    with pytest.raises(ValueError, match="different inputs"):
        generate(32)
    with pytest.raises(ValueError, match="numerical implementation"):
        referral_report.load_group(cache, "all", entry)


def test_rounded_forward_law_and_inverse_agree_with_independent_normal_truth():
    a = np.array([37.4, 45.0, 54.8])
    w = np.array([0.3, 0.5, 0.2])
    theta = np.zeros((4, 16, 5))
    theta[:2, :, 0] = 3
    theta[2:, :, 0] = -12
    theta[..., 1] = np.log(9.0)
    joint = class_joint(PredictiveKernel(theta), a, w)
    edges = np.r_[-np.inf, DISPLAY - 0.5, 100.5, np.inf]
    for chain in range(4):
        truth = np.diff(norm.cdf(edges[:, None], loc=a - theta[chain, 0, 0], scale=3), axis=0)
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
    np.testing.assert_allclose(m["efficiency"], 0.78)
    assert np.all(m["FNR"] <= 0.05) and np.all(m["FPR"] <= 0.05)
    none = evaluate_policy(x, (19, 101))
    assert np.all(none["referral"] == 1)
    assert np.isnan(none["negative_wrong"]).all() and np.isnan(none["positive_wrong"]).all()
    assert np.all(none["FN"] + none["FP"] == 0)


def test_inclusive_probability_cap_and_common_draw_plot_denominators():
    from scripts.manuscript.referral_report import figure_counts, summarize

    q = np.linspace(0.05, 0.95, 81)
    assert display_policy(q, q, 0.05) == (20, 100)
    rng = np.random.default_rng(9801)
    x = rng.dirichlet(np.ones(166), size=(4, 64)).reshape(4, 64, 2, 83)
    mean = x.mean(axis=(0, 1))
    metrics = evaluate_policy(x, (39, 51))
    counts = figure_counts(mean, (39, 51))
    np.testing.assert_allclose(counts.sum(axis=0), mean.sum(axis=0) * 100)
    assert counts[1].sum() == pytest.approx(metrics["FN"].mean() * 100)
    assert counts[3].sum() == pytest.approx(metrics["FP"].mean() * 100)
    assert counts[4].sum() == pytest.approx(metrics["referral"].mean() * 100)
    row = summarize(x, (39, 51), q, q, q)
    np.testing.assert_allclose(
        row["median_cri"]["efficiency"],
        np.quantile(metrics["efficiency"], [0.025, 0.5, 0.975]) * 100,
    )


def test_full_point_control_recovers_rounded_risk_with_equal_drawwise_midpoints():
    from tcco2_accuracy.core.referral_policy import corrected_bin_risk

    a = np.array([36.0, 45.0, 57.0])
    w = np.array([0.4, 0.4, 0.2])
    theta = np.zeros((4, 16, 5))
    theta[..., 1] = np.log(16)
    theta[2:, :, 0] = 5
    joint = class_joint(PredictiveKernel(theta), a, w)
    offsets = a[:, None] - DISPLAY
    density = norm.pdf(offsets[None, None] - theta[..., 0, None, None], scale=4)
    point = np.stack(
        [np.einsum("cdak,a->cdk", density, w * mask) for mask in (a < 45, a >= 45)], axis=2
    )
    center = point.mean(axis=(0, 1))
    point_q = center[1] / center.sum(axis=0)
    inferred, se = corrected_bin_risk(joint, point, point_q, np.zeros_like(point_q))
    expected, _, _ = risk_bounds(joint, joint)
    np.testing.assert_allclose(inferred, expected, atol=1e-14)
    assert np.all(np.isfinite(se))
