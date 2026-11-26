"""A tilted two-component mixture distinguishes inverse risk from forward classification."""

from pathlib import Path

import numpy as np
from scipy.stats import norm


def test_inverse_risk_and_credible_bounds_follow_the_observation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/manuscript"))
    from inverse_figure import ExactReadingLaw, classification_proportions, summarize

    a = np.array([40.0, 45.0, 52.0, 90.0])
    w = np.array([0.83, 0.10, 0.06, 0.01])
    readings = np.array([44, 45, 52])
    theta = np.zeros((4, 128, 5))
    theta[:, ::64, 0] = -5
    law = ExactReadingLaw(a, w, readings)
    joint = law.joint(theta, 32)
    result = summarize(joint, joint)
    for j, reading in enumerate(readings):
        component = w * norm.pdf(a - reading - np.array([[-5.0], [0.0]]))
        evidence = component.sum(axis=1)
        observed_weights = evidence * [1 / 64, 63 / 64]
        for h, cut in enumerate((45, 52)):
            ratios = component[:, a >= cut].sum(axis=1) / evidence
            order = ratios.argsort()
            cumulative = np.cumsum(observed_weights[order]) / observed_weights.sum()
            expected = ratios[order][np.searchsorted(cumulative, [0.025, 0.975])]
            np.testing.assert_allclose(result["edges"][j, h], expected, atol=1e-13)
            np.testing.assert_allclose(
                result["mean"][j, h], np.average(ratios, weights=observed_weights), atol=1e-13
            )
            if reading == cut == 45:
                assert abs(result["mean"][j, h] - np.average(ratios, weights=[1, 63])) > 0.05
                assert expected[0] < 0.01 < expected[1]
    shares = classification_proportions(readings, result["mean"][:, 0])
    np.testing.assert_allclose(shares.sum(axis=1), 1)
    # Order is TN, FN, TP, FP. An observed reading of exactly 45 is a positive call.
    np.testing.assert_allclose(shares[0], [1 - result["mean"][0, 0], result["mean"][0, 0], 0, 0])
    np.testing.assert_allclose(shares[1], [0, 0, result["mean"][1, 0], 1 - result["mean"][1, 0]])


def test_unqualified_band_edges_are_rejected_independently(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/manuscript"))
    from inverse_figure import summarize

    # Histories with stable draw-level risk should pass. Move only the second threshold
    # at coarse quadrature, preserving total evidence and the first threshold's risk.
    joint = np.ones((4, 128, 1, 3))
    coarse = joint.copy()
    coarse[..., 1] += 0.04
    coarse[..., 2] -= 0.04
    result = summarize(joint, coarse)
    assert result["qualified"][0, 0].all()
    assert not result["qualified"][0, 1].any()


def test_streamed_selection_preserves_draw_order_and_checks_the_entire_file(tmp_path, monkeypatch):
    import hashlib

    import pytest

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/manuscript"))
    from inverse_draws import select_saved

    path = tmp_path / "theta.npy"
    theta = np.arange(4 * 512 * 5, dtype=float).reshape(4, 512, 5)
    np.save(path, theta)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    for n in (16, 73, 128, 512):
        np.testing.assert_array_equal(
            select_saved(path, sha, 512, n), theta[:, np.linspace(0, 511, n, dtype=int)]
        )
    with path.open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        select_saved(path, sha, 512, 23)
