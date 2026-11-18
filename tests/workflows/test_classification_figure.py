"""Exercise the forward direction, inclusive arterial cutoff and full-support aggregation."""

from pathlib import Path

import numpy as np
from scipy.stats import norm

from tcco2_accuracy.core.future_setting import PredictiveKernel


def test_classification_figure_with_known_offset(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/manuscript"))
    from classification_figure import calculate

    # D = A - T is N(2, 1). Include mass outside both plot limits and at exactly 45.
    theta = np.array([[[2.0, 0, 0, 0, 0]]])
    arterial = np.array([25.0, 44.0, 45.0, 47.0, 80.0])
    mass = np.array([0.15, 0.20, 0.30, 0.25, 0.10])
    x, curves, cells = calculate(PredictiveKernel(theta), arterial, mass)
    p = norm.cdf(arterial - 47)
    expected = {
        "TP": sum(mass[2:] * p[2:]),
        "FN": sum(mass[2:] * (1 - p[2:])),
        "FP": sum(mass[:2] * p[:2]),
        "TN": sum(mass[:2] * (1 - p[:2])),
    }
    np.testing.assert_allclose(curves, np.tile(norm.cdf(x - 47), (3, 1)), atol=1e-14)
    for name, value in expected.items():
        np.testing.assert_allclose(cells[name], 100 * value)
