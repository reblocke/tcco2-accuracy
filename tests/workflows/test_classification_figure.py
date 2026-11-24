"""Exercise the forward direction, inclusive arterial cutoff and full-support aggregation."""

from pathlib import Path

import numpy as np
import pytest
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


def test_public_supplement_must_reproduce_credible_bounds(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/manuscript"))
    from classification_figure import CELLS, check_cells

    estimates = dict.fromkeys(CELLS, (1.0, 2.0, 4.0))
    rows = [{"Measure": label + " per 100", "All": "2.00 (1.00–4.00)"} for label in CELLS.values()]
    check_cells(estimates, "All", [rows, rows])
    for wrong in ("2.00", "2.00 (1.00–3.00)"):
        supplement = [dict(rows[0], All=wrong), *rows[1:]]
        with pytest.raises(ValueError, match="True positives"):
            check_cells(estimates, "All", [rows, supplement])
