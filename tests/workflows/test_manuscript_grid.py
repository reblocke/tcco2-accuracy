"""Exercise the manuscript grid, including independent numerical and hold decisions."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/manuscript"


def test_full_grid_matches_analytic_truth_and_detects_bad_rows(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from predictions import inverse
    from verify_grid import verify_rows

    lib = tmp_path / "mixture.so"
    flags = (
        ["-dynamiclib", "-framework", "Accelerate"]
        if sys.platform == "darwin"
        else ["-shared", "-fPIC"]
    )
    subprocess.run(
        ["cc", "-O2", *flags, str(SCRIPTS / "mixture.c"), "-o", str(lib), "-lm"], check=True
    )
    # Include exact threshold support points, two biases, and short final batches.
    a = np.array([15.0, 36.0, 45.0, 52.0, 71.0, 110.0])
    w = np.array([0.01, 0.26, 0.44, 0.18, 0.10, 0.01])
    theta = np.tile(
        [[4.0, np.log(16), 0.0, 0.0, 0.0], [-5.0, np.log(49), 0.0, 0.0, 0.0]], (4, 33, 1)
    )
    rows, checks = inverse(theta, a, w, lib)
    frame = pd.DataFrame(
        rows, columns=["tc_mmhg", "lower95_mmhg", "median", "upper95_mmhg", "p_ge45", "p_ge52"]
    )
    assert frame.tc_mmhg.tolist() == list(range(20, 101))
    analytic = norm.pdf(
        a[:, None, None] - np.arange(20, 101)[None, :, None], [4.0, -5.0], [4.0, 7.0]
    ).mean(axis=2)
    mass = w[:, None] * analytic
    mass /= mass.sum(axis=0)
    for row, probs in zip(frame.itertuples(), mass.T, strict=True):
        np.testing.assert_array_equal(
            [row.lower95_mmhg, row.median, row.upper95_mmhg],
            a[np.searchsorted(probs.cumsum(), [0.025, 0.5, 0.975])],
        )
        np.testing.assert_allclose(
            [row.p_ge45, row.p_ge52], [probs[a >= k].sum() for k in (45, 52)], atol=1e-12, rtol=0
        )
    assert verify_rows(theta, a, w, frame, checks)["rows"] == 81
    for tc, field in [(20, "upper95_mmhg"), (68, "median"), (100, "p_ge52")]:
        altered = frame.copy()
        altered.loc[altered.tc_mmhg == tc, field] = np.nan
        with pytest.raises(ValueError, match=str(tc)):
            verify_rows(theta, a, w, altered, checks)
    with pytest.raises(ValueError, match="20.*100"):
        verify_rows(theta, a, w, frame.iloc[:-1], checks)

    # A quantile has to be the identical empirical support point, even for close neighbors.
    shifted = frame.copy()
    shifted.loc[shifted.tc_mmhg == 29, "median"] += 1e-12
    with pytest.raises(ValueError, match="29"):
        verify_rows(theta, a, w, shifted, checks)

    # Distinct chain evidence and correlated setting effects create legitimate holds.
    theta[:, :, 2:] = [1.8, 0.35, -0.4]
    theta[:2, :, 0] -= 8
    theta[2:, :, 0] += 10
    rows, checks = inverse(theta, a, w, lib)
    held = pd.DataFrame(rows, columns=frame.columns)
    assert held.p_ge45.isna().any() and held["median"].isna().any()
    verify_rows(theta, a, w, held, checks)
    i = held.index[held.p_ge45.isna()][0]
    held.loc[i, "p_ge45"] = checks[i]["ratios"][0]
    with pytest.raises(ValueError, match=str(20 + i)):
        verify_rows(theta, a, w, held, checks)
