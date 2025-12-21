from __future__ import annotations

import numpy as np
import pandas as pd

from tcco2_accuracy.bootstrap import bootstrap_conway_parameters
from tcco2_accuracy.data import load_conway_group


def test_bootstrap_reproducible() -> None:
    data = load_conway_group("main")
    draws_a = bootstrap_conway_parameters(data, n_boot=25, seed=123)
    draws_b = bootstrap_conway_parameters(data, n_boot=25, seed=123)

    pd.testing.assert_frame_equal(draws_a, draws_b)


def test_bootstrap_tau2_nonnegative_and_loa_spread() -> None:
    data = load_conway_group("icu")
    draws = bootstrap_conway_parameters(data, n_boot=30, seed=456)

    assert np.isfinite(draws[["delta", "sigma2", "tau2", "sd_total", "loa_l", "loa_u"]]).all().all()
    assert (draws["tau2"] >= 0).all()
    assert draws["loa_u"].max() - draws["loa_u"].min() > 0.1
    assert draws["loa_l"].max() - draws["loa_l"].min() > 0.1
