from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import tcco2_accuracy.core.bootstrap as bootstrap_module
from tcco2_accuracy.bootstrap import bootstrap_conway_parameters
from tcco2_accuracy.conway_meta import AGREEMENT_METHOD_VERSION, RESULTS_STATUS
from tcco2_accuracy.data import load_conway_group


def test_bootstrap_reproducible() -> None:
    data = load_conway_group("main")
    draws_a = bootstrap_conway_parameters(data, n_boot=25, seed=123)
    draws_b = bootstrap_conway_parameters(data, n_boot=25, seed=123)

    pd.testing.assert_frame_equal(draws_a, draws_b)
    assert draws_a["agreement_method_version"].unique().tolist() == [AGREEMENT_METHOD_VERSION]
    assert draws_a["results_status"].unique().tolist() == [RESULTS_STATUS]


def test_bootstrap_tau2_nonnegative_and_loa_spread() -> None:
    data = load_conway_group("icu")
    draws = bootstrap_conway_parameters(data, n_boot=30, seed=456)

    assert np.isfinite(draws[["delta", "sigma2", "tau2", "sd_total", "loa_l", "loa_u"]]).all().all()
    assert (draws["tau2"] >= 0).all()
    assert draws["loa_u"].max() - draws["loa_u"].min() > 0.1
    assert draws["loa_l"].max() - draws["loa_l"].min() > 0.1


def test_bootstrap_truncates_negative_tau2() -> None:
    data = pd.DataFrame(
        {
            "study": ["a", "b"],
            "n": [10.0, 10.0],
            "n_2": [10.0, 10.0],
            "bias": [0.0, 0.0],
            "s2": [4.0, 4.0],
        }
    )

    draws = bootstrap_conway_parameters(data, n_boot=10, seed=123, truncate_tau2=True)

    assert (draws["tau2"] >= 0).all()


def test_publication_cluster_sample_contributes_all_effect_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = pd.DataFrame(
        {
            "study": ["A (one)", "A (two)", "B"],
            "study_base": ["A", "A", "B"],
            "n": [20.0, 20.0, 20.0],
            "n_2": [20.0, 20.0, 20.0],
            "bias": [10.0, 20.0, 30.0],
            "s2": [4.0, 4.0, 4.0],
        }
    )
    observed_bias_rows: list[tuple[float, ...]] = []

    def fake_loa_summary(bias: object, *_args: object, **_kwargs: object) -> SimpleNamespace:
        values = tuple(np.asarray(bias, dtype=float))
        observed_bias_rows.append(values)
        return SimpleNamespace(bias=float(np.mean(values)), sd=1.0, tau2=0.0)

    monkeypatch.setattr(bootstrap_module, "loa_summary", fake_loa_summary)
    seed = 19
    sampled_clusters = np.random.default_rng(seed).choice(
        np.array(["A", "B"], dtype=object),
        size=2,
        replace=True,
    )
    expected_rows = tuple(
        bias
        for cluster in sampled_clusters
        for bias in ({"A": (10.0, 20.0), "B": (30.0,)}[str(cluster)])
    )

    bootstrap_conway_parameters(
        data,
        n_boot=1,
        seed=seed,
        study_id="study_base",
        bootstrap_mode="cluster_only",
    )

    assert observed_bias_rows == [expected_rows]
