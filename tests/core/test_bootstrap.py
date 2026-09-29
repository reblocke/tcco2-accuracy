from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import tcco2_accuracy.core.bootstrap as bootstrap_module
from tcco2_accuracy.bootstrap import bootstrap_conway_parameters


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
