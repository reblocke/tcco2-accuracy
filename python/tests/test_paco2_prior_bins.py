from __future__ import annotations

import pytest

from tcco2_accuracy.data import (
    PACO2_PRIOR_BINS_PATH,
    PACO2_PRIOR_GROUPS,
    load_paco2_prior_bins,
)


def test_default_prior_bins_load() -> None:
    assert PACO2_PRIOR_BINS_PATH.exists()
    prior = load_paco2_prior_bins(PACO2_PRIOR_BINS_PATH)
    assert set(PACO2_PRIOR_GROUPS).issubset(set(prior["group"]))
    weight_sums = prior.groupby("group")["weight"].sum()
    for group in PACO2_PRIOR_GROUPS:
        assert weight_sums.loc[group] == pytest.approx(1.0, abs=1e-6)
