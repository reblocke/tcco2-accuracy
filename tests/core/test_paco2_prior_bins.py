from __future__ import annotations

import pandas as pd
import pytest

from tcco2_accuracy.data import (
    PACO2_PRIOR_GROUPS,
    PACO2_PUBLIC_PRIOR_PATH,
    load_paco2_prior_bins,
    validate_paco2_prior_bins,
)


def test_public_prior_loads_without_counts() -> None:
    assert PACO2_PUBLIC_PRIOR_PATH.exists()
    prior = load_paco2_prior_bins(PACO2_PUBLIC_PRIOR_PATH)

    assert list(prior.columns) == ["group", "paco2_bin", "weight"]
    assert "count" not in prior.columns
    assert "density" not in prior.columns
    assert set(PACO2_PRIOR_GROUPS).issubset(set(prior["group"]))
    assert (prior["paco2_bin"] % 1 == 0).all()
    assert (prior["weight"] >= 0).all()
    weight_sums = prior.groupby("group")["weight"].sum()
    for group in PACO2_PRIOR_GROUPS:
        assert weight_sums.loc[group] == pytest.approx(1.0, abs=1e-6)


def test_weight_only_prior_requires_normalized_group_weights() -> None:
    rows = []
    for group in PACO2_PRIOR_GROUPS:
        rows.extend(
            [
                {"group": group, "paco2_bin": 40, "weight": 0.25},
                {"group": group, "paco2_bin": 41, "weight": 0.25},
            ]
        )

    with pytest.raises(ValueError, match="Prior weights must sum to 1"):
        validate_paco2_prior_bins(pd.DataFrame(rows))
