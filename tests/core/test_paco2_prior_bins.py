from __future__ import annotations

import numpy as np
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


def _valid_prior() -> pd.DataFrame:
    return pd.DataFrame(
        [{"group": group, "paco2_bin": 40.0, "weight": 1.0} for group in PACO2_PRIOR_GROUPS]
    )


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_prior_rejects_invalid_paco2_support(value: float) -> None:
    prior = _valid_prior()
    prior.loc[0, "paco2_bin"] = value

    with pytest.raises(ValueError, match="PaCO2 values"):
        validate_paco2_prior_bins(prior)


@pytest.mark.parametrize("value", [None, "", "unknown"])
def test_prior_rejects_invalid_group_labels(value: str | None) -> None:
    prior = _valid_prior()
    prior.loc[0, "group"] = value

    with pytest.raises(ValueError, match="group|groups"):
        validate_paco2_prior_bins(prior)


def test_prior_has_no_hard_upper_paco2_limit() -> None:
    prior = _valid_prior()
    prior.loc[0, "paco2_bin"] = 500.0

    validated = validate_paco2_prior_bins(prior)

    assert 500.0 in validated["paco2_bin"].to_numpy()
