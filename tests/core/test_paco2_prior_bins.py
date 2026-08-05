from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcco2_accuracy.data import (
    PACO2_PRIOR_GROUPS,
    load_default_paco2_prior,
    load_paco2_distribution,
    load_paco2_prior,
    load_paco2_prior_bins,
    load_paco2_prior_bins_bytes,
    validate_paco2_prior_bins,
)

SYNTHETIC_PRIOR_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_paco2_prior.csv"
)


def test_synthetic_prior_fixture_loads_without_counts() -> None:
    prior = load_paco2_prior_bins(SYNTHETIC_PRIOR_PATH)

    assert list(prior.columns) == ["group", "paco2_bin", "weight"]
    assert "count" not in prior.columns
    assert "density" not in prior.columns
    assert set(PACO2_PRIOR_GROUPS).issubset(set(prior["group"]))
    assert (prior["weight"] >= 0).all()
    weight_sums = prior.groupby("group")["weight"].sum()
    for group in PACO2_PRIOR_GROUPS:
        assert weight_sums.loc[group] == pytest.approx(1.0, abs=1e-6)


def test_legacy_xls_prior_is_rejected_as_uninspectable(tmp_path: Path) -> None:
    path = tmp_path / "prior.xls"
    path.write_bytes(b"legacy workbook placeholder")

    with pytest.raises(ValueError, match="Unsupported prior bin format"):
        load_paco2_prior_bins(path)
    with pytest.raises(ValueError, match="Uploaded prior must be CSV/XLSX"):
        load_paco2_prior_bins_bytes(path.read_bytes(), path.name)


def test_prior_loader_requires_upload_or_explicit_private_path() -> None:
    result = load_paco2_prior("all")

    assert result.values is None
    assert result.source is None
    assert result.paths_checked == ()
    assert result.error is not None
    assert "explicit" in result.error.message


def test_restricted_distribution_loader_never_auto_discovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied_read(*_args: object, **_kwargs: object) -> None:
        pytest.fail("read_stata must not be called")

    monkeypatch.setattr("tcco2_accuracy.data.pd.read_stata", denied_read)

    with pytest.raises(ValueError, match="explicit private PaCO2 source path"):
        load_paco2_distribution()


def test_legacy_default_prior_helper_requires_explicit_path() -> None:
    with pytest.raises(ValueError, match="explicit private PaCO2 prior-bin path"):
        load_default_paco2_prior("all")


def test_prior_loader_accepts_explicit_synthetic_path() -> None:
    result = load_paco2_prior("all", default_bins_path=SYNTHETIC_PRIOR_PATH)

    assert result.values is not None
    assert result.weights is not None
    assert result.source == "explicit_bins"
    assert result.paths_checked == (SYNTHETIC_PRIOR_PATH,)


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
