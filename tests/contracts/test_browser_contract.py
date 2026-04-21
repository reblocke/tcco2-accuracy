from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from tcco2_accuracy.browser_contract import build_bootstrap_payload, compute_ui_payload
from tcco2_accuracy.data import PACO2_PRIOR_GROUPS, prior_distribution_from_bins
from tcco2_accuracy.ui_api import predict_paco2_from_tcco2

ROOT = Path(__file__).resolve().parents[2]


def test_browser_contract_matches_ui_api_canonical_prior_weighted() -> None:
    params_csv = _read_text(ROOT / "artifacts" / "bootstrap_params.csv")
    prior_csv = _read_text(ROOT / "Data" / "paco2_public_prior.csv")
    payload = {
        "tcco2": 50.0,
        "subgroup": "all",
        "threshold": 45.0,
        "mode": "prior_weighted",
        "interval": 0.95,
        "params_csv": params_csv,
        "prior_bins_csv": prior_csv,
        "n_param_draws": 1000,
        "seed": 202401,
    }

    browser = compute_ui_payload(payload)

    params = pd.read_csv(StringIO(params_csv))
    prior_bins = pd.read_csv(StringIO(prior_csv))
    prior_values, prior_weights = prior_distribution_from_bins(prior_bins, "all")
    direct = predict_paco2_from_tcco2(
        tcco2=50.0,
        subgroup="all",
        threshold=45.0,
        mode="prior_weighted",
        interval=0.95,
        params_draws=params,
        paco2_prior_values=prior_values,
        paco2_prior_weights=prior_weights,
        n_param_draws=1000,
        seed=202401,
    )

    assert browser["paco2_median"] == pytest.approx(direct.paco2_median)
    assert browser["paco2_q_low"] == pytest.approx(direct.paco2_q_low)
    assert browser["paco2_q_high"] == pytest.approx(direct.paco2_q_high)
    assert browser["p_ge_threshold"] == pytest.approx(direct.p_ge_threshold)
    assert sum(browser["posterior_prob"]) == pytest.approx(1.0)
    assert direct.likelihood_prob is not None
    assert browser["likelihood_prob"] is not None
    assert browser["likelihood_prob"] == pytest.approx(direct.likelihood_prob)
    assert sum(browser["likelihood_prob"]) == pytest.approx(1.0)


@pytest.mark.parametrize("subgroup", ["all", "pft", "ed_inp", "icu"])
@pytest.mark.parametrize("mode", ["prior_weighted", "likelihood_only"])
def test_browser_contract_canonical_cases_are_serializable(subgroup: str, mode: str) -> None:
    payload = {
        "tcco2": 46.5,
        "subgroup": subgroup,
        "threshold": 50.0,
        "mode": mode,
        "interval": 0.95,
        "params_csv": _read_text(ROOT / "artifacts" / "bootstrap_params.csv"),
        "prior_bins_csv": _read_text(ROOT / "Data" / "paco2_public_prior.csv"),
        "n_param_draws": 50,
        "seed": 123,
    }

    result = compute_ui_payload(payload)

    assert result["subgroup"] == subgroup
    assert result["mode"] == mode
    assert result["paco2_q_low"] < result["paco2_q_high"]
    assert 0.0 <= result["p_ge_threshold"] <= 1.0
    assert isinstance(result["paco2_bin"], list)
    assert isinstance(result["posterior_prob"], list)
    if mode == "prior_weighted":
        assert isinstance(result["likelihood_prob"], list)
        assert len(result["likelihood_prob"]) == len(result["paco2_bin"])
    else:
        assert result["likelihood_prob"] is None


def test_browser_contract_accepts_custom_prior_bins() -> None:
    prior_csv = "\n".join(
        ["group,paco2_bin,count,weight"]
        + [f"{group},40,1,0.5\n{group},60,1,0.5" for group in PACO2_PRIOR_GROUPS]
    )
    payload = {
        "tcco2": 50.0,
        "subgroup": "pft",
        "threshold": 45.0,
        "mode": "prior_weighted",
        "params_csv": _read_text(ROOT / "artifacts" / "bootstrap_params.csv"),
        "prior_bins_csv": prior_csv,
        "n_param_draws": 25,
        "seed": 1,
    }

    result = compute_ui_payload(payload)

    assert result["metadata"]["prior_source"] == "provided_bins"
    assert 0.0 <= result["p_ge_threshold"] <= 1.0


def test_browser_contract_accepts_weight_only_prior_bins() -> None:
    prior_csv = "\n".join(
        ["group,paco2_bin,weight"]
        + [f"{group},40,0.25\n{group},60,0.75" for group in PACO2_PRIOR_GROUPS]
    )
    payload = {
        "tcco2": 50.0,
        "subgroup": "pft",
        "threshold": 45.0,
        "mode": "prior_weighted",
        "params_csv": _read_text(ROOT / "artifacts" / "bootstrap_params.csv"),
        "prior_bins_csv": prior_csv,
        "n_param_draws": 25,
        "seed": 1,
    }

    result = compute_ui_payload(payload)

    assert result["metadata"]["prior_source"] == "provided_bins"
    assert 0.0 <= result["p_ge_threshold"] <= 1.0


def test_browser_contract_recomputes_from_uploaded_study_table() -> None:
    studies = pd.read_csv(ROOT / "Data" / "conway_studies.csv")
    studies.loc[studies.index[0], "bias"] = float(studies.loc[studies.index[0], "bias"]) + 0.25
    payload = {
        "subgroup": "pft",
        "study_csv": studies.to_csv(index=False),
        "n_boot": 25,
        "seed": 123,
        "bootstrap_mode": "cluster_plus_withinstudy",
    }

    bootstrap = build_bootstrap_payload(payload)

    assert bootstrap["subgroup"] == "pft"
    assert bootstrap["n_rows"] == 25
    assert bootstrap["params"]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")
