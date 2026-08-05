from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from tcco2_accuracy.browser_contract import build_bootstrap_payload, compute_ui_payload
from tcco2_accuracy.core.conway_meta import AGREEMENT_METHOD_VERSION, RESULTS_STATUS
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
    assert browser["metadata"]["agreement_method_version"] == AGREEMENT_METHOD_VERSION
    assert browser["metadata"]["results_status"] == RESULTS_STATUS
    assert browser["metadata"]["requested_group"] == "all"
    assert browser["metadata"]["parameter_group_used"] == "main"


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
    assert result["metadata"]["agreement_method_version"] == AGREEMENT_METHOD_VERSION
    assert result["metadata"]["results_status"] == RESULTS_STATUS
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
    assert bootstrap["metadata"] == {
        "agreement_method_version": AGREEMENT_METHOD_VERSION,
        "results_status": RESULTS_STATUS,
        "requested_group": "pft",
        "parameter_group_used": "single_model",
    }
    assert {row["agreement_method_version"] for row in bootstrap["params"]} == {
        AGREEMENT_METHOD_VERSION
    }
    assert {row["results_status"] for row in bootstrap["params"]} == {RESULTS_STATUS}
    assert {row["requested_group"] for row in bootstrap["params"]} == {"pft"}
    assert {row["parameter_group_used"] for row in bootstrap["params"]} == {"single_model"}


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("agreement_method_version", "legacy_mixed_log_v0"),
        ("results_status", "final"),
    ],
)
def test_browser_contract_rejects_stale_parameter_provenance(column: str, replacement: str) -> None:
    params = pd.read_csv(ROOT / "artifacts" / "bootstrap_params.csv")
    params[column] = replacement

    with pytest.raises(ValueError, match=f"provenance `{column}`"):
        compute_ui_payload(_likelihood_payload(params))


@pytest.mark.parametrize("column", ["agreement_method_version", "results_status"])
def test_browser_contract_rejects_missing_parameter_provenance(column: str) -> None:
    params = pd.read_csv(ROOT / "artifacts" / "bootstrap_params.csv").drop(columns=column)

    with pytest.raises(ValueError, match="missing required provenance columns"):
        compute_ui_payload(_likelihood_payload(params))


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("agreement_method_version", "legacy_mixed_log_v0"),
        ("results_status", "final"),
    ],
)
def test_browser_contract_rejects_mixed_parameter_provenance(column: str, replacement: str) -> None:
    params = pd.read_csv(ROOT / "artifacts" / "bootstrap_params.csv")
    params.loc[params.index[0], column] = replacement

    with pytest.raises(ValueError, match=f"provenance `{column}`"):
        compute_ui_payload(_likelihood_payload(params))


def test_browser_contract_default_and_uploaded_paths_share_method_provenance() -> None:
    default_result = compute_ui_payload(
        {
            "tcco2": 50.0,
            "subgroup": "pft",
            "mode": "likelihood_only",
            "params_csv": _read_text(ROOT / "artifacts" / "bootstrap_params.csv"),
            "n_param_draws": 25,
            "seed": 123,
        }
    )
    uploaded_result = compute_ui_payload(
        {
            "tcco2": 50.0,
            "subgroup": "pft",
            "mode": "likelihood_only",
            "study_csv": _read_text(ROOT / "Data" / "conway_studies.csv"),
            "n_boot": 25,
            "n_param_draws": 25,
            "seed": 123,
            "bootstrap_mode": "cluster_plus_withinstudy",
        }
    )

    assert (
        default_result["metadata"]["agreement_method_version"]
        == uploaded_result["metadata"]["agreement_method_version"]
    )
    assert (
        default_result["metadata"]["results_status"]
        == uploaded_result["metadata"]["results_status"]
    )
    assert default_result["metadata"]["requested_group"] == "pft"
    assert default_result["metadata"]["parameter_group_used"] == "lft"
    assert uploaded_result["metadata"]["requested_group"] == "pft"
    assert uploaded_result["metadata"]["parameter_group_used"] == "single_model"


def test_browser_contract_rejects_missing_requested_parameter_group() -> None:
    params = pd.read_csv(ROOT / "artifacts" / "bootstrap_params.csv")
    params = params.loc[params["group"] == "main"]
    payload = _likelihood_payload(params)
    payload["subgroup"] = "pft"

    with pytest.raises(ValueError, match="No parameters found for requested subgroup 'pft'"):
        compute_ui_payload(payload)


def _likelihood_payload(params: pd.DataFrame) -> dict[str, object]:
    return {
        "tcco2": 50.0,
        "subgroup": "all",
        "mode": "likelihood_only",
        "params_csv": params.to_csv(index=False),
        "n_param_draws": 25,
        "seed": 1,
    }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")
