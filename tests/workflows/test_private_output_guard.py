from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

import scripts.build_paco2_prior_bins as prior_script
from tcco2_accuracy.reporting import manuscript as manuscript_reporting
from tcco2_accuracy.workflows import conditional, infer, paco2, sim
from tcco2_accuracy.workflows._private_output import (
    REPOSITORY_ROOT,
    require_private_output_path,
)


@pytest.mark.parametrize("scratch_name", [".tmp", ".pytest_tmp"])
def test_private_output_guard_accepts_repository_scratch_roots(scratch_name: str) -> None:
    path = REPOSITORY_ROOT / scratch_name / "private" / "output.csv"

    assert require_private_output_path(path) == path.resolve()


def test_private_output_guard_accepts_explicit_external_path(tmp_path: Path) -> None:
    path = tmp_path / "private" / "output.csv"

    assert require_private_output_path(path) == path.resolve()


@pytest.mark.parametrize(
    "path",
    [
        REPOSITORY_ROOT,
        REPOSITORY_ROOT / "artifacts" / "restricted.csv",
        REPOSITORY_ROOT / ".tmp-output" / "restricted.csv",
    ],
)
def test_private_output_guard_rejects_other_repository_paths(path: Path) -> None:
    with pytest.raises(ValueError, match="Restricted-data-derived outputs"):
        require_private_output_path(path)


@pytest.mark.parametrize(
    ("module", "workflow"),
    [
        (paco2, paco2.run_paco2_summary),
        (sim, sim.run_forward_simulation_summary),
        (infer, infer.run_inference_demo),
        (conditional, conditional.run_conditional_classification),
        (manuscript_reporting, manuscript_reporting.run_manuscript_outputs),
    ],
)
def test_restricted_workflows_reject_tracked_output_before_loading_data(
    module: object,
    workflow: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    denied_load = Mock(side_effect=AssertionError("restricted input was read"))
    monkeypatch.setattr(module, "load_paco2_distribution", denied_load)

    with pytest.raises(ValueError, match="Restricted-data-derived outputs"):
        workflow(out_dir=REPOSITORY_ROOT / "artifacts")

    denied_load.assert_not_called()


@pytest.mark.parametrize(
    "workflow",
    [
        paco2.run_paco2_summary,
        sim.run_forward_simulation_summary,
        infer.run_inference_demo,
        conditional.run_conditional_classification,
        manuscript_reporting.run_manuscript_outputs,
    ],
)
def test_restricted_workflows_require_an_explicit_input_source(workflow: object) -> None:
    with pytest.raises(ValueError, match="paco2_data or an explicit private paco2_path"):
        workflow()


def test_agreement_only_manuscript_writer_is_exempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_writer = Mock()
    text_writer = Mock()
    monkeypatch.setattr(manuscript_reporting, "_write_csv", csv_writer)
    monkeypatch.setattr(manuscript_reporting, "write_text", text_writer)
    params = pd.DataFrame({"group": ["main"], "delta": [0.0], "sigma2": [4.0], "tau2": [1.0]})

    manuscript_reporting.run_manuscript_parameters(
        params=params,
        out_dir=REPOSITORY_ROOT / "artifacts",
    )

    assert csv_writer.call_count == 1
    assert text_writer.call_count == 1


def test_prior_build_script_requires_explicit_input_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["build_paco2_prior_bins.py"])

    with pytest.raises(SystemExit):
        prior_script.parse_args()


@pytest.mark.parametrize(
    ("output", "xlsx"),
    [
        (REPOSITORY_ROOT / "artifacts" / "prior.csv", None),
        (
            REPOSITORY_ROOT / ".tmp" / "prior.csv",
            REPOSITORY_ROOT / "artifacts" / "prior.xlsx",
        ),
    ],
)
def test_prior_build_script_guards_all_outputs_before_loading_input(
    output: Path,
    xlsx: Path | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        input=REPOSITORY_ROOT / "restricted.dta",
        output=output,
        xlsx=xlsx,
        bin_width=1.0,
        include_counts=False,
    )
    denied_load = Mock(side_effect=AssertionError("restricted input was read"))
    monkeypatch.setattr(prior_script, "parse_args", lambda: args)
    monkeypatch.setattr(prior_script, "load_paco2_distribution", denied_load)

    with pytest.raises(ValueError, match="Restricted-data-derived outputs"):
        prior_script.main()

    denied_load.assert_not_called()


def test_prior_build_script_writes_normalized_weights_to_external_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "private" / "prior.csv"
    args = argparse.Namespace(
        input=REPOSITORY_ROOT / "restricted.dta",
        output=output,
        xlsx=None,
        bin_width=1.0,
        include_counts=False,
    )
    data = pd.DataFrame(
        {
            "paco2": [35.0, 45.0, 55.0],
            "subgroup": ["pft", "ed_inp", "icu"],
        }
    )
    monkeypatch.setattr(prior_script, "parse_args", lambda: args)
    monkeypatch.setattr(prior_script, "load_paco2_distribution", lambda _path: data)

    prior_script.main()

    result = pd.read_csv(output)
    assert result.columns.tolist() == ["group", "paco2_bin", "weight"]
    assert result.groupby("group")["weight"].sum().to_dict() == {
        "all": 1.0,
        "ed_inp": 1.0,
        "icu": 1.0,
        "pft": 1.0,
    }
