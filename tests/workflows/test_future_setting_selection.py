"""Exercise run scope without requiring CmdStan or generating scientific candidates."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from scripts import future_setting as workflow


def test_source_selection_and_resume_keep_the_bridge_and_membership_in_scope(monkeypatch):
    # Model every public row as held so the real reporter runs without any sampling.
    ledger = workflow.load_ledger().assign(status="held", decision="Test hold")
    monkeypatch.setattr(workflow, "load_ledger", lambda: ledger)
    monkeypatch.setattr(workflow, "offline_model", lambda: (None, {}))
    sources = (
        "models/future_setting/source_ledger.csv",
        "tests/fixtures/conway_table1.csv",
        "Data/conway_studies.csv",
    )
    monkeypatch.setattr(
        workflow,
        "evidence_revision",
        lambda: {"input_sha256": {p: workflow.digest(workflow.ROOT / p) for p in sources}},
    )
    scratch = workflow.ROOT / ".pytest_tmp/future-setting"
    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="selection-test-", dir=scratch) as tmp:
        out = Path(tmp)
        out.rmdir()  # Let the workflow create the run, as the public command does.
        run_id = out.name
        workflow.run("public", run_id, stratum="main")
        workflow.run("public", run_id, resume=True)
        manifest = json.loads((out / "manifest.json").read_text())
        assert list(manifest["strata"]) == ["main"]
        selected = workflow.select_units(ledger, "main")
        assert manifest["strata"]["main"]["membership"] == selected.analysis_unit_id.tolist()
        assert manifest["strata"]["main"]["status"] == "held_source"
        with (out / "interval_bridge.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5 and {r["stratum"] for r in rows} == {"main"}
        manifest_bytes = (out / "manifest.json").read_bytes()
        with pytest.raises(ValueError, match="strat"):
            workflow.run("public", run_id, resume=True, stratum="icu")
        assert (out / "manifest.json").read_bytes() == manifest_bytes

        # Prior manifests have no requested_stratum. They can resume all four groups,
        # including runs interrupted before every group had a status.
        manifest.pop("requested_stratum")
        workflow.write(out / "manifest.json", manifest)
        with pytest.raises(ValueError, match="strat"):
            workflow.run("public", run_id, resume=True, stratum="main")
        legacy = workflow.run("public", run_id, resume=True)
        assert tuple(legacy["strata"]) == workflow.GROUPS
        with (out / "interval_bridge.csv").open() as f:
            assert len(list(csv.DictReader(f))) == 5 * len(workflow.GROUPS)


def test_recovery_rejects_public_stratum_before_sampling():
    with pytest.raises(ValueError, match="public"):
        workflow.run("recovery", "test-invalid-recovery-selection", stratum="main")
