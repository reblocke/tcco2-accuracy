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
        workflow.run("public", run_id, resume=True, sample_only=True)
        assert (out / "manifest.json").read_bytes() == manifest_bytes
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


def test_failed_preflight_leaves_run_id_available(tmp_path, monkeypatch):
    def missing_toolchain():
        raise ValueError("missing toolchain")

    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "offline_model", missing_toolchain)
    with pytest.raises(ValueError, match="toolchain"):
        workflow.run("public", "retry")
    assert not (tmp_path / ".pytest_tmp/future-setting/retry").exists()


def test_resume_verifies_repeated_measures_helper_against_original_commit(tmp_path, monkeypatch):
    """Use actual Git objects and the resume entry point without changing a candidate run."""
    import shutil
    import subprocess

    from tcco2_accuracy.workflows import future_setting_fit as fitter

    repo = tmp_path.resolve()
    helper = Path("src/tcco2_accuracy/core/conway_meta.py")
    inputs = [p.relative_to(fitter.ROOT) for p in fitter.FILES]
    for rel in set(inputs) | {helper}:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fitter.ROOT / rel, repo / rel)

    def git(*args):
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.org",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "Original fit inputs")
    monkeypatch.setattr(fitter, "ROOT", repo)
    monkeypatch.setattr(fitter, "FILES", [repo / rel for rel in inputs])
    monkeypatch.setattr(workflow, "ROOT", repo)
    monkeypatch.setattr(workflow, "offline_model", lambda: (None, {}))
    monkeypatch.setattr(workflow, "load_ledger", lambda: None)
    revision = fitter.evidence_revision()
    assert revision["input_sha256"][str(helper)] == fitter.digest(repo / helper)

    out = repo / ".pytest_tmp/future-setting/test-resume"
    out.mkdir(parents=True)
    path = out / "manifest.json"
    manifest = dict(**revision, run_id=out.name, mode="public", run_status="complete", backend={})
    legacy = {**manifest, "input_sha256": dict(manifest["input_sha256"])}
    del legacy["input_sha256"][str(helper)]

    def resume():
        return workflow.run("public", out.name, resume=True, sample_only=True)

    for original in [manifest, legacy]:
        workflow.write(path, original)
        before = path.read_bytes()
        assert resume() == original
        assert path.read_bytes() == before

    workflow.write(path, {**legacy, "git_commit": "0" * 40})
    with pytest.raises(ValueError, match="evidence"):
        resume()
    with (repo / helper).open("a") as f:
        f.write("\n# Different observation preparation revision.\n")
    git("commit", "-qam", "Change adjustment implementation")
    for original in [manifest, legacy]:
        workflow.write(path, original)
        before = path.read_bytes()
        with pytest.raises(ValueError, match="evidence"):
            resume()
        assert path.read_bytes() == before
