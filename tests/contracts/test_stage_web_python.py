from __future__ import annotations

import json
from pathlib import Path

from scripts.stage_web_python import stage_web_python

ROOT = Path(__file__).resolve().parents[2]


def test_stage_web_python_copies_package_and_assets(tmp_path: Path) -> None:
    web_dir = tmp_path / "web"
    data_dir = web_dir / "assets" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("paco2_public_prior.csv", "paco2_prior_bins.csv"):
        (data_dir / filename).write_text("group,paco2_bin,count\nall,40,1\n")

    manifest = stage_web_python(ROOT, web_dir=web_dir)
    release_contract = json.loads((ROOT / "docs" / "data_release_contract.json").read_text())
    expected_data = {
        relative.removeprefix("web/") for relative in release_contract["pages_data_allowlist"]
    }

    assert "tcco2_accuracy/browser_contract.py" in manifest["files"]
    assert "tcco2_accuracy/ui_api.py" in manifest["files"]
    assert "tcco2_accuracy/core/paco2.py" in manifest["files"]
    assert "tcco2_accuracy/core/downstream.py" not in manifest["files"]
    assert "assets/data/bootstrap_params.csv" in manifest["data"]
    assert "assets/data/conway_studies.csv" in manifest["data"]
    assert "assets/data/paco2_public_prior.csv" not in manifest["data"]
    assert "assets/data/paco2_prior_bins.csv" not in manifest["data"]
    assert set(manifest["data"]) == expected_data
    assert (web_dir / "assets" / "py" / "manifest.json").exists()
    assert (web_dir / "assets" / "data" / "conway_studies.csv").exists()
    assert not (web_dir / "assets" / "data" / "paco2_public_prior.csv").exists()
    assert (web_dir / "assets" / "data" / "bootstrap_params.csv").read_bytes() == (
        ROOT / "artifacts" / "bootstrap_params.csv"
    ).read_bytes()
    assert not (web_dir / "assets" / "data" / "paco2_prior_bins.csv").exists()
    assert not list((web_dir / "assets" / "py").rglob("__pycache__"))
    assert not (web_dir / "assets" / "py" / "tcco2_accuracy" / "workflows").exists()
    assert not (web_dir / "assets" / "py" / "tcco2_accuracy" / "reporting").exists()
    assert not (web_dir / "assets" / "py" / "tcco2_accuracy" / "io.py").exists()


def test_browser_upload_surface_rejects_legacy_xls() -> None:
    index = (ROOT / "web" / "index.html").read_text()
    app = (ROOT / "web" / "assets" / "js" / "app.js").read_text()

    assert ".csv,.xlsx,text/csv" in index
    assert ".xls,text/csv" not in index
    assert 'name.endsWith(".xls")' not in app
