from __future__ import annotations

from pathlib import Path

from scripts.stage_web_python import stage_web_python

ROOT = Path(__file__).resolve().parents[1]


def test_stage_web_python_copies_package_and_assets(tmp_path: Path) -> None:
    web_dir = tmp_path / "web"

    manifest = stage_web_python(ROOT, web_dir=web_dir)

    assert "tcco2_accuracy/browser_contract.py" in manifest["files"]
    assert "tcco2_accuracy/ui_api.py" in manifest["files"]
    assert "assets/data/bootstrap_params.csv" in manifest["data"]
    assert (web_dir / "assets" / "py" / "manifest.json").exists()
    assert (web_dir / "assets" / "data" / "conway_studies.csv").exists()
    assert not list((web_dir / "assets" / "py").rglob("__pycache__"))
    assert not (web_dir / "assets" / "py" / "tcco2_accuracy" / "workflows").exists()
