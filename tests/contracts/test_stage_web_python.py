from __future__ import annotations

import json
from pathlib import Path

from scripts.stage_web_python import stage_web_python

ROOT = Path(__file__).resolve().parents[2]


def test_stage_web_python_enforces_public_data_allowlist_and_clears_stale_priors(
    tmp_path: Path,
) -> None:
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

    assert set(manifest["data"]) == expected_data
    assert not (web_dir / "assets" / "data" / "paco2_public_prior.csv").exists()
    assert (web_dir / "assets" / "data" / "bootstrap_params.csv").read_bytes() == (
        ROOT / "artifacts" / "bootstrap_params.csv"
    ).read_bytes()
    assert not (web_dir / "assets" / "data" / "paco2_prior_bins.csv").exists()
