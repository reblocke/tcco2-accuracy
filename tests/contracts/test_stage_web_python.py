from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

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
    assert (web_dir / "assets" / "data" / "main.npz").read_bytes() == (
        ROOT / "models/pages" / "main.npz"
    ).read_bytes()
    assert not (web_dir / "assets" / "data" / "paco2_prior_bins.csv").exists()


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (repo / "docs").mkdir()
    (repo / "web").mkdir()
    shutil.copy(ROOT / "web/index.html", repo / "web")
    shutil.copy(ROOT / "docs/data_release_contract.json", repo / "docs")
    shutil.copytree(ROOT / "models/pages", repo / "models/pages")
    return repo


def test_stage_rejects_substituted_lookup_even_with_valid_schema(checkout):
    repo = checkout
    file = repo / "models/pages/synthetic-lookup.json"
    doc = json.loads(file.read_text())
    doc["groups"]["all"]["rows"][25]["interval"] = None
    file.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="whole-file"):
        stage_web_python(repo)
    assert not (repo / "web/assets/data/synthetic-lookup.json").exists()


def test_pinned_private_chart_at_summary_path_never_reaches_pages(checkout):
    contract_file = checkout / "docs/data_release_contract.json"
    contract = json.loads(contract_file.read_text())
    private = json.dumps(
        {
            "version": "conditional_lookup_private_chart_v1",
            "source": "trinetx",
            "target": "all",
            "support": [42.0, 47.0],
            "prior": [0.25, 0.75],
        }
    ).encode()
    (checkout / contract["conditional_lookup_public_source"]).write_bytes(private)
    contract["approved_conditional_lookup_sha256"] = hashlib.sha256(private).hexdigest()
    contract["pages_data_allowlist"].append("web/assets/data/conditional-lookup.json")
    contract_file.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="summary lookup version"):
        stage_web_python(checkout)
    assert not (checkout / "web/assets/data/conditional-lookup.json").exists()
