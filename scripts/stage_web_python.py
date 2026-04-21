"""Stage Python package and static data assets for the GitHub Pages app."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_DATA_ASSETS = {
    "conway_studies.csv": Path("Data/conway_studies.csv"),
    "paco2_public_prior.csv": Path("Data/paco2_public_prior.csv"),
    "bootstrap_params.csv": Path("artifacts/bootstrap_params.csv"),
}

BROWSER_PYTHON_FILES = (
    "__init__.py",
    "_files.py",
    "_params.py",
    "_posterior.py",
    "bland_altman.py",
    "bootstrap.py",
    "browser_contract.py",
    "conway_meta.py",
    "data.py",
    "inference.py",
    "simulation.py",
    "ui_api.py",
    "utils.py",
    "validate_inputs.py",
    "core/__init__.py",
    "core/_params.py",
    "core/_posterior.py",
    "core/bland_altman.py",
    "core/bootstrap.py",
    "core/constants.py",
    "core/conway_meta.py",
    "core/inference.py",
    "core/paco2.py",
    "core/simulation.py",
    "core/utils.py",
    "core/validate_inputs.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--web-dir", type=Path, default=None)
    return parser.parse_args()


def stage_web_python(repo_root: Path, web_dir: Path | None = None) -> dict[str, list[str]]:
    """Copy browser assets into web/ and return the staged manifest."""

    repo_root = repo_root.resolve()
    web_dir = (web_dir or repo_root / "web").resolve()
    package_src = repo_root / "src" / "tcco2_accuracy"
    package_dst = web_dir / "assets" / "py" / "tcco2_accuracy"
    data_dst = web_dir / "assets" / "data"

    if not package_src.exists():
        raise FileNotFoundError(f"Package source not found: {package_src}")

    if package_dst.exists():
        shutil.rmtree(package_dst)
    package_dst.parent.mkdir(parents=True, exist_ok=True)
    package_dst.mkdir(parents=True, exist_ok=True)
    for relative_name in BROWSER_PYTHON_FILES:
        source = package_src / relative_name
        if not source.exists():
            raise FileNotFoundError(f"Required browser Python file not found: {source}")
        destination = package_dst / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    if data_dst.exists():
        shutil.rmtree(data_dst)
    data_dst.mkdir(parents=True, exist_ok=True)
    staged_data: list[str] = []
    for asset_name, source_rel in DEFAULT_DATA_ASSETS.items():
        source = repo_root / source_rel
        if not source.exists():
            raise FileNotFoundError(f"Required web data asset not found: {source}")
        shutil.copy2(source, data_dst / asset_name)
        staged_data.append(f"assets/data/{asset_name}")

    py_files = sorted(
        str(path.relative_to(package_dst.parent)).replace("\\", "/")
        for path in package_dst.rglob("*.py")
    )
    manifest = {
        "python_root": "assets/py",
        "files": py_files,
        "data": sorted(staged_data),
    }
    manifest_path = web_dir / "assets" / "py" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (web_dir / ".nojekyll").write_text("")
    return manifest


def main() -> None:
    args = parse_args()
    stage_web_python(repo_root=args.repo_root, web_dir=args.web_dir)


if __name__ == "__main__":
    main()
