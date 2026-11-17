"""Stage Python package and static data assets for the GitHub Pages app."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np

from tcco2_accuracy.pages_prediction import GROUPS
from tcco2_accuracy.workflows.conditional_lookup import validate_lookup

DEFAULT_DATA_ASSETS = {
    name: Path("models/pages") / name
    for name in (
        "summary.json",
        "synthetic-lookup.json",
        "main.npz",
        "icu.npz",
        "arf.npz",
        "lft.npz",
    )
}

BROWSER_PYTHON_FILES = (
    "__init__.py",
    "bland_altman.py",
    "inference.py",
    "pages_prediction.py",
    "core/__init__.py",
    "core/_params.py",
    "core/_posterior.py",
    "core/bland_altman.py",
    "core/constants.py",
    "core/conditional_reporting.py",
    "core/future_setting.py",
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


def stage_web_python(repo_root: Path, web_dir: Path | None = None) -> dict:
    """Copy browser assets into web/ and return the staged manifest."""

    repo_root = repo_root.resolve()
    web_dir = (web_dir or repo_root / "web").resolve()
    package_src = repo_root / "src" / "tcco2_accuracy"
    package_dst = web_dir / "assets" / "py" / "tcco2_accuracy"
    data_dst = web_dir / "assets" / "data"

    if not package_src.exists():
        raise FileNotFoundError(f"Package source not found: {package_src}")

    if package_dst.parent.exists():
        shutil.rmtree(package_dst.parent)
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

    assets = DEFAULT_DATA_ASSETS.copy()
    assets["prediction-plots.json"] = Path("models/pages/prediction-plots.json")
    contract = json.loads((repo_root / "docs/data_release_contract.json").read_text())
    release_hash = contract["approved_conditional_lookup_sha256"]
    if release_hash is not None:
        assets["conditional-lookup.json"] = Path(contract["conditional_lookup_public_source"])
    if set(contract["pages_data_allowlist"]) != {f"web/assets/data/{name}" for name in assets}:
        raise ValueError("Lookup approval and Pages allowlist disagree")
    summary = json.loads((repo_root / assets["summary.json"]).read_text())
    verified_lookups = {}
    for filename, expected_source in (
        ("synthetic-lookup.json", "synthetic"),
        ("conditional-lookup.json", "trinetx"),
    ):
        if filename not in assets:
            continue
        content = (repo_root / assets[filename]).read_bytes()
        doc = json.loads(content)
        if expected_source == "trinetx" and (
            not isinstance(doc, dict)
            or doc.get("version") not in contract["conditional_lookup_public_versions"]
        ):
            raise ValueError("Pages requires an eligible summary lookup version for TriNetX")
        validate_lookup(doc)
        if doc["source"] != expected_source or any(
            doc["groups"][target][
                "reference_posterior" if doc["version"] == "conditional_lookup_v3" else "posterior"
            ]
            != summary[group]["posterior"]
            for target, group in GROUPS.items()
        ):
            raise ValueError("Lookup provenance or agreement posterior mismatch")
        expected = (
            release_hash
            if expected_source == "trinetx"
            else contract["synthetic_conditional_lookup_sha256"]
        )
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Lookup differs from the whole-file release approval")
        verified_lookups[filename] = content
    plot_bytes = (repo_root / assets["prediction-plots.json"]).read_bytes()
    if hashlib.sha256(plot_bytes).hexdigest() != contract["approved_plot_sha256"]:
        raise ValueError("Public plot checksum mismatch")
    plot = json.loads(plot_bytes)
    if plot["lookup_sha256"] != release_hash or plot["version"] != "trinetx_display_v1":
        raise ValueError("Public plot and summary revisions differ")
    for asset_name, source_rel in assets.items():
        source = repo_root / source_rel
        if not source.exists():
            raise FileNotFoundError(f"Required web data asset not found: {source}")
        if asset_name in verified_lookups:
            (data_dst / asset_name).write_bytes(verified_lookups[asset_name])
        else:
            shutil.copy2(source, data_dst / asset_name)
        staged_data.append(f"assets/data/{asset_name}")

    groups = {
        "main": "Overall",
        "icu": "Intensive care",
        "arf": "Acute respiratory failure",
        "lft": "Lung function testing",
    }
    columns = {
        ("published", "reference_limits"): "Conway limits",
        ("published", "endpoint_confidence_envelope"): "Conway envelope",
        ("corrected_legacy", "reference_limits"): "Corrected limits",
        ("corrected_legacy", "endpoint_confidence_envelope"): "Corrected envelope",
        ("hierarchical", "posterior_predictive_disagreement"): "Predictive interval",
    }
    rows = []
    for group, label in groups.items():
        fit = summary[group]
        if (
            hashlib.sha256((data_dst / f"{group}.npz").read_bytes()).hexdigest()
            != fit["kernel_sha256"]
        ):
            raise ValueError(f"Browser kernel checksum mismatch for {group}")
        pairs = {(r["method"], r["quantity_type"]): r for r in fit["intervals"]}
        if (
            set(pairs) != set(columns)
            or len(fit["intervals"]) != len(columns)
            or fit["method"] != "future_setting_joint_v1"
            or fit["status"] != "candidate"
            or any(row["unit"] != "mmHg" for row in pairs.values())
        ):
            raise ValueError(f"Invalid interval evidence for {group}")
        cells = []
        for key, heading in columns.items():
            pair = pairs[key]
            interval = f"{pair['lower']:+.1f} to {pair['upper']:+.1f}".replace("-", "−")
            cells.append(f'<td data-label="{heading}">{interval}</td>')
        rows.append(f'<tr><th scope="row">{label}</th>{"".join(cells)}</tr>')
    page = (repo_root / "web/research.html").read_text()
    page = re.sub(
        r"<tbody>.*?</tbody>", "<tbody>\n" + "\n".join(rows) + "\n</tbody>", page, flags=re.S
    )
    # Bind the initial table and calculator to the same revision, including on a cached page.
    display = {
        g: {
            k: fit[k]
            for k in (
                "quantiles",
                "method",
                "run",
                "status",
                "kernel_sha256",
                "posterior",
                "variance",
            )
        }
        for g, fit in summary.items()
    }
    # Plot knots come from the checksum-verified predictive density itself.
    for group in display:
        with np.load(data_dst / f"{group}.npz", allow_pickle=False) as kernel:
            display[group]["curve"] = {
                "x": kernel["x"].tolist(),
                "density": np.exp(kernel["logpdf"][0]).tolist(),
            }
    page = re.sub(
        r'(<script type="application/json" id="fit-summary">).*?(</script>)',
        lambda match: (
            match[1] + json.dumps(display, allow_nan=False).replace("<", r"\u003c") + match[2]
        ),
        page,
        flags=re.S,
    )
    hashes = {
        name: hashlib.sha256((data_dst / name).read_bytes()).hexdigest()
        for name in assets
        if name.endswith("-lookup.json")
    }
    page = re.sub(
        r'(<script type="application/json" id="lookup-hashes">).*?(</script>)',
        lambda m: m[1] + json.dumps(hashes) + m[2],
        page,
        flags=re.S,
    )

    py_files = sorted(
        str(path.relative_to(package_dst.parent)).replace("\\", "/")
        for path in package_dst.rglob("*.py")
    )
    bundle = package_dst.parent / "pages.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name in py_files:
            # Stable filenames and timestamps make the bundle reproducible in a fresh checkout.
            archive.writestr(
                zipfile.ZipInfo(name),
                (package_dst.parent / name).read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
            )
    manifest = {
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "python_root": "assets/py",
        "files": py_files,
        "data": sorted(staged_data),
    }
    worker = (repo_root / "web/pyodide_worker.js").read_bytes()
    worker_name = f"worker-{hashlib.sha256(worker).hexdigest()}.js"
    (package_dst.parent / worker_name).write_bytes(worker)
    worker_url = f"assets/py/{worker_name}?bundle={manifest['bundle_sha256']}"
    entrypoint = (repo_root / "web/assets/js/app.js").read_bytes()
    integrity = base64.b64encode(hashlib.sha256(entrypoint).digest()).decode("ascii")
    page, count = re.subn(
        r'<script id="upload-calculator"[^>]*></script>',
        f'<script id="upload-calculator" type="module" src="assets/js/app.js" '
        f'integrity="sha256-{integrity}" data-worker="{worker_url}"></script>',
        page,
    )
    if count != 1:
        raise ValueError("Expected exactly one upload calculator entrypoint")
    (web_dir / "research.html").write_text(page)
    primary = (repo_root / "web/index.html").read_text()
    hashes["prediction-plots.json"] = contract["approved_plot_sha256"]
    primary = re.sub(r'(?<=id="lookup-hashes">).*?(?=</script>)', json.dumps(hashes), primary)
    (web_dir / "index.html").write_text(primary)
    manifest_path = web_dir / "assets" / "py" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (web_dir / ".nojekyll").write_text("")
    return manifest


def main() -> None:
    args = parse_args()
    stage_web_python(repo_root=args.repo_root, web_dir=args.web_dir)


if __name__ == "__main__":
    main()
