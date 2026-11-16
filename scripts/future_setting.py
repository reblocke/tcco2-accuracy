"""Run the offline TCCO2-008 public or synthetic workflow from the repository."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from tcco2_accuracy.core.future_setting import PARAMETERS
from tcco2_accuracy.workflows.future_setting_data import (
    GROUPS,
    ROOT,
    SPEC,
    digest,
    load_ledger,
    prepare_summaries,
    recovery_data,
    select_units,
)
from tcco2_accuracy.workflows.future_setting_fit import (
    evidence_revision,
    offline_model,
    sample_attempt,
)


def write(path, obj):
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as tmp:
        json.dump(obj, tmp, indent=2, allow_nan=False)
        tmp.write("\n")
    Path(tmp.name).replace(path)


def finalize_run(out, manifest):
    """Bind the accepted posteriors and completed candidate tables to this run."""
    from importlib.metadata import version

    for group, info in manifest["strata"].items():
        if info["status"] != "candidate":
            continue
        attempt = out / group / info["attempt"]
        sampler = json.loads((attempt / "sampler.json").read_text())
        precision = json.loads((attempt / "precision.json").read_text())
        if not (sampler["sampler_pass"] and precision["passed"]):
            raise ValueError("Candidate failed its recorded diagnostics")
        info["posterior_sha256"] = digest(attempt / "posterior.npz")
        if info["posterior_sha256"] != sampler["posterior_sha256"]:
            raise ValueError("Posterior does not match sampler receipt")
        info["precision_sha256"] = digest(attempt / "precision.json")
    manifest["output_sha256"] = {
        path.name: digest(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    report_files = (
        "scripts/future_setting.py",
        "scripts/future_setting_validation.py",
        "src/tcco2_accuracy/core/future_setting.py",
        "src/tcco2_accuracy/reporting/future_setting.py",
        "src/tcco2_accuracy/workflows/future_setting_fit.py",
        "src/tcco2_accuracy/workflows/future_setting_inverse.py",
    )
    manifest["reporting_git_commit"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if manifest["mode"] == "public":
        manifest["population_applicability"] = {
            "all": "Mixed-age overall source, including pediatric and neonatal studies.",
            "icu": "Apply to ICU index measurements if the literature ICU settings are comparable.",
            "ed_inp": "ED/inpatient index measurements use the acute respiratory failure source class; clinical case mix can differ.",
            "pft": "PFT index measurements use the LFT source class, which includes respiratory inpatients.",
        }
    manifest["workflow_sha256"] = {name: digest(ROOT / name) for name in report_files}
    manifest["reporting_software"] = {k: version(k) for k in ("numpy", "scipy", "pandas")}
    manifest["run_status"] = "complete"


def run(mode, run_id, resume=False, sample_only=False, *, stratum=None):
    if stratum is not None and (mode != "public" or stratum not in GROUPS):
        raise ValueError("A selected stratum must be a prespecified public source stratum")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError("Invalid run identifier")
    base = (ROOT / ".pytest_tmp/future-setting").resolve()
    out = base / run_id
    if base != ROOT.resolve() / ".pytest_tmp/future-setting" or out.resolve() != out:
        raise ValueError("Run directory must resolve under repository scratch")
    model, backend = offline_model()
    revision = evidence_revision()
    ledger = load_ledger()
    priors = pd.read_csv(ROOT / "tests/fixtures/synthetic_paco2_prior.csv")
    manifest_path = out / "manifest.json"
    if resume:
        manifest = json.loads(manifest_path.read_text())
        if stratum is not None and stratum != manifest.get("requested_stratum"):
            raise ValueError("Cannot change the selected stratum on resume")
        stratum = manifest.get("requested_stratum")
        if (
            manifest["mode"] != mode
            or manifest["input_sha256"] != revision["input_sha256"]
            or manifest["backend"] != backend
            or manifest["run_id"] != run_id
        ):
            raise ValueError("Cannot resume a different evidence revision, backend, or mode")
        if sample_only and manifest.get("run_status") == "complete":
            return manifest
    else:
        out.mkdir(parents=True, exist_ok=False)
        manifest = {
            **revision,
            "mode": mode,
            "requested_stratum": stratum,
            "run_id": run_id,
            "backend": backend,
            "method_version": SPEC["method_version"],
            "observation_version": SPEC["observation_version"],
            "prior_version": SPEC["prior_version"],
            "prior_uncertainty": "fixed_empirical",
            "inverse_prior_revision": "synthetic_fixture_v1",
            "private_application": "pending approved provenance and private destination",
            "human_review": "pending",
            "release": "candidate only",
            "strata": {},
        }
    manifest["run_status"] = "running"
    write(manifest_path, manifest)
    if mode == "recovery":
        from future_setting_validation import check_binary

        write(out / "scientific_checks.json", check_binary(model))
    groups = GROUPS if mode == "public" else ("synthetic",)
    if stratum is not None:
        groups = (stratum,)
    summaries = {}
    for g in groups:
        info = manifest["strata"].setdefault(g, {})
        if mode == "public":
            rows = select_units(ledger, g)
            info["membership"] = rows.analysis_unit_id.tolist()
            held = rows[rows.status != "cleared"]
            if len(held):
                info.update(
                    status="held_source", reasons=held[["study_id", "decision"]].to_dict("records")
                )
                write(manifest_path, manifest)
                continue
            blocks = [b for b in SPEC["observation"]["dependent_blocks"] if g in b["strata"]]
            data = prepare_summaries(rows, blocks)
        else:
            data = recovery_data()
            info["generator_seed"] = 88031
        groupdir = out / g
        groupdir.mkdir(exist_ok=True)
        last_accepted = (
            info.get("attempt") if resume and info.get("status") == "candidate" else None
        )
        for retained in (4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000):
            attempt = groupdir / f"draws-{retained}"
            if last_accepted and retained < int(last_accepted.removeprefix("draws-")):
                continue
            if attempt.is_symlink() or groupdir.is_symlink():
                raise ValueError("Invalid sample destination")
            if resume and (attempt / "sampler.json").is_file():
                receipt = json.loads((attempt / "sampler.json").read_text())
                if receipt["posterior_sha256"] != digest(attempt / "posterior.npz"):
                    raise ValueError("Posterior checksum mismatch")
                with np.load(attempt / "posterior.npz") as saved:
                    if (
                        list(saved["parameters"]) != list(PARAMETERS)
                        or saved["method_version"].item() != SPEC["method_version"]
                        or not np.array_equal(saved["chain_ids"], [1, 2, 3, 4])
                        or not np.array_equal(saved["draw_ids"], np.arange(1, retained + 1))
                    ):
                        raise ValueError("Wrong posterior method or indexing")
                    theta = saved["theta"]
                if theta.shape != (4, retained, 5):
                    raise ValueError("Incomplete posterior")
            else:
                if resume and attempt.exists():
                    incomplete = (
                        groupdir
                        / f"{attempt.name}-incomplete-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
                    )
                    attempt.rename(incomplete)
                theta, receipt = sample_attempt(model, data, attempt, retained)
            info.update(
                status="sampled", attempt=attempt.name, sampler_pass=receipt["sampler_pass"]
            )
            write(manifest_path, manifest)
            if not receipt["sampler_pass"]:
                info.update(status="held_sampler")
                break
            if sample_only:
                break
            from tcco2_accuracy.reporting.future_setting import assess

            target = next((k for k, v in SPEC["target_mapping"].items() if v == g), "all")
            prior = priors.loc[priors.group == target]
            result, accuracy = assess(theta, data, prior, g, mode)
            write(attempt / "precision.json", accuracy)
            if accuracy["passed"]:
                summaries[g] = result
                info.update(status="candidate", target_group=target)
                break
            info.update(status="held_precision")
        write(manifest_path, manifest)
    if not sample_only:
        from tcco2_accuracy.reporting.future_setting import finish_run

        finish_run(out, manifest, summaries)
        finalize_run(out, manifest)
    else:
        manifest["run_status"] = "sampled"
    write(manifest_path, manifest)
    print(out, flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["public", "recovery"])
    parser.add_argument("--run-id")
    parser.add_argument("--stratum", choices=GROUPS, help="Fit a single public source stratum")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Stop after writing sampler receipts, resume to evaluate precision and report",
    )
    args = parser.parse_args()
    if args.resume and not args.run_id:
        parser.error("--resume requires --run-id")
    if args.stratum and args.mode != "public":
        parser.error("--stratum requires public mode")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = run(
        args.mode,
        args.run_id or f"{args.mode}-{stamp}",
        args.resume,
        args.sample_only,
        stratum=args.stratum,
    )
    failed = any(
        s["status"] in {"held_precision", "held_sampler", "failed_recovery"}
        for s in result["strata"].values()
    )
    raise SystemExit(int(failed))
