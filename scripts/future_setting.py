"""Run the offline TCCO2-008 public or synthetic workflow from the repository."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone

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
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def run(mode, run_id, resume=False, sample_only=False):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise ValueError("Invalid run identifier")
    base = (ROOT / ".pytest_tmp/future-setting").resolve()
    out = base / run_id
    if base != ROOT.resolve() / ".pytest_tmp/future-setting" or out.resolve() != out:
        raise ValueError("Run directory must resolve under repository scratch")
    out.mkdir(parents=True, exist_ok=resume)
    model, backend = offline_model()
    revision = evidence_revision()
    ledger = load_ledger()
    priors = pd.read_csv(ROOT / "tests/fixtures/synthetic_paco2_prior.csv")
    manifest_path = out / "manifest.json"
    if resume:
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["mode"] != mode
            or manifest["input_sha256"] != revision["input_sha256"]
            or manifest["backend"] != backend
            or manifest["run_id"] != run_id
        ):
            raise ValueError("Cannot resume a different evidence revision, backend, or mode")
    else:
        manifest = {
            **revision,
            "mode": mode,
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
        write(manifest_path, manifest)
    groups = GROUPS if mode == "public" else ("synthetic",)
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
        for retained in (4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000):
            attempt = groupdir / f"draws-{retained}"
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
        write(manifest_path, manifest)
    print(out, flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["public", "recovery"])
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Stop after writing sampler receipts, resume to evaluate precision and report",
    )
    args = parser.parse_args()
    if args.resume and not args.run_id:
        parser.error("--resume requires --run-id")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = run(args.mode, args.run_id or f"{args.mode}-{stamp}", args.resume, args.sample_only)
    failed = any(
        s["status"] in {"held_precision", "held_sampler", "failed_recovery"}
        for s in result["strata"].values()
    )
    raise SystemExit(int(failed))
