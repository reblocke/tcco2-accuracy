"""Build a conditional PaCO₂ lookup for a custodian-approved eligible extract."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from tcco2_accuracy.core._posterior import weighted_quantile
from tcco2_accuracy.core.downstream import PatientInputColumns
from tcco2_accuracy.core.future_setting import empirical_support
from tcco2_accuracy.pages_prediction import GROUPS
from tcco2_accuracy.workflows._private_output import require_private_output_path
from tcco2_accuracy.workflows.conditional_lookup import (
    READINGS,
    lookup_document,
    lookup_row,
    validate_lookup,
)
from tcco2_accuracy.workflows.future_setting_data import ROOT, digest
from tcco2_accuracy.workflows.future_setting_inverse import (
    EmpiricalPriorMetadata,
    index_support,
    load_candidate_draws,
)


def group_lookup(theta, order, a, w):
    a, w = empirical_support(a, w)
    return dict(
        prior_interval=weighted_quantile(a, w, [0.025, 0.975]).tolist(),
        rows=[lookup_row(theta, a, w, t, order=order) for t in READINGS],
    )


def private_build(receipt_path, runs):
    # The custodian supplies an approved, already eligibility/window-filtered extract.
    receipt = json.loads(receipt_path.read_text())
    meta = EmpiricalPriorMetadata(**receipt["revisions"])
    destination = require_private_output_path(receipt["private_destination"])
    if not destination.is_dir():
        raise ValueError("Choose an existing approved private directory")
    source = Path(receipt["eligible_extract"]).resolve(strict=True)
    with source.open("rb") as f:
        source_hash = hashlib.file_digest(f, "sha256").hexdigest()
    if (
        source_hash != receipt["eligible_extract_sha256"]
        or receipt["use"] != "approved_private_index_analysis"
    ):
        raise ValueError("Private eligibility receipt does not match this extract or use")
    columns = PatientInputColumns(**receipt.get("columns", {}))
    output_dir = destination / "conditional-lookup"
    if output_dir.exists():
        raise FileExistsError("Use a fresh approved private output directory")
    if source.suffix.lower() == ".dta":
        records = pd.read_stata(source, convert_categoricals=False)
    elif source.suffix.lower() == ".csv":
        records = pd.read_csv(
            source, dtype={columns.patient_id: "string", columns.encounter_id: "string"}
        )
    else:
        raise ValueError("Expected an eligible CSV or DTA extract")
    supports = {target: index_support(records, target, columns) for target in GROUPS}
    result = lookup_document("trinetx")
    fits = {}
    for target, group in GROUPS.items():
        theta, order, m = load_candidate_draws(
            runs / ("public-main-simon-2003" if target == "all" else "public-adjudicated-v2"),
            target,
        )
        fits[target] = {"run": m["run_id"], "posterior": m["strata"][group]["posterior_sha256"]}
        result["groups"][target] = dict(
            posterior=fits[target]["posterior"], **group_lookup(theta, order, *supports[target])
        )
    validate_lookup(result)
    # Publish the lookup and its receipt together. Concurrent successful writers
    # cannot replace a completed, nonempty directory.
    with TemporaryDirectory(dir=destination) as tmp:
        folder = Path(tmp)
        out = folder / "conditional-lookup.json"
        out.write_text(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
        audit = dict(
            revisions=asdict(meta),
            use="private_candidate",
            fits=fits,
            input_receipt_sha256=digest(receipt_path),
            lookup_sha256=digest(out),
            prior_uncertainty="fixed_empirical",
            index_policy="earliest eligible measurement in earliest eligible encounter per patient within target",
        )
        (folder / "receipt.json").write_text(json.dumps(audit, indent=2) + "\n")
        folder.rename(output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("approved_receipt", type=Path)
    parser.add_argument("--runs", type=Path, default=ROOT / ".pytest_tmp/future-setting")
    args = parser.parse_args()
    try:
        private_build(args.approved_receipt, args.runs)
    except (ValueError, KeyError, OSError, TypeError):
        # Parser errors may include private field values.
        parser.exit(
            1,
            "Private build failed. Check the eligibility receipt, schema, index keys and destination.\n",
        )


if __name__ == "__main__":
    main()
