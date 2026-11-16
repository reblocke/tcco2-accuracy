"""Build a conditional PaCO₂ lookup for a custodian-approved eligible extract."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

import numpy as np
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
from tcco2_accuracy.workflows.private_chart import private_chart

INDEX_POLICY = "earliest_eligible_measurement_in_earliest_eligible_encounter_per_patient"
FIRST_DAY_POLICY = "equal_patient_weight_split_across_eligible_draws_on_first_testing_day"


def group_lookup(theta, order, a, w):
    a, w = empirical_support(a, w)
    return dict(
        prior_interval=weighted_quantile(a, w, [0.025, 0.975]).tolist(),
        rows=[lookup_row(theta, a, w, t, order=order) for t in READINGS],
    )


def _frequency_support(frame):
    """Custodian-selected index frequencies at original measurement precision."""
    if (
        set(frame) not in ({"paco2_bin", "count"}, {"paco2_bin", "weight"})
        or frame.empty
        or not isinstance(frame.index, pd.RangeIndex)  # pandas can infer an undeclared ID column
        or any(pd.api.types.is_bool_dtype(v) for v in frame.dtypes)
    ):
        raise ValueError("Expected index frequency columns")
    a = frame["paco2_bin"].to_numpy(float)
    field = "count" if "count" in frame else "weight"
    w = frame[field].to_numpy(float)
    if (
        not np.isfinite(w).all()
        or not (w > 0).all()
        or len(np.unique(a)) != len(a)
        or (field == "count" and ((w > 2**53 - 1).any() or not (w == np.floor(w)).all()))
        or (field == "weight" and not np.isclose(w.sum(), 1, atol=1e-9, rtol=0))
    ):
        raise ValueError("Invalid index frequency table")
    return empirical_support(a, w)


def private_build(receipt_path, runs, *, chart_target=None):
    # The custodian supplies an approved, already eligibility/window-filtered extract.
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    meta = EmpiricalPriorMetadata(**receipt["revisions"])
    destination = require_private_output_path(receipt["private_destination"])
    if not destination.is_dir():
        raise ValueError("Choose an existing approved private directory")
    source = Path(receipt["eligible_extract"]).resolve(strict=True)
    if receipt["use"] != "approved_private_index_analysis":
        raise ValueError("Private eligibility receipt does not authorize this use")
    if chart_target is not None and chart_target not in GROUPS:
        raise ValueError("Unknown private chart target")
    frequency = receipt.get("input_kind") == "index_frequencies"
    first_day = receipt.get("index_policy") == FIRST_DAY_POLICY
    if first_day and not frequency:
        raise ValueError("First-testing-day analysis requires patient-weighted frequencies")
    if frequency:
        if (
            chart_target is None
            or receipt.get("index_target") != chart_target
            or receipt.get("index_policy") not in (INDEX_POLICY, FIRST_DAY_POLICY)
            or source.suffix.lower() != ".csv"
            or "columns" in receipt
        ):
            raise ValueError(
                "Index frequencies require a matching target and verified index policy"
            )
    elif "input_kind" in receipt:
        raise ValueError("Unknown private input kind")
    elif receipt.get("index_policy", INDEX_POLICY) != INDEX_POLICY:
        raise ValueError("Eligible records require the earliest-per-patient index policy")
    columns = PatientInputColumns(**receipt.get("columns", {}))
    output_dir = destination / (
        "conditional-lookup" if chart_target is None else f"private-chart-{chart_target}"
    )
    if output_dir.exists():
        raise FileExistsError("Use a fresh approved private output directory")
    if source.suffix.lower() not in {".dta", ".csv"}:
        raise ValueError("Expected an eligible CSV or DTA extract")
    # Snapshot the approved bytes in private scratch. The source path can be refreshed
    # while pandas parses a large export. The temporary copy closes before fitting.
    with TemporaryFile(dir=destination) as snapshot, source.open("rb") as extract:
        source_hash = hashlib.sha256()
        while chunk := extract.read(1024 * 1024):
            source_hash.update(chunk)
            snapshot.write(chunk)
        if source_hash.hexdigest() != receipt["eligible_extract_sha256"]:
            raise ValueError("Private eligibility receipt does not match this extract")
        snapshot.seek(0)
        if frequency:
            records = pd.read_csv(snapshot, float_precision="round_trip")
            if first_day and "weight" not in records:
                raise ValueError(
                    "First-testing-day frequencies must contain normalized patient weights"
                )
        elif source.suffix.lower() == ".dta":
            records = pd.read_stata(snapshot, convert_categoricals=False)
        else:
            records = pd.read_csv(
                snapshot,
                dtype={columns.patient_id: "string", columns.encounter_id: "string"},
                float_precision="round_trip" if chart_target is not None else None,
            )
    targets = GROUPS if chart_target is None else (chart_target,)
    supports = {
        target: _frequency_support(records)
        if frequency
        else index_support(records, target, columns)
        for target in targets
    }
    del records
    result = lookup_document("trinetx")
    fits = {}
    for target in targets:
        theta, order, m = load_candidate_draws(
            runs / ("public-main-simon-2003" if target == "all" else "public-adjudicated-v2"),
            target,
        )
        posterior = m["strata"][GROUPS[target]]["posterior_sha256"]
        fits[target] = {"run": m["run_id"], "posterior": posterior}
        if chart_target is None:
            result["groups"][target] = dict(
                posterior=posterior, **group_lookup(theta, order, *supports[target])
            )
        else:
            result = private_chart(
                theta, *supports[target], target=target, posterior=posterior, order=order
            )
            if first_day:
                result["index_policy"] = FIRST_DAY_POLICY
    if chart_target is None:
        validate_lookup(result)
    # Publish the lookup and its receipt together. Concurrent successful writers
    # cannot replace a completed, nonempty directory.
    with TemporaryDirectory(dir=destination) as tmp:
        folder = Path(tmp)
        out = folder / ("conditional-lookup.json" if chart_target is None else "private-chart.json")
        out.write_text(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
        audit = dict(
            revisions=asdict(meta),
            use="private_candidate",
            fits=fits,
            input_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
            prior_uncertainty="fixed_empirical",
            index_policy=(
                FIRST_DAY_POLICY
                if first_day
                else "earliest eligible measurement in earliest eligible encounter per patient within target"
            ),
        )
        sha = digest(out)
        if chart_target is None:
            audit["lookup_sha256"] = sha
        else:
            from tcco2_accuracy.workflows.private_preview_html import private_viewer_html

            audit.update(
                chart_sha256=sha,
                input_kind="index_frequencies" if frequency else "eligible_records",
            )
            (folder / "preview.html").write_text(private_viewer_html(sha))
        (folder / "receipt.json").write_text(json.dumps(audit, indent=2) + "\n")
        folder.rename(output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("approved_receipt", type=Path)
    parser.add_argument("--runs", type=Path, default=ROOT / ".pytest_tmp/future-setting")
    parser.add_argument(
        "--private-chart",
        choices=tuple(GROUPS),
        help="Write a local chart at TcCO₂ 47 for one setting",
    )
    args = parser.parse_args()
    try:
        private_build(args.approved_receipt, args.runs, chart_target=args.private_chart)
    except (ValueError, KeyError, OSError, TypeError):
        # Parser errors may include private field values.
        parser.exit(
            1,
            "Private build failed. Check the eligibility receipt, schema, index keys and destination.\n",
        )


if __name__ == "__main__":
    main()
