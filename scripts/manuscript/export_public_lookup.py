"""Export only investigator-approved prediction summaries from the verified private grid.

Run on the mini. Full support, masses, curves and audit certificates stay there.
The output remains private until its exact checksum is promoted in the release contract.
"""

import argparse
import json
import os
from pathlib import Path

from cache_private_grid import DRAWS, inputs, load
from private_inputs import digest, require_digest
from reporting_grid import build

from tcco2_accuracy.core._posterior import weighted_quantile
from tcco2_accuracy.pages_prediction import GROUPS
from tcco2_accuracy.workflows.conditional_lookup import validate_lookup


def export(run, dest, repo):
    # Recheck the archived certificates and every row against cached full-support likelihoods.
    build(run, dest, curves=True)
    plan = load(run / "plan.json")
    for name, expected in plan["commit"]["input_sha256"].items():
        require_digest(repo / name, expected)
    chart = load(dest / "private-chart.json")
    reference = load(repo / "models/pages/summary.json")
    doc = dict(
        version="conditional_lookup_v3",
        method=chart["method"],
        source="trinetx",
        prior_uncertainty="fixed_empirical",
        groups={},
    )
    fits = {}
    for setting, public_group in GROUPS.items():
        _, receipt, a, w, _ = inputs(run, setting)
        group = chart["groups"][setting]
        if group["posterior"] != receipt["posterior_sha256"]:
            raise ValueError("Private preview and accepted fit disagree")
        # Explicit allowlists: never copy a group or row wholesale from the private chart.
        doc["groups"][setting] = dict(
            posterior=group["posterior"],
            reference_posterior=reference[public_group]["posterior"],
            prior_interval=weighted_quantile(a, w, [0.025, 0.975]).tolist(),
            rows=[
                dict(interval=r["interval"], tails=r["tails"], report=r["report"], variance=None)
                for r in group["rows"]
            ],
        )
        fits[setting] = dict(
            posterior=group["posterior"],
            reference_posterior=reference[public_group]["posterior"],
            retained_per_chain=DRAWS[setting],
            chains=receipt["chains"],
            seed=receipt["seed"],
            adapt_delta=receipt["adapt_delta"],
            divergences=receipt["divergences"],
        )
    validate_lookup(doc)
    output = dest / "conditional-lookup.json"
    output.write_text(json.dumps(doc, allow_nan=False, separators=(",", ":")) + "\n")
    release = dict(
        lookup_sha256=digest(output),
        approval_date="2026-11-17",
        authorization="Investigator explicitly approved public summary lookup for all four settings.",
        index_policy=chart["index_policy"],
        observation_window="Original cohort, full available observation period",
        readings_mmhg=[20, 100],
        readings_per_setting=81,
        prior_uncertainty="fixed_empirical",
        reporting_amendment="docs/CONDITIONAL_REPORTING.md",
        verification="All 324 rows rechecked against the private cached likelihood and audited numerical brackets before export.",
        fits=fits,
        model_inputs=plan["commit"]["input_sha256"],
    )
    (dest / "conditional-release.json").write_text(json.dumps(release, indent=2) + "\n")
    print("Exported 324 summary rows; exact prior and curves remain private.")


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "output", "repository"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    export(args.run.resolve(), args.output.resolve(), args.repository.resolve())
