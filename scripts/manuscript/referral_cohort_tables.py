"""Write reviewable aggregate flow and selection-order comparisons on the controlled host."""

import argparse
import json
import os
from pathlib import Path

from private_inputs import private_destination
from referral_exhibits import table
from referral_report import GROUPS

SOURCE = [
    "The November 2023 request specified an initial qualifying 2022 encounter for each of six indications within each of three settings. The historical notebooks selected the first encounter for each indication and then joined the setting file. The maintained implementation selects within setting first and also changes gas eligibility. We replayed both orders on one recovered candidate snapshot with the historical gas rules. Both replays applied the historical demographic and regional exclusions. We used the archived patient–encounter membership for the primary predictions. No contemporaneous execution receipt binds the recovered snapshot and notebook to the archived May 2025 table.",
    "The historical arterial gas path accepted ordinary arterial PCO₂, temperature-adjusted arterial PCO₂ and total arterial CO₂ (2026-3). Its venous gas path accepted specimen-unspecified PCO₂ (11557-6). Numeric gas values had to exceed 5 and be below 200. The maintained gas rule uses values above 45 and below 200 after conversion. The legacy encounter-level Stata PaCO₂ averaged ordinary arterial, temperature-adjusted arterial and specimen-unspecified records. These choices can change which encounters enter before arterial outcome selection.",
    "The present arterial distribution uses source-recorded LOINC 2019-8 or 32771-8, a convertible pressure unit, and finite PaCO₂ above 2 and below 250 mmHg. We kept all eligible arterial result records on the first testing calendar day and divided each patient’s unit weight equally across those records. All and the three settings use independent first days. One patient can contribute to more than one setting. Records may share a specimen. The available dates do not establish an intraday order or pretreatment status.",
]
SOURCES = (
    "[Original cohort specification](https://github.com/reblocke/trinetx-hypercapnia-code/blob/d73c0559f480da16db0a5b46258a507ce4b75338/archive/legacy-projects/code-database-documentation/2023-11-27%20TriNetX%20Hypercapnia%20Model%20Data%20Specification.docx)",
    "[Historical encounter selection](https://github.com/reblocke/trinetx-preprocessing/blob/daf45eafcc87b86badd21a4c8dcdb434bd5673a3/Hypercapnia%20Final%20Dataset%20Generation%20-%20Master.ipynb)",
    "[Historical gas definitions](https://github.com/reblocke/trinetx-preprocessing/blob/daf45eafcc87b86badd21a4c8dcdb434bd5673a3/Hypercapnia%20NEW%20DATA%20-%20RFS%20Processing.ipynb)",
    "[Maintained definitions](https://github.com/reblocke/trinetx-preprocessing/blob/daf45eafcc87b86badd21a4c8dcdb434bd5673a3/docs/SPEC.md)",
    "[Archived Stata variable](https://github.com/reblocke/trinetx-hypercapnia-code/blob/d73c0559f480da16db0a5b46258a507ce4b75338/stata/do/10_preprocessing.do#L846-L877)",
)


def cohort_tables(cohort, work):
    flow = json.loads((cohort / "flow.json").read_bytes())
    flow_rows = [
        [k, f"{p:,}", f"{e:,}", "—" if k == "archived encounters" else f"{n:,}"]
        for k, (p, e, n) in flow.items()
    ]
    rows = []
    for group, setting in GROUPS.items():
        file = work / f"{group}-cohort-prediction.json"
        for name, r in json.loads(file.read_bytes())["results"].items():
            base = [f"{x:.1f}%" for x in r["baseline_percent"]]
            lo, median, hi = r["quantiles"]
            rows.append([setting, name, *base, median, f"{lo} to {hi}", *r["probabilities"]])
    flow_header = ["Selection in order", "Patients", "Encounters", "Arterial result records"]
    pred_header = [
        "Setting",
        "Population",
        "Baseline ≥45",
        "Baseline ≥52",
        "Median",
        "95% interval",
        "After ≥45",
        "After ≥52",
    ]
    return [(flow_header, flow_rows), (pred_header, rows)]


def displayed_change(alternative, reference):
    """Change between reported whole-unit values, preserving an ambiguous displayed bin."""

    def bounds(s):
        if s == "Held":
            return None
        if s.startswith(("<", ">")):
            return [0, 1] if s.startswith("<") else [99, 100]
        values = [int(v) for v in s.rstrip("%").split("–")]
        return values[0], values[-1]

    a, b = bounds(alternative), bounds(reference)
    if a is None or b is None:
        return "Held"
    lo, hi = a[0] - b[1], a[1] - b[0]
    return f"{lo:+d}" if lo == hi else f"{lo:+d} to {hi:+d}"


def build(cohort, work, output):
    private_destination(output)
    check = json.loads((work / "source-replay-check.json").read_bytes())
    if not (check["all_36_key_sets_match"] and check["shared_arterial_records_match"]):
        raise ValueError("Reconcile the source keys and laboratory records before reporting")
    overlap = json.loads((cohort / "key-overlap.json").read_bytes())
    matches = [
        v
        for k, v in overlap.items()
        if k.startswith("after|") and k.endswith("reason-first|archive")
    ]
    if len(matches) != 18 or any(sum(p[1:]) + sum(e[1:]) for p, e in matches):
        raise ValueError("The reason-first replay differs from the archived membership")
    (fh, fr), (ph, pr) = cohort_tables(cohort, work)
    sections = [
        "# How encounter selection affects the arterial reference population",
        "",
        *SOURCE,
        "The reason-first reconstruction matched all 36 saved before- and after-screen notebook key sets. Its final keys matched archived membership within all 18 indication-by-setting groups. It also reproduced each of the four patient-weighted arterial distributions. The setting-first reconstruction changed membership.",
        "",
        " · ".join(SOURCES),
        "",
        "## Ordered archived-cohort flow",
        "",
        *table(fh, fr),
        "Each row retains the preceding restrictions. The first row counts membership keys. Source results become countable at the arterial-code screen.",
        "",
        "## Prediction at observed TcCO₂ 47 mmHg",
        "",
        "Within each group all three rows use the same agreement fit. Probabilities are for simultaneous arterial PaCO₂ at or above 45 or 52 mmHg. An en dash within a rounded endpoint or probability spans two display bins allowed by its numerical bracket. An unresolved estimate reads Held.",
        "",
        *table(ph, pr),
    ]
    for stage in ("before", "after"):
        for first, second in [
            ("reason-first", "setting-first"),
            ("reason-first", "archive"),
            ("setting-first", "archive"),
        ]:
            matches = []
            for key, (p, e) in overlap.items():
                phase, setting, reason, va, vb = key.split("|")
                if (phase, va, vb) == (stage, first, second):
                    matches.append(
                        [setting, reason, *[f"{v:,}" for v in p], *[f"{v:,}" for v in e]]
                    )
            if len(matches) != 18:
                raise ValueError("Incomplete selection-order overlap")
            sections += [
                f"## {first} (A) and {second} (B), {stage} setting quality screens",
                "",
                "Reason and setting memberships overlap. Shared patients may have different encounters. Encounter identity includes both patient and encounter keys. Archive comparisons always use its final membership.",
                "",
                *table(
                    [
                        "Setting",
                        "Indication",
                        "Patients both",
                        "Only A",
                        "Only B",
                        "Encounters both",
                        "Only A",
                        "Only B",
                    ],
                    matches,
                ),
            ]
    changes = []
    for group, name in GROUPS.items():
        results = json.loads((work / f"{group}-cohort-prediction.json").read_bytes())["results"]
        a, b = results["setting-first"], results["archived"]
        changes.append(
            [
                name,
                *[
                    f"{(x - y):+.2f}"
                    for x, y in zip(a["baseline_percent"], b["baseline_percent"], strict=True)
                ],
                *[
                    displayed_change(x, y)
                    for x, y in zip(
                        a["quantiles"] + a["probabilities"],
                        b["quantiles"] + b["probabilities"],
                        strict=True,
                    )
                ],
            ]
        )
    sections += [
        "## Change with setting-first selection at 47 mmHg",
        "",
        "Setting-first minus archived. Baseline differences use the underlying patient-weighted probabilities and are in percentage points. For paired predictions the differences are between the certified whole-mmHg endpoints or median and whole-percentage-point displays. A zero in those columns means both outputs have the same displayed value.",
        "",
        *table(
            [
                "Setting",
                "Baseline ≥45 (pp)",
                "Baseline ≥52 (pp)",
                "Lower (mmHg)",
                "Median (mmHg)",
                "Upper (mmHg)",
                "After ≥45 (pp)",
                "After ≥52 (pp)",
            ],
            changes,
        ),
    ]
    (output / "encounter-selection.md").write_text("\n".join(sections).strip() + "\n")


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("cohort", "work", "output"):
        p.add_argument(n, type=lambda s: Path(s).resolve())
    build(**vars(p.parse_args()))
