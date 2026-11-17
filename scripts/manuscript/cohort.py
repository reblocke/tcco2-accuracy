"""Run on the private host against the fingerprinted original cohort and staged labs."""

import hashlib
import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from private_inputs import digest, private_destination

from tcco2_accuracy.core._posterior import weighted_quantile

CONVERTED = """
SELECT *, CASE
    WHEN unit_key IN ('mm[hg]', 'mmhg', 'torr') THEN value
    WHEN unit_key IN ('kpa','kilopascal','kilopascals') THEN value*7.50062
    WHEN unit_key IN ('pa','pascal','pascals') THEN value*0.00750062
    WHEN unit_key IN ('cm[h2o]','cmh2o') THEN value*(0.0980665*7.50062)
    END AS paco2
FROM (
    SELECT *, lower(replace(trim(unit),' ','')) AS unit_key
    FROM records WHERE lower(trim(derived_flag)) IN ('f','false','n','no','0')
      AND code IN ('2019-8','20198','32771-8','327718')
)
"""


def first_day_weights(db):
    db.execute(
        "CREATE TEMP TABLE eligible AS SELECT patient_id,encounter_id,result_id,gas_day,paco2 FROM ("
        + CONVERTED
        + ") WHERE isfinite(paco2) AND paco2>2 AND paco2<250"
    )
    key_check, dates_known = db.execute("""
        SELECT count(*)>0 AND count(*)=count(result_id) AND count(*)=count(DISTINCT result_id)
               AND count(*)=count(patient_id) AND count(*)=count(encounter_id), count(*)=count(gas_day)
        FROM eligible
    """).fetchone()
    if not key_check or not dates_known:
        raise ValueError("Missing day or nonunique source key in eligible observations")
    db.execute("""
        CREATE TEMP TABLE first_results AS
        SELECT * FROM eligible
        QUALIFY gas_day = min(gas_day) OVER (PARTITION BY patient_id)
    """)
    db.execute("""
        CREATE TEMP TABLE weighted AS SELECT *, 1.0/count(*) OVER (PARTITION BY patient_id) AS contribution
        FROM first_results
    """)
    per_patient, one_day_each = db.execute("""
        SELECT max(abs(patient_total-1)) <= 1e-12, max(days)=1
        FROM (SELECT patient_id, fsum(contribution) AS patient_total,
                     count(DISTINCT gas_day) AS days FROM weighted GROUP BY patient_id)
    """).fetchone()
    if not per_patient or not one_day_each:
        raise ValueError("Patient or calendar-day weights failed")
    sql = db.execute("""
        SELECT paco2 AS paco2_bin, fsum(contribution)/(SELECT count(DISTINCT patient_id) FROM eligible) AS weight
        FROM weighted GROUP BY paco2 ORDER BY paco2
    """).fetchdf()
    # Independently reconstruct using sorted pandas records and value multiplicities per patient.
    all_rows = db.execute("SELECT patient_id,gas_day,paco2 FROM eligible").fetchdf()
    ordered = all_rows.sort_values("gas_day", kind="stable")
    first_day = ordered.drop_duplicates("patient_id").set_index("patient_id")["gas_day"]
    first = all_rows.loc[all_rows["gas_day"].eq(all_rows["patient_id"].map(first_day))]
    by_value = first.groupby(["patient_id", "paco2"], sort=True).size()
    fractions = by_value.div(by_value.groupby(level=0).transform("sum"))
    independent = fractions.groupby(level=1).sum() / len(first_day)
    if (
        not np.array_equal(independent.index.to_numpy(), sql.paco2_bin)
        or not np.allclose(independent.to_numpy(), sql.weight, rtol=2e-12, atol=1e-16)
        or not np.isclose(sql.weight.sum(), 1, rtol=0, atol=1e-12)
        or not np.all(sql.weight > 0)
    ):
        raise ValueError("Independent support, mass or normalization check failed")
    return sql, dict(
        each_patient_contributes_one=bool(per_patient),
        one_first_day_per_patient=bool(one_day_each),
        independent_weights_match=True,
        unique_source_records=True,
        eligible_dates_known=True,
    )


def build(original, stage, archive, out):
    private_destination(out)

    cohort = original.parent / "cohort-keys.parquet"
    labs = stage / "arterial-candidates.parquet"
    policy = json.loads((stage / "stage-receipt.json").read_text())["policy"]
    for p, expected in [
        (cohort, policy["source_files_sha256"][cohort.name]),
        (labs, policy["staged_candidate_sha256"]),
        (archive, policy["archived_keyed_member_sha256"]),
    ]:
        with p.open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == expected
    flags = {
        "all": "true",
        "icu": "is_inp=1 AND cc_time=1 AND is_emer=0 AND is_amb=0",
        "ed_inp": "(is_inp=1 OR is_emer=1) AND cc_time=0 AND is_amb=0",
        "pft": "is_amb=1",
    }
    fields = [
        "male",
        "female",
        "white_race",
        "black_race",
        "asian_race",
        "hisp_eth",
        "osa",
        "asthma",
        "copd",
        "chf",
        "oud",
        "phtn",
        "ckd",
        "dm",
    ]
    cols = [
        "pat_enc_hash",
        "age_at_encounter",
        "paco2",
        "paco2_date",
        "is_inp",
        "is_emer",
        "is_amb",
        "cc_time",
        *fields,
    ]
    demo = pd.read_stata(archive, columns=cols, convert_categoricals=False)
    assert demo.pat_enc_hash.is_unique
    history = []
    for name, sel in [
        ("available PaCO2", demo.paco2.notna()),
        ("day zero", demo.paco2.notna() & demo.paco2_date.eq(0)),
    ]:
        s = demo.loc[sel]
        masks = [
            s.is_amb.eq(1),
            s.is_inp.eq(1) & s.cc_time.eq(1) & s.is_emer.eq(0) & s.is_amb.eq(0),
            s.is_inp.eq(1) & s.cc_time.eq(0) & s.is_emer.eq(0),
        ]
        history.append(
            [
                name,
                len(s),
                *[int(m.sum()) for m in masks],
                int((~(masks[0] | masks[1] | masks[2])).sum()),
            ]
        )
    (out / "historical-check.json").write_text(json.dumps(history))
    report = {}
    for group, condition in flags.items():
        with duckdb.connect() as db:
            db.read_parquet(str(labs)).create_view("labs")
            db.read_parquet(str(cohort)).create_view("cohort")
            db.execute(
                "create view records as select labs.* from labs join cohort using(patient_id,encounter_id) where "
                + condition
            )
            prior, checks = first_day_weights(db)
            a, w = prior.to_numpy().T
            if group == "all":
                anchor = pd.read_csv(
                    original / "first-day-weights.csv", float_precision="round_trip"
                )
                np.testing.assert_allclose(prior, anchor, rtol=2e-14, atol=1e-16)
            prior.to_csv(out / (group + "-weights.csv"), index=False, float_format="%.17g")
            flow = {}
            for table in ("eligible", "weighted"):
                flow[table] = db.execute(
                    "select count(distinct patient_id), count(distinct (patient_id,encounter_id)), count(*) from "
                    + table
                ).fetchone()
            db.register("demographics", demo)
            d = db.execute(
                "select d.*,w.contribution from weighted w left join demographics d on d.pat_enc_hash=w.patient_id||'-'||w.encounter_id"
            ).fetchdf()
            assert d.pat_enc_hash.notna().all()
            c = {}
            for col in ["age_at_encounter", *fields]:
                ok = d[col].notna()
                x = d.loc[ok, col].to_numpy()
                v = d.loc[ok, "contribution"].to_numpy()
                if col == "age_at_encounter":
                    ix = np.argsort(x)
                    val = weighted_quantile(x[ix], v[ix] / v.sum(), [0.25, 0.5, 0.75]).tolist()
                else:
                    assert np.isin(x, [0, 1]).all()
                    val = float(np.average(x, weights=v))
                c[col] = dict(value=val, known_fraction=float(v.sum() / d.contribution.sum()))
            report[group] = dict(
                prior_sha256=digest(out / (group + "-weights.csv")),
                flow=flow,
                checks=checks,
                characteristics=c,
                prior_interval=weighted_quantile(a, w, [0.025, 0.5, 0.975]).tolist(),
                prior_thresholds=[float(w[a >= k].sum()) for k in (45, 52)],
            )
    (out / "cohort.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.umask(0o077)
    build(*(Path(p).resolve() for p in sys.argv[1:]))
