"""Replay the two encounter-selection orders on one recovered set of historical candidates."""

import argparse
import hashlib
import json
import os
from pathlib import Path

REASONS = ("ABG", "VBG", "RESPFAIL", "OBESITY", "VENTSUPPORT", "PREDISPOSITION")
SETTINGS = ("AMB", "EMER", "INPAT")
GROUPS = {
    "all": "true",
    "icu": "is_inp=1 AND cc_time=1 AND is_emer=0 AND is_amb=0",
    "ed_inp": "(is_inp=1 OR is_emer=1) AND cc_time=0 AND is_amb=0",
    "pft": "is_amb=1",
}


def select_keys(db):
    # The notebook first sorts by qualification date and descending encounter identifier.
    db.execute("""CREATE OR REPLACE TEMP TABLE selected AS
        WITH earliest_global AS (SELECT * FROM events QUALIFY
            row_number() OVER (PARTITION BY patient_id ORDER BY event_day,encounter_id DESC)=1),
        joined AS (SELECT e.*, s.setting,s.start_date,s.end_date FROM events e JOIN enc s USING(encounter_id)),
        earliest_local AS (SELECT * FROM joined QUALIFY row_number() OVER
            (PARTITION BY setting,patient_id ORDER BY event_day,encounter_id DESC)=1),
        candidates AS (
            SELECT 'reason-first' AS variant, e.*,s.setting,s.start_date,s.end_date
              FROM earliest_global e JOIN enc s USING(encounter_id)
            UNION ALL SELECT 'setting-first' AS variant,* FROM earliest_local)
        SELECT DISTINCT variant, patient_id, encounter_id, setting,event_day
          FROM candidates JOIN age USING(patient_id)
         WHERE event_day BETWEEN start_date AND end_date AND year(event_day)-birth_year BETWEEN 18 AND 109""")


def counts(db, table):
    return list(
        db.execute(
            "SELECT count(DISTINCT patient_id),count(DISTINCT (patient_id,encounter_id)),count(*) FROM "
            + table
        ).fetchone()
    )


def overlap(db, query):
    # Count patients and compound encounters independently: a patient may change encounters.
    result = []
    for unit in ["patient_id", "patient_id,encounter_id"]:
        result.append(
            list(
                db.execute(f"""WITH keys AS ({query}), sides AS
            (SELECT {unit},bool_or(a) a,bool_or(b) b FROM keys GROUP BY {unit})
            SELECT count(*) FILTER(WHERE a AND b),count(*) FILTER(WHERE a AND NOT b),
                   count(*) FILTER(WHERE b AND NOT a) FROM sides""").fetchone()
            )
        )
    return result


# The executed notebook excludes Ex-US and unknown region, then applies pandas
# dropna to the gas-event key and the selected demographics before the first encounter.
DEMOGRAPHICS = """
    SELECT patient_id,coalesce(try_cast(year_of_birth AS int),0) birth_year FROM dem
    WHERE patient_id IS NOT NULL AND patient_regional_location NOT IN ('Ex-US','Unknown')
      AND patient_regional_location IS NOT NULL AND sex IS NOT NULL
      AND race IS NOT NULL AND ethnicity IS NOT NULL
"""


def write_overlaps(db, dest):
    overlaps = {}
    for stage_name, predicate in [("before", "true"), ("after", "passed")]:
        for s in SETTINGS:
            for r in REASONS:
                for va, vb in [
                    ("reason-first", "setting-first"),
                    ("reason-first", "archive"),
                    ("setting-first", "archive"),
                ]:

                    def query(v):
                        if v == "archive":
                            return f"select patient_id,encounter_id from archive_members where setting='{s}' and reason='{r}'"
                        return f"select patient_id,encounter_id from selections where variant='{v}' and {predicate} and setting='{s}' and reason='{r}'"

                    both = f"select *,true a,false b from ({query(va)}) union all select *,false a,true b from ({query(vb)})"
                    overlaps["|".join([stage_name, s, r, va, vb])] = overlap(db, both)
    (dest / "key-overlap.json").write_text(json.dumps(overlaps))


def write_populations(db, dest):
    from cohort import first_day_weights

    for variant in ("reason-first", "setting-first"):
        for group, flag in GROUPS.items():
            for t in ("eligible", "weighted", "first_results"):
                db.execute("drop table if exists " + t)
            db.execute(f"""create or replace temp view records as select l.* from newlabs l semi join
                (select * from selected_keys where variant='{variant}' and {flag}) s using(patient_id,encounter_id)""")
            weights, _ = first_day_weights(db)
            weights.to_csv(dest / f"{variant}-{group}-weights.csv", index=False)


def run(work, patient, store, original, stage, archive, canonical, dest):
    import duckdb
    import pandas as pd
    from cohort import CONVERTED
    from private_inputs import private_destination, verified_stage_inputs

    private_destination(dest)
    keys, labs = verified_stage_inputs(original, stage, archive)
    root = dest / "replay.duckdb"
    if root.exists():
        raise ValueError("Use a fresh replay directory")
    db = duckdb.connect(str(root))
    db.execute("set threads=4")
    db.execute("set memory_limit='5GB'")
    db.execute("set preserve_insertion_order=false")
    db.read_parquet(str(keys)).create_view("archive_keys")
    db.read_parquet(str(labs)).create_view("staged")
    db.execute("CREATE TEMP VIEW records AS SELECT * FROM staged")
    db.execute("CREATE TEMP VIEW converted AS " + CONVERTED)
    flow = {}
    flow["archived encounters"] = counts(db, "archive_keys")
    flow["archived encounters"][2] = None  # Source result records begin at the next stage.
    flow["candidate arterial codes"] = counts(db, "staged")
    flow["source-recorded"] = counts(
        db, "staged WHERE lower(trim(derived_flag)) IN ('f','false','n','no','0')"
    )
    flow["convertible pressure unit"] = counts(
        db,
        "converted WHERE unit_key IN ('mm[hg]','mmhg','torr','kpa','kilopascal','kilopascals','pa','pascal','pascals','cm[h2o]','cmh2o')",
    )
    flow["finite pressure"] = counts(db, "converted WHERE isfinite(paco2)")
    db.execute(
        "CREATE TEMP VIEW initial_eligible AS SELECT * FROM converted WHERE isfinite(paco2) AND paco2>2 AND paco2<250"
    )
    flow["eligible 2 < PaCO₂ < 250"] = counts(db, "initial_eligible")
    flow["first eligible testing day"] = counts(
        db,
        "(SELECT * FROM initial_eligible QUALIFY gas_day=min(gas_day) OVER (PARTITION BY patient_id))",
    )
    known = json.loads((store / "cohort.json").read_text())["all"]["flow"]
    assert (
        list(known["eligible"]) == flow["eligible 2 < PaCO₂ < 250"]
        and list(known["weighted"]) == flow["first eligible testing day"]
    )
    (dest / "flow.json").write_text(json.dumps(flow, indent=2))
    # Keep source metadata and digests alongside the controlled data.
    sources = {}

    def source(path, view):
        sources[view] = {"path": str(path), "bytes": path.stat().st_size}
        db.read_csv(str(path), header=True, all_varchar=True).create_view(view, replace=True)

    source(patient, "dem")
    db.execute("CREATE TABLE age AS " + DEMOGRAPHICS)
    for s in SETTINGS:
        source(work / (s + "_encounters.csv"), "setting_" + s)
        db.execute(f"""CREATE TABLE enc_{s} AS SELECT DISTINCT encounter_id, '{s}' setting,
            cast(start_date as date) start_date,cast(end_date as date) end_date FROM setting_{s}
            WHERE cast(start_date as date)<=date '2022-12-31' AND cast(end_date as date)>=date '2022-01-01'""")
    db.execute(
        "CREATE VIEW enc AS " + " UNION ALL ".join("SELECT * FROM enc_" + s for s in SETTINGS)
    )
    source(work / "data_checks" / "amb_enc_screen.csv", "amb_screen")
    source(work / "data_checks" / "inp_enc_screen.csv", "inp_screen")
    db.execute(
        "create table screens as select distinct encounter_id, 'AMB' AS kind from amb_screen union all select distinct encounter_id, 'IP' AS kind from inp_screen"
    )
    db.execute(
        "CREATE TABLE selections(variant VARCHAR,patient_id VARCHAR,encounter_id VARCHAR,setting VARCHAR,event_day DATE,reason VARCHAR,passed BOOLEAN)"
    )
    for reason in REASONS:
        f = work / ("RFS_" + reason + ".csv")
        source(f, "source_" + reason)
        if reason in ("ABG", "VBG"):
            codes = ("2019-8", "2026-3", "32771-8") if reason == "ABG" else ("11557-6", "2021-4")
            bad = db.execute(
                f"select count(*) from source_{reason} where code NOT IN ? OR try_cast(lab_result_num_val as double) <= 5 OR try_cast(lab_result_num_val as double) >= 200 OR try_cast(lab_result_num_val as double) IS NULL",
                [list(codes)],
            ).fetchone()[0]
            if bad:
                raise ValueError("Candidate gas criteria differ from the historical notebook")
        db.execute(f"""CREATE OR REPLACE TEMP TABLE events AS SELECT patient_id,encounter_id,
            try_cast(date as date) event_day FROM source_{reason}
            WHERE patient_id IS NOT NULL AND encounter_id IS NOT NULL
            AND try_cast(date as date) BETWEEN DATE '2022-01-01' AND DATE '2022-12-31'
            QUALIFY row_number() OVER (PARTITION BY encounter_id ORDER BY event_day,patient_id)=1""")
        # Confirm that encounter-level deduplication cannot exchange patients.
        collision = db.execute(f"""SELECT count(*) FROM (SELECT encounter_id FROM source_{reason}
               WHERE try_cast(date as date) BETWEEN DATE '2022-01-01' AND DATE '2022-12-31'
               GROUP BY encounter_id HAVING min(patient_id)<>max(patient_id))""").fetchone()[0]
        if collision:
            raise ValueError("Historical source-order ties require disambiguation")
        select_keys(db)
        db.execute(
            """INSERT INTO selections SELECT s.*,?, sc.encounter_id IS NOT NULL FROM selected s
             LEFT JOIN screens sc ON sc.encounter_id=s.encounter_id AND sc.kind=if(s.setting='AMB','AMB','IP')""",
            [reason],
        )
        print(reason, "selection orders replayed", flush=True)
    cols = ["pat_enc_hash", "is_amb", "is_emer", "is_inp", *[r + "_rfs" for r in REASONS]]
    data = pd.read_stata(archive, columns=cols, convert_categoricals=False)
    db.register("dta", data)
    ref = []
    for s, flag in zip(SETTINGS, ["is_amb", "is_emer", "is_inp"], strict=True):
        for r in REASONS:
            ref.append(f"""SELECT DISTINCT patient_id,encounter_id,'{s}' setting,'{r}' reason FROM archive_keys k
                JOIN dta d ON d.pat_enc_hash=k.patient_id||'-'||k.encounter_id WHERE d.{flag}=1 AND d.{r}_rfs=1""")
    db.execute("create table archive_members as " + " UNION ALL ".join(ref))
    write_overlaps(db, dest)
    source(work / "has_99291.csv", "billing_source")
    db.execute(
        "create table billing as select distinct patient_id,encounter_id from billing_source where try_cast(date as date)>=date '2022-01-01'"
    )
    db.execute("""create table selected_keys as select variant,patient_id,encounter_id,
        max((setting='AMB')::int) is_amb,max((setting='EMER')::int) is_emer,max((setting='INPAT')::int) is_inp,
        max((b.encounter_id is not null)::int) cc_time from selections s left join billing b using(patient_id,encounter_id)
        where passed group by variant,patient_id,encounter_id""")
    # Obtain candidates for both alternative memberships, including those outside the archived key list.
    quoted = str(canonical).replace("'", "''")
    db.execute(f"ATTACH '{quoted}' AS source_db (READ_ONLY)")
    db.execute("""create table newlabs as select g.patient_id,g.encounter_id,g.source_record_id as result_id,
           g.code,g.event_datetime::date as gas_day,g.numeric_value as value,g.units_of_measure as unit,g.derived_by_TriNetX as derived_flag
           from source_db.source_lab_measurement g semi join selected_keys k using(patient_id,encounter_id)
           where g.code IN ('2019-8','20198','32771-8','327718') and upper(trim(g.code_system))='LOINC'""")
    write_populations(db, dest)
    for v in sources.values():
        with Path(v["path"]).open("rb") as fp:
            v["sha256"] = hashlib.file_digest(fp, "sha256").hexdigest()
    (dest / "source-snapshot.json").write_text(json.dumps(sources, indent=2))
    db.close()
    print("Alternative first-day populations reconstructed", flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "patient", "store", "original", "stage", "archive", "canonical", "dest"):
        p.add_argument(name, type=lambda s: Path(s).resolve())
    run(**vars(p.parse_args()))
