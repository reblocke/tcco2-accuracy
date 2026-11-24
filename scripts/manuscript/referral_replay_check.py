"""Check reconstructed reason-first keys against the executed notebooks' 36 saved outputs."""

import argparse
import json
import os
from pathlib import Path

import duckdb
from private_inputs import private_destination, verified_stage_inputs
from referral_cohort import REASONS, overlap


def check(outputs, cohort, original, stage, archive, dest):
    private_destination(dest)
    keys, labs = verified_stage_inputs(original, stage, archive)
    db = duckdb.connect()
    source = str(cohort / "replay.duckdb").replace("'", "''")
    db.execute(f"ATTACH '{source}' AS replay (READ_ONLY)")
    db.read_parquet(str(labs)).create_view("primary_records")
    db.read_parquet(str(keys)).create_view("primary_keys")
    db.execute("""create temp view shared_keys as select distinct patient_id,encounter_id
        from replay.selected_keys semi join primary_keys using(patient_id,encounter_id)""")
    columns = "patient_id,encounter_id,result_id,gas_day,code,value,unit,derived_flag"

    def restrict(source):
        return (
            f"select {columns} from {source} semi join shared_keys using(patient_id,encounter_id)"
        )

    primary = restrict("primary_records")
    replay = restrict("replay.newlabs")
    differences = db.execute(
        f"select count(*) from (({primary} except all {replay}) union all ({replay} except all {primary}))"
    ).fetchone()[0]
    if differences:
        raise ValueError("Source arterial records on the overlapping admitted keys differ")
    comparisons = {}
    for setting, folder in [("AMB", "AMBULATORY"), ("EMER", "EMERGENCY"), ("INPAT", "INPATIENT")]:
        for reason in REASONS:
            for suffix, keep in [("BEFORE", "true"), ("AFTER", "passed")]:
                name = f"RFS_{reason}_ENC_{setting}_{suffix}.csv"
                db.read_csv(str(outputs / folder / name), all_varchar=True).create_view(
                    "saved_run", replace=True
                )
                query = f"""select patient_id,encounter_id,true a,false b from replay.selections
                    where variant='reason-first' and setting='{setting}' and reason='{reason}' and {keep}
                    union all select patient_id,encounter_id,false a,true b from saved_run"""
                comparisons[name] = overlap(db, query)
        print(setting, "executed notebook keys checked", flush=True)
    matches = all(not sum(row[1:]) for result in comparisons.values() for row in result)
    (dest / "source-replay-check.json").write_text(
        json.dumps(
            {
                "shared_arterial_records_match": True,
                "all_36_key_sets_match": matches,
                "notebook_key_counts": comparisons,
            },
            indent=2,
        )
    )
    if not matches:
        raise ValueError(
            "Reconstructed keys differ from at least one executed notebook output; see private receipt"
        )
    print("All 36 executed notebook key sets reconciled", flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("outputs", "cohort", "original", "stage", "archive", "dest"):
        p.add_argument(n, type=lambda s: Path(s).resolve())
    check(**vars(p.parse_args()))
