"""The historical reduction order can remove an entire setting for the same patient."""

import pytest


def test_replay_preserves_single_order_difference_and_date_tie():
    db = pytest.importorskip("duckdb").connect()
    from scripts.manuscript.referral_cohort import select_keys

    db.execute("""create table events as select * from (values
               ('p','e1',date '2022-01-01'),('p','e2',date '2022-02-01'),
               ('r','e3',date '2022-03-01'),('r','e4',date '2022-03-01'),
               ('z','e5',date '2022-03-01')) t(patient_id,encounter_id,event_day)""")
    db.execute("""create table enc as select *, date '2022-01-01' as start_date,
                  date '2022-12-31' as end_date from
               (values ('e1','AMB'),('e2','EMER'),('e3','AMB'),('e4','INPAT'),('e5','AMB'))
               t(encounter_id,setting)""")
    db.execute(
        "create table age as select * from (values ('p',1960),('r',1980),('z',2014)) t(patient_id,birth_year)"
    )
    select_keys(db)
    reason = db.execute(
        "select encounter_id from selected where variant='reason-first' order by 1"
    ).fetchall()
    setting = db.execute(
        "select encounter_id from selected where variant='setting-first' order by 1"
    ).fetchall()
    assert reason == [("e1",), ("e4",)]
    assert setting == [("e1",), ("e2",), ("e3",), ("e4",)]
