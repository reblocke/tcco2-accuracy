"""The restricted input chain must fail closed even under optimized Python."""

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/manuscript"


def test_self_consistent_replacement_stage_is_rejected_under_optimization(tmp_path):
    original = tmp_path / "original"
    stage = tmp_path / "stage"
    original.mkdir()
    stage.mkdir()
    paths = [
        tmp_path / "cohort-keys.parquet",
        stage / "arterial-candidates.parquet",
        tmp_path / "archive.dta",
        original / "first-day-weights.csv",
    ]
    for path in paths:
        path.write_text("invented fixture")
    h = sha256(paths[0].read_bytes()).hexdigest()
    policy = {
        "source_files_sha256": {"cohort-keys.parquet": h},
        "archived_keyed_member_sha256": h,
        "staged_candidate_sha256": h,
        "index_policy": "equal_patient_weight_split_across_eligible_draws_on_first_testing_day",
    }
    receipt = json.dumps({"policy": policy}).encode()
    orig_receipt = original / "stage-receipt.json"
    orig_receipt.write_bytes(receipt)
    active = stage / "stage-receipt.json"
    active.write_bytes(receipt)
    private = original / "private-input.json"
    private.write_text(
        json.dumps(
            {
                "use": "approved_private_index_analysis",
                "index_target": "all",
                "index_policy": policy["index_policy"],
                "eligible_extract_sha256": h,
                "extraction": {"stage_receipt_sha256": sha256(receipt).hexdigest()},
            }
        )
    )
    command = [
        sys.executable,
        "-O",
        "-c",
        f"import sys;sys.path.insert(0,{str(SCRIPTS)!r});from pathlib import Path;import private_inputs as p;p.PROVENANCE_SHA256={sha256(private.read_bytes()).hexdigest()!r};p.STAGE_RECEIPT_SHA256={sha256(receipt).hexdigest()!r};p.verified_stage_inputs(*(Path(a) for a in sys.argv[1:]))",
        str(original),
        str(stage),
        str(paths[2]),
    ]
    subprocess.run(command, check=True)
    paths[2].write_text("substituted archive")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode and "archive.dta" in result.stderr
    policy["archived_keyed_member_sha256"] = sha256(paths[2].read_bytes()).hexdigest()
    active.write_text(json.dumps({"policy": policy}))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode and "stage-receipt.json" in result.stderr
