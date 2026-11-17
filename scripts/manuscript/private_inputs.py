"""Verify derived inputs against the private cohort and public posterior receipts."""

import hashlib
import json
from pathlib import Path

# Reviewed fingerprints of the private source provenance record and manuscript restage.
PROVENANCE_SHA256 = "705492b52e2438ce15e6989fb247530a7042c0321dfa30aa2935dacc869ad55c"
STAGE_RECEIPT_SHA256 = "aff7a5d681c13abb15f11867fcb052ec7ea2c6404e3360e77dc6c3621d89d0f7"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require_digest(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Input differs from its receipt: {path.name}")


def model_receipts(original, manifest):
    require_digest(manifest, "e7d4b78abe88df745c4d8f4d41138a5d9b1f3572af1bec401b96a15c433c9795")
    [receipt] = original.glob("all-grid-review.*/verified-receipt.json")
    require_digest(receipt, "dfe704f116c196339553d719948a77f9cf589c25f61caaabf8cd3ce063c23c2d")
    return (
        json.loads(manifest.read_bytes())["strata"],
        receipt.parent,
        json.loads(receipt.read_bytes()),
    )


def verified_stage_inputs(original, stage, archive):
    provenance = original / "private-input.json"
    require_digest(provenance, PROVENANCE_SHA256)
    record = json.loads(provenance.read_bytes())
    policy_name = "equal_patient_weight_split_across_eligible_draws_on_first_testing_day"
    if (record["use"], record["index_target"], record["index_policy"]) != (
        "approved_private_index_analysis",
        "all",
        policy_name,
    ):
        raise ValueError("Unexpected original population or weighting policy")

    source = original / "stage-receipt.json"
    require_digest(source, record["extraction"]["stage_receipt_sha256"])
    original_policy = json.loads(source.read_bytes())["policy"]
    restaged = stage / "stage-receipt.json"
    require_digest(restaged, STAGE_RECEIPT_SHA256)
    policy = json.loads(restaged.read_bytes())["policy"]
    for key in ("index_policy", "source_files_sha256", "archived_keyed_member_sha256"):
        if policy[key] != original_policy[key]:
            raise ValueError(f"Restaged inputs differ from original provenance: {key}")
    cohort = original.parent / "cohort-keys.parquet"
    labs = stage / "arterial-candidates.parquet"
    require_digest(cohort, original_policy["source_files_sha256"][cohort.name])
    require_digest(labs, policy["staged_candidate_sha256"])
    require_digest(archive, original_policy["archived_keyed_member_sha256"])
    require_digest(original / "first-day-weights.csv", record["eligible_extract_sha256"])
    return cohort, labs


def private_destination(out):
    root = Path(__file__).resolve().parents[2]
    if (
        (root / ".git").exists()
        and out.is_relative_to(root)
        and not any(out.is_relative_to(root / p) for p in (".pytest_tmp", ".tmp"))
    ):
        raise ValueError("Use the approved private workspace for empirical outputs")
    if not out.is_dir():
        raise ValueError("Private output directory must exist")


def read_prior(out, target):
    import pandas as pd

    path = out / (target + "-weights.csv")
    info = json.loads((out / "cohort.json").read_text())[target]
    require_digest(path, info["prior_sha256"])
    return pd.read_csv(path, float_precision="round_trip").to_numpy().T
