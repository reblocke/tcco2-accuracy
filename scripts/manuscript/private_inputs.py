"""Verify derived inputs against the private cohort and public posterior receipts."""

import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require_digest(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Input differs from its receipt: {path.name}")


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
