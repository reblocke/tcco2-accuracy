#!/usr/bin/env python3
"""Fail closed when public Git history contains prohibited release-contract blobs.

The current-tree contract prevents newly tracked restricted-derived assets.  This
tool extends that protection to every blob reachable from public branches and
tags, without inspecting or printing data values.  It emits only paths, rule
identifiers, and Git blob identifiers; ``--write-blob-ids`` produces the exact
manifest consumed by ``git-filter-repo --strip-blobs-with-ids``.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BlobOccurrence:
    """A blob and one public-tree path at which it is reachable."""

    object_id: str
    relative_path: str


@dataclass(frozen=True)
class HistoryViolation:
    """A release-contract violation found in a reachable historical blob."""

    object_id: str
    relative_path: str
    rule_id: str


def load_contract(contract_path: Path) -> dict[str, Any]:
    """Load the machine-readable public data-release contract."""

    return json.loads(contract_path.read_text())


def public_refs(repo: Path, contract: dict[str, Any]) -> list[str]:
    """Return only refs that are public repository history, never local scratch refs."""

    history = contract["history"]
    prefixes = tuple(history["public_ref_prefixes"])
    result = _git_text(repo, "for-each-ref", "--format=%(refname)")
    refs = [
        ref
        for ref in result.stdout.splitlines()
        if ref.startswith(prefixes) and not ref.endswith("/HEAD")
    ]
    if not refs:
        raise RuntimeError("No public branch or tag refs were found for history validation.")
    return refs


def public_history_violations(repo: Path, contract: dict[str, Any]) -> list[HistoryViolation]:
    """Return every release-contract violation reachable from public heads and tags."""

    refs = public_refs(repo, contract)
    occurrences = _reachable_blob_occurrences(repo, refs)
    paths_by_blob: dict[str, set[str]] = {}
    for occurrence in occurrences:
        paths_by_blob.setdefault(occurrence.object_id, set()).add(occurrence.relative_path)

    blobs = _read_blobs(repo, paths_by_blob)
    violations: list[HistoryViolation] = []
    for object_id, paths in paths_by_blob.items():
        content = blobs[object_id]
        for relative_path in sorted(paths):
            for rule_id in _blob_rule_ids(relative_path, content, contract):
                violations.append(
                    HistoryViolation(
                        object_id=object_id,
                        relative_path=relative_path,
                        rule_id=rule_id,
                    )
                )
    return sorted(violations, key=lambda item: (item.relative_path, item.object_id, item.rule_id))


def write_blob_manifest(path: Path, violations: Iterable[HistoryViolation]) -> None:
    """Write one unique violating blob id per line for git-filter-repo."""

    object_ids = sorted({violation.object_id for violation in violations})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(object_ids) + ("\n" if object_ids else ""))


def _reachable_blob_occurrences(repo: Path, refs: list[str]) -> list[BlobOccurrence]:
    commits = _git_text(repo, "rev-list", *refs).stdout.splitlines()
    occurrences: set[BlobOccurrence] = set()
    for commit in commits:
        tree = _git_bytes(repo, "ls-tree", "-rz", "--full-tree", "-r", commit).stdout
        for record in tree.split(b"\0"):
            if not record:
                continue
            metadata, relative_path = record.split(b"\t", maxsplit=1)
            mode, object_type, object_id = metadata.split()
            if mode == b"160000" or object_type != b"blob":
                continue
            occurrences.add(
                BlobOccurrence(
                    object_id=object_id.decode("ascii"),
                    relative_path=relative_path.decode("utf-8", errors="surrogateescape"),
                )
            )
    return sorted(occurrences, key=lambda item: (item.object_id, item.relative_path))


def _read_blobs(repo: Path, paths_by_blob: dict[str, set[str]]) -> dict[str, bytes]:
    if not paths_by_blob:
        return {}
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=repo,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write("".join(f"{object_id}\n" for object_id in paths_by_blob).encode())
    process.stdin.close()

    blobs: dict[str, bytes] = {}
    for object_id in paths_by_blob:
        header = process.stdout.readline().decode("ascii").strip()
        parts = header.split()
        if len(parts) != 3 or parts[0] != object_id or parts[1] != "blob":
            process.kill()
            raise RuntimeError(f"Unable to read historical blob {object_id}.")
        size = int(parts[2])
        blobs[object_id] = process.stdout.read(size)
        newline = process.stdout.read(1)
        if newline != b"\n":
            process.kill()
            raise RuntimeError(f"Malformed git cat-file response for blob {object_id}.")
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"git cat-file failed: {stderr.strip()}")
    return blobs


def _blob_rule_ids(relative_path: str, content: bytes, contract: dict[str, Any]) -> list[str]:
    content_hash = hashlib.sha256(content).hexdigest()
    rules: list[str] = []
    allowlisted = _is_hash_allowlisted(relative_path, content_hash, contract)
    history = contract["history"]
    if not allowlisted:
        for pattern in history["prohibited_path_globs"]:
            if fnmatch.fnmatchcase(relative_path, pattern):
                rules.append(f"historical_path:{pattern}")

    suffix = Path(relative_path).suffix.lower()
    if suffix in contract["prohibited_tracked_extensions"]:
        rules.append(f"prohibited_extension:{suffix}")
    if suffix in contract["structured_schema_extensions"]:
        rules.extend(_structured_rule_ids(relative_path, content, contract, content_hash))
    if suffix in contract["prohibited_text_extensions"]:
        rules.extend(_text_rule_ids(content, contract))
    return sorted(set(rules))


def _is_hash_allowlisted(relative_path: str, content_hash: str, contract: dict[str, Any]) -> bool:
    expected = {
        **contract["canonical_agreement_artifacts"],
        **contract["retained_frozen_aggregate_artifacts"],
    }.get(relative_path)
    return expected == content_hash


def _structured_rule_ids(
    relative_path: str,
    content: bytes,
    contract: dict[str, Any],
    content_hash: str,
) -> list[str]:
    try:
        tables = _structured_column_sets(relative_path, content)
    except Exception:
        return ["uninspectable_structured_file"]

    exemptions = _schema_exemptions(relative_path, content_hash, contract)
    rules: list[str] = []
    for table_name, columns in tables:
        for rule in contract["prohibited_structured_schemas"]:
            if rule["id"] in exemptions:
                continue
            if not any(relative_path.startswith(root) for root in rule["roots"]):
                continue
            required = {column.lower() for column in rule["required_columns"]}
            prefixes = {prefix.lower() for prefix in rule.get("required_column_prefixes", [])}
            prefixes_present = all(
                any(column.startswith(prefix) for column in columns) for prefix in prefixes
            )
            if required.issubset(columns) and prefixes_present:
                rules.append(f"structured_schema:{table_name}:{rule['id']}")
    return rules


def _schema_exemptions(
    relative_path: str,
    content_hash: str,
    contract: dict[str, Any],
) -> set[str]:
    expected_hash = contract["synthetic_fixture_artifacts"].get(relative_path)
    if expected_hash != content_hash:
        return set()
    return set(contract["synthetic_fixture_schema_exemptions"].get(relative_path, []))


def _structured_column_sets(relative_path: str, content: bytes) -> list[tuple[str, set[str]]]:
    suffix = Path(relative_path).suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        text = content.decode("utf-8-sig")
        header = next(csv.reader(StringIO(text), delimiter=delimiter), [])
        columns = {column.strip().lower() for column in header}
        return [("table", columns)]
    if suffix == ".xlsx":
        sheets = pd.read_excel(BytesIO(content), sheet_name=None, nrows=0)
        return [
            (str(sheet_name), {str(column).strip().lower() for column in frame.columns})
            for sheet_name, frame in sheets.items()
        ]
    raise ValueError(f"Unsupported structured extension: {suffix}")


def _text_rule_ids(content: bytes, contract: dict[str, Any]) -> list[str]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return ["uninspectable_text_file"]
    return [
        f"text_pattern:{index}"
        for index, pattern in enumerate(contract["prohibited_text_patterns"], start=1)
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    ]


def _git_text(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )


def _git_bytes(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=ROOT,
        help="Repository to inspect (default: repository containing this script).",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        help="Release-contract JSON (default: <repo>/docs/data_release_contract.json).",
    )
    parser.add_argument(
        "--write-blob-ids",
        type=Path,
        help="Optional private manifest path for git-filter-repo --strip-blobs-with-ids.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    contract_path = args.contract or repo / "docs" / "data_release_contract.json"
    violations = public_history_violations(repo, load_contract(contract_path))
    if args.write_blob_ids:
        write_blob_manifest(args.write_blob_ids, violations)
    if not violations:
        print("Public branch/tag history satisfies the data-release contract.")
        return 0

    print(
        f"Found {len(violations)} prohibited historical occurrences in "
        f"{len({violation.object_id for violation in violations})} blobs:",
        file=sys.stderr,
    )
    for violation in violations:
        print(
            f"{violation.relative_path}: {violation.rule_id} (blob {violation.object_id})",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
