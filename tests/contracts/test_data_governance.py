from __future__ import annotations

import csv
import fnmatch
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs" / "data_release_contract.json"
PROVENANCE_TEMPLATE_PATH = ROOT / "docs" / "restricted_data_provenance.template.json"
HISTORY_CHECK_PATH = ROOT / "scripts" / "check_public_history.py"
CONWAY_CSV = ROOT / "Data" / "conway_studies.csv"
CONWAY_XLSX = ROOT / "Data" / "conway_studies.xlsx"
HUMAN_REVIEW_REQUIRED = "HUMAN REVIEW REQUIRED"


_HISTORY_CHECK_SPEC = importlib.util.spec_from_file_location(
    "check_public_history",
    HISTORY_CHECK_PATH,
)
assert _HISTORY_CHECK_SPEC is not None
assert _HISTORY_CHECK_SPEC.loader is not None
_HISTORY_CHECK_MODULE = importlib.util.module_from_spec(_HISTORY_CHECK_SPEC)
sys.modules[_HISTORY_CHECK_SPEC.name] = _HISTORY_CHECK_MODULE
_HISTORY_CHECK_SPEC.loader.exec_module(_HISTORY_CHECK_MODULE)


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text())


def _tracked_existing_files() -> list[str]:
    return [
        relative for relative in _git("ls-files").stdout.splitlines() if (ROOT / relative).is_file()
    ]


def test_release_contract_covers_public_history_and_external_limitations() -> None:
    contract = _contract()

    assert contract["contract_version"] == "1.1.0"
    assert contract["scope"]["history_rewritten"] is True
    assert "Public branch and tag refs" in contract["scope"]["history_note"]
    assert contract["history"]["public_ref_prefixes"] == [
        "refs/heads/",
        "refs/remotes/origin/",
        "refs/tags/",
    ]
    assert "Data/paco2_public_prior.*" in contract["history"]["prohibited_path_globs"]
    assert contract["private_output_roots"] == [
        ".pytest_tmp/",
        ".tmp/",
        "explicit path outside the repository",
    ]


def test_prohibited_paths_are_not_in_the_current_tracked_tree() -> None:
    contract = _contract()
    tracked = _tracked_existing_files()
    hash_locked = {
        *contract["canonical_agreement_artifacts"],
        *contract["retained_frozen_aggregate_artifacts"],
    }
    violations = sorted(
        relative
        for relative in tracked
        if relative not in hash_locked
        if any(
            fnmatch.fnmatchcase(relative, pattern)
            for pattern in contract["prohibited_tracked_globs"]
        )
    )

    assert violations == []


@pytest.mark.parametrize(
    "relative",
    [
        "Data/paco2_public_prior.xlsx",
        "artifacts/manuscript_table1.xlsx",
        "artifacts/conditional_classification_t60.tsv",
        "tests/fixtures/paco2_distribution_summary.xlsx",
    ],
)
def test_known_restricted_output_stems_are_format_independent(relative: str) -> None:
    assert any(
        fnmatch.fnmatchcase(relative, pattern)
        for pattern in _contract()["prohibited_tracked_globs"]
    )


def test_private_output_paths_are_ignored() -> None:
    for relative in _contract()["ignored_private_paths"]:
        result = _git("check-ignore", "--no-index", "--", relative)
        assert result.returncode == 0, f"Private output path is not ignored: {relative}"


def test_prohibited_structured_schemas_are_absent_from_public_tracked_roots() -> None:
    contract = _contract()
    violations: list[str] = []
    for relative in _tracked_existing_files():
        path = ROOT / relative
        if path.suffix.lower() not in contract["structured_schema_extensions"]:
            continue
        violations.extend(_prohibited_schema_violations(relative, path, contract))

    assert violations == []


def test_uninspectable_legacy_xls_is_not_tracked() -> None:
    prohibited = set(_contract()["prohibited_tracked_extensions"])

    assert prohibited == {".xls"}
    assert [
        relative
        for relative in _tracked_existing_files()
        if (ROOT / relative).suffix.lower() in prohibited
    ] == []


def test_structured_schema_scanner_checks_tsv_and_every_xlsx_sheet(tmp_path: Path) -> None:
    contract = _contract()
    tsv_path = tmp_path / "renamed.tsv"
    tsv_path.write_text("subgroup\tcount\tpaco2_q500\nall\t42\t45\n")
    xlsx_path = tmp_path / "renamed.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        pd.DataFrame({"safe": [1]}).to_excel(writer, sheet_name="safe", index=False)
        pd.DataFrame({"group": ["all"], "count": [42], "paco2_q025": [30]}).to_excel(
            writer, sheet_name="restricted", index=False
        )

    assert _prohibited_schema_violations("Data/renamed.tsv", tsv_path, contract) == [
        "Data/renamed.tsv: table: paco2_subgroup_count_summary"
    ]
    assert _prohibited_schema_violations("artifacts/renamed.xlsx", xlsx_path, contract) == [
        "artifacts/renamed.xlsx: restricted: paco2_group_count_summary"
    ]


@pytest.mark.parametrize(
    ("header", "expected_rule"),
    [
        ("paco2_bin,count", "exact_paco2_bin_counts"),
        ("paco2_bin,weight", "reconstructable_prior_weights"),
        ("n_encounters,value", "restricted_encounter_denominator"),
    ],
)
def test_structured_schema_scanner_normalizes_utf8_bom(
    tmp_path: Path,
    header: str,
    expected_rule: str,
) -> None:
    path = tmp_path / "renamed.csv"
    path.write_bytes(b"\xef\xbb\xbf" + f"{header}\n1,1\n".encode())

    assert _prohibited_schema_violations("renamed.csv", path, _contract()) == [
        f"renamed.csv: table: {expected_rule}"
    ]


def test_group_count_schema_requires_paco2_context(tmp_path: Path) -> None:
    path = tmp_path / "benign.csv"
    path.write_text("group,count\ncontrol,42\n")

    assert _prohibited_schema_violations("benign.csv", path, _contract()) == []


def test_synthetic_fixture_exemption_is_path_and_rule_specific() -> None:
    contract = _contract()
    relative = "tests/fixtures/synthetic_paco2_prior.csv"
    path = ROOT / relative

    assert contract["synthetic_fixture_schema_exemptions"] == {
        relative: ["reconstructable_prior_weights"]
    }
    assert _prohibited_schema_violations(relative, path, contract) == []
    assert _prohibited_schema_violations("tests/fixtures/renamed_prior.csv", path, contract) == [
        "tests/fixtures/renamed_prior.csv: table: reconstructable_prior_weights"
    ]
    assert contract["synthetic_fixture_artifacts"] == {
        relative: "08fcfa76380157cb8f613d19559dea29289d71d70cd5e661893b7a6248419e46"
    }
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == contract["synthetic_fixture_artifacts"][relative]
    )


def test_paco2_group_count_markdown_signature_is_specific() -> None:
    patterns = [
        re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
        for pattern in _contract()["prohibited_text_patterns"]
    ]

    assert any(pattern.search("| Group | Count | PaCO2 q2.5 |") for pattern in patterns)
    assert not any(pattern.search("| Group | Count | Mean age |") for pattern in patterns)


def test_prohibited_exact_count_text_schemas_are_absent() -> None:
    contract = _contract()
    patterns = [
        re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
        for pattern in contract["prohibited_text_patterns"]
    ]
    violations: list[str] = []
    for relative in _tracked_existing_files():
        path = ROOT / relative
        if path.suffix.lower() not in contract["prohibited_text_extensions"]:
            continue
        text = path.read_text()
        for pattern in patterns:
            if pattern.search(text):
                violations.append(f"{relative}: {pattern.pattern}")

    assert violations == []


def test_public_branch_and_tag_history_satisfies_release_contract() -> None:
    result = subprocess.run(
        [sys.executable, str(HISTORY_CHECK_PATH), "--repo", str(ROOT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_public_history_scans_deleted_ancestor_and_tag_only_blobs(tmp_path: Path) -> None:
    repo = tmp_path / "history-repo"
    repo.mkdir()
    _git_in_repo(repo, "init", "-q", "--initial-branch", "main")
    _git_in_repo(repo, "config", "user.email", "test@example.invalid")
    _git_in_repo(repo, "config", "user.name", "Release Contract Test")

    safe = repo / "README.md"
    safe.write_text("safe\n")
    _git_in_repo(repo, "add", "README.md")
    _git_in_repo(repo, "commit", "-qm", "Initial safe commit")

    restricted = repo / "Data" / "paco2_public_prior.csv"
    restricted.parent.mkdir()
    restricted.write_text("paco2_bin,weight\n40,1\n")
    _git_in_repo(repo, "add", "Data/paco2_public_prior.csv")
    _git_in_repo(repo, "commit", "-qm", "Add historical restricted fixture")
    _git_in_repo(repo, "tag", "ancestor-with-restricted-blob")
    restricted.unlink()
    _git_in_repo(repo, "add", "-u")
    _git_in_repo(repo, "commit", "-qm", "Delete historical restricted fixture")

    _git_in_repo(repo, "switch", "-qc", "tag-only")
    tag_only = repo / "artifacts" / "inference_demo.md"
    tag_only.parent.mkdir()
    tag_only.write_text("Subgroup priors: all (n=12)\n")
    _git_in_repo(repo, "add", "artifacts/inference_demo.md")
    _git_in_repo(repo, "commit", "-qm", "Add tag-only restricted output")
    _git_in_repo(repo, "tag", "tag-only-restricted-output")
    _git_in_repo(repo, "switch", "-q", "main")
    _git_in_repo(repo, "branch", "-D", "tag-only")

    violations = _HISTORY_CHECK_MODULE.public_history_violations(repo, _contract())

    paths = {violation.relative_path for violation in violations}
    assert "Data/paco2_public_prior.csv" in paths
    assert "artifacts/inference_demo.md" in paths


def test_history_hash_allowlist_is_exact_path_and_content() -> None:
    contract = _contract()
    safe_path = "artifacts/manuscript_confusion_matrix.md"
    safe_content = (ROOT / safe_path).read_bytes()

    assert not any(
        rule.startswith("historical_path:")
        for rule in _HISTORY_CHECK_MODULE._blob_rule_ids(safe_path, safe_content, contract)
    )
    assert any(
        rule.startswith("historical_path:")
        for rule in _HISTORY_CHECK_MODULE._blob_rule_ids(
            safe_path,
            safe_content + b"\n",
            contract,
        )
    )


def test_canonical_agreement_artifacts_are_unchanged() -> None:
    for relative, expected_sha256 in _contract()["canonical_agreement_artifacts"].items():
        path = ROOT / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256


def test_retained_frozen_aggregates_are_hash_locked() -> None:
    contract = _contract()
    retained = contract["retained_frozen_aggregate_artifacts"]

    assert retained
    assert set(retained).isdisjoint(contract["canonical_agreement_artifacts"])
    for relative, expected_sha256 in retained.items():
        path = ROOT / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256


def test_restricted_data_provenance_template_is_complete_and_unresolved() -> None:
    template = json.loads(PROVENANCE_TEMPLATE_PATH.read_text())
    record = template["record"]

    assert template["status"] == HUMAN_REVIEW_REQUIRED
    assert {
        "dataset_name",
        "ownership",
        "irb_or_protocol",
        "consent_or_waiver",
        "dua_or_authorization",
        "extract",
        "unit_of_observation",
        "repeated_patients",
        "fields_included",
        "permissions",
        "retention",
    }.issubset(record)
    assert set(_leaf_values(record)) == {HUMAN_REVIEW_REQUIRED}


def test_conway_csv_and_xlsx_mirrors_are_semantically_equal() -> None:
    pd.testing.assert_frame_equal(
        pd.read_csv(CONWAY_CSV),
        pd.read_excel(CONWAY_XLSX),
        check_dtype=False,
        check_exact=False,
        rtol=0,
        atol=1e-12,
    )


def test_human_and_machine_governance_indexes_cross_reference_the_contract() -> None:
    governance = (ROOT / "docs" / "DATA_GOVERNANCE.md").read_text()
    readme = (ROOT / "README.md").read_text()
    provenance = (ROOT / "Data" / "PROVENANCE.md").read_text()
    decisions = (ROOT / "docs" / "DECISIONS.md").read_text()
    llms = (ROOT / "llms.txt").read_text()
    web_llms = (ROOT / "web" / "llms.txt").read_text()

    for text in (governance, readme, provenance, decisions, llms):
        assert "data_release_contract.json" in text
    assert "HUMAN REVIEW REQUIRED" in provenance
    assert "current-tree" in governance.lower()
    assert "history" in governance.lower()
    assert "likelihood-only" in web_llms
    assert "upload" in web_llms.lower()


def _leaf_values(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaf_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaf_values(item)
    else:
        yield value


def _prohibited_schema_violations(
    relative: str,
    path: Path,
    contract: dict[str, Any],
) -> list[str]:
    violations: list[str] = []
    exempt_rule_ids = set(contract["synthetic_fixture_schema_exemptions"].get(relative, []))
    for table_name, columns in _structured_column_sets(path):
        for rule in contract["prohibited_structured_schemas"]:
            if rule["id"] in exempt_rule_ids:
                continue
            if not any(relative.startswith(root) for root in rule["roots"]):
                continue
            required = {column.lower() for column in rule["required_columns"]}
            required_prefixes = {
                prefix.lower() for prefix in rule.get("required_column_prefixes", [])
            }
            prefixes_present = all(
                any(column.startswith(prefix) for column in columns) for prefix in required_prefixes
            )
            if required.issubset(columns) and prefixes_present:
                violations.append(f"{relative}: {table_name}: {rule['id']}")
    return violations


def _structured_column_sets(path: Path) -> list[tuple[str, set[str]]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8-sig") as handle:
            columns = {
                column.strip().lower()
                for column in next(csv.reader(handle, delimiter=delimiter), [])
            }
        return [("table", columns)]
    if suffix == ".xlsx":
        sheets = pd.read_excel(path, sheet_name=None, nrows=0)
        return [
            (str(sheet_name), {str(column).strip().lower() for column in frame.columns})
            for sheet_name, frame in sheets.items()
        ]
    raise AssertionError(f"Unsupported structured schema extension in contract: {suffix}")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _git_in_repo(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
