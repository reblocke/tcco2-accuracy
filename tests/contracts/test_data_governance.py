from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_PRIOR = ROOT / "Data" / "paco2_public_prior.csv"
RESTRICTED_EXACT_OUTPUTS = (
    Path("Data/paco2_prior_bins.csv"),
    Path("artifacts/figure_paco2_distribution_bins.csv"),
)


def test_public_prior_csv_exposes_weights_not_counts() -> None:
    prior = pd.read_csv(PUBLIC_PRIOR)

    assert list(prior.columns) == ["group", "paco2_bin", "weight"]
    forbidden_columns = {"count", "counts", "n", "raw_count", "cell_n", "density"}
    assert forbidden_columns.isdisjoint({column.lower() for column in prior.columns})


def test_exact_count_outputs_are_ignored_and_untracked() -> None:
    for relative_path in RESTRICTED_EXACT_OUTPUTS:
        assert _git("ls-files", "--", str(relative_path)).stdout.strip() == ""
        assert _git("check-ignore", str(relative_path)).returncode == 0


def test_data_governance_docs_cover_public_and_restricted_boundaries() -> None:
    governance = (ROOT / "docs" / "DATA_GOVERNANCE.md").read_text()
    readme = (ROOT / "README.md").read_text()
    provenance = (ROOT / "Data" / "PROVENANCE.md").read_text()
    manuscript_outputs = (ROOT / "docs" / "MANUSCRIPT_OUTPUTS.md").read_text()
    decisions = (ROOT / "docs" / "DECISIONS.md").read_text()

    assert "Data/paco2_public_prior.csv" in governance
    assert "Data/paco2_prior_bins.csv" in governance
    assert "artifacts/figure_paco2_distribution_bins.csv" in governance
    assert ".pytest_tmp/" in governance
    assert ".tmp/" in governance
    assert "docs/DATA_GOVERNANCE.md" in readme
    assert "Data/PROVENANCE.md" in readme
    assert "weight-only" in provenance
    assert "restricted" in provenance
    assert "private manuscript" in manuscript_outputs
    assert "workspace" in manuscript_outputs
    assert "Current-tree remediation" in decisions


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
