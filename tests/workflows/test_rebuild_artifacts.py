from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "rebuild_artifacts.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("rebuild_artifacts_for_tests", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _public_args(
    out_dir: Path,
    *,
    seed: int = 123,
    n_boot: int = 4,
    input_study_table: Path | None = None,
    bootstrap_mode: str = "cluster_plus_withinstudy",
) -> list[str]:
    return [
        "--profile",
        "public-agreement",
        "--input-study-table",
        str(input_study_table or ROOT / "Data" / "conway_studies.csv"),
        "--out",
        str(out_dir),
        "--seed",
        str(seed),
        "--n-boot",
        str(n_boot),
        "--bootstrap-mode",
        bootstrap_mode,
    ]


def _assert_numeric_frames_close(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rtol=0,
        atol=1e-12,
    )


def _assert_numeric_csv_close(actual: Path, expected: Path) -> None:
    _assert_numeric_frames_close(pd.read_csv(actual), pd.read_csv(expected))


def test_profile_is_required() -> None:
    script = _load_script()

    with pytest.raises(SystemExit):
        script.parse_args([])


def test_public_agreement_is_allowlisted_deterministic_and_never_loads_paco2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    out_dir1 = tmp_path / "run1"
    out_dir2 = tmp_path / "run2"
    out_dir1.mkdir()
    frozen = out_dir1 / "frozen_downstream.md"
    frozen.write_text("do not rewrite\n")

    def _deny_restricted_load(*args, **kwargs):
        raise AssertionError("public-agreement attempted to load restricted PaCO2 data")

    monkeypatch.setattr(script.paco2, "run_paco2_summary", _deny_restricted_load)
    script.main(_public_args(out_dir1))
    script.main(_public_args(out_dir2))

    assert frozen.read_text() == "do not rewrite\n"
    assert {path.name for path in out_dir1.iterdir()} == {
        *script.PUBLIC_AGREEMENT_ARTIFACTS,
        frozen.name,
    }
    for filename in script.PUBLIC_AGREEMENT_ARTIFACTS:
        assert (out_dir1 / filename).read_bytes() == (out_dir2 / filename).read_bytes()

    params = pd.read_csv(out_dir1 / "bootstrap_params.csv")
    assert params["agreement_method_version"].unique().tolist() == [script.AGREEMENT_METHOD_VERSION]
    assert params["results_status"].unique().tolist() == [script.RESULTS_STATUS]
    parameter_summary = pd.read_csv(out_dir1 / "manuscript_parameters.csv")
    assert parameter_summary["agreement_method_version"].unique().tolist() == [
        script.AGREEMENT_METHOD_VERSION
    ]
    assert parameter_summary["results_status"].unique().tolist() == [script.RESULTS_STATUS]

    bootstrap_markdown = (out_dir1 / "bootstrap_summary.md").read_text()
    assert "corrected analytic CI" in bootstrap_markdown
    assert "Conway CI shown as reported" not in bootstrap_markdown
    assert script.AGREEMENT_METHOD_VERSION in bootstrap_markdown
    meta_markdown = (out_dir1 / "meta_loa_check.md").read_text()
    assert "Corrected versus published/legacy comparator" in meta_markdown
    assert "| Metric | Corrected | Published/legacy | Delta |" in meta_markdown
    assert script.PUBLISHED_CONWAY_TABLE1_SOURCE in meta_markdown
    assert str(ROOT) not in meta_markdown


def test_canonical_public_agreement_artifacts_match_same_seed_rebuild(tmp_path: Path) -> None:
    script = _load_script()
    rebuilt = tmp_path / "rebuilt"
    script.main(_public_args(rebuilt, seed=202401, n_boot=1000))

    numeric_csvs = {"bootstrap_params.csv", "manuscript_parameters.csv"}
    for filename in script.PUBLIC_AGREEMENT_ARTIFACTS:
        rebuilt_path = rebuilt / filename
        canonical_path = script.CANONICAL_ARTIFACTS_DIR / filename
        if filename in numeric_csvs:
            _assert_numeric_csv_close(rebuilt_path, canonical_path)
        else:
            assert rebuilt_path.read_bytes() == canonical_path.read_bytes(), (
                f"Canonical artifact is stale: {filename}"
            )


def test_numeric_artifact_comparison_uses_absolute_tolerance_only() -> None:
    expected = pd.DataFrame({"group": ["main"], "tau2": [1.0]})
    within_tolerance = expected.assign(tau2=[1.0 + 5e-13])
    above_tolerance = expected.assign(tau2=[1.0 + 2e-12])

    _assert_numeric_frames_close(within_tolerance, expected)
    with pytest.raises(AssertionError):
        _assert_numeric_frames_close(above_tolerance, expected)


def test_canonical_public_promotion_accepts_exact_contract_and_resolved_aliases(
    tmp_path: Path,
) -> None:
    script = _load_script()
    input_link = tmp_path / "conway.csv"
    output_link = tmp_path / "canonical-artifacts"
    input_link.symlink_to(script.DEFAULT_PUBLIC_CONWAY_PATH)
    output_link.symlink_to(script.CANONICAL_ARTIFACTS_DIR, target_is_directory=True)
    args = script.parse_args(
        _public_args(
            output_link,
            seed=script.CANONICAL_PUBLIC_SEED,
            n_boot=script.CANONICAL_PUBLIC_N_BOOT,
            input_study_table=input_link,
            bootstrap_mode=script.CANONICAL_PUBLIC_BOOTSTRAP_MODE,
        )
    )

    script._validate_canonical_public_promotion(
        args,
        conway_path=input_link,
        out_dir=output_link,
    )


def test_status_manifest_matches_canonical_promotion_contract() -> None:
    script = _load_script()
    status = (script.CANONICAL_ARTIFACTS_DIR / "STATUS.md").read_text()

    expected_contract = {
        "profile": "public-agreement",
        "output": "artifacts/",
        "input-study-table": "Data/conway_studies.csv",
        "seed": str(script.CANONICAL_PUBLIC_SEED),
        "n-boot": str(script.CANONICAL_PUBLIC_N_BOOT),
        "bootstrap-mode": script.CANONICAL_PUBLIC_BOOTSTRAP_MODE,
    }
    for key, value in expected_contract.items():
        assert f"- {key}: `{value}`" in status
    for filename in script.PUBLIC_AGREEMENT_ARTIFACTS:
        assert f"`{filename}`" in status


@pytest.mark.parametrize(
    ("seed", "n_boot", "bootstrap_mode", "mismatch"),
    [
        (7, 1000, "cluster_plus_withinstudy", "--seed=7"),
        (202401, 5, "cluster_plus_withinstudy", "--n-boot=5"),
        (202401, 1000, "cluster_only", "--bootstrap-mode=cluster_only"),
    ],
)
def test_canonical_public_promotion_rejects_noncanonical_settings_before_workflows(
    seed: int,
    n_boot: int,
    bootstrap_mode: str,
    mismatch: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    protected_paths = [
        script.CANONICAL_ARTIFACTS_DIR / filename for filename in script.PUBLIC_AGREEMENT_ARTIFACTS
    ] + [script.CANONICAL_ARTIFACTS_DIR / "STATUS.md"]
    before = {path: path.read_bytes() for path in protected_paths}

    def _deny_workflow(*args, **kwargs):
        raise AssertionError("canonical promotion guard ran too late")

    monkeypatch.setattr(script.meta, "run_meta_checks", _deny_workflow)
    with pytest.raises(script.ArtifactProfileError, match=mismatch):
        script.main(
            _public_args(
                script.CANONICAL_ARTIFACTS_DIR,
                seed=seed,
                n_boot=n_boot,
                bootstrap_mode=bootstrap_mode,
            )
        )

    assert {path: path.read_bytes() for path in protected_paths} == before


def test_canonical_public_promotion_rejects_custom_input_before_workflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    custom_input = tmp_path / "custom-conway.csv"
    pd.read_csv(script.DEFAULT_PUBLIC_CONWAY_PATH).to_csv(custom_input, index=False)
    protected_paths = [
        script.CANONICAL_ARTIFACTS_DIR / filename for filename in script.PUBLIC_AGREEMENT_ARTIFACTS
    ] + [script.CANONICAL_ARTIFACTS_DIR / "STATUS.md"]
    before = {path: path.read_bytes() for path in protected_paths}

    def _deny_workflow(*args, **kwargs):
        raise AssertionError("canonical promotion guard ran too late")

    monkeypatch.setattr(script.meta, "run_meta_checks", _deny_workflow)
    with pytest.raises(script.ArtifactProfileError, match="--input-study-table=custom-conway.csv"):
        script.main(
            _public_args(
                script.CANONICAL_ARTIFACTS_DIR,
                seed=script.CANONICAL_PUBLIC_SEED,
                n_boot=script.CANONICAL_PUBLIC_N_BOOT,
                input_study_table=custom_input,
                bootstrap_mode=script.CANONICAL_PUBLIC_BOOTSTRAP_MODE,
            )
        )

    assert {path: path.read_bytes() for path in protected_paths} == before


def test_public_agreement_handles_homogeneous_three_study_input(tmp_path: Path) -> None:
    script = _load_script()
    study_table = tmp_path / "homogeneous_conway_studies.csv"
    pd.DataFrame(
        {
            "study_id": ["equal_a", "equal_b", "equal_c"],
            "bias": [0.5, 0.5, 0.5],
            "sd": [2.0, 2.0, 2.0],
            "s2": [4.0, 4.0, 4.0],
            "n_pairs": [20.0, 20.0, 20.0],
            "n_participants": [20.0, 20.0, 20.0],
            "c": [1.0, 1.0, 1.0],
            "is_icu": [1, 1, 1],
            "is_arf": [1, 1, 1],
            "is_lft": [1, 1, 1],
        }
    ).to_csv(study_table, index=False)
    out_dir = tmp_path / "artifacts"

    script.main(
        _public_args(
            out_dir,
            seed=321,
            n_boot=5,
            input_study_table=study_table,
        )
    )

    assert {path.name for path in out_dir.iterdir()} == script.PUBLIC_AGREEMENT_ARTIFACTS
    params = pd.read_csv(out_dir / "bootstrap_params.csv")
    tau2 = params["tau2"].to_numpy(dtype=float)
    assert tau2.size == 4 * 5
    assert np.isfinite(tau2).all()
    assert (tau2 >= 0).all()


def test_public_agreement_rejects_restricted_arguments_before_writing(tmp_path: Path) -> None:
    script = _load_script()
    out_dir = tmp_path / "out"

    with pytest.raises(script.ArtifactProfileError, match="rejects --paco2-path"):
        script.main(
            [
                "--profile",
                "public-agreement",
                "--paco2-path",
                str(tmp_path / "restricted.dta"),
                "--out",
                str(out_dir),
            ]
        )

    assert not out_dir.exists()


def test_public_agreement_rejects_dta_study_input_before_writing(tmp_path: Path) -> None:
    script = _load_script()
    out_dir = tmp_path / "out"
    dta_path = tmp_path / "study.dta"
    dta_path.write_bytes(b"not a public study table")

    with pytest.raises(script.ArtifactProfileError, match="CSV/XLSX"):
        script.main(
            [
                "--profile",
                "public-agreement",
                "--input-study-table",
                str(dta_path),
                "--out",
                str(out_dir),
            ]
        )

    assert not out_dir.exists()


def test_public_agreement_does_not_promote_partial_or_unexpected_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    out_dir = tmp_path / "out"
    original = script.manuscript.run_manuscript_parameters

    def _write_unexpected(*args, **kwargs):
        result = original(*args, **kwargs)
        Path(kwargs["out_dir"], "unexpected.csv").write_text("unexpected\n")
        return result

    monkeypatch.setattr(script.manuscript, "run_manuscript_parameters", _write_unexpected)

    with pytest.raises(script.ArtifactProfileError, match="unexpected=.*unexpected.csv"):
        script.main(_public_args(out_dir))

    assert not out_dir.exists()


def test_full_requires_existing_explicit_paco2_and_scratch_or_external_output(
    tmp_path: Path,
) -> None:
    script = _load_script()

    with pytest.raises(script.ArtifactProfileError, match="explicit --paco2-path"):
        script.main(["--profile", "full", "--out", str(tmp_path / "scratch")])

    with pytest.raises(script.ArtifactProfileError, match="not found"):
        script.main(
            [
                "--profile",
                "full",
                "--paco2-path",
                str(tmp_path / "missing.dta"),
                "--out",
                str(tmp_path / "scratch"),
            ]
        )

    paco2_path = tmp_path / "restricted.dta"
    paco2_path.write_bytes(b"test path is validated before workflow execution")
    blocked_outputs = (
        script.REPO_ROOT,
        script.REPO_ROOT / "docs",
        script.CANONICAL_ARTIFACTS_DIR,
        script.CANONICAL_ARTIFACTS_DIR / "private",
    )
    for blocked_output in blocked_outputs:
        with pytest.raises(script.ArtifactProfileError, match="requires in-repo output"):
            script.main(
                [
                    "--profile",
                    "full",
                    "--paco2-path",
                    str(paco2_path),
                    "--out",
                    str(blocked_output),
                ]
            )

    script._validate_full_output_path(script.REPO_ROOT / ".pytest_tmp" / "full")
    script._validate_full_output_path(script.REPO_ROOT / ".tmp" / "full")
    script._validate_full_output_path(tmp_path / "external-private")
