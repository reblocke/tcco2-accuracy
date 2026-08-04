"""Regenerate TcCO2 accuracy artifacts from canonical study inputs."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence

from tcco2_accuracy.bootstrap import BOOTSTRAP_MODES
from tcco2_accuracy.core.conway_meta import AGREEMENT_METHOD_VERSION, RESULTS_STATUS
from tcco2_accuracy.simulation import DEFAULT_CLASSIFICATION_THRESHOLDS
from tcco2_accuracy.workflows import bootstrap, conditional, infer, manuscript, meta, paco2, sim

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ARTIFACTS_DIR = REPO_ROOT / "artifacts"
IN_REPO_FULL_OUTPUT_ROOTS = (REPO_ROOT / ".pytest_tmp", REPO_ROOT / ".tmp")
DEFAULT_PUBLIC_CONWAY_PATH = REPO_ROOT / "Data" / "conway_studies.csv"
CANONICAL_PUBLIC_SEED = 202401
CANONICAL_PUBLIC_N_BOOT = 1000
CANONICAL_PUBLIC_BOOTSTRAP_MODE = "cluster_plus_withinstudy"
PUBLISHED_CONWAY_TABLE1_PATH = REPO_ROOT / "tests" / "fixtures" / "conway_table1.csv"
PUBLISHED_CONWAY_TABLE1_SOURCE = "tests/fixtures/conway_table1.csv"
PUBLIC_CONWAY_SUFFIXES = frozenset({".csv", ".xlsx"})
PUBLIC_AGREEMENT_ARTIFACTS = frozenset(
    {
        "bootstrap_params.csv",
        "bootstrap_summary.md",
        "manuscript_parameters.csv",
        "manuscript_parameters.md",
        "meta_loa_check.md",
    }
)


class ArtifactProfileError(ValueError):
    """Raised when an artifact profile would cross its data or output boundary."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild TcCO2 accuracy artifacts.")
    parser.add_argument(
        "--profile",
        choices=("public-agreement", "full"),
        required=True,
        help="Fail-closed artifact profile to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CANONICAL_PUBLIC_SEED,
        help=f"Random seed. Canonical public promotion requires {CANONICAL_PUBLIC_SEED}.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=CANONICAL_PUBLIC_N_BOOT,
        help=(
            "Bootstrap draws per subgroup. Canonical public promotion requires "
            f"{CANONICAL_PUBLIC_N_BOOT}."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Comma-separated thresholds (mmHg). Defaults to pipeline defaults.",
    )
    parser.add_argument(
        "--input-study-table",
        type=Path,
        default=None,
        help=(
            "Public Conway study table (CSV/XLSX). Canonical public promotion requires "
            "Data/conway_studies.csv; use a noncanonical output for overrides."
        ),
    )
    parser.add_argument(
        "--paco2-path",
        type=Path,
        default=None,
        help="Restricted local in-silico PaCO2 .dta source override.",
    )
    parser.add_argument(
        "--bootstrap-mode",
        type=str,
        choices=BOOTSTRAP_MODES,
        default=CANONICAL_PUBLIC_BOOTSTRAP_MODE,
        help=(
            "Bootstrap uncertainty mode. Canonical public promotion requires "
            f"{CANONICAL_PUBLIC_BOOTSTRAP_MODE}."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("analytic", "monte_carlo"),
        default="analytic",
        help="Forward simulation mode.",
    )
    parser.add_argument(
        "--true-threshold",
        type=float,
        default=45.0,
        help="True hypercapnia threshold for reporting outputs.",
    )
    parser.add_argument(
        "--two-stage-lower",
        type=float,
        default=40.0,
        help="Lower TcCO2 zone bound for two-stage strategy.",
    )
    parser.add_argument(
        "--two-stage-upper",
        type=float,
        default=50.0,
        help="Upper TcCO2 zone bound for two-stage strategy.",
    )
    parser.add_argument(
        "--tcco2-values",
        type=str,
        default="35,40,45,50,55",
        help="Comma-separated TcCO2 values for prediction interval table.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts"),
        help=(
            "Output directory. The canonical artifacts/ destination accepts only the locked "
            "public promotion contract; use scratch output for custom runs."
        ),
    )
    return parser.parse_args(argv)


def _parse_thresholds(raw: str | None) -> list[float]:
    if raw is None or not raw.strip():
        return list(DEFAULT_CLASSIFICATION_THRESHOLDS)
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_float_list(raw: str | None, default: Sequence[float]) -> list[float]:
    if raw is None or not raw.strip():
        return list(default)
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.profile == "public-agreement":
        _run_public_agreement(args)
        return
    _run_full(args)


def _run_public_agreement(args: argparse.Namespace) -> None:
    if args.paco2_path is not None:
        raise ArtifactProfileError(
            "--profile public-agreement rejects --paco2-path and never loads restricted PaCO2 data"
        )

    conway_path = Path(args.input_study_table or DEFAULT_PUBLIC_CONWAY_PATH)
    if conway_path.suffix.lower() not in PUBLIC_CONWAY_SUFFIXES:
        raise ArtifactProfileError(
            "--profile public-agreement accepts only public Conway CSV/XLSX study tables"
        )
    if not conway_path.is_file():
        raise ArtifactProfileError(f"Public Conway study table not found: {conway_path}")

    out_dir = Path(args.out)
    _validate_canonical_public_promotion(args, conway_path=conway_path, out_dir=out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".tcco2-public-agreement-", dir=out_dir.parent
    ) as temp_dir:
        staged_dir = Path(temp_dir)
        meta.run_meta_checks(
            conway_path=conway_path,
            source_label=_portable_source_label(conway_path),
            published_comparator_path=PUBLISHED_CONWAY_TABLE1_PATH,
            published_comparator_source=PUBLISHED_CONWAY_TABLE1_SOURCE,
            out_dir=staged_dir,
        )
        bootstrap_result = bootstrap.run_bootstrap(
            n_boot=args.n_boot,
            seed=args.seed,
            conway_path=conway_path,
            bootstrap_mode=args.bootstrap_mode,
            out_dir=staged_dir,
        )
        _require_current_metadata(bootstrap_result.draws)
        manuscript.run_manuscript_parameters(params=bootstrap_result.draws, out_dir=staged_dir)
        _promote_public_agreement_artifacts(staged_dir, out_dir)


def _run_full(args: argparse.Namespace) -> None:
    if args.paco2_path is None:
        raise ArtifactProfileError("--profile full requires an explicit --paco2-path")
    paco2_path = Path(args.paco2_path)
    if not paco2_path.is_file():
        raise ArtifactProfileError(f"Restricted PaCO2 source not found: {paco2_path}")

    out_dir = Path(args.out)
    _validate_full_output_path(out_dir)

    thresholds = _parse_thresholds(args.thresholds)
    tcco2_values = _parse_float_list(args.tcco2_values, (35.0, 40.0, 45.0, 50.0, 55.0))
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_result = meta.run_meta_checks(conway_path=args.input_study_table, out_dir=out_dir)
    bootstrap_result = bootstrap.run_bootstrap(
        n_boot=args.n_boot,
        seed=args.seed,
        conway_path=args.input_study_table,
        bootstrap_mode=args.bootstrap_mode,
        out_dir=out_dir,
    )
    paco2_result = paco2.run_paco2_summary(paco2_path=paco2_path, out_dir=out_dir)
    sim.run_forward_simulation_summary(
        params=bootstrap_result.draws,
        paco2_data=paco2_result.data,
        thresholds=thresholds,
        mode=args.mode,
        seed=args.seed,
        out_dir=out_dir,
    )
    infer.run_inference_demo(
        params=bootstrap_result.draws,
        paco2_data=paco2_result.data,
        thresholds=thresholds,
        seed=args.seed,
        out_dir=out_dir,
    )
    for threshold in thresholds:
        conditional.run_conditional_classification(
            params=bootstrap_result.draws,
            paco2_data=paco2_result.data,
            threshold=threshold,
            seed=args.seed,
            bootstrap_mode=args.bootstrap_mode,
            out_dir=out_dir,
        )
    manuscript.run_manuscript_outputs(
        params=bootstrap_result.draws,
        paco2_data=paco2_result.data,
        thresholds=thresholds,
        true_threshold=args.true_threshold,
        two_stage_lower=args.two_stage_lower,
        two_stage_upper=args.two_stage_upper,
        tcco2_values=tcco2_values,
        mode=args.mode,
        seed=args.seed,
        out_dir=out_dir,
    )

    _ = meta_result


def _require_current_metadata(params: object) -> None:
    columns = getattr(params, "columns", ())
    for column, expected in (
        ("agreement_method_version", AGREEMENT_METHOD_VERSION),
        ("results_status", RESULTS_STATUS),
    ):
        if column not in columns:
            raise ArtifactProfileError(f"Public agreement draws are missing required {column}")
        values = getattr(params, column).dropna().astype(str).unique()
        if values.size != 1 or values[0] != expected:
            raise ArtifactProfileError(
                f"Public agreement draws require {column}={expected!r}; found {values.tolist()}"
            )


def _portable_source_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _validate_canonical_public_promotion(
    args: argparse.Namespace,
    *,
    conway_path: Path,
    out_dir: Path,
) -> None:
    """Reject noncanonical inputs before replacing repository promotion artifacts."""

    if out_dir.resolve() != CANONICAL_ARTIFACTS_DIR.resolve():
        return

    mismatches: list[str] = []
    if conway_path.resolve() != DEFAULT_PUBLIC_CONWAY_PATH.resolve():
        mismatches.append(f"--input-study-table={_portable_source_label(conway_path)}")
    if args.seed != CANONICAL_PUBLIC_SEED:
        mismatches.append(f"--seed={args.seed}")
    if args.n_boot != CANONICAL_PUBLIC_N_BOOT:
        mismatches.append(f"--n-boot={args.n_boot}")
    if args.bootstrap_mode != CANONICAL_PUBLIC_BOOTSTRAP_MODE:
        mismatches.append(f"--bootstrap-mode={args.bootstrap_mode}")

    if mismatches:
        raise ArtifactProfileError(
            "Canonical public promotion to artifacts/ requires "
            "--input-study-table=Data/conway_studies.csv, "
            f"--seed={CANONICAL_PUBLIC_SEED}, --n-boot={CANONICAL_PUBLIC_N_BOOT}, "
            f"and --bootstrap-mode={CANONICAL_PUBLIC_BOOTSTRAP_MODE}; mismatches: "
            f"{', '.join(mismatches)}. Use a noncanonical --out such as "
            ".pytest_tmp/public-agreement-candidate for custom runs."
        )


def _validate_full_output_path(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    repo_root = REPO_ROOT.resolve()
    if resolved.is_relative_to(repo_root) and not any(
        resolved.is_relative_to(root.resolve()) for root in IN_REPO_FULL_OUTPUT_ROOTS
    ):
        raise ArtifactProfileError(
            "--profile full requires in-repo output under .pytest_tmp/ or .tmp/; "
            "otherwise use an external private path"
        )


def _promote_public_agreement_artifacts(staged_dir: Path, out_dir: Path) -> None:
    staged_files = {
        path.relative_to(staged_dir).as_posix() for path in staged_dir.rglob("*") if path.is_file()
    }
    if staged_files != PUBLIC_AGREEMENT_ARTIFACTS:
        missing = sorted(PUBLIC_AGREEMENT_ARTIFACTS - staged_files)
        unexpected = sorted(staged_files - PUBLIC_AGREEMENT_ARTIFACTS)
        raise ArtifactProfileError(
            f"Public agreement artifact boundary mismatch: missing={missing}, unexpected={unexpected}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    prior_artifacts = {
        filename: (out_dir / filename).read_bytes() if (out_dir / filename).exists() else None
        for filename in sorted(PUBLIC_AGREEMENT_ARTIFACTS)
    }
    try:
        for filename in sorted(PUBLIC_AGREEMENT_ARTIFACTS):
            os.replace(staged_dir / filename, out_dir / filename)
    except BaseException as exc:
        rollback_errors = _restore_public_agreement_artifacts(
            staged_dir,
            out_dir,
            prior_artifacts,
        )
        if rollback_errors:
            exc.add_note(
                "Public agreement artifact rollback was incomplete: " + "; ".join(rollback_errors)
            )
        raise


def _restore_public_agreement_artifacts(
    staged_dir: Path,
    out_dir: Path,
    prior_artifacts: dict[str, bytes | None],
) -> list[str]:
    """Restore the complete allowlisted destination state after promotion failure."""

    rollback_errors: list[str] = []
    for filename, prior_bytes in prior_artifacts.items():
        destination = out_dir / filename
        try:
            if prior_bytes is None:
                destination.unlink(missing_ok=True)
                continue
            restore_path = staged_dir / f".rollback-{filename}"
            restore_path.write_bytes(prior_bytes)
            os.replace(restore_path, destination)
        except Exception as rollback_exc:  # pragma: no cover - requires filesystem failure
            rollback_errors.append(f"{filename}: {rollback_exc}")
    return rollback_errors


if __name__ == "__main__":
    try:
        main()
    except ArtifactProfileError as exc:
        raise SystemExit(str(exc)) from exc
