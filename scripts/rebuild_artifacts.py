"""Regenerate TcCO2 accuracy artifacts from canonical study inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from tcco2_accuracy.bootstrap import BOOTSTRAP_MODES
from tcco2_accuracy.simulation import DEFAULT_CLASSIFICATION_THRESHOLDS
from tcco2_accuracy.workflows import bootstrap, conditional, infer, manuscript, meta, paco2, sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild TcCO2 accuracy artifacts.")
    parser.add_argument("--seed", type=int, default=202401, help="Random seed.")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap draws per subgroup.")
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
        help="Canonical Conway study table (CSV/XLSX) override.",
    )
    parser.add_argument(
        "--bootstrap-mode",
        type=str,
        choices=BOOTSTRAP_MODES,
        default="cluster_plus_withinstudy",
        help="Bootstrap uncertainty mode.",
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
        help="Output directory for artifacts.",
    )
    return parser.parse_args()


def _parse_thresholds(raw: str | None) -> list[float]:
    if raw is None or not raw.strip():
        return list(DEFAULT_CLASSIFICATION_THRESHOLDS)
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_float_list(raw: str | None, default: Sequence[float]) -> list[float]:
    if raw is None or not raw.strip():
        return list(default)
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    args = parse_args()
    thresholds = _parse_thresholds(args.thresholds)
    tcco2_values = _parse_float_list(args.tcco2_values, (35.0, 40.0, 45.0, 50.0, 55.0))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_result = meta.run_meta_checks(conway_path=args.input_study_table, out_dir=out_dir)
    bootstrap_result = bootstrap.run_bootstrap(
        n_boot=args.n_boot,
        seed=args.seed,
        conway_path=args.input_study_table,
        bootstrap_mode=args.bootstrap_mode,
        out_dir=out_dir,
    )
    paco2_result = paco2.run_paco2_summary(out_dir=out_dir)
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


if __name__ == "__main__":
    main()
