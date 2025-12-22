"""Regenerate TcCO2 accuracy artifacts from canonical study inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from tcco2_accuracy.bootstrap import BOOTSTRAP_MODES
from tcco2_accuracy.simulation import DEFAULT_CLASSIFICATION_THRESHOLDS
from tcco2_accuracy.workflows import bootstrap, conditional, infer, meta, paco2, sim


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


def main() -> None:
    args = parse_args()
    thresholds = _parse_thresholds(args.thresholds)
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

    _ = meta_result


if __name__ == "__main__":
    main()
