"""Run all workflow stages and regenerate artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from tcco2_accuracy.workflows import bootstrap, infer, meta, paco2, sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TcCO2 accuracy workflows and write artifacts.")
    parser.add_argument("--seed", type=int, default=202401, help="Seed for bootstrap and simulations.")
    parser.add_argument("--out", type=Path, default=Path("artifacts"), help="Output directory.")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap draws per subgroup.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("analytic", "monte_carlo"),
        default="analytic",
        help="Forward simulation mode.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Repo root or data file path override.",
    )
    return parser.parse_args()


def resolve_input_paths(input_path: Path | None) -> tuple[Path | None, Path | None]:
    if input_path is None:
        return None, None
    if input_path.is_dir():
        conway_path = input_path / "Conway Meta" / "data.dta"
        paco2_path = input_path / "Data" / "In Silico TCCO2 Database.dta"
        _validate_path(conway_path, "Conway meta-analysis data")
        _validate_path(paco2_path, "In-silico PaCO2 data")
        return conway_path, paco2_path
    if input_path.is_file():
        if input_path.name == "data.dta":
            _validate_path(input_path, "Conway meta-analysis data")
            return input_path, None
        _validate_path(input_path, "In-silico PaCO2 data")
        return None, input_path
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _validate_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} at {path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    conway_path, paco2_path = resolve_input_paths(args.input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta.run_meta_checks(conway_path=conway_path, out_dir=out_dir)
    bootstrap_result = bootstrap.run_bootstrap(
        n_boot=args.n_boot,
        seed=args.seed,
        conway_path=conway_path,
        out_dir=out_dir,
    )
    paco2_result = paco2.run_paco2_summary(paco2_path=paco2_path, out_dir=out_dir)
    sim.run_forward_simulation_summary(
        params=bootstrap_result.draws,
        paco2_data=paco2_result.data,
        mode=args.mode,
        seed=args.seed,
        out_dir=out_dir,
    )
    infer.run_inference_demo(
        params=bootstrap_result.draws,
        paco2_data=paco2_result.data,
        seed=args.seed,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
