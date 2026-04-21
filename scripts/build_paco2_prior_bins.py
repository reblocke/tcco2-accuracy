"""Build public or local binned PaCO2 prior distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

from tcco2_accuracy.data import (
    PACO2_PUBLIC_PRIOR_PATH,
    build_paco2_prior_bins,
    load_paco2_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build binned PaCO2 priors.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="In-silico PaCO2 .dta source. Defaults to package path/fallback lookup.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACO2_PUBLIC_PRIOR_PATH,
        help="Output CSV for the binned prior.",
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=None,
        help="Optional XLSX output path.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=1.0,
        help="Bin width for PaCO2 values (mmHg).",
    )
    parser.add_argument(
        "--include-counts",
        action="store_true",
        help="Write exact count columns for restricted local use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_paco2_distribution(args.input)
    # build_paco2_prior_bins pools subgroups into "all" using subgroup sample sizes.
    prior_bins = build_paco2_prior_bins(data, bin_width=float(args.bin_width))
    prior_bins = prior_bins.sort_values(["group", "paco2_bin"]).reset_index(drop=True)
    if not args.include_counts:
        prior_bins = prior_bins[["group", "paco2_bin", "weight"]]

    # Store a compact public prior so the static app can run without the full .dta.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prior_bins.to_csv(args.output, index=False)

    if args.xlsx is not None:
        prior_bins.to_excel(args.xlsx, index=False)


if __name__ == "__main__":
    main()
