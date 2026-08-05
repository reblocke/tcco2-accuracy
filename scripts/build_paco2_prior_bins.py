"""Build private/scratch binned PaCO2 prior distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

from tcco2_accuracy.data import (
    build_paco2_prior_bins,
    load_paco2_distribution,
)
from tcco2_accuracy.workflows._private_output import require_private_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build binned PaCO2 priors.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Restricted in-silico PaCO2 .dta source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Private/scratch output CSV for the binned prior.",
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
    output_path = require_private_output_path(args.output)
    xlsx_path = require_private_output_path(args.xlsx) if args.xlsx is not None else None
    data = load_paco2_distribution(args.input)
    # build_paco2_prior_bins pools subgroups into "all" using subgroup sample sizes.
    prior_bins = build_paco2_prior_bins(data, bin_width=float(args.bin_width))
    prior_bins = prior_bins.sort_values(["group", "paco2_bin"]).reset_index(drop=True)
    if not args.include_counts:
        prior_bins = prior_bins[["group", "paco2_bin", "weight"]]

    # Normalized weights remain restricted-derived even when exact counts are omitted.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prior_bins.to_csv(output_path, index=False)

    if xlsx_path is not None:
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        prior_bins.to_excel(xlsx_path, index=False)


if __name__ == "__main__":
    main()
