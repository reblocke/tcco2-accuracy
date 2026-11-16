"""Evaluate a local binned prior on the full, uncompressed accepted fit."""

import argparse
from pathlib import Path

import numpy as np

from tcco2_accuracy.core.future_setting import PredictiveKernel, empirical_support, inverse_summary
from tcco2_accuracy.core.paco2 import prior_distribution_from_bins
from tcco2_accuracy.data import load_paco2_prior_bins
from tcco2_accuracy.workflows.future_setting_inverse import (
    _empirical_precision,
    load_candidate_kernel,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("target", choices=["all", "icu", "ed_inp", "pft"])
    parser.add_argument("tc", type=float)
    parser.add_argument("--threshold", type=float, default=45)
    args = parser.parse_args()
    kernel, _ = load_candidate_kernel(args.run, args.target)
    a, w = empirical_support(
        *prior_distribution_from_bins(load_paco2_prior_bins(args.prior), args.target)
    )
    result = inverse_summary(kernel, a, w, args.tc, threshold=args.threshold)
    coarse = inverse_summary(
        PredictiveKernel(kernel.theta, order=kernel.order // 2),
        a,
        w,
        args.tc,
        threshold=args.threshold,
    )
    se, resolved = _empirical_precision(kernel, a, w, args.tc, result, threshold=args.threshold)
    if (
        not resolved
        or se > 0.00025
        or abs(coarse["probability_ge"] - result["probability_ge"]) > 0.0001
        or not np.array_equal(coarse["quantiles"], result["quantiles"])
    ):
        raise ValueError("Numerical precision insufficient for this prior.")
    print("PaCO₂ lower, median, upper (mmHg):", *result["quantiles"])
    print(f"P(PaCO₂ ≥ {args.threshold:g}):", result["probability_ge"])


if __name__ == "__main__":
    main()
