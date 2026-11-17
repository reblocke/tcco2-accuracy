"""Evaluate a local fixed prior on the full accepted fit using conditional reporting v2."""

import argparse
from math import isqrt
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from tcco2_accuracy.core.conditional_reporting import display_range, qualify
from tcco2_accuracy.core.future_setting import PredictiveKernel, empirical_support
from tcco2_accuracy.core.paco2 import prior_distribution_from_bins
from tcco2_accuracy.data import load_paco2_prior_bins
from tcco2_accuracy.workflows.future_setting_inverse import load_candidate_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("target", choices=["all", "icu", "ed_inp", "pft"])
    parser.add_argument("tc", type=float)
    parser.add_argument("--threshold", type=float, default=45)
    args = parser.parse_args()
    if not np.isfinite([args.tc, args.threshold]).all() or min(args.tc, args.threshold) <= 0:
        raise ValueError("Pressures must be positive and finite")
    kernel, _ = load_candidate_kernel(args.run, args.target)
    a, w = empirical_support(
        *prior_distribution_from_bins(load_paco2_prior_bins(args.prior), args.target)
    )
    size = isqrt(kernel.theta.shape[1])
    chunks = [chain[i : i + size] for chain in kernel.theta for i in range(0, len(chain), size)]
    lengths = np.asarray([len(c) for c in chunks])
    means = []
    for order in (kernel.order // 2, kernel.order):
        logs = np.stack(
            [PredictiveKernel(c[None, :, :], order=order).logpdf(a - args.tc) for c in chunks]
        )
        means.append(logsumexp(logs + np.log(lengths[:, None]), axis=0) - np.log(lengths.sum()))
    joint = np.stack(means) + np.log(w)
    norm = logsumexp(joint, axis=1)
    coarse, mass = np.exp(joint - norm[:, None])
    blocks = np.exp(logs[lengths == size] + np.log(w) - norm[1])
    thresholds = list(dict.fromkeys([45, 52, args.threshold]))
    _, _, report, _ = qualify(a, mass, coarse, thresholds, [blocks])
    print("PaCO₂ lower, median, upper (mmHg)", *map(display_range, report["quantiles"]))
    for threshold, bracket in zip(thresholds, report["probabilities"], strict=True):
        print(f"P(PaCO₂ ≥ {threshold:g})", display_range(bracket, percent=True))


if __name__ == "__main__":
    main()
