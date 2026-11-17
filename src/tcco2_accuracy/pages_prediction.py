"""Evaluate a verified public disagreement kernel with a local, fixed empirical prior."""

from io import StringIO

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp

from .core._posterior import normalize_log_weights, weighted_quantile
from .core.conditional_reporting import qualify
from .core.future_setting import LEVELS, empirical_support
from .core.paco2 import prior_distribution_from_bins, validate_paco2_prior_bins

GROUPS = {"all": "main", "icu": "icu", "ed_inp": "arf", "pft": "lft"}


THRESHOLDS = (45, 52)


def evidence_masks(support, quantiles, thresholds):
    return np.array(
        [support >= k for k in thresholds]
        + [m for q in quantiles for m in (support < q, support <= q)]
    )


def resolved_summaries(a, quantiles, coarse, masks, ratios, errors, *, margin=0):
    """Check each tail separately, and all three discrete endpoints as one interval."""
    n = len(masks) - 2 * len(LEVELS)
    tails = []
    for mask, p, se in zip(masks[:n], ratios[:n], errors[:n], strict=True):
        if mask.all() or not mask.any():
            tails.append(float(mask.all()))
        elif se <= 0.00025 and abs(p - coarse[mask].sum()) <= 0.0001:
            tails.append(float(np.clip(p, 0, 1)))
        else:
            tails.append(None)
    resolved = np.array_equal(quantiles, weighted_quantile(a, coarse, LEVELS)) and all(
        ratios[j] + 2 * errors[j] + margin < level <= ratios[j + 1] - 2 * errors[j + 1] - margin
        for j, level in zip(range(n, len(masks), 2), LEVELS, strict=True)
    )
    return quantiles.tolist() if resolved else None, tails


def conditional(kernel, target, prior_csv, tcco2, threshold):
    if GROUPS.get(target) != str(kernel["group"]):
        raise ValueError("No matched fit for this setting.")
    if not np.isfinite([tcco2, threshold]).all() or min(tcco2, threshold) <= 0:
        raise ValueError("Enter positive finite CO₂ pressures.")
    prior = validate_paco2_prior_bins(pd.read_csv(StringIO(prior_csv)))
    a, w = empirical_support(*prior_distribution_from_bins(prior, target))
    offsets = a - tcco2
    grid = kernel["x"]
    if offsets.min() < grid[0] or offsets.max() > grid[-1]:
        raise ValueError(
            f"The verified arterial-minus-transcutaneous range is {grid[0]:g} to {grid[-1]:g} mmHg. Use the offline calculation for this prior."
        )
    logw = np.log(w)
    loglike = CubicSpline(grid, kernel["logpdf"], axis=1)(offsets)
    logjoint = loglike + logw
    mass, coarse = map(normalize_log_weights, logjoint)
    thresholds = list(dict.fromkeys([*THRESHOLDS, threshold]))
    evidence = logsumexp(logjoint[0])

    def batches():
        for start in range(0, len(kernel["batch_logpdf"]), 32):
            block = kernel["batch_logpdf"][start : start + 32]
            spline = CubicSpline(grid, block, axis=1)
            yield np.exp(spline(offsets) + logw - evidence)

    interval, tails, report, errors = qualify(
        a, mass, coarse, thresholds, batches(), interpolation_error=1e-5
    )
    idx = thresholds.index(threshold)
    return dict(
        quantiles=interval,
        report=report,
        probability=tails[idx],
        tails=tails[:2],
        se=float(errors[idx]),
        support=a.tolist(),
        mass=mass.tolist(),
        prior=w.tolist(),
        likelihood=np.exp(loglike[0] - loglike[0].max()).tolist(),
    )
