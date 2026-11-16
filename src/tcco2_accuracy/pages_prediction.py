"""Evaluate a verified public disagreement kernel with a local, fixed empirical prior."""

from io import StringIO

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp

from .core._posterior import normalize_log_weights, weighted_quantile
from .core.future_setting import LEVELS, empirical_support
from .core.paco2 import prior_distribution_from_bins, validate_paco2_prior_bins

GROUPS = {"all": "main", "icu": "icu", "ed_inp": "arf", "pft": "lft"}


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
    logjoint = CubicSpline(grid, kernel["logpdf"], axis=1)(offsets) + logw
    mass, coarse = map(normalize_log_weights, logjoint)
    quantiles = weighted_quantile(a, mass, LEVELS)
    masks = [a >= threshold]
    for q in quantiles:
        masks.extend([a < q, a <= q])
    masks = np.asarray(masks)
    ratios = masks @ mass
    if abs(ratios[0] - coarse[masks[0]].sum()) > 1e-4 or not np.array_equal(
        quantiles, weighted_quantile(a, coarse, LEVELS)
    ):
        raise ValueError("Quadrature precision is insufficient for this prior.")
    # Sum over the support before estimating the ratio SE from contiguous MCMC batches.
    # The all-draw denominator includes any incomplete final batch in each chain.
    evidence = logsumexp(logjoint[0])
    batch_logs = kernel["batch_logpdf"]
    deviations = []
    for block in np.array_split(batch_logs, range(32, len(batch_logs), 32)):
        spline = CubicSpline(grid, block, axis=1)
        residual = np.zeros((len(block), len(ratios)))
        for start in range(0, len(a), 512):
            sl = slice(start, start + 512)
            joint = np.exp(spline(offsets[sl]) + logw[sl] - evidence)
            residual += joint @ (masks[:, sl].T - ratios)
        deviations.extend(residual)
    errors = np.std(deviations, axis=0, ddof=1) / np.sqrt(len(deviations))
    # Include interpolation margin when resolving a discrete support point.
    if errors[0] > 0.00025 or any(
        ratios[j] + 2 * errors[j] + 1e-5 >= level
        or ratios[j + 1] - 2 * errors[j + 1] - 1e-5 < level
        for j, level in zip((1, 3, 5), LEVELS, strict=True)
    ):
        raise ValueError(
            "Posterior simulation precision does not resolve this result for the supplied prior."
        )
    return dict(
        quantiles=quantiles.tolist(),
        probability=float(np.clip(ratios[0], 0, 1)),
        se=float(errors[0]),
        support=a.tolist(),
        mass=mass.tolist(),
    )
