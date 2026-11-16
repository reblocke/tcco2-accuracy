"""One local PaCO₂ prediction at TcCO₂ 47, using the entire fixed empirical prior."""

import re
from math import isqrt

import numpy as np
from scipy.special import logsumexp

from ..core._posterior import normalize_log_weights, weighted_quantile
from ..core.future_setting import LEVELS, METHOD, PredictiveKernel, empirical_support
from ..pages_prediction import GROUPS, THRESHOLDS, evidence_masks, resolved_summaries

READING = 47


def display_curves(support, prior, mass):
    """Gaussian density displays (bandwidth 1 mmHg); no inference uses these curves."""
    a, w, p = (np.asarray(v, dtype=float) for v in (support, prior, mass))
    # Evaluate each occupied region on a quarter-mmHg mesh. Flat gaps between
    # regions can span arbitrary distances without allocating a grid per mmHg.
    # At nine SDs the omitted density is below 1.1e-18 per mmHg per unit mass.
    breaks = np.flatnonzero(np.diff(a) > 18) + 1
    grids = []
    size = 0
    for part in np.split(a, breaks):
        lo, hi = max(0.0, part[0] - 9), part[-1] + 9
        if hi - lo > 4096 or hi == lo:
            raise ValueError("Private display exceeds its grid resolution")
        n = int(np.ceil((hi - lo) * 4)) + 1
        size += n
        if size > 16384:
            raise ValueError("Private display exceeds its grid resolution")
        grids.append(np.linspace(lo, hi, n))
    x = np.concatenate(grids)
    if not np.all(np.diff(x) > 0):
        raise ValueError("Private display grid is not representable")
    densities = np.zeros((2, len(x)))
    for i, (left, right) in enumerate(
        zip(np.searchsorted(a, x - 9), np.searchsorted(a, x + 9, side="right"), strict=True)
    ):
        sl = slice(left, right)
        k = np.exp(-0.5 * (x[i] - a[sl]) ** 2) / np.sqrt(2 * np.pi)
        densities[:, i] = k @ w[sl], k @ p[sl]
    return dict(
        x=x.tolist(),
        prior_density=densities[0].tolist(),
        posterior_density=densities[1].tolist(),
        bandwidth=1,
    )


def _batch_logpdf(theta, order, offsets, logw, masks=None):
    """Average over every draw; retain equal chain batches for the ratio MCSE."""
    n = theta.shape[1]
    size = isqrt(n)
    total = np.full(len(offsets), -np.inf)
    batches = []
    for start in range(0, n, size):
        draws = theta[:, start : start + size]
        kernel = PredictiveKernel(draws, order=order)
        # Neither support × all draws nor support × all batches is materialized.
        # The final short batch contributes to the estimate but not its MCSE.
        logs = np.stack([logsumexp(kernel.draw_logpdf(d), axis=1) for d in offsets], axis=1)
        total = np.logaddexp(total, logsumexp(logs, axis=0))
        if masks is not None and draws.shape[1] == size:
            joint = logs + logw - np.log(size)
            batches.append(
                np.column_stack(
                    [logsumexp(joint, axis=1)]
                    + [logsumexp(joint[:, mask], axis=1) for mask in masks]
                )
            )
    return total - np.log(theta.shape[0] * n), batches


def private_chart(theta, support, weights, *, target, posterior, order, source="trinetx"):
    """Private chart payload: exact point masses and gated summaries, plus display densities."""
    if (
        target not in GROUPS
        or source not in ("trinetx", "synthetic")
        or not re.fullmatch(r"[0-9a-f]{64}", posterior)
        or len(theta) != 4
        or order < 4
    ):
        raise ValueError("Expected a matched candidate and paired quadrature orders")
    a, w = empirical_support(support, weights)
    if len(a) == 1:
        mass = relative = np.ones(1)
        interval = [float(a[0])] * 3
        tails = [float(a[0] >= k) for k in THRESHOLDS]
    else:
        offsets, logw = a - READING, np.log(w)
        coarse_loglike, _ = _batch_logpdf(theta, order // 2, offsets, logw)
        coarse = normalize_log_weights(coarse_loglike + logw)
        # If coarse and fine quantiles disagree, resolved_summaries holds the
        # entire interval. Using the coarse candidates lets one fine pass supply
        # both the exact probabilities and the batch evidence at each boundary.
        masks = evidence_masks(a, weighted_quantile(a, coarse, LEVELS), THRESHOLDS)
        loglike, batches = _batch_logpdf(theta, order, offsets, logw, masks)
        logjoint = loglike + logw
        mass = normalize_log_weights(logjoint)
        relative = np.exp(loglike - loglike.max())
        ratios = masks @ mass
        # Compare batch numerator - ratio * batch denominator on the same
        # all-draw evidence scale. Individual fits are never normalized first.
        batch = np.exp(np.concatenate(batches) - logsumexp(logjoint))
        residuals = batch[:, 1:] - batch[:, :1] * ratios
        errors = residuals.std(axis=0, ddof=1) / np.sqrt(len(residuals))
        interval, tails = resolved_summaries(
            a, weighted_quantile(a, mass, LEVELS), coarse, masks, ratios, errors
        )
    return dict(
        version="conditional_lookup_private_chart_v1",
        source=source,
        method=METHOD,
        prior_uncertainty="fixed_empirical",
        target=target,
        tcco2=READING,
        posterior=posterior,
        support=a.tolist(),
        prior=w.tolist(),
        mass=mass.tolist(),
        likelihood=relative.tolist(),
        interval=interval,
        tails=tails,
        display=display_curves(a, w, mass),
    )
