"""Pointwise precision of fixed-prior conditional estimates (November 2026 amendment)."""

from decimal import ROUND_HALF_UP, Decimal

import numpy as np

LEVELS = (0.025, 0.5, 0.975)


def display_range(bounds, *, percent=False):
    if bounds is None:
        return "Held"
    lo, hi = bounds
    scale = 100 if percent else 1
    digits = [
        int((Decimal(str(v)) * scale).quantize(Decimal(1), rounding=ROUND_HALF_UP)) for v in bounds
    ]
    if percent and digits[0] == 0 and hi < 0.01 and hi > 0:
        return "<1%"
    if percent and digits[1] == 100 and lo > 0.99 and lo < 1:
        return ">99%"
    text = str(digits[0]) if digits[0] == digits[1] else f"{digits[0]}–{digits[1]}"
    return text + ("%" if percent else "")


def qualify(a, mass, coarse, thresholds, batches, *, interpolation_error=0.0):
    """Batches contain fine-order joint masses on the common all-draw evidence scale."""
    masks = np.asarray([a >= k for k in thresholds])
    p = masks @ mass
    cdf = mass.cumsum()
    cdf[-1] = 1.0
    ratios = np.r_[cdf, p]
    count = 0
    sums = np.zeros_like(ratios)
    squares = np.zeros_like(ratios)
    for joint in batches:
        scores = np.column_stack((joint.cumsum(axis=1), joint @ masks.T))
        scores -= joint.sum(axis=1)[:, None] * ratios
        sums += scores.sum(axis=0)
        squares += np.square(scores).sum(axis=0)
        count += len(scores)
    if count < 2:
        raise ValueError("At least two complete chain batches required")
    se = np.sqrt(np.maximum(0, squares - sums * sums / count) / (count * (count - 1)))
    delta = np.abs(ratios - np.r_[coarse.cumsum(), masks @ coarse])
    radius = 2 * se + delta + interpolation_error
    err = radius[: len(a)]
    ceiling = np.maximum.accumulate(np.minimum(1, cdf + err))
    floor = np.minimum.accumulate(np.maximum(0, cdf - err)[::-1])[::-1]
    ceiling[-1] = floor[-1] = 1
    brackets = []
    q = []
    for level in LEVELS:
        low, high = a[np.searchsorted(ceiling, level)], a[np.searchsorted(floor, level)]
        ok = high - low <= 0.5
        brackets.append([float(low), float(high)] if ok else None)
        q.append(float(a[np.searchsorted(cdf, level)]) if ok else None)
    tails, bounds = [], []
    for i, (mask, value) in enumerate(zip(masks, p, strict=True)):
        j = len(a) + i
        value = float(np.clip(value, 0, 1))
        if mask.all() or not mask.any():
            value = float(mask.all())
            bound = [value, value]
        elif radius[j] <= 0.001 and delta[j] <= 0.0001:
            bound = [
                min(max(0.0, value - radius[j]), np.nextafter(1.0, 0.0)),
                max(min(1.0, value + radius[j]), np.nextafter(0.0, 1.0)),
            ]
        else:
            value = bound = None
        tails.append(value)
        bounds.append([float(v) for v in bound] if bound is not None else None)
    return q, tails, dict(quantiles=brackets, probabilities=bounds), se[len(a) :]
