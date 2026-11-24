"""Joint law of the rounded transcutaneous display and arterial class on a testing occasion."""

import numpy as np
from scipy import sparse, special

from .future_setting import PredictiveKernel, empirical_support

DISPLAY = np.arange(20, 101)
NO_NEGATIVE, NO_POSITIVE = 19, 101


class DisplayIntegral:
    """Reuse common disagreement offsets while preserving the exact arterial support."""

    def __init__(self, arterial, weights):
        a, w = empirical_support(arterial, weights)
        centers, inverse = np.unique(a[:, None] - DISPLAY, return_inverse=True)
        self.centers = centers
        rows = (a[:, None] >= 45) * len(DISPLAY) + np.arange(len(DISPLAY))
        self.project = sparse.csr_matrix(
            (np.broadcast_to(w[:, None], rows.shape).ravel(), (rows.ravel(), inverse.ravel())),
            shape=(2 * len(DISPLAY), len(centers)),
        )
        coordinates = [centers - 0.5, centers + 0.5, a - 19.5, a - 100.5]
        self.edges = np.unique(np.concatenate(coordinates))
        self.lo, self.hi, self.first, self.last = [
            np.searchsorted(self.edges, item) for item in coordinates
        ]
        self.by_class = np.array([w * (a < 45), w * (a >= 45)])

    def point_block(self, kernel):
        """Joint unit-width midpoint approximation on exactly the same draws as block()."""
        mean = kernel.mean.reshape(-1, kernel.order)
        sd = kernel.sd.reshape(mean.shape)
        weights = kernel.weights / (np.sqrt(2 * np.pi) * sd)
        density = np.empty((len(self.centers), len(mean)))
        step = max(1, 250_000 // mean.size)
        for start in range(0, len(density), step):
            sl = slice(start, start + step)
            z = (self.centers[sl, None, None] - mean) / sd
            np.square(z, out=z)
            z *= -0.5
            np.exp(z, out=z)
            density[sl] = np.einsum("ijk,jk->ij", z, weights)
        return (self.project @ density).T.reshape(*kernel.theta.shape[:2], 2, len(DISPLAY))

    def block(self, kernel):
        """Return (chain, draw, arterial class, display) with overflow bins on each end."""
        shape = kernel.theta.shape[:2]
        mean = kernel.mean.reshape(-1, kernel.order)
        sd = kernel.sd.reshape(mean.shape)
        center = kernel.theta[..., 0].ravel()
        upper = self.edges[:, None] >= center
        tail = np.empty(upper.shape)
        step = max(1, 250_000 // mean.size)
        for start in range(0, len(tail), step):
            sl = slice(start, start + step)
            z = (self.edges[sl, None, None] - mean) / sd
            z *= np.where(upper[sl], -1, 1)[..., None]
            special.ndtr(z, out=z)
            tail[sl] = np.einsum("ijk,k->ij", z, kernel.weights)
        # F(d) = signed_tail(d) + I(d >= mu_b). Far-right bins subtract survival values.
        signed = np.where(upper, -tail, tail)
        interval = signed[self.hi] - signed[self.lo] + (upper[self.hi] & ~upper[self.lo])
        low = np.where(upper[self.first], tail[self.first], 1 - tail[self.first])
        high = np.where(upper[self.last], 1 - tail[self.last], tail[self.last])
        middle = (self.project @ interval).reshape(2, len(DISPLAY), -1)
        joint = np.concatenate(
            [(self.by_class @ low)[:, None], middle, (self.by_class @ high)[:, None]], axis=1
        )
        if joint.min() < -1e-14 or not np.allclose(
            joint.sum(axis=1), self.by_class.sum(axis=1)[:, None], rtol=0, atol=3e-13
        ):
            raise ArithmeticError("Display bins do not partition each arterial class")
        return np.moveaxis(joint.clip(min=0), -1, 0).reshape(*shape, 2, len(DISPLAY) + 2)


def class_joint(kernel, arterial, weights):
    integral = DisplayIntegral(arterial, weights)
    return np.concatenate(
        [
            integral.block(PredictiveKernel(kernel.theta[:, i : i + 16], order=kernel.order))
            for i in range(0, kernel.theta.shape[1], 16)
        ],
        axis=1,
    )


def ratio(num, den):
    num, den = np.broadcast_arrays(num, den)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def risk_bounds(joint, coarse):
    """Numerical bounds from contiguous within-chain ratio scores and quadrature change."""
    fine = joint[..., 1:-1]
    mean = fine.mean(axis=(0, 1))
    q = ratio(mean[1], mean.sum(axis=0))
    other = coarse[..., 1:-1].reshape(-1, 2, len(DISPLAY)).mean(axis=0)
    delta = abs(q - ratio(other[1], other.sum(axis=0)))
    n = joint.shape[1]
    size = max(1, int(n**0.5))
    batches = fine[:, : n // size * size].reshape(-1, size, 2, len(DISPLAY)).mean(axis=1)
    score = ratio(batches[:, 1] - q * batches.sum(axis=1), mean.sum(axis=0))
    se = score.std(axis=0, ddof=1) / np.sqrt(len(batches))
    radius = 2 * se + delta
    return q, (q - radius).clip(0, 1), (q + radius).clip(0, 1)


def _check_cap(cap):
    if not 0 < cap < 0.5:
        raise ValueError("The wrong-call cap must be strictly between zero and one half")


def display_policy(lower, upper, cap):
    _check_cap(cap)
    low = np.logical_and.accumulate(np.asarray(upper) <= cap)
    high = np.logical_and.accumulate(np.asarray(lower)[::-1] >= 1 - cap)
    return NO_NEGATIVE + int(low.sum()), NO_POSITIVE - int(high.sum())


def evaluate_policy(joint, cuts):
    last_neg, first_pos = cuts
    if not NO_NEGATIVE <= last_neg < first_pos <= NO_POSITIVE:
        raise ValueError("Call ranges must be disjoint and inside the display grid")
    low = joint[..., 1 : last_neg - NO_NEGATIVE + 1].sum(axis=-1)
    high = joint[..., first_pos - NO_NEGATIVE : -1].sum(axis=-1)
    total = joint.sum(axis=-1)
    negative, positive = low.sum(axis=-1), high.sum(axis=-1)
    fn, fp = low[..., 1], high[..., 0]
    efficiency = negative + positive
    return {
        "negative": negative,
        "positive": positive,
        "efficiency": efficiency,
        "referral": 1 - efficiency,
        "off_grid": joint[..., [0, -1]].sum(axis=(-1, -2)),
        "FN": fn,
        "FP": fp,
        "negative_wrong": ratio(fn, negative),
        "positive_wrong": ratio(fp, positive),
        "FNR": ratio(fn, total[..., 1]),
        "FPR": ratio(fp, total[..., 0]),
    }


def class_policy(mean, cap, *, constraints=None):
    """Select the least-referring rule under whole-strategy reference-class constraints."""
    _check_cap(cap)
    n = len(DISPLAY)
    prefix = np.column_stack([np.zeros(2), mean[:, 1:-1].cumsum(axis=1)])
    suffix = np.column_stack([mean[:, -2:0:-1].cumsum(axis=1)[:, ::-1], np.zeros(2)])
    fnr = ratio(prefix[1], mean[1].sum())
    fpr = ratio(suffix[0], mean[0].sum())
    if not np.isfinite(fnr).all() or not np.isfinite(fpr).all():
        raise ValueError("Both reference classes are required for class-conditional selection")
    bound_fn, bound_fp = (fnr, fpr) if constraints is None else constraints
    candidates = []
    for left in np.flatnonzero(bound_fn <= cap):
        for right in np.flatnonzero(bound_fp <= cap):
            if right >= left:
                efficiency = prefix[:, left].sum() + suffix[:, right].sum()
                candidates.append(
                    (
                        round(-efficiency, 12),
                        round(max(fnr[left], fpr[right]), 12),
                        round(fnr[left], 12),
                        int(left),
                        int(right),
                    )
                )
    _, _, _, left, right = min(candidates)
    return NO_NEGATIVE + left, NO_POSITIVE - n + right


def corrected_bin_risk(joint, point, point_q, point_se):
    """Use the high-precision exact-reading ratio as a control for the small rounding change.

    E(point_pos - point_q * point_total) = 0. Estimate the remaining ratio
    on common draws. Add the two MCSEs conservatively even if chains overlap.
    """
    interior = joint[..., 1:-1]
    total = interior.sum(axis=2)
    point_total = point.sum(axis=2)
    change = (interior[:, :, 1] - point[:, :, 1]) - point_q * (total - point_total)
    mean_total = total.mean(axis=(0, 1))
    correction = ratio(change.mean(axis=(0, 1)), mean_total)
    influence = ratio(change - correction * total, mean_total)
    n = influence.shape[1]
    size = max(1, int(n**0.5))
    batch = influence[:, : n // size * size].reshape(-1, size, len(DISPLAY)).mean(axis=1)
    se = batch.std(axis=0, ddof=1) / np.sqrt(len(batch))
    return point_q + correction, se + ratio(point_total.mean(axis=(0, 1)), mean_total) * point_se
