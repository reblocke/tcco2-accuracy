"""Offline conditional prediction and variance for the four-setting Pages lookup."""

from __future__ import annotations

import re

import numpy as np
from scipy import special

from ..core._posterior import weighted_quantile
from ..core.future_setting import LEVELS, METHOD, PredictiveKernel, empirical_support
from ..pages_prediction import GROUPS, THRESHOLDS, evidence_masks, resolved_summaries
from ..reporting.future_setting import batch_mcse

READINGS = range(20, 101)
VERSION = "conditional_lookup_v2"


def lookup_document(source):
    return dict(
        version=VERSION,
        source=source,
        method=METHOD,
        prior_uncertainty="fixed_empirical",
        groups={},
    )


def _setting_moments(theta, offsets, logw, order, log_evidence):
    """Return E[Z Var(A|setting,t)] and E[Z] per fit draw, on one common evidence scale."""
    nodes, weights = special.roots_hermitenorm(order)
    weights /= weights.sum()
    result = np.empty((*theta.shape[:2], 2))
    for c, chain in enumerate(theta):
        # Stream over both draws and support. Memory is independent of prior cardinality.
        block = max(1, 200_000 // order**2)
        for start in range(0, len(chain), block):
            mb, ml, tb, tl, rho = chain[start : start + block].T[:, :, None, None]
            loading = np.where(tl == 0, 0, tb * rho)
            bias = mb + loading * nodes[:, None] + np.sqrt(tb**2 - loading**2) * nodes
            log_sd = (ml + tl * nodes[:, None]) * 0.5
            inv_sd = np.exp(-log_sd)
            # A common marginal-evidence scale preserves weights across settings.
            # Online central moments avoid subtraction of nearly equal raw moments.
            den = np.zeros_like(bias)
            mean = np.zeros_like(bias)
            numerator = np.zeros_like(bias)
            base = -log_sd - 0.5 * np.log(2 * np.pi) - log_evidence
            for d, lw in zip(offsets, logw, strict=True):
                mass = np.exp(base + lw - 0.5 * ((d - bias) * inv_sd) ** 2)
                new = den + mass
                fraction = np.divide(mass, new, where=new > 0, out=np.zeros_like(new))
                delta = d - mean
                numerator += den * fraction * delta**2
                mean += fraction * delta
                den = new
            joint_weights = weights[:, None] * weights
            result[c, start : start + len(mb), 0] = (joint_weights * numerator).sum(axis=(1, 2))
            result[c, start : start + len(mb), 1] = (joint_weights * den).sum(axis=(1, 2))
    return result


def lookup_row(theta, support, weights, t, *, order):
    """Marginalize likelihood first; gate interval and variance outputs independently."""
    a, w = empirical_support(support, weights)
    if t not in READINGS or order < 4 or len(theta) != 4:
        raise ValueError(
            "Expected an exact lookup reading, four chains and paired quadrature orders"
        )
    if len(a) == 1:
        x = float(a[0])
        return dict(
            interval=[x, x, x], tails=[float(x >= k) for k in THRESHOLDS], variance=[0.0, 0.0, 0.0]
        )
    d = a - t
    logw = np.log(w)

    # Accumulate marginal masses without storing support × all draws. Use the
    # same row-specific scaling for the fine likelihood and latent-setting integrals.
    def marginal(n):
        kernel = PredictiveKernel(theta, order=n)
        logs = kernel.logpdf(d) + logw
        logz = special.logsumexp(logs)
        mass = np.exp(logs - logz)
        return kernel, float(logz), mass, weighted_quantile(a, mass, LEVELS)

    coarse, _, low_mass, _ = marginal(order // 2)
    del coarse
    kernel, logz, mass, qs = marginal(order)
    center = mass @ d
    total_var = mass @ (d - center) ** 2
    masks = evidence_masks(a, qs, THRESHOLDS)
    ratios = masks @ mass
    # Z, centered first and second moments, then indicator numerators per draw.
    den, first, second, *indicators = np.zeros((3 + len(masks), *theta.shape[:2]))
    for i, x in enumerate(d):
        joint = np.exp(kernel.draw_logpdf(x) + logw[i] - logz)
        den += joint
        first += joint * (x - center)
        second += joint * (x - center) ** 2
        for num, include in zip(indicators, masks[:, i], strict=True):
            if include:
                num += joint
    del kernel
    errors = [batch_mcse(v - r * den) for v, r in zip(indicators, ratios, strict=True)]
    interval, tails = resolved_summaries(a, qs, low_mass, masks, ratios, errors)
    if total_var <= 0 or not np.isfinite(total_var):
        return dict(interval=interval, tails=tails, variance=None)

    # E_theta[Z (E[A|theta,t] - E[A|t])²]; a zero-evidence draw has zero weight.
    parameter = np.divide(first, den, where=den > 0, out=np.zeros_like(first)) * first

    def budget(n):
        within, integral = np.moveaxis(_setting_moments(theta, d, logw, n, logz), -1, 0)
        draws = np.array([within, second - parameter - within, parameter])
        values = draws.mean(axis=(1, 2))
        return draws, values, abs(integral.mean() - 1)

    _, low, low_error = budget(12)
    draws, values, error = budget(24)
    # Influence of the common evidence denominator, including for variance shares.
    diff = np.abs(values - low)
    shares = values / total_var
    mcse = np.array([batch_mcse(x - v * den) for x, v in zip(draws, values, strict=True)])
    share_mcse = [
        batch_mcse((x - r * second) / total_var) for x, r in zip(draws, shares, strict=True)
    ]
    ok = (
        max(low_error, error) <= 0.001
        and np.all(values >= -1e-10)
        and np.isclose(values.sum(), total_var, atol=1e-8)
        and max(diff) <= 0.05
        and max(mcse) <= 0.05
        and max(diff) / total_var <= 0.005
        and max(share_mcse) <= 0.005
    )
    return dict(
        interval=interval, tails=tails, variance=np.maximum(values, 0).tolist() if ok else None
    )


def validate_lookup(doc):
    """Exact aggregate schema only. Release authorization applies separately to its bytes."""

    def numbers(xs, n):
        return (
            isinstance(xs, list)
            and len(xs) == n
            and all(type(x) in (int, float) and np.isfinite(x) for x in xs)
        )

    summary = isinstance(doc, dict) and doc.get("version") == "conditional_lookup_v3"
    amended = summary or (
        isinstance(doc, dict) and doc.get("version") == "conditional_lookup_dense_v2"
    )
    dense = isinstance(doc, dict) and doc.get("version") in (
        "conditional_lookup_dense_v1",
        "conditional_lookup_dense_v2",
    )
    if (
        not isinstance(doc, dict)
        or set(doc) != {"version", "method", "source", "prior_uncertainty", "groups"}
        or (not dense and not summary and doc["version"] not in ("conditional_lookup_v1", VERSION))
        or (summary and doc["source"] != "trinetx")
        or (dense and doc["source"] != "synthetic")
        or doc["method"] != METHOD
        or doc["source"] not in ("synthetic", "trinetx")
        or doc["prior_uncertainty"] != "fixed_empirical"
        or set(doc["groups"]) != set(GROUPS)
    ):
        raise ValueError("Invalid lookup header")
    legacy = doc["version"] == "conditional_lookup_v1"
    chart = not legacy and doc["source"] == "synthetic"
    for group in doc["groups"].values():
        if (
            set(group)
            != {"prior_interval", "posterior", "rows"}
            | ({"reference_posterior"} if summary else set())
            | ({"support", "prior"} if chart else set())
            | ({"curve"} if dense else set())
            or not isinstance(group["posterior"], str)
            or not re.fullmatch("[a-f0-9]{64}", group["posterior"])
            or not numbers(group["prior_interval"], 2)
            or not 0 < group["prior_interval"][0] <= group["prior_interval"][1]
            or not isinstance(group["rows"], list)
            or len(group["rows"]) != len(READINGS)
        ):
            raise ValueError("Invalid group summary or incomplete lookup grid")
        if summary and (
            not isinstance(group["reference_posterior"], str)
            or not re.fullmatch("[a-f0-9]{64}", group["reference_posterior"])
        ):
            raise ValueError("Invalid reference posterior identity")
        if chart:
            a, w = group["support"], group["prior"]
            if (
                not isinstance(a, list)
                or not a
                or not numbers(a, len(a))
                or min(a) <= 0
                or not np.all(np.diff(a) > 0)
                or not numbers(w, len(a))
                or min(w) <= 0
                or abs(sum(w) - 1) > 1e-9
            ):
                raise ValueError("Invalid synthetic prior")
        if dense:
            curve = group["curve"]
            if (
                not isinstance(curve, dict)
                or set(curve) != {"x", "density"}
                or not isinstance(curve["x"], list)
                or len(curve["x"]) < 2
                or not numbers(curve["x"], len(curve["x"]))
                or not numbers(curve["density"], len(curve["x"]))
                or min(curve["density"]) <= 0
                or not np.all(np.diff(curve["x"]) == 0.25)
                or not np.all(np.diff(a) == 0.25)
                or len(a) != 281
                or curve["x"][0] > min(a) - max(READINGS)
                or curve["x"][-1] < max(a) - min(READINGS)
            ):
                raise ValueError("Invalid dense demonstration grid")
        for row in group["rows"]:
            if set(row) != {"interval", "variance"} | ({"report"} if amended else set()) | (
                set() if legacy else {"tails"}
            ) | (({"mass"} if dense else {"mass", "likelihood"}) if chart else set()):
                raise ValueError("Unexpected lookup fields")
            q, v = row["interval"], row["variance"]
            if amended:
                report = row["report"]
                if not isinstance(report, dict) or set(report) != {"quantiles", "probabilities"}:
                    raise ValueError("Invalid reporting brackets")
                for key, values, count, limit in (
                    ("quantiles", q, 3, 0.5),
                    ("probabilities", row["tails"], 2, 0.002),
                ):
                    if (
                        not isinstance(values, list)
                        or len(values) != count
                        or not isinstance(report[key], list)
                        or len(report[key]) != count
                    ):
                        raise ValueError("Invalid reporting dimensions")
                    for value, pair in zip(values, report[key], strict=True):
                        if value is None and pair is None:
                            continue
                        if (
                            not numbers([value], 1)
                            or not numbers(pair, 2)
                            or not pair[0] <= value <= pair[1]
                            or pair[1] - pair[0] > limit + 1e-12
                        ):
                            raise ValueError("Invalid reporting bracket")
                        if key == "probabilities" and not 0 <= pair[0] <= pair[1] <= 1:
                            raise ValueError("Invalid probability bracket")
                        if key == "quantiles" and (
                            pair[0] <= 0
                            or (chart and (value not in a or any(v not in a for v in pair)))
                        ):
                            raise ValueError("Unknown arterial support point")
                if any(
                    x > y
                    for i, x in enumerate(q)
                    for y in q[i + 1 :]
                    if x is not None and y is not None
                ):
                    raise ValueError("Unordered quantiles")
            elif q is not None and (
                not numbers(q, 4 if legacy else 3)
                or not 0 < q[0] <= q[1] <= q[2]
                or (legacy and not 0 <= q[3] <= 1)
            ):
                raise ValueError("Invalid conditional interval")
            if v is not None and (not numbers(v, 3) or min(v) < 0):
                raise ValueError("Invalid conditional variance")
            if not legacy:
                tails = row["tails"]
                if (
                    not isinstance(tails, list)
                    or len(tails) != 2
                    or any(
                        p is not None and (not numbers([p], 1) or not 0 <= p <= 1) for p in tails
                    )
                    or (None not in tails and tails[1] > tails[0] + 1e-12)
                ):
                    raise ValueError("Invalid threshold probabilities")
            if chart:
                mass = row["mass"]
                like = row.get("likelihood")
                if (
                    not numbers(mass, len(a))
                    or min(mass) < 0
                    or abs(sum(mass) - 1) > 1e-9
                    or (
                        not dense and (not numbers(like, len(a)) or min(like) < 0 or max(like) != 1)
                    )
                ):
                    raise ValueError("Invalid synthetic chart")
