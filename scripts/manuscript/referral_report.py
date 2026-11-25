"""Evaluate frozen referral rules, independently check counts and write private review tables."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from tcco2_accuracy.core.referral_policy import (
    DISPLAY,
    class_policy,
    display_policy,
    evaluate_policy,
)

GROUPS = {
    "all": "All settings",
    "icu": "Critical Care Billing",
    "ed_inp": "ED/noncritical inpatient",
    "pft": "Ambulatory",
}
CAPS = (0.01, 0.05, 0.1)


def figure_counts(mean, cuts):
    negative = DISPLAY <= cuts[0]
    positive = DISPLAY >= cuts[1]
    counts = np.zeros((5, 83))
    counts[0, 1:-1] = negative * mean[0, 1:-1]
    counts[1, 1:-1] = negative * mean[1, 1:-1]
    counts[2, 1:-1] = positive * mean[1, 1:-1]
    counts[3, 1:-1] = positive * mean[0, 1:-1]
    counts[4] = mean.sum(axis=0) * np.r_[True, ~(negative | positive), True]
    return 100 * counts


def limits(x):
    if np.isnan(x).all():
        return None
    if not np.isfinite(x).all():
        raise ValueError("Partially undefined rate")
    return (100 * np.quantile(x, [0.025, 0.5, 0.975])).tolist()


def summarize(joint, cut, q, lo, hi):
    stats = evaluate_policy(joint, cut)
    mean = joint.mean(axis=(0, 1))
    bars = figure_counts(mean, cut)
    np.testing.assert_allclose(bars.sum(axis=0), 100 * mean.sum(axis=0), rtol=0, atol=2e-10)
    for name, row in [("FN", 1), ("FP", 3), ("referral", 4)]:
        np.testing.assert_allclose(bars[row].sum(), 100 * stats[name].mean(), rtol=0, atol=2e-10)
    oracle = mean[:, 1:-1]
    display_mass = oracle.sum(axis=0)
    direct_q = oracle[1] / display_mass
    neg, pos = DISPLAY <= cut[0], DISPLAY >= cut[1]
    for probs, mask, name in [(direct_q, neg, "FN"), (1 - direct_q, pos, "FP")]:
        np.testing.assert_allclose(display_mass[mask] @ probs[mask], stats[name].mean(), atol=2e-12)
    region = {}
    for label, mask, risk, bound, edge in [
        ("negative", neg, q, hi, -1),
        ("positive", pos, 1 - q, 1 - lo, 0),
    ]:
        region[label] = (
            None
            if not mask.any()
            else {
                "boundary_risk": float(100 * risk[mask][edge]),
                "worst_risk": float(100 * risk[mask].max()),
                "worst_upper_numerical_bound": float(100 * bound[mask].max()),
            }
        )
    return {
        "cuts": list(cut),
        "median_cri": {k: limits(v) for k, v in stats.items()},
        "design_risk_percent": region,
        "figure_counts": bars.tolist(),
    }


def class_bounds(joint, coarse):
    """Numerical limits on cumulative whole-class error at every possible cutoff."""
    result = []
    for c, direction in [(1, 1), (0, -1)]:

        def cumulative(array):
            bins = array[:, :, c, 1:-1][:, :, ::direction]
            values = bins.cumsum(axis=-1) / array[:, :, c].sum(axis=-1)[..., None]
            return np.pad(values, ((0, 0), (0, 0), (1, 0)))[:, :, ::direction]

        fine = cumulative(joint)
        mean = fine.mean(axis=(0, 1))
        delta = abs(mean - cumulative(coarse).mean(axis=(0, 1)))
        size = int(joint.shape[1] ** 0.5)
        batch = (
            fine[:, : joint.shape[1] // size * size].reshape(-1, size, fine.shape[-1]).mean(axis=1)
        )
        rad = 2 * batch.std(axis=0, ddof=1) / np.sqrt(len(batch)) + delta
        result.append((np.maximum(0, mean - rad), np.minimum(1, mean + rad)))
    return list(zip(*result, strict=True))


def file_sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_group(cache, group, entry):
    from referral_analysis import check_numerical_sources

    if entry.get("precision_code") != file_sha(Path(__file__).with_name("referral_precision.py")):
        raise ValueError("Rounded risk certificate has a different precision implementation")
    path = cache / (group + "-bin-certificate.npz")
    if file_sha(path) != entry["output"]:
        raise ValueError("Rounded risk certificate digest differs")
    with np.load(path) as cert:
        n = int(cert["per_chain"])
        q, lo, hi = [cert[k] for k in ("q", "lower", "upper")]
        noise = {
            "max_numerical_radius_pp": float(100 * np.maximum(q - lo, hi - q).max()),
            "max_quadrature_change_pp": float(100 * cert["quadrature"].max()),
        }
    arrays = []
    for order in (32, 64):
        f = cache / f"{group}-{n}-{order}.npy"
        receipt = entry["sources"][str(order)]
        check_numerical_sources(receipt)
        if file_sha(f) != receipt["output"]:
            raise ValueError("Draw-level bin file differs from certificate source")
        arrays.append(np.load(f, mmap_mode="r"))
    return *arrays, q, lo, hi, noise


def report(cache, output, caps=CAPS):
    from private_inputs import private_destination

    private_destination(output)
    entries = json.loads((cache / "bin-certificates.json").read_bytes())
    for group in GROUPS:
        coarse, joint, q, lo, hi, noise = load_group(cache, group, entries[group])
        mean = joint.mean(axis=(0, 1))
        ranges = {"fixed": (39, 51)}
        holds = {}
        for cap in sorted(caps):
            label = f"cap {100 * cap:g}%"
            ranges[label] = display_policy(lo, hi, cap)
            optimistic = display_policy(hi, lo, cap)
            holds[label] = [
                int(k)
                for i, k in enumerate(DISPLAY)
                if (ranges[label][0] < k <= optimistic[0] and lo[i] <= cap < hi[i])
                or (optimistic[1] <= k < ranges[label][1] and lo[i] < 1 - cap <= hi[i])
            ]
        lower, upper = class_bounds(joint, coarse)
        class_status = {}
        for cap in sorted(caps):
            label = f"class {100 * cap:g}%"
            ranges[label] = class_policy(mean, cap, constraints=upper)
            optimistic = class_policy(mean, cap, constraints=lower)
            class_status[label] = {
                "numerically_unique": ranges[label] == optimistic,
                "optimistic_cuts": list(optimistic),
            }
        policies = {label: summarize(joint, cut, q, lo, hi) for label, cut in ranges.items()}
        previous = None
        for cap in sorted(caps):
            label = f"cap {100 * cap:g}%"
            cuts = ranges[label]
            for r in policies[label]["design_risk_percent"].values():
                if r is not None and r["worst_upper_numerical_bound"] > 100 * cap + 1e-11:
                    raise ArithmeticError("Unqualified definitive call")
            if previous is not None and not (cuts[0] >= previous[0] and cuts[1] <= previous[1]):
                raise ArithmeticError("Non-nested display policies")
            previous = cuts
        labels = [f"cap {100 * c:g}%" for c in sorted(caps)]
        contrasts = {}
        for name, base in [(s, "fixed") for s in labels] + list(zip(labels, labels[1:])):
            a = evaluate_policy(joint, ranges[name])
            b = evaluate_policy(joint, ranges[base])
            contrasts[name + " minus " + base] = {
                k: limits(a[k] - b[k]) for k in ("referral", "efficiency", "FN", "FP")
            }
            if base != "fixed" and (
                np.any(a["FN"] > b["FN"] + 1e-13)
                or np.any(a["FP"] > b["FP"] + 1e-13)
                or np.any(a["referral"] < b["referral"] - 1e-13)
            ):
                raise ArithmeticError("Stricter cap has incompatible outcomes on a common draw")
        knots = np.unique(
            np.r_[0.01, 0.05, 0.1, np.maximum.accumulate(hi), np.maximum.accumulate(1 - lo[::-1])]
        )
        curve = []
        for cap in knots[(knots >= 0.01) & (knots <= 0.1)]:
            cuts = display_policy(lo, hi, np.nextafter(cap, np.inf))
            v = evaluate_policy(joint, cuts)["efficiency"]
            curve.append([float(100 * cap), *limits(v)])
        meta = dict(
            setting=group,
            per_chain=joint.shape[1],
            **noise,
            policies=policies,
            contrasts=contrasts,
            efficiency_curve=curve,
            numerical_holds=holds,
            class_optimum=class_status,
            reversals=[int(DISPLAY[j + 1]) for j in np.flatnonzero(np.diff(q) < 0)],
            off_grid_per_100=(100 * mean[:, [0, -1]].sum(axis=0)).tolist(),
            boundary_examples={str(k): float(100 * q[k - 20]) for k in (39, 51)},
            source_certificate=entries[group]["output"],
        )
        (output / (group + "-policy.json")).write_text(json.dumps(meta, indent=2, allow_nan=False))
        print(group, "policy identities and comparisons passed", flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=lambda s: Path(s).resolve())
    p.add_argument("output", type=lambda s: Path(s).resolve())
    p.add_argument(
        "--caps", nargs="+", type=float, default=CAPS, help="Fractions strictly between 0 and 0.5"
    )
    args = p.parse_args()
    report(args.cache, args.output, sorted(set(args.caps)))
