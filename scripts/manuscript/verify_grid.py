"""Independent, ordinary-density verification of every row of the manuscript grid."""

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from private_inputs import model_receipts, private_destination, read_prior, require_digest

READINGS = range(20, 101)
LEVELS = [0.025, 0.5, 0.975]
COLUMNS = ["tc_mmhg", "lower95_mmhg", "median", "upper95_mmhg", "p_ge45", "p_ge52"]


def batch_density(draws, offsets, nodes, weights):
    """Direct normal densities with independently constructed quadrature components."""
    mb, ml, tb, tl, rho = draws.T[:, :, None]
    loading = np.where(tl == 0, 0, tb * rho)
    location = (mb + loading * nodes).ravel()
    variance = (np.exp(ml + tl * nodes) + tb**2 - loading**2).ravel()
    coefficient = np.tile(weights, len(draws)) / (len(draws) * np.sqrt(2 * np.pi * variance))
    exponent = -0.5 / variance
    result = np.empty(len(offsets))
    for start in range(0, len(offsets), 32):
        chunk = slice(start, start + 32)
        distance = offsets[chunk, None] - location
        np.square(distance, out=distance)
        distance *= exponent
        np.exp(distance, out=distance)
        result[chunk] = np.einsum("ij,j->i", distance, coefficient)
    return result


def check_value(actual, expected, reading, quantity, *, discrete=False):
    same = (
        np.array_equal(actual, expected, equal_nan=True)
        if discrete
        else np.shape(actual) == np.shape(expected)
        and np.allclose(actual, expected, rtol=0, atol=2e-11, equal_nan=True)
    )
    if not same:
        raise ValueError(f"Mismatch at TcCO₂ {reading} mmHg for {quantity}")


def verify_rows(theta, support, weights, frame, diagnostics):
    """Verify normalized results, both quadrature orders, MCSEs and precision holds."""
    if list(frame.columns) != COLUMNS or frame.tc_mmhg.tolist() != list(READINGS):
        raise ValueError("Expected each integer reading from 20 through 100 in order")
    if [d["tc"] for d in diagnostics] != list(READINGS):
        raise ValueError("Missing or unordered full-grid precision evidence")
    if not (
        len(weights) == len(support) > 0
        and np.all(np.diff(support) > 0)
        and np.isfinite(support).all()
        and (weights > 0).all()
        and np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Invalid empirical distribution for grid verification")
    batch_size = math.isqrt(theta.shape[1])
    blocks = [
        chain[i : i + batch_size] for chain in theta for i in range(0, len(chain), batch_size)
    ]
    sizes = np.array([len(block) for block in blocks])
    differences = np.unique(np.concatenate([support - reading for reading in READINGS]))
    means = []
    for order in (32, 64):
        x, w = np.polynomial.hermite.hermgauss(order)
        nodes = x * np.sqrt(2)
        normal_weights = w / np.sqrt(np.pi)
        with ThreadPoolExecutor(max_workers=5) as pool:
            batches = np.stack(
                list(
                    pool.map(
                        lambda block: batch_density(block, differences, nodes, normal_weights),
                        blocks,
                    )
                )
            )
        means.append(np.average(batches, axis=0, weights=sizes))
    # The full-draw estimate includes the final partial batch from every chain.
    # Only equal-length, contiguous within-chain batches enter the MCSE.
    complete = batches[sizes == batch_size]
    for row, recorded in zip(frame.itertuples(index=False), diagnostics, strict=True):
        reading = row.tc_mmhg
        indices = np.searchsorted(differences, support - reading)
        unnormalized = weights * np.array(means)[:, indices]
        evidence = unnormalized.sum(axis=1)
        if not np.all(np.isfinite(evidence) & (evidence > 0)):
            raise ValueError(f"Unrepresentable reference density at TcCO₂ {reading}")
        coarse, fine = unnormalized / evidence[:, None]
        quantiles = [support[np.searchsorted(mass.cumsum(), LEVELS)] for mass in (coarse, fine)]
        masks = [support >= 45, support >= 52]
        for q in quantiles[1]:
            masks.extend([support < q, support <= q])
        # Sum weighted likelihood over each event and its full denominator before centering.
        event = np.array(masks)
        p = event @ fine
        low = event @ coarse
        joint = complete[:, indices] * weights
        scores = joint @ event.T - joint.sum(axis=1)[:, None] * p
        se = scores.std(axis=0, ddof=1) / (evidence[1] * np.sqrt(len(complete)))
        for name, expected in [
            ("coarse_q", quantiles[0]),
            ("fine_q", quantiles[1]),
            ("ratios", p),
            ("coarse", low),
            ("se", se),
        ]:
            check_value(
                recorded[name], expected, reading, name, discrete=name in {"coarse_q", "fine_q"}
            )
        resolved = (
            np.array_equal(*quantiles)
            and np.all(p[2::2] + 2 * se[2::2] < LEVELS)
            and np.all(np.array(LEVELS) <= p[3::2] - 2 * se[3::2])
        )
        interval = quantiles[1] if resolved else [np.nan] * 3
        check_value(row[1:4], interval, reading, "prediction interval and median", discrete=True)
        for i, threshold in enumerate((45, 52)):
            homogeneous = not masks[i].any() or masks[i].all()
            qualified = homogeneous or (se[i] <= 0.00025 and abs(p[i] - low[i]) <= 0.0001)
            check_value(
                row[4 + i], p[i] if qualified else np.nan, reading, f"probability ≥{threshold}"
            )
    return {
        "rows": len(frame),
        "intervals": int(frame["median"].count()),
        "p_ge45": int(frame.p_ge45.count()),
        "p_ge52": int(frame.p_ge52.count()),
    }


def run(out, original, manifest):
    private_destination(out)
    fits, _, receipt = model_receipts(original, manifest)
    all_hash = receipt["posterior_sha256"]
    for name, fit in [("all", "main"), ("icu", "icu"), ("ed_inp", "arf"), ("pft", "lft")]:
        path = (
            next(original.glob("precision-64000/public-model/*/main/*/posterior.npz"))
            if name == "all"
            else out / f"{fit}.npz"
        )
        require_digest(path, all_hash if name == "all" else fits[fit]["posterior_sha256"])
        a, w = read_prior(out, name)
        frame = pd.read_csv(out / f"{name}-predictions.csv", float_precision="round_trip")
        qa = json.loads((out / f"{name}-precision.json").read_text())
        with np.load(path, allow_pickle=False) as saved:
            result = verify_rows(saved["theta"], a, w, frame, qa)
        print(name, json.dumps(result), flush=True)


if __name__ == "__main__":
    run(*(Path(arg).resolve() for arg in sys.argv[1:]))
