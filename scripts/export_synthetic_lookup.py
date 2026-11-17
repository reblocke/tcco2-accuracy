"""Apply conditional reporting v2 to the checksum-pinned, same-fit synthetic demonstration."""

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from tcco2_accuracy.core.conditional_reporting import qualify
from tcco2_accuracy.core.future_setting import PredictiveKernel
from tcco2_accuracy.pages_prediction import GROUPS
from tcco2_accuracy.workflows.conditional_lookup import READINGS, validate_lookup
from tcco2_accuracy.workflows.future_setting_data import ROOT
from tcco2_accuracy.workflows.future_setting_inverse import load_candidate_draws

BASE = ROOT / ".pytest_tmp/future-setting"
DEST = ROOT / "models/pages"


BASELINE = "90ab00b7f1338b593a0aaa6571358d3370d13b1738cffea5f15fc8e55262e7b3"


def grid_variance(theta, order, a, w, readings, density):
    """Reuse latent-setting quadrature across a grid of readings for the invented prior.

    Each cell tabulates Z(setting,t)*Var(A|setting,t). Deposit the tensor quadrature
    weights for a contiguous posterior batch on this grid by bilinear interpolation.
    The posterior integrals are linear in these weights, even though conditional
    variance at a single setting is nonlinear in the empirical prior.
    """
    from scipy.special import roots_hermitenorm

    readings = np.asarray(readings)
    a, w = np.asarray(a), np.asarray(w) / sum(w)
    joint = w[:, None] * density
    z = joint.sum(axis=0)
    center = (a @ joint) / z
    deviation = a[:, None] - center
    total = (joint * deviation**2).sum(axis=0) / z
    offsets, indices = np.unique(a[:, None] - readings, return_inverse=True)
    indices = indices.reshape(len(a), -1)
    rules = {}
    for n in (12, 24):
        nodes, weights = roots_hermitenorm(n)
        rules[n] = nodes, np.outer(weights, weights) / weights.sum() ** 2
    mb, ml, tb, tl, r = theta.reshape(-1, 5).T
    maximum = rules[24][0][-1]
    radius = maximum * tb * (abs(r) + np.sqrt(1 - r * r))
    # Both meshes include every quadrature node, including tails. Integer-aligned
    # endpoints make the coarse mesh an exact subset of the fine mesh.
    db, dl = 0.25, 0.025

    def axis(lo, hi, step):
        return np.arange(2 * np.floor(lo / (2 * step)), 2 * np.ceil(hi / (2 * step)) + 3) * step

    biases = axis((mb - radius).min(), (mb + radius).max(), db)
    logs = axis(((ml - maximum * tl) / 2).min(), ((ml + maximum * tl) / 2).max(), dl)
    # For a given SD the arterial likelihood depends only on (bias + reading).
    # Integer readings and the quarter-mmHg bias mesh share most locations.
    locs, gather = np.unique(biases[:, None] + readings, return_inverse=True)
    gather = gather.reshape(len(biases), len(readings))
    table = np.empty((len(biases), len(logs), len(readings)))
    delta = a - np.mean(a)
    for j, ls in enumerate(logs):
        sd = np.exp(ls)
        joint = np.exp(-0.5 * ((locs[:, None] - a) / sd) ** 2) * w / (sd * np.sqrt(2 * np.pi))
        den = joint.sum(axis=1)
        mean = np.divide(joint @ delta, den, out=np.zeros_like(den), where=den > 0)
        within = np.sum(joint * (delta - mean[:, None]) ** 2, axis=1)
        table[:, j, :] = within[gather] / z

    surfaces = {
        1: table.reshape(-1, len(readings)),
        2: np.ascontiguousarray(table[::2, ::2]).reshape(-1, len(readings)),
    }

    def integrate(b, ls, weights, step):
        x = (b - biases[0]) / (db * step)
        y = (ls - logs[0]) / (dl * step)
        i, j = x.astype(int), y.astype(int)
        fx, fy = x - i, y - j
        ncols = len(logs[::step])
        flat = (i * ncols + j).ravel()
        hist = np.zeros(surfaces[step].shape[0])
        for shift, factor in (
            (0, (1 - fx) * (1 - fy)),
            (1, (1 - fx) * fy),
            (ncols, fx * (1 - fy)),
            (ncols + 1, fx * fy),
        ):
            hist += np.bincount(
                flat + shift, weights=(factor * weights).ravel(), minlength=len(hist)
            )
        used = np.flatnonzero(hist)
        return hist[used] @ surfaces[step][used]

    nchain, ndraw = theta.shape[:2]
    batch_size = int(np.sqrt(ndraw))
    accum = np.zeros((6, len(readings)))
    batches = []
    for chain in theta:
        for start in range(0, ndraw, batch_size):
            rows = chain[start : start + batch_size]
            b, log_variance, tb, tl, r = rows.T[:, :, None, None]
            loading = np.where(tl == 0, 0, tb * r)
            spread = np.sqrt(tb**2 - loading**2)
            latent = {}
            for n, (nodes, weights) in rules.items():
                loc = b + loading * nodes[:, None] + spread * nodes
                ls = (log_variance + tl * nodes[:, None]) / 2
                latent[n] = integrate(loc, ls, weights, 1)
                if n == 24:
                    mesh_check = integrate(loc, ls, weights, 2)
            ker = PredictiveKernel(rows[None, :, :], order=order)
            scale = ker.weights / (ker.sd[0] * np.sqrt(2 * np.pi))
            inv = 1 / ker.sd[0]
            dens = np.stack(
                [
                    (scale * np.exp(-0.5 * ((d - ker.mean[0]) * inv) ** 2)).sum(axis=1)
                    for d in offsets
                ],
                axis=1,
            )
            weighted = dens[:, indices] * w[None, :, None] / z
            evidence = weighted.sum(axis=1)
            first = (weighted * deviation).sum(axis=1)
            parameter = (
                np.divide(first, evidence, out=np.zeros_like(first), where=evidence > 0) * first
            )
            second = (weighted * deviation**2).sum(axis=(0, 1))
            stats = np.array(
                [
                    latent[24],
                    parameter.sum(axis=0),
                    second,
                    evidence.sum(axis=0),
                    latent[12],
                    mesh_check,
                ]
            )
            accum += stats
            if len(rows) == batch_size:
                batches.append(stats[:4] / batch_size)
    within, param, second, evidence, coarse, mesh = accum / (nchain * ndraw)
    values = np.array([within, second - within - param, param])
    bw, bp, bs, bz = np.moveaxis(batches, 1, 0)
    comps = np.stack([bw, bs - bw - bp, bp], axis=1)

    def se(x):
        return x.std(axis=0, ddof=1) / np.sqrt(len(batches))

    mcse = se(comps - values * bz[:, None, :])
    share_se = se((comps - (values / total) * bs[:, None, :]) / total)
    error = np.maximum(abs(within - coarse), abs(within - mesh))
    ok = (
        (values.min(axis=0) >= -1e-10)
        & (abs(evidence - 1) < 1e-7)
        & np.isclose(second, total, atol=1e-8)
        & (error <= 0.05)
        & (error <= 0.005 * total)
        & (mcse.max(axis=0) <= 0.05)
        & (share_se.max(axis=0) <= 0.005)
    )
    return [np.maximum(values[:, i], 0).tolist() if passed else None for i, passed in enumerate(ok)]


def build():
    from manuscript.direct_density import direct_batch_density

    raw = subprocess.check_output(
        ["git", "show", "713f5c3:models/pages/synthetic-lookup.json"], cwd=DEST
    )
    if hashlib.sha256(raw).hexdigest() != BASELINE:
        raise ValueError("Unexpected synthetic baseline")
    doc = json.loads(raw)
    summary = json.loads((DEST / "summary.json").read_bytes())
    with tempfile.TemporaryDirectory() as scratch:
        import sys

        library = Path(scratch) / "mixture.so"
        flags = ["-framework", "Accelerate"] if sys.platform == "darwin" else ["-lm"]
        subprocess.run(
            [
                "cc",
                "-O3",
                "-shared",
                "-fPIC",
                str(Path(__file__).parent / "manuscript/mixture.c"),
                *flags,
                "-o",
                library,
            ],
            check=True,
        )
        fn = direct_batch_density(library)
        for target, group in doc["groups"].items():
            info = summary[GROUPS[target]]
            theta, order, manifest = load_candidate_draws(BASE / info["run"], target)
            if group["posterior"] != manifest["strata"][GROUPS[target]]["posterior_sha256"]:
                raise ValueError("Posterior differs from synthetic baseline")
            a, w = np.array(group["support"]), np.array(group["prior"])
            x, ix = np.unique(a[None, :] - np.array(READINGS)[:, None], return_inverse=True)
            size = math.isqrt(theta.shape[1])
            chunks = [c[i : i + size] for c in theta for i in range(0, len(c), size)]
            lengths = np.asarray([len(c) for c in chunks])
            means = []
            for n in (order // 2, order):
                with ThreadPoolExecutor(max_workers=5) as pool:
                    logs = np.stack(list(pool.map(lambda c: fn(c, n, x), chunks)))
                means.append(
                    logsumexp(logs + np.log(lengths[:, None]), axis=0) - np.log(lengths.sum())
                )
            for idx, row in zip(ix.reshape(len(READINGS), -1), group["rows"], strict=True):
                joint = np.stack(means)[:, idx] + np.log(w)
                norm = logsumexp(joint, axis=1)
                coarse, fine = np.exp(joint - norm[:, None])
                np.testing.assert_allclose(fine, row["mass"], rtol=0, atol=1e-9)
                block = np.exp(logs[lengths == size][:, idx] + np.log(w) - norm[1])
                q, tails, report, _ = qualify(a, fine, coarse, [45, 52], [block])
                row.update(interval=q, tails=tails, report=report)
            print(target, "requalified", flush=True)
    doc["version"] = "conditional_lookup_dense_v2"
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEST / "synthetic-lookup.json")
    args = parser.parse_args()
    doc = build()
    validate_lookup(doc)
    args.output.write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False) + "\n")
