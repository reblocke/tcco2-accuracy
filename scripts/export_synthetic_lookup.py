"""Reproduce the Pages example from the four accepted fits and the public invented prior."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from tcco2_accuracy.core._posterior import weighted_quantile
from tcco2_accuracy.core.future_setting import PredictiveKernel
from tcco2_accuracy.pages_prediction import GROUPS, conditional
from tcco2_accuracy.workflows.conditional_lookup import (
    READINGS,
    lookup_document,
    lookup_row,
    validate_lookup,
)
from tcco2_accuracy.workflows.future_setting_data import ROOT, digest
from tcco2_accuracy.workflows.future_setting_inverse import load_candidate_draws

BASE = ROOT / ".pytest_tmp/future-setting"
DEST = ROOT / "models/pages"


def tail_densities(chain, order, offsets):
    # The integer readings share offsets. Calculate each missing knot once, in exactly
    # the same contiguous chain batches as the existing public density kernel.
    size = int(np.sqrt(len(chain)))
    total = np.zeros((2, len(offsets)))
    batches = []
    for start in range(0, len(chain), size):
        draws = chain[start : start + size]
        for i, n in enumerate((order, order // 2)):
            kernel = PredictiveKernel(draws[None, :, :], order=n)
            mean = kernel.mean.ravel()
            inv = 1 / kernel.sd.ravel()
            factor = (kernel.weights / (kernel.sd * np.sqrt(2 * np.pi))).ravel()
            sums = np.array(
                [np.sum(factor * np.exp(-0.5 * ((d - mean) * inv) ** 2)) for d in offsets]
            )
            total[i] += sums
            if i == 0 and len(draws) == size:
                batches.append(sums / size)
    return total, np.array(batches)


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


def fitted_curve(target, offset_min, offset_max):
    """Evaluate a plotting grid independent of the prior, including all demo offsets."""
    group = GROUPS[target]
    info = json.loads((DEST / "summary.json").read_text())[group]
    theta, order, _ = load_candidate_draws(BASE / info["run"], target)
    offsets = np.arange(min(-48, offset_min), max(48, offset_max) + 0.125, 0.25)
    fitted = PredictiveKernel(theta, order=order)
    return dict(x=offsets.tolist(), density=np.exp(fitted.logpdf(offsets)).tolist())


def finish_dense(doc, curves):
    """Compare all rows to the independent plotting-grid evaluation before staging."""
    doc["version"] = "conditional_lookup_dense_v1"
    for target, g in doc["groups"].items():
        a, w = np.array(g["support"]), np.array(g["prior"])
        curve = curves[target]
        g["curve"] = curve
        for t, row in zip(READINGS, g["rows"], strict=True):
            offsets = a - t
            indices = np.searchsorted(curve["x"], offsets)
            np.testing.assert_array_equal(np.array(curve["x"])[indices], offsets)
            like = np.asarray(curve["density"])[indices]
            mass = like * w
            mass /= mass.sum()
            np.testing.assert_allclose(row.pop("likelihood"), like / like.max(), atol=1e-9, rtol=0)
            np.testing.assert_allclose(row["mass"], mass, atol=1e-9, rtol=0)
            if row["interval"] is not None:
                np.testing.assert_array_equal(
                    row["interval"], weighted_quantile(a, mass, [0.025, 0.5, 0.975])
                )
            for p, k in zip(row["tails"], [45, 52], strict=True):
                if p is not None:
                    np.testing.assert_allclose(p, mass[a >= k].sum(), atol=1e-9, rtol=0)
            if row["variance"] is not None:
                np.testing.assert_allclose(
                    sum(row["variance"]), mass @ (a - mass @ a) ** 2, atol=1e-8, rtol=0
                )
    validate_lookup(doc)
    return doc


def build():
    obj = lookup_document("synthetic")
    prior_path = ROOT / "tests/fixtures/synthetic_paco2_prior.csv"
    contract = json.loads((ROOT / "docs/data_release_contract.json").read_text())
    if (
        digest(prior_path)
        != contract["synthetic_fixture_artifacts"][str(prior_path.relative_to(ROOT))]
    ):
        raise ValueError("Synthetic prior differs from the approved test fixture")
    groups = []
    for target, fixture in pd.read_csv(prior_path).groupby("group"):
        mean = fixture.paco2_bin @ fixture.weight
        sd = np.sqrt((fixture.paco2_bin - mean) ** 2 @ fixture.weight)
        support = mean + np.arange(-140, 141) * 0.25
        edges = np.r_[support - 0.125, support[-1] + 0.125]
        weights = np.diff(ndtr((edges - mean) / sd))
        groups.append(
            pd.DataFrame(dict(group=target, paco2_bin=support, weight=weights / weights.sum()))
        )
    prior = pd.concat(groups, ignore_index=True)
    prior_csv = prior.to_csv(index=False)
    summary = json.loads((DEST / "summary.json").read_text())
    for target, group in GROUPS.items():
        g = prior.loc[prior.group == target]
        a, w = g.paco2_bin.to_numpy(), g.weight.to_numpy()
        info = summary[group]
        theta, order, manifest = load_candidate_draws(BASE / info["run"], target)
        path = DEST / f"{group}.npz"
        if (
            digest(path) != info["kernel_sha256"]
            or manifest["strata"][group]["posterior_sha256"] != info["posterior"]
        ):
            raise ValueError("Accepted posterior or public kernel revision mismatch")
        with np.load(path) as f:
            kernel = dict(f)
        offsets = np.unique((a[:, None] - np.array(READINGS)).ravel())
        missing = np.setdiff1d(offsets, kernel["x"])
        if len(missing):
            with ThreadPoolExecutor(max_workers=4) as pool:
                chains = list(pool.map(lambda c: tail_densities(c, order, missing), theta))
            extra = np.log(sum(c[0] for c in chains) / np.prod(theta.shape[:2]))
            batches = np.log(np.concatenate([c[1] for c in chains]))
            knots = np.r_[kernel["x"], missing]
            idx = knots.argsort()
            kernel["x"] = knots[idx]
            kernel["logpdf"] = np.column_stack((kernel["logpdf"], extra))[:, idx]
            kernel["batch_logpdf"] = np.column_stack((kernel["batch_logpdf"], batches))[:, idx]
        from scipy.interpolate import CubicSpline

        marginal = np.exp(
            CubicSpline(kernel["x"], kernel["logpdf"][0])(a[:, None] - np.array(READINGS))
        )
        print(group, "evaluating dense variance grid", flush=True)
        variances = grid_variance(theta, order, a, w, READINGS, marginal)
        rows = []
        for t in READINGS:
            result = conditional(kernel, target, prior_csv, t, 45)
            interval, tails = result["quantiles"], result["tails"]
            variance = variances[t - READINGS.start]
            if t == 45:
                print(group, "checking direct integration", flush=True)
                direct = lookup_row(theta, a, w, t, order=order)
                if any(
                    p is not None and (ref is None or abs(p - ref) > 1e-9)
                    for p, ref in zip(tails, direct["tails"], strict=True)
                ):
                    raise ValueError(
                        "Synthetic threshold probabilities disagree with full posterior"
                    )
                if interval is not None and (
                    direct["interval"] is None
                    or not np.allclose(interval, direct["interval"], atol=1e-9, rtol=0)
                ):
                    raise ValueError("Synthetic lookup disagrees with direct posterior evaluation")
                if variance is not None and (
                    direct["variance"] is None
                    or not np.allclose(variance, direct["variance"], atol=0.025, rtol=0)
                ):
                    raise ValueError("Conditional variance fails direct tensor quadrature check")
            rows.append(
                dict(
                    interval=interval,
                    tails=tails,
                    variance=variance,
                    mass=result["mass"],
                    likelihood=result["likelihood"],
                )
            )
        obj["groups"][target] = dict(
            support=a.tolist(),
            prior=(w / w.sum()).tolist(),
            posterior=info["posterior"],
            prior_interval=weighted_quantile(a, w, [0.025, 0.975]).tolist(),
            rows=rows,
        )
        print(group, "checked", flush=True)
    validate_lookup(obj)
    curves = {
        target: fitted_curve(
            target, g["support"][0] - max(READINGS), g["support"][-1] - min(READINGS)
        )
        for target, g in obj["groups"].items()
    }
    return finish_dense(obj, curves)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEST / "synthetic-lookup.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), separators=(",", ":"), allow_nan=False) + "\n")
