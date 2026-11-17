"""Evaluate full-support inverse predictions privately from canonical chain draws."""

import ctypes
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from private_inputs import private_destination, read_prior, require_digest
from scipy.special import logsumexp

from tcco2_accuracy.core._posterior import weighted_quantile
from tcco2_accuracy.core.future_setting import LEVELS, PredictiveKernel
from tcco2_accuracy.pages_prediction import THRESHOLDS, evidence_masks, resolved_summaries


def inverse(theta, a, w, library):
    # Native evaluator has the same unnormalized normal components as draw_logpdf.
    native = ctypes.CDLL(str(library)).mixture
    vec = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    native.argtypes = [ctypes.c_int, ctypes.c_int, *([vec] * 5)]
    native.restype = ctypes.c_int
    offsets, idx = np.unique(a[None, :] - np.arange(20, 101)[:, None], return_inverse=True)
    idx = idx.reshape(81, -1)
    size = math.isqrt(theta.shape[1])
    blocks = [chain[i : i + size] for chain in theta for i in range(0, len(chain), size)]
    lengths = np.array([len(b) for b in blocks])

    def batch(b):
        k = PredictiveKernel(b[None, :, :], order=order)
        v = np.empty(len(offsets))
        if native(
            k.mean.size,
            len(v),
            k.mean.ravel(),
            (1 / k.sd).ravel(),
            k.log_factor.ravel(),
            offsets,
            v,
        ):
            raise MemoryError()
        tiny = v < np.finfo(float).tiny * k.mean.size
        v[~tiny] = np.log(v[~tiny]) - np.log(len(b))
        for j in np.flatnonzero(tiny):
            v[j] = logsumexp(k.draw_logpdf(offsets[j])) - np.log(len(b))
        return v

    marginals = []
    for order in (32, 64):
        with ThreadPoolExecutor(max_workers=5) as pool:
            likelihood = np.stack(list(pool.map(batch, blocks)))
        marginals.append(
            logsumexp(likelihood + np.log(lengths[:, None]), axis=0) - np.log(lengths.sum())
        )
    rows = []
    checks = []
    for tc, ix in enumerate(idx, 20):
        joint = np.log(w) + np.stack(marginals)[:, ix]
        logz = logsumexp(joint, axis=1)
        coarse, mass = np.exp(joint - logz[:, None])
        qs = weighted_quantile(a, mass, LEVELS)
        masks = evidence_masks(a, qs, THRESHOLDS)
        ratios = masks @ mass
        scaled = np.exp(likelihood[lengths == size][:, ix] + np.log(w) - logz[1])
        resid = scaled @ (masks.T - ratios)
        err = resid.std(axis=0, ddof=1) / np.sqrt(len(resid))
        interval, tails = resolved_summaries(a, qs, coarse, masks, ratios, err)
        checks.append(
            dict(
                tc=tc,
                se=err.tolist(),
                ratios=ratios.tolist(),
                coarse=(masks @ coarse).tolist(),
                coarse_q=weighted_quantile(a, coarse, LEVELS).tolist(),
                fine_q=qs.tolist(),
            )
        )
        rows.append([tc, *(interval or [None] * 3), *tails])
    return rows, checks


def run(out, original, manifest):
    private_destination(out)
    grid = next(original.glob("all-grid-review.*/verified-receipt.json")).parent
    receipt = json.loads((grid / "verified-receipt.json").read_text())
    table = grid / "all-settings-review.csv"
    assert hashlib.sha256(table.read_bytes()).hexdigest() == receipt["grid_sha256"]
    require_digest(grid / "mixture.dylib", receipt["scripts"]["mixture.dylib"])
    accepted = pd.read_csv(table)
    diag = json.loads((grid / "precision.json").read_text())
    median = []
    for row in accepted.itertuples():
        if pd.isna(row.lower95_mmhg):
            median.append(np.nan)
            continue
        q = diag[str(row.tc_mmhg)]["endpoints"][1]
        assert (
            q["q"] == q["coarse"]
            and q["cdf_before"] + 2 * q["se_before"] < 0.5 <= q["cdf_at"] - 2 * q["se_at"]
        )
        median.append(q["q"])
    accepted.insert(2, "median", median)
    accepted.to_csv(out / "all-predictions.csv", index=False)
    np.testing.assert_allclose(
        accepted.loc[accepted.tc_mmhg == 47].iloc[0, 1:],
        [35, 44, 53.2, 0.4695596108483035, 0.051765875591940874],
    )
    assert accepted.lower95_mmhg.count() == 69 and len(accepted) == 81
    main_draws = next(original.glob("precision-64000/public-model/*/main/*/posterior.npz"))
    assert hashlib.sha256(main_draws.read_bytes()).hexdigest() == receipt["posterior_sha256"]
    a, w = read_prior(out, "all")
    main_rows, main_checks = inverse(np.load(main_draws)["theta"], a, w, grid / "mixture.dylib")
    np.testing.assert_allclose(
        pd.DataFrame(main_rows).to_numpy(dtype=float),
        accepted.to_numpy(dtype=float),
        atol=1e-11,
        rtol=0,
        equal_nan=True,
    )
    (out / "all-precision.json").write_text(json.dumps(main_checks, allow_nan=False))
    fits = json.loads(manifest.read_text())["strata"]
    for target, group in [("icu", "icu"), ("ed_inp", "arf"), ("pft", "lft")]:
        path = out / (group + ".npz")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fits[group]["posterior_sha256"]
        a, w = read_prior(out, target)
        result, checks = inverse(np.load(path)["theta"], a, w, grid / "mixture.dylib")
        (out / (target + "-precision.json")).write_text(json.dumps(checks, allow_nan=False))
        pd.DataFrame(result, columns=accepted.columns).to_csv(
            out / (target + "-predictions.csv"), index=False
        )
        print(target, "completed", flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    run(*(Path(v).resolve() for v in sys.argv[1:]))
