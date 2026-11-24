"""Compare first-testing-day populations at TcCO2 47 on the same verified saved fit."""

import argparse
import ctypes
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from cache_private_grid import DRAWS, inputs
from direct_density import direct_batch_density
from private_inputs import digest, private_destination, require_digest
from scipy.special import logsumexp, roots_hermitenorm

from tcco2_accuracy.core.conditional_reporting import display_range, qualify


def compare(run, cohort, output, group, workers):
    private_destination(output)
    path, receipt, arterial, weights, _ = inputs(run, group)
    fit = path / "theta-cache.npy"
    require_digest(fit, receipt["theta_cache_sha256"])
    theta = np.load(fit, mmap_mode="r")
    if theta.shape != (4, DRAWS[group], 5):
        raise ValueError("Invalid saved chain dimensions")
    populations = {"archived": (arterial, weights)}
    fingerprints = {}
    for variant in ("reason-first", "setting-first"):
        file = cohort / f"{variant}-{group}-weights.csv"
        table = pd.read_csv(file, float_precision="round_trip")
        a, w = table.paco2_bin.to_numpy(), table.weight.to_numpy()
        if not (np.all(np.diff(a) > 0) and np.all(w > 0) and abs(w.sum() - 1) < 1e-12):
            raise ValueError("Invalid first-day distribution")
        if variant == "reason-first":
            np.testing.assert_array_equal(a, arterial)
            np.testing.assert_allclose(w, weights, atol=1e-14, rtol=0)
        populations[variant] = (a, w)
        fingerprints[variant] = digest(file)
    offsets = np.unique(np.concatenate([a - 47 for a, _ in populations.values()]))
    root = run / "reporting-v2"
    lib = root / "cache.dylib"
    fn = ctypes.CDLL(str(lib)).cached_density
    vec = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    fn.argtypes = [vec, ctypes.c_int, vec, vec, ctypes.c_int, vec, ctypes.c_int, vec]
    fn.restype = ctypes.c_int
    nodes = {}
    for order in (32, 64):
        z, w = roots_hermitenorm(order)
        nodes[order] = z, w / w.sum()

    def density(chunk, x, order):
        z, w = nodes[order]
        values = np.zeros_like(x)
        if fn(chunk, len(chunk), z, w, order, x, len(x), values):
            raise ArithmeticError("Invalid mixture batch")
        return values

    pilot = np.ascontiguousarray(theta[:, :128].reshape(-1, 5))
    direct = direct_batch_density(root / "direct.dylib")
    for order in nodes:
        np.testing.assert_allclose(
            density(pilot, offsets, order),
            np.exp(direct(pilot, order, offsets)),
            atol=1e-14,
            rtol=0,
        )
    with np.load(root / "cache" / (group + ".npz")) as grid:
        xp = grid["x"]
        z, w = nodes[64]
        key = hashlib.sha256(
            (receipt["theta_cache_sha256"] + digest(lib)).encode()
            + xp.tobytes()
            + z.tobytes()
            + w.tobytes()
        ).hexdigest()
        if str(grid["posterior"]) != receipt["posterior_sha256"] or key != str(grid["cache_key"]):
            raise ValueError("Density cache does not match fit and quadrature")
    ix = np.minimum(np.searchsorted(xp, offsets), len(xp) - 1)
    exists = xp[ix] == offsets
    step = 256000
    if theta.shape[1] % step:
        raise ValueError("Unequal chain batches")

    def batch(item):
        c, i = item
        chunk = theta[c, i : i + step]
        fine = np.empty_like(offsets)
        fine[exists] = np.load(root / "cache" / group / key / f"{c}-{i}.npy")[ix[exists]]
        if not exists.all():
            fine[~exists] = density(chunk, offsets[~exists], 64)
        return density(chunk, offsets, 32), fine

    jobs = [(c, i) for c in range(4) for i in range(0, theta.shape[1], step)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        data = np.array(list(pool.map(batch, jobs)))
    if not np.isfinite(data).all() or np.any(data < 0):
        raise ArithmeticError("Invalid batch density")
    coarse, fine = data.mean(axis=0)
    ref = (
        pd.read_csv(root / (group + "-predictions.csv"), float_precision="round_trip")
        .set_index("tc_mmhg")
        .loc[47]
    )
    results = {}
    for name, (a, w) in populations.items():
        idx = np.searchsorted(offsets, a - 47)
        log_joint = np.log(w) + np.log(fine[idx])
        log_z = logsumexp(log_joint)
        mass = np.exp(log_joint - log_z)
        cm = w * coarse[idx]
        batches = data[:, 1, idx] * w / np.exp(log_z)
        q, p, bounds, _ = qualify(
            a, mass, cm / cm.sum(), [45, 52], [batches], interpolation_error=4e-14 / np.exp(log_z)
        )
        if name == "archived":
            np.testing.assert_allclose(
                [*q, *p],
                ref[["lower", "median", "upper", "p_ge45", "p_ge52"]].to_numpy(dtype=float),
                rtol=0,
                atol=2e-10,
            )
        results[name] = {
            "baseline_percent": [100 * float(w[a >= t].sum()) for t in (45, 52)],
            "quantiles": [display_range(v) for v in bounds["quantiles"]],
            "probabilities": [display_range(v, percent=True) for v in bounds["probabilities"]],
        }
    (output / f"{group}-cohort-prediction.json").write_text(
        json.dumps(
            {
                "setting": group,
                "results": results,
                "posterior": receipt["posterior_sha256"],
                "population_sha256": fingerprints,
                "batch_count": len(jobs),
            },
            indent=2,
        )
    )
    print(group, "archived prediction reproduced and both selection orders evaluated", flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "cohort", "output"):
        parser.add_argument(name, type=lambda s: Path(s).resolve())
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--setting", choices=list(DRAWS))
    args = parser.parse_args()
    for group in [args.setting] if args.setting else DRAWS:
        compare(args.run, args.cohort, args.output, group, args.workers)
