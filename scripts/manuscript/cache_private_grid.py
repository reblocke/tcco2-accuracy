"""Materialize row-shared order-64 likelihoods from receipt-bound saved chains on the mini."""

import argparse
import ctypes
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from private_inputs import digest, private_destination, read_prior, require_digest
from scipy.special import roots_hermitenorm

DRAWS = {"all": 16384000, "icu": 8192000, "ed_inp": 32768000, "pft": 65536000}


def load(path):
    return json.loads(path.read_bytes())


def inputs(run, setting):
    root = run / "pft-accept099" if setting == "pft" else run
    n = DRAWS[setting]
    plan = load(root / "plan.json")
    path = root / "public-draws" / setting / f"draws-{n}"
    receipt = load(path / "sampler.json")
    if not (
        receipt["sampler_pass"]
        and receipt["chains"] == 4
        and receipt["seed"] == 88032
        and receipt["retained_per_chain"] == n
        and receipt["divergences"] == 0
        and receipt["adapt_delta"] == (0.99 if setting == "pft" else 0.95)
        and all(
            d["R_hat"] < 1.01 and min(d["ESS_bulk"], d["ESS_tail"]) >= 400
            for d in receipt["diagnostics"].values()
        )
    ):
        raise ValueError("Unqualified archived fit")
    require_digest(run.parent / (setting + "-weights.csv"), plan["models"][setting]["prior_sha256"])
    a, w = read_prior(run.parent, setting)
    grid = root / "grids" / setting / f"draws-{n}" / "screen"
    return path, receipt, a, w, grid


def generate(run, setting, output, library, workers):
    private_destination(output.resolve())
    if not output.resolve().is_relative_to(run.resolve()):
        raise ValueError("Keep empirical density caches inside the private analysis run")
    path, receipt, a, _, _ = inputs(run, setting)
    require_digest(path / "posterior.npz", receipt["posterior_sha256"])
    require_digest(path / "theta-cache.npy", receipt["theta_cache_sha256"])
    theta = np.load(path / "theta-cache.npy", mmap_mode="r")
    if theta.shape != (4, DRAWS[setting], 5):
        raise ValueError("Unexpected chain dimensions")
    x = np.unique(a[None, :] - np.arange(20, 101)[:, None])
    z, w = roots_hermitenorm(64)
    w /= w.sum()
    fn = ctypes.CDLL(str(library.resolve())).cached_density
    vec = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    fn.argtypes = [vec, ctypes.c_int, vec, vec, ctypes.c_int, vec, ctypes.c_int, vec]
    fn.restype = ctypes.c_int
    key = hashlib.sha256(
        (receipt["theta_cache_sha256"] + digest(library)).encode()
        + x.tobytes()
        + z.tobytes()
        + w.tobytes()
    ).hexdigest()
    folder = output / setting / key
    folder.mkdir(parents=True, exist_ok=True)
    step = 256000

    def batch(item):
        chain, start = item
        chunk = theta[chain, start : start + step]
        file = folder / f"{chain}-{start}.npy"
        if not file.exists():
            density = np.zeros_like(x)
            if fn(chunk, len(chunk), z, w, len(z), x, len(x), density):
                raise ValueError("Invalid normal component")
            with tempfile.NamedTemporaryFile(dir=folder, delete=False) as f:
                np.save(f, density)
            os.replace(f.name, file)
        density = np.load(file, allow_pickle=False)
        if density.shape != x.shape or not np.isfinite(density).all():
            raise ValueError("Invalid density cache")
        return len(chunk) * density

    total = np.zeros_like(x)
    jobs = [(c, i) for c in range(4) for i in range(0, theta.shape[1], step)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for values in pool.map(batch, jobs):
            total += values
    mean = total / (theta.shape[0] * theta.shape[1])
    if not np.all(mean > 0):
        raise ValueError("Invalid marginal density")
    with tempfile.NamedTemporaryFile(dir=output, delete=False) as f:
        np.savez(f, x=x, density=mean, posterior=receipt["posterior_sha256"], cache_key=key)
    os.replace(f.name, output / (setting + ".npz"))
    print(setting, "verified saved draws and cached likelihood", flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("run", "output", "library"):
        parser.add_argument(k, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for setting in DRAWS:
        generate(args.run, setting, args.output, args.library, args.workers)
