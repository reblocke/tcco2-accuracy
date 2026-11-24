"""Compute the private rounded-display joint law from authenticated saved fits."""

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from private_inputs import digest, private_destination, read_prior
from strategy import classification_draws

from tcco2_accuracy.core import future_setting, referral_policy


def numerical_sources():
    return {
        "code": digest(Path(__file__)),
        "integral": digest(Path(referral_policy.__file__)),
        "predictive_kernel": digest(Path(future_setting.__file__)),
    }


def check_numerical_sources(receipt):
    for name, expected in numerical_sources().items():
        if receipt["inputs"].get(name) != expected:
            raise ValueError(f"Cached numerical implementation differs for {name}")


def calculate(theta, integral, order, workers, *, midpoint=False):
    n = theta.shape[1]
    positions = [(c, start) for c in range(4) for start in range(0, n, 64)]

    def block(position):
        c, start = position
        kernel = future_setting.PredictiveKernel(theta[c : c + 1, start : start + 64], order=order)
        values = integral.point_block(kernel) if midpoint else integral.block(kernel)
        return c, start, values[0]

    result = np.empty((*theta.shape[:2], 2, 81 if midpoint else 83))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for c, start, batch in pool.map(block, positions):
            result[c, start : start + len(batch)] = batch
    return result


def run(store, original, folder, setting, per_chain, order, workers):
    private_destination(folder)
    manifest = store / "manifest.json"
    for group, theta in classification_draws(
        store, original, manifest, per_chain=per_chain, setting=setting
    ):
        a, w = read_prior(store, group)
        stem = f"{group}-{per_chain}-{order}"
        dest = folder / (stem + ".npy")
        receipt = folder / (stem + ".json")
        inputs = {
            "theta": hashlib.sha256(theta.tobytes()).hexdigest(),
            "prior": digest(store / (group + "-weights.csv")),
            "manifest": digest(manifest),
            "per_chain": per_chain,
            "selection": "four original chains, equally spaced including both endpoints",
            "order": order,
            **numerical_sources(),
        }
        if receipt.exists():
            saved = json.loads(receipt.read_bytes())
            if saved["inputs"] == inputs and saved["output"] == digest(dest):
                continue
            raise ValueError("Existing cache has different inputs or bytes")
        result = calculate(theta, referral_policy.DisplayIntegral(a, w), order, workers)
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as f:
            np.save(f, result)
        os.replace(f.name, dest)
        receipt.write_text(json.dumps({"inputs": inputs, "output": digest(dest)}, indent=2))
        print(group, per_chain, order, "joint mass verified", flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("store", "original", "output"):
        parser.add_argument(arg, type=lambda s: Path(s).resolve())
    parser.add_argument("--setting", choices=["all", "icu", "ed_inp", "pft"])
    parser.add_argument("--draws", type=int, default=2048, help="Retained per chain")
    parser.add_argument("--order", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for order in args.order:
        run(args.store, args.original, args.output, args.setting, args.draws, order, args.workers)
