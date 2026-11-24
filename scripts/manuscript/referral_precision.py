"""Refine rounded-bin risks using certified point-reading risks and a paired rounding correction."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from cache_private_grid import DRAWS, inputs
from private_inputs import digest, private_destination, require_digest
from referral_analysis import calculate
from reporting_grid import CERTIFICATES
from strategy import classification_draws

from tcco2_accuracy.core.referral_policy import DISPLAY, DisplayIntegral, corrected_bin_risk


def precision(store, original, run, cache, counts, workers):
    private_destination(cache)
    receipts = {}
    for group, n in zip(DRAWS, counts, strict=True):
        _, _, a, w, screen = inputs(run, group)
        cert = run / "reporting-audit" / f"{group}-{DRAWS[group]}-private-certificates.json"
        require_digest(cert, CERTIFICATES[group])
        table = json.loads((screen / "precision.json").read_bytes())
        checks = json.loads(cert.read_bytes())
        if [r["tc"] for r in table] != DISPLAY.tolist() or [
            r["t"] for r in checks
        ] != DISPLAY.tolist():
            raise ValueError("Unexpected point grid")
        small_se = np.array([r["se"][0] for r in table])
        high = np.array([r["ratios"][0] for r in table])
        low = np.array([r["coarse"][0] for r in table])
        np.testing.assert_allclose(
            100 * (2 * small_se + abs(high - low)),
            [r["tail_radius_pp"][0] for r in checks],
            rtol=0,
            atol=1e-9,
        )
        published = np.genfromtxt(
            run / "reporting-v2" / (group + "-predictions.csv"), delimiter=",", names=True
        )
        np.testing.assert_array_equal(published["tc_mmhg"], DISPLAY)
        np.testing.assert_allclose(high, published["p_ge45"], atol=1e-12, rtol=0)
        _, theta = next(
            classification_draws(
                store, original, store / "manifest.json", per_chain=n, setting=group
            )
        )
        integral = DisplayIntegral(a, w)
        estimates = []
        sources = {}
        for order, baseline in [(32, low), (64, high)]:
            stem = f"{group}-{n}-{order}"
            path = cache / (stem + ".npy")
            receipt = json.loads((cache / (stem + ".json")).read_bytes())
            require_digest(path, receipt["output"])
            if receipt["inputs"]["prior"] != digest(store / (group + "-weights.csv")):
                raise ValueError("Rounding correction has a different arterial distribution")
            if receipt["inputs"]["theta"] != hashlib.sha256(theta.tobytes()).hexdigest():
                raise ValueError("Rounding correction requires the same chain/draw selection")
            joint = np.load(path, mmap_mode="r")
            point = calculate(theta, integral, order, workers, midpoint=True)
            estimates.append(corrected_bin_risk(joint, point, baseline, small_se))
            sources[str(order)] = receipt
        q, se = estimates[-1]
        delta = abs(q - estimates[0][0])
        radius = 2 * se + delta
        if not (
            np.isfinite(radius).all()
            and (delta <= 0.0001).all()
            and (radius <= 0.001).all()
            and ((q >= 0) & (q <= 1)).all()
        ):
            raise ValueError(f"Unqualified rounded display certificate for {group}")
        np.savez(
            cache / f"{group}-bin-certificate.npz",
            q=q,
            lower=(q - radius).clip(0, 1),
            upper=(q + radius).clip(0, 1),
            se=se,
            quadrature=delta,
            per_chain=n,
        )
        receipts[group] = dict(
            point_certificate=digest(cert),
            point_source=digest(screen / "precision.json"),
            output=digest(cache / f"{group}-bin-certificate.npz"),
            sources=sources,
        )
        print(group, "rounded risks qualified", flush=True)
    (cache / "bin-certificates.json").write_text(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "original", "run", "cache"):
        p.add_argument(name, type=lambda s: Path(s).resolve())
    p.add_argument("--counts", type=int, nargs=4, default=[8192, 8192, 32768, 32768])
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    precision(**vars(args))
