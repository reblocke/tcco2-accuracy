"""Build the public drawing coordinates on the mini from the verified private grid."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

X = np.arange(4001) / 16
EDGES = np.unique(np.r_[np.arange(0, 251, 2), 45])
SCALE = 100000


def drawing(arterial, masses, tails):
    cumulative = np.r_[0, np.cumsum(masses)]
    cuts = np.round(cumulative[np.searchsorted(arterial, EDGES, side="left")], 3)
    cuts[-1] = 1
    density = PchipInterpolator(EDGES, cuts).derivative()(X)
    ordinates = np.rint(density * SCALE).astype(int)
    total = np.trapezoid(ordinates / SCALE, X)
    if ordinates.min() < 0 or abs(total - 1) > 0.001:
        raise ValueError("Invalid plot area")
    for cutoff, expected in zip((45, 52), tails, strict=True):
        mask = X >= cutoff
        area = np.trapezoid(ordinates[mask] / SCALE, X[mask])
        if max(abs(area - expected), abs(area / total - expected)) > 0.001:
            raise ValueError("Plotted tail differs from verified inclusive probability")
    return ordinates.tolist()


def export(folder):
    lookup = (folder / "conditional-lookup.json").read_bytes()
    reference = json.loads(lookup)
    private = json.loads((folder / "private-chart.json").read_bytes())
    groups = {}
    for setting, accepted in reference["groups"].items():
        group = private["groups"][setting]
        if group["posterior"] != accepted["posterior"]:
            raise ValueError("Posterior mismatch")
        a = np.array(group["support"])
        w = np.array(group["prior"])
        with np.load(folder / "cache" / (setting + ".npz")) as cache:
            if str(cache["posterior"]) != accepted["posterior"]:
                raise ValueError("Offset cache mismatch")
            offsets, density = cache["x"], cache["density"]
        # Regular signed offsets within the fitted cache. No empirical abscissae exported.
        mesh = np.arange(np.ceil(offsets.min() * 4), np.floor(offsets.max() * 4) + 1) / 4
        kernel = np.interp(mesh, offsets, density)
        rows = []
        for t, (r, ref) in enumerate(zip(group["rows"], accepted["rows"], strict=True), 20):
            if any(r[k] != ref[k] for k in ("interval", "tails", "report")):
                raise ValueError("Private chart differs from approved row")
            mass = np.array(r["mass"])
            like = np.interp(a - t, offsets, density)
            np.testing.assert_allclose(like * w / (like @ w), mass, atol=2e-10, rtol=0)
            lo, hi = a[np.searchsorted(mass.cumsum(), [0.0005, 0.9995])]
            rows.append(
                dict(
                    view=[int(np.floor(min(45, lo) / 5) * 5), int(np.ceil(max(52, hi) / 5) * 5)],
                    density=drawing(a, mass, ref["tails"]),
                )
            )
        groups[setting] = dict(
            posterior=accepted["posterior"],
            prior_density=drawing(a, w, [w[a >= k].sum() for k in (45, 52)]),
            offset_start=mesh[0],
            relative=np.rint(kernel / kernel.max() * SCALE).astype(int).tolist(),
            rows=rows,
        )
    result = dict(
        version="trinetx_display_v1",
        lookup_sha256=hashlib.sha256(lookup).hexdigest(),
        groups=groups,
    )
    (folder / "prediction-plots.json").write_text(
        json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    export(parser.parse_args().folder)
