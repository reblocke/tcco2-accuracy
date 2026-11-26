"""Pointwise risk of arterial hypercapnia given an exact transcutaneous reading."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from cache_private_grid import DRAWS, inputs
from inverse_draws import select_saved
from private_inputs import digest, private_destination, require_digest
from scipy import sparse

from tcco2_accuracy.core._posterior import weighted_quantile
from tcco2_accuracy.core.future_setting import PredictiveKernel, empirical_support

GRID = np.arange(20, 101)
CUTS = (45, 52)
LEVELS = np.array([0.025, 0.975])


class ExactReadingLaw:
    """Project densities at signed offsets to three disjoint arterial classes."""

    def __init__(self, arterial, weights, readings=GRID):
        a, w = empirical_support(arterial, weights)
        self.readings = np.asarray(readings)
        offsets = a[:, None] - self.readings
        self.x, indices = np.unique(offsets, return_inverse=True)
        k = len(self.readings)
        rows = np.searchsorted(CUTS, a, side="right")[:, None] * k + np.arange(k)
        self.project = sparse.csr_matrix(
            (np.broadcast_to(w[:, None], offsets.shape).ravel(), (rows.ravel(), indices.ravel())),
            shape=(3 * k, len(self.x)),
        )

    def joint(self, theta, order):
        kernel = PredictiveKernel(theta, order=order)
        means = kernel.mean.reshape(-1, order)
        inv_sd = 1 / kernel.sd.reshape(means.shape)
        scale = inv_sd * (kernel.weights / np.sqrt(2 * np.pi))
        density = np.empty((len(self.x), len(means)))
        step = max(1, 250_000 // means.size)
        for start in range(0, len(density), step):
            part = slice(start, start + step)
            z = (self.x[part, None, None] - means) * inv_sd
            np.square(z, out=z)
            z *= -0.5
            np.exp(z, out=z)
            density[part] = np.einsum("ijk,jk->ij", z, scale)
        joint = (self.project @ density).T.reshape(*theta.shape[:2], 3, len(self.readings))
        return joint.swapaxes(-1, -2)


def classification_proportions(readings, p45):
    p = np.asarray(p45, dtype=float)
    positive = np.asarray(readings) >= 45
    if p.shape != positive.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid conditional probabilities or reading grid")
    return np.stack(((1 - p) * ~positive, p * ~positive, p * positive, (1 - p) * positive), axis=-1)


def batch_se(score):
    """Monte Carlo SE of a mean using equal-length batches within the four original chains."""
    chains, n = score.shape[:2]
    if chains != 4 or n < 16:
        raise ValueError("Four ordered chains with at least 16 draws are required")
    size = 2 ** int(np.log2(n) // 2)
    batches = score[:, : n // size * size].reshape(-1, size, *score.shape[2:]).mean(axis=1)
    return batches.std(axis=0, ddof=1) / np.sqrt(len(batches))


def conditional(joint):
    total = joint.sum(axis=-1)
    if not (np.isfinite(joint).all() and np.all(joint >= 0) and np.all(total > 0)):
        raise ValueError("Invalid joint density or zero predictive density for the reading")
    tail = np.stack((joint[..., 1:].sum(axis=-1), joint[..., 2]), axis=-1)
    return total, tail / total[..., None]


def summarize(fine, coarse):
    if fine.shape != coarse.shape or fine.ndim != 4 or fine.shape[-1] != 3:
        raise ValueError("Expected matched chain × draw × reading × arterial class arrays")
    mean = np.empty((fine.shape[2], len(CUTS)))
    mean_se = np.empty_like(mean)
    edges = np.empty((*mean.shape, 2))
    mc_radius = np.empty_like(edges)
    quad = np.empty_like(edges)
    for j in range(mean.shape[0]):
        evidence, risks = conditional(fine[:, :, j])
        other_evidence, other_risk = conditional(coarse[:, :, j])
        weights = evidence / evidence.mean()
        mean[j] = (weights[..., None] * risks).mean(axis=(0, 1))
        mean_se[j] = batch_se(weights[..., None] * (risks - mean[j]))
        for h in range(len(CUTS)):
            values = risks[:, :, h]

            def quantile(p):
                return weighted_quantile(values.ravel(), weights.ravel(), p)

            q = quantile(LEVELS)
            edges[j, h] = q
            quad[j, h] = abs(
                q - weighted_quantile(other_risk[:, :, h].ravel(), other_evidence.ravel(), LEVELS)
            )
            for i, (p, edge) in enumerate(zip(LEVELS, q, strict=True)):
                indicator = values <= edge
                actual = (weights * indicator).mean()
                probability_se = batch_se(weights * (indicator - actual))
                bracket = quantile(np.clip(p + np.array([-2, 2]) * probability_se, 0, 1))
                mc_radius[j, h, i] = max(abs(bracket - edge))
    qualified = (mc_radius + quad <= 0.001) & (quad <= 0.0001)
    return dict(
        mean=mean,
        mean_se=mean_se,
        edges=edges,
        mc_radius=mc_radius,
        quadrature=quad,
        qualified=qualified,
    )


def evaluate(theta, law, order, workers):
    chains, n = theta.shape[:2]
    result = np.empty((chains, n, len(law.readings), 3))

    def block(idx):
        chain, start = idx
        portion = theta[chain : chain + 1, start : start + 64]
        result[chain, start : start + len(portion[0])] = law.joint(portion, order)[0]

    positions = ((c, start) for c in range(chains) for start in range(0, n, 64))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(block, positions):
            pass
    return result


def calculate(run, pages, out, group, n, workers):
    private_destination(out)
    implementation = digest(Path(__file__))
    path, sampler, a, w, _ = inputs(run, group)
    receipt = json.loads((pages / "conditional-release.json").read_bytes())
    require_digest(pages / "conditional-lookup.json", receipt["lookup_sha256"])
    model = receipt["fits"][group]
    if sampler["posterior_sha256"] != model["posterior"] or n > model["retained_per_chain"]:
        raise ValueError("Selected fit differs from the released fit or has insufficient draws")
    theta = select_saved(
        path / "theta-cache.npy", sampler["theta_cache_sha256"], model["retained_per_chain"], n
    )
    lookup = json.loads((pages / "conditional-lookup.json").read_bytes())["groups"][group]
    released = np.asarray([row["tails"] for row in lookup["rows"]])
    law = ExactReadingLaw(a, w)
    # Independent full-draw control: locally integrate the saved, averaged likelihood.
    with np.load(run / "reporting-v2" / "cache" / (group + ".npz")) as saved:
        if str(saved["posterior"]) != model["posterior"]:
            raise ValueError("Full-draw cached likelihood belongs to a different posterior")
        np.testing.assert_array_equal(law.x, saved["x"])
        joint = (law.project @ saved["density"]).reshape(3, len(GRID)).T
    _, central = conditional(joint)
    np.testing.assert_allclose(central, released, rtol=0, atol=2e-10)
    print(group, "full-draw controls verified; evaluating saved draws", n, flush=True)
    coarse = evaluate(theta, law, 32, workers)
    fine = evaluate(theta, law, 64, workers)
    stats = summarize(fine, coarse)
    # Subset mean estimates the full-draw line; compare on its own MC scale.
    if np.any(abs(stats["mean"] - released) > 6 * stats["mean_se"] + 2e-10):
        raise ValueError("Conditional weighted mean disagrees with the released probability")
    record = {
        "group": group,
        "posterior": model["posterior"],
        "lookup": receipt["lookup_sha256"],
        "theta": sampler["theta_cache_sha256"],
        "draws_per_chain": n,
        "selection": "equally spaced within each of four ordered chains, including endpoints",
        "code": implementation,
        "selection_code": digest(Path(__file__).with_name("inverse_draws.py")),
        "readings": GRID.tolist(),
        **{k: v.tolist() for k, v in stats.items()},
    }
    target = out / f"{group}-{n}.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, allow_nan=False))
    os.replace(tmp, target)
    print(
        group,
        int(stats["qualified"].sum()),
        "of",
        stats["qualified"].size,
        "band edges qualified",
        flush=True,
    )


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "pages", "out"):
        p.add_argument(key, type=lambda s: Path(s).resolve())
    p.add_argument("--group", choices=DRAWS, required=True)
    p.add_argument("--draws", dest="n", type=int, default=8192)
    p.add_argument("--workers", type=int, default=6)
    calculate(**vars(p.parse_args()))
