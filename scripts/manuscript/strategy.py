"""Conditional operating characteristics for a fixed arterial distribution, by posterior draw."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from private_inputs import model_receipts, private_destination, read_prior, require_digest

from tcco2_accuracy.core.future_setting import PredictiveKernel


def metrics(pos, neg):
    # Each array holds joint probability for a true arterial class and a Tc reading >= c.
    total_pos, s40, s45, s50 = pos
    total_neg, n40, n45, n50 = neg
    tp, fp, fn, tn = s45, n45, total_pos - s45, total_neg - n45
    pos_lo = total_pos - s40
    neg_lo = total_neg - n40
    reflex = s40 - s50 + n40 - n50
    sens = tp / total_pos
    spec = tn / total_neg
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "sensitivity": sens,
        "specificity": spec,
        "PPV": tp / (tp + fp),
        "NPV": tn / (tn + fn),
        "LR+": sens / (1 - spec),
        "LR-": (1 - sens) / spec,
        "low": pos_lo + neg_lo,
        "reflex_ABG": reflex,
        "high": s50 + n50,
        "residual_FN": pos_lo,
        "residual_FP": n50,
        "classified_accuracy": (s50 + neg_lo) / (1 - reflex),
        "classified_sensitivity": s50 / (s50 + pos_lo),
        "classified_specificity": neg_lo / (neg_lo + n50),
        "whole_strategy_accuracy": 1 - pos_lo - n50,
    }


def classification_draws(out, original, manifest, *, per_chain=2048, setting=None):
    """Yield the checked, evenly spaced posterior draws used for secondary analyses."""
    fits, _, all_receipt = model_receipts(original, manifest)
    for group, model in [("all", "main"), ("icu", "icu"), ("ed_inp", "arf"), ("pft", "lft")]:
        if setting is not None and setting != group:
            continue
        path = (
            next(original.glob("precision-64000/public-model/*/main/*/posterior.npz"))
            if model == "main"
            else out / (model + ".npz")
        )
        require_digest(
            path,
            all_receipt["posterior_sha256"] if model == "main" else fits[model]["posterior_sha256"],
        )
        with np.load(path) as posterior:
            theta = posterior["theta"]
        if theta.shape[0] != 4 or not 2 <= per_chain <= theta.shape[1]:
            raise ValueError("Requested draws are unavailable in the four saved chains")
        yield group, theta[:, np.linspace(0, theta.shape[1] - 1, per_chain, dtype=int)]


def run(out, original, manifest):
    private_destination(out)
    summaries = {}
    for group, theta in classification_draws(out, original, manifest):
        a, w = read_prior(out, group)
        mask = a >= 45
        by_class = []
        for values, mass in [(a[mask], w[mask]), (a[~mask], w[~mask])]:
            sums = np.zeros((4, 4, 2048))
            sums[0] = mass.sum()
            kernel = PredictiveKernel(theta, order=64)
            cache = {}
            for j, cut in enumerate((40, 45, 50), 1):
                for arterial, weight in zip(values, mass, strict=True):
                    offset = arterial - cut
                    if offset not in cache:
                        cache[offset] = kernel.draw_cdf(offset)
                    sums[j] += weight * cache[offset]
            by_class.append(sums)
        draws = metrics(*by_class)
        np.testing.assert_allclose(sum(draws[k] for k in ("TP", "FP", "FN", "TN")), 1)
        np.testing.assert_allclose(sum(draws[k] for k in ("low", "reflex_ABG", "high")), 1)
        summaries[group] = {
            name: np.quantile(v, [0.025, 0.5, 0.975]).tolist() for name, v in draws.items()
        }
        print(group, "completed", flush=True)
    (out / "strategy.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == "__main__":
    os.umask(0o077)
    run(*(Path(p).resolve() for p in sys.argv[1:]))
