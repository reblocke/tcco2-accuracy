"""Reconcile the saved full-support curves with the independently audited reporting brackets."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from cache_private_grid import DRAWS, inputs, load
from exhibits import render
from private_inputs import digest, private_destination, require_digest
from scipy.special import logsumexp

from tcco2_accuracy.core.conditional_reporting import display_range
from tcco2_accuracy.pages_prediction import evidence_masks
from tcco2_accuracy.workflows.private_chart import display_curves
from tcco2_accuracy.workflows.private_preview_html import private_viewer_html

CERTIFICATES = {
    "all": "d9551152c910abd4f8433f1bf3dc85322ca0d2a9a13f9f706e763074ff6f80a3",
    "icu": "962ad118358bdb9aae9d8724cb5ea6562905f3b6365d1c345b13c7b52f0b4ce5",
    "ed_inp": "cb37c6ba900d6303806a6ffc6cc59e7a568ed042150549f0349d73dfd9fa51a4",
    "pft": "798ca1d3fe0132eac8d8f09aadba4cb6ad3d5c01822577b26ad969fcddb8f895",
}


def build(run, dest, *, curves):
    private_destination(dest)
    if not dest.is_relative_to(run):
        raise ValueError("Use a private output directory within the archived run")
    audit = run / "reporting-audit"
    checks = load(audit / "split-validation.json")
    if (
        set(checks) != set(DRAWS)
        or sum(v["passed"] for v in checks.values()) != 48
        or any(v["passed"] != v["separate_half_cut_checks"] for v in checks.values())
    ):
        raise ValueError("Incomplete saved split checks")
    groups = {}
    for setting, n in DRAWS.items():
        _, receipt, a, w, screen = inputs(run, setting)
        cert = audit / f"{setting}-{n}-private-certificates.json"
        require_digest(cert, CERTIFICATES[setting])
        grid = load(screen / "precision.json")
        rows = []
        data = []
        if curves:
            with np.load(dest / "cache" / (setting + ".npz")) as arr:
                if str(arr["posterior"]) != receipt["posterior_sha256"]:
                    raise ValueError("Curve posterior mismatch")
                x, density = arr["x"], arr["density"]
        for t, r, c in zip(range(20, 101), grid, load(cert), strict=True):
            if t != r["tc"] or t != c["t"]:
                raise ValueError("Unordered reading")
            q = r["fine_q"]
            np.testing.assert_array_equal(q, c["q"])
            limits = c["limits"]
            p = np.array(r["ratios"][:2])
            delta = np.abs(p - np.array(r["coarse"][:2]))
            rad = 2 * np.array(r["se"][:2]) + delta
            np.testing.assert_allclose(100 * rad, c["tail_radius_pp"], atol=1e-9, rtol=0)
            np.testing.assert_allclose(100 * delta, c["tail_quad_pp"], atol=1e-9, rtol=0)
            if not all(
                lo <= v <= hi and hi - lo <= 0.5 for (lo, hi), v in zip(limits, q, strict=True)
            ) or not np.all((rad <= 0.001) & (delta <= 0.0001)):
                raise ValueError("Numerical precision hold in audited release")
            bounds = np.column_stack((np.maximum(0, p - rad), np.minimum(1, p + rad))).tolist()
            report = dict(quantiles=limits, probabilities=bounds)
            data.append(
                [
                    t,
                    *q,
                    *p,
                    *[display_range(v) for v in limits],
                    *[display_range(v, percent=True) for v in bounds],
                ]
            )
            if curves:
                ix = np.searchsorted(x, a - t)
                np.testing.assert_array_equal(x[ix], a - t)
                like = density[ix]
                joint = np.log(w) + np.log(like)
                mass = np.exp(joint - logsumexp(joint))
                np.testing.assert_array_equal(
                    a[np.searchsorted(mass.cumsum(), [0.025, 0.5, 0.975])], q
                )
                np.testing.assert_allclose(
                    evidence_masks(a, q, [45, 52]) @ mass, r["ratios"], atol=2e-10, rtol=0
                )
                rows.append(
                    dict(
                        interval=q,
                        tails=p.tolist(),
                        report=report,
                        mass=mass.tolist(),
                        likelihood=(like / like.max()).tolist(),
                        display=display_curves(a, w, mass),
                    )
                )
        pd.DataFrame(
            data,
            columns=[
                "tc_mmhg",
                "lower",
                "median",
                "upper",
                "p_ge45",
                "p_ge52",
                "lower_display",
                "median_display",
                "upper_display",
                "p45_display",
                "p52_display",
            ],
        ).to_csv(dest / (setting + "-predictions.csv"), index=False)
        if curves:
            groups[setting] = dict(
                support=a.tolist(),
                prior=w.tolist(),
                posterior=receipt["posterior_sha256"],
                rows=rows,
            )
    render(run.parent, dest, grid=dest)
    if curves:
        doc = dict(
            version="conditional_lookup_private_chart_v2",
            method="future_setting_joint_v1",
            prior_uncertainty="fixed_empirical",
            source="trinetx",
            index_policy="equal_patient_weight_split_across_eligible_draws_on_first_testing_day",
            groups=groups,
        )
        chart = dest / "private-chart.json"
        tmp = dest / "private-chart.tmp"
        tmp.write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False))
        tmp.replace(chart)
        (dest / "preview.html").write_text(private_viewer_html(digest(chart)))
    print("Checked all four settings and 81 readings per setting.", flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--curves", action="store_true")
    args = parser.parse_args()
    build(args.run.resolve(), args.output.resolve(), curves=args.curves)
