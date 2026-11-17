"""Render only manuscript exhibits on the private host. Full-support arrays stay there."""

import json
import sys
from pathlib import Path

import matplotlib
from private_inputs import read_prior

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUPS = {
    "all": "All settings",
    "icu": "ICU proxy",
    "ed_inp": "ED / inpatient",
    "pft": "Ambulatory",
}
READINGS = [35, 40, 45, 47, 50, 55]


def table(headers, rows):
    return (
        "\n".join(
            "| " + " | ".join(map(str, r)) + " |" for r in [headers, ["---"] * len(headers), *rows]
        )
        + "\n"
    )


def render(private, dest, *, grid=None):
    dest.mkdir(exist_ok=True)
    data = json.loads((private / "cohort.json").read_text())
    strategy = json.loads((private / "strategy.json").read_text())
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.8), sharex=True, sharey=True, layout="constrained")
    pred, panels = plt.subplots(
        4, 2, figsize=(8.2, 9), sharex=True, sharey="col", layout="constrained"
    )
    examples = []
    quality = []
    for (key, name), ax, (left, right) in zip(GROUPS.items(), axes.flat, panels, strict=True):
        a, w = read_prior(private, key)
        edges = np.arange(0, 106, 5)
        bins = np.histogram(np.minimum(a, 102), edges, weights=w)[0]
        np.testing.assert_allclose(bins.sum(), 1)
        ax.bar(edges[:-1], 100 * bins, width=4.7, align="edge", color="#47776e")
        ax.set_title(name, loc="left", fontsize=12)
        for c, style in [(45, "--"), (52, ":")]:
            ax.axvline(c, color="#252525", ls=style)
        ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100+"])
        d = pd.read_csv((grid or private) / (key + "-predictions.csv")).set_index("tc_mmhg")
        if list(d.index) != list(range(20, 101)):
            raise ValueError(f"Incomplete or unordered prediction grid for {key}")
        lower, median, upper, p, q = d.iloc[:, :5].to_numpy().T
        t = d.index.to_numpy()
        tails = np.stack([p, q])
        if np.any((tails < 0) | (tails > 1)) or np.any(q > p):
            raise ValueError(f"Invalid threshold probabilities for {key}")
        left.fill_between(t, lower, upper, color="#cadae0")
        left.plot(t, median, ".-", markersize=2, color="#173e53")
        left.plot(t, lower, ".", ms=1.8, color="#476273")
        left.plot(t, upper, ".", ms=1.8, color="#476273")
        right.plot(t, p * 100, ".-", ms=2, color="#15574b", label="≥45")
        right.plot(t, q * 100, ".--", ms=2, color="#743e28", label="≥52")
        left.set_ylabel(name + "\nPaCO₂ (mmHg)")
        right.set_ylim(-3, 103)
        quality.append(
            [
                name,
                int(np.sum(np.isfinite(lower) & np.isfinite(upper))),
                int(d.p_ge45.count()),
                int(d.p_ge52.count()),
            ]
        )
        for tc in READINGS:
            if grid is not None:
                lo, mid, hi, p45, p52 = d.loc[tc].iloc[5:]
                examples.append([name, tc, mid, f"{lo}–{hi}", p45, p52])
                continue
            lo, mid, hi, p45, p52 = d.loc[tc]
            examples.append(
                [
                    name,
                    tc,
                    f"{mid:g}" if np.isfinite(mid) else "Held",
                    f"{lo:g}–{hi:g}" if np.isfinite(lo) else "Held",
                    f"{100 * p45:.2f}%" if np.isfinite(p45) else "Held",
                    f"{100 * p52:.2f}%" if np.isfinite(p52) else "Held",
                ]
            )
    fig.supxlabel("Arterial PaCO₂ (mmHg)")
    fig.supylabel("Patient-weighted probability per bin (%)")
    fig.suptitle("Arterial CO₂ on the first eligible testing day")
    fig.savefig(dest / "arterial-distribution.png", dpi=175)
    panels[0, 0].set_title("Median and 95% prediction interval", fontsize=12)
    panels[0, 1].set_title("Probability of arterial hypercapnia (%)", fontsize=12)
    panels[0, 1].legend(loc="lower right", frameon=False)
    pred.supxlabel("Observed TcCO₂ (mmHg)")
    pred.savefig(dest / "paired-prediction.png", dpi=175)
    plt.close("all")
    flow = []
    for key, name in GROUPS.items():
        v = data[key]
        f = v["flow"]
        flow.append(
            [name, *[f"{n:,}" for n in f["eligible"]], *[f"{n:,}" for n in f["weighted"][1:]]]
        )
    text = "# Cohort and paired predictions\n\n" + table(
        [
            "Setting",
            "Eligible patients",
            "Eligible encounters",
            "Eligible draws",
            "First-day encounters",
            "First-day draws",
        ],
        flow,
    )
    text += "\nEach patient contributes total weight one within the target. Settings can overlap across a patient’s encounters. All uses the earliest eligible day across the entire cohort.\n\n"
    chars = []
    for col, label in {
        "age_at_encounter": "Age, median (IQR)",
        "female": "Female",
        "male": "Male",
        "white_race": "White race",
        "black_race": "Black race",
        "asian_race": "Asian race",
        "hisp_eth": "Hispanic ethnicity",
        "copd": "COPD",
        "asthma": "Asthma",
        "osa": "Obstructive sleep apnea",
        "chf": "Heart failure",
        "ckd": "Chronic kidney disease",
        "dm": "Diabetes",
        "oud": "Opioid use disorder",
        "phtn": "Pulmonary hypertension",
    }.items():
        cells = []
        for key in GROUPS:
            obj = data[key]["characteristics"][col]
            v = obj["value"]
            label_value = (
                f"{v[1]:g} ({v[0]:g}–{v[2]:g})" if isinstance(v, list) else f"{100 * v:.1f}%"
            )
            if obj["known_fraction"] < 0.9995:
                label_value += f" ({100 * obj['known_fraction']:.1f}% observed)"
            cells.append(label_value)
        chars.append([label, *cells])
    text += table(["Characteristic", *GROUPS.values()], chars)
    text += "\nAge and coded characteristics use first-day draw weights and the associated encounter. Percentages use nonmissing values. Race and ethnicity are separate recorded indicators.\n\n"
    pre = []
    for key, name in GROUPS.items():
        v = data[key]
        lo, med, hi = v["prior_interval"]
        p, q = v["prior_thresholds"]
        pre.append([name, f"{med:g}", f"{lo:g}–{hi:g}", f"{100 * p:.2f}%", f"{100 * q:.2f}%"])
    text += "## Before observing TcCO₂\n\n" + table(
        ["Setting", "Median", "Central 95% range (mmHg)", "PaCO₂ ≥45", "PaCO₂ ≥52"], pre
    )
    text += "\n## Given observed TcCO₂\n\n" + table(
        ["Setting", "TcCO₂", "Median PaCO₂", "95% prediction interval", "PaCO₂ ≥45", "PaCO₂ ≥52"],
        examples,
    )
    text += (
        "\nPressures are in mmHg. Held denotes a result that did not meet the numerical precision criteria.\n\n"
        + table(["Qualified readings out of 81", "Interval", "≥45", "≥52"], quality)
    )
    (dest / "cohort-and-prediction.md").write_text(text)
    metric_names = {
        "TP": "True positives per 100",
        "FP": "False positives per 100",
        "FN": "False negatives per 100",
        "TN": "True negatives per 100",
        "sensitivity": "Sensitivity (%)",
        "specificity": "Specificity (%)",
        "PPV": "Positive predictive value (%)",
        "NPV": "Negative predictive value (%)",
        "LR+": "Positive likelihood ratio",
        "LR-": "Negative likelihood ratio",
        "low": "TcCO₂ <40 (%)",
        "reflex_ABG": "Reflex ABG, 40–50 (%)",
        "high": "TcCO₂ >50 (%)",
        "residual_FN": "Residual false negatives per 100 eligible",
        "residual_FP": "Residual false positives per 100 eligible",
        "classified_sensitivity": "Sensitivity among classified (%)",
        "classified_specificity": "Specificity among classified (%)",
        "classified_accuracy": "Accuracy among classified (%)",
        "whole_strategy_accuracy": "Overall strategy accuracy (%)",
    }
    rows = []
    for k, label in metric_names.items():
        factor = 1 if k.startswith("LR") else 100
        vals = []
        for group in GROUPS:
            lo, m, hi = np.array(strategy[group][k]) * factor
            vals.append(f"{m:.2f} ({lo:.2f}–{hi:.2f})")
        rows.append([label, *vals])
    (dest / "classification.md").write_text(
        "# Model-based threshold classification and ABG referral\n\nPosterior medians and pointwise 95% credible intervals. A single TcCO₂ ≥45 predicts arterial PaCO₂ ≥45 mmHg. The two-stage strategy assigns arterial status using the reference ABG throughout the inclusive 40–50 mmHg reflex range. Remaining errors are expressed per 100 eligible patients.\n\n"
        + table(["Measure", *GROUPS.values()], rows)
    )


if __name__ == "__main__":
    render(*(Path(p) for p in sys.argv[1:]))
