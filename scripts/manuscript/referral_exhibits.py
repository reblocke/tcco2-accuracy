"""Build the private four-policy figures and the human-readable aggregate review tables."""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from private_inputs import private_destination, require_digest
from referral_report import CAPS, GROUPS

from tcco2_accuracy.core.referral_policy import display_policy

LABELS = {
    "cap 1%": "1% wrong-call cap",
    "cap 5%": "5% wrong-call cap",
    "cap 10%": "10% wrong-call cap",
    "fixed": "Fixed 40–50 mmHg",
}
OUTCOMES = [
    "Correct negative",
    "False negative",
    "Correct positive",
    "False positive",
    "ABG referral",
]
COLORS = ["#437778", "#cf782e", "#34465e", "#ac4135", "#d2d3d0"]
HATCHES = ["", "////", "", "xxxx", ""]


def band(cuts):
    a, b = cuts
    return "none" if b == a + 1 else f"{a + 1}–{b - 1}"


def fmt(x, digits=2):
    if x is None:
        return "Undefined"
    return f"{x[1]:.{digits}f} ({x[0]:.{digits}f}–{x[2]:.{digits}f})"


def save(fig, dest, stem):
    fig.savefig(dest / (stem + ".png"), dpi=300, facecolor="white")
    fig.savefig(dest / (stem + ".pdf"), metadata={"CreationDate": None})
    plt.close(fig)


def panels(data, dest):
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 7.35), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.976, top=0.858, bottom=0.115, hspace=0.92)
    fig.text(
        0.085,
        0.968,
        "When would a TcCO₂ reading lead to an ABG?",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.085,
        0.932,
        GROUPS[data["setting"]] + "  ·  Posterior mean per 100 eligible testing occasions",
        fontsize=10.5,
    )
    xs = np.r_[13, np.arange(20, 101), 107]
    for ax, (key, title) in zip(axes, LABELS.items(), strict=True):
        row = data["policies"][key]
        counts = np.array(row["figure_counts"])
        lo, hi = row["cuts"]
        ax.set_title(
            title + "   ·   Refer displays " + band((lo, hi)) + " mmHg",
            loc="left",
            fontsize=10.4,
            weight="bold",
            pad=16,
        )
        if (
            counts.shape != (5, len(xs))
            or not np.all(np.isfinite(counts))
            or np.any(counts < -1e-10)
        ):
            raise ValueError("Expected five disposition counts for each of the 83 display bins")
        np.testing.assert_allclose(counts.sum(), 100, rtol=0, atol=1e-8)
        eff = counts[:4].sum()
        ax.text(
            0,
            1.035,
            f"Classified {eff:.1f} (including {counts[1].sum():.2f} false negative, {counts[3].sum():.2f} false positive)   ·   Referred {counts[4].sum():.1f}",
            transform=ax.transAxes,
            fontsize=8.0,
        )
        bottom = np.zeros(83)
        for heights, color, pattern in zip(counts, COLORS, HATCHES, strict=True):
            ax.bar(
                xs,
                heights,
                bottom=bottom,
                width=0.97,
                color=color,
                edgecolor="#465057",
                linewidth=0.14,
                hatch=pattern,
            )
            bottom += heights
        if hi - lo > 1:
            ax.axvspan(lo + 0.5, hi - 0.5, color="#f0f0ee", zorder=-2)
        for cut in (lo + 0.5, hi - 0.5):
            if 19.5 <= cut <= 100.5:
                ax.axvline(cut, ls=(0, (2, 3)), lw=0.65, color="#666", zorder=-1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(10, 110)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8)
    axes[-1].set_xticks(
        [13, 20, 30, 40, 50, 60, 70, 80, 90, 100, 107],
        ["<20", "20", "30", "40", "50", "60", "70", "80", "90", "100", ">100"],
    )
    axes[-1].set_xlabel("Displayed transcutaneous CO₂ (mmHg)", fontsize=10)
    legend = [
        Patch(facecolor=c, edgecolor="#485158", linewidth=0.4, hatch=h, label=label)
        for c, h, label in zip(COLORS, HATCHES, OUTCOMES, strict=True)
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.023),
        ncols=5,
        fontsize=8,
        frameon=False,
        columnspacing=1.05,
        handlelength=1.4,
    )
    save(fig, dest, "referral-" + data["setting"])


def curve(groups, dest):
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.835, bottom=0.15, hspace=0.36, wspace=0.15)
    fig.text(
        0.08,
        0.948,
        "How does the tolerated error change referral burden?",
        weight="bold",
        fontsize=14,
    )
    fig.text(
        0.08,
        0.897,
        "Testing occasions classified without ABG (%)  ·  Median and pointwise 95% credible interval",
        fontsize=9.2,
    )
    for ax, (g, d) in zip(axes.flat, groups.items(), strict=True):
        x, low, median, high = np.array(d["efficiency_curve"]).T
        ax.fill_between(x, low, high, step="post", color="#cadadd")
        ax.step(x, median, where="post", color="#294c5a", lw=1.8)
        ax.scatter(
            [1, 5, 10],
            [d["policies"][f"cap {int(c * 100)}%"]["median_cri"]["efficiency"][1] for c in CAPS],
            color="#294c5a",
            s=18,
            zorder=3,
        )
        ax.set_title(GROUPS[g], loc="left", fontsize=10, weight="bold")
        ax.set(
            xlim=(0.85, 10.15), ylim=(-2, 102), yticks=[0, 25, 50, 75, 100], xticks=[1, 3, 5, 7, 10]
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", lw=0.4, color="#ddd")
    fig.supxlabel("Maximum modeled wrong-call probability at a display (%)", y=0.035, fontsize=10)
    save(fig, dest, "referral-efficiency")


def table(head, rows):
    return [
        "| " + " | ".join(head) + " |",
        "| " + " | ".join("---" for _ in head) + " |",
        *["| " + " | ".join(map(str, r)) + " |" for r in rows],
        "",
    ]


def run(work, dest):
    private_destination(dest)
    data = {g: json.loads((work / (g + "-policy.json")).read_bytes()) for g in GROUPS}
    lines = [
        "# TcCO₂ referral policies in an ABG-tested population",
        "",
        "Each eligible patient has total weight one, split over arterial result records on the first eligible testing day. Each policy assumes a usable, contemporaneous TcCO₂ and a timely, definitive ABG after referral. Other indications for arterial sampling include pH and PaO₂. Their frequency, sensor failure frequency, and actual punctures avoided are unknown.",
        "",
        "The primary policies limit the modeled probability of a wrong call at each displayed integer. The class-denominator alternatives limit FNR among all reference-positive and FPR among all reference-negative occasions. All arterial thresholds are inclusive at 45 mmHg. Parentheses give pointwise 95% credible intervals for fixed policies. Operating characteristics are posterior medians. Figure bars and totals are posterior means.",
        "",
    ]
    for g, d in data.items():
        panels(d, dest)
        policies = {k: d["policies"][k] for k in LABELS}
        policies.update((k, v) for k, v in d["policies"].items() if k not in policies)
        lines += ["## " + GROUPS[g], "", "### Cutoffs and efficiency", ""]
        lines += table(
            [
                "Policy",
                "Last negative",
                "ABG displays",
                "First positive",
                "Classified without ABG (%)",
                "Referrals /100",
                "Residual FN /100",
                "Residual FP /100",
            ],
            [
                (
                    k,
                    r["cuts"][0] if r["cuts"][0] >= 20 else "None",
                    band(r["cuts"]),
                    r["cuts"][1] if r["cuts"][1] <= 100 else "None",
                    *[fmt(r["median_cri"][n]) for n in ("efficiency", "referral", "FN", "FP")],
                )
                for k, r in policies.items()
            ],
        )
        lines += ["### Class and call denominators", ""]
        lines += table(
            [
                "Policy",
                "Negative /100",
                "Positive /100",
                "Off-grid referrals /100",
                "Wrong among negative (%)",
                "Wrong among positive (%)",
                "Whole-strategy FNR (%)",
                "Whole-strategy FPR (%)",
            ],
            [
                (
                    k,
                    *[
                        fmt(r["median_cri"][n])
                        for n in (
                            "negative",
                            "positive",
                            "off_grid",
                            "negative_wrong",
                            "positive_wrong",
                            "FNR",
                            "FPR",
                        )
                    ],
                )
                for k, r in policies.items()
            ],
        )
        lines += [
            "### Reading-specific design risks",
            "",
            "Values are posterior-integrated percentages. The numerical upper bounds incorporate chain-batch Monte Carlo and quadrature error. Off-grid displays always go to ABG.",
            "",
        ]
        rows = []
        for k, r in policies.items():
            for side, entry in r["design_risk_percent"].items():
                rows.append(
                    [
                        k,
                        side,
                        *(
                            ["Undefined"] * 3
                            if entry is None
                            else [
                                f"{entry[n]:.4f}"
                                for n in (
                                    "boundary_risk",
                                    "worst_risk",
                                    "worst_upper_numerical_bound",
                                )
                            ]
                        ),
                    ]
                )
        lines += table(
            [
                "Policy",
                "Definitive call",
                "Boundary risk (%)",
                "Worst risk (%)",
                "Numerical upper bound (%)",
            ],
            rows,
        )
        lines += ["### Paired changes per 100 eligible occasions", ""]
        lines += table(
            [
                "Policy contrast",
                "Referrals",
                "Classified without ABG",
                "False negatives",
                "False positives",
            ],
            [
                [k, *[fmt(v[i]) for i in ("referral", "efficiency", "FN", "FP")]]
                for k, v in d["contrasts"].items()
            ],
        )
        lines += [
            "Display-risk holds: "
            + "; ".join(k + " " + str(v) for k, v in d["numerical_holds"].items() if v)
            if any(d["numerical_holds"].values())
            else "No unresolved named display boundary.",
            "The two reference-class constraints gave a unique numerical optimum at each cap."
            if all(x["numerically_unique"] for x in d["class_optimum"].values())
            else "Some conservative reference-class optima remain numerically unresolved. See the calculation receipt.",
            "",
        ]
    pages = Path(__file__).resolve().parents[2] / "models/pages"
    lookup = pages / "conditional-lookup.json"
    receipt = json.loads((pages / "conditional-release.json").read_bytes())
    require_digest(lookup, receipt["lookup_sha256"])
    reference = json.loads(lookup.read_bytes())["groups"]
    bridge = []
    for group, info in data.items():
        rows = reference[group]["rows"]
        points = np.array([r["tails"][0] for r in rows])
        lo, hi = np.array([r["report"]["probabilities"][0] for r in rows]).T
        for cap in CAPS:
            k = f"cap {100 * cap:g}%"
            bridge.append(
                [
                    GROUPS[group],
                    k,
                    band(display_policy(points, points, cap)),
                    band(display_policy(lo, hi, cap)),
                    band(info["policies"][k]["cuts"]),
                ]
            )
    lines += [
        "## Does rounding the display change the ABG band?",
        "",
        "The first column of bands uses exact-reading point estimates at integer values. The second applies their numerical bounds. The final column integrates the entire rounded display bin and applies its bounds. Every band is inclusive and confined to the 20–100 grid.",
        "",
        *table(
            [
                "Setting",
                "Cap",
                "Exact reading, point estimate",
                "Exact reading, qualified",
                "Rounded display, qualified",
            ],
            bridge,
        ),
    ]
    for group, info in data.items():
        reversals = ", ".join(map(str, info["reversals"])) or "none"
        lines.append(
            f"{GROUPS[group]} had a maximum numerical radius of {info['max_numerical_radius_pp']:.4f} percentage points and a maximum quadrature change of {info['max_quadrature_change_pp']:.4f} percentage points. Downward risk steps occurred at the following displays in mmHg: {reversals}."
        )
    curve(data, dest)
    (dest / "referral-tables.md").write_text("\n".join(lines).strip() + "\n")
    print("Review tables and five figures rendered", flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("work", "dest"):
        p.add_argument(n, type=lambda s: Path(s).resolve())
    run(**vars(p.parse_args()))
