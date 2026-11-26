"""Render checked, model-based classification probabilities at exact arterial values."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def render(x, curves, historical, output):
    groups = {
        "all": "All settings",
        "pft": "Ambulatory",
        "ed_inp": "ED / inpatient",
        "icu": "Critical Care Billing",
    }
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(7.0, 7.6))
    fig.subplots_adjust(left=0.105, right=0.97, top=0.84, bottom=0.11, hspace=0.53)
    fig.text(0.105, 0.963, "How often does TcCO₂ reach 45 mmHg?", size=16, weight="bold")
    fig.text(
        0.105,
        0.927,
        "At each arterial value, the line gives the modeled chance that TcCO₂ is ≥45 mmHg.",
        size=9.8,
    )
    fig.text(
        0.105,
        0.899,
        "The line and band show the posterior median and pointwise 95% credible interval.",
        size=9.8,
    )
    for ax, (key, label) in zip(axes, groups.items(), strict=True):
        low, median, high = curves[key]
        ax.set_facecolor("#f2f6f6")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, 1)
        ax.fill_between(x, low, high, color="#87a7b2", linewidth=0)
        ax.plot(x, median, color="#103f4e", lw=1.7)
        ax.axvline(45, color="#4d5051", ls=(0, (3, 2)), lw=0.9)
        ax.set_title(label, loc="left", size=11, weight="bold", pad=4)
        ax.set_yticks([0, 0.5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#899597")
        ax.tick_params(length=3, labelsize=8.5, color="#899597")
    axes[-1].set_xticks(range(30, 61, 5))
    axes[-1].set_xlabel("Arterial PaCO₂ (mmHg)", fontsize=11)
    fig.text(
        0.105,
        0.019,
        "The dashed line marks arterial hypercapnia (PaCO₂ ≥45 mmHg).",
        fontsize=8.8,
    )
    name = output / "forward-classification.png"
    fig.savefig(name, dpi=300, facecolor="white")
    fig.savefig(output / "forward-classification.pdf", metadata={"CreationDate": None})
    plt.close(fig)

    comparison = plt.figure(figsize=(14.4, 8.2), facecolor="white")
    comparison.text(0.025, 0.955, "December 2025  ·  Original figure", size=15, weight="bold")
    comparison.text(0.522, 0.955, "Current analysis  ·  Four settings", size=15, weight="bold")
    comparison.text(
        0.025,
        0.906,
        "Three settings. Bars group arterial readings rounded to whole mmHg.",
        size=10.2,
    )
    comparison.text(
        0.025,
        0.878,
        "The original 45 mmHg bar can include readings on either side of the cutoff.",
        size=10.2,
    )
    for source, bounds in [
        (historical, (0.012, 0.0, 0.479, 0.74)),
        (name, (0.50, 0.004, 0.494, 0.925)),
    ]:
        ax = comparison.add_axes(bounds)
        ax.imshow(plt.imread(source), interpolation="lanczos")
        ax.axis("off")
    comparison.savefig(output / "forward-classification-comparison.png", dpi=180)
    plt.close(comparison)
