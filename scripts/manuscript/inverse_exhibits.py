"""Render exact-reading hypercapnia probabilities and conditional classification percentages."""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from inverse_figure import CUTS, GRID, classification_proportions
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from private_inputs import digest, private_destination, require_digest
from referral_exhibits import COLORS, HATCHES, save

SETTINGS = {
    "all": "All settings",
    "icu": "Critical Care Billing",
    "ed_inp": "ED / inpatient",
    "pft": "Ambulatory",
}


def read_checked(pages, checks, draws):
    receipt = json.loads((pages / "conditional-release.json").read_bytes())
    lookup = pages / "conditional-lookup.json"
    require_digest(lookup, receipt["lookup_sha256"])
    groups = json.loads(lookup.read_bytes())["groups"]
    data = {}
    for group, n in zip(SETTINGS, draws, strict=True):
        record = json.loads((checks / f"{group}-{n}.json").read_bytes())
        if (
            record["group"],
            record["posterior"],
            record["lookup"],
            record["draws_per_chain"],
            record["code"],
            record["selection_code"],
        ) != (
            group,
            receipt["fits"][group]["posterior"],
            receipt["lookup_sha256"],
            n,
            digest(Path(__file__).with_name("inverse_figure.py")),
            digest(Path(__file__).with_name("inverse_draws.py")),
        ):
            raise ValueError(f"Band receipt differs from the requested inputs for {group}")
        np.testing.assert_array_equal(record["readings"], GRID)
        p = np.asarray([row["tails"] for row in groups[group]["rows"]])
        edges = np.asarray(record["edges"])
        mask = np.asarray(record["qualified"], dtype=bool)
        mc = np.asarray(record["mc_radius"])
        quad = np.asarray(record["quadrature"])
        if (
            mc.shape != edges.shape
            or quad.shape != edges.shape
            or not (
                np.isfinite(mc).all()
                and np.isfinite(quad).all()
                and np.all(mc >= 0)
                and np.all(quad >= 0)
            )
        ):
            raise ValueError("Invalid endpoint precision evidence")
        np.testing.assert_array_equal(mask, (mc + quad <= 0.001) & (quad <= 0.0001))
        if (
            p.shape != (81, 2)
            or edges.shape != (81, 2, 2)
            or edges.shape != mask.shape
            or not np.isfinite(edges).all()
            or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))
            or np.any(p[:, 1] > p[:, 0])
            or np.any((edges < 0) | (edges > 1))
            or np.any(edges[:, :, 0] > edges[:, :, 1])
            or np.any(edges[:, 1] > edges[:, 0])
        ):
            raise ValueError("Invalid threshold probabilities or pointwise bands")
        data[group] = p, edges, mask
    return data


def style(ax):
    ax.set(xlim=(19.5, 100.5), ylim=(0, 1), yticks=[0, 0.5, 1], xticks=[20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#819093")
    ax.tick_params(labelsize=8.5, length=3, color="#819093")


def risks(data, dest):
    fig, axes = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(7.1, 6.8))
    fig.subplots_adjust(left=0.102, right=0.977, top=0.787, bottom=0.126, hspace=0.67, wspace=0.13)
    fig.text(
        0.102,
        0.973,
        "Given this TcCO₂ reading, how likely is PaCO₂",
        weight="bold",
        fontsize=15,
        va="top",
    )
    fig.text(0.102, 0.934, "to be ≥45 or ≥52 mmHg?", weight="bold", fontsize=15, va="top")
    fig.text(
        0.102,
        0.865,
        "Probability of simultaneous arterial hypercapnia in the ABG-tested population",
        fontsize=9.3,
    )
    colors = ["#165e59", "#824421"]
    for j, threshold in enumerate(CUTS):
        fig.text(
            axes[0, j].get_position().x1,
            0.815,
            f"PaCO₂ ≥{threshold} mmHg",
            color=colors[j],
            weight="bold",
            ha="right",
            fontsize=10,
        )
    for row, (group, label) in enumerate(SETTINGS.items()):
        p, edges, qualified = data[group]
        for h, ax in enumerate(axes[row]):
            style(ax)
            valid = qualified[:, h].all(axis=-1)
            lower, upper = edges[:, h].T
            ax.fill_between(
                GRID, lower, upper, where=valid, color=colors[h], alpha=0.22, linewidth=0
            )
            # An isolated qualified point also has an interval, even between two gaps.
            isolated = valid & ~np.r_[False, valid[:-1]] & ~np.r_[valid[1:], False]
            ax.vlines(
                GRID[isolated],
                lower[isolated],
                upper[isolated],
                color=colors[h],
                alpha=0.45,
                linewidth=0.7,
            )
            ax.plot(GRID, p[:, h], color=colors[h], linestyle=["-", "--"][h], lw=1.6)
            if h == 0:
                ax.set_title(label, loc="left", pad=2.5, fontsize=9.5, weight="bold")
    fig.supxlabel("Observed TcCO₂ (mmHg)", y=0.065, fontsize=10.5)
    fig.text(
        0.102,
        0.033,
        "Lines give model-averaged probabilities. Shading gives pointwise 95% credible intervals.",
        fontsize=8,
    )
    if any(not mask.all() for _, _, mask in data.values()):
        fig.text(
            0.102,
            0.011,
            "A gap in shading marks insufficient numerical precision for a band endpoint.",
            fontsize=8,
        )
    save(fig, dest, "classification-figure")


def bars(data, dest):
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(7.1, 6.5))
    fig.subplots_adjust(left=0.102, right=0.978, top=0.819, bottom=0.194, hspace=0.59)
    fig.text(
        0.102, 0.975, "If TcCO₂ ≥45 mmHg is called positive,", weight="bold", fontsize=15, va="top"
    )
    fig.text(0.102, 0.935, "what would a paired ABG show?", weight="bold", fontsize=15, va="top")
    fig.text(
        0.102,
        0.860,
        "Each column represents 100 modeled testing occasions with that exact TcCO₂ reading.",
        fontsize=9.2,
    )
    labels = ["True negative", "False negative", "True positive", "False positive"]
    for ax, (group, label) in zip(axes, SETTINGS.items(), strict=True):
        style(ax)
        ax.set_xticks([20, 30, 40, 45, 50, 60, 70, 80, 90, 100])
        p = data[group][0][:, 0]
        shares = classification_proportions(GRID, p)
        np.testing.assert_allclose(shares.sum(axis=1), 1, rtol=0, atol=2e-15)
        base = np.zeros(len(GRID))
        for i, part in enumerate(shares.T):
            use = part > 0
            ax.bar(
                GRID[use],
                part[use],
                bottom=base[use],
                width=0.96,
                color=COLORS[i],
                edgecolor="#30444a",
                linewidth=0.15,
                hatch=HATCHES[i],
            )
            base += part
        ax.axvline(44.5, color="#121e22", lw=0.8, ls=(0, (3, 2)))
        ax.set_title(label, loc="left", fontsize=9.5, weight="bold", pad=3)
    fig.supxlabel("Observed TcCO₂ (mmHg)", y=0.124, fontsize=10.5)
    handles = [
        Patch(facecolor=COLORS[i], edgecolor="#30444a", hatch=HATCHES[i], label=v)
        for i, v in enumerate(labels)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.065),
        ncols=4,
        frameon=False,
        fontsize=8.4,
    )
    fig.text(
        0.102,
        0.041,
        "The dashed divider separates negative and positive TcCO₂ calls. PaCO₂ ≥45 mmHg defines hypercapnia.",
        fontsize=7.9,
    )
    fig.text(
        0.102,
        0.016,
        "At TcCO₂ 45 in All settings, about 32 of 100 calls are true positives and 68 are false positives.",
        fontsize=7.9,
    )
    save(fig, dest, "classification-by-reading")


def comparison(historical, dest):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.9))
    fig.subplots_adjust(left=0.015, right=0.99, top=0.87, bottom=0.015, wspace=0.055)
    for ax, source, title, subtitle in zip(
        axes,
        [historical, dest / "classification-by-reading.png"],
        ["December 2025  ·  Given arterial PaCO₂", "Current model  ·  Given observed TcCO₂"],
        [
            "Each bar groups rounded arterial values in the earlier analysis.",
            "Each bar groups testing occasions at one exact transcutaneous reading.",
        ],
        strict=True,
    ):
        ax.imshow(plt.imread(source), interpolation="lanczos")
        ax.axis("off")
        pos = ax.get_subplotspec().get_position(fig)
        fig.text(pos.x0, 0.956, title, fontsize=13, weight="bold")
        fig.text(pos.x0, 0.912, subtitle, fontsize=9.6)
    fig.savefig(dest / "classification-comparison.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("pages", "checks", "historical", "dest"):
        parser.add_argument(arg, type=lambda s: Path(s).resolve())
    parser.add_argument(
        "--draws",
        type=int,
        nargs=4,
        default=[32768, 131072, 262144, 262144],
        metavar=("ALL", "ICU", "ED", "PFT"),
    )
    args = parser.parse_args()
    private_destination(args.dest)
    data = read_checked(args.pages, args.checks, args.draws)
    risks(data, args.dest)
    bars(data, args.dest)
    comparison(args.historical, args.dest)
