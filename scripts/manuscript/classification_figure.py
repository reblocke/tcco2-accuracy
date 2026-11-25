"""Rebuild the four-setting classification exhibit and check both manuscript tables."""

import argparse
import os
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from private_inputs import private_destination, read_prior
from strategy import classification_draws

from tcco2_accuracy.core.future_setting import LEVELS, PredictiveKernel

SETTINGS = {
    "all": "All settings",
    "pft": "Ambulatory",
    "ed_inp": "ED / inpatient",
    "icu": "Critical Care Billing",
}
CELLS = {
    "TP": "True positives",
    "FP": "False positives",
    "FN": "False negatives",
    "TN": "True negatives",
}


def calculate(kernel, arterial, mass):
    """Return conditional probability bands and full-support joint counts per 100."""
    x = np.linspace(30, 60, 121)
    bands = np.empty((3, len(x)))
    joint = {name: np.zeros(kernel.theta.shape[:2]) for name in CELLS}
    if not (
        np.isfinite(arterial).all()
        and np.all(mass >= 0)
        and np.isclose(mass.sum(), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("Invalid arterial distribution")
    for value, weight in zip(arterial, mass, strict=True):
        p = kernel.draw_cdf(value - 45)
        positive, negative = ("TP", "FN") if value >= 45 else ("FP", "TN")
        joint[positive] += weight * p
        joint[negative] += weight * (1 - p)
    np.testing.assert_allclose(sum(joint.values()), 1, rtol=0, atol=1e-10)
    for i, value in enumerate(x):
        bands[:, i] = np.quantile(kernel.draw_cdf(value - 45), LEVELS)
    if not (
        np.isfinite(bands).all()
        and (np.diff(bands, axis=1) >= -1e-12).all()
        and np.all((-1e-14 <= bands) & (bands <= 1 + 1e-14))
    ):
        raise ValueError("Invalid classification probability curve")
    return x, np.clip(bands, 0, 1), {k: np.quantile(v * 100, LEVELS) for k, v in joint.items()}


def tables(wip):
    text = (wip / "classification.md").read_text()
    [section] = [
        block
        for block in re.findall(r"(?m)(?:^\|.*\n)+", text + "\n")
        if block.startswith("| Measure |") and "| True positives per 100 |" in block
    ]
    md = [line.strip("|").split("|") for line in section.splitlines()]
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(wip / "supplement.docx") as doc:
        root = ET.fromstring(doc.read("word/document.xml"))
    word = []
    for table in root.findall(".//w:tbl", ns):
        rows = [
            [
                "".join(t.text or "" for t in c.findall(".//w:t", ns))
                for c in row.findall("w:tc", ns)
            ]
            for row in table.findall("w:tr", ns)
        ]
        if (
            rows
            and rows[0][0] == "Measure"
            and any(row[0] == "True positives per 100" for row in rows)
        ):
            word.append(rows)
    [supplement] = word
    return [
        [
            dict(zip((v.strip() for v in rows[0]), (v.strip() for v in row), strict=True))
            for row in rows[1:]
        ]
        for rows in (md, supplement)
    ]


def check_cells(cells, setting, references):
    for code, label in CELLS.items():
        low, median, high = cells[code]
        expected = f"{median:.2f} ({low:.2f}–{high:.2f})"
        for rows in references:
            [actual] = [row[setting] for row in rows if row["Measure"] == label + " per 100"]
            if actual != expected:
                raise ValueError(f"{setting}, {label}: calculated {expected}, manuscript {actual}")


def run(private, original, manifest, wip, output):
    private_destination(output)
    references = tables(wip)
    curves = {}
    for group, theta in classification_draws(private, original, manifest):
        x, curves[group], cells = calculate(
            PredictiveKernel(theta, order=64), *read_prior(private, group)
        )
        check_cells(cells, SETTINGS[group], references)
        print(f"{SETTINGS[group]} matched both manuscript tables", flush=True)
    from classification_plot import render

    render(x, curves, wip / "december-2025/classification.png", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("private", "original", "manifest", "wip", "output"):
        parser.add_argument(name, type=Path)
    os.umask(0o077)
    run(**{name: path.resolve() for name, path in vars(parser.parse_args()).items()})
