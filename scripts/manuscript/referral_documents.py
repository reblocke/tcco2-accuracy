"""Revise the November manuscript reading copies in a private review directory."""

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt

NAMES = {
    "all": "All settings",
    "icu": "Critical Care Billing",
    "ed_inp": "ED/noncritical inpatient",
    "pft": "Ambulatory",
}
DESIGN = (
    "In a post hoc analysis we modeled a device that displays TcCO₂ to the nearest whole mmHg. "
    "For each display from 20 to 100, we integrated the signed disagreement likelihood over its 1-mmHg bin. "
    "At a cap of 1%, 5% or 10%, a low or high display could classify PaCO₂ relative to 45 mmHg if its modeled probability of a wrong call met the cap. "
    "We took consecutive qualifying displays from each end. Patients with other displays received an ABG, including those outside the supported range or at a numerically unresolved boundary. "
    "We evaluated these fixed rules and a display-matched 40–50 comparator on the same posterior draws."
)
CORRECTION = (
    "On the same 75 archived rows, using natural logarithms consistently widened the overall reference interval from 14.0 mmHg (−7.1 to 6.9) to 20.0 mmHg (−10.1 to 9.9). "
    "Using the variance of the estimated between-study variance on its proper scale narrowed the endpoint confidence envelope. "
    "Together the repairs gave an outer envelope of −11.2 to 11.1 mmHg. "
    "Our model uses the adjudicated study summaries to predict one paired difference in a comparable new setting (overall 95% prediction interval −11.5 to 11.2 mmHg)."
)
COHORT = (
    "The 2023 request specified a first qualifying encounter for each indication within each setting. "
    "The historical notebooks selected each patient’s first encounter for an indication before selecting the setting. "
    "The maintained code reverses that order and changes which gas results qualify. We retained archived encounter membership for the primary analysis. "
    "To isolate the effect of order, we replayed both rules on the same recovered candidates with identical historical gas criteria. "
    "The reason-first replay reproduced archived membership across all 18 indication-by-setting groups and the four arterial distributions. Changing the order changed encounter membership. The surviving records do not establish which execution produced the May 2025 archive."
)
CAPTION = (
    "Who receives an arterial blood gas (ABG) per 100 eligible testing occasions? "
    "At each whole-number display classified without an ABG, the modeled chance of a wrong call is at most the stated 1%, 5% or 10%. "
    "The fixed policy uses 40–50 mmHg inclusive. Under every policy, patients with displays outside 20–100 also receive an ABG. "
    "Hatching shows errors among those classified without an ABG. Bars and totals are posterior means assuming a usable reading and a timely ABG."
)
HEADERS = [
    "Policy",
    "ABG band (mmHg)",
    "Classified without ABG (%)",
    "Receives an ABG /100",
    "False negatives /100",
    "False positives /100",
]


def append(p, s):
    p.add_run(" " + s)


def replace(p, s):
    # Retain paragraph layout, comment boundaries and other XML around edited text.
    if p._p.xpath(".//w:instrText | .//w:fldChar"):
        raise ValueError("A field-bearing paragraph requires an anchored edit")
    nodes = p._p.xpath(".//w:t")
    lengths = [len(n.text or "") for n in nodes]
    if not sum(lengths):
        p.add_run(s)
        return
    pos = 0
    for i, node in enumerate(nodes):
        end = round(len(s) * sum(lengths[: i + 1]) / sum(lengths))
        node.text = s[pos:end]
        node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        pos = end


def insert_figure(doc, figure, caption, before=None):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.page_break_before = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(figure), width=Inches(6.0))
    caption_style = "Caption" if "Caption" in doc.styles else "Normal"
    c = doc.add_paragraph(caption, style=caption_style)
    if caption_style == "Normal":
        c.runs[0].font.size = Pt(10)
    c.paragraph_format.line_spacing = 1.1
    c.paragraph_format.keep_with_next = False
    if before is not None:
        before._p.addprevious(p._p)
        before._p.addprevious(c._p)


def tab(doc, title, rows, widths):
    h = doc.add_paragraph(title, style="Heading 2")
    h.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=0, cols=len(widths))
    t.style = "Table Grid"
    t.autofit = False
    for col, w in zip(t.columns, widths, strict=True):
        col.width = Inches(w)
    for index, values in enumerate(rows):
        row = t.add_row()
        props = row._tr.get_or_add_trPr()
        props.append(OxmlElement("w:cantSplit"))
        if not index:
            props.append(OxmlElement("w:tblHeader"))
        for cell, w, value in zip(row.cells, widths, values, strict=True):
            cell.width = Inches(w)
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.1
            p.paragraph_format.space_after = Pt(3)
            run = p.add_run(str(value))
            run.bold = index == 0
            run.font.name = "Times New Roman"
            run.font.size = Pt(10)
    return t


def tables(text):
    for block in re.findall(r"(?m)(?:^\|[^\n]*\n)+", text + "\n"):
        row = [list(map(str.strip, x.strip("|").split("|"))) for x in block.splitlines()]
        yield [row[0], *row[2:]]


def band(r):
    lo, hi = r["cuts"]
    return f"{lo + 1}–{hi - 1}" if hi > lo + 1 else "None"


def val(r, k):
    return r["median_cri"][k][1]


def ci(r, k):
    low, median, high = r["median_cri"][k]
    return f"{median:.2f} ({low:.2f}–{high:.2f})"


def short_result(pol):
    a, b = pol["cap 5%"], pol["fixed"]
    return (
        f"For displayed TcCO₂, the 5% rule referred {band(a)} mmHg and the fixed rule referred {band(b)} mmHg. "
        f"Per 100 eligible testing occasions, the two rules referred {val(a, 'referral'):.1f} and {val(b, 'referral'):.1f}, "
        f"leaving {val(a, 'FN'):.2f} and {val(b, 'FN'):.2f} false negatives and {val(a, 'FP'):.2f} and {val(b, 'FP'):.2f} false positives, respectively (Figure 2)."
    )


def short(parent, dest, pol, figure):
    d = Document(parent)
    p = d.paragraphs
    if not p[20].text.startswith("We also modeled") or p[36].text != "References":
        raise ValueError("Short Report parent has changed")
    append(
        p[16],
        "TriNetX contributed source-recorded ordinary or temperature-adjusted arterial results with convertible units and values strictly between 2 and 250 mmHg. Testing could occur after admission or treatment.",
    )
    append(
        p[17],
        "In Conway’s archived code we corrected the logarithm used for within-study variance and the variance scale used for endpoint uncertainty. On the same 75 rows the overall reference interval widened from 14.0 mmHg (−7.1 to 6.9) to 20.0 mmHg (−10.1 to 9.9). The corrected endpoint envelope was −11.2 to 11.1 mmHg. Our 95% prediction interval for one new paired difference was −11.5 to 11.2 mmHg.",
    )
    replace(
        p[20],
        "For a secondary analysis, we assumed a usable TcCO₂ displayed to the nearest whole mmHg. We compared referral for displays of 40–50 mmHg with three post hoc rules. Each limited the probability of misclassifying simultaneous PaCO₂ ≥45 at a definitive display to 1%, 5% or 10%. We integrated over each display bin, referred readings outside 20–100 and assumed a timely ABG resolved each referral.",
    )
    replace(p[25], short_result(pol))
    append(
        p[28],
        "The proportion classified without ABG estimates potential efficiency for CO₂ assessment. The indication for sampling and the frequency of unusable readings were unknown, so actual punctures avoided could differ.",
    )
    insert_figure(d, figure, "Figure 2. " + CAPTION, before=p[36])
    d.save(dest)

    # Word reserializes comments even when no annotation was edited. Keep the
    # original part verbatim after verifying the complete XML trees are equivalent.
    with ZipFile(parent) as package:
        has_comments = "word/comments.xml" in package.namelist()
    if has_comments:
        from lxml import etree

        with ZipFile(parent) as old, ZipFile(dest) as current:
            original = old.read("word/comments.xml")

            def c14n(data):
                return etree.tostring(etree.fromstring(data), method="c14n")

            if c14n(original) != c14n(current.read("word/comments.xml")):
                raise ValueError("Editable comments changed during the revision")
            out = io.BytesIO()
            with ZipFile(out, "w") as new:
                for item in current.infolist():
                    data = original if item.filename == "word/comments.xml" else current.read(item)
                    new.writestr(item, data)
        dest.write_bytes(out.getvalue())


def main_version(parent, dest, pol, figure):
    d = Document(parent)
    p = d.paragraphs
    if "95% of cases" not in p[16].text or p[60].text != "References":
        raise ValueError("Original Research parent has changed")
    replace(
        p[16],
        p[16].text.replace(
            "the model predicted a simultaneous PaCO₂ of 35–53 mmHg in 95% of cases under its assumptions",
            "the central 95% model prediction interval for simultaneous PaCO₂ was 35–53 mmHg",
        ),
    )
    append(p[23], CORRECTION)
    append(p[26], COHORT)
    append(
        p[27],
        "The first eligible testing day may follow treatment. Multiple results can come from one specimen.",
    )
    append(
        p[31],
        "We chose 45 mmHg to describe arterial elevation and 52 mmHg as a higher threshold used in COPD ventilatory-support research. The ATS definition of stable hypercapnic COPD uses resting PaCO₂ >45, and the ONMAP expert panel discussed ≥52 in COPD.",
    )
    p[31].add_run("6,7").font.superscript = True
    append(
        p[31], "We applied both inclusive study endpoints to every selected ABG-tested population."
    )
    append(p[33], DESIGN)
    append(
        p[41],
        "Interpretation at these high readings assumes that the disagreement seen in the paired studies applies across arterial CO₂ levels.",
    )
    replace(
        p[43],
        p[43].text.replace(
            "The two-stage strategy referred",
            "The historical continuous 40–50 mmHg strategy referred",
            1,
        ),
    )
    p[44].insert_paragraph_before(short_result(pol))
    at51 = pol["fixed"]["design_risk_percent"]["positive"]["boundary_risk"]
    p[44].insert_paragraph_before(
        f"At a displayed 51 mmHg, the fixed rule called PaCO₂ positive with a {at51:.1f}% modeled probability that PaCO₂ was below 45. Table 2 gives the referral burden and remaining errors at each cap."
    )
    append(
        p[52],
        "The subgroup tested specifically to assess CO₂ could have a different arterial distribution, in either direction. A failed or unavailable sensor reading would still require evaluation. Failure frequency and actual arterial sampling avoided could not be estimated.",
    )
    append(
        p[53],
        "Validation requires contemporaneous paired measurements below, inside and above the referral band, including attempted readings that fail.",
    )
    p[66].insert_paragraph_before(
        "6. Macrea M, Oczkowski S, Rochwerg B, et al. Long-term noninvasive ventilation in chronic stable hypercapnic chronic obstructive pulmonary disease. Am J Respir Crit Care Med. 2020;202:e74–e87. doi:10.1164/rccm.202006-2382ST."
    )
    p[66].insert_paragraph_before(
        "7. Hill NS, Criner GJ, Branson RD, et al. Optimal NIV Medicare Access Promotion: Patients With COPD. Chest. 2021;160:e389–e397. doi:10.1016/j.chest.2021.06.082."
    )
    d.add_page_break()
    rows = [HEADERS]
    for key in ("cap 1%", "cap 5%", "cap 10%", "fixed"):
        r = pol[key]
        rows.append([key, band(r), *[ci(r, n) for n in ("efficiency", "referral", "FN", "FP")]])
    tab(
        d,
        "Table 2. Referral and residual classification errors in All settings",
        rows,
        [0.74, 0.85, 1.32, 1.15, 1.15, 1.15],
    )
    d.add_paragraph(
        "Estimates are posterior medians (pointwise 95% credible intervals) per 100 patient-weighted testing occasions with usable readings. All usable displays outside 20–100 also receive ABG. The percentage classified without ABG quantifies potential efficiency when arterial CO₂ classification is the sole purpose of testing."
    )
    insert_figure(d, figure, "Figure 2. " + CAPTION)
    d.save(dest)


def supplement(parent, dest, review, work):
    d = Document(parent)
    d.add_page_break()
    d.add_heading("Encounter selection and arterial results", 1)
    encounter = (review / "encounter-selection.md").read_text()
    for para in encounter.split("\n"):
        if para.startswith(
            (
                "The November 2023",
                "The historical arterial",
                "The present arterial",
                "The reason-first reconstruction",
            )
        ):
            d.add_paragraph(para)
    es = list(tables(encounter))
    tab(
        d,
        "Supplementary Table 20. Ordered flow in the archived population",
        es[0],
        [2.45, 1.2, 1.2, 1.45],
    )
    d.add_paragraph(
        "Each successive row retains the preceding restrictions. Archived membership consists of encounter keys. The first-day row counts distinct patients, encounters and source result records."
    )
    for group, name in NAMES.items():
        rows = [es[1][0][1:], *[r[1:] for r in es[1][1:] if r[0] == name]]
        tab(
            d,
            "Supplementary Table 21. Selection order at TcCO₂ 47 mmHg, " + name,
            rows,
            [1.0, 0.83, 0.83, 0.63, 1.24, 0.82, 0.82],
        )
    d.add_paragraph(
        "Pressures are in mmHg. Baseline probabilities use fixed patient weights. For each setting we applied the same saved agreement fit to all three populations. Held denotes a numerical reporting hold. Whole-number display ranges follow the original numerical reporting rule."
    )
    t = es[-1]
    tab(
        d,
        "Supplementary Table 21B. Change with setting-first selection at TcCO₂ 47",
        [
            [
                "Setting",
                "Baseline ≥45 (pp)",
                "Baseline ≥52 (pp)",
                "Lower / median / upper (mmHg)",
                "After ≥45 (pp)",
                "After ≥52 (pp)",
            ]
        ]
        + [[r[0], r[1], r[2], " / ".join(r[3:6]), r[6], r[7]] for r in t[1:]],
        [1.1, 0.95, 0.95, 1.45, 0.95, 0.95],
    )
    d.add_paragraph(
        "Changes are setting-first minus archived. Baseline differences use the full probabilities. Prediction differences compare the certified reported whole-unit values. pp denotes percentage points."
    )
    d.add_heading("Testing occasions classified without an arterial blood gas", 1)
    d.add_paragraph(
        DESIGN
        + " We chose the risk caps after examining the exact-reading lookup. The caps represent hypothetical preferences. The relative clinical harms of a missed arterial elevation, a false positive and arterial sampling were unavailable."
    )
    d.add_paragraph(
        "We first averaged the positive and total joint display masses over posterior draws and then took their ratio. To improve numerical precision, we used the qualified exact-reading risk as a control for the bin-integrated risk and estimated the small rounding correction on common draws. We allowed two within-chain batch Monte Carlo standard errors and the change from 32 to 64 quadrature nodes. A bound crossing a cap led to referral. For the reference-class alternatives we maximized classification without ABG subject to both whole-strategy FNR and FPR caps. Boundary risks, uncertainty in policy performance and numerical uncertainty have different interpretations."
    )
    d.add_paragraph(
        "The original continuous comparator uses 40 through 50 inclusive throughout the real line. For the display comparator, the same labels span underlying values from 39.5 up to 50.5 mmHg. Under every policy, a patient with an off-grid display receives an ABG. Empty call regions have undefined conditional error rates."
    )
    all_tables = list(tables((review / "referral-tables.md").read_text()))
    for j, (group, name) in enumerate(NAMES.items()):
        cuts, den, risk, contrast = all_tables[4 * j : 4 * j + 4]
        d.add_page_break()
        d.add_heading(name, 1)

        def cols(table, indexes):
            return [[r[i] for i in indexes] for r in table]

        n = 22 + j
        tab(
            d,
            f"Supplementary Table {n}A. Displays and modeled outcomes per 100",
            cols(cuts, [0, 2, 4, 5, 6, 7]),
            [0.86, 0.73, 1.28, 1.17, 1.14, 1.18],
        )
        tab(
            d,
            f"Supplementary Table {n}B. Definitive calls and ABGs outside the grid",
            cols(den, [0, 1, 2, 3]),
            [1.25, 1.55, 1.55, 1.9],
        )
        tab(
            d,
            f"Supplementary Table {n}C. Conditional errors by call and by reference class (%)",
            cols(den, [0, 4, 5, 6, 7]),
            [1.0, 1.35, 1.35, 1.28, 1.28],
        )
        tab(
            d,
            f"Supplementary Table {n}D. Modeled risks at definitive displays (%)",
            risk,
            [1.07, 1.12, 1.25, 1.2, 1.62],
        )
        tab(
            d,
            f"Supplementary Table {n}E. Within-draw changes per 100",
            contrast,
            [1.36, 1.22, 1.22, 1.22, 1.22],
        )
        meta = json.loads((work / (group + "-policy.json")).read_bytes())
        holds = [
            f"{key} at " + ", ".join(map(str, ds)) + " mmHg"
            for key, ds in meta["numerical_holds"].items()
            if ds
        ]
        d.add_paragraph(
            "Numerically unresolved displays referred to ABG: " + ", ".join(holds) + "."
            if holds
            else "Every named policy boundary resolved numerically."
        )
        d.add_paragraph(
            "The conservative and optimistic bounds selected the same least-referring rule at every reference-class cap."
            if all(v["numerically_unique"] for v in meta["class_optimum"].values())
            else "The conservative rule is shown where the numerical bounds admit competing reference-class optima."
        )
    for num, (group, name) in enumerate(list(NAMES.items())[1:], 3):
        insert_figure(
            d, review / f"referral-{group}.png", f"Supplementary Figure {num}. {name}. " + CAPTION
        )
    insert_figure(
        d,
        review / "referral-efficiency.png",
        "Supplementary Figure 6. Percentage classified without ABG as the maximum modeled probability of a wrong call at a display rises. Steps mark changes in the consecutive qualifying displays. Dots identify the three named caps. Shading gives pointwise 95% credible intervals conditional on usable readings, the selected rule and fixed arterial distribution.",
    )
    d.add_page_break()
    d.add_heading("Encounter membership by indication and setting", 1)
    d.add_paragraph(
        "Counts compare keys on one recovered source snapshot. Each group of three columns gives shared, A-only and B-only patients or encounters. A shared patient can have different encounters. Reasons and settings overlap. The archive comparison uses final archived keys at both stages."
    )
    titles = re.findall(r"(?m)^## (.+\(A\).+)$", encounter)
    for i, (title, t) in enumerate(zip(titles, es[2:-1], strict=True)):
        # Unpack the patient and encounter triples into separate columns to keep type readable.
        rows = [
            ["Setting", "Indication", "Patients (both / A / B)", "Encounters (both / A / B)"]
        ] + [[r[0], r[1], " / ".join(r[2:5]), " / ".join(r[5:8])] for r in t[1:]]
        tab(d, f"Supplementary Table 26{chr(65 + i)}. {title}", rows, [0.7, 1.38, 2.08, 2.13])
    tab(
        d,
        "Supplementary Table 27. Effect of conditioning on a rounded display",
        all_tables[-1],
        [1.25, 0.8, 1.38, 1.38, 1.45],
    )
    d.add_paragraph(
        "All ranges are inclusive ABG bands within 20–100 mmHg. Exact-reading estimates come from the released lookup. Qualification includes the numerical uncertainty of the relevant estimate."
    )
    d.save(dest)


def run(parents, dest, review, work, editable):
    from private_inputs import private_destination

    private_destination(dest)
    pol = json.loads((work / "all-policy.json").read_bytes())["policies"]
    art = review / "referral-all.png"
    short(parents / "letter.docx", dest / "letter.docx", pol, art)
    main_version(parents / "original-research.docx", dest / "original-research.docx", pol, art)
    supplement(parents / "supplement.docx", dest / "supplement.docx", review, work)
    if editable:
        short(editable, work / "Short Report - editable with comments.docx", pol, art)
    receipt = {
        f: {
            role: hashlib.sha256((folder / f).read_bytes()).hexdigest()
            for role, folder in (("parent", parents), ("successor", dest))
        }
        for f in ("letter.docx", "original-research.docx", "supplement.docx")
    }
    (work / "document-parents.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("parents", "dest", "review", "work"):
        parser.add_argument(n, type=lambda s: Path(s).resolve())
    parser.add_argument("--editable", type=lambda s: Path(s).resolve())
    run(**vars(parser.parse_args()))
