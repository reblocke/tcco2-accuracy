"""Update the review copies from the checksum-verified November 25 predecessors."""

import argparse
import io
import json
import os
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from classification_example import positive_calls_per_100, read_lookup
from docx import Document
from docx.oxml.ns import qn
from PIL import Image
from private_inputs import digest, private_destination, require_digest
from referral_documents import CAPTION, insert_figure

PARENTS = {
    # Build the supplement first so a stale figure is detected before the other manuscripts are written.
    "supplement": "152c755dcb2e35323a1d7c0fa3a715e4a5c1d64f25a73412598e82bff6ed214d",
    "letter": "cd38fd8df865b7eb819f8c0ddf46b42c0233c24a44c9ad60f57d5a7604a11414",
    "original-research": "789d2303bc6f98622eda25d2f1ec7ecf710600437ac46a3ad513cd9785951612",
}
DEFAULT_PAGES = Path(__file__).resolve().parents[2] / "models/pages"
# PR #26 predates the per-figure source receipt. These are its reviewed bytes.
ARCHIVED_BAR_SOURCE = {
    "lookup_sha256": "6de657bd14e5b0a2de471000b4abd692dc42eca9e601ac999c9114c67964bbaa",
    "png_sha256": "414feaefcc500e4ce77688e1375eb81932ddd6263fd01f2b16b82ce3168adc6f",
}
RISK_CAPTION = (
    "Supplementary Figure 2a. Modeled probabilities of simultaneous PaCO₂ ≥45 and ≥52 mmHg at each exact observed TcCO₂. "
    "Lines give the full posterior-averaged predictions. Shading gives pointwise 95% credible intervals for these probabilities, conditional on the fixed arterial distribution. "
    "Within each agreement-parameter draw we computed both threshold risks, then weighted each draw by the density it assigned to the observed reading. "
    "A gap in shading indicates that an endpoint failed its numerical precision check."
)
REVISIONS = {
    "We compared referral for displays of 40–50 mmHg with three post hoc rules. Each limited the probability of misclassifying simultaneous PaCO₂ ≥45 at a definitive display to 1%, 5% or 10%. We integrated over each display bin, referred readings outside 20–100 and assumed a timely ABG resolved each referral.": "We compared a fixed rule in which patients with displays of 40–50 mmHg receive an ABG with three post hoc rules. A reading could classify simultaneous PaCO₂ relative to 45 mmHg only if its modeled chance of error was at most 1%, 5% or 10%, respectively. We integrated over each display bin. Patients with displays outside 20–100 received an ABG under every rule. We assumed a timely ABG resolved arterial status.",
    "For displayed TcCO₂, the 5% rule referred 39–60 mmHg and the fixed rule referred 40–50 mmHg. Per 100 eligible testing occasions, the two rules referred 51.4 and 37.0, leaving 0.77 and 1.03 false negatives and 0.12 and 1.79 false positives, respectively (Figure 2).": "Within the 20–100 mmHg display range, patients received an ABG at 39–60 under the 5% policy and at 40–50 under the fixed policy. Across all eligible occasions, posterior medians per 100 were 51.4 and 37.0 receiving an ABG, 0.77 and 1.03 false negatives, and 0.12 and 1.79 false positives, respectively. Figure 2 shows the posterior means.",
    "We also modeled classification errors and ABG referrals under two illustrative testing rules.": "We also estimated how many patients would receive an ABG and how many classification errors would remain under alternative testing rules.",
    "Modeled classification and referral": "Modeled classification and receipt of an ABG",
    "The historical continuous 40–50 mmHg strategy referred 30.07 per 100 eligible testing occasions": "Under the historical continuous 40–50 mmHg strategy, 30.07 per 100 eligible testing occasions received an ABG",
    "and left 1.20 false negatives": "and there were 1.20 false negatives",
    "Table 2 gives the referral burden and remaining errors at each cap.": "Table 2 shows how many patients receive an ABG and how many errors remain under each policy.",
    "Table 2. Referral and residual classification errors in All settings": "Table 2. Receipt of an ABG and residual classification errors in All settings",
    "All usable displays outside 20–100 also receive ABG.": "Patients with usable displays outside 20–100 also receive an ABG.",
    "Refer both endpoints and the intervening readings. Sensitivity and specificity among those classified exclude referrals. With timely definitive ABGs the continuous rule refers 30.07 per 100 in All settings. Rounding and off-grid referrals raise this to 36.95 for the displayed fixed rule.": "Patients with readings from 40 through 50 mmHg receive an ABG. Sensitivity and specificity among those classified use only occasions classified without an ABG. In All settings, 30.07 per 100 receive an ABG under the continuous rule and 36.95 under the displayed fixed rule, which includes rounding and off-grid ABGs.",
    "Other displays went to ABG, including both out-of-range tails and numerically unresolved boundaries.": "Patients with other displays received an ABG, including those outside the supported range or at a numerically unresolved boundary.",
    "Both overflow displays require referral under every display policy.": "Under every policy, a patient with an off-grid display receives an ABG.",
    "A bound crossing a cap led to referral.": "When a numerical bound crossed a cap, the patient received an ABG.",
    "Definitive calls and referrals outside the grid": "Definitive calls and ABGs outside the grid",
    "Numerically unresolved displays referred to ABG: cap 1% at 100 mmHg.": "Under the 1% policy, a patient with a display of 100 mmHg receives an ABG because its error probability could not be resolved at the required precision.",
}
HEADS = {
    "Referrals /100": "Receives an ABG /100",
    "Off-grid referrals /100": "Receives an ABG outside the grid /100",
    "Referrals": "Receives an ABG",
    "ABG displays": "Receives an ABG at (mmHg)",
}


def text_edit(paragraph, new):
    old = paragraph.text
    if old == new:
        return
    nodes = paragraph._p.xpath(".//w:t")
    if (
        not nodes
        or "".join(n.text or "" for n in nodes) != old
        or paragraph._p.xpath(".//w:instrText | .//w:fldChar | .//w:drawing")
    ):
        raise ValueError("Paragraph requires a field- or object-aware edit")
    positions = []
    i = 0
    for node in nodes:
        positions.append((i, i + len(node.text or "")))
        i = positions[-1][1]
    fragments = [""] * len(nodes)
    for action, lo, hi, a, b in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if action == "equal":
            for j, (start, end) in enumerate(positions):
                if max(lo, start) < min(hi, end):
                    fragments[j] += old[max(lo, start) : min(hi, end)]
        elif action in ("insert", "replace"):
            j = next(
                (j for j, (start, end) in enumerate(positions) if start <= lo < end), len(nodes) - 1
            )
            fragments[j] += new[a:b]
    for node, fragment in zip(nodes, fragments, strict=True):
        node.text = fragment
        node.set(qn("xml:space"), "preserve")
    if paragraph.text != new:
        raise ValueError("Run-preserving edit changed the intended reading")


def bar_caption(image, pages):
    release, probabilities = read_lookup(pages)
    sidecar = image.with_suffix(".source.json")
    source = json.loads(sidecar.read_bytes()) if sidecar.exists() else ARCHIVED_BAR_SOURCE
    if source["lookup_sha256"] != release["lookup_sha256"]:
        raise ValueError("Classification figure and Word caption use different lookup revisions")
    require_digest(image, source["png_sha256"])
    true_positives, false_positives = positive_calls_per_100(probabilities)
    return (
        "Supplementary Figure 2b. Results of classifying TcCO₂ ≥45 mmHg as positive. Each column represents 100 eligible occasions with that exact reading. "
        "Below an observed 45, arterial hypercapnia is a false negative. At or above an observed 45, an arterial value below 45 is a false positive. "
        f"In All settings at an observed 45, about {true_positives} calls per 100 are true positives and {false_positives} are false positives."
    )


def revise(parents, exhibits, dest, stem, *, pages=DEFAULT_PAGES):
    source = parents / (stem + ".docx")
    require_digest(source, PARENTS[stem])
    doc = Document(source)
    if stem == "supplement":
        anchor = next(p for p in doc.paragraphs if p.text.startswith("Supplementary Figure 2."))
        text_edit(anchor, RISK_CAPTION)
        after = next(p for p in doc.paragraphs if p.text == "Model-dependence analyses")
        image = exhibits / "classification-by-reading.png"
        insert_figure(doc, image, bar_caption(image, pages), before=after)
    for p in doc.paragraphs:
        t = p.text
        if "Disposition per 100 eligible testing occasions under the 1%" in t:
            t = t.split("Disposition per 100", 1)[0] + CAPTION
        else:
            for a, b in REVISIONS.items():
                t = t.replace(a, b)
        text_edit(p, t)
    for t in doc.tables:
        for cell in t.rows[0].cells:
            for p in cell.paragraphs:
                text_edit(p, HEADS.get(p.text, p.text))
    old_images = {digest(p): p for p in parents.glob("*.png")}
    for shape in doc.inline_shapes:
        rel = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        part = doc.part.related_parts[rel]
        old = old_images.get(sha256(part.blob).hexdigest())
        if old is None or not (exhibits / old.name).is_file():
            continue
        file = exhibits / old.name
        part._blob = file.read_bytes()
        with Image.open(file) as image:
            shape.height = round(shape.width * image.height / image.width)
        alt = (
            "Modeled arterial threshold probabilities by exact observed TcCO₂ with pointwise credible intervals."
            if old.name == "classification-figure.png"
            else "Modeled outcomes under whole-number TcCO₂ policies for receipt of an ABG."
        )
        shape._inline.docPr.set("descr", alt)
    if stem == "supplement":
        insert = next(
            s
            for s in doc.inline_shapes
            if doc.part.related_parts[s._inline.graphic.graphicData.pic.blipFill.blip.embed].blob
            == (exhibits / "classification-by-reading.png").read_bytes()
        )
        insert._inline.docPr.set(
            "descr",
            "True and false classifications per 100 occasions with each exact observed TcCO₂.",
        )
    updated = io.BytesIO()
    doc.save(updated)
    # Keep original package parts byte-for-byte apart from the document, relationships and exhibits.
    with (
        ZipFile(source) as original,
        ZipFile(updated) as edited,
        ZipFile(dest / source.name, "w", ZIP_DEFLATED) as output,
    ):
        for name in edited.namelist():
            if (
                name in ("word/document.xml", "word/_rels/document.xml.rels")
                or name.startswith("word/media/")
                or name not in original.namelist()
            ):
                output.writestr(name, edited.read(name))
            else:
                output.writestr(original.getinfo(name), original.read(name))


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("parents", "exhibits", "dest"):
        parser.add_argument(arg, type=lambda s: Path(s).resolve())
    parser.add_argument(
        "--pages",
        type=Path,
        default=DEFAULT_PAGES,
        help="Directory containing the lookup and its release receipt",
    )
    args = parser.parse_args()
    private_destination(args.dest)
    for name in PARENTS:
        revise(args.parents, args.exhibits, args.dest, name, pages=args.pages)
