"""Inventory the designated private Word folders locally. Emit visible text only on request."""

import hashlib
import io
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MATH = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
ROLES = {"letter": "letter.docx", "supplement": "supplement.docx"}


def files(roots):
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for p in sorted(root.glob("*.docx")):
            if p.name.startswith("~$") or p.is_symlink():
                continue
            if "letter" in p.stem.lower():
                yield "letter", p
            elif "supplement" in p.stem.lower():
                yield "supplement", p


def equation(node):
    """Copy the native mathematical structure and its visible tokens without Word metadata."""
    if not node.tag.startswith(MATH):
        raise ValueError(
            "Native equation contains unsupported embedded content. Check the source equation in Word."
        )
    clean = ET.Element(
        node.tag,
        {
            k: v
            for k, v in node.attrib.items()
            if k.startswith(MATH) or k == "{http://www.w3.org/XML/1998/namespace}space"
        },
    )
    if node.tag == MATH + "t":
        clean.text = node.text
    for child in node:
        if child.tag != W + "rPr":
            clean.append(equation(child))
    return clean


def elements(data):
    with ZipFile(io.BytesIO(data)) as z:
        doc = ET.fromstring(z.read("word/document.xml"))
        body = doc.find(W + "body")
        for parent in body.iter():
            for child in list(parent):
                if child.tag in {
                    W + "del",
                    W + "moveFrom",
                    W + "pict",
                    W + "object",
                    W + "txbxContent",
                } or (
                    child.tag in {W + "r", MATH + "r"}
                    and any(
                        child.find(W + "rPr/" + W + k) is not None for k in ("vanish", "webHidden")
                    )
                ):
                    parent.remove(child)
        rels = {
            e.get("Id"): e.get("Target", "")
            for e in ET.fromstring(z.read("word/_rels/document.xml.rels"))
            if e.get("TargetMode") != "External"
        }

        # Replace each equation in its original position, including equations in table cells.
        equations = {}
        for parent in body.iter():
            for index, child in enumerate(list(parent)):
                if child.tag == MATH + "oMath":
                    xml = ET.tostring(equation(child), encoding="unicode")
                    token = "OMML" + hashlib.sha256(xml.encode()).hexdigest()
                    equations[token] = xml
                    run = ET.Element(W + "r")
                    ET.SubElement(run, W + "t").text = token
                    parent.remove(child)
                    parent.insert(index, run)

        def paragraph(p):
            return "".join(t.text or "" for run in p.iter(W + "r") for t in run.findall(W + "t"))

        result = []
        for item in body:
            if item.tag in (W + "p", MATH + "oMathPara", W + "r"):
                s = paragraph(item)
                if s.strip():
                    result.append(s)
                for picture in item.findall(".//{*}blip"):
                    ref = picture.get(
                        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                    )
                    path = rels.get(ref, "")
                    if path.startswith("media/") and "word/" + path in z.namelist():
                        result.append({"image": hashlib.sha256(z.read("word/" + path)).hexdigest()})
            elif item.tag == W + "tbl":
                result.append(
                    [
                        [
                            " ".join(paragraph(p) for p in cell.findall(W + "p"))
                            for cell in row.findall(W + "tc")
                        ]
                        for row in item.findall(W + "tr")
                    ]
                )
        return {"blocks": result, "equations": equations}


def main():
    primary = Path(sys.argv[1]).expanduser()
    if not primary.is_absolute():
        primary = Path.home() / primary
    backup = Path.home() / "Documents/Codex/llm-readiness-working/tcco2-accuracy/Drafts"
    rows = []
    for role, p in files([primary, backup]):
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(sys.argv) > 2 and digest == sys.argv[2]:
            print(json.dumps(elements(data)))
            return
        rows.append([role, p.name, digest])
    if len(sys.argv) > 2:
        raise ValueError("Source changed during inspection")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
