"""Inventory the designated private Word folders locally. Emit visible text only on request."""

import hashlib
import io
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
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
                    child.tag == W + "r"
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

        def paragraph(p):
            # Only visible Word text. Skip field instructions, deleted/hidden text and objects.
            text = []
            for run in p.iter(W + "r"):
                if any(run.find(W + "rPr/" + W + k) is not None for k in ("vanish", "webHidden")):
                    continue
                text.extend(t.text or "" for t in run.findall(W + "t"))
            return "".join(text)

        result = []
        for item in body:
            if item.tag == W + "p":
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
        return result


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
