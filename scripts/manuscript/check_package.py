"""Check the named public Word package and the exact figure bytes allowed inside it."""

import hashlib
import io
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = Path("docs/data_release_contract.json")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strip_empty_pandoc_parts(path):
    """Pandoc emits empty comment/property parts even for plain visible text."""
    output = io.BytesIO()
    omit = {"word/comments.xml", "docProps/custom.xml"}
    with ZipFile(path) as source, ZipFile(output, "w", ZIP_DEFLATED) as dest:
        for name in source.namelist():
            data = source.read(name)
            if name in omit:
                if len(ET.fromstring(data)):
                    raise ValueError("Unexpected comments or custom properties in text export")
                continue
            if name.endswith(".rels") or name == "[Content_Types].xml":
                root = ET.fromstring(data)
                for item in list(root):
                    if item.get("PartName", "").lstrip("/") in omit or item.get("Type", "").rsplit(
                        "/", 1
                    )[-1] in {"comments", "custom-properties"}:
                        root.remove(item)
                data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            dest.writestr(name, data)
    path.write_bytes(output.getvalue())


def inspect_word(data, figures):
    with ZipFile(io.BytesIO(data)) as z:
        if z.testzip() or len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("Invalid Word package")
        for name in z.namelist():
            if name.startswith("word/media/") and sha(z.read(name)) in figures:
                continue
            if not re.fullmatch(
                r"\[Content_Types\]\.xml|_rels/\.rels|docProps/(core|app)\.xml|word/(_rels/[\w.]+\.rels|theme/theme\d+\.xml|(document|styles|stylesWithEffects|numbering|settings|webSettings|fontTable|header\d+|footer\d+|footnotes|endnotes)\.xml)",
                name,
            ):
                raise ValueError(f"Unreviewed Word part: {name}")
            tree = ET.fromstring(z.read(name))
            for e in tree.iter():
                tag = e.tag.split("}")[-1]
                if e.get("TargetMode") == "External" or tag in {
                    "ins",
                    "del",
                    "commentReference",
                    "object",
                    "oleObject",
                    "altChunk",
                    "vanish",
                    "webHidden",
                    "moveFrom",
                    "moveTo",
                    "fldData",
                    "binData",
                }:
                    raise ValueError(f"Unexpected {tag} in {name}")
                if tag == "instrText" and (e.text or "").strip() != "PAGE":
                    raise ValueError("Unexpected Word field instruction")
                if (
                    tag == "fldSimple"
                    and e.get(
                        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}instr", ""
                    ).strip()
                    != "PAGE"
                ):
                    raise ValueError("Unexpected Word field")
        if "word/document.xml" not in z.namelist():
            raise ValueError("Missing document")


def check(root):
    config = json.loads((root / CONTRACT).read_text())["manuscript_wip"]
    figures = {sha((root / p).read_bytes()) for p in config["reviewed"] if p.endswith(".png")}
    # Comparison documents also show the previously published version of an exhibit.
    comparison_figures = {
        digest
        for p, revisions in config["reviewed"].items()
        if p.endswith(".png")
        for digest in revisions
    }
    allowed = set(config["reviewed"])
    found = {
        p.relative_to(root).as_posix()
        for p in (root / "works-in-progress").rglob("*")
        if p.is_file() and p.name != ".DS_Store"
    }
    if found != allowed:
        raise ValueError(f"Manuscript package differs from contract: {sorted(found ^ allowed)}")
    for name, hashes in config["reviewed"].items():
        file = root / name
        data = file.read_bytes()
        if file.is_symlink() or sha(data) not in hashes:
            raise ValueError(f"Unreviewed manuscript file: {name}")
        if name.endswith(".docx"):
            comparison = name.endswith(("-review.docx", "-december-comparison.docx"))
            inspect_word(data, comparison_figures if comparison else figures)


if __name__ == "__main__":
    check(ROOT)
