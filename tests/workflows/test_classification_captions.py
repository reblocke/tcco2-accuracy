import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_classification_captions_follow_the_lookup(tmp_path, monkeypatch):
    reason = "Rendering the manuscript exhibits requires the manuscript dependency group"
    Document = pytest.importorskip("docx", reason=reason).Document
    Figure = pytest.importorskip("matplotlib.figure", reason=reason).Figure
    Image = pytest.importorskip("PIL.Image", reason=reason)
    monkeypatch.syspath_prepend(str(ROOT / "scripts/manuscript"))
    import inverse_documents
    import inverse_exhibits

    pages, parents, exhibits, output = (
        tmp_path / name for name in ("pages", "parents", "exhibits", "word")
    )
    for folder in (pages, parents, exhibits, output):
        folder.mkdir()
    parent = Document()
    old_image = parents / "earlier-figure.png"
    Image.new("RGB", (2, 2), color="white").save(old_image)
    parent.add_picture(str(old_image))
    parent.add_paragraph("Supplementary Figure 2. Earlier figure")
    parent.add_paragraph("Model-dependence analyses")
    source = parents / "supplement.docx"
    parent.save(source)
    monkeypatch.setitem(
        inverse_documents.PARENTS, "supplement", hashlib.sha256(source.read_bytes()).hexdigest()
    )

    # Inspect the All-settings bar at exactly 45 as the real PNG is written.
    rendered = {}
    savefig = Figure.savefig

    def capture(fig, filename, *args, **kwargs):
        if Path(filename).name == "classification-by-reading.png":
            rendered["text"] = [label.get_text() for label in fig.texts]
            rendered["segments"] = [
                bar.get_height()
                for bar in fig.axes[0].patches
                if abs(bar.get_x() + bar.get_width() / 2 - 45) < 1e-8
            ]
        return savefig(fig, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    lookup = json.loads((ROOT / "models/pages/conditional-lookup.json").read_bytes())
    release = json.loads((ROOT / "models/pages/conditional-release.json").read_bytes())
    row = lookup["groups"]["all"]["rows"][45 - 20]

    for probability, true_positive, false_positive in [(0.32, 32, 68), (0.625, 63, 37)]:
        row["tails"][0] = probability
        row["report"]["probabilities"][0] = [probability, probability]
        payload = json.dumps(lookup).encode()
        (pages / "conditional-lookup.json").write_bytes(payload)
        release["lookup_sha256"] = hashlib.sha256(payload).hexdigest()
        (pages / "conditional-release.json").write_text(json.dumps(release))
        if (exhibits / "classification-by-reading.png").exists():
            with pytest.raises(ValueError, match="lookup"):
                inverse_documents.revise(parents, exhibits, output, "supplement", pages=pages)

        receipt, probabilities = inverse_exhibits.read_lookup(pages)
        inverse_exhibits.bars(probabilities, exhibits, lookup_sha256=receipt["lookup_sha256"])
        inverse_documents.revise(parents, exhibits, output, "supplement", pages=pages)
        np.testing.assert_allclose(rendered["segments"], [probability, 1 - probability])
        assert any(
            f"{true_positive} of 100 calls are true positives and {false_positive} are false positives"
            in text
            for text in rendered["text"]
        )
        supplement = Document(output / "supplement.docx")
        (caption,) = (
            p.text for p in supplement.paragraphs if p.text.startswith("Supplementary Figure 2b.")
        )
        assert (
            f"{true_positive} calls per 100 are true positives and {false_positive} are false positives"
            in caption
        )
        _, shape = supplement.inline_shapes
        image_id = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        assert (
            supplement.part.related_parts[image_id].blob
            == (exhibits / "classification-by-reading.png").read_bytes()
        )

    image = exhibits / "classification-by-reading.png"
    image.write_bytes(image.read_bytes() + b"altered image")
    with pytest.raises(ValueError, match="classification-by-reading.png"):
        inverse_documents.revise(parents, exhibits, output, "supplement", pages=pages)
