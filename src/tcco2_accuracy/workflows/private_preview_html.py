"""Package the data-free chart viewer for use from a local file without a server."""

import base64
import hashlib
import re

from .future_setting_data import ROOT


def private_viewer_html(chart_sha256: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", chart_sha256):
        raise ValueError("Expected the private chart file's SHA-256")
    assets = ROOT / "web/assets"
    template = (ROOT / "scripts/private_preview.html").read_text()
    html = (
        template.replace("/* SHARED_STYLES */", (assets / "css/styles.css").read_text())
        .replace("/* DISTRIBUTION_MODULE */", (assets / "js/distribution.js").read_text())
        .replace("/* INTERPRETATION_MODULE */", (assets / "js/prediction.js").read_text())
        .replace("EXPECTED_CHART_SHA256", chart_sha256)
    )
    script = html.split('<script type="module">', 1)[1].split("</script>", 1)[0]
    script_hash = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    return html.replace("VIEWER_SCRIPT_HASH", script_hash)
