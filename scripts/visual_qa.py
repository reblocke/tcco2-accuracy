"""Generate local visual QA screenshots for the static browser app."""

from __future__ import annotations

import contextlib
import hashlib
import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage_web_python import stage_web_python  # noqa: E402
from tcco2_accuracy.workflows.private_chart import display_curves  # noqa: E402
from tcco2_accuracy.workflows.private_preview_html import private_viewer_html  # noqa: E402

OUTPUT_DIR = ROOT / ".pytest_tmp" / "visual-qa"
SYNTHETIC_PRIOR_PATH = ROOT / "tests" / "fixtures" / "synthetic_paco2_prior.csv"


def _synthetic_private_preview():
    """Prepare the offline viewer using only the public, invented reference row."""
    directory = ROOT / ".pytest_tmp/e2e/private-preview"
    directory.mkdir(parents=True, exist_ok=True)
    reference = json.loads((ROOT / "models/pages/synthetic-lookup.json").read_text())
    if reference["source"] != "synthetic" or reference["version"] != "conditional_lookup_dense_v2":
        raise ValueError("Private visual QA requires the public synthetic fixture")
    group = reference["groups"]["all"]
    row = group["rows"][47 - 20]
    a = np.asarray(group["support"])
    curve = group["curve"]
    grid = np.asarray(curve["x"])
    indices = np.searchsorted(grid, a - 47)
    np.testing.assert_array_equal(grid[indices], a - 47)
    density = np.asarray(curve["density"])[indices]
    doc = dict(
        version="conditional_lookup_private_chart_v1",
        source="synthetic",
        method=reference["method"],
        prior_uncertainty=reference["prior_uncertainty"],
        target="all",
        tcco2=47,
        posterior=group["posterior"],
        support=group["support"],
        prior=group["prior"],
        mass=row["mass"],
        interval=row["interval"],
        tails=row["tails"],
        likelihood=(density / density.max()).tolist(),
        display=display_curves(a, group["prior"], row["mass"]),
    )
    chart = directory / "private-chart.json"
    chart.write_text(json.dumps(doc, allow_nan=False, separators=(",", ":")) + "\n")
    (directory / "preview.html").write_text(
        private_viewer_html(hashlib.sha256(chart.read_bytes()).hexdigest())
    )
    return directory


def _capture_private_preview(directory, browser, device, width, height):
    with browser.new_context(
        offline=True, viewport={"width": width, "height": height}, is_mobile=device == "mobile"
    ) as context:
        page = context.new_page()
        page.goto((directory / "preview.html").as_uri())
        page.locator("#chart-file").set_input_files(directory / "private-chart.json")
        page.locator("#private-bounds").get_by_text("38.75 to 56.25 mmHg").wait_for()
        assert "Synthetic example" in page.locator("#private-context").inner_text()
        chart = page.locator("#private-chart")
        assert chart.locator(".tail-value").all_text_contents() == ["71.61%", "13.43%"]
        if device == "mobile":
            chart.get_by_role("button", name="Highlight ≥52 mmHg").click()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.mouse.move(0, 0)
        view = page.locator("main")
        view.screenshot(path=directory / f"{device}-47.png")
        gray = page.add_style_tag(content="main {filter: grayscale(1)}")
        view.screenshot(path=directory / f"{device}-47-grayscale.png")
        gray.evaluate("(style) => style.remove()")
        if device == "desktop":
            svg = chart.locator("svg")
            svg.scroll_into_view_if_needed()
            box = svg.bounding_box()
            y = box["y"] + box["height"] * 0.40
            page.mouse.move(box["x"] + box["width"] * 0.33, y)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] * 0.62, y, steps=10)
            page.mouse.up()
            chart.get_by_role("button", name="Reset zoom").wait_for()
            page.mouse.move(0, 0)
            view.screenshot(path=directory / "desktop-47-zoom.png")


def main() -> None:
    stage_web_python(ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    previews = ROOT / ".pytest_tmp/e2e/prediction-interval-preview"
    previews.mkdir(parents=True, exist_ok=True)
    private_preview = _synthetic_private_preview()
    measurements = []
    with _serve_web() as url, sync_playwright() as playwright:
        for device, width, height in [("desktop", 1440, 950), ("mobile", 390, 900)]:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(
                    viewport={"width": width, "height": height}, is_mobile=device == "mobile"
                )
                start = time.perf_counter()
                page.goto(url, wait_until="domcontentloaded")
                assert page.locator("#lookup-tc").input_value() == ""
                page.fill("#lookup-tc", "47")
                page.locator("#lookup-chart .posterior-line").wait_for()
                chart = page.locator("#lookup-chart")
                page.locator("#lookup-bounds").get_by_text("35 to 53 mmHg").wait_for()
                assert page.locator("#lookup-values .tail-value").all_text_contents() == [
                    "47%",
                    "5%",
                ]
                assert page.locator("#lookup-tc").input_value() == "47"
                initial = time.perf_counter() - start
                view = page.locator(".predict .workspace")
                view.screenshot(path=str(previews / f"{device}-opening.png"))
                gray = page.add_style_tag(content=".workspace {filter: grayscale(1)}")
                view.screenshot(path=str(previews / f"{device}-opening-grayscale.png"))
                gray.evaluate("style => style.remove()")
                page.goto(url + "/research.html")
                page.screenshot(path=str(OUTPUT_DIR / f"{device}-research.png"), full_page=True)
                page.locator("#lookup-tools > summary").click()
                page.locator("#lookup-demo").click()
                page.locator("#lookup-bounds").get_by_text("39 to 56 mmHg").wait_for()
                assert chart.locator(".tail-value").all_text_contents() == ["72%", "13%"]
                if device == "mobile":
                    chart.get_by_role("button", name="Highlight ≥52 mmHg").click()
                page.mouse.move(0, 0)
                view.screenshot(path=str(previews / f"{device}-47.png"))
                monochrome = page.add_style_tag(content="#lookup-chart {filter: grayscale(1)}")
                chart.screenshot(path=str(previews / f"{device}-47-grayscale.png"))
                monochrome.evaluate("(style) => style.remove()")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.screenshot(path=str(OUTPUT_DIR / f"{device}-prediction.png"), full_page=False)
                page.screenshot(
                    path=str(OUTPUT_DIR / f"{device}-prediction-full.png"), full_page=True
                )
                page.locator("#advanced > summary").click()
                page.locator("#bounds").get_by_text("−11.5 to +11.2 mmHg").wait_for()
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.screenshot(path=str(OUTPUT_DIR / f"{device}-agreement.png"), full_page=True)
                page.select_option("#subgroup", "ed_inp")
                page.locator("#inverse").check()
                page.locator("#prior-file").set_input_files(SYNTHETIC_PRIOR_PATH)
                page.fill("#tcco2", "45")
                start = time.perf_counter()
                page.click("#calculate")
                page.wait_for_function(
                    "document.querySelector('#estimate').textContent === '48 mmHg' || document.querySelector('#error').textContent !== ''",
                    timeout=90000,
                )
                assert not page.locator("#error").inner_text(), page.locator("#error").inner_text()
                latency = time.perf_counter() - start
                assert page.locator("#probability").inner_text() == "82%"
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.locator(".calc").screenshot(path=str(OUTPUT_DIR / f"{device}-conditional.png"))
                cdp = browser.new_browser_cdp_session()
                processes = cdp.send("SystemInfo.getProcessInfo")["processInfo"]
                pids = [str(proc["id"]) for proc in processes if proc["type"] == "renderer"]
                rss = subprocess.check_output(["ps", "-o", "rss=", "-p", ",".join(pids)], text=True)
                measurements.append(
                    dict(
                        device=device,
                        viewport=[width, height],
                        stratum="arf",
                        kernel_bytes=(ROOT / "models/pages/arf.npz").stat().st_size,
                        agreement_seconds=initial,
                        cold_inverse_seconds=latency,
                        renderer_rss_kib_after_calculation=sum(map(int, rss.split())),
                    )
                )
                _capture_private_preview(private_preview, browser, device, width, height)
            finally:
                browser.close()
    (OUTPUT_DIR / "measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")
    print(OUTPUT_DIR)


@contextlib.contextmanager
def _serve_web(directory: Path = ROOT / "web") -> Iterator[str]:
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=directory,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_server(port)
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(port: int) -> None:
    deadline = time.time() + 10
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError("Timed out waiting for local web server.")


if __name__ == "__main__":
    main()
