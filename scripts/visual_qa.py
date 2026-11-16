"""Generate local visual QA screenshots for the static browser app."""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage_web_python import stage_web_python  # noqa: E402

OUTPUT_DIR = ROOT / ".pytest_tmp" / "visual-qa"
SYNTHETIC_PRIOR_PATH = ROOT / "tests" / "fixtures" / "synthetic_paco2_prior.csv"


def main() -> None:
    stage_web_python(ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    previews = ROOT / ".pytest_tmp/e2e/prediction-interval-preview"
    previews.mkdir(parents=True, exist_ok=True)
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
                chart = page.locator("#lookup-chart")
                chart.locator(".density-line").wait_for()
                assert page.locator("#lookup-tc").input_value() == "47"
                initial = time.perf_counter() - start
                view = page.locator(".predict .workspace")
                view.screenshot(path=str(previews / f"{device}-opening.png"))
                page.locator("#lookup-demo").click()
                page.locator("#lookup-bounds").get_by_text("38.75 to 56.25 mmHg").wait_for()
                assert chart.locator(".tail-value").all_text_contents() == ["71.61%", "13.43%"]
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
                    "document.querySelector('#estimate').textContent === '48.0 mmHg' || document.querySelector('#error').textContent !== ''",
                    timeout=90000,
                )
                assert not page.locator("#error").inner_text(), page.locator("#error").inner_text()
                latency = time.perf_counter() - start
                assert page.locator("#probability").inner_text() == "0.8182"
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
