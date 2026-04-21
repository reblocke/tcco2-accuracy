"""Generate local visual QA screenshots for the static browser app."""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from playwright.sync_api import Browser, Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage_web_python import stage_web_python  # noqa: E402

OUTPUT_DIR = ROOT / ".pytest_tmp" / "visual-qa"
MIN_SCREENSHOT_BYTES = 5_000
DEFAULT_TRACES = ["Posterior", "Likelihood (scaled)", "Prior"]


def main() -> None:
    """Stage the app, run browser states, and write review screenshots."""

    stage_web_python(ROOT)
    _assert_staged_public_prior()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with _serve_web() as base_url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                _capture_desktop_states(browser, base_url)
                _capture_mobile_state(browser, base_url)
            finally:
                browser.close()

    print("Visual QA screenshots:")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"- {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")


def _capture_desktop_states(browser: Browser, base_url: str) -> None:
    context = browser.new_context(viewport={"width": 1440, "height": 950})
    page = context.new_page()
    try:
        page.goto(base_url, wait_until="domcontentloaded")
        _wait_for_ready(page)
        _assert_public_prior_fetchable(page)
        _wait_for_trace_names(page, DEFAULT_TRACES)
        _assert_chart_contract(page, DEFAULT_TRACES, require_prior_labels=True)
        _screenshot(page, "desktop-default.png")

        page.locator("input[name='mode'][value='likelihood_only']").check()
        page.locator("#calculate").click()
        _wait_for_trace_names(page, ["Posterior"])
        _assert_chart_contract(page, ["Posterior"], require_prior_labels=False)
        _screenshot(page, "desktop-likelihood-only.png")

        page.locator("input[name='mode'][value='prior_weighted']").check()
        page.locator("#threshold").fill("150")
        page.locator("#calculate").click()
        _wait_for_trace_names(page, DEFAULT_TRACES)
        page.wait_for_function(
            """
            () => document
              .querySelector("#chart-caption")
              ?.textContent
              .includes("outside the focused plot range")
            """,
            timeout=180_000,
        )
        _assert_chart_contract(page, DEFAULT_TRACES, require_prior_labels=True)
        _screenshot(page, "desktop-threshold-outside.png")
    finally:
        context.close()


def _capture_mobile_state(browser: Browser, base_url: str) -> None:
    context = browser.new_context(
        viewport={"width": 390, "height": 900},
        is_mobile=True,
        device_scale_factor=1,
    )
    page = context.new_page()
    try:
        page.goto(base_url, wait_until="domcontentloaded")
        _wait_for_ready(page)
        _wait_for_trace_names(page, DEFAULT_TRACES)
        _assert_chart_contract(page, DEFAULT_TRACES, require_prior_labels=True)
        _screenshot(page, "mobile-default.png")
    finally:
        context.close()


def _assert_chart_contract(
    page: Page,
    trace_names: list[str],
    *,
    require_prior_labels: bool,
) -> None:
    state = _chart_state(page)
    if state["trace_names"] != trace_names:
        raise AssertionError(f"Unexpected chart traces: {state['trace_names']}")
    if state["showlegend"]:
        raise AssertionError("Expected direct labels with detached legend hidden.")
    if "Median" not in state["annotation_text"]:
        raise AssertionError("Expected median marker label in chart annotations.")
    if require_prior_labels:
        for label in ("Prior", "Likelihood (scaled)"):
            if label not in state["annotation_text"]:
                raise AssertionError(f"Missing direct chart label: {label}")
    else:
        for label in ("Prior", "Likelihood (scaled)"):
            if label in state["annotation_text"]:
                raise AssertionError(f"Unexpected direct chart label: {label}")


def _assert_staged_public_prior() -> None:
    data_dir = ROOT / "web" / "assets" / "data"
    public_prior = data_dir / "paco2_public_prior.csv"
    exact_prior = data_dir / "paco2_prior_bins.csv"
    if not public_prior.exists():
        raise AssertionError(f"Missing staged public prior: {public_prior}")
    if exact_prior.exists():
        raise AssertionError(f"Exact prior should not be staged: {exact_prior}")


def _assert_public_prior_fetchable(page: Page) -> None:
    public_ok = page.evaluate(
        "() => fetch('assets/data/paco2_public_prior.csv').then((response) => response.ok)"
    )
    exact_ok = page.evaluate(
        "() => fetch('assets/data/paco2_prior_bins.csv').then((response) => response.ok)"
    )
    if public_ok is not True:
        raise AssertionError("Expected public prior asset to load over HTTP.")
    if exact_ok is True:
        raise AssertionError("Exact count prior should not be served by the static app.")


def _screenshot(page: Page, filename: str) -> None:
    path = OUTPUT_DIR / filename
    page.screenshot(path=path, full_page=True)
    size = path.stat().st_size
    if size < MIN_SCREENSHOT_BYTES:
        raise AssertionError(f"Screenshot appears empty: {path} ({size} bytes)")


def _chart_state(page: Page) -> dict[str, Any]:
    return page.evaluate(
        """
        () => {
          const chart = document.querySelector("#posterior-chart");
          const annotations = chart.layout.annotations ?? [];
          return {
            trace_names: chart.data.map((trace) => trace.name),
            showlegend: Boolean(chart._fullLayout.showlegend),
            annotation_text: annotations.map((annotation) => annotation.text),
          };
        }
        """
    )


def _wait_for_ready(page: Page) -> None:
    page.get_by_text("Calculation complete.").wait_for(timeout=180_000)


def _wait_for_trace_names(page: Page, trace_names: list[str]) -> None:
    page.wait_for_function(
        """
        (expectedNames) => {
          const chart = document.querySelector("#posterior-chart");
          const names = chart?.data?.map((trace) => trace.name) ?? [];
          return names.length === expectedNames.length &&
            names.every((name, index) => name === expectedNames[index]);
        }
        """,
        arg=trace_names,
        timeout=180_000,
    )


@contextlib.contextmanager
def _serve_web() -> Iterator[str]:
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=ROOT / "web",
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
