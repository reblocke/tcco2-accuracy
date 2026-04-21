from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from playwright.sync_api import expect

from scripts.stage_web_python import stage_web_python

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def web_server() -> str:
    stage_web_python(ROOT)
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
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)


def test_static_app_default_calculation(page, web_server: str) -> None:
    page.goto(web_server, wait_until="domcontentloaded")

    page.get_by_text("Calculation complete.").wait_for(timeout=180_000)

    assert page.locator("#metric-interval").inner_text() != "-"
    assert page.locator("#metric-probability").inner_text() != "-"
    assert page.locator("#posterior-chart .main-svg").count() >= 1
    expect(page.get_by_text("TcCO2 threshold result", exact=True)).to_be_visible()
    expect(page.get_by_text("Threshold classification mass", exact=True)).to_be_visible()
    expect(page.get_by_text("At/above threshold", exact=True)).to_be_visible()
    expect(page.get_by_text("Posterior mass at/above threshold:")).to_be_visible()
    expect(page.get_by_text("TcCO2 decision", exact=True)).to_have_count(0)
    expect(page.get_by_text("Decision correctness", exact=True)).to_have_count(0)
    expect(page.get_by_text("True positive")).to_have_count(0)
    expect(page.get_by_text("False positive")).to_have_count(0)


def test_static_app_prior_weighted_chart_uses_posterior_focused_axis(page, web_server: str) -> None:
    page.goto(web_server, wait_until="domcontentloaded")
    page.get_by_text("Calculation complete.").wait_for(timeout=180_000)

    state = _chart_state(page)

    assert state["trace_names"] == ["Posterior", "Likelihood (scaled)", "Prior"]
    assert state["yaxis_title"] == "Probability per bin"
    assert state["showlegend"] is False
    assert "Likelihood (scaled)" in state["annotation_text"]
    assert "Prior" in state["annotation_text"]
    assert state["range_width"] < state["trace_width"] * 0.4
    assert "Median" in state["annotation_text"]
    assert "PI low" in state["annotation_text"]
    assert "PI high" in state["annotation_text"]
    assert state["annotation_lanes"] > 1


def test_static_app_likelihood_only_chart_uses_posterior_focused_axis(
    page, web_server: str
) -> None:
    page.goto(web_server, wait_until="domcontentloaded")
    page.get_by_text("Calculation complete.").wait_for(timeout=180_000)

    page.locator("input[name='mode'][value='likelihood_only']").check()
    page.locator("#calculate").click()
    _wait_for_trace_names(page, ["Posterior"])

    state = _chart_state(page)

    assert state["trace_names"] == ["Posterior"]
    assert state["showlegend"] is False
    assert "Likelihood (scaled)" not in state["annotation_text"]
    assert "Prior" not in state["annotation_text"]
    assert state["range_width"] < state["trace_width"]
    assert "Median" in state["annotation_text"]
    assert state["annotation_lanes"] > 1


def test_static_app_threshold_change_updates_metric(page, web_server: str) -> None:
    page.goto(web_server, wait_until="domcontentloaded")
    page.get_by_text("Calculation complete.").wait_for(timeout=180_000)

    page.locator("#threshold").fill("150")
    page.locator("#calculate").click()

    expect(page.locator("#metric-threshold-label")).to_contain_text("150", timeout=180_000)
    expect(page.locator("#chart-caption")).to_contain_text("outside the focused plot range")
    assert page.locator("#posterior-chart .main-svg").count() >= 1


def _chart_state(page) -> dict[str, bool | float | int | str | list[str]]:
    return page.evaluate(
        """
        () => {
          const chart = document.querySelector("#posterior-chart");
          const range = chart._fullLayout.xaxis.range.map(Number);
          const traceX = chart.data[0].x.map(Number);
          const annotations = chart.layout.annotations ?? [];
          const markerAnnotations = annotations.filter((annotation) => annotation.yref === "paper");
          const annotationYs = markerAnnotations.map((annotation) =>
            Number(annotation.y).toFixed(3)
          );
          return {
            range_width: range[1] - range[0],
            trace_width: Math.max(...traceX) - Math.min(...traceX),
            trace_names: chart.data.map((trace) => trace.name),
            yaxis_title: chart._fullLayout.yaxis.title.text,
            showlegend: Boolean(chart._fullLayout.showlegend),
            annotation_text: annotations.map((annotation) => annotation.text),
            annotation_lanes: new Set(annotationYs).size,
          };
        }
        """
    )


def _wait_for_trace_names(page, trace_names: list[str]) -> None:
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
