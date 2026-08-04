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
EXPECTED_AGREEMENT_METHOD_VERSION = "agreement_natural_log_tau2_direct_v1"
EXPECTED_RESULTS_STATUS = "provisional"
SYNTHETIC_STUDY_CSV = """study_id,bias,s2,n_pairs,n_participants,c,is_icu,is_arf,is_lft
synthetic_a,-1.0,4.0,20,20,1,0,0,1
synthetic_b,0.5,9.0,24,24,1,1,0,0
synthetic_c,1.5,16.0,30,30,1,0,1,0
"""
INVALID_STUDY_CSV = """study_id,bias
invalid_a,-1.0
"""


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

    revision_notice = page.locator("#revision-notice")
    expect(revision_notice).to_be_visible()
    expect(revision_notice).to_contain_text("agreement-model equations were corrected")
    expect(revision_notice).to_contain_text("provisional pending independent biostatistical review")
    expect(revision_notice).to_contain_text("downstream manuscript numbers remain frozen")
    expect(revision_notice).to_contain_text("Research use only—not for clinical decision-making")
    page.get_by_text("Calculation complete.").wait_for(timeout=180_000)

    assert page.locator("body").get_attribute("data-agreement-method-version") == (
        EXPECTED_AGREEMENT_METHOD_VERSION
    )
    assert page.locator("body").get_attribute("data-results-status") == EXPECTED_RESULTS_STATUS
    assert page.locator("body").get_attribute("data-params-source") == "payload_params"
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


def test_static_app_uploaded_studies_use_current_provisional_method(page, web_server: str) -> None:
    page.goto(web_server, wait_until="domcontentloaded")
    page.get_by_text("Calculation complete.").wait_for(timeout=180_000)
    default_method = page.locator("body").get_attribute("data-agreement-method-version")
    default_status = page.locator("body").get_attribute("data-results-status")

    page.locator("details.panel > summary").click()
    page.locator("#study-file").set_input_files(
        {
            "name": "synthetic_conway.csv",
            "mimeType": "text/csv",
            "buffer": SYNTHETIC_STUDY_CSV.encode(),
        }
    )
    # One synthetic draw is enough to exercise the uploaded-study Pyodide path;
    # native contract tests carry the higher-draw numerical checks.
    page.locator("#n-boot").evaluate(
        "element => { element.min = '1'; element.step = '1'; element.value = '1'; }"
    )
    page.locator("#n-param-draws").evaluate(
        "element => { element.step = '1'; element.value = '1'; }"
    )
    calculate = page.locator("#calculate")
    calculate.click()
    page.wait_for_function(
        """
        () => document.body.dataset.paramsSource === "computed" ||
          !document.querySelector("#error").hidden
        """,
        timeout=180_000,
    )
    assert page.locator("#error").inner_text() == ""
    expect(page.locator("body")).to_have_attribute("data-params-source", "computed")
    expect(page.locator("#status")).to_have_text("Calculation complete.")

    assert page.locator("body").get_attribute("data-agreement-method-version") == default_method
    assert page.locator("body").get_attribute("data-results-status") == default_status
    assert default_method == EXPECTED_AGREEMENT_METHOD_VERSION
    assert default_status == EXPECTED_RESULTS_STATUS


def test_static_app_failed_recalculation_clears_previous_result(page, web_server: str) -> None:
    page.goto(web_server, wait_until="domcontentloaded")
    page.get_by_text("Calculation complete.").wait_for(timeout=180_000)

    assert page.locator("#metric-interval").inner_text() != "-"
    assert page.locator("#posterior-chart .main-svg").count() >= 1
    assert page.locator("body").get_attribute("data-params-source") == "payload_params"

    page.locator("details.panel > summary").click()
    page.locator("#study-file").set_input_files(
        {
            "name": "invalid_studies.csv",
            "mimeType": "text/csv",
            "buffer": INVALID_STUDY_CSV.encode(),
        }
    )
    page.locator("#n-boot").evaluate(
        "element => { element.min = '1'; element.step = '1'; element.value = '1'; }"
    )
    page.locator("#n-param-draws").evaluate(
        "element => { element.step = '1'; element.value = '1'; }"
    )
    page.locator("#calculate").click()

    expect(page.locator("#status")).to_have_text("Calculation failed.", timeout=180_000)
    expect(page.locator("#error")).to_be_visible()
    expect(page.locator("#error")).not_to_be_empty()
    expect(page.locator("#metrics")).to_be_hidden()
    expect(page.locator("#metric-interval")).to_have_text("-")
    expect(page.locator("#metric-probability")).to_have_text("-")
    expect(page.locator("#metric-decision")).to_have_text("-")
    expect(page.locator("#posterior-chart")).to_be_hidden()
    expect(page.locator("#posterior-chart .main-svg")).to_have_count(0)
    expect(page.locator("#chart-caption")).to_be_empty()
    expect(page.locator("#decision-text")).to_have_text(
        "Run a calculation to show posterior threshold mass."
    )
    assert page.locator("body").get_attribute("data-agreement-method-version") is None
    assert page.locator("body").get_attribute("data-results-status") is None
    assert page.locator("body").get_attribute("data-params-source") is None


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
