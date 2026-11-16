"""Exercise the offline file preview with an invented index distribution and known fit."""

import hashlib
import json

import numpy as np
import pytest
from playwright.sync_api import expect
from scipy.stats import norm

from scripts import conditional_lookup
from tcco2_accuracy.workflows.private_preview_html import private_viewer_html


@pytest.mark.parametrize("split_first_day", [False, True])
def test_local_chart_from_receipt_shows_full_curves_and_exact_tails(
    page, tmp_path, monkeypatch, split_first_day
):
    support = np.array([34.0, 44.5, 45, 49, 52, 55, 105.4321])
    counts = np.array([1, 3, 3, 4, 2, 1, 1])
    weights = counts / counts.sum() if split_first_day else counts
    policy = (
        "equal_patient_weight_split_across_eligible_draws_on_first_testing_day"
        if split_first_day
        else "earliest_eligible_measurement_in_earliest_eligible_encounter_per_patient"
    )
    extract = tmp_path / "invented.csv"
    field = "weight" if split_first_day else "count"
    extract.write_text(
        f"paco2_bin,{field}\n"
        + "".join(f"{a},{w}\n" for a, w in zip(support, weights, strict=True))
    )
    receipt = tmp_path / "source.json"
    receipt.write_text(
        json.dumps(
            dict(
                use="approved_private_index_analysis",
                private_destination=str(tmp_path),
                eligible_extract=str(extract),
                eligible_extract_sha256=hashlib.sha256(extract.read_bytes()).hexdigest(),
                input_kind="index_frequencies",
                index_target="all",
                index_policy=policy,
                revisions=dict(
                    cohort_revision="fixture",
                    eligibility_revision="fixture",
                    window_revision="fixture",
                ),
            )
        )
    )

    # A fitted Gaussian with SD 6 makes the expected result independently calculable.
    def test_fit(run, target):
        assert target == "all"
        return (
            np.tile([0, np.log(36), 0, 0, 0], (4, 100, 1)),
            16,
            dict(run_id=run.name, strata={"main": {"posterior_sha256": "a" * 64}}),
        )

    monkeypatch.setattr(conditional_lookup, "load_candidate_draws", test_fit)
    conditional_lookup.private_build(receipt, tmp_path, chart_target="all")
    directory = tmp_path / "private-chart-all"
    viewer = directory / "preview.html"
    chart = directory / "private-chart.json"
    if split_first_day:
        assert json.loads((directory / "receipt.json").read_text())["index_policy"] == policy

    assert "105.4321" not in viewer.read_text()
    traffic = []
    page.on("request", lambda req: traffic.append(req.url))
    page.context.set_offline(True)
    page.goto(viewer.as_uri())
    page.locator("#chart-file").set_input_files(chart)
    joint = counts * norm.pdf(support - 47, scale=6)
    mass = joint / joint.sum()
    lower, upper = support[np.searchsorted(mass.cumsum(), [0.025, 0.975])]
    expect(page.locator("#private-bounds")).to_have_text(f"{lower:g} to {upper:g} mmHg")
    assert ("first testing day" in page.locator("#private-context").inner_text()) == split_first_day
    probabilities = [f"{100 * mass[support >= t].sum():.2f}%" for t in (45, 52)]
    expect(page.locator(".tail-value")).to_have_text(probabilities)
    expect(page.locator(".prediction-band")).to_be_visible()
    expect(page.locator("#private-interpretation")).to_contain_text(
        "across the eligible settings in TriNetX"
    )
    for selector in (".prior-line", ".posterior-line"):
        assert page.locator(selector).get_attribute("d").count("L") > 30
    page.get_by_role("button", name="Highlight ≥52 mmHg").click()
    expect(page.locator(".tail-area")).to_be_visible()
    expect(page.locator(".tail-value")).to_have_text(probabilities)
    expect(page.locator("figcaption")).to_contain_text("30 to 55 mmHg")
    expect(page.locator("figcaption")).to_contain_text("93.33% of the prior")
    expect(page.locator(".threshold-reference")).to_have_text("45")
    page.get_by_role("button", name="Full range").click()
    expect(page.locator("figcaption")).to_contain_text("100.00% of the prior")
    svg = page.locator("svg")
    svg.focus()
    page.keyboard.press("End")
    expect(page.locator("output")).to_contain_text("105.432")
    assert not [url for url in traffic if url.startswith(("http:", "https:", "ws:", "wss:"))]

    # A different private file (even a syntactically valid chart) must clear the previous result.
    other = tmp_path / "other.json"
    other.write_bytes(chart.read_bytes() + b" ")
    page.locator("#chart-file").set_input_files(other)
    expect(page.locator("#chart-status")).to_contain_text("does not match")
    expect(page.locator("#private-result")).to_be_hidden()
    expect(page.locator(".posterior-line")).to_have_count(0)

    held = json.loads(chart.read_bytes())
    held.update(interval=None, tails=[None, held["tails"][1]])
    other.write_text(json.dumps(held))
    viewer.write_text(private_viewer_html(hashlib.sha256(other.read_bytes()).hexdigest()))
    page.reload()
    page.locator("#chart-file").set_input_files(other)
    expect(page.locator("#private-bounds")).to_have_text("Unresolved")
    expect(page.locator(".tail-value")).to_have_text(["Unresolved", probabilities[1]])
    expect(page.locator("#private-interpretation")).to_be_hidden()
    expect(page.locator(".prediction-band")).to_have_count(0)
    expect(page.locator(".posterior-line")).to_be_visible()
    expect(page.get_by_role("button", name="Full range")).to_be_visible()
