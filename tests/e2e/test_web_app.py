import hashlib
import json
from pathlib import Path

import pytest
from playwright.sync_api import expect

from scripts.stage_web_python import stage_web_python
from scripts.visual_qa import _serve_web

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def web_server():
    stage_web_python(ROOT)
    with _serve_web() as url:
        yield url


def test_comparison_without_javascript(browser, web_server):
    ctx = browser.new_context(java_script_enabled=False)
    page = ctx.new_page()
    page.goto(web_server)
    rows = page.locator("tbody tr")
    expect(rows).to_have_count(4)
    for i, group in enumerate(["main", "icu", "arf", "lft"]):
        row = json.loads((ROOT / "models/pages/summary.json").read_text())[group]
        for j, pair in enumerate(row["intervals"]):
            label = f"{pair['lower']:+.1f} to {pair['upper']:+.1f}".replace("-", "−")
            expect(rows.nth(i).locator("td").nth(j)).to_have_text(label)
    ctx.close()


def test_all_settings_against_offline(page, web_server):
    page.goto(web_server)
    page.locator("#advanced > summary").click()
    expect(page.locator("#bounds")).to_contain_text("−11.5")
    page.locator("#inverse").check()
    page.locator("#calculate").click()
    expect(page.locator("#error")).to_contain_text("prior")
    page.locator("#prior-file").set_input_files(ROOT / "tests/fixtures/synthetic_paco2_prior.csv")
    cases = json.loads(
        (ROOT / "tests/fixtures/future_setting_synthetic_reference.json").read_text()
    )
    for r in cases:
        page.select_option("#subgroup", r["target_stratum"])
        page.fill("#tcco2", r["tcco2"])
        page.click("#calculate")
        expect(page.locator("#estimate")).to_have_text(
            f"{float(r['median']):.1f} mmHg", timeout=90000
        )
        expect(page.locator("#bounds")).to_have_text(
            f"{float(r['lower']):.1f} to {float(r['upper']):.1f} mmHg"
        )
        expect(page.locator("#probability")).to_have_text(f"{float(r['probability_ge']):.4f}")
        interpretation = page.locator("#upload-interpretation")
        expect(interpretation).to_contain_text(
            "the population represented by the uploaded PaCO₂ distribution"
        )
        expect(interpretation).to_contain_text(page.locator("#bounds").inner_text())
        expect(interpretation).to_contain_text("given an observed TcCO₂ of " + r["tcco2"] + " mmHg")
    page.fill("#tcco2", "999")
    expect(interpretation).to_be_empty()
    expect(page.locator("#results")).to_be_hidden()
    page.click("#calculate")
    expect(page.locator("#error")).to_contain_text("offline")
    expect(page.locator("#results")).to_be_hidden()


def test_local_spreadsheet_equality_and_no_prior_fetch(page, web_server, tmp_path):
    import pandas as pd

    traffic = []
    page.context.on("request", lambda request: traffic.append((request.method, request.url)))
    page.goto(web_server, wait_until="domcontentloaded")
    page.locator("#advanced > summary").click()
    expect(page.locator("#bounds")).to_be_visible()
    assert not any("pyodide.js" in url or ".npz" in url for _, url in traffic)
    page.locator("#inverse").focus()
    page.keyboard.press("Space")
    expect(page.locator("#prior-file")).to_be_visible()
    sheet = tmp_path / "invented.xlsx"
    pd.DataFrame(dict(group=["all", "icu", "ed_inp", "pft"], paco2_bin=45, weight=1)).to_excel(
        sheet, index=False
    )
    page.locator("#prior-file").set_input_files(sheet)
    page.click("#calculate")
    expect(page.locator("#probability")).to_have_text("1.0000", timeout=90000)
    expect(page.locator("#plot .tail-value")).to_have_text(["100.00%", "0.00%"])
    expect(page.locator("#plot .posterior-bar")).to_have_count(1)
    assert page.locator("#plot .likelihood-line").get_attribute("d").count("L") >= 20
    page.locator("#plot").get_by_role("button", name="Highlight ≥52 mmHg").click()
    expect(page.locator("#plot figcaption")).to_contain_text("52 mmHg is outside")
    page.fill("#tcco2", "47.25")
    page.fill("#threshold", "45.01")
    expect(page.locator("#results")).to_be_hidden()
    page.click("#calculate")
    expect(page.locator("#probability")).to_have_text("0.0000")
    # Both uploaded pressures fall between adjacent fitted likelihood grid points.
    narrow = tmp_path / "narrow.csv"
    pd.DataFrame(
        [
            (group, pressure, 0.5)
            for group in ("all", "icu", "ed_inp", "pft")
            for pressure in (44.1, 44.2)
        ],
        columns=["group", "paco2_bin", "weight"],
    ).to_csv(narrow, index=False)
    page.locator("#prior-file").set_input_files(narrow)
    page.fill("#tcco2", "47")
    page.click("#calculate")
    expect(page.locator("#bounds")).to_have_text("44.1 to 44.2 mmHg")
    path = page.locator("#plot .likelihood-line")
    expect(path).to_be_visible()
    axis = page.locator("#plot .axis-line").first
    geometry = "(node) => [node.getBBox().x, node.getBBox().width]"
    assert path.evaluate(geometry) == pytest.approx(axis.evaluate(geometry), abs=0.02)
    assets = {url.split("/")[-1] for _, url in traffic if "/assets/data/" in url}
    assert assets == {"main.npz"}
    assert all(method == "GET" for method, _ in traffic)


def test_conditional_lookup_first_and_no_rounding(page, web_server, tmp_path):

    sent = []
    page.on("request", lambda req: sent.append(req.url))
    page.goto(web_server)
    expect(page.locator("#lookup-setting")).to_have_value("all")
    expect(page.locator("#lookup-status")).to_contain_text("TriNetX")
    data = json.loads((ROOT / "models/pages/synthetic-lookup.json").read_text())
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(data))
    page.locator("#lookup-file").set_input_files(path)
    for setting in ("all", "icu", "ed_inp", "pft"):
        page.select_option("#lookup-setting", setting)
        page.fill("#lookup-tc", "45")
        row = data["groups"][setting]["rows"][25]
        if row["interval"]:
            expect(page.locator("#lookup-bounds")).to_have_text(
                f"{row['interval'][0]:g} to {row['interval'][2]:g} mmHg"
            )
        else:
            expect(page.locator("#lookup-held")).to_be_visible()
        expect(page.locator("#lookup-status")).to_be_empty()
        expect(page.locator("#conditional-budget")).to_be_visible()
    before = list(sent)
    page.fill("#lookup-tc", "45.5")
    expect(page.locator("#lookup-values")).to_be_hidden()
    expect(page.locator("#lookup-status")).to_contain_text("45 and 46")
    page.fill("#lookup-tc", "101")
    expect(page.locator("#lookup-status")).to_contain_text("20 to 100")
    assert sent == before


def test_staged_demo_all_readings_and_offline_references(page, web_server):
    page.goto(web_server)
    page.locator("#lookup-demo").click()
    expect(page.locator("#lookup-interpretation")).to_contain_text("In this example")
    display = page.evaluate("""() => {
      const out = {};
      const select = document.querySelector('#lookup-setting');
      const tc = document.querySelector('#lookup-tc');
      for (const option of select.options) {
        select.value = option.value; select.dispatchEvent(new Event('change'));
        out[option.value] = [];
        for (let t = 20; t <= 100; t++) {
          tc.value = t; tc.dispatchEvent(new Event('input'));
          const visible = !document.querySelector('#lookup-values').hidden;
          out[option.value].push([
            visible ? document.querySelector('#lookup-bounds').textContent : null,
            visible ? document.querySelector('#lookup-estimates').textContent : null,
            [...document.querySelectorAll('#lookup-chart .tail-value')].map(x=>x.textContent),
            [...document.querySelectorAll('#conditional-budget dd')].map(x=>x.textContent)]);
        }
      }
      return out;
    }""")
    payload = json.loads((ROOT / "models/pages/synthetic-lookup.json").read_text())
    for target, g in payload["groups"].items():
        for row, (bounds, metrics, tails, parts) in zip(g["rows"], display[target], strict=True):
            q = row["interval"]
            assert bounds == (f"{q[0]:g} to {q[2]:g} mmHg" if q else None)
            if q:
                assert f"Median {q[1]:g} mmHg" in metrics
            assert tails == [
                f"{100 * p:.2f}%" if p is not None else "Unresolved" for p in row["tails"]
            ]
            assert [float(p.split()[0]) for p in parts] == [
                round(x, 2) for x in row["variance"] or []
            ]
    page.locator("#lookup-reset").click()
    expect(page.locator("#lookup-values")).to_be_hidden()


def test_distribution_inspection_and_summary_compatibility(page, web_server, tmp_path):
    page.goto(web_server)
    chart = page.locator("#lookup-chart")
    expect(chart.get_by_role("heading")).to_have_text("95% Prediction Interval for PaCO₂ − TcCO₂")
    curve = chart.locator(".density-line").get_attribute("d")
    page.fill("#lookup-tc", "47")
    assert chart.locator(".density-line").get_attribute("d") == curve
    page.select_option("#lookup-setting", "icu")
    assert chart.locator(".density-line").get_attribute("d") != curve
    page.select_option("#lookup-setting", "all")
    page.click("#lookup-demo")
    expect(chart.locator(".tail-value")).to_have_text(["71.61%", "13.43%"])
    expect(page.locator("#lookup-bounds")).to_have_text("38.75 to 56.25 mmHg")
    expect(page.locator("#lookup-values small")).to_have_text("95% Prediction Interval for PaCO₂")
    interpretation = page.locator("#lookup-interpretation")
    expect(interpretation.locator("p").first).to_have_text(
        "In this example, given an observed TcCO₂ of 47 mmHg, the 95% prediction interval "
        "for a PaCO₂ measured at the same time is 38.75 to 56.25 mmHg."
    )
    original = interpretation.inner_text()
    expect(chart.locator(".range-label")).to_have_text("95% Prediction Interval")
    assert "Synthetic" not in chart.inner_text()
    band = chart.locator(".prediction-band")

    def check_band():
        # Use the two pressure-axis ticks to project the *reported* limits independently.
        band_edges = band.evaluate("""rect => {
          const svg = rect.ownerSVGElement;
          const [a,b] = [...svg.querySelectorAll("text")].filter(
            t => !t.classList.length && t.getAttribute("text-anchor") === "middle" &&
              t.y.baseVal[0].value > svg.querySelector(".likelihood-line").getBBox().y &&
              Number.isFinite(Number(t.textContent)));
          const sx = v => +a.getAttribute("x") + (v-Number(a.textContent)) *
            (+b.getAttribute("x")-a.getAttribute("x"))/(Number(b.textContent)-Number(a.textContent));
          const box = rect.getBBox();
          return [box.x - sx(38.75), box.x + box.width - sx(56.25),
            box.y + box.height < svg.querySelector(".likelihood-line").getBBox().y];
        }""")
        assert max(map(abs, band_edges[:2])) < 0.01
        assert band_edges[2]
        expect(chart.locator(".endpoint-label")).to_have_text(["38.75", "56.25"])
        return band.bounding_box()

    initial_band = check_band()
    before = chart.locator(".likelihood-line").get_attribute("d")
    area = chart.locator(".tail-area").get_attribute("d")
    chart.get_by_role("button", name="Highlight ≥52 mmHg").click()
    assert chart.locator(".likelihood-line").get_attribute("d") == before
    assert chart.locator(".tail-area").get_attribute("d") != area
    assert band.bounding_box() == initial_band
    assert interpretation.inner_text() == original
    expect(chart).to_contain_text("86.57% below 52")
    assert chart.locator(".posterior-line").get_attribute("d").count("L") > 60
    assert chart.locator(".likelihood-line").get_attribute("d").count("L") > 35
    expect(chart.locator(".tail-area")).to_be_visible()
    chart.get_by_role("button", name="Full range").click()
    check_band()
    assert chart.locator(".posterior-line").get_attribute("d").count("L") == 280
    expect(chart.locator(".tail-value")).to_have_text(["71.61%", "13.43%"])
    expect(chart.locator(".median-marker")).to_have_count(1)
    chart.locator("svg").focus()
    page.keyboard.press("ArrowRight")
    expect(chart.locator("output")).to_contain_text("mmHg")
    page.fill("#lookup-tc", "47.5")
    expect(interpretation).to_be_empty()
    expect(band).to_have_count(0)
    expect(chart.locator(".tail-value")).to_have_count(0)
    page.fill("#lookup-tc", "47")
    page.click("#lookup-reset")
    expect(chart.locator(".density-line")).to_be_visible()
    expect(chart.locator(".tail-value")).to_have_count(0)
    # A local v1 lookup has a single inclusive tail and no recoverable shape.
    doc = json.loads((ROOT / "models/pages/synthetic-lookup.json").read_text())
    doc.update(source="trinetx", version="conditional_lookup_v1")
    for g in doc["groups"].values():
        g.pop("curve")
        g.pop("support")
        g.pop("prior")
        g["rows"] = [
            dict(
                interval=[*r["interval"], r["tails"][0]]
                if r["interval"] and r["tails"][0] is not None
                else None,
                variance=r["variance"],
            )
            for r in g["rows"]
        ]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(doc))
    page.locator("#lookup-file").set_input_files(path)
    expect(chart.locator(".tail-value")).to_have_text(["71.61%", "Unavailable"])
    expect(chart.locator(".posterior-bar")).to_have_count(0)
    expect(chart.locator(".median-marker")).to_have_count(1)
    expect(chart).to_contain_text("Summary-only lookup")
    expect(interpretation).to_contain_text("across the eligible settings in TriNetX")
    expect(page.locator("#lookup-status")).to_contain_text("Local candidate lookup")
    # The release path accepts only the hash pinned in the page. Exercise it with
    # illustrative summaries; no restricted support or weights enter this test.
    doc["groups"]["all"]["rows"][28]["interval"] = None
    payload = json.dumps(doc).encode()

    def with_release(route):
        response = route.fetch()
        original_html = response.text()
        marker = 'id="lookup-hashes">{'
        html = original_html.replace(
            marker,
            marker + '"conditional-lookup.json":"' + hashlib.sha256(payload).hexdigest() + '",',
        )
        route.fulfill(response=response, body=html)

    page.route(web_server + "/", with_release)
    page.route(
        "**/assets/data/conditional-lookup.json",
        lambda route: route.fulfill(body=payload, content_type="application/json"),
    )
    page.goto(web_server)
    page.locator("#lookup-released").click()
    for group, setting in [
        ("all", "across the eligible settings"),
        ("icu", "in intensive care"),
        ("pft", "in ambulatory or pulmonary function testing settings"),
        ("ed_inp", "in the emergency department or during an inpatient stay"),
    ]:
        page.select_option("#lookup-setting", group)
        expect(interpretation).to_contain_text(
            f"undergo arterial blood gas testing {setting} in TriNetX"
        )
        expect(interpretation).to_contain_text(page.locator("#lookup-bounds").inner_text())
    page.select_option("#lookup-setting", "all")
    page.fill("#lookup-tc", "48")
    expect(page.locator("#lookup-held")).to_be_visible()
    expect(interpretation).to_be_empty()
    expect(band).to_have_count(0)
