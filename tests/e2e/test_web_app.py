import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pytest
from playwright.sync_api import expect

from scripts.stage_web_python import stage_web_python
from scripts.visual_qa import _serve_web
from tcco2_accuracy.core.conditional_reporting import display_range
from tcco2_accuracy.pages_prediction import GROUPS, conditional

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def web_server(tmp_path_factory):
    stage_web_python(ROOT)
    directory = tmp_path_factory.mktemp("pages-project")
    (directory / "tcco2-accuracy").symlink_to(ROOT / "web", target_is_directory=True)
    with _serve_web(directory) as url:
        yield url + "/tcco2-accuracy/"


def test_comparison_without_javascript(browser, web_server):
    ctx = browser.new_context(java_script_enabled=False)
    page = ctx.new_page()
    page.goto(web_server + "research.html")
    expect(page.locator("#results-across-settings img")).to_have_attribute(
        "src", re.compile(r"/works-in-progress/paired-prediction\.png$")
    )
    expect(page.get_by_role("link", name="Cohort and example predictions")).to_be_visible()
    expect(page.locator("#lookup-tools")).not_to_have_attribute("open", "")
    page.locator("#comparison > summary").click()
    rows = page.locator("tbody tr")
    expect(rows).to_have_count(4)
    for i, group in enumerate(["main", "icu", "arf", "lft"]):
        row = json.loads((ROOT / "models/pages/summary.json").read_text())[group]
        for j, pair in enumerate(row["intervals"]):
            label = f"{pair['lower']:+.1f} to {pair['upper']:+.1f}".replace("-", "−")
            expect(rows.nth(i).locator("td").nth(j)).to_have_text(label)
    ctx.close()


@pytest.mark.parametrize("width", [1280, 390])
def test_public_trinetx_full_grid_and_release_boundary(page, web_server, width):
    """Release acceptance: real summaries, every reading, no private curves or input traffic."""
    data = json.loads((ROOT / "models/pages/conditional-lookup.json").read_text())
    assert data["version"] == "conditional_lookup_v3"
    assert data["source"] == "trinetx"
    traffic = []
    page.on("request", lambda req: traffic.append((req.method, req.url)))
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(web_server)
    expect(page.locator("#lookup-tc")).to_have_value("")
    expect(page.locator("#lookup-values")).to_be_hidden()
    expect(page.locator("#lookup-setting")).to_have_value("all")
    expect(page.locator("#lookup-context")).to_contain_text("TriNetX")
    page.fill("#lookup-tc", "47")
    expect(page.locator("#lookup-chart .posterior-line")).to_be_visible()
    expect(page.locator("#lookup-bounds")).to_have_text("35 to 53 mmHg")
    expect(page.locator("#lookup-values .tail-value")).to_have_text(["47%", "5%"])
    expect(page.locator("#lookup-interpretation")).to_contain_text("in TriNetX")
    before = list(traffic)
    display = page.evaluate("""() => {
      const out = {};
      const select = document.querySelector('#lookup-setting');
      const tc = document.querySelector('#lookup-tc');
      for (const option of select.options) {
        select.value = option.value; select.dispatchEvent(new Event('change'));
        out[option.value] = [];
        for (let t = 20; t <= 100; t++) {
          tc.value=t; tc.dispatchEvent(new Event('input'));
          const chart = document.querySelector('#lookup-chart');
          const line = () => chart.querySelector('.posterior-line').getAttribute('d');
          const before = line();
          const results = document.querySelector('#lookup-values');
          const labels = results.textContent;
          const highlight = results.querySelector('[aria-label="Highlight ≥52 mmHg"]');
          highlight.click();
          if (highlight.getAttribute('aria-pressed') !== 'true' ||
              chart.querySelector('.cut-label').textContent !== '≥52') throw Error('Wrong tail');
          if (line() !== before) throw Error('Threshold changed the distribution');
          const svg = chart.querySelector('svg');
          for (const key of ['Home','Enter','ArrowRight','ArrowRight','Enter']) {
            svg.dispatchEvent(new KeyboardEvent('keydown',{key}));
          }
          if (line() === before || labels !== results.textContent) throw Error('Zoom mismatch');
          svg.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape'}));
          if (line() !== before || labels !== results.textContent) throw Error('Reset mismatch');
          out[option.value].push({
            bounds: document.querySelector('#lookup-bounds').textContent,
            estimates: document.querySelector('#lookup-estimates').textContent,
            tails: [...document.querySelectorAll('#lookup-values .tail-value')].map(x=>x.textContent),
            interpretation: document.querySelector('#lookup-interpretation').textContent,
          });
        }
      }
      return out;
    }""")
    for setting, g in data["groups"].items():
        assert set(g) == {"prior_interval", "posterior", "reference_posterior", "rows"}
        for t, row, seen in zip(range(20, 101), g["rows"], display[setting], strict=True):
            assert set(row) == {"interval", "tails", "report", "variance"}
            report = row["report"]
            bounds = f"{display_range(report['quantiles'][0])} to {display_range(report['quantiles'][2])} mmHg"
            assert seen["bounds"] == bounds
            assert seen["estimates"].startswith(
                f"Median {display_range(report['quantiles'][1])} mmHg"
            )
            assert seen["tails"] == [
                display_range(p, percent=True) for p in report["probabilities"]
            ]
            assert bounds in seen["interpretation"] and f"{t} mmHg" in seen["interpretation"]
    page.select_option("#lookup-setting", "all")
    page.fill("#lookup-tc", "47")
    interpretation = page.locator("#lookup-interpretation").inner_text()
    page.get_by_role("button", name="Highlight ≥52 mmHg").click()
    assert page.locator("#lookup-interpretation").inner_text() == interpretation
    expect(
        page.locator(
            "#lookup-chart .posterior-line, #lookup-chart .prior-line, #lookup-chart .likelihood-line"
        )
    ).to_have_count(3)
    expect(page.locator("#conditional-budget")).to_be_hidden()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert traffic == before
    assert all(method == "GET" for method, _ in traffic)
    assert {url.split("/")[-1] for _, url in traffic if "/assets/data/" in url} == {
        "conditional-lookup.json",
        "prediction-plots.json",
    }
    evidence = ROOT / ".pytest_tmp/e2e/trinetx-release"
    evidence.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(evidence / f"public-{width}.png"), full_page=True)
    (evidence / f"grid-{width}.json").write_text(json.dumps(display))
    page.fill("#lookup-tc", "47.5")
    expect(page.locator("#lookup-values")).to_be_hidden()
    expect(page.locator("#lookup-interpretation")).to_be_empty()
    expect(page.locator("#neighbors button")).to_have_text(["47 mmHg", "48 mmHg"])
    page.get_by_role("button", name="48 mmHg", exact=True).click()
    expect(page.locator("#lookup-tc")).to_have_value("48")

    def corrupt(route):
        route.fulfill(body="{}", content_type="application/json")

    page.route("**/assets/data/prediction-plots.json", corrupt)
    page.reload()
    page.fill("#lookup-tc", "47")
    expect(page.locator("#lookup-status")).to_contain_text("chart could not be verified")
    expect(page.locator("#lookup-bounds")).to_have_text("35 to 53 mmHg")
    expect(page.locator("#lookup-estimates")).to_have_text("Median 44 mmHg")
    expect(page.locator("#lookup-values .tail-value")).to_have_text(["47%", "5%"])
    expect(page.locator("#lookup-chart svg")).to_have_count(0)
    expect(page.get_by_role("button", name="Highlight ≥52 mmHg")).to_be_disabled()
    page.unroute("**/assets/data/prediction-plots.json", corrupt)
    page.route(
        "**/assets/data/conditional-lookup.json",
        lambda route: route.fulfill(body="{}", content_type="application/json"),
    )
    page.reload()
    page.fill("#lookup-tc", "47")
    expect(page.locator("#lookup-status")).to_contain_text("could not be verified")
    expect(page.locator("#lookup-values")).to_be_hidden()
    expect(page.locator("#lookup-interpretation")).to_be_empty()


def test_all_settings_against_offline(page, web_server):
    page.goto(web_server + "research.html")
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
        with np.load(ROOT / "models/pages" / (GROUPS[r["target_stratum"]] + ".npz")) as kernel:
            reference = conditional(
                kernel,
                r["target_stratum"],
                (ROOT / "tests/fixtures/synthetic_paco2_prior.csv").read_text(),
                float(r["tcco2"]),
                45,
            )
        page.select_option("#subgroup", r["target_stratum"])
        page.fill("#tcco2", r["tcco2"])
        page.click("#calculate")
        expect(page.locator("#estimate")).to_have_text(
            f"{float(r['median']):g} mmHg", timeout=90000
        )
        expect(page.locator("#bounds")).to_have_text(
            f"{float(r['lower']):g} to {float(r['upper']):g} mmHg"
        )
        expect(page.locator("#probability")).to_have_text(
            display_range(reference["report"]["probabilities"][0], percent=True)
        )
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


def test_current_page_rejects_matching_stale_manifest_and_python(page, web_server):
    # The old ZIP is valid Python and its manifest agrees with its bytes.
    assets = ROOT / "web/assets/py"
    stale = io.BytesIO((assets / "pages.zip").read_bytes())
    with zipfile.ZipFile(stale, "a") as archive:
        archive.comment = b"previous deployment"
    stale_bytes = stale.getvalue()
    manifest = json.loads((assets / "manifest.json").read_text())
    manifest["bundle_sha256"] = hashlib.sha256(stale_bytes).hexdigest()
    page.context.route("**/assets/py/manifest.json*", lambda r: r.fulfill(json=manifest))
    page.context.route("**/assets/py/pages.zip*", lambda r: r.fulfill(body=stale_bytes))
    page.goto(web_server + "research.html")
    page.locator("#advanced > summary").click()
    page.locator("#inverse").check()
    page.locator("#prior-file").set_input_files(ROOT / "tests/fixtures/synthetic_paco2_prior.csv")
    with page.expect_worker() as created:
        page.click("#calculate")
    page.wait_for_function(
        "document.querySelector('#error').textContent || !document.querySelector('#results').hidden",
        timeout=90_000,
    )
    expect(page.locator("#error")).to_contain_text("checksum mismatch")
    expect(page.locator("#results")).to_be_hidden()
    # A stale worker at the previous deployment's fixed URL cannot bypass the page pin.
    worker_url = urlsplit(created.value.url)
    worker_hash = hashlib.sha256((ROOT / "web/pyodide_worker.js").read_bytes()).hexdigest()
    assert worker_url.path.endswith(f"/assets/py/worker-{worker_hash}.js")
    assert (assets / f"worker-{worker_hash}.js").read_bytes() == (
        ROOT / "web/pyodide_worker.js"
    ).read_bytes()

    # A stale upload entrypoint must not be able to start its former unpinned worker.
    page.route(
        "**/assets/js/app.js",
        lambda r: r.fulfill(
            content_type="text/javascript", body="document.body.dataset.stale = 'yes'"
        ),
    )
    page.reload()
    assert page.locator("body").get_attribute("data-stale") is None


def test_local_spreadsheet_equality_and_no_prior_fetch(page, web_server, tmp_path):
    import pandas as pd

    traffic = []
    page.context.on("request", lambda request: traffic.append((request.method, request.url)))
    page.goto(web_server + "research.html", wait_until="domcontentloaded")
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
    expect(page.locator("#probability")).to_have_text("100%", timeout=90000)
    expect(page.locator("#plot .tail-value")).to_have_text(["100%", "0%"])
    expect(page.locator("#plot .posterior-bar")).to_have_count(1)
    assert page.locator("#plot .likelihood-line").get_attribute("d").count("L") >= 20
    page.locator("#plot").get_by_role("button", name="Highlight ≥52 mmHg").click()
    expect(page.locator("#plot figcaption")).to_contain_text("45 to 55 mmHg")
    expect(page.locator("#plot .cut-label")).to_have_text("≥52")
    expect(page.locator("#plot .threshold-reference")).to_have_text("45")
    page.fill("#tcco2", "47.25")
    page.fill("#threshold", "45.01")
    expect(page.locator("#results")).to_be_hidden()
    page.click("#calculate")
    expect(page.locator("#probability")).to_have_text("0%")
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
    expect(page.locator("#bounds")).to_have_text("44 to 44 mmHg")
    path = page.locator("#plot .likelihood-line")
    expect(path).to_be_visible()
    axis = page.locator("#plot .axis-line").first
    geometry = "(node) => [node.getBBox().x, node.getBBox().width]"
    assert path.evaluate(geometry) == pytest.approx(axis.evaluate(geometry), abs=0.02)
    assets = {url.split("/")[-1] for _, url in traffic if "/assets/data/" in url}
    assert assets == {"main.npz", "conditional-lookup.json"}
    assert all(method == "GET" for method, _ in traffic)


def test_conditional_lookup_first_and_no_rounding(page, web_server, tmp_path):

    sent = []
    page.on("request", lambda req: sent.append(req.url))
    page.goto(web_server + "research.html")
    page.locator("#lookup-tools > summary").click()
    expect(page.locator("#lookup-setting")).to_have_value("all")
    expect(page.locator("#lookup-interpretation")).to_contain_text("in TriNetX")
    data = json.loads((ROOT / "models/pages/synthetic-lookup.json").read_text())
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(data))
    page.locator("#lookup-file").set_input_files(path)
    for setting in ("all", "icu", "ed_inp", "pft"):
        page.select_option("#lookup-setting", setting)
        page.fill("#lookup-tc", "45")
        row = data["groups"][setting]["rows"][25]
        if all(row["interval"][j] is not None for j in (0, 2)):
            expect(page.locator("#lookup-bounds")).to_have_text(
                f"{display_range(row['report']['quantiles'][0])} to {display_range(row['report']['quantiles'][2])} mmHg"
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
    page.goto(web_server + "research.html")
    page.locator("#lookup-tools > summary").click()
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
            lo, median, hi = [display_range(v) for v in row["report"]["quantiles"]]
            assert bounds == (
                f"{lo} to {hi} mmHg" if q[0] is not None and q[2] is not None else None
            )
            if bounds and q[1] is not None:
                assert f"Median {median} mmHg" in metrics
            assert tails == [
                display_range(b, percent=True) if b else "Unresolved"
                for b in row["report"]["probabilities"]
            ]
            assert [float(p.split()[0]) for p in parts] == [
                round(x, 2) for x in row["variance"] or []
            ]
    page.locator("#lookup-reset").click()
    expect(page.locator("#lookup-values")).to_be_hidden()


def test_distribution_inspection_and_summary_compatibility(page, web_server, tmp_path):
    page.goto(web_server + "research.html")
    page.locator("#lookup-tools > summary").click()
    expect(page.locator("#lookup-values")).to_be_visible()
    page.click("#lookup-reset")
    chart = page.locator("#lookup-chart")
    expect(chart.get_by_role("heading")).to_have_text("95% Prediction Interval for PaCO₂ − TcCO₂")
    curve = chart.locator(".density-line").get_attribute("d")
    page.fill("#lookup-tc", "47")
    assert chart.locator(".density-line").get_attribute("d") == curve
    page.select_option("#lookup-setting", "icu")
    assert chart.locator(".density-line").get_attribute("d") != curve
    page.select_option("#lookup-setting", "all")
    page.click("#lookup-demo")
    expect(chart.locator(".tail-value")).to_have_text(["72%", "13%"])
    expect(page.locator("#lookup-bounds")).to_have_text("39 to 56 mmHg")
    expect(page.locator("#lookup-values small")).to_have_text("95% Prediction Interval for PaCO₂")
    interpretation = page.locator("#lookup-interpretation")
    expect(interpretation.locator("p").first).to_have_text(
        "In this example, given an observed TcCO₂ of 47 mmHg, the 95% prediction interval "
        "for a PaCO₂ measured at the same time is 39 to 56 mmHg."
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
        expect(chart.locator(".endpoint-label")).to_have_text(["39", "56"])
        return band.bounding_box()

    initial_band = check_band()
    before = chart.locator(".likelihood-line").get_attribute("d")
    area = chart.locator(".tail-area").get_attribute("d")
    chart.get_by_role("button", name="Highlight ≥52 mmHg").click()
    assert chart.locator(".likelihood-line").get_attribute("d") == before
    assert chart.locator(".tail-area").get_attribute("d") != area
    assert band.bounding_box() == initial_band
    assert interpretation.inner_text() == original
    expect(chart).to_contain_text("87% below 52")
    assert chart.locator(".posterior-line").get_attribute("d").count("L") > 60
    assert chart.locator(".likelihood-line").get_attribute("d").count("L") > 35
    expect(chart.locator(".tail-area")).to_be_visible()
    chart.get_by_role("button", name="Full range").click()
    check_band()
    assert chart.locator(".posterior-line").get_attribute("d").count("L") == 280
    expect(chart.locator(".tail-value")).to_have_text(["72%", "13%"])
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
                if None not in r["interval"] and r["tails"][0] is not None
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
        html = re.sub(
            r'(<script type="application/json" id="lookup-hashes">).*?(</script>)',
            lambda m: (
                m[1]
                + json.dumps({"conditional-lookup.json": hashlib.sha256(payload).hexdigest()})
                + m[2]
            ),
            original_html,
            flags=re.S,
        )
        route.fulfill(response=response, body=html)

    page.route(web_server + "research.html", with_release)
    page.route(
        "**/assets/data/conditional-lookup.json",
        lambda route: route.fulfill(body=payload, content_type="application/json"),
    )
    page.goto(web_server + "research.html")
    page.locator("#lookup-tools > summary").click()
    expect(page.locator("#lookup-values")).to_be_visible()
    expect(page.locator('#lookup-setting option[value="icu"]')).to_have_text(
        "Critical Care Billing"
    )
    for group, setting in [
        ("all", "across the eligible settings"),
        ("icu", "in inpatient encounters with critical-care billing"),
        ("pft", "in ambulatory encounters"),
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


def test_amended_independent_holds_and_rounding(page, web_server, tmp_path):
    doc = json.loads((ROOT / "models/pages/synthetic-lookup.json").read_text())
    rows = doc["groups"]["all"]["rows"]
    rows[27]["interval"][1] = None
    rows[27]["report"]["quantiles"][1] = None
    rows[28]["interval"][0] = None
    rows[28]["report"]["quantiles"][0] = None
    rows[28]["tails"][1] = None
    rows[28]["report"]["probabilities"][1] = None
    path = tmp_path / "independent.json"
    path.write_text(json.dumps(doc))
    page.goto(web_server + "research.html")
    page.locator("#lookup-tools > summary").click()
    page.locator("#lookup-file").set_input_files(path)
    expect(page.locator("#lookup-bounds")).to_have_text("39 to 56 mmHg")
    expect(page.locator("#lookup-estimates")).to_contain_text("Median unresolved")
    expect(page.locator("#lookup-chart .median-marker")).to_have_count(0)
    expect(page.locator("#lookup-chart .prediction-band")).to_have_count(1)
    page.fill("#lookup-tc", "48")
    expect(page.locator("#lookup-held")).to_contain_text("Median")
    expect(page.locator("#lookup-interpretation")).to_be_empty()
    expect(page.locator("#lookup-chart .prediction-band")).to_have_count(0)
    expect(page.locator("#lookup-chart .median-marker")).to_have_count(1)
    expect(page.locator("#lookup-chart .marker-label")).to_contain_text("Median")
    expect(page.locator("#lookup-chart .tail-value").nth(1)).to_have_text("Unresolved")
    assert page.locator("#lookup-chart .tail-value").nth(0).inner_text() != "Unresolved"
