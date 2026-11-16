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
    page.fill("#tcco2", "999")
    expect(page.locator("#results")).to_be_hidden()
    page.click("#calculate")
    expect(page.locator("#error")).to_contain_text("offline")
    expect(page.locator("#results")).to_be_hidden()


def test_local_spreadsheet_equality_and_no_prior_fetch(page, web_server, tmp_path):
    import pandas as pd

    traffic = []
    page.context.on("request", lambda request: traffic.append((request.method, request.url)))
    page.goto(web_server, wait_until="domcontentloaded")
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
    page.fill("#threshold", "45.01")
    expect(page.locator("#results")).to_be_hidden()
    page.click("#calculate")
    expect(page.locator("#probability")).to_have_text("0.0000")
    assets = {url.split("/")[-1] for _, url in traffic if "/assets/data/" in url}
    assert assets == {"main.npz"}
    assert all(method == "GET" for method, _ in traffic)
