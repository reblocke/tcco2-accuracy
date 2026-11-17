"""Exercise range selection without changing the fixed-prior inference."""

from playwright.sync_api import expect

from scripts.stage_web_python import stage_web_python
from scripts.visual_qa import ROOT, _serve_web


def pressure_ticks(svg):
    # The pressure tick labels are 30 viewBox units above its lower edge.
    return svg.evaluate("""s => [...s.querySelectorAll('text')].filter(t =>
        t.y.baseVal[0].value === s.viewBox.baseVal.height - 30
    ).map(t => Number(t.textContent))""")


def test_distribution_zoom_keeps_inference_and_resets(page):
    stage_web_python(ROOT)
    with _serve_web() as url:
        page.goto(url)
        page.locator("#lookup-demo").click()
        chart = page.locator("#lookup-chart")
        svg = chart.locator("svg")
        expect(chart.locator(".tail-value")).to_have_text(["72%", "13%"])
        bounds = page.locator("#lookup-bounds").inner_text()
        explanation = page.locator("#lookup-interpretation").inner_text()
        initial_ticks = pressure_ticks(svg)
        svg.scroll_into_view_if_needed()
        box = svg.bounding_box()
        y = box["y"] + box["height"] * 0.36
        start = box["x"] + box["width"] * 0.48
        end = box["x"] + box["width"] * 0.66
        page.mouse.move(end, y)
        page.mouse.down()
        page.mouse.move(start, y, steps=6)
        page.mouse.up()
        zoomed = pressure_ticks(svg)
        assert min(initial_ticks) < min(zoomed) < max(zoomed) < max(initial_ticks)
        expect(chart.locator("figcaption")).to_contain_text("of the posterior")
        chart.get_by_role("button", name="Highlight ≥52 mmHg").click()
        assert pressure_ticks(svg) == zoomed
        expect(chart.locator(".tail-value")).to_have_text(["72%", "13%"])
        expect(page.locator("#lookup-bounds")).to_have_text(bounds)
        assert page.locator("#lookup-interpretation").inner_text() == explanation
        chart.get_by_role("button", name="Reset zoom").click()
        assert pressure_ticks(svg) == initial_ticks
        chart.get_by_role("button", name="Full range").click()
        svg.focus()
        page.keyboard.press("Home")
        page.keyboard.press("Enter")
        page.keyboard.press("ArrowRight")
        page.keyboard.press("Enter")
        key_ticks = pressure_ticks(svg)
        assert max(key_ticks) < max(initial_ticks)
        expect(chart.locator(".threshold-offscreen")).to_have_count(2)
        svg.dblclick()
        assert pressure_ticks(svg) == initial_ticks


def test_sparse_posterior_axis_includes_thresholds_and_uses_full_weights(page):
    stage_web_python(ROOT)
    with _serve_web() as url:
        page.goto(url)
        page.set_viewport_size({"width": 390, "height": 900})
        for center, limits in [(29, "25 to 55"), (81, "45 to 85")]:
            page.evaluate(
                """async center => {
                const {showDistribution} = await import("./assets/js/distribution.js");
                const host = document.querySelector("#lookup-chart");
                const x = [5, center, 250];
                showDistribution(host, {kind: "posterior", x, tc: 47, q: null,
                    prior: [.25,.5,.25], mass: [.0004,.9992,.0004],
                    curve: {x: [-42,center-47,203], density: [.01,1,.01]},
                    title: "Invented sparse prior", tails: [null,null]});
                }""",
                center,
            )
            chart = page.locator("#lookup-chart")
            expect(chart.locator("figcaption")).to_contain_text(limits + " mmHg")
            expect(chart.locator("figcaption")).to_contain_text("99.92% of the posterior")
            expect(chart.locator("figcaption")).to_contain_text("50.00% of the prior")
            expect(chart.locator(".threshold-reference")).to_have_text("52")
            chart.get_by_role("button", name="Full range").click()
            expect(chart.locator("figcaption")).to_contain_text("100.00% of the prior")
            chart.get_by_role("button", name="Focused range").click()
