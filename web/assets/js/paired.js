import { showDistribution, reportLabel } from "./distribution.js";
import { showInterpretation } from "./prediction.js";

const el = (id) => document.getElementById(id);
const hashes = JSON.parse(el("lookup-hashes").textContent);
const axis = Array.from({ length: 4001 }, (_, i) => i / 16);
let lookup,
  plots,
  failure = "";

async function verified(name) {
  const response = await fetch("assets/data/" + name);
  if (!response.ok) throw Error();
  const bytes = await response.arrayBuffer();
  const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
    .map((v) => v.toString(16).padStart(2, "0"))
    .join("");
  if (hash !== hashes[name]) throw Error();
  return JSON.parse(new TextDecoder().decode(bytes));
}

function render() {
  const input = el("lookup-tc"),
    t = input.valueAsNumber,
    setting = el("lookup-setting").value;
  el("lookup-values").hidden = true;
  el("neighbors").replaceChildren();
  showInterpretation(el("lookup-interpretation"), null);
  el("lookup-status").textContent = failure;
  if (!input.value || !Number.isInteger(t) || t < 20 || t > 100 || !lookup) {
    showDistribution(el("lookup-chart"), null);
    if (!failure)
      el("lookup-status").textContent = !input.value
        ? "Enter a reading from 20 to 100 mmHg."
        : !lookup
          ? "Loading predictions…"
          : "Choose a whole-number reading from 20 to 100 mmHg.";
    if (Number.isFinite(t) && t > 20 && t < 100 && !Number.isInteger(t)) {
      for (const n of [Math.floor(t), Math.ceil(t)]) {
        const b = document.createElement("button");
        b.textContent = n + " mmHg";
        b.onclick = () => {
          input.value = n;
          render();
        };
        el("neighbors").append(b);
      }
    }
    return;
  }
  const row = lookup.groups[setting].rows[t - 20];
  const [lo, median, hi] = row.report.quantiles.map((pair) =>
    reportLabel(pair),
  );
  const bounds = lo + " to " + hi + " mmHg";
  el("lookup-bounds").textContent = bounds;
  el("lookup-estimates").textContent = "Median " + median + " mmHg";
  el("lookup-values").hidden = false;
  showInterpretation(el("lookup-interpretation"), {
    source: "trinetx",
    setting,
    reading: String(t),
    bounds,
    brief: true,
  });
  const model = {
    title: "PaCO₂ given TcCO₂ " + t + " mmHg",
    q: row.interval,
    tails: row.tails,
    report: row.report,
    tc: t,
    endpointLabel: (_, i) => reportLabel(row.report.quantiles[i]),
    kind: "summary",
  };
  if (plots) {
    const group = plots.groups[setting],
      display = group.rows[t - 20];
    Object.assign(model, {
      kind: "posterior",
      x: axis,
      view: display.view,
      curve: group.curve,
      display: {
        x: axis,
        prior_density: group.prior,
        posterior_density: display.density.map((v) => v / 100000),
      },
    });
  }
  showDistribution(el("lookup-chart"), model);
}
el("lookup-tc").addEventListener("input", render);
el("lookup-setting").addEventListener("change", render);
render();
try {
  lookup = await verified("conditional-lookup.json");
  render();
  const chart = await verified("prediction-plots.json");
  if (chart.lookup_sha256 !== hashes["conditional-lookup.json"]) throw Error();
  for (const [setting, g] of Object.entries(chart.groups)) {
    if (g.posterior !== lookup.groups[setting].posterior) throw Error();
    g.prior = g.prior_density.map((v) => v / 100000);
    g.curve = {
      x: g.relative.map((_, i) => g.offset_start + i / 4),
      density: g.relative,
    };
  }
  plots = chart;
} catch {
  failure = lookup
    ? "The chart could not be verified. Showing the verified summary."
    : "The prediction lookup could not be verified. Reload the page.";
}
render();
