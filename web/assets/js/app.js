const $ = (id) => document.getElementById(id);
const groups = {
  all: ["main", "Overall · mixed-age studies, including children and neonates"],
  icu: ["icu", "ICU · comparable intensive care settings"],
  ed_inp: [
    "arf",
    "Acute respiratory failure · ED and inpatient case mix can differ",
  ],
  pft: ["lft", "Lung function testing · includes respiratory inpatients"],
};
let fits,
  worker,
  revision = 0,
  currentPlot;
const signed = (value) => (value >= 0 ? "+" : "−") + Math.abs(value).toFixed(1);
const priorMode = () => $("inverse").checked;

function invalidate() {
  revision++;
  $("results").hidden = true;
  currentPlot = null;
  $("offline-help").hidden = true;
  $("status").textContent = "";
  $("error").textContent = "";
}
function update() {
  invalidate();
  const [group, description] = groups[$("subgroup").value];
  $("source").textContent = description;
  $("inputs").hidden = !priorMode();
  if (!priorMode() && fits) render(fits[group].quantiles);
}
function render(q, result) {
  const inverse = Boolean(result);
  const fmt = inverse ? (v) => v.toFixed(1) : signed;
  $("result-title").textContent = inverse
    ? "Conditional arterial CO₂"
    : "Arterial minus transcutaneous CO₂";
  $("estimate").textContent = fmt(q[1]) + " mmHg";
  $("bounds").textContent = fmt(q[0]) + " to " + fmt(q[2]) + " mmHg";
  $("prob-card").hidden = !inverse;
  if (inverse) {
    $("prob-label").textContent =
      "P(PaCO₂ ≥ " + $("threshold").value + " mmHg)";
    $("probability").textContent = result.probability.toFixed(4);
  }
  $("results").hidden = false;
  currentPlot = { q, result };
  plot();
}
function plot() {
  if (!currentPlot) return;
  const { q, result } = currentPlot;
  const plotWidth = Math.max(260, $("plot").clientWidth);
  const x0 = result ? result.support[0] : -20;
  const x1 = result ? result.support.at(-1) : 20;
  const scale = (x) =>
    x0 === x1 ? plotWidth / 2 : 26 + ((plotWidth - 52) * (x - x0)) / (x1 - x0);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 " + plotWidth + " 111");
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    result
      ? "Conditional probability mass across arterial CO₂ values"
      : "Predictive interval, median and zero disagreement line",
  );
  function draw(tag, attrs, caption) {
    const el = document.createElementNS(svg.namespaceURI, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    if (caption !== undefined) el.textContent = caption;
    svg.append(el);
  }
  const line = (x1, y1, x2, y2, color, width) =>
    draw("line", { x1, y1, x2, y2, stroke: color, "stroke-width": width });
  const label = (x, y, s) => draw("text", { x, y, "text-anchor": "middle" }, s);
  line(26, 65, plotWidth - 26, 65, "#8fa6a0", 1);
  if (result) {
    // Aggregate only for screen resolution. All reported statistics use the original support.
    const bins = new Map();
    result.support.forEach((a, i) => {
      const x = Math.round(scale(a) / 4) * 4;
      bins.set(x, (bins.get(x) || 0) + result.mass[i]);
    });
    const max = Math.max(...bins.values());
    bins.forEach((p, x) => {
      const top = 65 - (42 * p) / max;
      line(x, 65, x, top, "#086863", 4);
      if (bins.size <= 7) label(x, top - 6, (100 * p).toFixed(1) + "%");
    });
    const ticks = x0 === x1 ? [x0] : [x0, x1];
    ticks.forEach((x) => label(scale(x), 85, x));
  } else {
    line(scale(0), 12, scale(0), 66, "#849892", 1);
    line(scale(q[0]), 40, scale(q[2]), 40, "#086863", 5);
    for (const x of [q[0], q[2]])
      line(scale(x), 32, scale(x), 48, "#086863", 2);
    draw("circle", {
      cx: scale(q[1]),
      cy: 40,
      r: 6,
      fill: "#123e40",
      stroke: "white",
      "stroke-width": 1.5,
    });
    for (const x of [-20, -10, 0, 10, 20]) label(scale(x), 84, x);
  }
  label(
    plotWidth / 2,
    103,
    result ? "Arterial CO₂ (mmHg)" : "Paired difference (mmHg)",
  );
  $("plot").replaceChildren(svg);
}
new ResizeObserver(plot).observe($("results"));
async function readUploadAsCsv(file) {
  if (!file) {
    return null;
  }
  const name = file.name.toLowerCase();
  if (name.endsWith(".csv")) {
    return file.text();
  }
  if (name.endsWith(".xlsx")) {
    if (!globalThis.XLSX) {
      await new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src =
          "https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js";
        script.onload = resolve;
        script.onerror = () =>
          reject(
            new Error("Spreadsheet reader unavailable. Upload CSV or retry."),
          );
        document.head.append(script);
      });
    }
    const data = await file.arrayBuffer();
    const workbook = globalThis.XLSX.read(data, { type: "array" });
    const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
    return globalThis.XLSX.utils.sheet_to_csv(firstSheet);
  }
  throw new Error("Uploads must be CSV or XLSX files.");
}
$("controls").addEventListener("input", update);
$("controls").addEventListener("submit", async (e) => {
  e.preventDefault();
  invalidate();
  const id = revision;
  try {
    if (!fits) throw Error("Model summary unavailable. Reload this page.");
    const file = $("prior-file").files[0];
    if (!file)
      throw Error("Choose a fixed prior file for conditional inference.");
    $("status").textContent = "Computing from your local prior…";
    const prior = await readUploadAsCsv(file);
    if (id !== revision) return;
    if (!worker) {
      worker = new Worker("pyodide_worker.js");
      worker.onmessage = ({ data }) => {
        if (data.id !== revision) return;
        $("status").textContent = "";
        if (data.error) {
          $("error").textContent = data.error;
          if (data.error.includes("offline")) {
            $("offline-help").hidden = false;
            $("offline-command").textContent =
              "uv run python scripts/future_setting_prior.py .pytest_tmp/future-setting/" +
              fits[groups[$("subgroup").value][0]].run +
              " prior.csv " +
              $("subgroup").value +
              " " +
              $("tcco2").value +
              " --threshold " +
              $("threshold").value;
          }
        } else render(data.result.quantiles, data.result);
      };
      worker.onerror = () => {
        worker.terminate();
        worker = null;
        if (!priorMode() || !$("status").textContent) return;
        invalidate();
        $("error").textContent =
          "Browser runtime failed to load. Retry the calculation.";
      };
    }
    const target = $("subgroup").value,
      group = groups[target][0];
    worker.postMessage({
      id,
      group,
      target,
      prior,
      hash: fits[group].kernel_sha256,
      tc: $("tcco2").valueAsNumber,
      threshold: $("threshold").valueAsNumber,
    });
  } catch (err) {
    if (id === revision) {
      $("status").textContent = "";
      $("error").textContent = err.message;
    }
  }
});
fits = JSON.parse($("fit-summary").textContent);
document.body.dataset.agreementMethodVersion = fits.main.method;
document.body.dataset.resultsStatus = fits.main.status;
update();
