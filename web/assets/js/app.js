import { showDistribution } from "./distribution.js";
import {
  agreementIntervalTitle,
  arterialIntervalTitle,
  showInterpretation,
} from "./prediction.js";
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
  revision = 0;
const signed = (value) => (value >= 0 ? "+" : "−") + Math.abs(value).toFixed(1);
const priorMode = () => $("inverse").checked;

function invalidate() {
  revision++;
  $("results").hidden = true;
  showInterpretation($("upload-interpretation"), null);
  showDistribution($("plot"), null);
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
  const fmt = inverse
    ? (v) =>
        v.toLocaleString("en-US", {
          minimumFractionDigits: 1,
          maximumFractionDigits: 12,
        })
    : signed;
  $("interval-title").textContent = inverse
    ? arterialIntervalTitle
    : agreementIntervalTitle;
  $("result-title").textContent = inverse
    ? "Conditional arterial CO₂"
    : "Arterial minus transcutaneous CO₂";
  $("estimate").textContent = q ? fmt(q[1]) + " mmHg" : "Unresolved";
  $("bounds").textContent = q
    ? fmt(q[0]) + " to " + fmt(q[2]) + " mmHg"
    : "Unresolved";
  $("prob-card").hidden = !inverse;
  if (inverse) {
    if (q)
      showInterpretation($("upload-interpretation"), {
        source: "upload",
        reading: $("tcco2").value,
        bounds: $("bounds").textContent,
      });
    $("prob-label").textContent =
      "P(PaCO₂ ≥ " + $("threshold").value + " mmHg)";
    $("probability").textContent =
      result.probability === null
        ? "Unresolved"
        : result.probability.toFixed(4);
  }
  $("results").hidden = false;
  const fit = fits[groups[$("subgroup").value][0]];
  showDistribution(
    $("plot"),
    inverse
      ? {
          kind: "posterior",
          title: "PaCO₂ given TcCO₂ " + $("tcco2").value + " mmHg",
          formatEndpoint: fmt,
          q,
          x: result.support,
          ...result,
          curve: fit.curve,
          tc: $("tcco2").valueAsNumber,
          active: $("threshold").valueAsNumber,
          customProbability: result.probability,
        }
      : {
          kind: "agreement",
          title: agreementIntervalTitle,
          subtitle: $("subgroup").selectedOptions[0].text,
          q,
          ...fit.curve,
        },
  );
}
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
      worker = new Worker($("upload-calculator").dataset.worker);
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
