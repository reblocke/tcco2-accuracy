const DEFAULTS = {
  nBoot: 1000,
  seed: 202401,
  bootstrapMode: "cluster_plus_withinstudy",
};

const elements = {
  form: document.querySelector("#controls"),
  calculate: document.querySelector("#calculate"),
  status: document.querySelector("#status"),
  error: document.querySelector("#error"),
  metrics: document.querySelector("#metrics"),
  interval: document.querySelector("#metric-interval"),
  thresholdLabel: document.querySelector("#metric-threshold-label"),
  probability: document.querySelector("#metric-probability"),
  decision: document.querySelector("#metric-decision"),
  decisionText: document.querySelector("#decision-text"),
  caption: document.querySelector("#chart-caption"),
  chart: document.querySelector("#posterior-chart"),
};

let worker;
let requestId = 0;
const pending = new Map();

function getWorker() {
  if (worker) {
    return worker;
  }
  worker = new Worker("pyodide_worker.js");
  worker.onmessage = (event) => {
    const { id, type, result, error } = event.data;
    const handlers = pending.get(id);
    if (!handlers) {
      return;
    }
    pending.delete(id);
    if (type === "error") {
      handlers.reject(new Error(error || "Browser runtime failed."));
    } else {
      handlers.resolve(result);
    }
  };
  return worker;
}

function postToWorker(type, payload = {}) {
  const id = ++requestId;
  const activeWorker = getWorker();
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    activeWorker.postMessage({ id, type, payload });
  });
}

async function initialize() {
  try {
    await postToWorker("init");
    elements.status.textContent = "Runtime ready. Default data are loaded from repo assets.";
    await calculate();
  } catch (error) {
    showError(error.message);
    elements.status.textContent = "Runtime failed to load.";
  }
}

async function calculate() {
  setBusy(true);
  hideError();
  try {
    const payload = await buildPayload();
    const result = await postToWorker("compute", payload);
    renderResult(result);
    elements.status.textContent = "Calculation complete.";
  } catch (error) {
    showError(error.message);
    elements.status.textContent = "Calculation failed.";
  } finally {
    setBusy(false);
  }
}

async function buildPayload() {
  const formData = new FormData(elements.form);
  const mode = formData.get("mode");
  const studyCsv = await readUploadAsCsv(document.querySelector("#study-file").files[0]);
  const priorCsv =
    mode === "prior_weighted"
      ? await readUploadAsCsv(document.querySelector("#prior-file").files[0])
      : null;
  const nBoot = Number(document.querySelector("#n-boot").value);
  const seed = Number(document.querySelector("#seed").value);
  const bootstrapMode = document.querySelector("#bootstrap-mode").value;
  const useCanonicalParams =
    !studyCsv &&
    nBoot === DEFAULTS.nBoot &&
    seed === DEFAULTS.seed &&
    bootstrapMode === DEFAULTS.bootstrapMode;

  return {
    tcco2: Number(formData.get("tcco2")),
    subgroup: formData.get("subgroup"),
    threshold: Number(formData.get("threshold")),
    mode,
    interval: Number(formData.get("interval")),
    n_boot: nBoot,
    bootstrap_mode: bootstrapMode,
    n_param_draws: Number(document.querySelector("#n-param-draws").value),
    seed,
    bin_width: Number(document.querySelector("#bin-width").value),
    use_canonical_params: useCanonicalParams,
    study_csv: studyCsv || null,
    prior_bins_csv: priorCsv || null,
    prior_source: priorCsv ? "uploaded_bins" : "default_bins",
  };
}

async function readUploadAsCsv(file) {
  if (!file) {
    return null;
  }
  const name = file.name.toLowerCase();
  if (name.endsWith(".csv")) {
    return file.text();
  }
  if (name.endsWith(".xlsx") || name.endsWith(".xls")) {
    if (!globalThis.XLSX) {
      throw new Error("XLSX upload support is unavailable; upload CSV instead.");
    }
    const data = await file.arrayBuffer();
    const workbook = globalThis.XLSX.read(data, { type: "array" });
    const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
    return globalThis.XLSX.utils.sheet_to_csv(firstSheet);
  }
  throw new Error("Uploads must be CSV or XLSX files.");
}

function renderResult(result) {
  elements.metrics.hidden = false;
  elements.interval.textContent =
    `${result.paco2_median.toFixed(1)} [` +
    `${result.paco2_q_low.toFixed(1)}, ${result.paco2_q_high.toFixed(1)}]`;
  elements.thresholdLabel.textContent = `P(PaCO2 >= ${formatNumber(result.threshold)})`;
  elements.probability.textContent = result.p_ge_threshold.toFixed(3);
  elements.decision.textContent = titleCase(result.decision_label);

  if (result.decision_label === "positive") {
    elements.decisionText.textContent =
      `True positive: ${result.p_true_positive.toFixed(3)} · ` +
      `False positive: ${result.p_false_positive.toFixed(3)}`;
  } else {
    elements.decisionText.textContent =
      `True negative: ${result.p_true_negative.toFixed(3)} · ` +
      `False negative: ${result.p_false_negative.toFixed(3)}`;
  }
  elements.caption.textContent =
    `Posterior mass above threshold: ${(result.p_ge_threshold * 100).toFixed(1)}%.`;
  renderChart(result);
}

function renderChart(result) {
  const colors = result.paco2_bin.map((value) =>
    value >= result.threshold ? "rgba(195, 78, 63, 0.76)" : "rgba(55, 117, 151, 0.76)",
  );
  const traces = [
    {
      type: "bar",
      x: result.paco2_bin,
      y: result.posterior_prob,
      marker: { color: colors },
      name: "Posterior",
    },
  ];
  if (result.prior_prob) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x: result.paco2_bin,
      y: result.prior_prob,
      line: { color: "rgba(82, 91, 95, 0.78)", width: 2 },
      name: "Prior",
    });
  }

  const verticals = [
    [result.threshold, "Threshold", "rgba(195, 78, 63, 0.92)", "dash"],
    [result.paco2_q_low, "PI low", "rgba(23, 32, 38, 0.55)", "dot"],
    [result.paco2_median, "Median", "rgba(23, 32, 38, 0.92)", "solid"],
    [result.paco2_q_high, "PI high", "rgba(23, 32, 38, 0.55)", "dot"],
  ];
  const layout = {
    title: "Posterior PaCO2 distribution conditioned on observed TcCO2",
    xaxis: { title: "PaCO2 (mmHg)" },
    yaxis: { title: "Posterior probability" },
    bargap: 0.05,
    margin: { t: 52, r: 20, b: 58, l: 66 },
    shapes: verticals.map(([x, , color, dash]) => ({
      type: "line",
      x0: x,
      x1: x,
      y0: 0,
      y1: 1,
      xref: "x",
      yref: "paper",
      line: { color, dash, width: 2 },
    })),
    annotations: verticals.map(([x, label]) => ({
      x,
      y: 1,
      xref: "x",
      yref: "paper",
      yanchor: "bottom",
      text: label,
      showarrow: false,
      font: { size: 11 },
    })),
  };
  globalThis.Plotly.newPlot(elements.chart, traces, layout, {
    responsive: true,
    displayModeBar: false,
  });
}

function setBusy(isBusy) {
  elements.calculate.disabled = isBusy;
  elements.calculate.textContent = isBusy ? "Calculating..." : "Calculate";
}

function showError(message) {
  elements.error.hidden = false;
  elements.error.textContent = message;
}

function hideError() {
  elements.error.hidden = true;
  elements.error.textContent = "";
}

function titleCase(value) {
  return `${value.charAt(0).toUpperCase()}${value.slice(1)}`;
}

function formatNumber(value) {
  return Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1);
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  calculate();
});

initialize();
