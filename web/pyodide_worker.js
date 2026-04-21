const PYODIDE_VERSION = "0.29.0";
const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let pyodidePromise;
let defaultAssetsPromise;

self.onmessage = async (event) => {
  const { id, type, payload } = event.data;
  try {
    if (type === "init") {
      await initializePyodide();
      self.postMessage({ id, type: "result", result: { ok: true } });
      return;
    }
    if (type === "compute") {
      const result = await compute(payload);
      self.postMessage({ id, type: "result", result });
      return;
    }
    throw new Error(`Unknown worker message: ${type}`);
  } catch (error) {
    self.postMessage({ id, type: "error", error: error.message || String(error) });
  }
};

async function initializePyodide() {
  if (pyodidePromise) {
    return pyodidePromise;
  }
  pyodidePromise = (async () => {
    importScripts(`${PYODIDE_INDEX_URL}pyodide.js`);
    const pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });
    await pyodide.loadPackage(["numpy", "pandas", "scipy"]);
    await stagePythonPackage(pyodide);
    pyodide.runPython(`
import sys
if "/home/pyodide/src" not in sys.path:
    sys.path.insert(0, "/home/pyodide/src")
from tcco2_accuracy.browser_contract import compute_ui_payload
`);
    return pyodide;
  })();
  return pyodidePromise;
}

async function stagePythonPackage(pyodide) {
  const manifest = await fetchJson("assets/py/manifest.json");
  pyodide.FS.mkdirTree("/home/pyodide/src");
  for (const file of manifest.files) {
    const source = await fetchText(`assets/py/${file}`);
    const destination = `/home/pyodide/src/${file}`;
    pyodide.FS.mkdirTree(destination.split("/").slice(0, -1).join("/"));
    pyodide.FS.writeFile(destination, source);
  }
}

async function compute(inputPayload) {
  const pyodide = await initializePyodide();
  const defaults = await defaultAssets();
  const payload = {
    ...inputPayload,
    prior_bins_csv: inputPayload.prior_bins_csv || defaults.priorBins,
  };
  if (inputPayload.use_canonical_params) {
    payload.params_csv = defaults.params;
    delete payload.study_csv;
  } else {
    payload.study_csv = inputPayload.study_csv || defaults.studies;
    delete payload.params_csv;
  }

  const pyPayload = pyodide.toPy(payload);
  pyodide.globals.set("_browser_payload", pyPayload);
  const pyResult = await pyodide.runPythonAsync("compute_ui_payload(_browser_payload)");
  const result = pyResult.toJs({ dict_converter: Object.fromEntries });
  pyResult.destroy();
  pyPayload.destroy();
  return result;
}

function defaultAssets() {
  if (!defaultAssetsPromise) {
    defaultAssetsPromise = Promise.all([
      fetchText("assets/data/bootstrap_params.csv"),
      fetchText("assets/data/conway_studies.csv"),
      fetchText("assets/data/paco2_public_prior.csv"),
    ]).then(([params, studies, priorBins]) => ({ params, studies, priorBins }));
  }
  return defaultAssetsPromise;
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status}`);
  }
  return response.text();
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status}`);
  }
  return response.json();
}
