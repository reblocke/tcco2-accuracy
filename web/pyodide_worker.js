// Only static public files are fetched. Uploaded values stay inside this worker.
const indexURL = "https://cdn.jsdelivr.net/pyodide/v0.29.0/full/";
let initialization;
async function getRuntime() {
  const expected = new URL(self.location.href).searchParams.get("bundle");
  const archive = await verifiedAsset("pages.zip", expected);
  importScripts(indexURL + "pyodide.js");
  const py = await loadPyodide({ indexURL });
  await py.loadPackage(["numpy", "pandas", "scipy"]);
  py.FS.writeFile("/home/pyodide/pages.zip", archive);
  py.runPython("import sys; sys.path.insert(0,'/home/pyodide/pages.zip')");
  py.runPython(
    "import io, json, numpy as np\nfrom tcco2_accuracy.pages_prediction import conditional",
  );
  return py;
}
async function read(url) {
  const r = await fetch(url);
  if (!r.ok) throw Error("Cannot load public model asset.");
  return r;
}
async function verifiedAsset(path, expected) {
  const bytes = await read(path).then((response) => response.arrayBuffer());
  const digest = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    (v) => v.toString(16).padStart(2, "0"),
  ).join("");
  if (digest !== expected)
    throw Error("Public asset checksum mismatch. Reload this page.");
  return new Uint8Array(bytes);
}
const kernels = new Map();
self.onmessage = async ({ data }) => {
  const { id, group, hash, target, prior, tc, threshold } = data;
  try {
    const py = await (initialization ??= getRuntime());
    if (!kernels.has(group)) {
      kernels.set(
        group,
        await verifiedAsset("../data/" + group + ".npz", hash),
      );
    }
    const locals = py.toPy({
      blob: kernels.get(group),
      target,
      prior,
      tc,
      threshold,
    });
    try {
      const result = JSON.parse(
        py.runPython(
          "with np.load(io.BytesIO(bytes(blob))) as kernel:\n    answer=conditional(kernel,target,prior,tc,threshold)\njson.dumps(answer,allow_nan=False)",
          { locals },
        ),
      );
      self.postMessage({ id, result });
    } finally {
      locals.destroy();
    }
  } catch (e) {
    self.postMessage({
      id,
      error: String(e.message).split("\n").filter(Boolean).at(-1),
    });
  }
};
