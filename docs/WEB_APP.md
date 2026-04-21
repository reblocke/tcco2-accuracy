# TcCO2 to PaCO2 Static Web App

## What The App Does
- Computes a PaCO2 median with a prediction interval conditional on an observed TcCO2.
- Reports `P(PaCO2 >= threshold)` and the observed TcCO2 threshold classification.
- Visualizes the posterior PaCO2 distribution, threshold region, direct-labeled prior and scaled likelihood curves in prior-weighted mode, and PI/median markers.
- Supports custom study table and binned prior uploads in CSV; XLSX is parsed client-side when the browser SheetJS dependency is available.
- Runs the Python package in a Pyodide worker. JavaScript does not implement the statistical model.

## Default Values
- TcCO2 = 50 mmHg
- Hypercapnia threshold = 45 mmHg
- Prediction interval = 95% PI
- Inference mode = prior-weighted
- Setting = All
- Bootstrap draws = 1000
- Bootstrap mode = `cluster_plus_withinstudy`
- Seed = 202401

## Data Assets
- `web/assets/data/conway_studies.csv` - generated staged canonical study table.
- `web/assets/data/paco2_public_prior.csv` - generated staged public 1 mmHg PaCO2 density prior.
- `web/assets/data/bootstrap_params.csv` - generated staged canonical bootstrap parameters.
- `web/assets/py/tcco2_accuracy/` - generated staged Python package copied from the browser allowlist.

Run `make stage-web` after changing Python browser-facing code or default data
assets. Generated staged assets are not tracked.

## Browser Behavior
- Default calculations use the staged canonical bootstrap parameters for responsiveness.
- Uploaded study tables or changed bootstrap settings recompute bootstrap parameters in the browser worker using the same Python model.
- Uploaded prior bins replace the staged public prior for prior-weighted inference.
- The staged public prior uses 1 mmHg normalized weights and omits exact bin counts from restricted local data.
- The posterior chart uses a posterior-focused x-axis for readability; calculations still use the full posterior and prior support.
- In prior-weighted mode, the likelihood curve is Python-computed and normalized to sum to 1 over the histogram bins for visual comparison only.
- Browser copy uses threshold-classification and posterior-mass terms rather than clinical correctness language.
- The browser app does not read local filesystem paths, store values, send user inputs to a backend, or encode patient values in URLs.

## Local Run
```bash
uv sync --locked
make serve
```

Open http://127.0.0.1:8000.

## Verification
```bash
make test
make e2e
make visual-qa
make verify
```

`make e2e` runs Playwright against the staged static app and verifies the Pyodide contract executes in Chromium.
`make visual-qa` writes desktop and mobile screenshots to `.pytest_tmp/visual-qa/`
for chart-label review. It is an explicit local/reviewer aid and is not part of
`make verify`.
