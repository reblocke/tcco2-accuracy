# TcCO2 to PaCO2 Static Web App

## What The App Does
- Computes a PaCO2 median with a prediction interval conditional on an observed TcCO2.
- Reports `P(PaCO2 >= threshold)` and the TcCO2 threshold decision.
- Visualizes the posterior PaCO2 distribution, threshold region, prior overlay when used, and PI/median markers.
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
- `web/assets/data/conway_studies.csv` - staged canonical study table.
- `web/assets/data/paco2_prior_bins.csv` - staged binned PaCO2 prior.
- `web/assets/data/bootstrap_params.csv` - staged canonical bootstrap parameters.
- `web/assets/py/tcco2_accuracy/` - staged Python package copied from `src/tcco2_accuracy/`.

Run `make stage-web` after changing Python browser-facing code or default data assets.

## Browser Behavior
- Default calculations use the staged canonical bootstrap parameters for responsiveness.
- Uploaded study tables or changed bootstrap settings recompute bootstrap parameters in the browser worker using the same Python model.
- Uploaded prior bins replace the staged default prior bins for prior-weighted inference.
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
make verify
```

`make e2e` runs Playwright against the staged static app and verifies the Pyodide contract executes in Chromium.
