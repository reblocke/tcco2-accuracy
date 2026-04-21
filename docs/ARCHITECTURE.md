# Architecture

## Package Layout
The Python package lives under `src/tcco2_accuracy/`. The pure numerical source
of truth is `src/tcco2_accuracy/core/`, which owns Conway reproduction,
bootstrap uncertainty, simulation, inverse inference, conditional classification,
two-stage calculations, PaCO2 subgrouping, prior-bin expansion, and validation.

`src/tcco2_accuracy/data.py` owns repository paths and file loading. Top-level
modules such as `tcco2_accuracy.inference` remain import-compatible wrappers for
the core layer. `src/tcco2_accuracy/reporting/` builds manuscript/reviewer
tables, figure-data CSVs, and result snippets; `src/tcco2_accuracy/workflows/`
keeps command-facing orchestration entry points.

Tests live under `tests/core/`, `tests/contracts/`, `tests/workflows/`, and
`tests/e2e/`. Fixtures remain under `tests/fixtures/`.

## Static App
The app under `web/` is static HTML/CSS/JS. It uses:
- `web/assets/js/app.js` for UI state, uploads, worker messages, and Plotly rendering.
- `web/pyodide_worker.js` for Pyodide initialization and Python execution.
- `web/assets/data/` for generated staged canonical CSV assets.
- `web/assets/py/` for generated staged Python source copied from an explicit allowlist.

The posterior chart uses a posterior-focused x-axis so long prior tails do not
compress the clinically relevant posterior view. The full posterior/prior arrays
still drive all numeric summaries. In prior-weighted mode, the optional
likelihood overlay is computed in Python and JavaScript only renders the
serialized curve.

## Browser Contract
`tcco2_accuracy.browser_contract` exposes JSON-like functions for the browser:
- `compute_ui_payload(payload)` computes a serializable posterior result.
- `build_bootstrap_payload(payload)` computes serializable bootstrap draws for custom study inputs.

The contract accepts CSV text or record lists and returns plain dictionaries,
lists, strings, and numbers. It delegates validation, parameter routing,
bootstrap generation, prior weighting, and posterior summaries to existing
package modules.

## Staging
`scripts/stage_web_python.py` copies only the browser-required Python allowlist
and default data assets into `web/`. It writes `web/assets/py/manifest.json`,
which the worker uses to mount the staged package into Pyodide. Generated
`web/assets/py/` and `web/assets/data/` files are ignored by git and should not
be edited by hand.

Staged assets:
- `Data/conway_studies.csv`
- `Data/paco2_prior_bins.csv`
- `artifacts/bootstrap_params.csv`
- Python source files required by the browser contract

## Data Strategy
The browser default path uses repo-shipped static assets only. It does not
require the large in-silico `.dta`. Offline workflows may still use the `.dta`
when present for artifact generation and prior-bin rebuilding.

## Privacy Boundary
The app has no backend, telemetry, persistence, or PHI-bearing URL state.
Browser CDN requests load runtime libraries only; user values and uploaded files
remain in the browser worker.
