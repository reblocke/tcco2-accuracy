# Architecture

## Package Layout
The Python package lives under `src/tcco2_accuracy/`. It remains the numerical
source of truth for Conway reproduction, bootstrap uncertainty, simulation,
inverse inference, conditional classification, manuscript workflows, and the
browser contract.

Tests live under `tests/`. Browser E2E tests live under `tests/e2e/`.

## Static App
The app under `web/` is static HTML/CSS/JS. It uses:
- `web/assets/js/app.js` for UI state, uploads, worker messages, and Plotly rendering.
- `web/pyodide_worker.js` for Pyodide initialization and Python execution.
- `web/assets/data/` for staged canonical CSV assets.
- `web/assets/py/` for staged Python source copied from `src/`.

## Browser Contract
`tcco2_accuracy.browser_contract` exposes JSON-like functions for the browser:
- `compute_ui_payload(payload)` computes a serializable posterior result.
- `build_bootstrap_payload(payload)` computes serializable bootstrap draws for custom study inputs.

The contract accepts CSV text or record lists and returns plain dictionaries,
lists, strings, and numbers. It delegates validation, parameter routing,
bootstrap generation, prior weighting, and posterior summaries to existing
package modules.

## Staging
`scripts/stage_web_python.py` copies the package and default data assets into
`web/`. It writes `web/assets/py/manifest.json`, which the worker uses to mount
the staged package into Pyodide.

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
