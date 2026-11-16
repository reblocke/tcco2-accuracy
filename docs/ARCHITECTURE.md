# Architecture

## Package Layout
The Python package lives under `src/tcco2_accuracy/`. The pure numerical source
of truth is `src/tcco2_accuracy/core/`, which owns corrected Conway-input agreement calculations,
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
- `web/assets/js/lookup.js` for exact conditional row selection and the variance display.
- `web/assets/js/app.js` for UI state, uploads, worker messages, and SVG rendering.
- `web/pyodide_worker.js` for Pyodide initialization and Python execution.
- `web/assets/data/` for generated staged public models and comparison data.
- `web/assets/py/` for generated staged Python source copied from an explicit allowlist.

The homepage initially plots the signed density from the verified setting kernel. Both calculators use the same distribution chart renderer. In the advanced calculator, the agreement chart shows signed disagreement and the zero reference. Its conditional chart presents the empirical posterior masses computed by `tcco2_accuracy.pages_prediction.conditional` in the worker. Its compressed, setting-specific kernel and uploaded prior provide all numerical inputs. The previous bootstrap contract is retained for offline clients.

## Staging
`scripts/stage_web_python.py` copies only the browser-required Python allowlist
and default data assets into `web/`. It writes `web/assets/py/manifest.json`,
which the worker uses to mount the staged package into Pyodide. Generated
`web/assets/py/` and `web/assets/data/` files are ignored by git and should not
be edited by hand.

Staged assets:
- `models/pages/summary.json` and the four public NPZ disagreement kernels
- Python source files required by the browser contract

## Data Strategy
The browser default path uses repo-shipped agreement assets only and displays signed future-setting
disagreement. It does not require the restricted in-silico `.dta`, stage a PaCO2 prior, or fetch one.
Prior weighting is upload-only and remains client-side. Offline workflows may use an explicitly
authorized `.dta` only with outputs under `.pytest_tmp/`, `.tmp/`, or an explicitly approved
external private workspace. Normalized restricted-derived weights are governed like exact counts
because they may reconstruct the source distribution. See `docs/data_release_contract.json`.

## Privacy Boundary
The app has no backend, telemetry, persistence, or PHI-bearing URL state.
Browser CDN requests load runtime libraries only; user values and uploaded files
remain in the browser worker.
