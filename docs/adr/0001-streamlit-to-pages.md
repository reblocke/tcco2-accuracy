# ADR 0001: Migrate Streamlit App To Static GitHub Pages

## Status
Accepted.

## Context
The project originally exposed app-facing inference through Streamlit. That made
public hosting depend on a server-side Python session and Streamlit Cloud
deployment assumptions. The repository also carried parallel dependency files
and nested package/test paths.

## Decision
Use a static GitHub Pages app with Pyodide in a web worker. Keep Python as the
single numerical source of truth by staging `src/tcco2_accuracy/` into
`web/assets/py/` and calling `tcco2_accuracy.browser_contract` from the worker.
Use `pyproject.toml` plus `uv.lock` as the single dependency path.

## Consequences
- The public app can run with no backend and no server-local path controls.
- Default browser calculations use staged canonical CSV/bootstrap assets and do
  not require the large in-silico `.dta`.
- Custom study tables can trigger in-browser bootstrap recomputation, which is
  slower than the canonical default path.
- Pyodide, Plotly.js, and SheetJS are pinned browser dependencies loaded from
  CDNs unless a later decision vendors them locally.
