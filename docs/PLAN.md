# TcCO2 Accuracy — Active Plan

## Stabilized Architecture
1. Keep the Python package in `src/tcco2_accuracy/` as the numerical source of truth.
2. Keep scientific and workflow tests in `tests/`.
3. Serve the public app as static GitHub Pages from `web/`.
4. Stage Python and canonical CSV assets with `scripts/stage_web_python.py`.
5. Verify changes through `make verify`.

## Current Priorities
1. Preserve Conway reproduction, bootstrap uncertainty, simulation, inference, and manuscript workflow behavior.
2. Keep the browser contract numerically aligned with `predict_paco2_from_tcco2`.
3. Keep app input handling client-side with no backend, telemetry, persistence, or PHI-bearing URLs.
4. Update validation artifacts only when scientific behavior intentionally changes.
