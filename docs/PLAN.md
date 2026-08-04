# TcCO2 Accuracy — Active Plan

## Stabilized Architecture
1. Keep the Python package in `src/tcco2_accuracy/` as the numerical source of truth.
2. Keep scientific and workflow tests in `tests/`.
3. Serve the public app as static GitHub Pages from `web/`.
4. Stage Python and canonical CSV assets with `scripts/stage_web_python.py`.
5. Verify changes through `make verify`.

## Current Priorities
1. Close the corrected-provisional wave: preserve canonical numerical outputs, enforce the locked
   public promotion contract, and require the GitHub Actions x86 gate to pass.
2. Complete independent statistical validation (TCCO2-002/005/007), including an independently
   authored reference comparison and biostatistical review of near-zero tau-squared CI behavior.
3. Lock the analysis specification: estimand, PaCO2 sampling unit, repeated patients, publication
   clustering, bootstrap model, supported range, proportional bias, subgroup mappings, ED/inpatient
   definition, and row-level Conway provenance.
4. Close downstream safety and governance gaps: missing-group behavior, stable tail calculations,
   conditional-bin truth assignment, input validation, and exact-count/IRB/DUA provenance.
5. Rebuild downstream results only in a private workspace, jointly propagate agreement and target-
   distribution uncertainty, run clustering/model sensitivities with at least 10,000 release draws,
   and document Monte Carlo stability before approved aggregate promotion.
6. Restore an authoritative text-diffable manuscript in an approved workspace and require
   traceability, clean-room reproduction, and independent sign-off before unfreezing results,
   tagging, or archiving a release.

Throughout these priorities, keep the browser contract aligned with the Python numerical source of
truth, keep user inputs client-side, and update validation artifacts only for intentional scientific
changes.
