# TcCO2 Accuracy — Validation Targets

## Conway Table 1
- Fixtures extracted from `Conway Meta/Thorax 2019 TcCO2 metaanalysis.pdf`.
- Invariants: `tests/core/test_conway_meta.py` validates bias/SD/τ²/LoA against Table 1.

## Workflow validation stages

### Meta-analysis reproduction
- Purpose: recompute bias/SD/τ²/LoA per subgroup from Conway inputs.
- Invariants: LoA identity checks in `src/tcco2_accuracy/workflows/meta.py` and tests in `tests/core/test_conway_meta.py`.
- Artifacts: `artifacts/meta_loa_check.md`, `artifacts/conway_table1_fixture_summary.md`.
- Scientific claim: Conway study-level synthesis reproduces Table 1 magnitudes.

### Bootstrap uncertainty propagation
- Purpose: propagate δ/σ²/τ² uncertainty with study-level (route-1) bootstrap.
- Invariants: reproducibility and τ² >= 0 checks in `tests/core/test_bootstrap.py`.
- Artifacts: `artifacts/bootstrap_params.csv`, `artifacts/bootstrap_summary.md`.
- Scientific claim: between-study uncertainty in δ, σ², τ² is propagated.

### PaCO2 distribution ingestion
- Purpose: ingest PaCO2 distributions and assign mutually exclusive subgroups.
- Invariants: subgroup assignment and quantile checks in `tests/core/test_paco2_distribution.py`.
- Artifacts: `artifacts/paco2_distribution_summary.md`.
- Scientific claim: empirical PaCO2 priors are correctly stratified.

### Forward simulation
- Purpose: propagate bootstrap parameters through TcCO2 accuracy metrics.
- Invariants: moment/interval checks in `tests/core/test_simulation.py`.
- Artifacts: `artifacts/simulation_summary.md`.
- Scientific claim: TcCO2 accuracy metrics propagate parameter uncertainty.

### Inverse inference
- Purpose: compute TcCO2 → PaCO2 posterior intervals and exceedance probabilities.
- Invariants: likelihood/prior checks in `tests/core/test_inference.py` and determinism in `tests/workflows/test_workflows.py`.
- Artifacts: `artifacts/inference_demo.md`.
- Scientific claim: TcCO2 measurements map to PaCO2 intervals with uncertainty.

### Conditional misclassification curves
- Purpose: summarize conditional TN/FP/FN/TP probabilities by true PaCO2 bin.
- Invariants: probability mass and branching checks in `tests/core/test_hybrid_bootstrap_and_conditional.py`.
- Artifacts: `artifacts/conditional_classification_t45.csv`, `artifacts/conditional_classification_t45.md`.
- Scientific claim: conditional error rates vary smoothly with PaCO2 and include parameter uncertainty.

### Manuscript reporting outputs
- Purpose: generate manuscript-ready tables, figures, and results snippets.
- Invariants: smoke test in `tests/workflows/test_manuscript_workflow.py`.
- Artifacts: `artifacts/manuscript_table1.csv`, `artifacts/manuscript_table2_two_stage.csv`,
  `artifacts/manuscript_table3_prediction_intervals.csv`, `artifacts/manuscript_results_snippets.md`.
- Scientific claim: reported operating characteristics, two-stage strategy metrics, and prediction intervals
  are reproducible from the bootstrap parameter uncertainty model.

### Browser contract and static app
- Purpose: verify that the Pages app calls the Python source of truth through a JSON-safe contract.
- Invariants: `tests/contracts/test_browser_contract.py` compares contract outputs to `predict_paco2_from_tcco2`
  for canonical prior-weighted inference and exercises subgroups, inference modes, custom priors,
  and uploaded study-table recomputation.
- Staging: `tests/contracts/test_stage_web_python.py` verifies package/data staging into `web/assets/`.
- E2E: `tests/e2e/test_web_app.py` verifies Pyodide loads, default calculation completes, metrics render,
  and threshold changes update the browser result.
- Scientific claim: browser-facing outputs are a serialization of the authoritative Python model, not
  a separate JavaScript implementation.
