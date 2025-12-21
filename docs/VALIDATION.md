# TcCO2 Accuracy — Validation Targets

## Conway Table 1
- Fixtures extracted from `Conway Meta/Thorax 2019 TcCO2 metaanalysis.pdf`.
- Invariants: `python/tests/test_conway_meta.py` validates bias/SD/τ²/LoA against Table 1.

## Workflow validation stages

### Meta-analysis reproduction
- Purpose: recompute bias/SD/τ²/LoA per subgroup from Conway inputs.
- Invariants: LoA identity checks in `python/src/tcco2_accuracy/workflows/meta.py` and tests in `python/tests/test_conway_meta.py`.
- Artifacts: `artifacts/meta_loa_check.md`, `artifacts/conway_table1_fixture_summary.md`.
- Scientific claim: Conway study-level synthesis reproduces Table 1 magnitudes.

### Bootstrap uncertainty propagation
- Purpose: propagate δ/σ²/τ² uncertainty with study-level (route-1) bootstrap.
- Invariants: reproducibility and τ² ≥ 0 checks in `python/tests/test_bootstrap.py`.
- Artifacts: `artifacts/bootstrap_params.csv`, `artifacts/bootstrap_summary.md`.
- Scientific claim: between-study uncertainty in δ, σ², τ² is propagated.

### PaCO2 distribution ingestion
- Purpose: ingest PaCO2 distributions and assign mutually exclusive subgroups.
- Invariants: subgroup assignment and quantile checks in `python/tests/test_paco2_distribution.py`.
- Artifacts: `artifacts/paco2_distribution_summary.md`.
- Scientific claim: empirical PaCO2 priors are correctly stratified.

### Forward simulation
- Purpose: propagate bootstrap parameters through TcCO2 accuracy metrics.
- Invariants: moment/interval checks in `python/tests/test_simulation.py`.
- Artifacts: `artifacts/simulation_summary.md`.
- Scientific claim: TcCO2 accuracy metrics propagate parameter uncertainty.

### Inverse inference
- Purpose: compute TcCO2 → PaCO2 posterior intervals and exceedance probabilities.
- Invariants: likelihood/prior checks in `python/tests/test_inference.py` and determinism in `python/tests/test_workflows.py`.
- Artifacts: `artifacts/inference_demo.md`.
- Scientific claim: TcCO2 measurements map to PaCO2 intervals with uncertainty.
