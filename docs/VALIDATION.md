# TcCO2 Accuracy — Validation Targets

## Conway Table 1
- Fixtures extracted from `Conway Meta/Thorax 2019 TcCO2 metaanalysis.pdf`.
- Initial checks: column integrity and LoA presence.

## Workflow validation stages

### Meta-analysis reproduction
- Stage: `workflows.meta.run_meta_checks`.
- Evidence: recomputed bias/SD/τ²/LoA per subgroup against Conway inputs.
- Scientific claim: Conway study-level synthesis reproduces Table 1 magnitudes.

### Bootstrap uncertainty propagation
- Stage: `workflows.bootstrap.run_bootstrap`.
- Evidence: study-level cluster bootstrap (route-1) yields stable LoA ranges.
- Scientific claim: between-study uncertainty in δ, σ², τ² is propagated.

### PaCO2 distribution ingestion
- Stage: `workflows.paco2.run_paco2_summary`.
- Evidence: subgroup counts + quantiles (ED included in `ed_inp`).
- Scientific claim: empirical PaCO2 priors are correctly stratified.

### Forward simulation
- Stage: `workflows.sim.run_forward_simulation_summary`.
- Evidence: summary tables of d moments, LoA, and classification metrics.
- Scientific claim: TcCO2 accuracy metrics propagate parameter uncertainty.

### Inverse inference
- Stage: `workflows.infer.run_inference_demo`.
- Evidence: posterior medians + 95% prediction intervals per subgroup.
- Scientific claim: TcCO2 measurements map to PaCO2 intervals with uncertainty.
