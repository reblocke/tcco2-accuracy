# TcCO2 Accuracy — Validation Targets

## Corrected agreement method
- Method revision: `agreement_natural_log_tau2_direct_v1`; result status: `provisional`.
- Statistical authority: Tipton and Shuster equations 4.4-4.5, 4.13, and 4.16
  (PMCID `PMC5585060`). The linked Conway Figshare source and Table 1 are preserved as
  published/legacy comparators, not correctness targets for the corrected equations.
- Equation tests use an independent test-only oracle, a one-row hand calculation, coherent
  natural/base-10 scale equivalence, direct-scale τ² uncertainty, a negative raw DL estimate with
  a zero-truncated production-summary boundary, a single-study boundary, and unit-rescaling
  equivariance. Production helpers are not imported into the oracle.
- Point estimates and LoA are required to remain finite at the τ²=0 boundary. The corresponding
  confidence-interval behavior remains provisional pending independent biostatistical review.
- The published fixture remains numerically immutable and its schema/LoA identity continue to be
  checked separately from corrected-method values.
- Independent biostatistical sign-off is pending. It blocks final release and downstream manuscript
  promotion, but not a visibly provisional Pages deployment.

## Workflow validation stages

### Corrected agreement comparison
- Purpose: recompute corrected bias/SD/τ²/LoA per subgroup from public Conway inputs and compare
  them with the frozen published values.
- Invariants: LoA identity plus independent equation, dimensional, scaling, and boundary tests in
  `tests/core/`.
- Artifacts: `artifacts/meta_loa_check.md`, `artifacts/conway_table1_fixture_summary.md`.
- Scientific claim: the corrected implementation follows the cited equations; it is expected to
  diverge materially from the inherited Table 1 reproduction because the inherited log transform
  and τ² uncertainty scales were inconsistent.

### Bootstrap uncertainty propagation
- Purpose: propagate δ/σ²/τ² uncertainty with study-level (route-1) bootstrap.
- Invariants: reproducibility, τ² >= 0, method/status metadata, and committed-versus-recomputed
  same-seed checks.
- Artifacts: `artifacts/bootstrap_params.csv`, `artifacts/bootstrap_summary.md`.
- Scientific claim: between-study uncertainty in δ, σ², and τ² is propagated using the
  corrected provisional method.

### PaCO2 distribution ingestion
- Purpose: ingest PaCO2 distributions and assign mutually exclusive subgroups.
- Invariants: subgroup assignment and quantile checks in `tests/core/test_paco2_distribution.py`.
- Artifacts: `artifacts/paco2_distribution_summary.md`.
- Status: frozen at the legacy agreement-method wave pending downstream regeneration and governance
  review; no current corrected-method claim is made from this artifact.

### Forward simulation
- Purpose: propagate bootstrap parameters through TcCO2 accuracy metrics.
- Invariants: moment/interval checks in `tests/core/test_simulation.py`.
- Artifacts: `artifacts/simulation_summary.md`.
- Status: frozen at the legacy agreement-method wave; no current corrected-method claim is made.

### Inverse inference
- Purpose: compute TcCO2 → PaCO2 posterior intervals and exceedance probabilities.
- Invariants: likelihood/prior checks in `tests/core/test_inference.py` and determinism in `tests/workflows/test_workflows.py`.
- Artifacts: `artifacts/inference_demo.md`.
- Status: frozen at the legacy agreement-method wave; no current corrected-method claim is made.

### Conditional misclassification curves
- Purpose: summarize conditional TN/FP/FN/TP probabilities by true PaCO2 bin.
- Invariants: probability mass and branching checks in `tests/core/test_hybrid_bootstrap_and_conditional.py`.
- Artifacts: `artifacts/conditional_classification_t45.csv`, `artifacts/conditional_classification_t45.md`.
- Status: frozen at the legacy agreement-method wave; no current corrected-method claim is made.

### Manuscript reporting outputs
- Purpose: generate manuscript-ready tables, figures, and results snippets.
- Invariants: smoke test in `tests/workflows/test_manuscript_workflow.py`.
- Artifacts: `artifacts/manuscript_table1.csv`, `artifacts/manuscript_table2_two_stage.csv`,
  `artifacts/manuscript_table3_prediction_intervals.csv`, `artifacts/manuscript_results_snippets.md`.
- Status: frozen at the legacy agreement-method wave. These values must not be copied into a final
  manuscript or described as corrected until the downstream regeneration gate is complete.

### Browser contract and static app
- Purpose: verify that the Pages app calls the Python source of truth through a JSON-safe contract.
- Invariants: `tests/contracts/test_browser_contract.py` compares contract outputs to `predict_paco2_from_tcco2`
  for canonical prior-weighted inference and exercises subgroups, inference modes, custom priors,
  and uploaded study-table recomputation.
- Staging: `tests/contracts/test_stage_web_python.py` verifies package/data staging into `web/assets/`.
- E2E: `tests/e2e/test_web_app.py` verifies Pyodide loads, default calculation completes, metrics render,
  threshold changes update the browser result, and a failed custom-study recalculation clears prior
  metrics, chart output, and result-provenance attributes before displaying the error.
- Scientific claim: browser-facing outputs are a serialization of the authoritative Python model, not
  a separate JavaScript implementation.
- Version gate: canonical parameter assets must contain exactly one current method revision and
  provisional status; missing, stale, or mixed metadata fail closed. The UI must keep the corrected
  method, independent-review-pending, frozen-manuscript, and research-only notice visible.

## Artifact and release gates
- `artifacts/STATUS.md` is the current/frozen manifest. Only five public Conway-derived outputs are
  promoted by `--profile public-agreement`; restricted PaCO2 loaders must be unreachable in that
  profile.
- Promotion to repository `artifacts/` requires the canonical CSV, seed 202401, 1,000 draws per
  subgroup, and `cluster_plus_withinstudy`. Contract tests reject each noncanonical option before
  workflow execution, preserve existing deployed bytes on failure, enforce CSV/XLSX semantic
  parity, and verify that browser staging copies the canonical parameter artifact unchanged. A
  fault-injection regression also fails sequential promotion after multiple replacements and checks
  exact restoration of every preexisting artifact plus removal of newly introduced files.
- `--profile full` requires an explicit restricted source and a scratch/private output directory.
  It is a comparison workflow only until TCCO2-006 and the governance review are complete.
- Final tag, removal of provisional copy, manuscript unfreeze, submission-readiness claims, and final
  downstream promotion require independent biostatistical review plus resolution of the remaining
  source/reference and exact-count governance tickets.
