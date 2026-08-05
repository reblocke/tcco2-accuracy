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

## Scientific input validation
- Conway table validation fails closed for an empty table, an empty selected subgroup, blank or
  duplicate identifiers after whitespace stripping, non-integer counts, `n_participants <= 1`, or
  `n_pairs < n_participants`.
- Export regressions require missing or blank RData/count identifiers and non-integer observed
  counts to fail even under `--no-strict` or `--allow-missing-counts`.
- Supplied `sd` and `s2` values must be finite and strictly positive. When both are present,
  `sd²` and `s2` must agree at `rtol=1e-10`, `atol=1e-12`. Optional `c` must be finite,
  at least 1, and agree with `n_pairs / n_participants` at the same tolerance.
- Conway study flags may overlap; validation must accept a row selected into multiple Conway
  subgroups. Record-level PaCO2 distribution groups remain mutually exclusive.
- Retained PaCO2 observations and prior support values must be finite and strictly positive.
  Genuinely missing source-distribution observations are dropped before validation; nonnumeric,
  infinite, zero, and negative retained values fail closed. A finite positive value above the
  observed support is accepted because no evidence-based upper validation bound is specified.
- Prepared PaCO2 inputs drop rows with genuinely missing `paco2` before validating the retained
  rows; each retained subgroup must be non-missing and normalize to `pft`, `ed_inp`, or `icu`. Raw
  assignment flags must be binary 0/1 whenever non-missing; only genuinely missing flag values are
  replaced by 0.
- Prior group labels must be non-missing and nonblank and, after normalization, contain exactly
  `pft`, `ed_inp`, `icu`, and `all`, with neither missing nor extra groups.
- Classification thresholds, including the two-stage truth threshold, must be finite and strictly
  positive. Two-stage TcCO2 zone boundaries must be finite and strictly ordered; negative finite
  boundaries remain valid.
- Regression coverage belongs in `tests/core/test_validate_inputs.py` and the affected PaCO2,
  posterior, simulation, two-stage, workflow, and browser-contract tests. These validation changes
  do not authorize regeneration or promotion of frozen PaCO2-dependent artifacts.

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
- Invariants: subgroup assignment, prepared-label allowlisting, raw binary-flag validation, and
  quantile checks in `tests/core/test_paco2_distribution.py`.
- Artifacts: no restricted-derived distribution summary or prior is tracked. Tests use a clearly
  synthetic prior fixture.
- Status: the former distribution summary and reconstructable prior were removed from the current
  tree under `docs/data_release_contract.json`. Private future rebuilds remain frozen and gated.

### Forward simulation
- Purpose: propagate bootstrap parameters through TcCO2 accuracy metrics.
- Invariants: moment/interval checks plus finite analytic likelihood-ratio comparisons at normal
  z values of ±8 and ±12 in `tests/core/test_simulation.py`. Survival and LR calculations are
  performed without `1-cdf` subtraction.
- Artifacts: `artifacts/simulation_summary.md`.
- Status: retained as a frozen aggregate legacy-method comparator; it is not corrected or
  release-approved.

### Inverse inference
- Purpose: compute TcCO2 → PaCO2 posterior intervals and exceedance probabilities.
- Invariants: likelihood/prior checks, exact survival-tail checks at z values of ±8 and ±12 in
  `tests/core/test_inference.py`, and determinism in `tests/workflows/test_workflows.py`.
- Artifacts: the detailed inference demo was removed. Prediction-interval CSV/Markdown files remain
  tracked only as frozen aggregate legacy-method comparators.
- Status: no current corrected-method or release-approved downstream claim is made.

### Conditional misclassification curves
- Purpose: summarize conditional TN/FP/FN/TP probabilities by true PaCO2 bin.
- Invariants: probability mass, raw-value truth assignment, half-open edge handling, legacy-display
  independence, decimal-width exact-edge handling, and 44.49/44.50/44.99/45.00/45.01 mmHg boundary checks in
  `tests/core/test_hybrid_bootstrap_and_conditional.py`. `paco2_bin` and `paco2_bin_upper`
  identify `[lower, upper)`; truth is never derived from either edge.
- Artifacts: detailed conditional-classification CSV/Markdown files were removed from the current
  tracked tree.
- Status: future private regeneration remains frozen; no current corrected-method claim is made.

### Two-stage probability stability
- Purpose: calculate lower, reflex, and upper-zone probabilities without normal-tail cancellation.
- Invariants: zone mass, stable `[8 SD, 12 SD)` interval probability, positive mass for an ordered
  interval only `1e-16` SD wide, nonzero 12-SD survival, and finite extreme-tail LR checks in
  `tests/core/test_two_stage.py`.
- Status: implementation corrected for future private rebuilds; retained rounded
  `artifacts/two_stage_summary.md` and `artifacts/manuscript_table2_two_stage.md` remain frozen,
  were not regenerated, and are not release-approved. Their detailed CSVs were removed.

### Manuscript reporting outputs
- Purpose: generate manuscript-ready tables, figures, and results snippets.
- Invariants: smoke test in `tests/workflows/test_manuscript_workflow.py`.
- Artifacts: retained files are enumerated in `artifacts/STATUS.md`; detailed Table 1 and selected
  exact/reconstructable CSV outputs were removed under the release contract.
- Status: frozen at the legacy agreement-method wave. These values must not be copied into a final
  manuscript or described as corrected until the downstream regeneration gate is complete.

### Browser contract and static app
- Purpose: verify that the Pages app calls the Python source of truth through a JSON-safe contract.
- Invariants: `tests/contracts/test_browser_contract.py` compares contract outputs to
  `predict_paco2_from_tcco2`, verifies likelihood-only default behavior, requires an explicit
  synthetic prior for prior-weighted mode, and exercises subgroups and uploaded study-table
  recomputation.
- Staging: `tests/contracts/test_stage_web_python.py` verifies package/data staging into
  `web/assets/` and confirms no PaCO2 prior is staged.
- E2E: `tests/e2e/test_web_app.py` verifies Pyodide loads, the default likelihood-only calculation
  completes, an uploaded synthetic prior enables prior-weighted mode, a missing prior fails closed,
  threshold changes update the browser result, and failed recalculation clears prior metrics, chart
  output, and result-provenance attributes before displaying the error.
- Scientific claim: browser-facing outputs are a serialization of the authoritative Python model, not
  a separate JavaScript implementation.
- Version gate: canonical parameter assets must contain exactly one current method revision and
  provisional status; missing, stale, or mixed metadata fail closed. The UI must keep the corrected
  method, independent-review-pending, frozen-manuscript, and research-only notice visible.
- Routing gate: grouped parameter inputs must contain the requested mapped group. Missing groups
  fail closed in Python and Pages, clear any prior browser result, and expose no pooled fallback.
  Successful outputs record `requested_group` and `parameter_group_used`; uploaded subgroup-specific
  study calculations are marked `single_model`.

## Artifact and release gates
- `docs/data_release_contract.json` rejects prohibited tracked paths and exact or reconstructable
  restricted-derived schemas, verifies that only Conway/bootstrap data are staged to Pages, and
  preserves exact hashes for the five canonical agreement artifacts and eight retained frozen
  downstream comparators. Alternate known output extensions and prohibited CSV/TSV/XLSX schemas
  are covered so a format change does not bypass the gate. Legacy binary `.xls` is unsupported and
  prohibited from the tracked tree because the locked environment cannot inspect it reliably.
- `docs/restricted_data_provenance.template.json` retains `HUMAN REVIEW REQUIRED` for unresolved
  ownership, IRB/protocol, waiver, DUA/authorization, extract, source-system, observation-unit,
  repeated-patient, permissions, and retention fields.
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
- Restricted-derived prior or exact-count regeneration is allowed only under `.pytest_tmp/`,
  `.tmp/`, or an explicitly approved external private workspace. The browser defaults to
  likelihood-only and does not stage or fetch such a prior.
- Retained rounded/aggregate downstream files remain frozen historical comparators, not
  release-approved outputs. No PaCO2-dependent result was regenerated, promoted, or unfrozen in
  this governance wave.
- Current-tree remediation does not rewrite Git history. Prior commits, tags, clones, caches, and
  historical deployments remain possible disclosure surfaces pending institutional review.
- Final tag, removal of provisional copy, manuscript unfreeze, submission-readiness claims, and final
  downstream promotion require independent biostatistical review plus resolution of the remaining
  source/reference and exact-count governance tickets.
