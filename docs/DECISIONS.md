# TcCO2 Accuracy — Decisions

## Open decisions
- Independent biostatistical review is required before removing provisional warnings,
  unfreezing downstream manuscript outputs, creating a final release tag, or making
  submission-readiness claims.
- The distinction between 73 reported studies and 76 modeled effect rows, together with
  any estimator, clustering, or estimand changes, remains outside the TCCO2-003/004
  correction. The documented downstream implementation contract records the code design;
  any separate review is outside this repository implementation.

## Workflow and monorepo stabilization
- The repository remains a monorepo for this wave: Python package, static browser app, source/reference
  data, artifacts, and manuscript drafts stay in one repository with nested `AGENTS.md` guardrails
  where rules differ.
- `pyproject.toml` and `uv.lock` are authoritative for Python dependencies. Use `uv sync --locked`
  and the root `Makefile` command surface for local and CI verification.
- The active package/test layout is root `src/` and root `tests/`; the prior nested `python/`
  package layout has been retired.
- The public app deployment target is static GitHub Pages. See `docs/adr/0001-streamlit-to-pages.md`.
- Browser computation loads staged Python through Pyodide in a worker. JavaScript manages UI,
  uploads, and plotting but does not duplicate the statistical model.
- Historical behavior (superseded 2026-08-05 below): browser default inference staged
  `Data/conway_studies.csv`, `Data/paco2_public_prior.csv`, and `artifacts/bootstrap_params.csv`;
  the weight-only prior was treated as public because exact count columns were omitted.
- 2026-08-05: Browser default inference is likelihood-only and stages only the Conway study table
  and canonical bootstrap parameters. Prior-weighted inference requires a user-supplied binned-prior
  upload that remains client-side; the app neither stages nor fetches a repository PaCO2 prior.
- 2026-08-05: Exact counts and normalized weights derived from the restricted PaCO2 source are both
  restricted-derived because weights may reconstruct the exact histogram. The current tree removes
  `Data/paco2_public_prior.csv` and prohibited count-bearing/reconstructable outputs. Private
  regeneration is limited to `.pytest_tmp/`, `.tmp/`, or an explicitly approved external private
  workspace.
- 2026-08-05: `docs/data_release_contract.json` established the machine-readable current-tree and
  Pages authority. It was extended to public-history verification on 2026-08-06.
  `docs/restricted_data_provenance.template.json` is the unresolved human-review record; unknown
  fields remain `HUMAN REVIEW REQUIRED`. Known restricted-output stems are blocked across file
  formats, structured schema checks include BOM-safe CSV/TSV and every XLSX sheet, and retained
  frozen aggregates plus the synthetic prior fixture are hash-locked against accidental
  regeneration. Legacy binary `.xls` is unsupported and prohibited from tracking.
- 2026-08-05: Restricted PaCO2 package and workflow loaders no longer auto-discover repository-local
  extracts. Callers must supply in-memory `paco2_data` or an explicit private source path.
- 2026-08-05: This wave did not rewrite Git history. This decision is superseded by the 2026-08-06
  public-reference remediation below.
- 2026-08-06: Public branch and tag history was rewritten to remove prohibited restricted-derived
  blobs. `scripts/check_public_history.py` continuously scans public refs in CI and Pages, with
  exact path-and-hash exceptions only for approved artifacts. The baseline and archive marker names
  are retained as sanitized tags. Independently retained clones, caches, and historical deployments
  outside repository-controlled refs are not asserted removed.
- 2026-08-07: `actions/deploy-pages@v5` is the required official Node 24-native Pages action, but
  its current released bundle emits Node `DEP0040` by importing deprecated `punycode`
  ([upstream issue 413](https://github.com/actions/deploy-pages/issues/413)). The deployment step
  sets `NODE_OPTIONS=--disable-warning=DEP0040` only for that action until an official fixed release
  is available. It intentionally does not suppress other warning codes, all deprecations, or all
  process warnings.
- 2026-08-07: Commit `12062da` passed the x86 CI and matching Pages gates. This completes the
  corrected-provisional engineering wave but does not satisfy independent statistical review.
  TCCO2-002 source recovery is complete through source-linked, checksummed records for the
  author-supplied Tipton-Shuster R supplement and the Conway application archive in
  `Data/PROVENANCE.md`. Its deliberate link-and-checksum-only non-retention disposition is
  complete. TCCO2-005 has an executable equation-derived reference comparison; TCCO2-007 external
  review remains pending. TCCO2-006 is intentionally phased: agreement recomputation is complete
  and hash-locked, and the new downstream code remains in-memory and aggregate-only. These project
  states stay in `docs/PLAN.md`; `docs/data_release_contract.json` remains limited to enforceable
  release and data boundaries rather than duplicating a project-management register.
- 2026-08-07: A read-only audit of the full private workflow found that explicit restricted-input,
  private-output, and no-public-promotion guards are already sufficient. The in-memory TCCO2-006
  code implements draw-aligned agreement/target resampling, deterministic publication clustering,
  patient clustering, a non-sensitive manifest, an enforced minimum of 10,000 draws, and target-
  scale Monte Carlo stability checks. The legacy full profile remains unchanged.
- 2026-08-07: Superseding the earlier pending TCCO2-002 archival note, the repository records a
  link-and-checksum-only disposition for the author-supplied Tipton-Shuster R supplement. The
  repository records its authoritative URLs, 2,521-byte size, and SHA-256 but deliberately retains
  no repository or private copy because no explicit redistribution license was identified. This
  completes TCCO2-002 without creating an unlicensed mirror.
- 2026-08-07: The downstream implementation contract in `docs/SPEC.md` defines the in-memory code
  choices: caller-supplied eligible records; earliest encounter then earliest measurement selection;
  deterministic 73-publication-cluster agreement resampling; truth-pattern-stratified patient
  resampling; current setting mapping; all-positive-value support with a central-95% sensitivity;
  no aggregate-data proportional-bias model; specified 45 mmHg and two-stage outputs; percentile
  intervals; and at least 10,000 draws with bounded Monte Carlo stability repeats. This records
  code requirements only; source-data use, publication, result promotion, manuscript status, and
  any external review are handled outside this repository implementation.
- 2026-08-07: Superseding only the downstream resampling and stability details in the preceding
  decision, the source-like new-patient estimand uses ordinary patient-cluster resampling rather than
  fixed truth-pattern strata. Setting-specific analyses select an index record within setting, pooled
  All selects one index record overall, and truth-class-degenerate proposals are redrawn subject to a
  100-attempt and 1%-per-setting limit. Monte Carlo precision uses one predetermined independent
  repeat and an all-output combined-MCSE gate; two-MCSE repeat agreement is descriptive. The new API
  requires a non-sensitive target-data revision label, rejects factorial sensitivities, returns
  explicit bootstrap quantiles/probability names, and labels noncontract development runs.
- 2026-08-07: Downstream manifests accept `target_data_revision` only as a bounded opaque ASCII
  token, rejecting paths, free text, and control characters before manifest construction. Token
  syntax reduces accidental disclosure but does not replace the caller's responsibility to exclude
  patient or other sensitive identifiers.
- 2026-08-07: Seeded downstream runs canonicalize patient clusters, values within patient clusters,
  and prepared Conway effects before resampling, making results invariant to caller row order while
  preserving repeated measurements and multi-effect publications. Numeric order normalization
  retains exact integer precision; custom patient column roles and workflow controls fail closed if
  semantically aliased or incorrectly typed. The generic agreement bootstrap and protected artifacts
  are unchanged.
- Pyodide 0.29.0, Plotly.js 2.35.2, and SheetJS 0.18.5 are pinned CDN browser dependencies.
- User-entered values and uploads remain client-side; the app has no backend, telemetry,
  persistence, or patient-value URL state.
- Generated `.pytest_tmp/`, `.tmp/`, and `*.egg-info/` outputs are not source artifacts and should
  not be tracked.
- Generated static app assets under `web/assets/py/` and `web/assets/data/` are not tracked; Pages,
  E2E tests, and local serving regenerate them from `src/`, `Data/`, and `artifacts/` with
  `scripts/stage_web_python.py`.
- Visual QA screenshots are generated only on request with `make visual-qa` under
  `.pytest_tmp/visual-qa/`; they are not part of `make verify`.
- The browser posterior chart uses a posterior-focused x-axis to keep the displayed distribution
  readable when the prior has a long tail. This does not change posterior/prior arrays or numeric
  summaries.
- Prior-weighted browser plots include a Python-computed `Likelihood (scaled)` curve normalized to
  bin mass for visual comparison after an explicit prior upload; likelihood-only mode omits it
  because it duplicates the displayed distribution shape. This does not change numeric summaries.
- Browser UI copy avoids clinical correctness wording and reports threshold classification with
  posterior mass summaries because the app is a research interpretation tool, not clinical decision
  support.
- Browser initialization and recalculation failures fail closed: before an error is displayed, the
  app removes prior result-provenance attributes, hides and resets metrics, and purges the prior
  Plotly chart so a failed custom run cannot appear to retain a valid result.
- Agreement calculations, canonical browser parameters, and public agreement artifacts carry
  method revision `agreement_natural_log_tau2_direct_v1` and status `provisional`. Pages may deploy
  this corrected provisional method, but independent biostatistical review remains a final-release
  and manuscript-unfreeze gate.
- Public artifact regeneration uses the explicit `public-agreement` profile and public Conway data
  only. Full PaCO2-dependent regeneration requires an explicit restricted input and a scratch/private
  output directory. In-repository full output is permitted only below `.pytest_tmp/` or `.tmp/`;
  otherwise the destination must be an external private path.
- Retained rounded/aggregate PaCO2-dependent artifacts are frozen historical comparators, not
  corrected or release-approved outputs. They remain frozen; this governance correction does not
  regenerate, promote, or unfreeze them.
- Canonical promotion to repository `artifacts/` is additionally locked to
  `Data/conway_studies.csv`, seed 202401, 1,000 draws per subgroup, and
  `cluster_plus_withinstudy`. Path aliases resolving to those canonical locations are accepted;
  custom studies or settings must write to scratch. `artifacts/STATUS.md` remains the hand-authored
  authority because scientific status and frozen-output decisions cannot be inferred from CLI args.
  Promotion snapshots the complete five-file destination state before the first replacement. If
  any replacement fails, all five destinations are rolled back byte-for-byte and files absent
  before the attempt are removed; unrelated manifest and frozen files are never part of promotion.
- Pure numerical code is separated under `src/tcco2_accuracy/core/`; top-level modules remain
  compatibility wrappers for existing public imports.
- Malformed continuity-ledger paths are retired; durable project decisions belong in this file or
  `docs/adr/`.

## Logged divergences
- Superseding the legacy reproduction decision, `logs2` inputs now use the coherent natural-log
  expression `log(S2*) + 1/(n_2 - 1)` with `v_logs2 = 2/(n_2 - 1)` and an `exp()`
  back-transform, following Tipton and Shuster equations 4.4-4.5. The Figshare RData and inherited
  Stata/Python calculation used `log10` with natural-log correction and back-transformation; those
  values are retained only as frozen published/legacy comparators.
- Analytic LoA uncertainty now treats `2 / sum((v_bias + tau2)^-2)` as direct-scale `Var(tau2)`
  and applies the equation 4.16 coefficient `1/(sigma2 + tau2)`. The inherited implementation
  instead paired that direct-scale variance with the log-scale coefficient
  `tau2^2/(sigma2 + tau2)`.
- Bootstrap τ² draws are truncated at 0 to enforce non-negative between-study variance for simulation/inference draws in `src/tcco2_accuracy/core/bootstrap.py`.
- Bootstrap workflows default to `cluster_plus_withinstudy` to align outer CI scale with Conway, while low-level bootstrap functions default to `cluster_only` in `src/tcco2_accuracy/workflows/bootstrap.py` and `src/tcco2_accuracy/core/bootstrap.py`.
- Hybrid bootstrap perturbations treat bias and log-variance inputs as independent due to missing covariance data in `src/tcco2_accuracy/core/bootstrap.py`.
- Historical provisional choice (superseded 2026-08-04 below): meta-analysis τ² retained
  untruncated analytic behavior for published/legacy-comparator review.
- 2026-08-04: Production `loa_summary`, `conway_group_summary`, and meta workflow calculations
  zero-truncate a negative method-of-moments τ² estimate before random-effects weights, LoA, and
  equation 4.13. `random_effects_meta(..., truncate_tau2=False)` retains the raw estimate as a
  diagnostic-only path. Confidence-interval behavior at and near the τ²=0 boundary remains
  provisional pending independent biostatistical review.
- LoA confidence intervals are undefined for single-study summaries, so CI bounds are returned as NaN when df ≤ 0 in `src/tcco2_accuracy/core/conway_meta.py`.
- Main-analysis descriptive counts aggregate by `study_base` (strip trailing parentheses) and treat identical-bias multi-row citations as overlapping cohorts (use max counts) in `src/tcco2_accuracy/core/conway_meta.py`.
- PaCO2 subgroup assignment follows `docs/SPEC.md:20-27`, which includes ED in `ed_inp` by construction; this differs from `Code/2_trinetx_cleaning_do.do:8-13`, where `ed_inp_group` excludes ED (`is_emer==0`) and requires `cc_time==0`.
- PaCO2 subgroups map to Conway bootstrap groups via the shared helper in `src/tcco2_accuracy/core/_params.py` (pft→lft, ed_inp→arf, icu→icu, all→main) to align ambulatory and acute respiratory failure sub-analyses.
- Simulation/inference parameter validation requires finite numeric values with non-negative σ² and τ² in `src/tcco2_accuracy/core/utils.py`.
- Historical behavior (superseded 2026-08-05 below): missing subgroup parameters fell back to all
  parameter rows with a warning.
- 2026-08-05: Grouped simulation, inference, browser, conditional, and manuscript calculations fail
  closed when the resolved Conway parameter group is missing. An explicit `fallback="main"` option
  selects only the pooled `main` group; it never combines unrelated group rows. Ungrouped tables are
  treated as an explicitly supplied single model. Downstream results record the requested and used
  parameter groups.
- 2026-08-05: Analytic survival probabilities now use the normal `sf` directly; modeled
  likelihood ratios aggregate `logsf`/`logcdf` probabilities with `logsumexp`, and two-stage
  middle-zone probabilities use a stable interval calculation. This prevents representable
  extreme tails from becoming zero or infinity through `1-cdf` subtraction alone.
- 2026-08-05: Conditional curves assign truth and calculate test-positive probability from original
  unbinned PaCO2 values before aggregation. Production defaults use half-open bins
  `[paco2_bin, paco2_bin_upper)`; legacy round/floor grouping may alter display bins but never
  truth. These corrections apply to future private downstream rebuilds only; frozen tracked
  PaCO2-dependent artifacts were not regenerated or promoted.
- 2026-08-05: Scientific inputs now fail closed on empty study tables or subgroup selections,
  invalid study identifiers/counts/repeated-measure ratios, inconsistent variance fields, and
  nonpositive or nonfinite retained PaCO2 supports and thresholds. Genuine missing distribution
  rows are dropped; no unsupported PaCO2 upper bound is imposed. Conway study flags may overlap,
  while record-level PaCO2 groups remain mutually exclusive; two-stage zone boundaries need only
  be finite and strictly ordered and may be negative. Frozen PaCO2-dependent outputs remain frozen.
- 2026-08-05: Prepared PaCO2 inputs drop genuinely missing PaCO2 rows first; every retained row
  requires one of the three record-level subgroup labels. Raw assignment flags are binary when
  present; supplied binned priors require exactly all four allowed group labels. Conway export never permits
  missing/blank identifiers or non-integer observed counts, even when other source checks are
  configured as permissive.
- `format_inference_demo` only supports a single threshold and raises a ValueError otherwise in `src/tcco2_accuracy/workflows/infer.py`.
- Legacy Conway study exports read bias/S2 and subgroup membership from the RData objects (`main`, `ICU`, `ARF`, `LFT`) and merge counts from `data.dta` (or `data_counts.csv` fallback); the Bolliger ICU row uses a 49/49/1 count fallback with bias/S2 pulled from `ICU` when absent from `main` to preserve the published/legacy source export (`scripts/export_conway_rdata.py:29`, `scripts/export_conway_rdata.py:107`).
