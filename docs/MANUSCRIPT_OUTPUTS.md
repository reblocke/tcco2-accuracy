# Manuscript outputs

## How to generate
- Promote only current public agreement artifacts with:
  `uv run python scripts/rebuild_artifacts.py --profile public-agreement --input-study-table Data/conway_studies.csv --out artifacts --seed 202401 --n-boot 1000 --bootstrap-mode cluster_plus_withinstudy`.
- The `artifacts/` destination enforces that exact source and settings. Build custom-input or
  sensitivity candidates under `.pytest_tmp/`, `.tmp/`, or another noncanonical destination.
- A future full restricted comparison requires an explicit approved source and private output, for
  example: `uv run python scripts/rebuild_artifacts.py --profile full --paco2-path /approved/restricted/source/in_silico_tcco2_db.dta --out /approved/private/workspace/tcco2-corrected-full --seed 202401 --n-boot 1000 --thresholds 45`.
- In-repository full output is accepted only under `.pytest_tmp/` or `.tmp/`; otherwise use an
  external private destination. The full profile is not a promotion command.
- Exact-count and reconstructable restricted-derived outputs, including normalized weights, must be
  written only to `.pytest_tmp/`, `.tmp/`, or an explicitly approved external private workspace.
- The full profile is not authorized for current-wave regeneration, promotion, or unfreezing. Use it
  only after the analysis specification, restricted-data authority, and independent-review gates are
  complete.

## Current status
- `artifacts/STATUS.md` is authoritative for corrected versus frozen outputs.
- `artifacts/manuscript_parameters.*` contains corrected-provisional agreement parameters under
  method revision `agreement_natural_log_tau2_direct_v1`.
- Every retained PaCO2-dependent aggregate remains frozen at the legacy method revision. Files
  removed under the exact-count governance contract are not public manuscript placeholders.
- Frozen values must not be copied into a final manuscript, described as corrected, or used to make
  submission-readiness claims. Independent biostatistical review and a separate downstream rebuild
  are required before unfreezing them.

## Current tracked manuscript-support files
- Error-model parameters: `artifacts/manuscript_parameters.csv` + `artifacts/manuscript_parameters.md`
  - Corrected-provisional; may support method review but not final manuscript promotion.
- Confusion matrix (rounded aggregate): `artifacts/manuscript_confusion_matrix.md`
  - Retained frozen legacy-method comparator; not release-approved.
- Two-stage strategy (rounded aggregate): `artifacts/manuscript_table2_two_stage.md`
  - Retained frozen legacy-method comparator; not release-approved.
- TcCO2 → PaCO2 prediction intervals: `artifacts/manuscript_table3_prediction_intervals.csv` +
  `artifacts/manuscript_table3_prediction_intervals.md`
  - Frozen legacy-method output.
- Results snippets: `artifacts/manuscript_results_snippets.md`
  - Frozen legacy-method output; do not copy into a current manuscript.

Also retained and frozen are `artifacts/simulation_summary.md` and
`artifacts/two_stage_summary.md`. `artifacts/STATUS.md` is the authoritative complete allowlist.

## Removed from the current tracked tree

- Operating-characteristic Table 1 CSV/Markdown.
- Exact/reconstructable confusion-matrix and two-stage CSVs.
- PaCO2 distribution, conditional-classification, misclassification-figure, and inference-demo
  outputs.

These files may be generated only in approved private output locations. Their removal is current-tree
remediation; prior commits, tags, clones, caches, and historical deployments remain possible
disclosure surfaces. See `docs/data_release_contract.json`.

## Interval definitions
- Parameter summaries (δ/σ²/τ²/LoA): 95% uncertainty interval (bootstrap percentile).
- Forward classification metrics and two-stage metrics: 95% CI (bootstrap percentile).
- Inference outputs: 95% prediction interval (PI).
