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

These files may be generated only in approved private output locations. Public branch and tag
history is checked against `docs/data_release_contract.json`; independently retained clones,
caches, and historical deployments outside repository-controlled refs may remain.

## Interval definitions
- Parameter summaries (δ/σ²/τ²/LoA): 95% uncertainty interval (bootstrap percentile).
- Forward classification metrics and two-stage metrics: 95% CI (bootstrap percentile).
- Inference outputs: 95% prediction interval (PI).

## November 2026 prediction letter

The frozen legacy catalog above describes the files in `artifacts/`. The current Word review package and results are in [Works in progress](../works-in-progress/README.md). The methods supplement describes the cohort, study observation model, hyperpriors, source differences, variance components and numerical checks.

Reproduce the empirical runs inside the private workspace on the mini. Install the private analysis dependencies with `uv sync --locked --group manuscript`. Use `uv run --locked --group manuscript python` to run the scripts below. The eligible-record stage selects source-recorded arterial LOINC 2019-8 and 32771-8 results from encounters matching the original patient/encounter keys and records the input fingerprints. The December Stata file supplies encounter covariates. Use the original All first-day workspace and its verified full-grid receipt as the baseline. The scripts accept explicit private paths and write their full outputs there.

1. Run `cohort.py ORIGINAL STAGE ARCHIVED_DTA PRIVATE_OUTPUT` from `scripts/manuscript/`. The stage contains the arterial candidate parquet and its receipt. Its parent is the original cohort kit with keyed membership parquet. The versioned fingerprints in `private_inputs.py` bind the original private provenance record and manuscript stage receipt before any file digests are accepted. Both stages must identify the same archived Stata member, source and weighting policy. SQL and pandas independently construct the patient-weighted distribution within each setting. Eligibility counts precede first-day selection.
2. Place the accepted ICU, ARF and LFT posterior files under their stratum names in the private output directory. Use the public-adjudicated-v2 manifest and the extended, verified All posterior in the original workspace. Run `predictions.py PRIVATE_OUTPUT ORIGINAL MANIFEST`. The grid evaluator uses the mini’s Accelerate normal-mixture library from the original verified grid; its source is `mixture.c` (build with `clang -O3 -dynamiclib -framework Accelerate mixture.c -o mixture.dylib`). It integrates over the observed arterial support at quadrature orders 32 and 64, preserving contiguous chain batches for MCSE. It rechecks the All results including verified medians and stores numerical evidence privately for all four populations.
3. Run `strategy.py PRIVATE_OUTPUT ORIGINAL MANIFEST` for draw-specific forward classification, followed by `exhibits.py PRIVATE_OUTPUT PRIVATE_EXHIBITS`. Inspect the two rendered PNGs and the two Markdown summary tables before transferring those four exhibits. The full grids, numerical evidence and empirical weights belong in the private output directory. Rebuilding legacy artifacts is a separate workflow.

Review scope includes source and model checks, private calculations and rendering. Independent biostatistical review, clinical validation and journal submission remain open.

Run `verify_grid.py PRIVATE_OUTPUT ORIGINAL MANIFEST` on the mini to independently recompute every integer TcCO₂ reading from 20 through 100 mmHg in all four settings (324 rows). It uses the full arterial support and posterior chains, ordinary-scale Gaussian quadrature, and separate calculations of threshold masses and chain-batch errors. It compares both quadrature orders, each median and interval, both inclusive threshold probabilities, and every decision to withhold an estimate. The 47 mmHg reading remains a worked example. The portable CI acceptance test exercises known mixture predictions across the full grid and detects incorrect holds or missing rows. Run `validate.py PATH_TO_MIXTURE_DYLIB` for the analytic mixture and ABG-referral cases. Public fit provenance is in `models/pages/summary.json`; the All precision extension and original grid receipt are on the mini.
