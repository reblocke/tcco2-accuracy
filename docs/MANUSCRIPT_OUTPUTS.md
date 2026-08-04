# Manuscript outputs

## How to generate
- Promote only current public agreement artifacts with:
  `uv run python scripts/rebuild_artifacts.py --profile public-agreement --input-study-table Data/conway_studies.csv --out artifacts --seed 202401 --n-boot 1000 --bootstrap-mode cluster_plus_withinstudy`.
- The `artifacts/` destination enforces that exact source and settings. Build custom-input or
  sensitivity candidates under `.pytest_tmp/`, `.tmp/`, or another noncanonical destination.
- Run a full restricted comparison only with an explicit local `.dta` and a scratch/private output,
  for example: `uv run python scripts/rebuild_artifacts.py --profile full --paco2-path Data/in_silico_tcco2_db.dta --out .pytest_tmp/tcco2-corrected-full --seed 202401 --n-boot 1000 --thresholds 45`.
- In-repository full output is accepted only under `.pytest_tmp/` or `.tmp/`; otherwise use an
  external private destination. The full profile is not a promotion command.
- Exact count-bearing outputs such as PaCO2 prior bins or distribution figure
  bins should be written to `.pytest_tmp/`, `.tmp/`, or a private manuscript
  workspace unless explicitly approved for public release.

## Current status
- `artifacts/STATUS.md` is authoritative for corrected versus frozen outputs.
- `artifacts/manuscript_parameters.*` contains corrected-provisional agreement parameters under
  method revision `agreement_natural_log_tau2_direct_v1`.
- Every PaCO2-distribution-dependent table, figure dataset, confusion matrix, prediction interval,
  two-stage result, and results snippet remains frozen at the legacy method revision.
- Frozen values must not be copied into a final manuscript, described as corrected, or used to make
  submission-readiness claims. Independent biostatistical review and a separate downstream rebuild
  are required before unfreezing them.

## Artifacts and manuscript placeholders
- Error-model parameters: `artifacts/manuscript_parameters.csv` + `artifacts/manuscript_parameters.md`
  - Corrected-provisional; may support method review but not final manuscript promotion.
- Operating characteristics by setting: `artifacts/manuscript_table1.csv` + `artifacts/manuscript_table1.md`
  - Frozen legacy-method output.
- Confusion matrix (per 1000 tested): `artifacts/manuscript_confusion_matrix.csv` + `artifacts/manuscript_confusion_matrix.md` (frozen legacy-method output)
- Two-stage strategy: `artifacts/manuscript_table2_two_stage.csv` + `artifacts/manuscript_table2_two_stage.md`
  - Frozen legacy-method output.
- TcCO2 → PaCO2 prediction intervals: `artifacts/manuscript_table3_prediction_intervals.csv` +
  `artifacts/manuscript_table3_prediction_intervals.md`
  - Frozen legacy-method output.
- Results snippets: `artifacts/manuscript_results_snippets.md`
  - Frozen legacy-method output; do not copy into a current manuscript.
- Figure data:
  - `artifacts/figure_paco2_distribution_bins.csv` (PaCO2 distributions by setting; restricted local/generated output, not tracked)
  - `artifacts/figure_misclassification_vs_paco2.csv` (misclassification vs true PaCO2)

## Interval definitions
- Parameter summaries (δ/σ²/τ²/LoA): 95% uncertainty interval (bootstrap percentile).
- Forward classification metrics and two-stage metrics: 95% CI (bootstrap percentile).
- Inference outputs: 95% prediction interval (PI).
