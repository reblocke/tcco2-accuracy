# Manuscript outputs

## How to generate
- Run: `python scripts/rebuild_artifacts.py --out artifacts --seed 202401 --n-boot 1000 --thresholds 45 --true-threshold 45 --two-stage-lower 40 --two-stage-upper 50 --tcco2-values 35,40,45,50,55`
- Outputs are written under `artifacts/`.

## Artifacts and manuscript placeholders
- Error-model parameters: `artifacts/manuscript_parameters.csv` + `artifacts/manuscript_parameters.md`
  - Fills the Results placeholder describing δ/σ²/τ² and marginal LoA.
- Operating characteristics by setting: `artifacts/manuscript_table1.csv` + `artifacts/manuscript_table1.md`
  - Fills the Results placeholder describing sensitivity/specificity/LR+/LR− and misclassification burden.
- Confusion matrix (per 1000 tested): `artifacts/manuscript_confusion_matrix.csv` + `artifacts/manuscript_confusion_matrix.md`
- Two-stage strategy: `artifacts/manuscript_table2_two_stage.csv` + `artifacts/manuscript_table2_two_stage.md`
  - Fills the Results placeholder for zone proportions, interval LRs, reflex ABG fraction, and residual errors.
- TcCO2 → PaCO2 prediction intervals: `artifacts/manuscript_table3_prediction_intervals.csv` +
  `artifacts/manuscript_table3_prediction_intervals.md`
  - Fills the Results placeholder with example prediction intervals and P(PaCO2≥T).
- Results snippets: `artifacts/manuscript_results_snippets.md`
  - Copy/paste blocks for manuscript Results placeholders.
- Figure data:
  - `artifacts/figure_paco2_distribution_bins.csv` (PaCO2 distributions by setting)
  - `artifacts/figure_misclassification_vs_paco2.csv` (misclassification vs true PaCO2)

## Interval definitions
- Parameter summaries (δ/σ²/τ²/LoA): 95% uncertainty interval (bootstrap percentile).
- Forward classification metrics and two-stage metrics: 95% CI (bootstrap percentile).
- Inference outputs: 95% prediction interval (PI).
