# Artifact status

Agreement method revision: `agreement_natural_log_tau2_direct_v1`
Result status: `provisional`

## Canonical promotion contract

- profile: `public-agreement`
- output: `artifacts/`
- input-study-table: `Data/conway_studies.csv`
- seed: `202401`
- n-boot: `1000`
- bootstrap-mode: `cluster_plus_withinstudy`

`scripts/rebuild_artifacts.py` rejects any other input or setting when the output
resolves to `artifacts/`. Custom inputs and sensitivity settings must write to a
noncanonical scratch destination such as `.pytest_tmp/public-agreement-candidate/`.
This manifest remains hand-authored because it records scientific status and
governance decisions in addition to executable settings.

Independent biostatistical review is pending. Only the five artifacts below are
current for the corrected agreement method. All other numerical artifacts remain
frozen at the legacy method revision and must not be described as corrected,
submission-ready, or suitable for clinical decision-making.

## Corrected-provisional artifacts

| Artifact | Role |
| --- | --- |
| `meta_loa_check.md` | Corrected analytic agreement estimates and comparison with published legacy values |
| `bootstrap_params.csv` | Canonical corrected bootstrap draws staged for Pages |
| `bootstrap_summary.md` | Corrected bootstrap spread versus corrected analytic intervals |
| `manuscript_parameters.csv` | Corrected agreement-parameter review table |
| `manuscript_parameters.md` | Human-readable corrected agreement-parameter review table |

These files are regenerated with:

```bash
uv run python scripts/rebuild_artifacts.py --profile public-agreement --input-study-table Data/conway_studies.csv --out artifacts --seed 202401 --n-boot 1000 --bootstrap-mode cluster_plus_withinstudy
```

## Frozen legacy-method artifacts

The following files are intentionally not regenerated in the corrected agreement
wave because they depend on the PaCO2 distribution, downstream simulation or
inference, or manuscript reporting workflows:

- `paco2_distribution_summary.md`
- `simulation_summary.md`
- `inference_demo.md`
- `conditional_classification_t45.csv` and `conditional_classification_t45.md`
- `figure_misclassification_vs_paco2.csv`
- `manuscript_confusion_matrix.csv` and `manuscript_confusion_matrix.md`
- `manuscript_table1.csv` and `manuscript_table1.md`
- `manuscript_table2_two_stage.csv` and `manuscript_table2_two_stage.md`
- `manuscript_table3_prediction_intervals.csv` and `manuscript_table3_prediction_intervals.md`
- `two_stage_summary.csv` and `two_stage_summary.md`
- `manuscript_results_snippets.md`

`conway_table1_fixture_summary.md` is separately preserved as an immutable
published/legacy comparator. `ui_overview.md` is descriptive documentation, not a
numerical validation artifact.

Some frozen tracked outputs contain exact setting counts. Their deletion,
coarsening, or public-release approval is a separate governance decision; until
that decision is resolved, the repository as a whole must not be called public-safe.

## Promotion gates

- Pages may deploy the corrected browser calculation only with its visible
  provisional and research-only warning.
- A final tag, removal of the warning, downstream artifact regeneration/promotion,
  manuscript unfreeze, and submission-readiness claims require independent
  biostatistical review and resolution of the remaining source/reference and
  exact-count governance tickets.
