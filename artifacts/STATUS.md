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
current for the corrected agreement method. The machine-readable current-tree allowlists and
prohibited schemas are in `docs/data_release_contract.json`.

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

## Removed restricted-derived outputs

The current tracked tree no longer contains the following exact-count or reconstructable
restricted-derived outputs:

- `paco2_distribution_summary.md`
- `conditional_classification_t45.csv` and `conditional_classification_t45.md`
- `figure_misclassification_vs_paco2.csv`
- `inference_demo.md`
- `manuscript_confusion_matrix.csv`
- `manuscript_table1.csv` and `manuscript_table1.md`
- `manuscript_table2_two_stage.csv`
- `two_stage_summary.csv`

The reconstructable normalized prior `Data/paco2_public_prior.csv` and its Pages copy were also
removed. The restricted-derived test fixture and aggregate `scripts/run_all_workflows.py` command
were retired so routine verification or deployment cannot recreate these files in the public tree.

## Retained frozen aggregates

These rounded or aggregate downstream files remain tracked as frozen historical comparators:

- `manuscript_confusion_matrix.md`
- `manuscript_results_snippets.md`
- `manuscript_table2_two_stage.md`
- `manuscript_table3_prediction_intervals.csv` and `manuscript_table3_prediction_intervals.md`
- `simulation_summary.md`
- `two_stage_summary.md`

They are not corrected-method outputs, release-approved results, submission-ready evidence, or
suitable for clinical decision-making. They must not be regenerated, promoted, or used to infer
the downstream effect of the wider corrected agreement distribution. Their current bytes are
SHA-256 locked in `docs/data_release_contract.json`, so accidental regeneration fails verification.
`conway_table1_fixture_summary.md` is separately preserved and hash-locked as an immutable
published/legacy comparator. `ui_overview.md` is
descriptive documentation, not a numerical validation artifact.

This is current-tree remediation only. No Git history was rewritten; prior commits, tags, clones,
caches, and historical deployments remain possible disclosure surfaces pending institutional
review.

## Promotion gates

- Pages may deploy the corrected browser calculation only with its visible
  provisional and research-only warning, likelihood-only default, and no staged PaCO2 prior.
- A final tag, removal of the warning, downstream artifact regeneration/promotion,
  manuscript unfreeze, and submission-readiness claims require independent
  biostatistical review, a locked analysis specification, and resolution of restricted-data
  provenance and historical-disclosure decisions.
