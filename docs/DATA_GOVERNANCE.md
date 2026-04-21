# Data Governance

## Scope
This repository is public-facing research software. It includes source code,
reference study summaries, a static browser app, and small derived outputs. It
does not include patient-level protected health information (PHI), and it should
not include exact count-bearing tables derived from restricted local PaCO2 data
unless that release is explicitly approved.
Treat the local PaCO2 source distribution and count-derived outputs as
data-use-agreement-sensitive unless institutional review explicitly says
otherwise.

## Public Assets
- `Data/conway_studies.csv` and `Data/conway_studies.xlsx` contain reference
  Conway study summaries used by the model.
- `Data/paco2_public_prior.csv` contains the public PaCO2 prior used by the
  browser app. It keeps 1 mmHg bins and normalized `weight` values only.
- `artifacts/bootstrap_params.csv` contains canonical bootstrap parameters for
  responsive browser defaults.
- Descriptive summaries such as `artifacts/paco2_distribution_summary.md` may be
  tracked when they do not expose exact restricted-data bin counts.

## Restricted Or Local-Only Assets
- Local `.dta` files, including `Data/in_silico_tcco2_db.dta` and
  `Data/In Silico TCCO2 Database.dta`, are restricted source inputs and must not
  be committed.
- `Data/paco2_prior_bins.csv` is an exact count-bearing prior output and must
  remain local/generated unless explicitly approved.
- `artifacts/figure_paco2_distribution_bins.csv` is an exact count-bearing
  manuscript figure-data output and must remain local/generated unless explicitly
  approved.
- Any table with exact prior counts, raw rows, small-cell indicators, or
  count-reconstructing columns from restricted data should stay in `.pytest_tmp/`,
  `.tmp/`, or a private manuscript workspace.

## Workflow Rules
- Browser and Pages assets must be staged from canonical sources with
  `make stage-web`; do not hand-edit generated files under `web/assets/`.
- The browser prior is public and weight-only. Local manuscript workflows may use
  exact restricted data through `scripts/rebuild_artifacts.py --paco2-path PATH`.
- Exact count-bearing outputs should be generated to scratch or private paths,
  for example:

```bash
uv run python scripts/rebuild_artifacts.py --out .pytest_tmp/exact-artifacts --paco2-path Data/in_silico_tcco2_db.dta --seed 202401 --n-boot 1000 --thresholds 45
```

- The static app should remain client-side only: no backend, telemetry,
  persistence, or patient-value URL state unless a future change explicitly
  documents the data path and compliance assumptions.
- Current-tree remediation is the active decision. Git history rewriting is out
  of scope for this pass.
