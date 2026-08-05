# Data Dictionary

This dictionary documents the public inputs, browser app uploads, aggregate
artifacts, and restricted local-only data boundaries used by this repository.
The machine-readable companion is `data_dictionary.csv`.

## Canonical Public Inputs

| Asset | Purpose | Public status |
| --- | --- | --- |
| `Data/conway_studies.csv` | Operational canonical promotion and browser-staging source | Public |
| `Data/conway_studies.xlsx` | Human-editable review mirror of the canonical CSV | Public |
| `Data/conway_studies_template.xlsx` | Template for future study additions | Public |
| `Data/data_counts.csv` | Source-derived count fallback for Conway export workflows | Public, review provenance before reuse |
| `Data/paco2_public_prior.csv` | Weight-only PaCO2 prior used by the browser app | Public |

The Conway study table follows `docs/CONWAY_DATA_SCHEMA.md`. The public PaCO2
prior keeps 1 mmHg bins and normalized weights only; exact bin counts are not
included.

## Conway Study Table Fields

| Field | Definition | Validation |
| --- | --- | --- |
| `study_id` | Unique study or cohort label | Non-empty and unique |
| `bias` | Mean PaCO2 minus TcCO2 difference | Finite numeric, mmHg |
| `sd` | Within-study SD of differences | Positive if present |
| `s2` | Within-study variance of differences | Positive if present |
| `n_pairs` | Number of paired PaCO2/TcCO2 measurements | Positive integer |
| `n_participants` | Number of participants contributing pairs | Positive integer |
| `c` | Average repeated measurements per participant | Positive if present |
| `is_icu` | ICU subgroup membership | Boolean-like |
| `is_arf` | Acute respiratory failure subgroup membership | Boolean-like |
| `is_lft` | Ambulatory/PFT subgroup membership | Boolean-like |

## Public Prior Fields

| Field | Definition | Validation |
| --- | --- | --- |
| `group` | Clinical setting group | `all`, `pft`, `ed_inp`, or `icu` |
| `paco2_bin` | PaCO2 bin value | Finite numeric, mmHg |
| `weight` | Normalized public prior weight | Nonnegative; should sum approximately 1 within group |

## Browser App Uploads

The app accepts optional CSV/XLSX uploads for a custom Conway-compatible study
table and a custom binned PaCO2 prior. These files are parsed in the browser and
passed to the Pyodide worker. They are not sent to a backend, persisted, logged,
or encoded in URLs.

## Parameter Routing Fields

| Field | Definition | Validation |
| --- | --- | --- |
| `requested_group` | PaCO2 setting requested by the calculation | Must match the analysis subgroup |
| `parameter_group_used` | Conway group actually supplying parameters | Mapped group, explicit `main`, or `single_model` |

Grouped inputs fail closed when the mapped parameter group is missing. `main` is used only through
an explicit fallback request; ungrouped inputs are recorded as `single_model`.

## Aggregate Artifacts

Small aggregate outputs under `artifacts/` support review and manuscript
workflows. `artifacts/STATUS.md` is the authority for whether each output is
corrected-provisional or frozen at the legacy agreement-method revision. The
canonical bootstrap parameters include `agreement_method_version` and
`results_status`; browser staging rejects missing, stale, or mixed values.
Canonical promotion also requires the locked source, seed, draw count, and bootstrap mode in
`artifacts/STATUS.md`; custom candidates must be written outside `artifacts/`.

Only public Conway-derived agreement outputs are promoted in the current wave.
Classification metrics, two-stage summaries, prediction intervals, confusion
matrices, and result snippets remain frozen until corrected downstream regeneration
and governance review. Aggregate outputs must not contain patient-level rows,
identifiers, exact restricted-source counts, or small-cell reconstruction fields.

| Asset | Purpose | Current status |
| --- | --- | --- |
| `artifacts/STATUS.md` | Current/frozen artifact manifest and promotion gates | Current documentation |
| `artifacts/bootstrap_params.csv` | Canonical browser bootstrap parameters | Corrected-provisional |
| PaCO2-dependent artifacts listed in `artifacts/STATUS.md` | Simulation, inference, classification, and manuscript review outputs | Frozen legacy-method |

## Restricted Or Local-Only Assets

| Asset | Rule |
| --- | --- |
| `Data/in_silico_tcco2_db.dta` | Local restricted source input; never track |
| `Data/In Silico TCCO2 Database.dta` | Alternate local restricted source filename; never track |
| `Data/paco2_prior_bins.csv` | Exact count-bearing prior output; keep local/generated unless explicitly approved |
| `artifacts/figure_paco2_distribution_bins.csv` | Exact count-bearing figure data; keep local/generated unless explicitly approved |
| `web/assets/data/` | Generated staged data for Pages; do not hand-edit |
| `web/assets/py/` | Generated staged Python package for Pyodide; do not hand-edit |

## Review Flags

- `Data/data_counts.csv` is public and useful for reproducibility, but source
  provenance should be checked before reuse outside this repository.
- The manuscript has not yet been submitted; do not publish full manuscript text
  or internal draft files as machine-readable surfaces.
- The static app is research software only and is not intended for clinical
  decision-making.
- Corrected browser outputs remain provisional pending independent biostatistical
  review; frozen downstream artifacts must not be described as corrected.
