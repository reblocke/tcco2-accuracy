# Data Dictionary

This dictionary documents the public inputs, browser app uploads, aggregate
artifacts, and restricted local-only data boundaries used by this repository.
The machine-readable companion is `data_dictionary.csv`.

## Canonical Public Inputs

| Asset | Purpose | Public status |
| --- | --- | --- |
| `Data/conway_studies.csv` | Canonical study-level input table for the TcCO2-PaCO2 agreement model | Public |
| `Data/conway_studies.xlsx` | Spreadsheet copy of the canonical Conway study table | Public |
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

## Aggregate Artifacts

Small aggregate outputs under `artifacts/` support review and manuscript
workflows. Public aggregate examples include bootstrap parameters, classification
metrics, two-stage strategy summaries, prediction interval examples, and result
snippets. These should not contain patient-level rows, identifiers, exact
restricted-source counts, or small-cell reconstruction fields.

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
