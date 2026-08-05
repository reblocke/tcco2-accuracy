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

The RData/count exporter rejects missing or blank study identifiers and non-integer observed
counts even when `--no-strict` or `--allow-missing-counts` is selected.

## Conway Study Table Fields

| Field | Definition | Validation |
| --- | --- | --- |
| `study_id` | Unique study or cohort label | Non-empty and unique after surrounding whitespace is stripped |
| `bias` | Mean PaCO2 minus TcCO2 difference | Finite numeric, mmHg |
| `sd` | Within-study SD of differences | Supplied column complete, finite, and positive; `sd²` consistent with supplied `s2` at `rtol=1e-10`, `atol=1e-12` |
| `s2` | Within-study variance of differences | Supplied column complete, finite, and positive; consistent with supplied `sd²` at `rtol=1e-10`, `atol=1e-12` |
| `n_pairs` | Number of paired PaCO2/TcCO2 measurements | Finite integer, `n_pairs >= n_participants` |
| `n_participants` | Number of participants contributing pairs | Finite integer greater than 1 |
| `c` | Average repeated measurements per participant | Optional; finite, at least 1, and consistent with `n_pairs / n_participants` at `rtol=1e-10`, `atol=1e-12` |
| `is_icu` | ICU subgroup membership | Boolean-like, non-missing; may overlap other Conway flags |
| `is_arf` | Acute respiratory failure subgroup membership | Boolean-like, non-missing; may overlap other Conway flags |
| `is_lft` | Ambulatory/PFT subgroup membership | Boolean-like, non-missing; may overlap other Conway flags |

The table and every requested subgroup analysis must be non-empty. Overlapping Conway study flags
are intentional and differ from the mutually exclusive record-level PaCO2 distribution groups.

## Public Prior Fields

| Field | Definition | Validation |
| --- | --- | --- |
| `group` | Clinical setting group | After normalization, exactly all four labels `pft`, `ed_inp`, `icu`, and `all`; no blanks, missing groups, or extras |
| `paco2_bin` | PaCO2 prior support value | Finite and strictly positive, mmHg; no hard upper cutoff |
| `weight` | Normalized public prior weight | Nonnegative; should sum approximately 1 within group |

## Scientific Numeric Inputs

| Field | Definition | Validation |
| --- | --- | --- |
| `paco2` | Retained PaCO2 observation | Finite and strictly positive, mmHg; no hard upper cutoff |
| prepared `subgroup` | Mutually exclusive PaCO2 record group | For each retained PaCO2 row, non-missing normalized `pft`, `ed_inp`, or `icu`; `all` is not a record label |
| `is_amb`, `is_emer`, `is_inp`, `cc_time` | Raw subgroup-assignment flags | Each non-missing value numeric binary 0/1; genuinely missing values become 0 |
| `threshold` / `true_threshold` | PaCO2 classification threshold | Finite and strictly positive, mmHg |
| `lower` / `upper` | Two-stage TcCO2 zone boundaries | Both finite and `lower < upper`; negative boundaries are allowed |

Genuinely missing PaCO2 source-distribution rows are dropped before retained values are validated.
Malformed, infinite, zero, or negative retained values fail closed. No evidence-supported hard upper
PaCO2 range has been established.

## Conditional Curve Fields

| Field | Definition | Validation |
| --- | --- | --- |
| `paco2_bin` | Inclusive lower edge of a conditional-curve display bin | Finite numeric, mmHg |
| `paco2_bin_upper` | Exclusive upper edge of that display bin | Finite numeric greater than `paco2_bin` |

Conditional TN/FP/FN/TP truth and test-positive probabilities are computed from original unbinned
PaCO2 values and then aggregated. The production bin contract is
`[paco2_bin, paco2_bin_upper)`; display grouping never determines truth.

## Browser App Uploads

The app accepts optional CSV/XLSX uploads for a custom Conway-compatible study
table and a custom binned PaCO2 prior. These files are parsed in the browser and
passed to the Pyodide worker. They are not sent to a backend, persisted, logged,
or encoded in URLs. Uploaded study tables and prior supports must satisfy the same validation
contract as canonical inputs.

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
