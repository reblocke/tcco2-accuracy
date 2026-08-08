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

The Conway study table follows `docs/CONWAY_DATA_SCHEMA.md`. No restricted-derived PaCO2 prior is
shipped or staged. Normalized weights can remain reconstructable when a denominator is known, so
prior-weighted browser inference requires an explicit client-side upload.

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

## Uploaded or Private Prior Fields

| Field | Definition | Validation |
| --- | --- | --- |
| `group` | Clinical setting group | After normalization, exactly all four labels `pft`, `ed_inp`, `icu`, and `all`; no blanks, missing groups, or extras |
| `paco2_bin` | PaCO2 prior support value | Finite and strictly positive, mmHg; no hard upper cutoff |
| `weight` | Normalized uploaded/private prior weight | Nonnegative; should sum approximately 1 within group |

The schema is public, but a prior generated from restricted data remains restricted-derived even
when it contains weights rather than counts. `tests/fixtures/synthetic_paco2_prior.csv` is a clearly
synthetic test-only example.

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

## In-Memory Downstream Caller Fields

These fields are accepted only by the caller-managed
`tcco2_accuracy.workflows.downstream.run_downstream_analysis(...)` API. The API writes, stages, logs,
and tracks nothing; patient, encounter, order, and PaCO2 cell values are not returned. Aggregate
result tables and a manifest containing the caller-vetted revision token and configured column-role
names are returned in memory. The four caller-configurable identifier/order column roles must be
nonblank, mutually distinct, and must not reuse fixed `paco2`, subgroup, or raw subgroup-flag columns.

| Field | Definition | Validation |
| --- | --- | --- |
| `patient_id` | Caller-local patient identifier used only to form resampling clusters | Nonblank after string trimming on retained rows |
| `encounter_id` | Caller-local encounter identifier used to choose an index encounter | Nonblank after string trimming on retained rows |
| `encounter_order` | Caller-local ordering value for the earliest eligible encounter | Entire retained field is finite numeric or valid datetime; exact integer precision is retained; one consistent value per patient/encounter; ties for earliest encounter fail closed |
| `measurement_order` | Caller-local ordering value for the earliest eligible PaCO2 measurement | Entire retained field is finite numeric or valid datetime; exact integer precision is retained; duplicate patient/encounter/measurement-order keys after numeric or UTC normalization and ties at the selected earliest measurement fail closed |
| `target_data_revision` | Caller-supplied opaque source-extract/version token recorded in the run manifest | 1-64 ASCII characters; starts alphanumeric; remaining characters limited to letters, digits, `.`, `_`, or `-`; caller must not encode patient values or direct identifiers |

The same input also requires `paco2` and either an already prepared `subgroup` or the raw subgroup
flags documented above. The caller is responsible for supplying only records eligible for the
intended observation window; this API does not adjudicate eligibility.

Returned `core`, `prediction`, and `two_stage` tables contain identifying analysis fields plus
`bootstrap_q025`, `bootstrap_q500`, and `bootstrap_q975`. Probability metrics use the 0-1 scale,
prediction-limit metrics use mmHg, and likelihood ratios are unitless. The returned stability table
contains aggregate MCSE diagnostics only. The JSON-safe manifest records configuration, the opaque
target revision, subgroup-input mode, redraw fractions, and contract-compliance status; it is not a
replacement for caller-managed private source provenance.

## Conditional Curve Fields

| Field | Definition | Validation |
| --- | --- | --- |
| `paco2_bin` | Inclusive lower edge of a conditional-curve display bin | Finite numeric, mmHg |
| `paco2_bin_upper` | Exclusive upper edge of that display bin | Finite numeric greater than `paco2_bin` |

Conditional TN/FP/FN/TP truth and test-positive probabilities are computed from original unbinned
PaCO2 values and then aggregated. The production bin contract is
`[paco2_bin, paco2_bin_upper)`; display grouping never determines truth.

## Browser App Uploads

The app accepts an optional CSV/XLSX upload for a custom Conway-compatible study table.
Prior-weighted mode additionally requires a binned PaCO2 prior upload; likelihood-only mode does
not use a prior. These files are parsed in the browser and
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
The exact current-tree allowlists and SHA-256 values are machine-readable in
`docs/data_release_contract.json`.

| Asset | Purpose | Current status |
| --- | --- | --- |
| `artifacts/STATUS.md` | Current/frozen artifact manifest and promotion gates | Current documentation |
| `artifacts/bootstrap_params.csv` | Canonical browser bootstrap parameters | Corrected-provisional |
| Rounded/aggregate PaCO2-dependent artifacts retained in `artifacts/STATUS.md` | Simulation, classification, two-stage, prediction-interval, and manuscript review outputs | Frozen legacy-method; not release-approved |

## Restricted Or Local-Only Assets

| Asset | Rule |
| --- | --- |
| `Data/in_silico_tcco2_db.dta` | Local restricted source input; never track |
| `Data/In Silico TCCO2 Database.dta` | Alternate local restricted source filename; never track |
| Restricted-derived prior CSV/XLSX, including normalized weights | Generate only under `.pytest_tmp/`, `.tmp/`, or an explicit external private workspace; never stage |
| Exact-count PaCO2 summaries, conditional curves, figures, and manuscript tables | Generate only under `.pytest_tmp/`, `.tmp/`, or an explicit external private workspace; never track |
| `web/assets/data/` | Generated staged data for Pages; do not hand-edit |
| `web/assets/py/` | Generated staged Python package for Pyodide; do not hand-edit |

Use `docs/restricted_data_provenance.template.json` before any private rebuild. Unresolved ownership,
IRB/protocol, waiver, authorization, extraction, observation-unit, repeated-patient, field,
permission, and retention items remain `HUMAN REVIEW REQUIRED`.

## Review Flags

- `Data/data_counts.csv` is public and useful for reproducibility, but source
  provenance should be checked before reuse outside this repository.
- The manuscript has not yet been submitted; do not publish full manuscript text
  or internal draft files as machine-readable surfaces.
- The static app is research software only and is not intended for clinical
  decision-making.
- Corrected browser outputs remain provisional pending independent biostatistical
  review; frozen downstream artifacts must not be described as corrected.
- Public branch and tag history is rewritten and checked against the release contract. Independently
  retained clones, caches, and historical Pages deployments outside repository-controlled refs may
  retain earlier material.
