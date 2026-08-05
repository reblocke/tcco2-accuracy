# Conway Study Input Schema

Canonical study-level inputs for the Conway meta-analysis live in:
- `Data/conway_studies.csv` (operational promotion and browser-staging source)
- `Data/conway_studies.xlsx` (human-editable review and authoring mirror)

These files represent one canonical table and must remain semantically equal within `1e-12`.
The CSV is the executable source of truth for canonical artifact promotion; the XLSX supports
human review and study additions.

## Required columns
- `study_id` (string): unique study identifier.
- `bias` (float): study-level mean difference in mmHg, defined as PaCO2 − TcCO2.
- `sd` (float): within-study SD of differences (mmHg). Optional if `s2` is provided.
- `s2` (float): within-study variance of differences (mmHg²). Optional if `sd` is provided.
- `n_pairs` (int): number of paired measurements.
- `n_participants` (int): number of participants contributing pairs.
- `is_icu` (bool-like): 1/0 flag for ICU subgroup.
- `is_arf` (bool-like): 1/0 flag for acute respiratory failure subgroup.
- `is_lft` (bool-like): 1/0 flag for lung-function-test outpatient subgroup.

## Optional columns
- `c` (float): repeated measures per participant. If omitted, it is derived as `n_pairs / n_participants`.

## Validation rules
- The study table is non-empty. Any requested subgroup analysis must select at least one row.
- `study_id` is stripped of leading/trailing whitespace before validation, non-empty after
  stripping, and unique after stripping.
- `bias` is finite.
- The table contains at least one of the `sd` or `s2` columns. Every value in each supplied column
  is finite and strictly positive. When both columns are supplied, `sd²` and `s2` must agree with
  `rtol=1e-10` and `atol=1e-12`.
- `n_pairs` and `n_participants` are finite integers, `n_participants > 1`, and
  `n_pairs >= n_participants`.
- If supplied, `c` is finite, `c >= 1`, and agrees with `n_pairs / n_participants` using
  `rtol=1e-10` and `atol=1e-12`.
- subgroup flags are boolean-like (0/1 or True/False) with no missing values.

## Notes
- Main analysis is the full table; subgroups are selected by the `is_*` flags.
- Conway subgroup flags are allowed to overlap: one study row may be selected into more than one
  subgroup. This differs from the mutually exclusive PaCO2 record groups defined in
  `docs/SPEC.md`.
- ARF subgroup includes both Kim 2014 cohorts (normotensive + hypotensive).
- If a reported `c` is rounded and does not satisfy the ratio check, omit the column and allow the
  package to derive the exact value; a supplied `c` column must be complete. If redundant `sd` and
  `s2` fields are discordant, retain only the authoritative source field rather than forcing them
  to agree.
- The RData/count exporter always rejects missing or blank study identifiers and non-integer
  observed counts. `--no-strict` and `--allow-missing-counts` do not relax those checks.
- Public PaCO2 prior weights used for browser prior-weighted inference are maintained separately in `Data/paco2_public_prior.csv`.
