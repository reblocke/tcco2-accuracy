# Conway Study Input Schema

Canonical study-level inputs for the Conway meta-analysis live in:
- `Data/conway_studies.csv`
- `Data/conway_studies.xlsx`

The table is the single source of truth for meta-analysis, bootstrap draws, and inference.

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
- `study_id` is unique and non-empty.
- `bias`, `sd`/`s2`, `n_pairs`, `n_participants`, `c` (if present) are finite.
- `sd` and `s2` are strictly positive.
- `n_pairs` and `n_participants` are positive integers.
- subgroup flags are boolean-like (0/1 or True/False) with no missing values.

## Notes
- Main analysis is the full table; subgroups are selected by the `is_*` flags.
- ARF subgroup includes both Kim 2014 cohorts (normotensive + hypotensive).
- `sd` and `s2` should be internally consistent (`s2 ≈ sd²`) if both are provided.
