# Adding/Editing Conway Studies

Maintain two semantically equivalent representations of the canonical table:
- `Data/conway_studies.xlsx`: human-editable review and authoring mirror.
- `Data/conway_studies.csv`: operational source for canonical artifact promotion and browser staging.
- Template: `Data/conway_studies_template.xlsx`

## Step-by-step
1. Open `Data/conway_studies_template.xlsx` and add a new row for each study.
2. Fill in:
   - `study_id`: unique short label that remains non-empty and unique after surrounding
     whitespace is stripped.
   - `bias`: PaCO2 − TcCO2 mean difference (mmHg).
   - `sd` or `s2`: a finite, positive within-study SD or variance of differences. Every value in a
     supplied column is required. If both columns are supplied, `sd²` and `s2` must agree within
     `rtol=1e-10`, `atol=1e-12`.
   - `n_pairs`: finite integer paired-measurement count, at least `n_participants`.
   - `n_participants`: finite integer participant count greater than 1.
   - `is_icu`, `is_arf`, `is_lft`: subgroup flags (0/1). Flags may overlap when supported by the
     source; they are not mutually exclusive.
   - Optional `c`: finite repeated measurements per participant, at least 1 and consistent with
     `n_pairs / n_participants` within `rtol=1e-10`, `atol=1e-12`.
3. Save the reviewed table as `Data/conway_studies.xlsx` and export the same rows and values to
   `Data/conway_studies.csv`. Contract tests require semantic equality within `1e-12`.

## Where the numbers come from
- `bias`: mean PaCO2 − TcCO2 difference reported in the study.
- `sd` or `s2`: SD/variance of the paired differences.
- `n_pairs`: total number of paired measurements.
- `n_participants`: number of participants contributing pairs.

Do not enter a rounded reported `c` if it differs from the exact count ratio beyond the validation
tolerance; omit the `c` column so the package derives `n_pairs / n_participants`. If the column is
supplied, every row must contain a valid value. If redundant `sd` and `s2` values disagree, resolve
which source field is authoritative and provide only that field rather than adjusting a value
merely to pass validation.

The canonical table must contain at least one row. Before relying on a subgroup result, confirm the
requested `is_*` flag selects at least one row; an empty subgroup fails closed.

The RData/count exporter's permissive options may relax other source checks, but they never permit
missing or blank study identifiers or non-integer observed `n_pairs`/`n_participants` values.

## PaCO2 priors
The public prior for browser prior-weighted inference lives separately in
`Data/paco2_public_prior.csv`. Updating the Conway study table does not change
this prior; use the prior build script if the restricted in-silico distribution
changes. Exact count-bearing prior bins are local/generated outputs and should
not be committed. PaCO2 prior support values must be finite and strictly positive; there is no
evidence-based hard upper validation bound. When rebuilding from a source distribution, genuinely
missing PaCO2 rows are dropped, while malformed, infinite, zero, or negative retained values fail.
Prior group labels must contain exactly `pft`, `ed_inp`, `icu`, and `all`, with no blanks or extras.

## Validate the table locally
```bash
uv run python - <<'PY'
import pandas as pd
from tcco2_accuracy.validate_inputs import validate_conway_studies_df

df = pd.read_excel('Data/conway_studies.xlsx')
validate_conway_studies_df(df)
csv_df = pd.read_csv('Data/conway_studies.csv')
validate_conway_studies_df(csv_df)
pd.testing.assert_frame_equal(
    csv_df, df, check_dtype=False, check_exact=False, rtol=0, atol=1e-12
)
print('OK')
PY
```

## Build and review a candidate
```bash
uv run python scripts/rebuild_artifacts.py --profile public-agreement \
  --input-study-table Data/conway_studies.xlsx \
  --out .pytest_tmp/public-agreement-candidate \
  --seed 202401 --n-boot 1000 --bootstrap-mode cluster_plus_withinstudy
```

This profile uses public Conway inputs only and does not regenerate frozen
PaCO2-dependent manuscript outputs. Review this scratch candidate and its provenance before
changing the canonical table or promoting anything.

## Promote canonical artifacts

After CSV/XLSX parity, provenance review, tests, and scientific approval, use the exact locked
command below:

```bash
uv run python scripts/rebuild_artifacts.py --profile public-agreement \
  --input-study-table Data/conway_studies.csv --out artifacts \
  --seed 202401 --n-boot 1000 --bootstrap-mode cluster_plus_withinstudy
```

The `artifacts/` destination rejects any different input, seed, draw count, or bootstrap mode.

## Run tests
```bash
uv run pytest -q
```

## Refresh static app assets
```bash
make stage-web
make verify
```

The browser app uses staged CSV assets, so run `make stage-web` after changing
canonical study inputs or bootstrap artifacts.
