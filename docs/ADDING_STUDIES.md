# Adding/Editing Conway Studies

Use the canonical spreadsheet as the single source of truth:
- `Data/conway_studies.xlsx`
- Template: `Data/conway_studies_template.xlsx`

## Step-by-step
1. Open `Data/conway_studies_template.xlsx` and add a new row for each study.
2. Fill in:
   - `study_id`: unique short label.
   - `bias`: PaCO2 − TcCO2 mean difference (mmHg).
   - `sd` (or `s2`): within-study SD (or variance) of differences.
   - `n_pairs`: paired measurements count.
   - `n_participants`: participant count contributing pairs.
   - `is_icu`, `is_arf`, `is_lft`: subgroup flags (0/1).
   - Optional `c`: repeated measures per participant.
3. Save as `Data/conway_studies.xlsx` (or export to CSV).

## Where the numbers come from
- `bias`: mean PaCO2 − TcCO2 difference reported in the study.
- `sd` or `s2`: SD/variance of the paired differences.
- `n_pairs`: total number of paired measurements.
- `n_participants`: number of participants contributing pairs.

## PaCO2 priors
The public prior for browser prior-weighted inference lives separately in
`Data/paco2_public_prior.csv`. Updating the Conway study table does not change
this prior; use the prior build script if the restricted in-silico distribution
changes. Exact count-bearing prior bins are local/generated outputs and should
not be committed.

## Validate the table locally
```bash
uv run python - <<'PY'
import pandas as pd
from tcco2_accuracy.validate_inputs import validate_conway_studies_df

df = pd.read_excel('Data/conway_studies.xlsx')
validate_conway_studies_df(df)
print('OK')
PY
```

## Regenerate artifacts
```bash
uv run python scripts/rebuild_artifacts.py --input-study-table Data/conway_studies.xlsx \
  --paco2-path Data/in_silico_tcco2_db.dta --seed 202401 --n-boot 1000 --thresholds 45
```

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
