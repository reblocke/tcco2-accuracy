# Adding/Editing Conway Studies

Maintain two semantically equivalent representations of the canonical table:
- `Data/conway_studies.xlsx`: human-editable review and authoring mirror.
- `Data/conway_studies.csv`: operational source for canonical artifact promotion and browser staging.
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
3. Save the reviewed table as `Data/conway_studies.xlsx` and export the same rows and values to
   `Data/conway_studies.csv`. Contract tests require semantic equality within `1e-12`.

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
