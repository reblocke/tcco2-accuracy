# Meta-analysis LoA Check

Source: `Data/conway_studies.csv`.
Agreement method version: `agreement_natural_log_tau2_direct_v1`.
Results status: `provisional`.
- Formula: SD_total = sqrt(sigma^2 + tau^2); LoA = delta ± 2 * SD_total.

| Population | Bias | SD | Tau2 | LoA L | LoA U | CI L | CI U |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Main analysis | -0.11 | 4.03 | 8.86 | -10.13 | 9.90 | -11.28 | 11.05 |
| ICU | -0.60 | 4.69 | 1.89 | -10.38 | 9.18 | -12.35 | 11.16 |
| Acute respiratory failure | 1.69 | 4.89 | 3.16 | -8.71 | 12.10 | -12.40 | 15.78 |
| Outpatients requiring lung function tests | -0.05 | 2.78 | 1.45 | -6.11 | 6.02 | -11.16 | 11.07 |

## Corrected versus published/legacy comparator

Published comparator: `tests/fixtures/conway_table1.csv`.
Published values are rounded to one decimal place; delta is corrected minus published and therefore includes source-rounding differences.

### Main analysis

| Metric | Corrected | Published/legacy | Delta |
| --- | --- | --- | --- |
| Bias | -0.11 | -0.10 | -0.01 |
| SD | 4.03 | 1.90 | +2.13 |
| Tau2 | 8.86 | 8.90 | -0.04 |
| LoA L | -10.13 | -7.10 | -3.03 |
| LoA U | 9.90 | 6.90 | +3.00 |
| CI L | -11.28 | -15.10 | +3.82 |
| CI U | 11.05 | 14.90 | -3.85 |

### ICU

| Metric | Corrected | Published/legacy | Delta |
| --- | --- | --- | --- |
| Bias | -0.60 | -0.60 | +0.00 |
| SD | 4.69 | 2.00 | +2.69 |
| Tau2 | 1.89 | 1.90 | -0.01 |
| LoA L | -10.38 | -5.40 | -4.98 |
| LoA U | 9.18 | 4.20 | +4.98 |
| CI L | -12.35 | -7.30 | -5.05 |
| CI U | 11.16 | 6.10 | +5.06 |

### Acute respiratory failure

| Metric | Corrected | Published/legacy | Delta |
| --- | --- | --- | --- |
| Bias | 1.69 | 1.70 | -0.01 |
| SD | 4.89 | 2.00 | +2.89 |
| Tau2 | 3.16 | 3.20 | -0.04 |
| LoA L | -8.71 | -3.70 | -5.01 |
| LoA U | 12.10 | 7.10 | +5.00 |
| CI L | -12.40 | -7.80 | -4.60 |
| CI U | 15.78 | 11.20 | +4.58 |

### Outpatients requiring lung function tests

| Metric | Corrected | Published/legacy | Delta |
| --- | --- | --- | --- |
| Bias | -0.05 | -0.10 | +0.05 |
| SD | 2.78 | 1.60 | +1.18 |
| Tau2 | 1.45 | 1.40 | +0.05 |
| LoA L | -6.11 | -4.00 | -2.11 |
| LoA U | 6.02 | 3.90 | +2.12 |
| CI L | -11.16 | -7.30 | -3.86 |
| CI U | 11.07 | 7.30 | +3.77 |