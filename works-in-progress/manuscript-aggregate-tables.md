# How arterial record selection and agreement assumptions affect PaCO₂ predictions

The custodian approved these aggregate tables on November 24, 2026, for the manuscript’s [draft PR #24](https://github.com/reblocke/tcco2-accuracy/pull/24). The approved source file has SHA-256 `9eda8d45034040a81f757854e55701008057a35f299b4428911d53e7ac67d4f3`. Independent statistical review is pending.

The prediction tables give the median and central 95% prediction interval (PI) for simultaneous arterial CO₂ (PaCO₂) given observed transcutaneous CO₂ (TcCO₂). Pressures are in mmHg. Disagreement is PaCO₂ − TcCO₂. Both threshold probabilities include equality. When an individual prediction displays a range for a probability, its numerical precision permits either whole percentage. Differences in probabilities are expressed in percentage points (pp). The strategy tables give modeled errors per 100 eligible patient-weighted testing occasions and 95% credible intervals (CrI).

## Reference measurement and population

The archived cohort specification selected first qualifying encounters in 2022 within reason for suspicion and setting. The observed encounter and result-date ranges below are separate from the specified enrollment window. Original data extraction date and number of contributing sites for this subset could not be established from the matched keys. Population summaries below describe the keyed extract.

Observed encounter dates 2022-01-01 to 2022-12-31. Eligible result dates 2021-11-28 to 2023-05-11. Recorded age among eligible encounters ranges from 18 to 96 years. There were 0 encounter ages below 18 and 0 missing ages.

Each arterial result record has a unique source ID. Specimen and panel identifiers are empty, and dates do not establish the order of same-day results. One specimen could therefore contribute more than one record across codes. LOINC 2019-8 identifies ordinary PaCO₂ and 32771-8 identifies temperature-adjusted PaCO₂. The first table counts each code before and after first-day selection using both codes. Patients can appear in both code rows.

| Population | Selection | Arterial LOINC | Result records | Distinct patients with this code |
| --- | --- | --- | --- | --- |
| All settings | eligible | 2019-8 | 925,699 | 126,686 |
| All settings | eligible | 32771-8 | 93,400 | 23,977 |
| All settings | first | 2019-8 | 335,016 | 126,432 |
| All settings | first | 32771-8 | 50,851 | 23,525 |
| Critical Care Billing | eligible | 2019-8 | 504,065 | 49,287 |
| Critical Care Billing | eligible | 32771-8 | 31,726 | 6,758 |
| Critical Care Billing | first | 2019-8 | 128,251 | 49,197 |
| Critical Care Billing | first | 32771-8 | 13,567 | 6,416 |
| ED/inpatient | eligible | 2019-8 | 359,625 | 66,746 |
| ED/inpatient | eligible | 32771-8 | 59,259 | 15,456 |
| ED/inpatient | first | 2019-8 | 186,889 | 66,615 |
| ED/inpatient | first | 32771-8 | 35,469 | 15,408 |
| Ambulatory | eligible | 2019-8 | 11,185 | 6,856 |
| Ambulatory | eligible | 32771-8 | 2,043 | 1,638 |
| Ambulatory | first | 2019-8 | 11,032 | 6,854 |
| Ambulatory | first | 32771-8 | 2,027 | 1,637 |

Restricting to ordinary PaCO₂ can change a patient’s first eligible day. The next table repeats eligibility and first-day selection under each definition, independently for All and within each setting. Each patient contributes total weight one, divided among eligible results on that first testing day.

| Population | Stage | Definition | Patients | Encounters | Result records |
| --- | --- | --- | --- | --- | --- |
| All settings | Eligible | Both codes | 142,156 | 147,028 | 1,019,099 |
| All settings | Eligible | 2019-8 only | 126,686 | 130,888 | 925,699 |
| All settings | First day | Both codes | 142,156 | 142,424 | 385,867 |
| All settings | First day | 2019-8 only | 126,686 | 126,948 | 336,010 |
| Critical Care Billing | Eligible | Both codes | 52,931 | 53,985 | 535,791 |
| Critical Care Billing | Eligible | 2019-8 only | 49,287 | 50,245 | 504,065 |
| Critical Care Billing | First day | Both codes | 52,931 | 52,968 | 141,818 |
| Critical Care Billing | First day | 2019-8 only | 49,287 | 49,322 | 128,373 |
| ED/inpatient | Eligible | Both codes | 77,119 | 78,561 | 418,884 |
| ED/inpatient | Eligible | 2019-8 only | 66,746 | 67,757 | 359,625 |
| ED/inpatient | First day | Both codes | 77,119 | 77,146 | 222,358 |
| ED/inpatient | First day | 2019-8 only | 66,746 | 66,771 | 187,596 |
| Ambulatory | Eligible | Both codes | 8,174 | 8,203 | 13,228 |
| Ambulatory | Eligible | 2019-8 only | 6,856 | 6,879 | 11,185 |
| Ambulatory | First day | Both codes | 8,174 | 8,178 | 13,059 |
| Ambulatory | First day | 2019-8 only | 6,856 | 6,860 | 11,034 |

The baseline percentages describe PaCO₂ before observing TcCO₂. A residual false negative (FN) is a modeled missed PaCO₂ ≥45 after applying the inclusive 40–50 mmHg ABG referral rule. Change is ordinary-only minus both codes, calculated within each posterior draw.

| Population | Both ≥45 | 2019-8 ≥45 | Both ≥52 | 2019-8 ≥52 | Ordinary residual FN per 100 (95% CrI) | Change in residual FN per 100 (95% CrI) |
| --- | --- | --- | --- | --- | --- | --- |
| All settings | 29.39% | 28.01% | 15.25% | 13.60% | 1.22 (0.92–1.60) | 0.02 (0.02–0.03) |
| Critical Care Billing | 31.63% | 28.20% | 18.83% | 14.67% | 1.12 (0.63–2.03) | 0.06 (0.03–0.11) |
| ED/inpatient | 27.81% | 27.46% | 12.89% | 12.47% | 2.97 (1.79–4.62) | 0.02 (0.01–0.03) |
| Ambulatory | 26.52% | 27.12% | 10.12% | 10.77% | 0.80 (0.19–2.98) | -0.00 (-0.01–-0.00) |

Across the 15 displayed characteristics in each setting, the percentage of patient weight with a missing stored field rounds to 0.000%. A zero diagnosis indicator can include unrecorded disease. [Field-level missingness audit](https://github.com/reblocke/tcco2-accuracy/blob/4903a7fc381b2f42094889e018d497c5157509c1/works-in-progress/manuscript-aggregate-tables.md#reference-measurement-and-population)

## Reference-code check at three readings

The first table uses ordinary arterial results and the assigned agreement fit. The second subtracts the matched both-code estimate from the ordinary-only estimate. Contrasts use the unrounded estimates, so a small shift can occur even when displayed values round to the same number.

| Arterial population | Agreement fit | TcCO₂ | Median | 95% PI | P(≥45) | P(≥52) |
| --- | --- | --- | --- | --- | --- | --- |
| All settings ordinary only | Overall | 35 | 36 | 28 to 44 | 2% | <1% |
| All settings ordinary only | Overall | 47 | 44 | 35 to 53 | 47% | 5% |
| All settings ordinary only | Overall | 55 | 52 | 39 to 62 | 87% | 49% |
| Critical Care Billing ordinary only | ICU | 35 | 36 | 27 to 44 | 2% | <1% |
| Critical Care Billing ordinary only | ICU | 47 | 44 | 35 to 54 | 46% | 6% |
| Critical Care Billing ordinary only | ICU | 55 | 52 | 40 to 62 | 89% | 49% |
| ED/inpatient ordinary only | Acute respiratory failure | 35 | 37 | 29 to 47 | 6% | 1% |
| ED/inpatient ordinary only | Acute respiratory failure | 47 | 45 | 35 to 56 | 54% | 10% |
| ED/inpatient ordinary only | Acute respiratory failure | 55 | 52 | 37 to 65 | 84% | 54% |
| Ambulatory ordinary only | Lung function | 35 | 36 | 30 to 44 | 1% | <1% |
| Ambulatory ordinary only | Lung function | 47 | 45 | 37 to 53 | 57% | 4% |
| Ambulatory ordinary only | Lung function | 55 | 53 | 41 to 61 | 93% | 63% |

| Population | TcCO₂ | Δ lower | Δ median | Δ upper | Δ ≥45 (pp) | Δ ≥52 (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| All settings | 35 | +0.0 | +0.0 | +0.0 | -0.0 | +0.0 |
| All settings | 47 | +0.0 | +0.0 | +0.0 | -0.1 | +0.0 |
| All settings | 55 | +0.0 | +0.0 | +0.0 | -0.0 | -0.0 |
| Critical Care Billing | 35 | +0.0 | +0.1 | +0.0 | +0.0 | -0.0 |
| Critical Care Billing | 47 | +0.0 | +0.0 | +0.0 | -0.1 | -0.2 |
| Critical Care Billing | 55 | +0.0 | -0.3 | -0.9 | -0.3 | -1.2 |
| ED/inpatient | 35 | +0.0 | +0.2 | +0.0 | -0.0 | +0.0 |
| ED/inpatient | 47 | +0.0 | +0.0 | +0.0 | -0.1 | +0.2 |
| ED/inpatient | 55 | +0.0 | +0.0 | +0.0 | +0.1 | +0.4 |
| Ambulatory | 35 | +0.0 | +0.0 | +0.1 | +0.0 | +0.0 |
| Ambulatory | 47 | +0.0 | +0.0 | +0.8 | -0.6 | +0.0 |
| Ambulatory | 55 | +0.0 | +0.0 | +0.0 | +0.1 | +1.1 |

## Arterial population and agreement fit

The 12 assigned pairs reproduce the approved unrounded lookup. Each other pair applies a saved fit to a different arterial population. Pointwise checks use four chains, contiguous equal-sized batches and separate quadrature orders. Differences use unrounded qualified estimates and have descriptive precision.

| Arterial population | Agreement fit | TcCO₂ | Median | 95% PI | P(≥45) | P(≥52) |
| --- | --- | --- | --- | --- | --- | --- |
| All settings | Overall | 35 | 36 | 28 to 44 | 2% | <1% |
| All settings | ICU | 35 | 36 | 28 to 45 | 2% | <1% |
| All settings | Acute respiratory failure | 35 | 37 | 29 to 47 | 6% | 1% |
| All settings | Lung function | 35 | 36 | 29 to 43 | 1% | <1% |
| Critical Care Billing | Overall | 35 | 36 | 27 to 44 | 2% | <1% |
| Critical Care Billing | ICU | 35 | 35 | 27 to 44 | 2% | <1% |
| Critical Care Billing | Acute respiratory failure | 35 | 37 | 28 to 47 | 6% | 1% |
| Critical Care Billing | Lung function | 35 | 35 | 29 to 43 | 1% | <1% |
| ED/inpatient | Overall | 35 | 36 | 28 to 44 | 2% | <1% |
| ED/inpatient | ICU | 35 | 36 | 28 to 45 | 3% | <1% |
| ED/inpatient | Acute respiratory failure | 35 | 37 | 29 to 47 | 6% | 1% |
| ED/inpatient | Lung function | 35 | 36 | 30 to 43 | 1% | <1% |
| Ambulatory | Overall | 35 | 36 | 29 to 45 | 2% | <1% |
| Ambulatory | ICU | 35 | 36 | 29 to 45 | 3% | <1% |
| Ambulatory | Acute respiratory failure | 35 | 37 | 30 to 47 | 6% | 1% |
| Ambulatory | Lung function | 35 | 36 | 30 to 44 | 1% | <1% |
| All settings | Overall | 47 | 44 | 35 to 53 | 47% | 5% |
| All settings | ICU | 47 | 44 | 35 to 54 | 45% | 5% |
| All settings | Acute respiratory failure | 47 | 45 | 35 to 56 | 55% | 10% |
| All settings | Lung function | 47 | 45 | 37 to 53 | 58% | 4% |
| Critical Care Billing | Overall | 47 | 44 | 35 to 54 | 48% | 6% |
| Critical Care Billing | ICU | 47 | 44 | 35 to 54 | 46% | 6% |
| Critical Care Billing | Acute respiratory failure | 47 | 45 | 34 to 57 | 57% | 12% |
| Critical Care Billing | Lung function | 47 | 45 | 37 to 53 | 59% | 5% |
| ED/inpatient | Overall | 47 | 44 | 35 to 53 | 46% | 5% |
| ED/inpatient | ICU | 47 | 44 | 35 to 53 | 44% | 5% |
| ED/inpatient | Acute respiratory failure | 47 | 45 | 35 to 56 | 54% | 10% |
| ED/inpatient | Lung function | 47 | 45 | 37 to 53 | 57% | 4% |
| Ambulatory | Overall | 47 | 44 | 35 to 53 | 47% | 5% |
| Ambulatory | ICU | 47 | 44 | 36 to 53 | 44–45% | 5% |
| Ambulatory | Acute respiratory failure | 47 | 45 | 35 to 55 | 55% | 9% |
| Ambulatory | Lung function | 47 | 45 | 37 to 52 | 58% | 4% |
| All settings | Overall | 55 | 52 | 39 to 62 | 87% | 49% |
| All settings | ICU | 55 | 51 | 40 to 62 | 88% | 47% |
| All settings | Acute respiratory failure | 55 | 52 | 37 to 66 | 85% | 55% |
| All settings | Lung function | 55 | 53 | 41 to 61 | 93% | 64% |
| Critical Care Billing | Overall | 55 | 52 | 39 to 63 | 88% | 53% |
| Critical Care Billing | ICU | 55 | 52 | 40 to 63 | 89% | 51% |
| Critical Care Billing | Acute respiratory failure | 55 | 53 | 37 to 67 | 87% | 59% |
| Critical Care Billing | Lung function | 55 | 53 | 41 to 62 | 94% | 67% |
| ED/inpatient | Overall | 55 | 51 | 38 to 62 | 86% | 47% |
| ED/inpatient | ICU | 55 | 51 | 39 to 62 | 87% | 46% |
| ED/inpatient | Acute respiratory failure | 55 | 52 | 37 to 65 | 84% | 53% |
| ED/inpatient | Lung function | 55 | 53 | 41 to 61 | 93% | 63% |
| Ambulatory | Overall | 55 | 51 | 39 to 61 | 86% | 46% |
| Ambulatory | ICU | 55 | 51 | 39 to 61 | 87% | 44% |
| Ambulatory | Acute respiratory failure | 55 | 52 | 37 to 64 | 83% | 51% |
| Ambulatory | Lung function | 55 | 53 | 41 to 61 | 93% | 61% |

### Change the agreement fit while holding the arterial population fixed

| Population | Fit | TcCO₂ | Δ lower | Δ median | Δ upper | Δ ≥45 (pp) | Δ ≥52 (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Critical Care Billing | Overall | 35 | +0.2 | +0.1 | -0.2 | -0.1 | +0.0 |
| Critical Care Billing | Overall | 47 | -0.1 | +0.3 | +0.0 | +2.3 | -0.2 |
| Critical Care Billing | Overall | 55 | -1.0 | +0.0 | +0.0 | -0.8 | +1.9 |
| ED/inpatient | Overall | 35 | -1.0 | -1.0 | -2.7 | -3.4 | -0.4 |
| ED/inpatient | Overall | 47 | +0.0 | -1.0 | -3.0 | -7.9 | -4.8 |
| ED/inpatient | Overall | 55 | +1.4 | -1.0 | -3.0 | +1.9 | -5.7 |
| Ambulatory | Overall | 35 | -1.0 | +0.0 | +1.3 | +1.0 | +0.1 |
| Ambulatory | Overall | 47 | -1.8 | -1.0 | +0.8 | -11.0 | +0.9 |
| Ambulatory | Overall | 55 | -2.5 | -2.0 | +0.0 | -7.1 | -15.7 |
| All settings | ICU | 35 | +0.0 | +0.0 | +0.6 | +0.1 | -0.0 |
| All settings | ICU | 47 | +0.0 | +0.0 | +0.4 | -2.4 | +0.1 |
| All settings | ICU | 55 | +0.9 | -0.6 | +0.0 | +0.9 | -2.0 |
| ED/inpatient | ICU | 35 | -1.0 | -1.0 | -2.0 | -3.2 | -0.4 |
| ED/inpatient | ICU | 47 | +0.3 | -1.0 | -3.0 | -10.3 | -4.7 |
| ED/inpatient | ICU | 55 | +2.0 | -1.0 | -3.0 | +2.8 | -7.7 |
| Ambulatory | ICU | 35 | -1.0 | +0.0 | +1.4 | +1.2 | +0.0 |
| Ambulatory | ICU | 47 | -1.2 | -1.0 | +0.8 | -13.4 | +1.0 |
| Ambulatory | ICU | 55 | -1.8 | -2.0 | +0.0 | -6.2 | -17.7 |
| All settings | Acute respiratory failure | 35 | +1.0 | +1.0 | +2.9 | +3.4 | +0.5 |
| All settings | Acute respiratory failure | 47 | +0.0 | +1.0 | +2.7 | +8.0 | +5.2 |
| All settings | Acute respiratory failure | 55 | -1.8 | +0.7 | +3.9 | -1.7 | +6.1 |
| Critical Care Billing | Acute respiratory failure | 35 | +1.0 | +1.6 | +2.8 | +3.3 | +0.5 |
| Critical Care Billing | Acute respiratory failure | 47 | -1.0 | +1.3 | +3.0 | +10.6 | +5.7 |
| Critical Care Billing | Acute respiratory failure | 55 | -3.1 | +1.0 | +4.0 | -2.3 | +8.5 |
| Ambulatory | Acute respiratory failure | 35 | -0.3 | +1.2 | +3.4 | +4.5 | +0.5 |
| Ambulatory | Acute respiratory failure | 47 | -2.0 | +0.0 | +2.8 | -3.3 | +5.4 |
| Ambulatory | Acute respiratory failure | 55 | -4.0 | -1.0 | +3.0 | -9.3 | -10.6 |
| All settings | Lung function | 35 | +1.0 | +0.0 | -1.1 | -0.9 | -0.1 |
| All settings | Lung function | 47 | +2.0 | +1.0 | -0.3 | +11.0 | -1.0 |
| All settings | Lung function | 55 | +2.2 | +1.4 | -0.6 | +6.7 | +14.8 |
| Critical Care Billing | Lung function | 35 | +2.0 | -0.3 | -1.2 | -1.0 | -0.0 |
| Critical Care Billing | Lung function | 47 | +2.0 | +1.3 | -1.0 | +13.3 | -1.3 |
| Critical Care Billing | Lung function | 55 | +1.0 | +1.0 | -1.0 | +5.3 | +16.1 |
| ED/inpatient | Lung function | 35 | +0.5 | -1.0 | -4.0 | -4.4 | -0.5 |
| ED/inpatient | Lung function | 47 | +2.0 | +0.0 | -3.3 | +3.2 | -5.7 |
| ED/inpatient | Lung function | 55 | +4.0 | +1.0 | -4.0 | +9.0 | +9.6 |

### Change the arterial population while holding the agreement fit fixed

| Population | Fit | TcCO₂ | Δ lower | Δ median | Δ upper | Δ ≥45 (pp) | Δ ≥52 (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Critical Care Billing | Overall | 35 | -0.8 | -0.5 | -0.1 | -0.1 | +0.0 |
| Critical Care Billing | Overall | 47 | -0.1 | +0.3 | +0.7 | +1.3 | +0.6 |
| Critical Care Billing | Overall | 55 | +0.2 | +0.4 | +1.0 | +1.4 | +3.2 |
| ED/inpatient | Overall | 35 | +0.0 | +0.0 | +0.2 | +0.1 | -0.0 |
| ED/inpatient | Overall | 47 | +0.0 | +0.0 | -0.3 | -0.8 | -0.3 |
| ED/inpatient | Overall | 55 | -0.4 | -0.6 | +0.0 | -0.9 | -1.9 |
| Ambulatory | Overall | 35 | +1.0 | +0.0 | +0.8 | +0.2 | -0.0 |
| Ambulatory | Overall | 47 | +0.2 | +0.0 | -0.3 | -0.0 | -0.5 |
| Ambulatory | Overall | 55 | -0.3 | -0.6 | -1.0 | -1.1 | -3.6 |
| All settings | ICU | 35 | +1.0 | +0.6 | +0.5 | +0.2 | -0.0 |
| All settings | ICU | 47 | +0.0 | +0.0 | -0.3 | -1.4 | -0.7 |
| All settings | ICU | 55 | -0.3 | -1.0 | -1.0 | -1.3 | -3.2 |
| ED/inpatient | ICU | 35 | +1.0 | +0.6 | +0.8 | +0.3 | -0.0 |
| ED/inpatient | ICU | 47 | +0.3 | +0.0 | -1.0 | -2.2 | -1.0 |
| ED/inpatient | ICU | 55 | -1.0 | -1.0 | -1.0 | -2.2 | -5.1 |
| Ambulatory | ICU | 35 | +2.0 | +0.6 | +0.8 | +0.4 | -0.0 |
| Ambulatory | ICU | 47 | +0.8 | +0.0 | -1.0 | -1.5 | -1.2 |
| Ambulatory | ICU | 55 | -0.8 | -1.0 | -2.0 | -2.4 | -6.9 |
| All settings | Acute respiratory failure | 35 | +0.0 | +0.0 | +0.0 | -0.1 | +0.0 |
| All settings | Acute respiratory failure | 47 | +0.0 | +0.0 | +0.0 | +1.0 | +0.7 |
| All settings | Acute respiratory failure | 55 | +0.0 | +0.3 | +0.9 | +1.1 | +2.2 |
| Critical Care Billing | Acute respiratory failure | 35 | -1.0 | +0.0 | +0.0 | -0.2 | +0.1 |
| Critical Care Billing | Acute respiratory failure | 47 | -1.0 | +0.3 | +1.0 | +2.5 | +2.0 |
| Critical Care Billing | Acute respiratory failure | 55 | -0.1 | +1.0 | +2.0 | +2.7 | +5.9 |
| Ambulatory | Acute respiratory failure | 35 | +0.7 | +0.2 | +0.0 | +0.3 | -0.0 |
| Ambulatory | Acute respiratory failure | 47 | +0.0 | +0.0 | -1.0 | +0.6 | -0.5 |
| Ambulatory | Acute respiratory failure | 55 | +0.0 | +0.0 | -1.0 | -0.4 | -2.3 |
| All settings | Lung function | 35 | -1.0 | +0.0 | -0.6 | -0.1 | +0.0 |
| All settings | Lung function | 47 | +0.0 | +0.0 | +0.8 | +0.1 | +0.5 |
| All settings | Lung function | 55 | +0.0 | +0.0 | +0.4 | +0.7 | +2.8 |
| Critical Care Billing | Lung function | 35 | -1.0 | -0.9 | -0.6 | -0.2 | +0.0 |
| Critical Care Billing | Lung function | 47 | +0.0 | +0.3 | +0.8 | +1.3 | +0.9 |
| Critical Care Billing | Lung function | 55 | +0.0 | +0.0 | +1.0 | +1.4 | +5.2 |
| ED/inpatient | Lung function | 35 | -0.5 | +0.0 | -0.6 | -0.1 | +0.0 |
| ED/inpatient | Lung function | 47 | +0.0 | +0.0 | +0.5 | -0.6 | +0.2 |
| ED/inpatient | Lung function | 55 | +0.0 | +0.0 | +0.0 | +0.1 | +1.3 |

### Interaction compared with All

Each entry is the effect of changing the fit from All in the indicated population minus the same change under the All arterial distribution. Interactions depend on the chosen reference and scale.

| Population | Fit | TcCO₂ | ΔΔ lower | ΔΔ median | ΔΔ upper | ΔΔ ≥45 (pp) | ΔΔ ≥52 (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Critical Care Billing | ICU | 35 | -0.2 | -0.1 | -0.4 | -0.0 | -0.0 |
| Critical Care Billing | ICU | 47 | +0.1 | -0.3 | -0.4 | +0.1 | +0.0 |
| Critical Care Billing | ICU | 55 | +0.1 | +0.6 | +0.0 | -0.1 | +0.0 |
| Critical Care Billing | Acute respiratory failure | 35 | -0.2 | +0.5 | +0.1 | -0.0 | +0.1 |
| Critical Care Billing | Acute respiratory failure | 47 | -0.9 | +0.0 | +0.3 | +0.2 | +0.7 |
| Critical Care Billing | Acute respiratory failure | 55 | -0.3 | +0.3 | +0.1 | +0.3 | +0.5 |
| Critical Care Billing | Lung function | 35 | +0.8 | -0.4 | +0.1 | +0.1 | -0.0 |
| Critical Care Billing | Lung function | 47 | +0.1 | +0.0 | -0.7 | -0.1 | -0.2 |
| Critical Care Billing | Lung function | 55 | -0.2 | -0.4 | -0.4 | -0.6 | -0.7 |
| ED/inpatient | ICU | 35 | +0.0 | +0.0 | +0.1 | +0.0 | +0.0 |
| ED/inpatient | ICU | 47 | +0.3 | +0.0 | -0.4 | -0.0 | -0.0 |
| ED/inpatient | ICU | 55 | -0.3 | +0.6 | +0.0 | +0.0 | -0.0 |
| ED/inpatient | Acute respiratory failure | 35 | +0.0 | +0.0 | -0.2 | -0.0 | -0.0 |
| ED/inpatient | Acute respiratory failure | 47 | +0.0 | +0.0 | +0.3 | -0.1 | -0.4 |
| ED/inpatient | Acute respiratory failure | 55 | +0.4 | +0.3 | -0.9 | -0.2 | -0.4 |
| ED/inpatient | Lung function | 35 | +0.5 | +0.0 | -0.2 | -0.0 | +0.0 |
| ED/inpatient | Lung function | 47 | +0.0 | +0.0 | +0.0 | +0.1 | +0.1 |
| ED/inpatient | Lung function | 55 | +0.4 | +0.6 | -0.4 | +0.4 | +0.4 |
| Ambulatory | ICU | 35 | +0.0 | +0.0 | -0.5 | +0.0 | +0.0 |
| Ambulatory | ICU | 47 | +0.6 | +0.0 | -0.4 | -0.0 | -0.0 |
| Ambulatory | ICU | 55 | -0.2 | +0.6 | +0.0 | +0.0 | -0.1 |
| Ambulatory | Acute respiratory failure | 35 | -0.3 | +0.2 | -0.8 | +0.1 | -0.0 |
| Ambulatory | Acute respiratory failure | 47 | -0.2 | +0.0 | -0.7 | -0.3 | -0.7 |
| Ambulatory | Acute respiratory failure | 55 | +0.3 | +0.3 | -0.9 | -0.4 | -0.9 |
| Ambulatory | Lung function | 35 | +0.0 | +0.0 | -0.2 | -0.1 | +0.0 |
| Ambulatory | Lung function | 47 | -0.2 | +0.0 | -0.5 | -0.1 | +0.1 |
| Ambulatory | Lung function | 55 | +0.3 | +0.6 | +0.6 | +0.5 | +0.9 |

## Four-unit ambulatory evidence

Each post hoc fit uses the same fixed ambulatory arterial distribution. Four fits each omit one lung-function study. The fifth retains all four and reduces the half-normal prior scale for between-study log-variance heterogeneity from 0.6 to 0.4. Retained draws are per chain. ESS denotes effective sample size. [Contrasts recorded before the runs](manuscript-contrasts.json)

| Variant | Retained/chain | Divergences | Max R-hat | Min bulk ESS | Min tail ESS | Sampler |
| --- | --- | --- | --- | --- | --- | --- |
| Omit Chhajed | 131072 | 0 | 1.00001 | 215890 | 190287 | passed |
| Omit Domingo | 131072 | 0 | 1.00002 | 208223 | 187497 | passed |
| Omit Ekkernkamp | 262144 | 0 | 1.00001 | 387805 | 355711 | passed |
| Omit Maniscalco | 131072 | 0 | 1.00001 | 172829 | 163050 | passed |
| Log-variance heterogeneity prior scale 0.4 | 131072 | 0 | 1.00002 | 241288 | 208040 | passed |

Changes compare each variant with the original four-study ambulatory fit at the same TcCO₂.

| Variant | TcCO₂ | Median | 95% PI | ≥45 | ≥52 | Δ lower | Δ median | Δ upper | Δ ≥45 (pp) | Δ ≥52 (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Omit Chhajed | 35 | 36 | 30 to 44 | 2% | <1% | +0.0 | +0.0 | +0.4 | +0.4 | +0.0 |
| Omit Chhajed | 47 | 45 | 37 to 53 | 59% | 4% | +0.0 | +0.2 | +0.8 | +1.4 | +0.7 |
| Omit Chhajed | 55 | 53 | 40 to 61 | 92% | 62% | -1.0 | +0.0 | +0.0 | -1.0 | +0.2 |
| Omit Domingo | 35 | 36 | 29 to 44 | 2% | <1% | -1.0 | +0.0 | +0.4 | +0.3 | +0.0 |
| Omit Domingo | 47 | 45 | 36 to 52 | 53% | 3% | -1.0 | +0.0 | -0.2 | -5.0 | -0.3 |
| Omit Domingo | 55 | 52 | 39 to 61 | 90% | 55% | -2.0 | -1.0 | -0.3 | -2.7 | -6.1 |
| Omit Ekkernkamp | 35 | 36 | 30 to 43 | 1% | <1% | +0.0 | +0.0 | -0.6 | -0.3 | -0.0 |
| Omit Ekkernkamp | 47 | 45 | 37 to 52 | 56% | 3% | +0.0 | +0.0 | -0.2 | -2.2 | -0.6 |
| Omit Ekkernkamp | 55 | 53 | 41 to 60 | 93% | 59–60% | +0.0 | +0.0 | -1.0 | -0.1 | -2.0 |
| Omit Maniscalco | 35 | 37 | 30 to 44 | 2% | <1% | -0.3 | +0.6 | +0.4 | +0.7 | +0.0 |
| Omit Maniscalco | 47 | 45 | 36 to 53 | 56% | 5–6% | -1.0 | +0.0 | +0.9 | -2.2 | +1.7 |
| Omit Maniscalco | 55 | 52 | 40 to 61 | 90% | 56% | -1.0 | -1.0 | +0.2 | -2.8 | -5.5 |
| Log-variance heterogeneity prior scale 0.4 | 35 | 36 | 30 to 43 | 1% | <1% | +0.0 | +0.0 | -0.6 | -0.3 | -0.0 |
| Log-variance heterogeneity prior scale 0.4 | 47 | 45 | 38 to 52 | 58–59% | 4% | +1.0 | +0.0 | -0.2 | +0.6 | -0.1 |
| Log-variance heterogeneity prior scale 0.4 | 55 | 53 | 42 to 61 | 95% | 63% | +1.0 | +0.0 | +0.0 | +1.8 | +1.9 |

For each reading, the final table shows the minimum and maximum estimates across these five variants.

| TcCO₂ | Lower limit range | Median range | Upper limit range | P(≥45) range, % | P(≥52) range, % |
| --- | --- | --- | --- | --- | --- |
| 35 | 29.0–30.0 | 36.0–36.6 | 43.0–44.0 | 1.2–2.1 | 0.0–0.1 |
| 47 | 36.0–38.0 | 45.0–45.2 | 52.0–53.1 | 52.9–59.3 | 3.1–5.5 |
| 55 | 39.0–42.0 | 52.0–53.0 | 60.0–61.2 | 89.9–94.6 | 55.4–63.3 |

## Modeled strategy differences

The following contrasts subtract the single-threshold errors from the two-stage errors within each posterior draw. Both strategies use the same eligible testing occasions. The reference ABG is assumed timely and correct.

| Population | Difference in false negatives per 100 (95% CrI) | Difference in false positives per 100 (95% CrI) |
| --- | --- | --- |
| All settings | -3.21 (-3.60–-2.85) | -6.25 (-7.14–-5.42) |
| Critical Care Billing | -2.86 (-3.57–-2.20) | -5.68 (-7.55–-4.30) |
| ED/inpatient | -4.28 (-5.16–-3.40) | -4.71 (-6.91–-3.23) |
| Ambulatory | -3.69 (-5.65–-2.04) | -5.29 (-10.40–-2.75) |
