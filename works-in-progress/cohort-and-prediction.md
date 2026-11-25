# Cohort and paired predictions

| Setting | Eligible patients | Eligible encounters | Eligible arterial result records | First-day encounters | First-day arterial result records |
| --- | --- | --- | --- | --- | --- |
| All settings | 142,156 | 147,028 | 1,019,099 | 142,424 | 385,867 |
| Critical Care Billing | 52,931 | 53,985 | 535,791 | 52,968 | 141,818 |
| ED / inpatient | 77,119 | 78,561 | 418,884 | 77,146 | 222,358 |
| Ambulatory | 8,174 | 8,203 | 13,228 | 8,178 | 13,059 |

Each patient contributes total weight one within the target. Settings can overlap across a patient’s encounters. All uses the earliest eligible day across the entire cohort.

| Characteristic | All settings | Critical Care Billing | ED / inpatient | Ambulatory |
| --- | --- | --- | --- | --- |
| Age, median (IQR) | 64 (52–74) | 65 (53–74) | 64 (52–74) | 64 (53–72) |
| Female | 43.5% | 42.4% | 44.6% | 43.4% |
| Male | 56.5% | 57.6% | 55.4% | 56.6% |
| White race | 63.8% | 65.9% | 63.3% | 60.2% |
| Black race | 16.7% | 19.6% | 15.3% | 8.8% |
| Asian race | 2.1% | 3.0% | 1.7% | 1.9% |
| Hispanic ethnicity | 6.5% | 7.8% | 5.3% | 4.6% |
| COPD | 18.1% | 18.4% | 17.8% | 23.6% |
| Asthma | 10.3% | 9.9% | 10.1% | 14.2% |
| Obstructive sleep apnea | 15.7% | 14.6% | 15.7% | 27.0% |
| Heart failure | 20.8% | 22.7% | 19.9% | 23.5% |
| Chronic kidney disease | 18.5% | 19.9% | 18.2% | 16.5% |
| Diabetes | 26.5% | 27.4% | 26.5% | 27.0% |
| Opioid use disorder | 4.2% | 4.6% | 3.9% | 3.5% |
| Pulmonary hypertension | 9.6% | 9.9% | 9.2% | 14.5% |

Age and coded characteristics use first-day arterial result weights and the associated encounter. Percentages use nonmissing values. Race and ethnicity are separate recorded indicators.

## Before observing TcCO₂

These are the patient-weighted arterial distributions before using a transcutaneous reading. Within each setting each patient contributes total weight one across eligible arterial results on their first testing day.

| Setting | Median | Central 95% range (mmHg) | PaCO₂ ≥45 | PaCO₂ ≥52 |
| --- | --- | --- | --- | --- |
| All settings | 40 | 23–90 | 29.39% | 15.25% |
| Critical Care Billing | 39.4 | 22–108 | 31.63% | 18.83% |
| ED / inpatient | 40 | 24–77.1 | 27.81% | 12.89% |
| Ambulatory | 40 | 26–64 | 26.52% | 10.12% |

## Given observed TcCO₂

Each row predicts an arterial measurement made at the same time. In All settings at TcCO₂ 47 mmHg, the predicted median is 44 mmHg, the 95% prediction interval is 35–53 mmHg, and the chances of PaCO₂ ≥45 and ≥52 are 47% and 5%. Every result ≥52 also counts toward ≥45.

| Setting | TcCO₂ (mmHg) | Median PaCO₂ (mmHg) | 95% prediction interval (mmHg) | PaCO₂ ≥45 | PaCO₂ ≥52 |
| --- | --- | --- | --- | --- | --- |
| All settings | 35 | 36 | 28–44 | 2% | <1% |
| All settings | 40 | 39 | 31–48 | 9% | <1% |
| All settings | 45 | 43 | 34–52 | 32% | 2% |
| All settings | 47 | 44 | 35–53 | 47% | 5% |
| All settings | 50 | 47 | 36–56 | 67% | 15% |
| All settings | 55 | 52 | 39–62 | 87% | 49% |
| Critical Care Billing | 35 | 35 | 27–44 | 2% | <1% |
| Critical Care Billing | 40 | 39 | 31–48 | 9% | 1% |
| Critical Care Billing | 45 | 43 | 34–52 | 32% | 3% |
| Critical Care Billing | 47 | 44 | 35–54 | 46% | 6% |
| Critical Care Billing | 50 | 47 | 37–57 | 67% | 16% |
| Critical Care Billing | 55 | 52 | 40–63 | 89% | 51% |
| ED / inpatient | 35 | 37 | 29–47 | 6% | 1% |
| ED / inpatient | 40 | 40 | 32–50 | 15% | 1% |
| ED / inpatient | 45 | 44 | 34–54 | 41% | 5% |
| ED / inpatient | 47 | 45 | 35–56 | 54% | 10% |
| ED / inpatient | 50 | 47 | 36–59 | 70% | 22% |
| ED / inpatient | 55 | 52 | 37–65 | 84% | 53% |
| Ambulatory | 35 | 36 | 30–44 | 1% | <1% |
| Ambulatory | 40 | 40 | 33–47 | 7% | <1% |
| Ambulatory | 45 | 44 | 36–51 | 37% | 2% |
| Ambulatory | 47 | 45 | 37–52 | 58% | 4% |
| Ambulatory | 50 | 48 | 39–55 | 80% | 14% |
| Ambulatory | 55 | 53 | 41–61 | 93% | 61% |

In each setting, the interval and both threshold probabilities met the numerical reporting criteria at all 81 integer readings from 20 through 100 mmHg. Individual-patient coverage requires a separate study with paired measurements. [Precision audit and earlier reporting counts](reviewer-questions.md#numerical-reporting-audit)
