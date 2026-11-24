# Cohort and paired predictions

| Setting | Eligible patients | Eligible encounters | Eligible arterial result records | First-day encounters | First-day arterial result records |
| --- | --- | --- | --- | --- | --- |
| All settings | 142,156 | 147,028 | 1,019,099 | 142,424 | 385,867 |
| ICU proxy | 52,931 | 53,985 | 535,791 | 52,968 | 141,818 |
| ED / inpatient | 77,119 | 78,561 | 418,884 | 77,146 | 222,358 |
| Ambulatory | 8,174 | 8,203 | 13,228 | 8,178 | 13,059 |

Each patient contributes total weight one within the target. Settings can overlap across a patient’s encounters. All uses the earliest eligible day across the entire cohort.

| Characteristic | All settings | ICU proxy | ED / inpatient | Ambulatory |
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

| Setting | Median | Central 95% range (mmHg) | PaCO₂ ≥45 | PaCO₂ ≥52 |
| --- | --- | --- | --- | --- |
| All settings | 40 | 23–90 | 29.39% | 15.25% |
| ICU proxy | 39.4 | 22–108 | 31.63% | 18.83% |
| ED / inpatient | 40 | 24–77.1 | 27.81% | 12.89% |
| Ambulatory | 40 | 26–64 | 26.52% | 10.12% |

## Given observed TcCO₂

| Setting | TcCO₂ | Median PaCO₂ | 95% prediction interval | PaCO₂ ≥45 | PaCO₂ ≥52 |
| --- | --- | --- | --- | --- | --- |
| All settings | 35 | 36 | ~~28–44.1~~ **28–44** | ~~2.29%~~ **2%** | ~~0.14%~~ **<1%** |
| All settings | 40 | 39 | ~~31–47.7~~ **31–48** | ~~8.65%~~ **9%** | ~~0.46%~~ **<1%** |
| All settings | 45 | 43 | ~~34–51.8~~ **34–52** | ~~32.28%~~ **32%** | ~~2.47%~~ **2%** |
| All settings | 47 | 44 | ~~35–53.2~~ **35–53** | ~~46.96%~~ **47%** | ~~5.18%~~ **5%** |
| All settings | 50 | 47 | ~~36.1–56.3~~ **36–56** | ~~67.34%~~ **67%** | ~~14.78%~~ **15%** |
| All settings | 55 | ~~51.6~~ **52** | ~~38.8–62~~ **39–62** | ~~86.75%~~ **87%** | ~~49.36%~~ **49%** |
| ICU proxy | 35 | ~~35.4~~ **35** | ~~27–44.2~~ **27–44** | ~~2.28%~~ **2%** | ~~0.12%~~ **<1%** |
| ICU proxy | 40 | 39 | 31–48 | ~~9.11%~~ **9%** | ~~0.52%~~ **1%** |
| ICU proxy | 45 | ~~42.8~~ **43** | 34–52 | ~~31.82%~~ **32%** | ~~2.96%~~ **3%** |
| ICU proxy | 47 | 44 | 35–54 | ~~45.95%~~ **46%** | ~~6.00%~~ **6%** |
| ICU proxy | 50 | 47 | ~~36.9–57~~ **37–57** | ~~66.99%~~ **67%** | ~~15.92%~~ **16%** |
| ICU proxy | 55 | 52 | 40–63 | ~~88.92%~~ **89%** | ~~50.62%~~ **51%** |
| ED / inpatient | 35 | 37 | 29–47 | ~~5.77%~~ **6%** | ~~0.56%~~ **1%** |
| ED / inpatient | 40 | 40 | 32–50 | ~~15.36%~~ **15%** | ~~1.44%~~ **1%** |
| ED / inpatient | 45 | 44 | 34–54 | ~~41.02%~~ **41%** | ~~5.42%~~ **5%** |
| ED / inpatient | 47 | 45 | 35–56 | ~~54.03%~~ **54%** | ~~9.65%~~ **10%** |
| ED / inpatient | 50 | ~~47.4~~ **47** | 36–59 | ~~70.02%~~ **70%** | ~~21.63%~~ **22%** |
| ED / inpatient | 55 | 52 | 37–65 | ~~83.93%~~ **84%** | ~~53.17%~~ **53%** |
| Ambulatory | 35 | ~~Held~~ **36** | ~~Held~~ **30–44** | ~~1.48%~~ **1%** | ~~0.08%~~ **<1%** |
| Ambulatory | 40 | 40 | 33–47 | ~~6.59%~~ **7%** | ~~0.24%~~ **<1%** |
| Ambulatory | 45 | ~~Held~~ **44** | ~~Held~~ **36–51** | ~~36.98%~~ **37%** | ~~1.55%~~ **2%** |
| Ambulatory | 47 | ~~Held~~ **45** | ~~Held~~ **37–52** | ~~57.89%~~ **58%** | ~~3.76%~~ **4%** |
| Ambulatory | 50 | ~~Held~~ **48** | ~~Held~~ **39–55** | ~~79.74%~~ **80%** | ~~14.42%~~ **14%** |
| Ambulatory | 55 | 53 | 41–61 | ~~92.75%~~ **93%** | ~~Held~~ **61%** |

Pressures are in mmHg. Held denotes a result that did not meet the numerical precision criteria.

| Qualified readings out of 81 | Interval | ≥45 | ≥52 |
| --- | --- | --- | --- |
| All settings | ~~69~~ **81** | 81 | 81 |
| ICU proxy | ~~48~~ **81** | 81 | 81 |
| ED / inpatient | ~~62~~ **81** | 81 | 81 |
| Ambulatory | ~~51~~ **81** | 81 | ~~76~~ **81** |
