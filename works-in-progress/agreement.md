# Agreement for PaCO₂ − TcCO₂ (mmHg)

All intervals use a nominal 95% level and mmHg. The published and corrected reference limits summarize a normal reference distribution. Their outer envelopes use the lower confidence bound for the lower endpoint and the upper confidence bound for the upper endpoint. The new interval averages over one observation in a comparable new setting and posterior uncertainty.

| Agreement study group | Published limits (mmHg) | Published outer envelope (mmHg) | Corrected limits (mmHg) | Corrected outer envelope (mmHg) | Future-pair 95% prediction interval (mmHg) |
| --- | --- | --- | --- | --- | --- |
| Overall | -7.1 to 6.9 | -15.1 to 14.9 | -10.1 to 9.9 | -11.3 to 11.1 | -11.5 to 11.2 |
| ICU | -5.4 to 4.2 | -7.3 to 6.1 | -10.4 to 9.2 | -12.4 to 11.2 | -11.2 to 11.1 |
| Acute respiratory failure | -3.7 to 7.1 | -7.8 to 11.2 | -8.7 to 12.1 | -12.4 to 15.8 | -12.0 to 17.9 |
| Lung function | -4.0 to 3.9 | -7.3 to 7.3 | -6.1 to 6.0 | -11.2 to 11.1 | -8.9 to 9.6 |

The corrected overall union contains 76 rows, including the ICU-only Bolliger cohort. The new overall fit uses 75 adjudicated units from 73 publications. The Word supplement gives variance components in mmHg² and interval widths in mmHg. [Source ledger](../models/future_setting/README.md)

## Corrections on the same 75 archived rows

The outer envelope spans the lower confidence bound for the estimated lower agreement limit through the upper confidence bound for the estimated upper limit. Each endpoint has its own approximate 95% confidence interval. The last column of the first table instead predicts a single difference in a comparable new setting.

| Calculation | Agreement limits (mmHg) | Outer envelope (mmHg) |
| --- | --- | --- |
| Archived calculation | −7.10 to +6.93 | −15.06 to +14.89 |
| Use natural logs throughout | −10.08 to +9.91 | −15.74 to +15.57 |
| Correct the between-study variance term in the endpoint calculation | −7.10 to +6.93 | −8.18 to +8.01 |
| Both corrections | −10.08 to +9.91 | −11.24 to +11.07 |

The archived code used a base-10 logarithm with a natural exponential back-transformation. Consistent natural logs raise the pooled within-study SD from 1.86 to 4.02 mmHg. The endpoint correction removes an extra squared-variance multiplier when propagating uncertainty in the estimated between-study variance. With both corrections the endpoint SE is 0.5827 mmHg. Multiplying by 1.9925 adds 1.161 mmHg per side. Independent derivation of the variance terms and critical value is requested. [Reviewer questions](reviewer-questions.md) · [Replay on the 75 archive rows](overall-envelope.R)
