# TcCO2 → PaCO2 UI overview

Research-only static GitHub Pages UI for TcCO2-based PaCO2 inference.

| Setting | Default |
| --- | --- |
| Setting | All (Conway main + pooled prior) |
| TcCO2 input | 50 mmHg |
| Inference mode | Prior-weighted |
| Hypercapnia threshold | 45 mmHg |
| Prediction interval | 95% PI |
| Prior source | `Data/paco2_prior_bins.csv` |
| Runtime | Pyodide worker running staged Python |

The UI reports a PaCO2 prediction interval (PI), not a confidence interval (CI).
The repo-shipped prior bins and canonical bootstrap asset keep deployments portable
(no in-silico .dta needed); "All" pools subgroups weighted by their counts.
