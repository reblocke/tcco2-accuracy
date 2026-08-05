# TcCO2 → PaCO2 UI overview

Research-only static GitHub Pages UI for TcCO2-based PaCO2 inference.

| Setting | Default |
| --- | --- |
| Setting | All (Conway main agreement parameters) |
| TcCO2 input | 50 mmHg |
| Inference mode | Likelihood-only |
| Hypercapnia threshold | 45 mmHg |
| Prediction interval | 95% PI |
| Prior source | None by default; user upload required for prior-weighted mode |
| Runtime | Pyodide worker running staged Python |
| Agreement method | `agreement_natural_log_tau2_direct_v1` (provisional) |

The UI reports a PaCO2 prediction interval (PI), not a confidence interval (CI).
The canonical bootstrap asset keeps default likelihood-only inference portable. The browser does
not ship or fetch a restricted-derived PaCO2 prior. Prior-weighted inference is available only
after the user uploads a binned prior, which remains client-side.

Corrected browser outputs remain provisional pending independent biostatistical
review. PaCO2-dependent manuscript outputs are frozen at the legacy agreement
method. This research app is not intended for clinical decision-making.
