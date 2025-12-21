# TcCO2 → PaCO2 inference demo

Bootstrap draws: 1000 per subgroup (seed=202401).
Parameter draws: full bootstrap set per subgroup.
Subgroup priors: empirical PaCO2 distributions for pft (n=10456), ed_inp (n=120538), icu (n=66704).
Interval type: 95% prediction interval (PI), not CI.

## Likelihood-only (bootstrap mixture)

| Group | TcCO2 | PaCO2 median [PI] | P(PaCO2≥45) |
| --- | --- | --- | --- |
| pft | 35 | 34.92 [31.17, 38.91] | 0.000 |
| pft | 45 | 44.92 [41.17, 48.91] | 0.485 |
| pft | 55 | 54.92 [51.17, 58.91] | 1.000 |
| ed_inp | 35 | 36.60 [31.37, 42.47] | 0.004 |
| ed_inp | 45 | 46.60 [41.37, 52.47] | 0.725 |
| ed_inp | 55 | 56.60 [51.37, 62.47] | 1.000 |
| icu | 35 | 34.44 [29.71, 39.55] | 0.000 |
| icu | 45 | 44.44 [39.71, 49.55] | 0.409 |
| icu | 55 | 54.44 [49.71, 59.55] | 1.000 |

## Prior-weighted (empirical PaCO2 prior)

| Group | TcCO2 | PaCO2 median [PI] | P(PaCO2≥45) |
| --- | --- | --- | --- |
| pft | 35 | 35.10 [32.00, 39.00] | 0.000 |
| pft | 45 | 44.00 [41.00, 48.00] | 0.453 |
| pft | 55 | 54.00 [51.00, 58.00] | 1.000 |
| ed_inp | 35 | 37.00 [32.00, 42.00] | 0.003 |
| ed_inp | 45 | 46.00 [41.00, 51.00] | 0.678 |
| ed_inp | 55 | 56.00 [50.00, 61.20] | 1.000 |
| icu | 35 | 35.00 [30.00, 39.60] | 0.000 |
| icu | 45 | 44.00 [39.00, 49.00] | 0.383 |
| icu | 55 | 54.00 [49.00, 59.00] | 1.000 |