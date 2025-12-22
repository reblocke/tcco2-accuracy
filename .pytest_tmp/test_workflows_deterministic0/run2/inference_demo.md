# TcCO2 → PaCO2 inference demo

Bootstrap draws: 25 per subgroup (seed=123).
Parameter draws: 10 draws per subgroup.
Subgroup priors: empirical PaCO2 distributions for pft (n=10456), ed_inp (n=120538), icu (n=66704).
Interval type: 95% prediction interval (PI), not CI.

## Likelihood-only (bootstrap mixture)

| Group | TcCO2 | PaCO2 median [PI] | P(PaCO2≥45) |
| --- | --- | --- | --- |
| pft | 35 | 34.94 [31.08, 38.91] | 0.000 |
| pft | 45 | 44.94 [41.08, 48.91] | 0.488 |
| pft | 55 | 54.94 [51.08, 58.91] | 1.000 |
| ed_inp | 35 | 36.35 [31.21, 42.26] | 0.004 |
| ed_inp | 45 | 46.35 [41.21, 52.26] | 0.701 |
| ed_inp | 55 | 56.35 [51.21, 62.26] | 1.000 |
| icu | 35 | 34.92 [29.41, 40.79] | 0.001 |
| icu | 45 | 44.92 [39.41, 50.79] | 0.489 |
| icu | 55 | 54.92 [49.41, 60.79] | 1.000 |

## Prior-weighted (empirical PaCO2 prior)

| Group | TcCO2 | PaCO2 median [PI] | P(PaCO2≥45) |
| --- | --- | --- | --- |
| pft | 35 | 35.20 [32.00, 39.00] | 0.000 |
| pft | 45 | 44.00 [41.00, 48.00] | 0.455 |
| pft | 55 | 54.00 [50.90, 58.00] | 1.000 |
| ed_inp | 35 | 37.00 [32.00, 42.00] | 0.003 |
| ed_inp | 45 | 46.00 [40.90, 51.00] | 0.659 |
| ed_inp | 55 | 55.40 [50.00, 61.00] | 1.000 |
| icu | 35 | 35.00 [30.00, 41.00] | 0.001 |
| icu | 45 | 44.00 [39.00, 50.00] | 0.437 |
| icu | 55 | 54.00 [48.30, 60.00] | 0.999 |