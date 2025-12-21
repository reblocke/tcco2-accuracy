# TcCO2 → PaCO2 inference demo

Bootstrap draws: 1000 per subgroup (seed=12345).
Parameter draws: full bootstrap set per subgroup.
Subgroup priors: empirical PaCO2 distributions for pft (n=10456), ed_inp (n=120538), icu (n=66704).
Interval type: 95% prediction interval (PI), not CI.

## Likelihood-only (bootstrap mixture)

| Group | TcCO2 | PaCO2 median [PI] | P(PaCO2≥45) |
| --- | --- | --- | --- |
| pft | 35 | 34.91 [31.17, 38.92] | 0.000 |
| pft | 45 | 44.91 [41.17, 48.92] | 0.482 |
| pft | 55 | 54.91 [51.17, 58.92] | 1.000 |
| ed_inp | 35 | 36.59 [31.34, 42.41] | 0.003 |
| ed_inp | 45 | 46.59 [41.34, 52.41] | 0.723 |
| ed_inp | 55 | 56.59 [51.34, 62.41] | 1.000 |
| icu | 35 | 34.46 [29.73, 39.55] | 0.000 |
| icu | 45 | 44.46 [39.73, 49.55] | 0.412 |
| icu | 55 | 54.46 [49.73, 59.55] | 1.000 |

## Prior-weighted (empirical PaCO2 prior)

| Group | TcCO2 | PaCO2 median [PI] | P(PaCO2≥45) |
| --- | --- | --- | --- |
| pft | 35 | 35.10 [32.00, 39.00] | 0.000 |
| pft | 45 | 44.00 [41.00, 48.00] | 0.451 |
| pft | 55 | 54.00 [51.00, 58.00] | 1.000 |
| ed_inp | 35 | 37.00 [32.00, 42.00] | 0.003 |
| ed_inp | 45 | 46.00 [41.00, 51.00] | 0.676 |
| ed_inp | 55 | 56.00 [50.00, 61.10] | 1.000 |
| icu | 35 | 35.00 [30.00, 39.50] | 0.000 |
| icu | 45 | 44.00 [39.00, 49.00] | 0.386 |
| icu | 55 | 54.00 [49.00, 59.00] | 1.000 |