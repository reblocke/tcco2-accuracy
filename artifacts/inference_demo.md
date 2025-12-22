# TcCO2 → PaCO2 inference demo

Bootstrap draws: 1000 per subgroup (seed=202401).
Parameter draws: full bootstrap set per subgroup.
Subgroup priors: empirical PaCO2 distributions for pft (n=10456), ed_inp (n=120538), icu (n=66704).
Interval type: 95% prediction interval (PI), not CI.

## Likelihood-only (bootstrap mixture)

| Group | TcCO2 | PaCO2 median [PI] | P(PaCO2≥45) |
| --- | --- | --- | --- |
| pft | 35 | 34.91 [31.13, 38.95] | 0.000 |
| pft | 45 | 44.91 [41.13, 48.95] | 0.483 |
| pft | 55 | 54.91 [51.13, 58.95] | 1.000 |
| ed_inp | 35 | 36.67 [31.32, 42.69] | 0.005 |
| ed_inp | 45 | 46.67 [41.32, 52.69] | 0.730 |
| ed_inp | 55 | 56.67 [51.32, 62.69] | 1.000 |
| icu | 35 | 34.42 [29.42, 39.77] | 0.000 |
| icu | 45 | 44.42 [39.42, 49.77] | 0.409 |
| icu | 55 | 54.42 [49.42, 59.77] | 1.000 |

## Prior-weighted (empirical PaCO2 prior)

| Group | TcCO2 | PaCO2 median [PI] | P(PaCO2≥45) |
| --- | --- | --- | --- |
| pft | 35 | 35.10 [32.00, 39.00] | 0.000 |
| pft | 45 | 44.00 [41.00, 48.00] | 0.450 |
| pft | 55 | 54.00 [51.00, 58.00] | 1.000 |
| ed_inp | 35 | 37.00 [32.00, 42.20] | 0.004 |
| ed_inp | 45 | 46.00 [41.00, 51.30] | 0.680 |
| ed_inp | 55 | 56.00 [50.00, 61.90] | 1.000 |
| icu | 35 | 35.00 [30.00, 40.00] | 0.000 |
| icu | 45 | 44.00 [39.00, 49.00] | 0.376 |
| icu | 55 | 54.00 [49.00, 59.00] | 1.000 |