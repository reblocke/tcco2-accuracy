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
| ed_inp | 35 | 36.63 [31.25, 42.62] | 0.004 |
| ed_inp | 45 | 46.63 [41.25, 52.62] | 0.724 |
| ed_inp | 55 | 56.63 [51.25, 62.62] | 1.000 |
| icu | 35 | 34.44 [29.42, 39.80] | 0.000 |
| icu | 45 | 44.44 [39.42, 49.80] | 0.413 |
| icu | 55 | 54.44 [49.42, 59.80] | 1.000 |

## Prior-weighted (empirical PaCO2 prior)

| Group | TcCO2 | PaCO2 median [PI] | P(PaCO2≥45) |
| --- | --- | --- | --- |
| pft | 35 | 35.10 [32.00, 39.00] | 0.000 |
| pft | 45 | 44.00 [41.00, 48.00] | 0.450 |
| pft | 55 | 54.00 [51.00, 58.00] | 1.000 |
| ed_inp | 35 | 37.00 [32.00, 42.00] | 0.004 |
| ed_inp | 45 | 46.00 [41.00, 51.10] | 0.673 |
| ed_inp | 55 | 56.00 [50.00, 61.60] | 1.000 |
| icu | 35 | 35.00 [30.00, 40.00] | 0.000 |
| icu | 45 | 44.00 [39.00, 49.00] | 0.379 |
| icu | 55 | 54.00 [48.90, 59.00] | 1.000 |