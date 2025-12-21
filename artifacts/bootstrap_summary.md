# Bootstrap LoA spread summary

Bootstrap draws: 1000 per subgroup (seed=202401).

LoA bounds shown as 2.5/50/97.5% bootstrap quantiles;
Conway CI shown as reported outer CI bounds.

| Group | LoA L q2.5 | LoA L q50 | LoA L q97.5 | LoA U q2.5 | LoA U q50 | LoA U q97.5 | Conway CI L | Conway CI U |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | -8.47 | -6.94 | -4.68 | 4.23 | 6.78 | 8.51 | -15.06 | 14.89 |
| icu | -6.38 | -5.31 | -4.47 | 2.81 | 4.15 | 7.05 | -7.33 | 6.14 |
| arf | -4.97 | -3.62 | -2.62 | 4.49 | 6.97 | 9.95 | -7.82 | 11.21 |
| lft | -4.20 | -4.01 | -2.46 | 1.89 | 3.93 | 5.03 | -7.36 | 7.26 |

Qualitative check: bootstrap LoA quantile ranges span a similar
scale to Conway's outer CI bounds across subgroups.