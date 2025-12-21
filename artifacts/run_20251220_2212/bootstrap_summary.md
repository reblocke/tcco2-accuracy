# Bootstrap LoA spread summary

Bootstrap draws: 1000 per subgroup (seed=12345).

LoA bounds shown as 2.5/50/97.5% bootstrap quantiles;
Conway CI shown as reported outer CI bounds.

| Group | LoA L q2.5 | LoA L q50 | LoA L q97.5 | LoA U q2.5 | LoA U q50 | LoA U q97.5 | Conway CI L | Conway CI U |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | -8.62 | -6.93 | -4.69 | 4.25 | 6.81 | 8.43 | -15.06 | 14.89 |
| icu | -6.32 | -5.27 | -4.45 | 2.90 | 4.18 | 7.00 | -7.33 | 6.14 |
| arf | -4.91 | -3.66 | -2.69 | 4.55 | 6.94 | 10.00 | -7.82 | 11.21 |
| lft | -4.20 | -4.01 | -2.46 | 1.89 | 3.93 | 5.03 | -7.36 | 7.26 |

Qualitative check: bootstrap LoA quantile ranges span a similar
scale to Conway's outer CI bounds across subgroups.