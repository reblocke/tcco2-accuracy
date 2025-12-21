# Bootstrap LoA spread summary

Bootstrap draws: 1000 per subgroup (seed=202401).
Bootstrap mode: cluster_plus_withinstudy.

LoA bounds shown as 2.5/50/97.5% bootstrap quantiles;
Conway CI shown as reported outer CI bounds.

| Group | LoA L q2.5 | LoA L q50 | LoA L q97.5 | LoA U q2.5 | LoA U q50 | LoA U q97.5 | Conway CI L | Conway CI U | Width ratio | Width gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | -8.67 | -6.99 | -4.77 | 4.34 | 6.92 | 8.53 | -15.06 | 14.89 | 0.57 | 12.74 |
| icu | -6.83 | -5.56 | -4.53 | 2.89 | 4.40 | 7.24 | -7.33 | 6.14 | 1.04 | -0.59 |
| arf | -5.29 | -3.69 | -2.70 | 4.58 | 7.06 | 10.09 | -7.82 | 11.21 | 0.81 | 3.65 |
| lft | -4.47 | -3.96 | -2.48 | 1.84 | 3.86 | 5.25 | -7.36 | 7.26 | 0.66 | 4.90 |

Width interpretation (bootstrap vs Conway outer CI):
- main: materially narrower than Conway CI.
- icu: comparable to Conway CI.
- arf: comparable to Conway CI.
- lft: materially narrower than Conway CI.