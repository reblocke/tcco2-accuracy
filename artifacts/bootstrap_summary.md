# Bootstrap LoA spread summary

Bootstrap draws: 1000 per subgroup (seed=202401).
Bootstrap mode: cluster_plus_withinstudy.

LoA bounds shown as 2.5/50/97.5% bootstrap quantiles;
Conway CI shown as reported outer CI bounds.

| Group | LoA L q2.5 | LoA L q50 | LoA L q97.5 | LoA U q2.5 | LoA U q50 | LoA U q97.5 | Conway CI L | Conway CI U | Width ratio | Width gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | -8.58 | -7.05 | -4.79 | 4.32 | 6.85 | 8.48 | -15.04 | 14.81 | 0.57 | 12.79 |
| icu | -6.82 | -5.59 | -4.47 | 2.84 | 4.37 | 7.22 | -7.33 | 6.14 | 1.04 | -0.56 |
| arf | -5.32 | -3.71 | -2.62 | 4.56 | 6.96 | 10.13 | -7.82 | 11.21 | 0.81 | 3.58 |
| lft | -4.47 | -3.96 | -2.48 | 1.84 | 3.86 | 5.25 | -7.36 | 7.26 | 0.66 | 4.90 |

Width interpretation (bootstrap vs Conway outer CI):
- main: materially narrower than Conway CI.
- icu: comparable to Conway CI.
- arf: comparable to Conway CI.
- lft: materially narrower than Conway CI.