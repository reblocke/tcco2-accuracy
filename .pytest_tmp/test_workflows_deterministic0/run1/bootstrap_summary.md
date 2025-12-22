# Bootstrap LoA spread summary

Bootstrap draws: 25 per subgroup (seed=123).
Bootstrap mode: cluster_plus_withinstudy.

LoA bounds shown as 2.5/50/97.5% bootstrap quantiles;
Conway CI shown as reported outer CI bounds.

| Group | LoA L q2.5 | LoA L q50 | LoA L q97.5 | LoA U q2.5 | LoA U q50 | LoA U q97.5 | Conway CI L | Conway CI U | Width ratio | Width gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | -8.51 | -7.05 | -4.97 | 4.32 | 7.03 | 8.06 | -15.04 | 14.81 | 0.56 | 13.27 |
| icu | -6.20 | -5.58 | -4.58 | 3.18 | 5.04 | 7.05 | -7.33 | 6.14 | 0.98 | 0.22 |
| arf | -5.17 | -3.70 | -2.86 | 4.49 | 6.16 | 9.40 | -7.98 | 11.42 | 0.75 | 4.83 |
| lft | -4.47 | -4.00 | -2.56 | 1.82 | 3.62 | 5.23 | -7.36 | 7.26 | 0.66 | 4.92 |

Width interpretation (bootstrap vs Conway outer CI):
- main: materially narrower than Conway CI.
- icu: comparable to Conway CI.
- arf: materially narrower than Conway CI.
- lft: materially narrower than Conway CI.