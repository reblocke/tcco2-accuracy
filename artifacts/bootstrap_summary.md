# Bootstrap LoA spread summary

Bootstrap draws: 1000 per subgroup (seed=202401).
Bootstrap mode: cluster_plus_withinstudy.
Agreement method version: `agreement_natural_log_tau2_direct_v1`.
Results status: `provisional`.

LoA bounds shown as 2.5/50/97.5% bootstrap quantiles;
corrected analytic CI shown as outer CI bounds from the same method revision.

| Group | LoA L q2.5 | LoA L q50 | LoA L q97.5 | LoA U q2.5 | LoA U q50 | LoA U q97.5 | Corrected analytic CI L | Corrected analytic CI U | Width ratio | Width gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main | -11.36 | -9.95 | -8.52 | 7.86 | 9.73 | 11.26 | -11.28 | 11.05 | 1.01 | -0.28 |
| icu | -12.32 | -10.47 | -8.74 | 7.15 | 9.24 | 12.40 | -12.35 | 11.16 | 1.05 | -1.20 |
| arf | -12.11 | -8.77 | -6.43 | 8.41 | 11.93 | 17.04 | -12.40 | 15.78 | 1.03 | -0.98 |
| lft | -8.36 | -6.06 | -4.65 | 2.98 | 6.05 | 9.39 | -11.16 | 11.07 | 0.80 | 4.48 |

Width interpretation (bootstrap vs corrected analytic outer CI):
- main: comparable to the corrected analytic CI.
- icu: comparable to the corrected analytic CI.
- arf: comparable to the corrected analytic CI.
- lft: materially narrower than the corrected analytic CI.