# TcCO2 Accuracy — Decisions

## Open decisions
- None yet.

## Logged divergences
- `logs2` inputs are recomputed as `log10(S2*) + 1/(n_2 - 1)` with `v_logs2 = 2/(n_2 - 1)` to match `Conway Meta/data.dta`, even though `Code/3_tcco2_uncertainty_and_simulation_do.do:94` describes `logs2` as a natural log.
- Bootstrap τ² draws are truncated at 0 to enforce non-negative between-study variance in `python/src/tcco2_accuracy/bootstrap.py:33`.
- PaCO2 subgroup assignment follows `docs/SPEC.md:11-19`, which includes ED in `ed_inp` by construction; this differs from `Code/2_trinetx_cleaning_do.do:8-13`, where `ed_inp_group` excludes ED (`is_emer==0`) and requires `cc_time==0`.
- Forward simulation maps PaCO2 subgroups to Conway bootstrap groups via `python/src/tcco2_accuracy/simulation.py:23` (pft→lft, ed_inp→arf, icu→icu) to align ambulatory and acute respiratory failure sub-analyses.
