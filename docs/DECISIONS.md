# TcCO2 Accuracy — Decisions

## Open decisions
- None yet.

## Logged divergences
- `logs2` inputs are recomputed as `log10(S2*) + 1/(n_2 - 1)` with `v_logs2 = 2/(n_2 - 1)` to match `Conway Meta/data.dta`, even though `Code/3_tcco2_uncertainty_and_simulation_do.do:94` describes `logs2` as a natural log.
