# TcCO2 Accuracy — Decisions

## Open decisions
- None yet.

## Logged divergences
- `logs2` inputs are recomputed as `log10(S2*) + 1/(n_2 - 1)` with `v_logs2 = 2/(n_2 - 1)` to match `Conway Meta/data.dta`, even though `Code/3_tcco2_uncertainty_and_simulation_do.do:94` describes `logs2` as a natural log.
- Bootstrap τ² draws are truncated at 0 to enforce non-negative between-study variance for simulation/inference draws in `python/src/tcco2_accuracy/bootstrap.py:66`.
- Bootstrap workflows default to `cluster_plus_withinstudy` to align outer CI scale with Conway, while low-level bootstrap functions default to `cluster_only` in `python/src/tcco2_accuracy/workflows/bootstrap.py:32` and `python/src/tcco2_accuracy/bootstrap.py:24`.
- Hybrid bootstrap perturbations treat bias and log-variance inputs as independent due to missing covariance data in `python/src/tcco2_accuracy/bootstrap.py:52`.
- Meta-analysis τ² defaults to untruncated values to reproduce Conway Table 1, but if the method-of-moments denominator is non-finite or zero we set τ² = 0 for stability in `python/src/tcco2_accuracy/conway_meta.py:133`.
- LoA confidence intervals are undefined for single-study summaries, so CI bounds are returned as NaN when df ≤ 0 in `python/src/tcco2_accuracy/conway_meta.py:199`.
- PaCO2 subgroup assignment follows `docs/SPEC.md:20-27`, which includes ED in `ed_inp` by construction; this differs from `Code/2_trinetx_cleaning_do.do:8-13`, where `ed_inp_group` excludes ED (`is_emer==0`) and requires `cc_time==0`.
- Forward simulation maps PaCO2 subgroups to Conway bootstrap groups via `python/src/tcco2_accuracy/simulation.py:21` (pft→lft, ed_inp→arf, icu→icu) to align ambulatory and acute respiratory failure sub-analyses.
- Simulation/inference parameter validation requires finite numeric values with non-negative σ² and τ² in `python/src/tcco2_accuracy/utils.py:40`.
- When subgroup-specific parameters are missing, simulation and inference fall back to using all parameters (with a warning) rather than dropping the subgroup in `python/src/tcco2_accuracy/simulation.py:48` and `python/src/tcco2_accuracy/inference.py:85`.
- `format_inference_demo` only supports a single threshold and raises a ValueError otherwise in `python/src/tcco2_accuracy/workflows/infer.py:123`.
- Canonical Conway study exports append the Bolliger 2007 (TOSCA - ICU) row from `Code/1_conway_tcco2_cleaning_do.do:13-24` when absent in `data.Rdata`, to preserve Table 1 reproduction in `scripts/export_conway_rdata.py:18`.
