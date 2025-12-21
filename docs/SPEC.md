# TcCO2 Accuracy — Specification

## Core definitions
- Difference (bias) is defined as PaCO2 − TcCO2 (mmHg), per Conway 2019.
- Let d = PaCO2 − TcCO2 (mmHg) to match Conway notation.

## Scope
- This document will capture intended behavior for the Python port.
- Meta-analysis, simulation, and inference details will be expanded as milestones land.

## Bootstrap uncertainty propagation
- Bootstrap modes: `cluster_only` (study-level resampling) and `cluster_plus_withinstudy`
  (cluster resampling plus parametric perturbations of study bias/log-variance).
- Within-study perturbations draw `bias* ~ Normal(bias, v_bias)` and
  `logs2* ~ Normal(logs2, v_logs2)` with independence between bias and log-variance.
- Workflow defaults use `cluster_plus_withinstudy` to align outer CI scale with Conway.
- For simulation/inference draws, τ² is truncated at 0 when requested to keep
  between-study variance non-negative (Table 1 reproduction uses untruncated τ²).

## In-silico PaCO2 distribution
- Source file: `Data/In Silico TCCO2 Database.dta` (configurable path in loaders).
- Use rows with non-missing `paco2`; PaCO2 values are in mmHg.
- Treat `is_amb`, `is_emer`, `is_inp`, `cc_time` as binary flags (missing → 0).
- Subgroup assignment is mutually exclusive, applied in order:
  1) `pft` (ambulatory/LFT): `is_amb == 1`.
  2) `icu`: `is_inp == 1` and `cc_time == 1` and `is_emer == 0` and `is_amb == 0`.
  3) `ed_inp`: `is_emer == 1` or `is_inp == 1` (after removing `pft`/`icu`).
- ED membership is included in `ed_inp` by construction.
