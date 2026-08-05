# TcCO2 Accuracy — Specification

## Core definitions
- Difference (bias) is defined as PaCO2 − TcCO2 (mmHg), per Conway 2019.
- Let d = PaCO2 − TcCO2 (mmHg) to match Conway notation.

## Scope
- This document captures intended behavior for the Python package, workflows, and app-facing inference API.
- Public-facing summaries should describe outputs as research estimates with uncertainty, not clinical validation.

## Agreement variance and limits of agreement
- Agreement calculations use method revision `agreement_natural_log_tau2_direct_v1` and remain
  provisional pending independent biostatistical review.
- For adjusted within-study variance `S2*`, the study-level input is
  `log(S2*) + 1/(n_participants - 1)` on the natural-log scale, with sampling variance
  `2/(n_participants - 1)`. The pooled within-study variance is back-transformed with `exp()`.
- Let `sigma2` be the pooled within-study variance and `tau2` the between-study variance.
  Marginal LoA use `delta +/- 2 * sqrt(sigma2 + tau2)`.
- Analytic LoA uncertainty uses direct-scale `var_tau2 = 2 / sum((v_bias + tau2)^-2)`.
  Its delta-method contribution is `var_tau2 / (sigma2 + tau2)`; no quantity named
  `var_log_tau2` is used for that direct-scale variance.
- The coefficient for pooled log-within-study-variance uncertainty is
  `sigma2^2 / (sigma2 + tau2)`.
- Production LoA and subgroup summaries truncate a negative method-of-moments `tau2`
  estimate to 0 before calculating random-effects weights, LoA, or equation 4.13.
  `random_effects_meta(..., truncate_tau2=False)` remains available only as a low-level
  raw diagnostic path.
- Confidence-interval behavior at and near the `tau2 = 0` boundary remains provisional
  pending independent biostatistical review.
- The Conway Table 1 values remain a frozen published/legacy comparator rather than a
  correctness target for the corrected method.

## Bootstrap uncertainty propagation
- Bootstrap modes: `cluster_only` (study-level resampling) and `cluster_plus_withinstudy`
  (cluster resampling plus parametric perturbations of study bias/log-variance).
- Within-study perturbations draw `bias* ~ Normal(bias, v_bias)` and
  `logs2* ~ Normal(logs2, v_logs2)` with independence between bias and log-variance.
- Workflow defaults use `cluster_plus_withinstudy` to align outer CI scale with Conway.
- Production simulation/inference draws use a zero-truncated τ² to keep between-study
  variance non-negative. Passing `truncate_tau2=False` is an explicit diagnostic-only path.

## Parameter-group routing
- PaCO2 groups map to Conway parameters as pft→lft, ed_inp→arf, icu→icu, and all→main.
- Grouped parameter tables fail closed when the resolved group is absent. Pooled parameters may
  be used only through the explicit `fallback="main"` API and select only `group == "main"`.
- An ungrouped parameter table is treated as one explicitly supplied model, not as an implicit
  fallback. Downstream rows record `requested_group` and `parameter_group_used`.

## In-silico PaCO2 distribution
- Source file: `Data/In Silico TCCO2 Database.dta` by package default, with
  `Data/in_silico_tcco2_db.dta` accepted as a local alias, or an explicitly supplied `.dta`
  path in workflow loaders.
- The static browser app uses `Data/paco2_public_prior.csv` by default so it can run without the full `.dta` or public exact bin counts.
- Use rows with non-missing `paco2`; PaCO2 values are in mmHg.
- Treat `is_amb`, `is_emer`, `is_inp`, `cc_time` as binary flags (missing → 0).
- Subgroup assignment is mutually exclusive, applied in order:
  1) `pft` (ambulatory/LFT): `is_amb == 1`.
  2) `icu`: `is_inp == 1` and `cc_time == 1` and `is_emer == 0` and `is_amb == 0`.
  3) `ed_inp`: `is_emer == 1` or `is_inp == 1` (after removing `pft`/`icu`).
- ED membership is included in `ed_inp` by construction.
