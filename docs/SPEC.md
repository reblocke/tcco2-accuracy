# TcCO2 Accuracy — Specification

## Core definitions
- Difference (bias) is defined as PaCO2 − TcCO2 (mmHg), per Conway 2019.
- Let d = PaCO2 − TcCO2 (mmHg) to match Conway notation.

## Scope
- This document captures intended behavior for the Python package, workflows, and app-facing inference API.
- Public-facing summaries should describe outputs as research estimates with uncertainty, not clinical validation.

## Downstream analysis decision gate

The corrected agreement method is implemented, but the PaCO2-dependent phase of TCCO2-006 remains
blocked until the study PI and independent biostatistical reviewer resolve and approve the items
below. Existing workflow defaults and frozen artifacts are not approval evidence.

| Decision | Current state | Required completion evidence |
| --- | --- | --- |
| Target estimand | `HUMAN REVIEW REQUIRED` | State whether the primary target is a new clinical context, a fixed observed context, or another precisely defined population. |
| PaCO2 observation unit and repeated patients | `HUMAN REVIEW REQUIRED` | Define the sampling unit, repeated-patient handling, and resampling unit consistently with the authorized source extract. |
| Publication/cohort dependence | `HUMAN REVIEW REQUIRED` | Reconcile 73 reported studies with 76 modeled effect rows and define stable publication/cohort clustering identifiers. |
| Joint uncertainty model | `HUMAN REVIEW REQUIRED` | Prespecify how agreement parameters and the target PaCO2 distribution are resampled together, including one primary bootstrap mode and one justified sensitivity. |
| Setting mappings | Implemented provisionally | Ratify or replace the Conway-to-local subgroup mapping and resolve the documented ED/inpatient Stata divergence. |
| Supported range and proportional bias | `HUMAN REVIEW REQUIRED` | Prespecify the supported PaCO2 range, proportional-bias assessment, and any range-restricted sensitivity without inventing an unsupported validation cutoff. |
| Final downstream outputs and reporting precision | `HUMAN REVIEW REQUIRED` | Identify the diagnostic, predictive-value, likelihood-ratio, conditional, and two-stage results retained for the manuscript and their reporting precision. |

Approval must be dated and recorded in `docs/DECISIONS.md`. Restricted execution additionally
requires completion of the private provenance record derived from
`docs/restricted_data_provenance.template.json`.

## Input validation contract
- Conway study tables and requested subgroup analyses must be non-empty. Study identifiers are
  stripped of leading/trailing whitespace and must then be non-empty and unique.
- Study counts are finite integers with `n_participants > 1` and
  `n_pairs >= n_participants`. Optional `c` must be finite, at least 1, and consistent with
  `n_pairs / n_participants` at `rtol=1e-10`, `atol=1e-12`.
- Each study table supplies at least one of the `sd` or `s2` columns; every value in each supplied
  column is finite and strictly positive. When both columns are supplied, `sd²` must equal `s2`
  within `rtol=1e-10`, `atol=1e-12`.
- Conway `is_icu`, `is_arf`, and `is_lft` study flags may overlap. They are not interpreted as
  mutually exclusive in the way the record-level PaCO2 groups below are.
- Retained PaCO2 values, including PaCO2 prior support values, and clinical classification
  thresholds are finite and strictly positive. Genuinely missing PaCO2 rows in a source
  distribution are excluded before retained values are validated; malformed, infinite, zero, or
  negative retained values fail closed.
- After whitespace and case normalization, prior group labels must form exactly
  `{pft, ed_inp, icu, all}` with no missing, blank, or additional labels.
- No evidence-supported hard upper PaCO2 limit is currently specified. Validation therefore
  imposes no finite upper cutoff beyond the requirement that the value be finite and positive.
- Two-stage TcCO2 zone boundaries are finite and strictly ordered (`lower < upper`). They may be
  negative because they are model boundaries rather than PaCO2 observations. The policy's true
  PaCO2 threshold remains finite and strictly positive.
- Conway RData/count export always rejects missing or blank study identifiers and non-integer
  observed counts, including with `--no-strict` or `--allow-missing-counts`.

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

## Tail probabilities and conditional classification
- Normal upper-tail probabilities use `sf`; their complements use `cdf` directly rather than
  subtracting a CDF from one. Likelihood ratios derived from modeled normal probabilities aggregate
  `logsf`/`logcdf` values with `logsumexp` before forming the ratio.
- Two-stage middle-zone probabilities use a stable normal interval probability rather than
  `1 - lower_tail - upper_tail`; if endpoint log probabilities are numerically indistinguishable,
  direct density quadrature preserves the mass of a valid narrow interval.
- Conditional classification determines true hypercapnia and test-positive probability from each
  original unbinned PaCO2 value. Display binning never determines truth.
- The production conditional-curve default uses half-open bins `[lower, upper)`. Output
  `paco2_bin` is the lower edge and `paco2_bin_upper` is the excluded upper edge. Legacy `round`
  and `floor` display grouping remains available, but uses the same raw-value truth calculation.
  Legacy `round` uses centered half-open intervals and assigns an exact midpoint to the interval
  beginning at that midpoint, rather than NumPy's ties-to-even convention. Bin indexing uses the
  decimal representation of the requested width so exact decimal edges are not shifted downward
  by binary floating-point division.
- Component quantiles summarize each TN/FP/FN/TP distribution separately. Probability-mass
  conservation is a per-draw invariant; lower or upper componentwise quantiles are not additive.

## In-silico PaCO2 distribution
- Restricted workflows require in-memory `paco2_data` or an explicitly supplied private `.dta`
  path. Package and workflow loaders do not auto-discover repository-local extracts. Historical
  local filenames may still be passed explicitly for compatibility.
- The static browser app defaults to likelihood-only inference and stages no PaCO2 prior. It can run
  without the restricted `.dta`; prior-weighted inference requires an explicit user-supplied binned
  prior that remains client-side.
- Exact counts and normalized weights derived from the restricted source are both private-only
  unless release is explicitly approved; a weight vector may reconstruct the exact source histogram
  even when count columns are absent. Generate them only under `.pytest_tmp/`, `.tmp/`, or an
  explicitly approved external private workspace.
- Drop genuinely missing `paco2` rows, then require every retained value to be finite and strictly
  positive; PaCO2 values are in mmHg. Do not apply an unvalidated hard upper cutoff.
- If `subgroup` is already supplied, rows with genuinely missing `paco2` are dropped first. Every
  retained row must have a non-missing subgroup label normalized to `pft`, `ed_inp`, or `icu`.
- Otherwise, treat genuinely missing `is_amb`, `is_emer`, `is_inp`, and `cc_time` values as 0.
  Every non-missing raw flag must be numeric binary 0/1; nonnumeric and nonbinary values fail.
- Subgroup assignment is mutually exclusive, applied in order:
  1) `pft` (ambulatory/LFT): `is_amb == 1`.
  2) `icu`: `is_inp == 1` and `cc_time == 1` and `is_emer == 0` and `is_amb == 0`.
  3) `ed_inp`: `is_emer == 1` or `is_inp == 1` (after removing `pft`/`icu`).
- ED membership is included in `ed_inp` by construction.

## Data-release boundary

- `docs/data_release_contract.json` defines current-tree, Pages, and public-history allowlists,
  prohibited paths and schemas, canonical agreement-artifact hashes, and retained frozen aggregate
  outputs.
- `docs/restricted_data_provenance.template.json` is required before restricted-data use or release
  review. Unknown authority, provenance, permission, and retention fields remain
  `HUMAN REVIEW REQUIRED`.
- Retained PaCO2-dependent aggregates remain frozen historical comparators and are not
  release-approved. No downstream regeneration, promotion, or unfreezing occurs without the locked
  analysis specification and independent review.
- Public branch and tag history is rewritten and continuously checked against
  `docs/data_release_contract.json`. Independently retained clones, caches, and historical
  deployments outside repository-controlled refs may still contain removed material.
