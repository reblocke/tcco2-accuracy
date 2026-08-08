# TcCO2 Accuracy — Specification

## Core definitions
- Difference (bias) is defined as PaCO2 − TcCO2 (mmHg), per Conway 2019.
- Let d = PaCO2 − TcCO2 (mmHg) to match Conway notation.

## Scope
- This document captures intended behavior for the Python package, workflows, and app-facing inference API.
- Public-facing summaries should describe outputs as research estimates with uncertainty, not clinical validation.

## In-memory downstream implementation contract

The choices below define the in-memory PaCO2-dependent TCCO2-006 workflow. They are a code
contract, not a claim of source-data approval, publication authorization, promotion, manuscript
status, or external review. Those decisions are outside this repository implementation.

### Target population and patient-level sampling

- The workflow models performance for a new patient in a source-like clinical setting.
- The primary setting-specific target-distribution unit is one index PaCO2 measurement per patient
  within each setting: the earliest caller-supplied eligible PaCO2 value in that patient's earliest
  caller-supplied eligible encounter for PFT, ED/inpatient, and ICU separately. Pooled `All` selects
  one earliest eligible record per patient across all settings. Eligibility and the observation
  window are supplied by the caller; this code does not adjudicate either.
- A downstream target input must contain patient identifier, encounter identifier, encounter-order,
  and measurement-order fields. Missing or unusable fields fail closed; no encounter-row fallback
  is permitted. Identifiers are stripped before grouping, duplicate patient/encounter/measurement-
  order keys after numeric or UTC normalization are rejected, and tied earliest encounters or
  measurements fail closed rather than selecting an arbitrary row.
- The primary target bootstrap resamples patient clusters with replacement without fixing observed
  truth-class counts. A proposal lacking either truth class is redrawn, with at most 100 attempts per
  accepted replicate. The rejected-proposal fraction must remain at or below 1% within every setting;
  otherwise the cohort fails as too sparse or imbalanced for the requested class-conditional metrics.
  A one-at-a-time measurement-policy sensitivity retains all eligible values while clustering them
  by patient.
- Every selected setting must contain values both below and at/above the true hypercapnia threshold
  before the bootstrap begins. A replicate with an undefined required metric fails closed.

### Agreement resampling and joint uncertainty

- The primary agreement resampling unit is the deterministic `study_base` derived from Conway
  `study_id` by stripping a trailing parenthetical qualifier. The canonical public table has 76
  effect rows and 73 publication clusters. A sampled publication contributes all of its effect rows.
- The only multi-row publication clusters are `Bolliger 2007` (TOSCA ICU and operating theatre),
  `Hirabayashi 2009` (non-ventilated and ventilated), and `Kim 2014` (hypotensive and normotensive).
  This rule is a reproducibility contract, not an additional data column.
- The one-at-a-time clustering sensitivity resamples the 76 effect rows by `study` instead.
- Each replicate independently resamples the publication-cluster agreement data and the patient
  target distribution, then pairs those two draws for every reported downstream quantity. This is
  the primary draw-aligned joint bootstrap.

### Settings, support, and model scope

- The primary mapping is PFT to LFT, ED/inpatient to ARF, ICU to ICU, and All to `main`; ED remains
  part of ED/inpatient. Repeating the analysis with pooled `main` parameters for every setting is a
  one-at-a-time sensitivity.
- Primary analyses retain every finite positive observed PaCO2 value. A one-at-a-time sensitivity
  restricts each requested setting, including pooled All, to that setting's own empirical 2.5th
  through 97.5th percentile range. No hard upper PaCO2 cutoff is introduced.
- Proportional bias is not estimable from the aggregate Conway inputs. No slope or scenario model
  will be fitted without separate evidence.

### Outputs, intervals, and precision

- The core 45 mmHg hypercapnia analysis reports prevalence, sensitivity, specificity, PPV, NPV,
  LR+, LR-, and TP/FP/TN/FN and total-misclassification probability contributions. Prediction
  output is reported at TcCO2 values 35, 40, 45, 50, and 55 mmHg; the primary prediction is
  prior-weighted, with likelihood-only output as a comparator.
- Secondary two-stage output uses TcCO2 zones `<40`, `40–50`, and `>50` mmHg and reports zone
  probabilities, zone likelihood ratios, posterior hypercapnia probabilities, reflex fraction, and
  residual misclassification probability. This API does not emit conditional curves.
- This workflow never returns exact counts or per-1,000 projections. Report mmHg and
  percentages to 0.1, probabilities to 0.001, and likelihood ratios to two decimals or two
  significant digits when large.
- Returned summary columns are `bootstrap_q025`, `bootstrap_q500`, and `bootstrap_q975`. Core
  confusion-cell outputs are joint probabilities named `tp_probability`, `fp_probability`,
  `tn_probability`, and `fn_probability`, not conditional rates. Prediction-limit metrics are in
  mmHg, probabilities are on the 0-1 scale, and likelihood ratios are unitless. Bootstrap quantiles
  around `paco2_pi_lower` or `paco2_pi_upper` quantify uncertainty in that prediction-limit endpoint;
  they are not themselves a single marginal prediction interval.
- Use 2.5th and 97.5th percentile bootstrap intervals. The default configuration enforces at least
  10,000 primary draws with seed 202401 and exactly one independent repeat with seed 202402. Every
  combined batch-quantile MCSE must be at most one tenth of reporting precision. Whether primary and
  repeat estimates differ by no more than two combined MCSE is returned as a descriptive diagnostic,
  not a hard gate. A failed MCSE gate requires more draws, never a search across additional seeds.
- Reduced-draw, disabled-stability, altered threshold, two-stage boundaries, prediction grid, or
  prescribed seed, and noncanonical-bootstrap runs remain available for controlled development and
  synthetic tests but are marked `contract_compliant: false` with reasons in the manifest.
- The only planned one-at-a-time sensitivities are effect-row versus publication clustering, pooled
  versus setting-specific parameters, all-measurements-versus-index-measurement policy, and central
  95% support restriction. The workflow rejects configurations containing more than one sensitivity
  deviation; a factorial sensitivity design is out of scope.

### Returned tables and manifest

| Result | Identifying fields | Values |
| --- | --- | --- |
| `core` | requested group, parameter group, true threshold, metric | bootstrap 2.5th/50th/97.5th percentiles |
| `prediction` | requested group, parameter group, true threshold, TcCO2, mode, metric | bootstrap 2.5th/50th/97.5th percentiles |
| `two_stage` | requested group, parameter group, true threshold, lower/upper zone bounds, metric | bootstrap 2.5th/50th/97.5th percentiles |
| `stability` | analysis component, requested/parameter groups, repeat seed, optional TcCO2/mode | primary/repeat values, MCSEs, combined MCSE, difference, precision, and pass/description flags |

Every call requires a caller-supplied non-sensitive `target_data_revision` label. The JSON-safe
manifest records that label, the Conway-table digest, complete configuration, actual subgroup-input
mode, seeds, sensitivity, redraw fractions, and contract-compliance status. It contains no patient
identifier, source path, target value, exact count, or patient-data hash and must remain paired with
the repository commit and caller-managed private source provenance.

The in-memory workflow implements this specification with synthetic validation evidence and does not
load data from a path, write results, or alter frozen artifacts. Source-data handling, output
retention, publication, and external review decisions are outside this code contract.

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
