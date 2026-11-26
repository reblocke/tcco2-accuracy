# Manuscript outputs

The [coauthor index](../works-in-progress/README.md) links the Original Research and standalone Short Report alternatives, the supplement, reversible comparisons and the [statistical review questions](../works-in-progress/reviewer-questions.md). The source-specific appraisal and same-75-row endpoint replay accompany the journal supplement. On November 24, 2026, the custodian approved the aggregate file for draft PR #24. It reports code-specific reweighting, crossed population and fit comparisons, ambulatory influence estimates and within-draw strategy differences. The exact empirical prior and full curves remain on the mini. [Manuscript aggregate tables and approved source checksum](../works-in-progress/manuscript-aggregate-tables.md)

## Next steps for manuscript review

The GitHub code review of the November 25 figure revision closed with no findings. Independent biostatistical review is pending. [Code review of PR #26](https://github.com/reblocke/tcco2-accuracy/pull/26#issuecomment-7179635877)

- [ ] **Send Dustin the packet for independent review.** Ask whether equal patient weighting on the first eligible testing day represents the intended ABG-tested population, and whether the signed disagreement distribution applies across arterial values within each setting. The ambulatory fit uses four lung-function studies. Ask him to check how the observed reading weights parameter draws when estimating uncertainty in its threshold probability. [Review packet](../works-in-progress/README.md) · [Reviewer questions](../works-in-progress/reviewer-questions.md) · [Sensitivity analyses](../works-in-progress/manuscript-aggregate-tables.md)
- [ ] **Derive the overall agreement envelope independently.** Use the same 75 archive rows to calculate both endpoint variances and the critical value. Compare the derivation with the reported endpoint SE of 0.5827 mmHg and outward margin of 1.161 mmHg per side. [Agreement audit](../works-in-progress/agreement.md) · [Calculation to compare after derivation](../works-in-progress/overall-envelope.R)
- [ ] **Verify who receives an ABG and how errors are counted.** For the 1%, 5% and 10% policies, check the maximum modeled error at each whole-number display classified without an ABG. Compare with fixed 40–50 inclusive, including both off-grid bins. Reconcile residual false negatives and false positives per 100 eligible testing occasions. Consider how ABGs obtained for pH or oxygenation affect the interpretation of potential tests avoided. [Policy tables](../works-in-progress/referral-tables.md)
- [ ] **Decide how to present ambulatory uncertainty.** Both probability curves cover all 81 readings. The pointwise credible bands qualify at 37 readings for PaCO₂ ≥45 and 34 for ≥52. Decide whether the visible gaps are sufficient for the paper or whether to evaluate more saved draws on the mini under the existing precision criteria. [Affected readings and precision checks](#threshold-risk-and-classification-given-an-exact-reading)
- [ ] **Generate the worked example from the lookup before refreshing the figures.** At an observed TcCO₂ of 45 mmHg in All settings, the released probability of PaCO₂ ≥45 is 32.29%. The figure and supplement correctly describe 32 true positives and 68 false positives per 100 calls. A synthetic change to 60% updated the bars while both captions still said 32 and 68. Have both captions read the same value as the bars, and verify them with a changed example. [Figure caption](../scripts/manuscript/inverse_exhibits.py) · [Word caption](../scripts/manuscript/inverse_documents.py)
- [ ] **Choose and proof the submission manuscript.** Coauthors should choose between Original Research and the Short Report, resolve authorship, affiliations, ethics and disclosures, and verify the AI contribution statement. For Original Research, double-space Table 2 for submission. For the Short Report, check that the Figure 1 caption stays with the figure in the final PDF on both Macs. The Short Report uses both of the journal’s two permitted exhibits. [Clean copies and redlines](../works-in-progress/README.md) · [Respiratory Care instructions](https://journals.sagepub.com/author-instructions/RSP)

**Future clinical validation** requires individual paired measurements. An independent cohort can test coverage of the 95% PaCO₂ prediction interval and calibration at ≥45 and ≥52 mmHg. The current packet supports review of the model-based estimates.

## How to generate
- Promote only current public agreement artifacts with:
  `uv run python scripts/rebuild_artifacts.py --profile public-agreement --input-study-table Data/conway_studies.csv --out artifacts --seed 202401 --n-boot 1000 --bootstrap-mode cluster_plus_withinstudy`.
- The `artifacts/` destination enforces that exact source and settings. Build custom-input or
  sensitivity candidates under `.pytest_tmp/`, `.tmp/`, or another noncanonical destination.
- A future full restricted comparison requires an explicit approved source and private output, for
  example: `uv run python scripts/rebuild_artifacts.py --profile full --paco2-path /approved/restricted/source/in_silico_tcco2_db.dta --out /approved/private/workspace/tcco2-corrected-full --seed 202401 --n-boot 1000 --thresholds 45`.
- In-repository full output is accepted only under `.pytest_tmp/` or `.tmp/`; otherwise use an
  external private destination. The full profile is not a promotion command.
- Exact-count and reconstructable restricted-derived outputs, including normalized weights, must be
  written only to `.pytest_tmp/`, `.tmp/`, or an explicitly approved external private workspace.
- The full profile is not authorized for current-wave regeneration, promotion, or unfreezing. Use it
  only after the analysis specification, restricted-data authority, and independent-review gates are
  complete.

## Current status
- `artifacts/STATUS.md` is authoritative for corrected versus frozen outputs.
- `artifacts/manuscript_parameters.*` contains corrected-provisional agreement parameters under
  method revision `agreement_natural_log_tau2_direct_v1`.
- The retained PaCO2-dependent aggregates under `artifacts/` remain frozen at the legacy method revision. Files
  removed under the exact-count governance contract are not public manuscript placeholders.
- Frozen values must not be copied into a final manuscript, described as corrected, or used to make
  submission-readiness claims. Independent biostatistical review and a separate downstream rebuild
  are required before unfreezing them.

## Current tracked manuscript-support files
- Error-model parameters: `artifacts/manuscript_parameters.csv` + `artifacts/manuscript_parameters.md`
  - Corrected-provisional; may support method review but not final manuscript promotion.
- Confusion matrix (rounded aggregate): `artifacts/manuscript_confusion_matrix.md`
  - Retained frozen legacy-method comparator; not release-approved.
- Two-stage strategy (rounded aggregate): `artifacts/manuscript_table2_two_stage.md`
  - Retained frozen legacy-method comparator; not release-approved.
- TcCO2 → PaCO2 prediction intervals: `artifacts/manuscript_table3_prediction_intervals.csv` +
  `artifacts/manuscript_table3_prediction_intervals.md`
  - Frozen legacy-method output.
- Results snippets: `artifacts/manuscript_results_snippets.md`
  - Frozen legacy-method output; do not copy into a current manuscript.

Also retained and frozen are `artifacts/simulation_summary.md` and
`artifacts/two_stage_summary.md`. `artifacts/STATUS.md` is the authoritative complete allowlist.

## Removed from the current tracked tree

- Operating-characteristic Table 1 CSV/Markdown.
- Exact/reconstructable confusion-matrix and two-stage CSVs.
- PaCO2 distribution, conditional-classification, misclassification-figure, and inference-demo
  outputs.

These files may be generated only in approved private output locations. Public branch and tag
history is checked against `docs/data_release_contract.json`; independently retained clones,
caches, and historical deployments outside repository-controlled refs may remain.

## Interval definitions
- Parameter summaries (δ/σ²/τ²/LoA): 95% uncertainty interval (bootstrap percentile).
- Forward classification metrics and two-stage metrics: 95% CI (bootstrap percentile).
- Inference outputs: 95% prediction interval (PI).

## November 2026 prediction letter

The frozen legacy catalog above describes the files in `artifacts/`. The current Word review package and results are in [Works in progress](../works-in-progress/README.md). The methods supplement describes the cohort, study observation model, hyperpriors, source differences, variance components and numerical checks.

Reproduce the empirical runs inside the private workspace on the mini. Install the private analysis dependencies with `uv sync --locked --group manuscript`. Use `uv run --locked --group manuscript python` to run the scripts below. The eligible-record stage selects source-recorded arterial LOINC 2019-8 and 32771-8 results from encounters matching the original patient/encounter keys and records the input fingerprints. The December Stata file supplies encounter covariates. Use the original All first-day workspace and its verified full-grid receipt as the baseline. The scripts accept explicit private paths and write their full outputs there.

1. Run `cohort.py ORIGINAL STAGE ARCHIVED_DTA PRIVATE_OUTPUT` from `scripts/manuscript/`. The stage contains the arterial candidate parquet and its receipt. Its parent is the original cohort kit with keyed membership parquet. The versioned fingerprints in `private_inputs.py` bind the original private provenance record and manuscript stage receipt before any file digests are accepted. Both stages must identify the same archived Stata member, source and weighting policy. SQL and pandas independently construct the patient-weighted distribution within each setting. Eligibility counts precede first-day selection.
2. Place the accepted ICU, ARF and LFT posterior files under their stratum names in the private output directory. Use the public-adjudicated-v2 manifest and the extended, verified All posterior in the original workspace. Run `predictions.py PRIVATE_OUTPUT ORIGINAL MANIFEST`. The grid evaluator uses the mini’s Accelerate normal-mixture library from the original verified grid; its source is `mixture.c` (build with `clang -O3 -dynamiclib -framework Accelerate mixture.c -o mixture.dylib`). It integrates over the observed arterial support at quadrature orders 32 and 64, preserving contiguous chain batches for MCSE. It rechecks the All results including verified medians and stores numerical evidence privately for all four populations.
3. Run `strategy.py PRIVATE_OUTPUT ORIGINAL MANIFEST` for draw-specific forward classification, followed by `exhibits.py PRIVATE_OUTPUT PRIVATE_EXHIBITS`. Inspect the two rendered PNGs and the two Markdown summary tables before transferring those four exhibits. The full grids, numerical evidence and empirical weights belong in the private output directory. Rebuilding legacy artifacts is a separate workflow.

For the historical forward classification calculation, run `classification_figure.py PRIVATE_OUTPUT ORIGINAL MANIFEST WIP PRIVATE_EXHIBITS` with the manuscript environment. Supply the current WIP directory with the Markdown classification table, Word supplement and archived December classification PNG. The builder integrates each posterior draw over the full patient-weighted arterial support and checks the four confusion-matrix cells in both tables before rendering the figure and its historical comparison. The historical plot spans arterial values of 30–60 mmHg. The current figures use observed TcCO₂ on the horizontal axis.

Review scope includes source and model checks, private calculations and rendering. Independent biostatistical review, clinical validation and journal submission remain open.

Run `verify_grid.py PRIVATE_OUTPUT ORIGINAL MANIFEST` on the mini to independently recompute every integer TcCO₂ reading from 20 through 100 mmHg in all four settings (324 rows). It uses the full arterial support and posterior chains, ordinary-scale Gaussian quadrature, and separate calculations of threshold masses and chain-batch errors. It compares both quadrature orders, each median and interval, both inclusive threshold probabilities, and every decision to withhold an estimate. The 47 mmHg reading remains a worked example. The portable CI acceptance test exercises known mixture predictions across the full grid and detects incorrect holds or missing rows. Run `validate.py PATH_TO_MIXTURE_DYLIB` for the analytic mixture and ABG-referral cases. Public fit provenance is in `models/pages/summary.json`; the All precision extension and original grid receipt are on the mini.

### Reproduce the amended predictions on the private mini

Use the archived interval-precision run and its saved audit certificates. From its approved private output directory, compile `scripts/manuscript/mixture_cache.cpp` with `c++ -O3 -std=c++17 -shared -fPIC` and create a private `cache/` directory. Run `cache_private_grid.py RUN OUTPUT/cache LIBRARY` and then `reporting_grid.py RUN OUTPUT --curves` with the manuscript environment. Both scripts are in `scripts/manuscript/`. Each chart row is checked against the archived full-grid ratio estimates at both thresholds and at the six quantile cuts. The HTML viewer authenticates its accompanying JSON file and opens each of the 324 rows without a network request. Only the named manuscript exhibits go into the WIP package.

The largest saved fits had 16,384,000 retained draws per chain for All, 8,192,000 for ICU, 32,768,000 for ED/inpatient and 65,536,000 for ambulatory patients. The ambulatory sampler target was 0.99. All 48 disjoint-prefix bounding-cut checks passed. The largest endpoint bracket across settings was 0.3 mmHg. Each reported numerical bracket had one whole-unit display value.

The reporting exporter checks each reading against the cached likelihood before producing manuscript exhibits, including when called without `--curves`. That option additionally writes the private chart and viewer.

### Threshold risk and classification given an exact reading

For an exact TcCO₂ reading `t`, each saved draw supplies a density for each candidate PaCO₂ `a` at the signed offset `a − t`. Multiply by the fixed arterial weights and sum over all arterial values to get the draw's marginal density of the reading, `m(t)`. Summing only over `a ≥ h` gives `j(h,t)`. The released line at each threshold is the sum of the joint densities across draws divided by the sum of the marginal densities. For the pointwise 95% credible band, take the 2.5th and 97.5th percentiles of the draw-specific risks `j(h,t)/m(t)`, weighting the draws in proportion to `m(t)`. Thus a draw giving the reading little support contributes less to the conditional band. This band describes uncertainty about the probability for that reading. The PaCO₂ prediction interval describes variation in an individual paired arterial value.

On the mini, use `inverse_figure.py RUN PAGES CHECKS --group GROUP --draws N` for each setting. Use the receipt-bound four chains of the released fit and equally spaced draws in each chain. Compare the full-draw cached density against the approved lookup at every reading and both thresholds (absolute tolerance 2 × 10⁻¹⁰). The smaller selection used for the bands must reproduce the full-draw risk within six chain-batch Monte Carlo standard errors. Within-chain batches give the Monte Carlo error of the weighted CDF at each endpoint. Invert the CDF at the target percentile plus and minus two standard errors to express that error on the risk scale. An endpoint qualifies if the larger outward change plus the absolute difference between 32- and 64-node quadrature is at most 0.1 percentage point, and the quadrature change alone is at most 0.01 percentage point. Shade only readings where both endpoints qualify. The full-draw risk line is available at every supported reading.

| Population | Saved draws used per chain for bands | Qualified bands at PaCO₂ ≥45 | Qualified bands at PaCO₂ ≥52 |
| --- | ---: | ---: | ---: |
| All settings | 32,768 | 81/81 | 81/81 |
| Critical Care Billing | 131,072 | 81/81 | 81/81 |
| ED / inpatient | 262,144 | 81/81 | 81/81 |
| Ambulatory | 262,144 | 37/81 | 34/81 |

For Ambulatory, shading is withheld at observed TcCO₂ 48–59 and 69–100 mmHg for PaCO₂ ≥45, and at 54–100 mmHg for PaCO₂ ≥52. At least one endpoint at those readings exceeded the numerical budget. The released probability lines and the calculator's PaCO₂ prediction intervals qualified at all 324 setting–reading combinations.

Run `inverse_exhibits.py PAGES CHECKS DECEMBER_PNG PRIVATE_EXHIBITS --draws N_ALL N_ICU N_ED N_PFT`. The second figure uses the approved ≥45 line and its complement. Below an observed 45, these are false negatives and true negatives. At and above 45, they are true positives and false positives. Every column sums to one before display rounding. These exact-reading proportions have a different denominator from population-wide counts in the forward tables and from the rounded-display ABG policies. The side-by-side comparison retains the archived December figure with its arterial horizontal axis.

For the Word exhibits, run `inverse_documents.py PARENTS PRIVATE_EXHIBITS PRIVATE_WIP` in the manuscript environment. The script checks the three clean parent documents against their SHA-256 hashes at Git revision `e7be2e6`. The supplement embeds the probability figure as Figure 2a and the classification bars as Figure 2b. The immediate redlines compare with those same November 25 parents. The cumulative Short Report redline compares with the December 30 DAB original. Verify that both text projections recover their respective versions and that the embedded figures match the reviewed PNGs.

### Approved summary export

On the mini, run `export_public_lookup.py RUN OUTPUT REPOSITORY` from `scripts/manuscript/` with the manuscript environment. It reruns the saved-grid reconciliation, checks the public model input fingerprints and explicitly selects summary fields. Only `conditional-lookup.json` and `conditional-release.json` are transferred for public promotion. Pin the lookup hash in the release contract before staging. The private chart, exact weights, caches and numerical certificates remain on the mini. No new sampling is required.

### Whole-mmHg referral policies and encounter-selection replay

The [post hoc specification](REFERRAL_POLICY.md) defines the policies for classifying simultaneous PaCO₂ ≥45. The primary manuscript analyses still predict the arterial interval and the probabilities of ≥45 and ≥52 given an observed TcCO₂. The efficiency outcome is the expected fraction of patient-weighted occasions classified without a reflex ABG, conditional on a usable reading and a timely reference result when referred. The frequency of sensor failures and other reasons for arterial sampling, including pH and oxygenation, is unknown.

Run these commands on the controlled host with the manuscript dependencies. Create private cache, work and review directories outside the checkout. Here `STORE` is the verified manuscript workspace, `ORIGINAL` is the original first-testing-day run and `RUN` is the saved interval-precision run. The commands below refer to scripts in `scripts/manuscript/`.

1. Replay `strategy.py` using the original cohort, manifest and four 2,048-draw-per-chain selections in a private output directory containing the verified input files. Compare its `strategy.json` with the archived continuous analysis.
2. Run `referral_analysis.py STORE ORIGINAL CACHE --setting GROUP --draws N` for each setting at quadrature orders 32 and 64. Use 8,192 retained draws per chain for `all` and `icu`, and 32,768 for `ed_inp` and `pft`. Each cache receipt binds the inputs, output, generator, display integral and predictive kernel. When an implementation hash differs or an older receipt lacks the kernel hash, preserve that cache and start a fresh one for the new run. Run `referral_precision.py STORE ORIGINAL RUN CACHE` to combine the signed bin-integrated likelihood with the high-precision exact-reading control. The report checks the precision script’s hash before using its certificates. If only the precision script changes, rerun that step on the verified arrays. The tables of displayed-reading risks remain private.
3. Run `referral_report.py CACHE WORK` followed by `referral_exhibits.py WORK REVIEW`. For another cap, include the named caps in the list, for example `--caps .01 .025 .05 .10`. Rules use both arterial classes, all 81 display bins and both overflow cells. The chart uses predicted occasions per 100 at each display. Totals are posterior means. The policy tables use medians and pointwise 95% credible intervals. The step plot traces the changes in classified proportion from 1% to 10%.
4. Run `referral_cohort.py HISTORICAL_CANDIDATES PATIENT_CSV STORE ORIGINAL STAGE ARCHIVED_DTA CANONICAL_DB COHORT` in a fresh private directory. Preserve the source snapshot and verify the archived flow. Run `referral_replay_check.py NOTEBOOK_OUTPUTS COHORT ORIGINAL STAGE ARCHIVED_DTA WORK` to compare all 36 saved key sets and all arterial source records on shared encounter keys. Run `referral_cohort_prediction.py RUN COHORT WORK` against the libraries and batch cache of the saved prediction run. It checks the original cache fingerprint, the two integration orders, and the primary 47 mmHg result on the unchanged fit before reporting each pair of comparisons. Run `referral_cohort_tables.py COHORT WORK REVIEW`. The exact first-day weights and membership keys are controlled inputs.
5. In the Word-editing environment with python-docx, run `referral_documents.py PARENTS PRIVATE_WIP REVIEW WORK --editable EDITABLE_PARENT`. Use the three pinned immediate parents and December wording for the cumulative comparison. Render the resulting Word files, check headings, figures and the article limits, and verify the old and new text projections of each redline. The approved source documents and figures are in the WIP index. New tables, rendered figures and manuscripts undergo release review together on the mini.

## Importing a newer Word revision

From the Air checkout, run `make manuscript-wip` with Python, Pandoc, GitHub CLI and SSH access to the mini. Both hosts are checked for letter and supplement Word files in the checkout’s top-level `Drafts/` and the older working clone’s top-level `Drafts/`. Subdirectories contain conference material and are outside this import. Identical editable copies named “Current TcCO2 Letter.docx” and “Current TcCO2 Supplement.docx” were placed in both checkout draft folders. New revisions may use dated filenames containing “letter” or “supplement.”

The command compares SHA-256 hashes against current remote main. It prepares visible text and tables in a local Word review copy. Inspect it beside the current public Word file and reconcile changes in Word. Existing approved figures are carried forward by content hash. Other illustrations receive a review marker. Native Word equations retain their mathematical structure. Verify the prose, tables, authorship, graphics and source inventory before using the printed digest to open a draft PR. A changed review copy produces a new digest. Compare divergent edits, stale copies and newly named revisions with current main in Word. Save the reconciled document under the current source filename on its Mac. Keep superseded revisions in a local archive outside the import folders. An unavailable Mac is identified in the command output.
