# TcCO2 Accuracy — Active Plan

## Stabilized Architecture
1. Keep the Python package in `src/tcco2_accuracy/` as the numerical source of truth.
2. Keep scientific and workflow tests in `tests/`.
3. Serve the public app as static GitHub Pages from `web/`.
4. Stage Python and canonical CSV assets with `scripts/stage_web_python.py`.
5. Use affected checks for local changes; documentation-only edits need reference checks and `git diff --check`. Use `make verify` for broad integration and release verification.

## Completed Engineering Gate

- On 2026-08-07, commit `12062da` passed the GitHub Actions x86 CI gate
  ([run 31136457759](https://github.com/reblocke/tcco2-accuracy/actions/runs/31136457759))
  and the matching Pages deployment
  ([run 31136457761](https://github.com/reblocke/tcco2-accuracy/actions/runs/31136457761)).
  This closes the corrected-provisional engineering wave; it does not constitute independent
  statistical review or authorize downstream promotion.

## Validation Tickets

| Ticket | Current state | Minimum completion evidence |
| --- | --- | --- |
| TCCO2-002 | Complete: link-and-checksum-only archival disposition | `Data/PROVENANCE.md` records the authoritative URLs, retrieval date, 2,521-byte size, SHA-256, and non-redistribution rationale for the Tipton-Shuster supplement. No repository or private copy is retained because no explicit redistribution license was identified. |
| TCCO2-005 | Implemented in code | A standalone equation-derived test oracle does not import production numerical functions and agrees with the production implementation on hand calculations, canonical subgroups, scale conversions, Eq. 4.13, and the zero-heterogeneity boundary within declared tolerances. |
| TCCO2-006 | Downstream specification and code implementation complete | The five corrected-provisional agreement artifacts remain hash-locked. The in-memory workflow implements setting-specific/pool index selection, publication clustering, ordinary patient-cluster resampling, draw-aligned uncertainty, one-at-a-time sensitivities, explicit aggregate schemas, reproducibility metadata, and a fail-closed MCSE gate. Synthetic tests and the 10,000-draw scale check access no restricted data and change no frozen artifacts. |
| TCCO2-007 | Outside repository code scope | Any human methods review, publication decision, or release authorization is handled separately and is not a code-completion criterion. |

The original audit's broad TCCO2-006 wording is implemented in phases so that completion of the
public agreement correction cannot silently authorize restricted downstream regeneration. The
repository does not duplicate these project-management states in a machine-readable contract;
`docs/data_release_contract.json` remains narrowly authoritative for enforceable data and release
boundaries.

## Current Priorities
1. Keep the implemented TCCO2-006 workflow covered by synthetic regression tests: within-setting
   and pooled index selection, ordinary patient-cluster resampling and redraw limits, publication-
   cluster resampling, draw-aligned joint uncertainty, exact aggregate schemas, run-manifest
   generation, development-contract labeling, and Monte Carlo stability gates.
2. Preserve the closed downstream safety gates: missing groups fail closed, tail calculations are
   stable, conditional truth uses original values, scientific inputs fail closed, the browser
   defaults to likelihood-only, and exact/reconstructable restricted-derived outputs stay outside
   the tracked tree and Pages. Keep `docs/data_release_contract.json` authoritative and resolve every
   `HUMAN REVIEW REQUIRED` field in `docs/restricted_data_provenance.template.json` before any
   restricted-data release review.
3. Manual decisions about source-data use, publication, review, promotion, and release remain
   outside this repository implementation.

Throughout these priorities, keep the browser contract aligned with the Python numerical source of
truth, keep user inputs client-side, and update validation artifacts only for intentional scientific
changes.

Public branch and tag history is continuously checked against the release contract. Independently
retained clones, caches, and historical deployments outside repository-controlled refs remain
outside this repository verification boundary.

## TCCO2-008

The [offline future-setting workflow](FUTURE_SETTING.md) has distinct records for source adjudication, actual fitting, predictive integration, synthetic recovery, public candidates, private application and human review. Inspect the run manifest for each stratum's evidence status. The pre-fit ledger uses the final Simon steady-state figure for the overall class and records its handling of repeated pairs.

### Pages preview for the four public fits

On November 15, 2026, the [live calculator](https://reblocke.github.io/tcco2-accuracy/) completed its default calculation with method `agreement_natural_log_tau2_direct_v1`. Its worker fetched `bootstrap_params.csv` and `conway_studies.csv`. The deployed Python manifest contains the legacy contract and omits the future-setting module. The current branch uses the same browser worker and staging manifest. The live runtime check and screenshot are in `.pytest_tmp/e2e/live-applet-audit/`. [Successful Pages deployment](https://github.com/reblocke/tcco2-accuracy/actions/runs/36619691711)

1. **Build the four-stratum comparison for review.** Generate one public summary from the accepted overall fit in `public-main-simon-2003` and the ICU, acute respiratory failure and lung-function fits in `public-adjudicated-v2`. Verify each manifest, candidate status and output checksum. Keep the full-precision endpoints, quantity, units, membership, method and source revision in the summary. Show all four rows in the initial page before Pyodide loads. Group the two Conway columns and the two corrected legacy columns under their respective headings. Emphasize the central 95% predictive interval for one new paired observation. Define the signed difference (arterial minus transcutaneous, mmHg) and outer confidence envelope next to the table. Include the overall membership difference (75 adjudicated units versus 76 legacy rows). On narrow screens, keep each stratum together with its labeled intervals. Keep the custom-study and bootstrap calculator offline.
2. **Package the fitted distribution for interactive use.** The five interval pairs per stratum suffice for the table. The calculator needs each stratum's posterior-averaged density and CDF. Generate a compact, versioned representation offline from the same accepted draws. Benchmark transfer size, load time and memory on a mobile browser. Compare its log density, CDF and quantiles against the [native Python calculation](../src/tcco2_accuracy/core/future_setting.py), including asymmetric tails. Specify how the evaluator handles offsets outside any lookup grid and how it certifies numerical precision on the supplied prior. If these checks cannot be met for an input, explain the limitation and clear that result. Stage only the reviewed public assets and Python evaluator through the [staging script](../scripts/stage_web_python.py) and [release contract](data_release_contract.json).
3. **Distinguish agreement from conditional PaCO₂.** Make signed disagreement the default prior-free result, with a labeled zero line and the selected stratum's median and central 95% interval. For a PaCO₂ posterior, require an explicit local upload of fixed prior masses. Evaluate the averaged density at every supported arterial value minus observed TcCO₂, multiply by its mass and normalize once in log space. Compute quantiles and threshold probabilities over the full support, independently of chart zoom or bin width. Map All to overall, ICU to ICU, ED/inpatient to acute respiratory failure, and PFT to lung-function testing. Show the source population and report an unavailable stratum explicitly. Keep legacy bootstrap and study-upload controls within the legacy workflow. Public summaries and synthetic test priors suffice for development. Private TriNetX support and weights stay off Pages.
4. **Verify in preview and after deployment.** Write acceptance cases before changing the worker or display. Check all 20 displayed intervals, their signed direction, and stratum mapping. For the overall fit, the default disagreement interval should read −11.5 to +11.2 mmHg. With the synthetic prior and TcCO₂ of 45 mmHg, the new overall fit gives a PaCO₂ median of 49 mmHg, a central 95% interval of 41 to 57 mmHg and P(PaCO₂ ≥45) of 0.6316. On the same inputs the deployed app displayed 49 [41, 49] mmHg and 0.655. Use this case to catch an accidental call to the legacy evaluator. Exercise all four strata at TcCO₂ 35, 40, 45, 50 and 55, the existing threshold and interval-level controls, missing priors, tail support and unresolved discrete quantiles. Check browser approximation error separately from the offline Monte Carlo and quadrature gates. Verify mobile and desktop readability, keyboard access, visible method/status labels, stale-result clearing and client-local handling of inputs. Run `make verify` and `make visual-qa`, then repeat the method, asset and numerical checks on the deployed URL after an authorized release. Record engineering verification, independent biostatistical review and release as separate statuses.
