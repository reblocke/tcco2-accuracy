# TcCO2 Accuracy — Active Plan

## Stabilized Architecture
1. Keep the Python package in `src/tcco2_accuracy/` as the numerical source of truth.
2. Keep scientific and workflow tests in `tests/`.
3. Serve the public app as static GitHub Pages from `web/`.
4. Stage Python and canonical CSV assets with `scripts/stage_web_python.py`.
5. Verify changes through `make verify`.

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
