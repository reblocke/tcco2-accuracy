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
| TCCO2-002 | Source recovered and checksummed; archival disposition pending | The author-supplied Tipton-Shuster R supplement and Conway application archive are source-linked in `Data/PROVENANCE.md` with retrieval dates, file identifiers, SHA-256 values, citations, and license/redistribution dispositions. A durable private archive requires documented retention authority; absent that, the independent reviewer must accept the linked/checksummed source as sufficient. |
| TCCO2-005 | Implemented; external review pending | A standalone equation-derived test oracle does not import production numerical functions and agrees with the production implementation on hand calculations, canonical subgroups, scale conversions, Eq. 4.13, and the zero-heterogeneity boundary within declared tolerances. |
| TCCO2-006 | Agreement phase complete; downstream phase blocked | The five corrected-provisional agreement artifacts are hash-locked. PaCO2-dependent outputs require the locked analysis specification, completed private provenance record, independent review, a private reproducible rebuild with at least 10,000 draws, and explicit aggregate-promotion approval. |
| TCCO2-007 | Pending human review | A dated independent biostatistical memo or reproducibility report reviews the variance transformation, Eq. 4.13, behavior at and near `tau2 = 0`, confidence intervals, estimand, clustering, and bootstrap model; every recommendation receives a recorded disposition. |

The original audit's broad TCCO2-006 wording is implemented in phases so that completion of the
public agreement correction cannot silently authorize restricted downstream regeneration. The
repository does not duplicate these project-management states in a machine-readable contract;
`docs/data_release_contract.json` remains narrowly authoritative for enforceable data and release
boundaries.

## Current Priorities
1. Complete TCCO2-007 independent biostatistical review, including near-zero tau-squared CI
   behavior, TCCO2-002 archival disposition, and the executable TCCO2-005 comparison.
2. Lock the analysis specification: estimand, PaCO2 sampling unit, repeated patients, publication
   clustering, bootstrap model, supported range, proportional bias, subgroup mappings, ED/inpatient
   definition, and row-level Conway provenance.
3. Preserve the closed downstream safety gates: missing groups fail closed, tail calculations are
   stable, conditional truth uses original values, scientific inputs fail closed, the browser
   defaults to likelihood-only, and exact/reconstructable restricted-derived outputs stay outside
   the tracked tree and Pages. Keep `docs/data_release_contract.json` authoritative and resolve every
   `HUMAN REVIEW REQUIRED` field in `docs/restricted_data_provenance.template.json` before any
   restricted-data release review.
4. Complete the downstream phase of TCCO2-006 only in a private workspace, jointly propagate
   agreement and target-distribution uncertainty, run clustering/model sensitivities with at least
   10,000 release draws, and document Monte Carlo stability before approved aggregate promotion.
5. Restore an authoritative text-diffable manuscript in an approved workspace and require
   traceability, clean-room reproduction, and independent sign-off before unfreezing results,
   tagging, or archiving a release.

Throughout these priorities, keep the browser contract aligned with the Python numerical source of
truth, keep user inputs client-side, and update validation artifacts only for intentional scientific
changes.

Public branch and tag history is continuously checked against the release contract. Independently
retained clones, caches, and historical deployments outside repository-controlled refs remain
outside this repository verification boundary.
