# Data Governance

## Scope and authority

This repository is public-facing research software. The current tracked tree contains source code,
public literature-derived study summaries, a static browser app, and a bounded set of aggregate
artifacts. It must not contain patient-level protected health information (PHI), exact-count tables
derived from the restricted local PaCO2 distribution, or normalized restricted-derived weights that
can reconstruct that distribution.

The machine-readable authority is [`data_release_contract.json`](data_release_contract.json). It
defines current-tree and Pages allowlists, public-history scanning rules, format-independent
prohibited tracked paths, structured and text schema rules, and SHA-256 locks for both canonical
agreement artifacts and retained frozen aggregates. The provenance worksheet for any future
restricted-data use is
[`restricted_data_provenance.template.json`](restricted_data_provenance.template.json). Unknown
ownership, IRB/protocol, waiver, DUA/authorization, extract, source-system, repeated-patient,
permission, or retention fields remain visibly `HUMAN REVIEW REQUIRED`.

## Current tracked public/reference assets

- `Data/conway_studies.csv` is the operational canonical promotion source;
  `Data/conway_studies.xlsx` is its human-editable review mirror. They contain public
  literature-derived reference summaries and must remain semantically equal.
- `artifacts/bootstrap_params.csv` is the canonical corrected-provisional legacy parameter asset.
  It must contain exactly one current agreement-method revision and provisional status.
- The other four corrected-provisional canonical agreement artifacts are listed in
  `artifacts/STATUS.md` and hash-locked by the release contract.
- Pages stages the four literature-derived predictive kernels, their comparison summary, and the browser Python allowlist.
  It does not stage or fetch a PaCO2 prior.

## Restricted or private-only material

- Local `.dta` files, including `Data/in_silico_tcco2_db.dta` and
  `Data/In Silico TCCO2 Database.dta`, are restricted source inputs and must not be committed.
- Exact counts, binned distributions, small-cell indicators, and normalized weights derived from
  the restricted PaCO2 source are restricted-derived outputs. Removing the `count` column does not
  establish release safety: normalized weights may preserve and reconstruct the exact histogram.
- `Data/paco2_public_prior.csv`, `Data/paco2_prior_bins.*`, count-bearing/reconstructable artifact
  tables, and their staged Pages copies are prohibited by the current-tree contract.
- CSV, TSV, and XLSX schemas are inspected across the tracked tree. Legacy binary `.xls` is rejected
  by loaders and prohibited from tracking because the locked environment cannot inspect it reliably.
- The clearly synthetic prior fixture is narrowly exempted from the prior-weight schema and
  SHA-256 locked; changing its authored contents fails the release contract.
- Restricted-derived outputs may be written only under `.pytest_tmp/`, `.tmp/`, or an explicitly
  approved external private workspace. An arbitrary directory elsewhere inside the repository is
  not an approved private destination.

## Browser and workflow rules

- On November 17, 2026 the investigator explicitly approved publication of the four-setting summary lookup. The default observed-TcCO₂ lookup uses these approved TriNetX summaries. The whole-file checksum is pinned in `data_release_contract.json`; the public release receipt is `models/pages/conditional-release.json`. The synthetic example remains separately labeled. Private lookup files and uploaded binned priors are read locally. A separate checksum-approved display file contains smoothed densities derived from rounded cumulative probabilities at fixed edges. Exact weights and private chart files stay on the mini. The standalone [private chart](FUTURE_SETTING.md#private-distribution-preview) reads the private prior and posterior from a local JSON file in the approved workspace. Its chart JSON is ineligible for public promotion, even at the approved summary path with a matching digest. A posterior plotted on its full support together with the known likelihood can reveal the underlying prior. A public TriNetX lookup requires approval of the complete aggregate file for Pages, a pinned SHA-256 and a matching allowlist entry. The tracked tree and public branch/tag history admit that file only at the approved source path with the approved bytes. The history check also recognizes renamed TriNetX lookup documents. Reading and setting selection stay in browser memory.
- Browser and Pages assets are staged with `make stage-web`; never hand-edit generated files under
  `web/assets/`.
- Canonical agreement artifacts are rebuilt only with the locked `public-agreement` command in
  `artifacts/STATUS.md`. Output resolving to `artifacts/` rejects custom input or bootstrap settings.
- Restricted prior generation requires explicit source and output paths:

```bash
uv run python scripts/build_paco2_prior_bins.py \
  --input /approved/restricted/source/in_silico_tcco2_db.dta \
  --output /approved/private/workspace/paco2_prior_bins.csv --include-counts
```

- A future full restricted comparison likewise requires an explicit source and private output:

```bash
uv run python scripts/rebuild_artifacts.py --profile full \
  --paco2-path /approved/restricted/source/in_silico_tcco2_db.dta \
  --out /approved/private/workspace/tcco2-corrected-full \
  --seed 202401 --n-boot 1000 --thresholds 45
```

  Neither command is a release or promotion path. Do not regenerate, promote, or unfreeze frozen
  PaCO2-dependent results until the analysis specification, restricted-data authority, and
  independent biostatistical review gates are complete.

## Public-reference history remediation and retained outputs

The current branch tip removes exact-count and reconstructable restricted-derived public-tree
outputs. CI and Pages check public branch and tag history and the ancestry of the checked-out commit.
This includes detached pull request candidates whose earlier commits have no branch on the host repository. `artifacts/STATUS.md` accurately enumerates
removed files and the rounded/aggregate downstream files retained as frozen historical comparators.
Retention is not release approval: those files are not corrected-method results, submission-ready
evidence, or suitable for clinical use. Their hashes are locked so accidental regeneration fails the
machine contract.

This verification covers repository-controlled public branch and tag refs. Independently retained
clones, caches, package/download copies, and historical deployments outside those refs are not
asserted removed.

## Manuscript review package

The approved public distribution exhibit is a rasterized, grouped manuscript figure. The Word letter, supplement and rounded derived summaries live in the named GitHub WIP package. Every WIP asset, including prose, tables and the source inventory, is checked by exact path and reviewed digest in the current package and public history. Private arterial weights, patient records and full posterior curves remain on the mini. Manuscript inclusion does not change the Pages allowlist.

On November 24, 2026, the custodian approved the aggregate tables from `manuscript-results-for-review.md` (SHA-256 `9eda8d45034040a81f757854e55701008057a35f299b4428911d53e7ac67d4f3`) for draft PR #24. The [public manuscript tables](../works-in-progress/manuscript-aggregate-tables.md) reproduce the approved table cells. Their public file digest is recorded in `docs/data_release_contract.json`.

The custodian subsequently approved the 16 files in `draft-pr-candidates.sha256` (manifest SHA-256 `8efe87d6b000c6f766a3659408a29f2788887a0e48e857ae10251a54a0483af7`) for the same draft PR. The release includes seven Word reading and comparison copies, two aggregate tables, five figures, the working index and the reviewer questions. Each published file must match its approved digest. [Manuscript index](../works-in-progress/README.md) · [Referral policies](../works-in-progress/referral-tables.md) · [Encounter selection](../works-in-progress/encounter-selection.md)

The custodian approved merging the package to `main` with two corrections to the working index. The revised index describes both Short Report figures and the line spacing in Original Research Table 2. The release contract records the index from the approved manifest and its successor. The private manuscript archive preserves the intermediate Word revisions and their Git history.

On November 25, 2026, the custodian reviewed and approved the 19 files in `reviewed-files.sha256` (manifest SHA-256 `c1d1e947bf5286c59c0e322d666d53cca6baea142bc672dacfc15c813e8a9ac7`) for the GitHub manuscript packet. They include eight raster exhibits, seven Word copies, the two classification and policy tables, the index and reviewer questions. The risk figure displays pointwise credible bands from the saved fits. The exact empirical distributions, draw-level risks and full-resolution curves remain on the mini. [Manuscript review packet](../works-in-progress/README.md)
