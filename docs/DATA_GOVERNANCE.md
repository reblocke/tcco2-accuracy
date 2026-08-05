# Data Governance

## Scope and authority

This repository is public-facing research software. The current tracked tree contains source code,
public literature-derived study summaries, a static browser app, and a bounded set of aggregate
artifacts. It must not contain patient-level protected health information (PHI), exact-count tables
derived from the restricted local PaCO2 distribution, or normalized restricted-derived weights that
can reconstruct that distribution.

The machine-readable authority is [`data_release_contract.json`](data_release_contract.json). It
defines current-tree and Pages allowlists, format-independent prohibited tracked paths, structured
and text schema rules, and SHA-256 locks for both canonical agreement artifacts and retained frozen
aggregates. The provenance worksheet for any future
restricted-data use is
[`restricted_data_provenance.template.json`](restricted_data_provenance.template.json). Unknown
ownership, IRB/protocol, waiver, DUA/authorization, extract, source-system, repeated-patient,
permission, or retention fields remain visibly `HUMAN REVIEW REQUIRED`.

## Current tracked public/reference assets

- `Data/conway_studies.csv` is the operational canonical promotion and browser-staging source;
  `Data/conway_studies.xlsx` is its human-editable review mirror. They contain public
  literature-derived reference summaries and must remain semantically equal.
- `artifacts/bootstrap_params.csv` is the canonical corrected-provisional browser parameter asset.
  It must contain exactly one current agreement-method revision and provisional status.
- The other four corrected-provisional canonical agreement artifacts are listed in
  `artifacts/STATUS.md` and hash-locked by the release contract.
- Pages stages only `conway_studies.csv`, `bootstrap_params.csv`, and the browser Python allowlist.
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

- Browser inference defaults to likelihood-only with canonical agreement parameters.
- Prior-weighted inference requires an explicit binned-prior upload. The upload remains client-side;
  the app has no backend, telemetry, persistence, or patient-value URL state.
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

## Current-tree remediation and retained history

The current branch tip removes exact-count and reconstructable restricted-derived public-tree
outputs. `artifacts/STATUS.md` accurately enumerates removed files and the rounded/aggregate
downstream files retained as frozen historical comparators. Retention is not release approval: those
files are not corrected-method results, submission-ready evidence, or suitable for clinical use.
Their hashes are locked so accidental regeneration fails the machine contract.

No Git history was rewritten in this wave. Prior commits, tags, clones, caches, package/download
copies, and historical Pages deployments remain possible disclosure surfaces pending an
institutional decision. The current-tree contract must not be represented as retroactive deletion
or proof that every historical copy is safe.
