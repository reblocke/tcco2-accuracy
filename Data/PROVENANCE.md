# Data Provenance

## Conway study table

- Canonical Conway table: `Data/conway_studies.csv` is the operational promotion/staging source;
  `Data/conway_studies.xlsx` is its human-editable review mirror. Contract tests require semantic
  equality within `1e-12`.
- Template for future additions: `Data/conway_studies_template.xlsx`.
- Count fallback for the Conway export workflow: `Data/data_counts.csv`.
- Source article: Conway A, Tipton E, Liu W-H, Conway Z, Soalheira K,
  Sutherland J, Fingleton J. *Accuracy and precision of transcutaneous carbon
  dioxide monitoring: a systematic review and meta-analysis.* Thorax.
  2019;74(2):157-163. doi:10.1136/thoraxjnl-2017-211466.
- Source data/code record: https://figshare.com/articles/dataset/Accuracy_of_TcCO2_monitoring_meta-analysis/6244058
- Statistical-method authority for the corrected agreement equations: Tipton E, Shuster J.
  *A framework for the meta-analysis of Bland-Altman studies based on a limits of agreement
  approach.* Statistics in Medicine. 2017;36(23):3621-3635. doi:10.1002/sim.7352;
  PMCID: PMC5585060. Equations 4.4-4.5, 4.13, and 4.16 establish the natural-log
  within-study variance transform and the direct-scale between-study variance contribution.
- Source comparison retrieved 2026-08-03 from Figshare record version 2 (CC BY 4.0):
  - `TcCO2 meta-analysis.Rmd`, Figshare file ID `11643593`, SHA-256
    `238bccbaf92c3cdea715db6960c1d585431423782b48c376ad9281864c29f100`.
  - `data.Rdata`, Figshare file ID `11409167`, SHA-256
    `7c195f92f96bce3667bd90a2a55702ad51c88c40ce3ae11772ca4d6c6ef2a935`.
- The Figshare implementation is used as legacy provenance, not as the corrected equation
  authority: it stores base-10 log-variance values while applying a natural-log correction and
  later uses a natural-exponential back-transform; it also pairs direct-scale `Var(tau2)` with a
  log-scale coefficient in the analytic LoA interval.
- No external RData, R Markdown source archive, or article PDF is redistributed in this repository.
  Stable identifiers and checksums are recorded so reviewers can obtain and verify the originals.
- Public-branch rule: keep the curated CSV/XLSX inputs and cite/link the source
  records. Do not mirror third-party PDFs, duplicated supplement folders, or
  binary RData source archives in the public branch tip.

## `paco2_public_prior.csv`

- Source: restricted local in-silico PaCO2 distribution used by the project workflows. The source file is not redistributed; local paths may include `Data/in_silico_tcco2_db.dta` or `Data/In Silico TCCO2 Database.dta`.
- Transformation: weight-only 1 mmHg binned PaCO2 prior weights by group; exact bin counts are omitted. The public schema is limited to `group,paco2_bin,weight`.
- Regeneration: `uv run python scripts/build_paco2_prior_bins.py --input Data/in_silico_tcco2_db.dta --output Data/paco2_public_prior.csv`.
- Access terms: governed by the original restricted data access terms and any applicable data-use agreement assumptions. Exact count-bearing prior bins are local/generated outputs and should not be committed.
- Local exact outputs: `Data/paco2_prior_bins.csv` and `artifacts/figure_paco2_distribution_bins.csv` may be regenerated for private manuscript work, but they are ignored and intentionally absent from the public tree.
- Governance reference: see [`docs/DATA_GOVERNANCE.md`](../docs/DATA_GOVERNANCE.md).
