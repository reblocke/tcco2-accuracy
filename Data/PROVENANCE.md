# Data Provenance

## Conway study table

- Canonical public inputs: `Data/conway_studies.csv` and `Data/conway_studies.xlsx`.
- Template for future additions: `Data/conway_studies_template.xlsx`.
- Count fallback for the Conway export workflow: `Data/data_counts.csv`.
- Source article: Conway A, Tipton E, Liu W-H, Conway Z, Soalheira K,
  Sutherland J, Fingleton J. *Accuracy and precision of transcutaneous carbon
  dioxide monitoring: a systematic review and meta-analysis.* Thorax.
  2019;74(2):157-163. doi:10.1136/thoraxjnl-2017-211466.
- Source data/code record: https://figshare.com/articles/dataset/Accuracy_of_TcCO2_monitoring_meta-analysis/6244058
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
