# Data Provenance

## `paco2_public_prior.csv`

- Source: restricted local in-silico PaCO2 distribution used by the project workflows.
- Transformation: 1 mmHg binned PaCO2 prior weights by group; exact bin counts are omitted.
- Regeneration: `uv run python scripts/build_paco2_prior_bins.py --input Data/in_silico_tcco2_db.dta --output Data/paco2_public_prior.csv`.
- Access terms: governed by the original restricted data access terms; exact count-bearing prior bins are local/generated outputs and should not be committed.
