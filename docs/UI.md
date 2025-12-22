# TcCO2 → PaCO2 inference UI

## What the UI does
- Computes a PaCO2 median with a prediction interval (PI) conditional on an observed TcCO2.
- Reports P(PaCO2 ≥ threshold) from the posterior distribution and a TcCO2 threshold decision.
- Visualizes the posterior PaCO2 distribution, with the hypercapnia threshold region highlighted.
- Supports uploaded study tables (CSV/XLSX) to override the canonical dataset.

**PI vs CI**
- The interval reported is a *prediction interval* for a hypothetical arterial blood gas given TcCO2.
- It integrates measurement variability and parameter uncertainty; it is **not** a Conway-style CI.

**Prior-weighted vs likelihood-only**
- Prior-weighted uses the empirical PaCO2 distribution for the selected setting as a pretest prior.
- Likelihood-only ignores the prior and uses only the bootstrap mixture likelihood.

## Run locally
```bash
python -m pip install -e ".[ui]"
streamlit run streamlit_app.py
```

## Data requirements
- Canonical study table: `Data/conway_studies.csv` (or upload a CSV/XLSX).
- PaCO2 priors: `Data/In Silico TCCO2 Database.dta` (preferred).
- Optional fallback binned prior: `artifacts/paco2_prior_bins.csv` with columns
  `subgroup`, `paco2_bin`, `count`.

If the PaCO2 prior files are missing, the UI will display an actionable error message.
