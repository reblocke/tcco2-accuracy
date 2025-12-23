# TcCO2 → PaCO2 inference UI

## What the UI does
- Computes a PaCO2 median with a prediction interval (PI) conditional on an observed TcCO2.
- Reports P(PaCO2 ≥ threshold) from the posterior distribution and a TcCO2 threshold decision.
- Visualizes the posterior PaCO2 distribution, with the hypercapnia threshold region highlighted.
- Supports uploaded study tables (CSV/XLSX) to override the canonical dataset.
- The "All" setting uses Conway main-analysis parameters and the pooled PaCO2 prior across groups.

**PI vs CI**
- The interval reported is a *prediction interval* for a hypothetical arterial blood gas given TcCO2.
- It integrates measurement variability and parameter uncertainty; it is **not** a Conway-style CI.

**Prior-weighted vs likelihood-only**
- Prior-weighted uses the empirical PaCO2 distribution for the selected setting as a pretest prior.
- Likelihood-only ignores the prior and uses only the bootstrap mixture likelihood.
- For "All", the prior is pooled across pft/ed_inp/icu by subgroup sample size.

## Default UI values
- TcCO2 = 50 mmHg
- Hypercapnia threshold = 45 mmHg
- Prediction interval = 95% PI
- Inference mode = prior-weighted
- Setting = All

## Run locally
```bash
python -m pip install -e ".[ui]"
streamlit run streamlit_app.py
```

## Data requirements
- Canonical study table: `Data/conway_studies.csv` (or upload a CSV/XLSX).
- PaCO2 priors (default): `Data/paco2_prior_bins.csv` with columns
  `group`, `paco2_bin`, `count`, `weight` (weights sum to 1 within each group).
- Optional fallback: `Data/In Silico TCCO2 Database.dta` (not required for Streamlit Cloud).

You can upload a custom binned prior CSV/XLSX in the Advanced panel; it must include
all four groups (`pft`, `ed_inp`, `icu`, `all`) so the UI can serve any setting.

If the PaCO2 prior files are missing, the UI will display an actionable error message.
