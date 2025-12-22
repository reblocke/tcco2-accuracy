# Streamlit Community Cloud Deployment

## Quick steps
1. Push this repo to GitHub.
2. In Streamlit Community Cloud, click **New app** → select the repo and branch.
3. Set the entrypoint to `streamlit_app.py`.
4. Deploy.

## Local parity with Cloud
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Updating the canonical study table
1. Edit `Data/conway_studies.xlsx` (or regenerate via `scripts/export_conway_rdata.py`).
2. Commit the updated `Data/conway_studies.csv` + `.xlsx`.
3. Redeploy (Streamlit Cloud auto-redeploys on push).

## Notes
- The app defaults to the canonical table; users may upload a CSV/XLSX to override it.
- Streamlit Cloud installs dependencies from `requirements.txt`.
