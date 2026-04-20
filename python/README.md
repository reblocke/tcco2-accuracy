# TcCO2 Accuracy — Python Workflows

## Install
- From the repo root: `python -m pip install -r requirements.txt -r requirements-dev.txt`
- Optional: create/activate a virtualenv first.

## Tests
- In an active environment: `pytest -q`
- Ephemeral check without adopting `uv`: `uv run --no-project --with-requirements requirements.txt --with-requirements requirements-dev.txt pytest -q`

## Deterministic workflow runner
- Regenerate artifacts: `python scripts/rebuild_artifacts.py --out artifacts --seed 202401 --n-boot 1000 --thresholds 45`
- Override study table: `python scripts/rebuild_artifacts.py --input-study-table Data/conway_studies.xlsx`
- Full artifact regeneration requires the in-silico PaCO2 `.dta` at the package default path,
  `Data/In Silico TCCO2 Database.dta`. The app and most tests use the shipped binned prior
  `Data/paco2_prior_bins.csv` when the large `.dta` is absent.

## Manuscript outputs
- Generate manuscript-ready tables/figures/snippets:
  `python scripts/rebuild_artifacts.py --out artifacts --seed 202401 --n-boot 1000 --thresholds 45 --true-threshold 45 --two-stage-lower 40 --two-stage-upper 50 --tcco2-values 35,40,45,50,55`
- Outputs include `artifacts/manuscript_table1.csv`, `artifacts/manuscript_table2_two_stage.csv`,
  `artifacts/manuscript_table3_prediction_intervals.csv`, and `artifacts/manuscript_results_snippets.md`.

## Notebooks
- Launch: `jupyter lab`
- Open `python/notebooks/00_smoke.ipynb` for step-by-step checks.
- Open `python/notebooks/01_inference_playground.ipynb` for inference sweeps.

## Streamlit UI
- Docs: `docs/UI.md`
- Install extras: `python -m pip install -e ".[ui]"`
- Run: `streamlit run streamlit_app.py`
