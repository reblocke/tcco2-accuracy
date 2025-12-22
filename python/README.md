# TcCO2 Accuracy — Python Workflows

## Install
- From the repo root: `python -m pip install -e .`
- Optional: create/activate a virtualenv first.

## Tests
- Run: `pytest`

## Deterministic workflow runner
- Regenerate artifacts: `python scripts/rebuild_artifacts.py --out artifacts --seed 202401 --n-boot 1000 --thresholds 45`
- Override study table: `python scripts/rebuild_artifacts.py --input-study-table Data/conway_studies.xlsx`

## Notebooks
- Launch: `jupyter lab`
- Open `python/notebooks/00_smoke.ipynb` for step-by-step checks.
- Open `python/notebooks/01_inference_playground.ipynb` for inference sweeps.

## Streamlit UI
- Docs: `docs/UI.md`
- Install extras: `python -m pip install -e ".[ui]"`
- Run: `streamlit run streamlit_app.py`
