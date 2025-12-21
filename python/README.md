# TcCO2 Accuracy — Python Workflows

## Install
- From the repo root: `python -m pip install -e .`
- Optional: create/activate a virtualenv first.

## Tests
- Run: `pytest`

## Deterministic workflow runner
- Regenerate artifacts: `python python/scripts/run_all_workflows.py --out artifacts --seed 202401 --n-boot 1000 --mode analytic`
- Override data root: `python python/scripts/run_all_workflows.py --input-path /path/to/repo-root`

## Notebooks
- Launch: `jupyter lab`
- Open `python/notebooks/00_smoke.ipynb` for step-by-step checks.
- Open `python/notebooks/01_inference_playground.ipynb` for inference sweeps.
