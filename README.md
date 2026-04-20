# TcCO₂ Accuracy

> Research code and materials for studying agreement between transcutaneous CO₂ (TcCO₂) monitoring and arterial PaCO₂ across clinical contexts. This repository contains the Python package, Streamlit app, validation artifacts, and manuscript-support workflows.

**Project status:** Presented as posters at CHEST and ATS. Manuscript in preparation. This is research software and is not intended for clinical decision-making.

---

## Links & Persistent Identifiers

- **Streamlit App** https://tcco2-accuracy-2025-12-22.streamlit.app/
- **Repository:** https://github.com/reblocke/tcco2-accuracy
- **CHEST poster:** https://scholar.google.com/citations?view_op=view_citation&hl=en&user=O1nydc8AAAAJ&sortby=pubdate&citation_for_view=O1nydc8AAAAJ:hC7cP41nSMkC
- **ATS poster:** https://scholar.google.com/citations?view_op=view_citation&hl=en&user=O1nydc8AAAAJ&sortby=pubdate&citation_for_view=O1nydc8AAAAJ:dhFuZR0502QC
- **Related evidence synthesis:** Conway A, et al. *Thorax* (2019). Accuracy and precision of transcutaneous CO₂ monitoring.
- **Public dataset / code archive (Conway et al.):** https://figshare.com/articles/dataset/Accuracy_of_TcCO2_monitoring_meta-analysis/6244058
- **Release DOI:** *To be minted via Zenodo at v1.0*
- **Exact analysis commit:** *<commit-hash to be pinned prior to submission>*

---

## How to Cite

Until journal publication, please cite the conference abstracts and this repository (tagged release):

> Anderson-Bell D, Locke BW. *TcCO₂ Accuracy: code for evaluating transcutaneous CO₂ monitoring accuracy.* GitHub repository (version <tag>). <Zenodo DOI once minted>.

Add a machine‑readable citation file at the repository root (`CITATION.cff`) prior to submission.

---

## Quick Start (Reproduce Main Results)

The active reproducible workflow is the Python package under `python/src/tcco2_accuracy/`. Stata files are retained as reference material.

**Requirements:** Python ≥3.10

```bash
python3 -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Run tests:

```bash
pytest -q
```

Or run the documented ephemeral check without adopting `uv` as the project environment:

```bash
uv run --no-project --with-requirements requirements.txt --with-requirements requirements-dev.txt pytest -q
```

Regenerate review/manuscript artifacts:

```bash
python scripts/rebuild_artifacts.py --out artifacts --seed 202401 --n-boot 1000 --thresholds 45
```

The full artifact rebuild requires the in-silico PaCO₂ distribution at the configured package default path, `Data/In Silico TCCO2 Database.dta`. The Streamlit app and tests can run without that large file by using the shipped binned prior, `Data/paco2_prior_bins.csv`.

---

## Data Access

- Primary inputs derive from publicly available supplemental tables and code associated with Conway et al. (*Thorax*, 2019).
- Canonical Conway study inputs are maintained in `Data/conway_studies.csv` and `Data/conway_studies.xlsx`.
- PaCO₂ prior bins for app deployment are maintained in `Data/paco2_prior_bins.csv`.
- **No patient‑level protected health information (PHI)** is included in this repository.

If future analyses require restricted data, do **not** commit raw files. Instead, provide synthetic examples and access instructions here.

---

## Computational Environment

- **Operating systems tested:** macOS and Linux-style CI/runtime environments
- **Languages:** Python, Stata
- **Dependencies:** `requirements.txt` and `requirements-dev.txt` are authoritative for this wave; `pyproject.toml` provides package metadata and optional UI extras
- **Hardware:** CPU‑only; no GPU required

---

## Repository Layout

```
├── app/                                 # Streamlit implementation
├── streamlit_app.py                     # Streamlit Cloud entrypoint
├── python/src/tcco2_accuracy/           # Importable Python package
├── python/tests/                        # Unit and workflow tests
├── scripts/                             # Artifact and data-prep commands
├── Data/                                # Source/reference inputs and deployable prior bins
├── artifacts/                           # Small generated review/manuscript outputs
├── docs/                                # Specifications, validation notes, and runbooks
├── Conway Meta/                         # Conway reference materials
├── Code/                                # Stata reference code
├── Drafts/                              # Manuscript/presentation drafts
├── requirements.txt                     # Runtime dependencies
├── requirements-dev.txt                 # Test dependencies
├── pyproject.toml                       # Package metadata
└── README.md
```

---

## Workflow Overview

1. Load canonical Conway study-level inputs.
2. Reproduce Conway bias, SD, τ², and limits of agreement.
3. Generate route-1 bootstrap parameter draws.
4. Summarize empirical PaCO₂ priors by setting.
5. Run forward simulation, inverse inference, conditional misclassification, and manuscript-reporting workflows.
6. Export deterministic review/manuscript outputs to `artifacts/`.

---

## Key Outputs

- Validation summaries: `artifacts/meta_loa_check.md`, `artifacts/bootstrap_summary.md`, `artifacts/paco2_distribution_summary.md`, `artifacts/simulation_summary.md`, `artifacts/inference_demo.md`.
- Manuscript tables/snippets: `artifacts/manuscript_table1.csv`, `artifacts/manuscript_table2_two_stage.csv`, `artifacts/manuscript_table3_prediction_intervals.csv`, `artifacts/manuscript_results_snippets.md`.
- Figure data: `artifacts/figure_paco2_distribution_bins.csv`, `artifacts/figure_misclassification_vs_paco2.csv`.

---

## Quality Checks

- `python/tests/test_conway_meta.py` validates Conway Table 1 reproduction.
- `python/tests/test_workflows.py` checks workflow determinism.
- `python/tests/test_manuscript_workflow.py` smoke-tests manuscript output generation.
- App-facing behavior is covered by compute-layer tests and a Streamlit import smoke test.

---

## License

- **Code:** MIT License (see `LICENSE`)
- **Data:** Governed by original source licenses (e.g., Thorax supplemental materials)

---

## Funding & Acknowledgements

- **Funding:** *ATS ASPIRE Program, NIH T32, and the Intermountain Fund*

---

## Contributing & Governance

Contributions are welcome via GitHub issues and pull requests. See `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and `SUPPORT.md`.

---

## Maintainer & Contact

- **Maintainer:** Brian Locke
- **Contact:** Open an issue at https://github.com/reblocke/tcco2-accuracy/issues
- **Maintenance status:** Active (poster phase; manuscript in preparation)
