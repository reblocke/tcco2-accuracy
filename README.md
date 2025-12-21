# TcCO₂ Accuracy

> Code and materials for studying the **accuracy of transcutaneous CO₂ (TcCO₂) monitoring** versus arterial PaCO₂ across clinical contexts. This repository hosts analysis code, data extraction assets, and artifacts supporting posters presented at **CHEST** and **ATS**; a manuscript is in preparation.

**Project status:** Presented as posters at CHEST and ATS. Manuscript in preparation.

---

## Links & Persistent Identifiers

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

This project uses **Python** and **Stata**. Choose the path appropriate to the analysis you wish to reproduce.

### A) Python

**Requirements:** Python ≥3.10

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .      # if pyproject.toml is configured
```

Run analyses (example; replace with finalized scripts):

```bash
python python/<analysis_script>.py --input data/ --out artifacts/
```

### B) Stata

**Requirements:** Stata 17+ recommended

From the Stata console:

```stata
cd "<repo_root>"
do "Conway Meta/<main_do_file>.do"
```

Outputs are written to `artifacts/`.

---

## Data Access

- Primary inputs derive from publicly available supplemental tables and code associated with Conway et al. (*Thorax*, 2019).
- These materials are mirrored or referenced in `Data/ Conway Thorax supplement and code/` for convenience.
- **No patient‑level protected health information (PHI)** is included in this repository.

If future analyses require restricted data, do **not** commit raw files. Instead, provide synthetic examples and access instructions here.

---

## Computational Environment

- **Operating systems tested:** macOS, Linux (confirm exact versions before release)
- **Languages:** Python, Stata
- **Dependencies:** Declared in `pyproject.toml` (Python) and within Stata `.do` file headers
- **Hardware:** CPU‑only; no GPU required

---

## Repository Layout

```
├── Conway Meta/                          # Meta‑analysis scripts and notes
├── Data/ Conway Thorax supplement and code/  # Public supplemental data/code
├── python/                              # Python analysis scripts and notebooks
├── artifacts/                           # Generated figures and tables
├── docs/                                # Project documentation
├── Drafts/                              # Exploratory or working files
├── pyproject.toml                       # Python project configuration
├── LICENSE                              # MIT License
└── README.md
```

---

## Workflow Overview

1. Acquire public supplemental data (Conway et al.).
2. Harmonize study‑level variables (device, site, temperature correction).
3. Compute bias and limits of agreement (Bland–Altman).
4. Perform subgroup and sensitivity analyses.
5. Export tables and figures to `artifacts/`.

---

## Results Mapping

| Poster / Paper Item | Script / Notebook | Command | Output |
|--------------------|-------------------|---------|--------|
| Overall bias & LOA | `<script>` | `python <script>` | `artifacts/fig1.png` |
| Subgroup analysis  | `<script>` | `python <script>` | `artifacts/fig2.png` |

Replace placeholders with final filenames prior to release.

---

## Quality Checks

- Manual review of generated tables against source supplements
- Optional future work: add smoke‑test dataset and automated checks

---

## License

- **Code:** MIT License (see `LICENSE`)
- **Data:** Governed by original source licenses (e.g., Thorax supplemental materials)

---

## Funding & Acknowledgements

- **Funding:** *ATS ASPIRE Program, NIH T32, and the Intermountain Fund*

---

## Contributing & Governance

Contributions are welcome via GitHub issues and pull requests. Please add the following prior to public release:

- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`

---

## Maintainer & Contact

- **Maintainer:** Brian Locke
- **Contact:** Open an issue at https://github.com/reblocke/tcco2-accuracy/issues
- **Maintenance status:** Active (poster phase; manuscript in preparation)

