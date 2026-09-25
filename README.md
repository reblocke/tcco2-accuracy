# TcCO2 Accuracy

[![ATS abstract](https://img.shields.io/badge/ATS%202025-10.1164%2Fajrccm.2025.211.Abstracts.A2683-blue)](https://doi.org/10.1164/ajrccm.2025.211.Abstracts.A2683)
[![CHEST abstract](https://img.shields.io/badge/CHEST%202025-10.1016%2Fj.chest.2025.07.3877-blue)](https://doi.org/10.1016/j.chest.2025.07.3877)
[![App](https://img.shields.io/badge/app-GitHub%20Pages-green)](https://reblocke.github.io/tcco2-accuracy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Research software and a static browser app for studying agreement between
transcutaneous CO2 (TcCO2) monitoring and arterial PaCO2 across clinical
contexts. The browser app runs the Python numerical model in Pyodide and keeps
all user-entered values client-side.

**Project status:** abstract/poster stage. Results have been presented at ATS
2025 and CHEST 2025; the manuscript has not yet been submitted. This is research
software and is not intended for clinical decision-making.

**Statistical revision:** the agreement-model logarithm and analytic LoA uncertainty
equations have been corrected under method revision
`agreement_natural_log_tau2_direct_v1`. Browser and agreement-parameter outputs are
provisional pending independent biostatistical review. PaCO2-dependent manuscript and
downstream outputs remain frozen at the legacy method revision; see
[`artifacts/STATUS.md`](artifacts/STATUS.md).

## Choose a Route

| Goal | Source/access class | Current evidence and start |
| --- | --- | --- |
| Explore the browser | Public staged copies; client-side input | [Static app](https://reblocke.github.io/tcco2-accuracy/) or [local serve](#quick-start); corrected-provisional, research-only |
| Develop or test locally | Repository code and public fixtures | [Locked setup](#quick-start) and [checks](#quality-checks); engineering checks do not grant statistical review |
| Understand or rebuild agreement artifacts | Public Conway table and five promoted outputs | [Artifact status](artifacts/STATUS.md) and [rebuild boundaries](#rebuild-outputs); corrected-provisional only |
| Try the downstream API | In-memory synthetic inputs | [Synthetic example](#in-memory-downstream-implementation) and [contract](docs/SPEC.md); example is noncompliant |
| Find private contract/review requirements | Restricted source, approved scratch output and human decisions | [Data governance](docs/DATA_GOVERNANCE.md), [release contract](docs/data_release_contract.json) and [artifact status](artifacts/STATUS.md); legacy results remain frozen |

## Links And Identifiers

| Item | Link |
| --- | --- |
| Static app | https://reblocke.github.io/tcco2-accuracy/ |
| Repository | https://github.com/reblocke/tcco2-accuracy |
| Machine-readable index | [llms.txt](llms.txt) |
| Active validation plan | [docs/PLAN.md](docs/PLAN.md) |
| ATS 2025 abstract | [10.1164/ajrccm.2025.211.Abstracts.A2683](https://doi.org/10.1164/ajrccm.2025.211.Abstracts.A2683) |
| CHEST 2025 abstract | [10.1016/j.chest.2025.07.3877](https://doi.org/10.1016/j.chest.2025.07.3877) |
| Conway evidence synthesis | [10.1136/thoraxjnl-2017-211466](https://doi.org/10.1136/thoraxjnl-2017-211466) |
| Conway data/code archive | [Figshare record](https://figshare.com/articles/dataset/Accuracy_of_TcCO2_monitoring_meta-analysis/6244058) |

## Authors, Funding, And Disclosures

| Contributor | Role | Affiliation |
| --- | --- | --- |
| Dustin Anderson-Bell, MD | Abstract author | University of Utah Health |
| Brian W. Locke, MD, MSc | Abstract author, repository maintainer | Intermountain Health; University of Utah |
| Ram Gouripeddi, MBBS, MS | ATS abstract author | University of Utah Biomedical Informatics |
| W. Richards, BS | ATS abstract author | University of Utah Biomedical Informatics |

Funding/support listed with the abstracts includes the American Thoracic Society
ASPIRE Fellowship, the Intermountain Fund, NIH NRSA `5T32HL105321`, and NCATS
`UM1TR004409`. Repository issues and pull requests are the preferred contact
route. Maintainer: Brian W. Locke (`@reblocke`, ORCID
[`0000-0002-3588-5238`](https://orcid.org/0000-0002-3588-5238)).

## Quick Start

Requirements: Git, Make, Python 3.11, `uv`, a free local port 8000, and
network access for the locked dependencies and browser CDNs. Run the following
from the repository root. It creates a fresh temporary clone and keeps its
virtual environment, dependency cache, and staged public web assets there;
the source checkout is not staged. The server remains active until `Ctrl-C`.
Successful setup opens <http://127.0.0.1:8000> and, with the default
likelihood-only form values, displays `Calculation complete.` plus estimates
and a chart. Dependency, port, or browser download failures stop that route;
check their messages before retrying with another fresh temporary directory.

```bash
(
set -eu
browser_work=$(mktemp -d "${TMPDIR:-/tmp}/tcco2-browser.XXXXXXXX")
git clone --quiet --local . "$browser_work/repo"
cd "$browser_work/repo"
export UV_CACHE_DIR="$browser_work/uv-cache"
export UV_PROJECT_ENVIRONMENT="$browser_work/repo/.venv"
export UV_PROJECT="$browser_work/repo"
uv sync --locked
make serve
)
```

The `serve` target first stages the Python allowlist and public
Conway/bootstrap data into the clone's `web/assets/py/` and
`web/assets/data/`, then starts an HTTP server. Open the local URL in a
browser with Web Worker support. The worker downloads Pyodide plus NumPy,
pandas and SciPy; the page also loads plotting/XLSX libraries. A working
browser route does not establish independent statistical review or clinical
validity. The temporary clone remains after stopping the server so its
generated files can be inspected and removed deliberately.

Staged files are generated copies, not canonical sources. The `verify` target is a
separate terminating integration gate that checks public history, stages assets,
checks formatting/lint, runs tests and browser E2E; it is not needed just to
start the app. See [Quality Checks](#quality-checks).

## Rebuild Outputs

Rebuilding artifacts is separate from local browser staging. The authoritative
promotion contract, accepted arguments, five corrected-provisional outputs,
and current review state are in [artifact status](artifacts/STATUS.md). The
public-agreement profile accepts only the public Conway table for canonical
promotion and rejects restricted PaCO2 inputs. Promotion to the repository's
`artifacts/` destination requires the canonical study table, seed 202401,
1,000 bootstrap draws, and `cluster_plus_withinstudy` mode. Custom inputs or
sensitivity settings belong in a fresh scratch destination after checking the
script's output contract. This README does not launch or authorize a rebuild.

The XLSX study table is the human-editable review mirror; the semantically equivalent
CSV is the operational source for canonical promotion and browser staging. PaCO2-dependent
outputs remain frozen. A future full-profile private comparison requires an
explicit approved restricted source and a fresh approved scratch/private output
directory; it is not a promotion route. See [data governance](docs/DATA_GOVERNANCE.md)
and [artifact status](artifacts/STATUS.md) for its access and review gates.

Within the repository, the full profile accepts output only under `.pytest_tmp/` or
`.tmp/`; otherwise it requires an explicitly approved external private destination. Do not run
this workflow to regenerate, promote, or unfreeze downstream results in the current wave.

Private prior generation likewise requires explicitly approved input and
output locations; see [data governance](docs/DATA_GOVERNANCE.md). It is not a
browser setup step.

Normalized restricted-derived weights are not automatically public-safe: they may reconstruct the
exact source distribution even without a `count` column. The static app therefore stages no PaCO2
prior and defaults to likelihood-only inference.

## In-Memory Downstream Implementation

`tcco2_accuracy.workflows.downstream.run_downstream_analysis(...)` is a caller-managed Python API
for the new draw-aligned downstream implementation. It accepts in-memory patient and Conway
`DataFrame` inputs only, requires nonblank patient/encounter identifiers plus encounter and
measurement ordering fields, and returns aggregate percentile summaries, aggregate Monte Carlo
diagnostics, and a non-sensitive manifest. It accepts no paths, writes no files, is not a rebuild or
promotion command, and is not staged to or callable from the Pages app. See
[`docs/SPEC.md`](docs/SPEC.md) for the exact code contract.

Seeded runs canonicalize patient clusters and Conway effect rows, so permuting either input table
does not change results or reproducibility metadata. Custom patient identifier/order column roles
must be nonblank, distinct, and separate from fixed PaCO2/subgroup fields.

`target_data_revision` must be a caller-managed, non-sensitive opaque token: 1-64 ASCII characters,
starting with a letter or digit and otherwise limited to letters, digits, `.`, `_`, and `-`. The
syntax rejects paths and free text but cannot establish that a token is free of identifiers; that
remains the caller's responsibility.

**Noncompliant synthetic development example:** this exercises the public API
without restricted data. It deliberately uses only 25 draws and disables the
independent-repeat stability gate; it is not a contract-grade run or validation
of research results. From a locked Python environment in a disposable clone,
use the public Conway table and synthetic in-memory rows only. It writes no
files or network results. Success prints aggregate rows and a noncompliant
contract status; missing dependencies or input-contract failures raise an
exception. Do not treat the output as a research result.

```python
import pandas as pd

from tcco2_accuracy.data import load_conway_studies
from tcco2_accuracy.workflows.downstream import (
    DownstreamWorkflowConfig,
    run_downstream_analysis,
)

patient_data = pd.DataFrame(
    [
        {
            "patient_id": f"synthetic-{group}-{number}",
            "encounter_id": f"synthetic-encounter-{group}-{number}",
            "encounter_order": 1,
            "measurement_order": 1,
            "paco2": 40.0 if number < 14 else 50.0,
            "subgroup": group,
        }
        for group in ("pft", "ed_inp", "icu")
        for number in range(20)
    ]
)
result = run_downstream_analysis(
    patient_data,
    load_conway_studies(),
    target_data_revision="synthetic-readme-v1",
    config=DownstreamWorkflowConfig(
        n_boot=25,
        enforce_minimum_draws=False,
        assess_stability=False,
        require_stability=False,
    ),
)
print(result.core.head())
print(result.manifest["contract_compliance"])
```

The example is intentionally marked noncompliant because it uses 25 draws and disables the
independent-repeat gate. A contract run begins with the default 10,000 draws and stability settings;
if the MCSE gate fails, increase `n_boot` rather than trying additional seeds or widening the
tolerance. Probabilities are returned on the 0-1 scale; prediction endpoints are in mmHg; likelihood
ratios are unitless.

## Repository Layout

```text
src/tcco2_accuracy/core/   Pure numerical/statistical source of truth
src/tcco2_accuracy/        I/O, contracts, wrappers, reporting, workflows
tests/                     Core, workflow, contract, and browser tests
web/                       Static GitHub Pages app
scripts/                   Staging, artifact, and data-prep commands
Data/                      Canonical public reference inputs and provenance
artifacts/                 Small aggregate review/manuscript outputs
docs/                      Architecture, deployment, validation, and decisions
Code/                      Stata reference code
Makefile                   Local command surface
pyproject.toml             Package and dependency metadata
uv.lock                    Locked Python environment
```

Internal drafts, editable poster decks, third-party PDFs, RData source archives,
local Stata `.dta` files, and exact or reconstructable restricted-derived PaCO2 outputs are
local-only or source-linked materials and are not part of the public branch tip.

## Static App Model

- Python remains the single source of truth for computation.
- JavaScript handles controls, uploads, worker messaging, and plotting.
- The posterior chart uses a posterior-focused x-axis for readability; numeric
  summaries still use the full posterior/prior support.
- Default calculations use repo-shipped canonical bootstrap parameters in likelihood-only mode.
- Prior-weighted inference requires an explicit binned-prior upload. No repository prior is staged
  or fetched; the uploaded prior remains client-side with the rest of the inputs.
- Custom study tables or changed bootstrap settings trigger in-browser
  recomputation through the staged Python package.
- Canonical and recomputed parameters must report the current agreement-method
  revision and provisional result status; stale or mixed canonical assets fail closed.
- Grouped parameter inputs must contain the mapped setting-specific group; missing groups fail
  closed. Result metadata records the requested group and the parameter group actually used.
- User-entered values and uploads remain in the browser. The app has no backend,
  telemetry, persistence, or patient-value URLs.
- Client-side processing does not grant permission to use or share restricted
  data; source access and release terms still apply.

## Data Access And Dictionary

- The canonical Conway table is maintained as a human-editable XLSX review mirror and a
  semantically equivalent CSV operational source for promotion and browser staging.
- Exact counts, binned distributions, and normalized weights derived from the restricted PaCO2
  source must not be tracked. Generate them only under `.pytest_tmp/`, `.tmp/`, or an explicitly
  approved external private workspace.
- No patient-level protected health information (PHI) is included in the current tracked tree.
- Public/restricted asset boundaries are documented in
  `docs/DATA_GOVERNANCE.md` and `Data/PROVENANCE.md`.
- The machine-readable current-tree authority is
  [`docs/data_release_contract.json`](docs/data_release_contract.json). Complete
  [`docs/restricted_data_provenance.template.json`](docs/restricted_data_provenance.template.json)
  before restricted-data use or release review; unresolved fields remain
  `HUMAN REVIEW REQUIRED`.
- Variable and artifact documentation is available in [data_dictionary.md](data_dictionary.md)
  and [data_dictionary.csv](data_dictionary.csv).

If future analyses require restricted data, do not commit raw files. Provide
synthetic examples and access instructions instead. Public branch and tag history is continuously
checked against the release contract. Independently retained clones, caches, and historical
deployments outside repository-controlled refs are not asserted removed. Retained
rounded/aggregate downstream artifacts are listed in `artifacts/STATUS.md`; they remain frozen and
are not release-approved.

## Quality Checks

| Make target | Purpose and local effects |
| --- | --- |
| `stage-web` | Replace staged Python and public data copies under `web/assets/` and write the staging manifest |
| `test` | Run Python unit, workflow, staging, and browser-contract tests |
| `e2e` | Run Playwright browser smoke tests against the staged app |
| `visual-qa` | Write local review screenshots under `.pytest_tmp/visual-qa/` |
| `verify` | Check public history, stage assets, check format/lint, and run unit and E2E tests |

Scientific validation targets are documented in `docs/VALIDATION.md`; current ticket states and
their minimum completion evidence are maintained in `docs/PLAN.md`.

## Citation

Until a manuscript or archived software DOI is available, cite the repository
release or commit and the conference abstracts:

> Anderson-Bell D, Locke BW. Simulation suggests transcutaneous CO2 sensors may
> accurately detect hypercapnia across settings. CHEST. 2025;168(4):A6917.
> doi:[10.1016/j.chest.2025.07.3877](https://doi.org/10.1016/j.chest.2025.07.3877)

> Anderson-Bell D, Locke BW, Gouripeddi R, Richards W. In silico estimation of
> the performance of transcutaneous CO2 sensors for detecting hypercapnia in
> newly admitted inpatients. American Journal of Respiratory and Critical Care
> Medicine. 2025;211(Supplement_1):A2683.
> doi:[10.1164/ajrccm.2025.211.Abstracts.A2683](https://doi.org/10.1164/ajrccm.2025.211.Abstracts.A2683)

See `CITATION.cff` for machine-readable metadata.

## License

- **Code and author-owned repository documentation:** MIT License (see `LICENSE`)
- **Data and external evidence sources:** governed by original source licenses
  and access terms.
- **Third-party article pages, source records, and abstract pages:** linked and
  cited rather than mirrored as publisher PDFs or full publisher text.
