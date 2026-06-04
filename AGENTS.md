# Codex AGENTS

## Purpose
- This repository contains the TcCO2 to PaCO2 agreement meta-analysis, uncertainty simulation, static GitHub Pages app, validation artifacts, and manuscript support files.
- Keep it as a stabilized monorepo. Do not split app, analysis, data, or manuscript assets in this wave.
- The importable Python package is `tcco2_accuracy` under `src/`.

## Repo Map
- `src/tcco2_accuracy/core/` - pure numerical/statistical code and input validation; no filesystem, web, artifact, or test dependencies.
- `src/tcco2_accuracy/reporting/` - reviewer/manuscript table, figure-data, and snippet builders.
- `src/tcco2_accuracy/workflows/` - workflow entry points and command-facing orchestration wrappers.
- `src/tcco2_accuracy/` - compatibility wrappers, file/data I/O, browser contract, and UI API.
- `tests/core/`, `tests/contracts/`, `tests/workflows/`, and `tests/e2e/` - purpose-specific QA/test code.
- `web/` - static GitHub Pages app; generated staged Python/data assets are created under `web/assets/` by `make stage-web`.
- `scripts/stage_web_python.py` - stages Python and data assets for the browser app.
- `Data/` - canonical public CSV/XLSX inputs, public prior weights, and data provenance notes.
- `artifacts/` - small derived outputs intended for review/manuscript workflows.
- `Code/` - Stata reference code; `Code/Legacy/` is read-only reference material.
- `llms.txt` and `web/llms.txt` - machine-readable repository and app indexes.
- `data_dictionary.md` and `data_dictionary.csv` - public/restricted input and artifact dictionary.
- `.agents/skills/` - focused local workflows for recurring agent tasks.

## Commands
- Install locked environment: `uv sync --locked`
- Stage browser assets: `make stage-web`
- Python tests: `make test`
- Browser E2E: `make e2e`
- Visual QA screenshots: `make visual-qa`
- Full local gate: `make verify`
- Serve static app: `make serve`
- Rebuild artifacts: `uv run python scripts/rebuild_artifacts.py --out artifacts`

## Authority
1. Conway Thorax 2019 paper, Figshare source record, and repository provenance docs.
2. `docs/SPEC.md`, `docs/VALIDATION.md`, `docs/CONWAY_DATA_SCHEMA.md`,
   `docs/DATA_GOVERNANCE.md`, and `docs/DECISIONS.md`.
3. Stata code in `Code/` as reference only; `Code/Legacy/` may contain bugs.
4. Existing Python code, tests, browser contract tests, and artifacts.

When Stata conflicts with the paper or docs, implement the paper/docs and record the divergence in `docs/DECISIONS.md` or an ADR.

## Working Rules
- Before non-trivial edits, state assumptions, ambiguities, tradeoffs, a brief plan, risks, and verification commands.
- Keep changes small and directly tied to the request; do not make drive-by refactors.
- Do not modify `Code/Legacy/` unless the user explicitly asks for legacy Stata reference changes.
- Do not add `Drafts/`, editable poster decks, Word lock files, third-party article PDFs, or publisher-formatted full text to the public branch.
- Current citable project outputs are abstracts only: CHEST 2025 `10.1016/j.chest.2025.07.3877` and ATS 2025 `10.1164/ajrccm.2025.211.Abstracts.A2683`.
- Do not add full manuscript Markdown until a submitted preprint, accepted author manuscript, or explicit public manuscript source is supplied.
- Do not commit patient-level or large raw extracts. Use small, de-identified fixtures under `tests/fixtures/`.
- Do not commit exact count-bearing PaCO2 prior or distribution-bin outputs; keep them in `.pytest_tmp/`,
  `.tmp/`, or a private manuscript workspace unless explicitly approved.
- Do not commit external RData source archives or mirrored publisher PDFs. Link DOI/Figshare records instead.
- Keep core math pure and isolate I/O in package I/O, workflow modules, or the browser contract.
- Do not edit generated `web/assets/py/` or `web/assets/data/` by hand; stage them from canonical sources.
- Python remains the single numerical source of truth. JavaScript may manage UI state, uploads, worker calls, and plotting, but must not reimplement the statistical model.
- Browser app changes must keep user-entered values client-side: no backend, telemetry, persistence, or PHI-bearing URLs unless explicitly approved.
- Keep `pyproject.toml` and `uv.lock` authoritative for dependencies.
- Generated scratch outputs belong in `.pytest_tmp/` or `.tmp/` and should not be tracked.
- Keep `README.md`, `llms.txt`, `web/llms.txt`, `CITATION.cff`, and the data dictionary internally consistent after repository-surface changes.

## Skill Triggers
- Planning a non-trivial change: `.agents/skills/implementation-strategy/SKILL.md`.
- Verifying a code change: `.agents/skills/code-change-verification/SKILL.md`.
- Updating docs after behavior/workflow changes: `.agents/skills/docs-sync/SKILL.md`.
- Static Pages/Pyodide/browser contract work: `.agents/skills/static-browser-pyodide-verification/SKILL.md`.
- Reviewing numerical/statistical behavior: `.agents/skills/scientific-validation/SKILL.md`.
- Reviewing clinical, privacy, public-copy, provenance, or app surfaces: use the matching focused skill in `.agents/skills/`.

## Done Criteria
- Relevant tests pass locally.
- `make stage-web` succeeds after browser-contract or package changes.
- Browser contract outputs match Python reference behavior for canonical cases.
- Validation targets and small artifacts are updated when behavior changes.
- Decisions, divergences, browser-runtime choices, and data provenance changes are documented.
- Public/restricted data boundaries remain aligned with `docs/DATA_GOVERNANCE.md` and `Data/PROVENANCE.md`.
- `CITATION.cff` validates with CFF tooling after citation edits.
- Static hygiene checks find no placeholder ORCID, stale "no DOI" wording, generic LLM appendix, tracked drafts, tracked third-party PDFs, `.Rhistory`, `data.Rdata`, local `.dta`, or tracked generated `web/assets/py` and `web/assets/data`.
- The final report names changed files, verification commands, warnings, skips, and remaining risks.
